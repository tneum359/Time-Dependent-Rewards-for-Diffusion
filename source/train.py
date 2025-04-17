import ImageReward as reward
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from peft import get_peft_model, LoraConfig, TaskType
import sys
import os
import matplotlib.pyplot as plt
import io
from PIL import Image
import numpy as np
import traceback
from torchvision.transforms.functional import to_pil_image

# Add parent directory path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__) if __file__ else '.', '..')))
from source.timedep_dataloader import load_diffusion_dataloader # Assumes dataloader returns (final_img, inter_img, ts, prompt_str)

class PEFTImageReward(nn.Module):
    def __init__(self, base_model_path="ImageReward-v1.0", timestep_dim=320):
        super().__init__()
        print(f"Initializing PEFTImageReward - Base: {base_model_path}")
        # Load base models
        self.base_reward_model = reward.load(base_model_path)
        self.original_reward_model = reward.load(base_model_path)
        self.original_reward_model.eval() # Frozen model for target scores

        # Access and wrap vision encoder with PEFT
        if not hasattr(self.base_reward_model, 'blip') or not hasattr(self.base_reward_model.blip, 'visual_encoder'):
            raise AttributeError("Cannot find '.blip.visual_encoder'.")
        self.vision_encoder = self.base_reward_model.blip.visual_encoder
        print("Identified vision component as '.blip.visual_encoder'")

        # Get vision feature dimension
        self.vision_feature_dim = self.vision_encoder.embed_dim if hasattr(self.vision_encoder, 'embed_dim') else self.vision_encoder.config.hidden_size
        print(f"Vision feature dimension: {self.vision_feature_dim}")

        # Apply PEFT to the vision_encoder
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION, r=16, lora_alpha=32, lora_dropout=0.1,
            target_modules=["qkv", "proj"]
        )
        self.vision_encoder = get_peft_model(self.vision_encoder, peft_config)
        print(f"PEFT applied to vision_encoder.")

        # Trainable timestep embedding
        self.timestep_embedding = TimestepEmbedding(timestep_dim)
        print(f"Timestep Embedding dim: {timestep_dim}")

        # Trainable layer to fuse PEFT-vision features and time embedding
        self.fusion_layer = nn.Linear(self.vision_feature_dim + timestep_dim, self.vision_feature_dim)
        print(f"Fusion layer: {self.vision_feature_dim + timestep_dim} -> {self.vision_feature_dim}")

        # Identify the original model's reward head (MLP)
        if hasattr(self.base_reward_model, 'mlp'):
             self.reward_head = self.base_reward_model.mlp
             try:
                 first_layer_of_mlp = self.reward_head.layers[0]
                 self.reward_head_in_dim = first_layer_of_mlp.in_features # e.g., 768
                 print(f"Identified reward head: 'mlp' (expects input dim: {self.reward_head_in_dim})")
             except Exception: raise AttributeError("Cannot get input dim for 'mlp' reward head.")
        else: raise AttributeError("Cannot find 'mlp' reward head.")

        # Trainable projection layer to map fused features to reward head input dimension
        self.fusion_to_reward_proj = nn.Linear(self.vision_feature_dim, self.reward_head_in_dim)
        print(f"Added projection layer: {self.vision_feature_dim} -> {self.reward_head_in_dim}")

        # --- Freeze Parameters ---
        # Freeze all parameters of the base model initially
        for name, param in self.base_reward_model.named_parameters():
            param.requires_grad = False
        # PEFT handles unfreezing LoRA layers within self.vision_encoder
        # Unfreeze our custom layers
        for param in self.timestep_embedding.parameters(): param.requires_grad = True
        for param in self.fusion_layer.parameters(): param.requires_grad = True
        for param in self.fusion_to_reward_proj.parameters(): param.requires_grad = True

        # Keep original_reward_model completely frozen (already in eval mode)
        for param in self.original_reward_model.parameters():
             param.requires_grad = False

        print("PEFTImageReward model configuration complete.")
        print(f"Total Trainable Params: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")


    def forward(self, image, timestep):
        """ Forward pass for the PEFT model predicting reward from intermediate image. """
        # 1. Get timestep embedding: [B, timestep_dim]
        timestep_emb = self.timestep_embedding(timestep)

        # 2. Get image features using PEFT-wrapped vision encoder: [B, vision_feature_dim]
        # Prepare image using the base model's processor
        if hasattr(self.base_reward_model, 'vis_processor'):
             processed_image = self.base_reward_model.vis_processor(image.to(next(self.vision_encoder.parameters()).device))
        else:
             print("Warning: Cannot find base_reward_model.vis_processor. Using raw image tensor.")
             processed_image = image.to(next(self.vision_encoder.parameters()).device)

        # --- EDIT: Call vision_encoder using 'pixel_values' keyword argument ---
        vision_outputs = self.vision_encoder(pixel_values=processed_image)
        # --- End EDIT ---

        image_features = vision_outputs.pooler_output

        # 3. Concatenate image features and timestep embedding: [B, vision_feature_dim + timestep_dim]
        combined_features = torch.cat([image_features, timestep_emb], dim=1)

        # 4. Fuse features: [B, vision_feature_dim]
        fused_features = self.fusion_layer(combined_features)

        # 5. Project fused features to match reward head input: [B, reward_head_in_dim]
        projected_features = self.fusion_to_reward_proj(fused_features)

        # 6. Pass projected features through the original model's reward head: [B, 1]
        try:
            reward_score = self.reward_head(projected_features)
        except Exception as e: print(f"Error in reward head: {e}"); raise
        return reward_score

class TimestepEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__(); self.dim = dim
        self.proj = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
    def forward(self, timestep):
        half_dim = self.dim // 2; timestep = timestep.float() / 1000.0
        freqs = torch.exp(-torch.arange(half_dim, device=timestep.device) * torch.log(torch.tensor(10000.0)) / half_dim)
        timestep = timestep.view(-1, 1); freqs = freqs.view(1, -1); args = timestep * freqs
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1: embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return self.proj(embedding)

def plot_loss_to_terminal(steps, losses, width=80, title="Training Loss per Step"):
    if not steps or not losses or len(steps) != len(losses): print("Plotting skipped: Invalid data."); return
    print(f"\n--- {title} ---"); avg_loss_val = np.mean(losses[-50:]) if losses else float('nan') # Avg last 50
    print(f"Current Avg Loss (last 50 steps): {avg_loss_val:.6f}")
    try:
        plt.figure(figsize=(12, 4)); plt.plot(steps, losses, linestyle='-', color='b', alpha=0.7)
        plt.title(title); plt.xlabel('Training Step'); plt.ylabel('Batch Loss')
        plt.grid(True, alpha=0.5)
        if max(losses) / min(l for l in losses if l > 0) > 50: plt.yscale('log'); plt.ylabel('Batch Loss (log scale)')
        plt.tight_layout(); buf = io.BytesIO(); plt.savefig(buf, format='png'); buf.seek(0); img = Image.open(buf); plt.close()
        aspect_ratio = img.width / img.height; new_height = min(30, int(width / aspect_ratio / 2))
        img = img.resize((width, new_height)).convert('L'); ascii_chars = ' .:-=+*#%@'; pixels = np.array(img)
        ascii_img = [''.join([ascii_chars[int(p * (len(ascii_chars) - 1) / 255)] for p in r]) for r in pixels]
        for row in ascii_img: print(row)
        print("--- End Loss Curve ---\n")
    except Exception as e: print(f"Failed to generate terminal plot: {e}\n{traceback.format_exc()}")

def train(
    prompts_file="prompts.txt",
    batch_size=4,
    learning_rate=1e-4,
    # num_epochs removed
    image_size=512,
    device="cuda" if torch.cuda.is_available() else "cpu",
    checkpoint_dir="checkpoints",
    plot_every_n_steps=1 # How often to plot the loss
):
    """ Train for one pass through the prompts file, plotting loss per step. """
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Checkpoints directory: {checkpoint_dir}")

    # Create model
    try: model = PEFTImageReward().to(device)
    except Exception as e: print(f"ERROR: Failed to init model: {e}\n{traceback.format_exc()}"); return None
    print(f"Model loaded on device: {device}")

    # Create dataloader
    try:
        # Pass device to dataloader so dataset uses it
        dataloader = load_diffusion_dataloader(
            prompts_file=prompts_file, batch_size=batch_size, image_size=image_size,
            shuffle=True, device=device # Pass device here
        )
        dataset_size = len(dataloader.dataset)
        if dataset_size == 0: raise ValueError("Dataloader empty.")
        print(f"Data loaded. Dataset size: {dataset_size} prompts.")
        num_steps = len(dataloader) # Total steps for one pass
        print(f"Total Steps for one pass: {num_steps}")
    except Exception as e: print(f"ERROR: Failed to load dataloader: {e}\n{traceback.format_exc()}"); return None

    # Setup Optimizer & Loss
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params: print("ERROR: No trainable parameters found."); return None
    print(f"Optimizing {len(trainable_params)} trainable params ({sum(p.numel() for p in trainable_params):,} total).")
    optimizer = optim.AdamW(trainable_params, lr=learning_rate, weight_decay=0.01)
    loss_fn = nn.MSELoss()

    # Metrics storage
    step_losses = []
    global_steps_list = [] # Renamed to avoid conflict
    global_step_counter = 0

    print(f"Starting training for one pass ({num_steps} steps)...")

    # --- Training Loop (Single Pass) ---
    model.train()
    total_loss_accum = 0.0
    batches_processed = 0

    for i, batch_data in enumerate(dataloader):
        global_step_counter += 1
        try:
            # Unpack batch (prompt is text, others are tensors)
            final_images_cpu, intermediate_images_cpu, timesteps_cpu, prompts_text = batch_data

            # Move tensors needed for training to the device
            intermediate_images = intermediate_images_cpu.to(device)
            timesteps = timesteps_cpu.to(device)
            # final_images_gpu = final_images_cpu.to(device) # Only move if score needed tensor

            # --- Convert final images to PIL for score method ---
            final_images_pil = []
            for img_tensor in final_images_cpu: # Iterate through batch on CPU
                # Convert from bfloat16 (or other) to float32
                img_tensor_float32 = img_tensor.to(torch.float32)
                # Now convert the float32 tensor to PIL
                final_images_pil.append(to_pil_image(img_tensor_float32))
            # --- End Conversion ---

            # Target rewards using the *original* model's score method (needs prompt and PIL image/tensor)
            # Note: original_reward_model.score expects PIL images or tensors and prompt text
            # We need to ensure final_images is in the correct format (likely tensor is fine)
            with torch.no_grad():
                model.original_reward_model.to(device).eval()
                # Pass the list of PIL images
                target_rewards_list = model.original_reward_model.score(prompts_text, final_images_pil) # USE PIL LIST
                if target_rewards_list is None: print(f"W: Skip step {global_step_counter}, None target"); continue
                # Score returns list, convert to tensor and move to device
                target_rewards = torch.tensor(target_rewards_list, device=device, dtype=torch.float32).unsqueeze(1) #[B, 1]

            # Predicted rewards using our PEFT model (takes image tensor, timestep tensor)
            predicted_rewards = model(intermediate_images, timesteps) # Output shape [B, 1]
            if predicted_rewards is None: print(f"W: Skip step {global_step_counter}, None prediction"); continue

            # Ensure shapes match [B, 1] vs [B, 1]
            if predicted_rewards.shape != target_rewards.shape:
                 print(f"W: Skip step {global_step_counter}, Shape mismatch Pred:{predicted_rewards.shape}, Targ:{target_rewards.shape}"); continue

            loss = loss_fn(predicted_rewards, target_rewards)

            optimizer.zero_grad(); loss.backward(); optimizer.step()

            batch_loss = loss.item()
            total_loss_accum += batch_loss
            batches_processed += 1

            # Store loss for plotting
            step_losses.append(batch_loss)
            global_steps_list.append(global_step_counter)
            print("STEP: ", global_step_counter)

            # Print progress and plot periodically
            if global_step_counter % plot_every_n_steps == 0:
                print(f"  Step {global_step_counter}/{num_steps}, Batch Loss: {batch_loss:.6f}, Avg Loss: {total_loss_accum/batches_processed:.6f}")
                plot_loss_to_terminal(global_steps_list, step_losses)

        except Exception as batch_e:
            print(f"\nERROR Step {global_step_counter}: {batch_e}\n{traceback.format_exc()}\nSkipping step...")
            continue

    # --- End of Training Pass ---
    avg_loss_final = total_loss_accum / batches_processed if batches_processed > 0 else float('nan')
    print(f"\n--- Training Pass Finished. Average Loss: {avg_loss_final:.6f} ---")

    # Final plot
    plot_loss_to_terminal(global_steps_list, step_losses, title="Final Training Loss per Step")

    # Save Final Checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, "final_checkpoint_onepass.pt")
    try:
        checkpoint_data = {
            'steps': global_step_counter, 'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(), 'final_avg_loss': avg_loss_final,
        }
        torch.save(checkpoint_data, checkpoint_path)
        print(f"Final checkpoint saved to {checkpoint_path}")
    except Exception as save_e: print(f"ERROR saving final checkpoint: {save_e}")

    print(f"Model processed {dataset_size} unique prompts in {global_step_counter} steps.")
    return model

if __name__ == "__main__":
    prompts_file_path = "prompts.txt"
    if not os.path.exists(prompts_file_path): print(f"ERROR: {prompts_file_path} not found.")
    else: train(prompts_file=prompts_file_path) # Run training

