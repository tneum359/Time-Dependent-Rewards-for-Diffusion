import ImageReward as reward
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from peft import get_peft_model, LoraConfig, TaskType, inject_adapter_in_model
import sys
import os
import matplotlib.pyplot as plt
import io
from PIL import Image
import numpy as np
import traceback
from torchvision.transforms.functional import to_pil_image
from transformers import AutoTokenizer, logging as hf_logging

# Suppress tokenizer warnings about legacy behavior
hf_logging.set_verbosity_error()

# Add parent directory path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__) if __file__ else '.', '..')))
from source.timedep_dataloader import load_diffusion_dataloader # Assumes dataloader returns (final_img, inter_img, ts, prompt_str)

class PEFTImageReward(nn.Module):
    def __init__(self, base_model_path="ImageReward-v1.0", timestep_dim=320, text_model_name="roberta-base"):
        super().__init__()
        print(f"Initializing PEFTImageReward - Base: {base_model_path}")
        self.base_reward_model = reward.load(base_model_path)
        self.original_reward_model = reward.load(base_model_path)
        self.original_reward_model.eval()

        # Access visual_encoder via 'blip'
        if not hasattr(self.base_reward_model, 'blip') or not hasattr(self.base_reward_model.blip, 'visual_encoder'):
            raise AttributeError("Cannot find '.blip.visual_encoder'.")
        # Keep the reference to the original encoder
        self.vision_encoder = self.base_reward_model.blip.visual_encoder
        print("Identified vision component as '.blip.visual_encoder'")

        # Get vision feature dimension
        self.vision_feature_dim = self.vision_encoder.embed_dim if hasattr(self.vision_encoder, 'embed_dim') else self.vision_encoder.config.hidden_size
        print(f"Vision feature dimension: {self.vision_feature_dim}")

        # Text Encoder & Projection (Keep references, remain frozen)
        if not hasattr(self.base_reward_model.blip, 'text_encoder') or not hasattr(self.base_reward_model.blip, 'text_proj'):
             raise AttributeError("Cannot find '.blip.text_encoder' or '.blip.text_proj'.")
        self.text_encoder = self.base_reward_model.blip.text_encoder
        self.text_proj = self.base_reward_model.blip.text_proj
        self.text_feature_dim = self.text_proj.out_features
        print(f"Text Encoder & Projection found. Projected Text dim: {self.text_feature_dim}")

        # Timestep Embedding
        self.timestep_embedding = TimestepEmbedding(timestep_dim)
        print(f"Timestep Embedding dim: {timestep_dim}")

        # Fusion Layer (Inputs: Vision + Time + Text)
        total_fused_dim = self.vision_feature_dim + timestep_dim + self.text_feature_dim
        intermediate_fusion_dim = self.vision_feature_dim
        self.fusion_layer = nn.Linear(total_fused_dim, intermediate_fusion_dim)
        print(f"Fusion layer input dim: {total_fused_dim}, output dim: {intermediate_fusion_dim}")

        # --- Define PEFT configuration ---
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION, r=16, lora_alpha=32, lora_dropout=0.1,
            target_modules=["qkv", "proj"]
        )

        # --- EDIT: Inject Adapters instead of wrapping ---
        # Apply PEFT by injecting adapters directly into self.vision_encoder
        self.vision_encoder = inject_adapter_in_model(peft_config, self.vision_encoder)
        # Mark adapters as trainable after injection (might be needed depending on peft version)
        for name, param in self.vision_encoder.named_parameters():
             if 'lora' in name:
                  param.requires_grad = True
        print(f"PEFT adapters injected into vision_encoder.")
        # --- End EDIT ---

        # Identify the reward head MLP
        if hasattr(self.base_reward_model, 'mlp'):
             self.reward_head = self.base_reward_model.mlp
             try: first_layer_of_mlp = self.reward_head.layers[0]; self.reward_head_in_dim = first_layer_of_mlp.in_features
             except Exception: raise AttributeError("Cannot get input dim for 'mlp' reward head.")
             print(f"Identified reward head: 'mlp' (expects input dim: {self.reward_head_in_dim})")
        else: raise AttributeError("Cannot find 'mlp' reward head.")

        # Projection layer
        self.fusion_to_reward_proj = nn.Linear(intermediate_fusion_dim, self.reward_head_in_dim)
        print(f"Added projection layer: {intermediate_fusion_dim} -> {self.reward_head_in_dim}")

        # --- Freeze Parameters ---
        # Freeze base model parameters FIRST
        for name, param in self.base_reward_model.named_parameters():
            param.requires_grad = False
            
        # Unfreeze ONLY our custom layers and injected PEFT layers explicitly
        # Note: Parameters within self.vision_encoder that are NOT LoRA layers will remain frozen.
        for param in self.timestep_embedding.parameters(): param.requires_grad = True
        for param in self.fusion_layer.parameters(): param.requires_grad = True
        for param in self.fusion_to_reward_proj.parameters(): param.requires_grad = True
        # Explicitly unfreeze LoRA params within the vision encoder (redundant if inject worked correctly, but safe)
        for name, param in self.vision_encoder.named_parameters():
             if 'lora' in name:
                  param.requires_grad = True
                  
        # Freeze original_reward_model used for scoring
        for param in self.original_reward_model.parameters(): param.requires_grad = False

        print("PEFTImageReward model configuration complete.")
        # Calculate trainable params AFTER potentially unfreezing LoRA params again
        print(f"Total Trainable Params: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")

    def forward(self, image, timestep, input_ids, attention_mask):
        """ Forward pass including image, timestep, and tokenized text. """
        timestep_emb = self.timestep_embedding(timestep)
        
        # --- EDIT: Correctly locate and use vis_processor ---
        # Prepare image using the base model's processor first
        # The processor is likely nested under 'blip' like other components
        if hasattr(self.base_reward_model, 'blip') and hasattr(self.base_reward_model.blip, 'vis_processor'):
            vis_processor = self.base_reward_model.blip.vis_processor
            # Move image tensor to the device where vision_encoder expects it
            device = next(self.vision_encoder.parameters()).device
            # Apply the processor
            processed_image = vis_processor(image.to(device))
            print(f"Applied vis_processor. Processed image shape: {processed_image.shape}") # Debug print
        else:
             # If no processor found, raise an error as processing is crucial
             raise AttributeError("Cannot find base_reward_model.blip.vis_processor. Image processing is required.")
             # processed_image = image.to(next(self.vision_encoder.parameters()).device) # Fallback removed - likely causes errors
        # --- End EDIT ---

        # Pass the processed image tensor directly as the first argument
        vision_outputs = self.vision_encoder(processed_image)
        
        image_features = vision_outputs.pooler_output
        
        # --- Text Processing (remains the same) ---
        device = image_features.device # Use device where image features ended up
        text_output = self.text_encoder.to(device)(
             input_ids=input_ids.to(device), attention_mask=attention_mask.to(device), return_dict=True
        )
        text_features = text_output.last_hidden_state[:, 0, :] # Use [CLS] token
        projected_text_features = self.text_proj.to(device)(text_features)
        # --- End Text Processing ---

        # --- Fusion and Prediction (remains the same) ---
        combined_features = torch.cat([
            image_features, 
            timestep_emb.to(device), 
            projected_text_features
        ], dim=1)
        fused_features = self.fusion_layer(combined_features)
        projected_features = self.fusion_to_reward_proj(fused_features)
        try: reward_score = self.reward_head.to(device)(projected_features)
        except Exception as e: print(f"Error in reward head: {e}"); raise
        # --- End Fusion ---
        
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
    avg_loss_val = np.mean(losses[-50:]) if len(losses) > 0 else float('nan') # Avg last 50 or available
    print(f"\n--- {title} (Avg last 50: {avg_loss_val:.6f}) ---")
    try:
        plt.figure(figsize=(12, 4)); plt.plot(steps, losses, linestyle='-', color='b', alpha=0.7)
        plt.title(title); plt.xlabel('Training Step'); plt.ylabel('Batch Loss')
        plt.grid(True, alpha=0.5)
        min_positive_loss = min((l for l in losses if l > 1e-9), default=1e-9) # Avoid zero/negative for log
        if max(losses) / min_positive_loss > 50: plt.yscale('log'); plt.ylabel('Batch Loss (log scale)')
        plt.tight_layout(); buf = io.BytesIO(); plt.savefig(buf, format='png'); buf.seek(0); img = Image.open(buf); plt.close()
        aspect_ratio = img.width / img.height; new_height = min(30, int(width / aspect_ratio / 2))
        img = img.resize((width, new_height)).convert('L'); ascii_chars = ' .:-=+*#%@'; pixels = np.array(img)
        ascii_img = [''.join([ascii_chars[int(p * (len(ascii_chars) - 1) / 255)] for p in r]) for r in pixels]
        for row in ascii_img: print(row)
        print("--- End Loss Curve ---\n")
    except Exception as e: print(f"Failed to generate terminal plot: {e}\n{traceback.format_exc()}")

def train(
    prompts_file="prompts.txt",
    batch_size=1,
    learning_rate=1e-4,
    image_size=512,
    device="cuda" if torch.cuda.is_available() else "cpu",
    checkpoint_dir="checkpoints",
    plot_every_n_steps=1,
    tokenizer_name="roberta-base",
    max_prompt_length=77
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Checkpoints directory: {checkpoint_dir}")

    # Load Tokenizer
    try: tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    except Exception as e: print(f"ERROR: Failed to load tokenizer '{tokenizer_name}': {e}"); return None
    print(f"Tokenizer '{tokenizer_name}' loaded.")

    # Create model
    try: model = PEFTImageReward(text_model_name=tokenizer_name).to(device)
    except Exception as e: print(f"ERROR: Failed to init model: {e}\n{traceback.format_exc()}"); return None
    print(f"Model loaded on device: {device}")
    
    # Create dataloader
    print(f"Loading data using prompts from: {prompts_file}")
    try:
        dataloader = load_diffusion_dataloader(
            prompts_file=prompts_file, batch_size=batch_size, image_size=image_size, shuffle=True, device=device
        )
        dataset_size = len(dataloader.dataset); num_steps = len(dataloader)
        if dataset_size == 0: raise ValueError("Dataloader empty.")
        print(f"Data loaded. Size: {dataset_size}. Steps/Pass: {num_steps}")
    except Exception as e: print(f"ERROR: Failed to load dataloader: {e}\n{traceback.format_exc()}"); return None

    # Setup Optimizer & Loss
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params: print("ERROR: No trainable params."); return None
    print(f"Optimizing {len(trainable_params)} params ({sum(p.numel() for p in trainable_params):,} total).")
    optimizer = optim.AdamW(trainable_params, lr=learning_rate, weight_decay=0.01)
    loss_fn = nn.MSELoss()
    
    # Metrics storage
    step_losses, global_steps_list = [], []; global_step_counter = 0
    print(f"Starting training for one pass ({num_steps} steps)... Plotting every {plot_every_n_steps} steps.")

    model.train(); total_loss_accum, batches_processed = 0.0, 0

    for i, batch_data in enumerate(dataloader):
        global_step_counter += 1
        try:
            final_images_cpu, intermediate_images_cpu, timesteps_cpu, prompts_text = batch_data

            # Tokenize Prompts
            tokenized_prompts = tokenizer(
                prompts_text, padding="max_length", truncation=True,
                max_length=max_prompt_length, return_tensors="pt"
            )
            # Move necessary tokens to device
            input_ids = tokenized_prompts["input_ids"].to(device)
            attention_mask = tokenized_prompts["attention_mask"].to(device)

            # Move images and timesteps to device
            intermediate_images = intermediate_images_cpu.to(device)
            timesteps = timesteps_cpu.to(device)
            final_images_gpu = final_images_cpu.to(device) # For potential use by score method

            # Convert final images to PIL for score method (run on CPU tensors)
            final_images_pil = [to_pil_image(img.to(torch.float32)) for img in final_images_cpu]

            # Target rewards using the original model's score method
            with torch.no_grad():
                model.original_reward_model.to(device).eval()
                target_rewards_output = model.original_reward_model.score(prompts_text, final_images_pil)
                if target_rewards_output is None: print(f"W: Skip step {global_step_counter}, None target"); continue

                # --- EDIT: Handle scalar or list return from score ---
                # Ensure we have a list of scores before creating the tensor
                if isinstance(target_rewards_output, list):
                    target_rewards_list = target_rewards_output
                else: # Assume scalar if not list
                    target_rewards_list = [target_rewards_output] 

                # Convert list (even if single item) to tensor and add dim
                target_rewards = torch.tensor(target_rewards_list, device=device, dtype=torch.float32).unsqueeze(1) #[B, 1]
                # --- End EDIT ---

            # Corrected Call: Pass required arguments to model
            predicted_rewards = model(intermediate_images, timesteps, input_ids, attention_mask)

            if predicted_rewards is None: print(f"W: Skip step {global_step_counter}, None prediction"); continue
            if predicted_rewards.shape != target_rewards.shape: print(f"W: Skip step {global_step_counter}, Shape mismatch"); continue

            loss = loss_fn(predicted_rewards, target_rewards)
            
            optimizer.zero_grad(); loss.backward(); optimizer.step()

            batch_loss = loss.item(); total_loss_accum += batch_loss; batches_processed += 1
            step_losses.append(batch_loss); global_steps_list.append(global_step_counter)

            # Plotting Condition
            if global_step_counter % plot_every_n_steps == 0:
                print(f"\n--- Plotting at Step {global_step_counter} ---")
                print(f"  Step {global_step_counter}/{num_steps}, Batch Loss: {batch_loss:.6f}, Running Avg Loss: {total_loss_accum/batches_processed:.6f}")
                plot_loss_to_terminal(global_steps_list, step_losses)

        except Exception as batch_e: print(f"\nERROR Step {global_step_counter}: {batch_e}\n{traceback.format_exc()}\nSkipping..."); continue

    # Final wrap-up
    avg_loss_final = total_loss_accum / batches_processed if batches_processed > 0 else float('nan')
    print(f"\n--- Training Pass Finished. Avg Loss: {avg_loss_final:.6f} ---")
    plot_loss_to_terminal(global_steps_list, step_losses, title="Final Training Loss per Step")
    checkpoint_path = os.path.join(checkpoint_dir, "final_checkpoint_onepass.pt")
    try:
        checkpoint_data = {'steps': global_step_counter, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'final_avg_loss': avg_loss_final}
        torch.save(checkpoint_data, checkpoint_path); print(f"Final checkpoint saved: {checkpoint_path}")
    except Exception as save_e: print(f"ERROR saving final checkpoint: {save_e}")
    print(f"Model processed {dataset_size} unique prompts in {global_step_counter} steps.")
    return model

if __name__ == "__main__":
    prompts_file_path = "prompts.txt"
    if not os.path.exists(prompts_file_path): print(f"ERROR: {prompts_file_path} not found.")
    else: train(prompts_file=prompts_file_path, plot_every_n_steps=1) # Set plotting frequency here

