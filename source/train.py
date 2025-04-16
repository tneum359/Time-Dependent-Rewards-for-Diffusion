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

# Add parent directory to path to import dataloader
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from source.timedep_dataloader import load_diffusion_dataloader

class PEFTImageReward(nn.Module):
    def __init__(self, base_model_path="ImageReward-v1.0", timestep_dim=320):
        super().__init__()
        print(f"Initializing PEFTImageReward model - loading base: {base_model_path} via ImageReward library")
        self.base_reward_model = reward.load(base_model_path)
        self.original_reward_model = reward.load(base_model_path)
        self.original_reward_model.eval()

        # --- Corrected Access Attempt 2: Try 'visual_encoder' ---
        # Check if 'visual_encoder' exists directly on the loaded object
        if hasattr(self.base_reward_model, 'visual_encoder'):
            self.vision_encoder = self.base_reward_model.visual_encoder
            print("Identified vision component as 'visual_encoder'")
        # --- Fallback/Error ---
        else:
             # If this also fails, we absolutely need to inspect the object structure
             print("\n--- DEBUG: Structure of loaded base_reward_model ---")
             print(self.base_reward_model) # Print the model structure
             print("--- END DEBUG ---\n")
             raise AttributeError("Loaded ImageReward object does not have 'vision_model' or 'visual_encoder'. Please check the printed structure above.")
        # --- End Correction ---

        self.timestep_embedding = TimestepEmbedding(timestep_dim)
        
        # Get feature dim from the vision_encoder's config
        # Ensure the identified encoder has a 'config' attribute
        if not hasattr(self.vision_encoder, 'config') or not hasattr(self.vision_encoder.config, 'hidden_size'):
             raise AttributeError("Identified vision encoder does not have '.config.hidden_size'. Check structure.")
        self.vision_feature_dim = self.vision_encoder.config.hidden_size
        print(f"Vision feature dimension: {self.vision_feature_dim}")
        
        self.fusion_layer = nn.Linear(self.vision_feature_dim + timestep_dim, self.vision_feature_dim)
        
        # Define PEFT configuration
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=16, 
            lora_alpha=32, 
            lora_dropout=0.1,
            target_modules=["query", "key", "value", "projection"] # Adjust if needed based on vision_encoder type
        )
        # Apply PEFT to the identified vision_encoder
        self.vision_encoder = get_peft_model(self.vision_encoder, peft_config)
        print(f"PEFT applied to vision_encoder. Trainable params in vision_encoder: {sum(p.numel() for p in self.vision_encoder.parameters() if p.requires_grad)}")

        # --- Identify the reward head (logic remains the same, check common names) ---
        if hasattr(self.base_reward_model, 'reward_head'):
             self.reward_head = self.base_reward_model.reward_head
             print("Identified reward head: 'reward_head'")
        elif hasattr(self.base_reward_model, 'score_head'):
             self.reward_head = self.base_reward_model.score_head
             print("Identified reward head: 'score_head'")
        elif hasattr(self.base_reward_model, 'mlp_head'):
             self.reward_head = self.base_reward_model.mlp_head
             print("Identified reward head: 'mlp_head'")
        elif hasattr(self.base_reward_model, 'fc') and isinstance(self.base_reward_model.fc, nn.Linear):
             self.reward_head = self.base_reward_model.fc
             print("Identified reward head: 'fc'")
        else:
             # Fallback attempt (less robust)
             try:
                 last_layer = list(self.base_reward_model.children())[-1]
                 while not isinstance(last_layer, nn.Linear) and list(last_layer.children()):
                      last_layer = list(last_layer.children())[-1] 
                 if isinstance(last_layer, nn.Linear):
                      self.reward_head = last_layer
                      print(f"Identified reward head dynamically as last Linear layer: {self.reward_head}")
                 else: raise AttributeError("Dynamic search failed.")
             except Exception as e:
                 raise AttributeError(f"Could not automatically identify the reward head layer. Error: {e}")

        # --- Freezing Parameters (logic remains the same) ---
        for name, param in self.base_reward_model.named_parameters():
             param.requires_grad = False 
        for param in self.timestep_embedding.parameters(): param.requires_grad = True
        for param in self.fusion_layer.parameters(): param.requires_grad = True
        # PEFT handles LoRA layers within self.vision_encoder

        print("PEFTImageReward model configuration complete.")
        
    def forward(self, image, timestep):
        timestep_emb = self.timestep_embedding(timestep)
        vision_outputs = self.vision_encoder(image) 
        image_features = vision_outputs.pooler_output 
        combined_features = torch.cat([image_features, timestep_emb], dim=1)
        fused_features = self.fusion_layer(combined_features) 
        try:
            reward_score = self.reward_head(fused_features)
        except Exception as e:
             print(f"Error passing fused features to identified reward head: {e}")
             print(f"Fused features shape: {fused_features.shape}")
             print(f"Reward head: {self.reward_head}")
             raise
        return reward_score

class TimestepEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(
            nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )

    def forward(self, timestep):
        """
        Convert timestep to sinusoidal embedding
        
        Args:
            timestep: Tensor of timesteps [batch_size]
            
        Returns:
            Timestep embeddings [batch_size, dim]
        """
        half_dim = self.dim // 2
        timestep = timestep.float() / 1000.0
        freqs = torch.exp(
            -torch.arange(half_dim, device=timestep.device) * torch.log(torch.tensor(10000.0)) / half_dim
        )
        timestep = timestep.view(-1, 1)
        freqs = freqs.view(1, -1)
        args = timestep * freqs
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return self.proj(embedding)

def plot_loss_to_terminal(epochs, losses, width=80):
    """Creates a loss plot and displays it as ASCII in the terminal."""
    if not epochs or not losses:
        print("No data to plot.")
        return
    
    print("\n--- Training Loss Curve ---")
    try:
        plt.figure(figsize=(10, 4))
        plt.plot(epochs, losses, marker='o', linestyle='-')
        plt.title('Training Loss per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Average Loss')
        plt.xticks(epochs)
        plt.grid(True)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf)
        plt.close()

        aspect_ratio = img.width / img.height
        new_height = min(30, int(width / aspect_ratio / 2))
        img = img.resize((width, new_height))
        img = img.convert('L')
        ascii_chars = ' .:-=+*#%@'
        pixels = np.array(img)
        ascii_img = []
        for row in pixels:
            ascii_row = ''.join([ascii_chars[int(pixel * (len(ascii_chars) - 1) / 255)] for pixel in row])
            ascii_img.append(ascii_row)
        
        for row in ascii_img:
            print(row)
        print("--- End Loss Curve ---\n")

    except Exception as e:
        print(f"Failed to generate terminal plot: {e}")
        print(traceback.format_exc())

def train(
    prompts_file="prompts.txt",
    batch_size=4,
    learning_rate=1e-4,
    num_epochs=10,
    image_size=512,
    device="cuda" if torch.cuda.is_available() else "cpu",
    save_path="finetuned_reward_model.pt",
    checkpoint_dir="checkpoints"
):
    """
    Train the PEFT-wrapped ImageReward model using prompts.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Checkpoints will be saved in: {checkpoint_dir}")

    # Create model
    try:
        model = PEFTImageReward().to(device)
        print(f"Model loaded on device: {device}")
    except Exception as model_init_e:
        print(f"ERROR: Failed to initialize PEFTImageReward model: {model_init_e}")
        print(traceback.format_exc())
        return None

    # Freeze original_reward_model (already done in PEFTImageReward __init__)
    # Note: Freezing base_reward_model happens *within* PEFTImageReward __init__ now too

    # Create dataloader
    print(f"Loading data using prompts from: {prompts_file}")
    try:
        dataloader = load_diffusion_dataloader(
            prompts_file=prompts_file, batch_size=batch_size, image_size=image_size, shuffle=True
        )
        dataset_size = len(dataloader.dataset)
        print(f"Data loaded. Dataset size: {dataset_size} prompts.")
        if dataset_size == 0: raise ValueError("Dataloader loaded an empty dataset.")
    except Exception as e:
        print(f"ERROR: Failed to load dataloader: {e}\n{traceback.format_exc()}")
        return None

    # Identify trainable parameters (PEFT layers, timestep_embedding, fusion_layer, potentially reward_head if not frozen)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
         print("ERROR: No trainable parameters found in the model. Check PEFT setup and freezing logic in PEFTImageReward __init__.")
         # Print parameter names and requires_grad status for debugging
         for name, param in model.named_parameters():
              print(f"  Param: {name}, Requires Grad: {param.requires_grad}")
         return None
    print(f"Optimizing {len(trainable_params)} trainable parameters ({sum(p.numel() for p in trainable_params):,} total).")

    optimizer = optim.AdamW(trainable_params, lr=learning_rate, weight_decay=0.01)
    loss_fn = nn.MSELoss()
    
    epoch_losses = []
    epochs_completed = []

    print("Starting training...")
    total_steps = len(dataloader) * num_epochs
    print(f"Total Epochs: {num_epochs}, Steps per Epoch: {len(dataloader)}, Total Steps: {total_steps}")

    for epoch in range(num_epochs):
        model.train()
        epoch_total_loss = 0.0
        num_batches = 0
        
        print(f"\n--- Starting Epoch {epoch+1}/{num_epochs} ---")
        for i, batch_data in enumerate(dataloader):
            try:
                final_images, intermediate_images, timesteps = batch_data
                final_images, intermediate_images, timesteps = final_images.to(device), intermediate_images.to(device), timesteps.to(device)

                # Target rewards using the frozen original model
                with torch.no_grad():
                    model.original_reward_model.to(device).eval()
                    target_rewards = model.original_reward_model.score(final_images)
                    if target_rewards is None: print(f"Warning: Got None target_rewards batch {i}. Skipping."); continue
                
                # Predicted rewards using our PEFT model
                predicted_rewards = model(intermediate_images, timesteps)
                if predicted_rewards is None: print(f"Warning: Got None predicted_rewards batch {i}. Skipping."); continue
                if predicted_rewards.shape != target_rewards.shape: print(f"Warning: Shape mismatch batch {i}. Pred: {predicted_rewards.shape}, Targ: {target_rewards.shape}. Skipping."); continue
                
                loss = loss_fn(predicted_rewards, target_rewards)
                
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                
                epoch_total_loss += loss.item(); num_batches += 1
                if (i + 1) % 50 == 0: print(f"  E{epoch+1} B{i+1}/{len(dataloader)}, BatchLoss: {loss.item():.6f}, EpochAvg: {epoch_total_loss/num_batches:.6f}")

            except Exception as batch_e:
                print(f"\nERROR E{epoch+1} B{i+1}: {batch_e}\n{traceback.format_exc()}\nSkipping batch...")
                continue # Continue to next batch

        avg_loss = epoch_total_loss / num_batches if num_batches > 0 else float('nan')
        print(f"--- Epoch {epoch+1}/{num_epochs} finished. Average Training Loss: {avg_loss:.6f} ---")
        
        epochs_completed.append(epoch + 1); epoch_losses.append(avg_loss)

        # --- Checkpoint Saving (remains the same) ---
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt")
        try:
            checkpoint_data = {
                'epoch': epoch + 1, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), 'train_loss': avg_loss,
            }
            torch.save(checkpoint_data, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
        except Exception as save_e: print(f"ERROR saving checkpoint E{epoch+1}: {save_e}")

        # --- Plotting (remains the same) ---
        plot_loss_to_terminal(epochs_completed, epoch_losses)
        
    # --- Final Save (remains the same) ---
    final_model_path = os.path.join(os.path.dirname(checkpoint_dir), "final_model_state_dict.pt")
    try:
        torch.save(model.state_dict(), final_model_path)
        print(f"Final model state dictionary saved to {final_model_path}")
    except Exception as final_save_e: print(f"ERROR saving final model state: {final_save_e}")

    print("\nTraining finished.")
    total_examples_seen = dataset_size * num_epochs
    print(f"Model saw approximately {total_examples_seen} examples over {num_epochs} epochs.")
    
    return model

if __name__ == "__main__":
    prompts_file_path = "prompts.txt" 
    if not os.path.exists(prompts_file_path):
         print(f"ERROR: {prompts_file_path} not found. Please create it with prompts.")
    else:
        train(prompts_file=prompts_file_path)

