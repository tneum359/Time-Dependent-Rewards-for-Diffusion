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
    def __init__(self, base_model_path="ImageReward/ImageReward", timestep_dim=320):
        super().__init__()
        print("Initializing PEFTImageReward model...")
        self.base_reward_model = reward.load(base_model_path)
        self.original_reward_model = reward.load(base_model_path)
        self.original_reward_model.eval()
        self.vision_encoder = self.base_reward_model.model.vision_model
        self.timestep_embedding = TimestepEmbedding(timestep_dim)
        self.vision_feature_dim = self.base_reward_model.model.vision_model.config.hidden_size
        self.fusion_layer = nn.Linear(self.vision_feature_dim + timestep_dim, self.vision_feature_dim)
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION, r=16, lora_alpha=32, lora_dropout=0.1,
            target_modules=["query", "key", "value", "projection"]
        )
        self.vision_encoder = get_peft_model(self.vision_encoder, peft_config)
        print("PEFTImageReward model initialized.")

    def forward(self, image, timestep):
        """
        Forward pass for the PEFT model predicting reward from intermediate image
        
        Args:
            image: Intermediate denoised image
            timestep: Diffusion timestep when the intermediate image was captured
            
        Returns:
            Predicted reward score
        """
        timestep_emb = self.timestep_embedding(timestep)
        vision_outputs = self.vision_encoder(image)
        image_features = vision_outputs.pooler_output
        combined_features = torch.cat([image_features, timestep_emb], dim=1)
        fused_features = self.fusion_layer(combined_features)
        if hasattr(vision_outputs, '_replace'):
            vision_outputs = vision_outputs._replace(pooler_output=fused_features)
        elif hasattr(vision_outputs, 'pooler_output'):
            vision_outputs.pooler_output = fused_features
        result = self.base_reward_model.model(
            vision_outputs=vision_outputs, pixel_values=None, return_dict=True
        )
        return result.logits

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
    model = PEFTImageReward().to(device)
    print(f"Model loaded on device: {device}")

    # Freeze original reward model
    for param in model.original_reward_model.parameters():
        param.requires_grad = False
    
    # Create dataloader using prompts
    print(f"Loading data using prompts from: {prompts_file}")
    try:
        dataloader = load_diffusion_dataloader(
            prompts_file=prompts_file,
            batch_size=batch_size,
            image_size=image_size,
            shuffle=True
        )
        dataset_size = len(dataloader.dataset)
        print(f"Data loaded. Dataset size: {dataset_size} prompts.")
        if dataset_size == 0:
            print("ERROR: Dataloader loaded an empty dataset. Check prompts file and dataloader logic.")
            return None
    except Exception as e:
        print(f"ERROR: Failed to load dataloader: {e}")
        print(traceback.format_exc())
        return None

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
         print("ERROR: No trainable parameters found in the model. Check PEFT setup.")
         return None
    print(f"Optimizing {len(trainable_params)} trainable parameters.")
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

                final_images = final_images.to(device)
                intermediate_images = intermediate_images.to(device)
                timesteps = timesteps.to(device)

                with torch.no_grad():
                    model.original_reward_model.to(device).eval()
                    target_rewards = model.original_reward_model.score(final_images)
                    if target_rewards is None:
                        print(f"Warning: Got None for target_rewards at batch {i}. Skipping.")
                        continue
                    
                predicted_rewards = model(intermediate_images, timesteps)
                if predicted_rewards is None:
                     print(f"Warning: Got None for predicted_rewards at batch {i}. Skipping.")
                     continue

                if predicted_rewards.shape != target_rewards.shape:
                    print(f"Warning: Shape mismatch at batch {i}. Predicted: {predicted_rewards.shape}, Target: {target_rewards.shape}. Skipping.")
                    continue
                
                loss = loss_fn(predicted_rewards, target_rewards)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_total_loss += loss.item()
                num_batches += 1

                if (i + 1) % 50 == 0:
                     current_avg_loss = epoch_total_loss / num_batches
                     print(f"  Epoch {epoch+1}, Batch {i+1}/{len(dataloader)}, Current Batch Loss: {loss.item():.6f}, Epoch Avg Loss: {current_avg_loss:.6f}")

            except Exception as batch_e:
                print(f"\nERROR during Epoch {epoch+1}, Batch {i+1}: {batch_e}")
                print(traceback.format_exc())
                print("Skipping this batch and continuing training...")
                continue

        avg_loss = epoch_total_loss / num_batches if num_batches > 0 else 0
        print(f"--- Epoch {epoch+1}/{num_epochs} finished. Average Training Loss: {avg_loss:.6f} ---")
        
        epochs_completed.append(epoch + 1)
        epoch_losses.append(avg_loss)

        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt")
        try:
            checkpoint_data = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_loss,
            }
            torch.save(checkpoint_data, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
        except Exception as save_e:
             print(f"ERROR: Failed to save checkpoint for epoch {epoch+1}: {save_e}")

        plot_loss_to_terminal(epochs_completed, epoch_losses)
        
    final_model_path = os.path.join(os.path.dirname(checkpoint_dir), "final_model_state_dict.pt")
    try:
        torch.save(model.state_dict(), final_model_path)
        print(f"Final model state dictionary saved to {final_model_path}")
    except Exception as final_save_e:
        print(f"ERROR: Failed to save final model state dictionary: {final_save_e}")

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

