import ImageReward as reward
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from peft import get_peft_model, LoraConfig, TaskType
import sys
import os
from evaluate import eval_callback

# Add parent directory to path to import dataloader
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from source.timedep_dataloader import load_diffusion_dataloader

class PEFTImageReward(nn.Module):
    def __init__(self, base_model_path="ImageReward/ImageReward", timestep_dim=320):
        super().__init__()
        
        # Load base ImageReward model
        self.base_reward_model = reward.load(base_model_path)
        
        # Original ImageReward model for scoring fully denoised images
        self.original_reward_model = reward.load(base_model_path)
        self.original_reward_model.eval()  # Set to eval mode as we won't train this
        
        # Extract the vision encoder part of the model for PEFT tuning
        self.vision_encoder = self.base_reward_model.model.vision_model
        
        # Timestep embedding layer using sinusoidal positional encoding
        self.timestep_embedding = TimestepEmbedding(timestep_dim)
        
        # Feature dimension from the vision encoder
        self.vision_feature_dim = self.base_reward_model.model.vision_model.config.hidden_size
        
        # Fusion layer to combine image features with timestep embedding
        self.fusion_layer = nn.Linear(self.vision_feature_dim + timestep_dim, self.vision_feature_dim)
        
        # Apply PEFT configuration
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=16,  # LoRA rank
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["query", "key", "value", "projection"]
        )
        
        # Wrap the vision encoder with PEFT
        self.vision_encoder = get_peft_model(self.vision_encoder, peft_config)
        
    def forward(self, image, timestep):
        """
        Forward pass for the PEFT model predicting reward from intermediate image
        
        Args:
            image: Intermediate denoised image
            timestep: Diffusion timestep when the intermediate image was captured
            
        Returns:
            Predicted reward score
        """
        # Get timestep embedding
        timestep_emb = self.timestep_embedding(timestep)
        
        # Get image features from vision encoder
        vision_outputs = self.vision_encoder(image)
        image_features = vision_outputs.pooler_output  # Shape: [batch_size, vision_feature_dim]
        
        # Concatenate image features with timestep embedding
        combined_features = torch.cat([image_features, timestep_emb], dim=1)
        
        # Fuse features
        fused_features = self.fusion_layer(combined_features)
        
        # Method 1: If vision_outputs is a dataclass or has a replace method
        if hasattr(vision_outputs, '_replace'):
            # For namedtuples or similar
            vision_outputs = vision_outputs._replace(pooler_output=fused_features)
        # Method 2: If vision_outputs is a dictionary-like object
        elif hasattr(vision_outputs, 'pooler_output'):
            vision_outputs.pooler_output = fused_features
        
        # Replace the original vision features in the model's forward pass
        # By using the modified feature representation with fused features
        result = self.base_reward_model.model(
            vision_outputs=vision_outputs,
            pixel_values=None,  # We already processed the image
            return_dict=True
        )
        
        # Get the reward score (mapped to a single value)
        reward_score = result.logits
        
        return reward_score

class TimestepEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        # Timestep projection layer
        self.proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
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
        # Scale timesteps to be between 0 and 1
        timestep = timestep.float() / 1000.0  # Assuming max timestep is 1000
        
        # Create sinusoidal embedding frequencies
        freqs = torch.exp(
            -torch.arange(half_dim, device=timestep.device) * torch.log(torch.tensor(10000.0)) / half_dim
        )
        
        # Reshape for broadcasting
        timestep = timestep.view(-1, 1)
        freqs = freqs.view(1, -1)
        
        # Create sinusoidal embeddings
        args = timestep * freqs
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        
        # Ensure correct dimension if dim is odd
        if self.dim % 2 == 1:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
            
        # Project to final dimension
        return self.proj(embedding)

def train(
    batch_size=4,
    learning_rate=1e-4,
    num_epochs=10,
    image_size=512,
    device="cuda" if torch.cuda.is_available() else "cpu",
    save_path="finetuned_reward_model.pt"
):
    """
    Train the PEFT-wrapped ImageReward model
    """
    # Create model
    model = PEFTImageReward().to(device)
    
    # Freeze original reward model
    for param in model.original_reward_model.parameters():
        param.requires_grad = False
    
    # Create dataloader
    dataloader = load_diffusion_dataloader(
        batch_size=batch_size,
        image_size=image_size
    )
    
    # Set up optimizer and loss function
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=0.01
    )
    loss_fn = nn.MSELoss()
    
    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0
        
        for final_images, intermediate_images, timesteps in dataloader:
            # Move data to device
            final_images = final_images.to(device)
            intermediate_images = intermediate_images.to(device)
            timesteps = timesteps.to(device)
            
            # Get target reward score from original model on fully denoised images
            with torch.no_grad():
                target_rewards = model.original_reward_model.score(final_images)
                
            # Get predicted reward from intermediate images
            predicted_rewards = model(intermediate_images, timesteps)
            
            # Calculate loss
            loss = loss_fn(predicted_rewards, target_rewards)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            num_batches += 1
            
        # Print epoch statistics
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")
        
        # Evaluate after each epoch
        eval_callback(epoch_num=epoch+1)
        
    # Save the fine-tuned model
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    
    return model

if __name__ == "__main__":
    train()

