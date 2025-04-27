import torch
import torch.nn as nn
import ImageReward as reward
from peft import LoraConfig, TaskType, inject_adapter_in_model

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

        # --- EDIT: Check for vis_processor OR image_processor under blip ---
        vis_processor = None
        # Prepare image using the base model's processor first
        if hasattr(self.base_reward_model, 'blip'):
            if hasattr(self.base_reward_model.blip, 'vis_processor'):
                vis_processor = self.base_reward_model.blip.vis_processor
                print("Found processor as 'vis_processor' under 'blip'.") # Debug
            elif hasattr(self.base_reward_model.blip, 'image_processor'): # Check alternative name
                vis_processor = self.base_reward_model.blip.image_processor
                print("Found processor as 'image_processor' under 'blip'.") # Debug

        if vis_processor:
            # Move image tensor to the device where vision_encoder expects it
            device = next(self.vision_encoder.parameters()).device
            # Apply the processor
            processed_image = vis_processor(image.to(device))
            print(f"Applied processor. Processed image shape: {processed_image.shape}") # Debug print
        else:
             # If no processor found after checking common names/locations, raise an error
             raise AttributeError("Cannot find base_reward_model's visual processor (checked blip.vis_processor, blip.image_processor). Image processing is required.")
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