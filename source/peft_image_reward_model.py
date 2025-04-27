import torch
import torch.nn as nn
import ImageReward as reward
from peft import LoraConfig, TaskType, inject_adapter_in_model, get_peft_model

class PEFTImageReward(nn.Module):
    def __init__(self, base_model_path="ImageReward-v1.0", timestep_dim=320, text_model_name="roberta-base"):
        super().__init__()
        print(f"Initializing PEFTImageReward - Base: {base_model_path}")
        self.base_reward_model = reward.load(base_model_path)
        self.original_reward_model = reward.load(base_model_path)
        self.original_reward_model.eval()

        # --- Debug: Inspect the loaded base_reward_model --- 
        print("--- Debugging base_reward_model --- ")
        try:
            print(f"Type: {type(self.base_reward_model)}")
            print(f"Attributes/Methods: {dir(self.base_reward_model)}")
            if hasattr(self.base_reward_model, 'blip'):
                 print(f"BLIP component attributes: {dir(self.base_reward_model.blip)}")
        except Exception as e:
            print(f"Error during debug inspection: {e}")
        print("--- End Debugging --- ")
        # --- End Debug ---

        # --- References to Base Model Components ---
        # Vision Encoder (Will be adapted)
        if not hasattr(self.base_reward_model, 'blip') or not hasattr(self.base_reward_model.blip, 'visual_encoder'):
            raise AttributeError("Cannot find '.blip.visual_encoder'.")
        self.vision_encoder = self.base_reward_model.blip.visual_encoder
        print("Identified vision component as '.blip.visual_encoder'")
        self.vision_feature_dim = self.vision_encoder.embed_dim if hasattr(self.vision_encoder, 'embed_dim') else self.vision_encoder.config.hidden_size
        print(f"Vision feature dimension: {self.vision_feature_dim}")

        # Text Encoder & Projection (Frozen)
        if not hasattr(self.base_reward_model.blip, 'text_encoder') or not hasattr(self.base_reward_model.blip, 'text_proj'):
             raise AttributeError("Cannot find '.blip.text_encoder' or '.blip.text_proj'.")
        self.text_encoder = self.base_reward_model.blip.text_encoder
        self.text_proj = self.base_reward_model.blip.text_proj
        self.text_feature_dim = self.text_proj.out_features
        print(f"Text Encoder & Projection found. Projected Text dim: {self.text_feature_dim}")

        # Reward Head MLP (Will be adapted)
        if not hasattr(self.base_reward_model, 'mlp'):
             raise AttributeError("Cannot find 'mlp' reward head.")
        self.reward_head = self.base_reward_model.mlp
        try: first_layer_of_mlp = self.reward_head.layers[0]; self.reward_head_in_dim = first_layer_of_mlp.in_features
        except Exception: raise AttributeError("Cannot get input dim for 'mlp' reward head.")
        print(f"Identified reward head: 'mlp' (expects input dim: {self.reward_head_in_dim})")

        # --- Visual Processor (Crucial!) ---
        self.vis_processor = None # Initialize to None
        if hasattr(self.base_reward_model, 'vis_processor'):
            self.vis_processor = self.base_reward_model.vis_processor
            print("Found visual processor as 'vis_processor' directly on base model.")
        elif hasattr(self.base_reward_model, 'image_processor'):
            self.vis_processor = self.base_reward_model.image_processor
            print("Found visual processor as 'image_processor' directly on base model.")
        elif hasattr(self.base_reward_model, 'blip') and hasattr(self.base_reward_model.blip, 'vis_processor'):
             self.vis_processor = self.base_reward_model.blip.vis_processor
             print("Found visual processor as 'vis_processor' under 'blip'.")
        elif hasattr(self.base_reward_model, 'blip') and hasattr(self.base_reward_model.blip, 'image_processor'):
             self.vis_processor = self.base_reward_model.blip.image_processor
             print("Found visual processor as 'image_processor' under 'blip'.")
        # else:
        #      # Temporarily comment out the error to see debug prints
        #      # raise AttributeError("Cannot find base_reward_model's visual processor (checked multiple locations). Required for processing image.")
        if self.vis_processor is None:
            print("WARNING: Visual processor not found in expected locations. Will likely cause errors later.")

        # --- Trainable Components ---
        self.timestep_embedding = TimestepEmbedding(timestep_dim)
        print(f"Timestep Embedding dim: {timestep_dim}")

        total_fused_dim = self.vision_feature_dim + timestep_dim + self.text_feature_dim
        intermediate_fusion_dim = self.vision_feature_dim
        self.fusion_layer = nn.Linear(total_fused_dim, intermediate_fusion_dim)
        print(f"Fusion layer input dim: {total_fused_dim}, output dim: {intermediate_fusion_dim}")

        self.fusion_to_reward_proj = nn.Linear(intermediate_fusion_dim, self.reward_head_in_dim)
        print(f"Added projection layer: {intermediate_fusion_dim} -> {self.reward_head_in_dim}")

        # --- PEFT Configs --- 
        # Vision Encoder PEFT
        peft_config_vision = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION, r=16, lora_alpha=32, lora_dropout=0.1,
            target_modules=["qkv", "proj"] # Common targets for ViT
        )
        print("Applying LoRA to Vision Encoder...")
        self.vision_encoder = inject_adapter_in_model(peft_config_vision, self.vision_encoder)
        # self.vision_encoder.print_trainable_parameters() # Optional: print details

        # MLP Head PEFT
        peft_config_mlp = LoraConfig(
            task_type=TaskType.SEQ_CLS, r=8, lora_alpha=16, lora_dropout=0.1,
        )
        print("Applying LoRA to MLP Reward Head...")
        self.reward_head = get_peft_model(self.reward_head, peft_config_mlp)
        # self.reward_head.print_trainable_parameters() # Optional: print details

        # --- Freeze Parameters --- 
        # Freeze base model EXCEPT adapted layers' original weights (handled by PEFT)
        for name, param in self.base_reward_model.named_parameters():
            param.requires_grad = False
            
        # Unfreeze custom layers and ensure LoRA adapters are trainable
        for param in self.timestep_embedding.parameters(): param.requires_grad = True
        for param in self.fusion_layer.parameters(): param.requires_grad = True
        for param in self.fusion_to_reward_proj.parameters(): param.requires_grad = True
        
        # Explicitly unfreeze LoRA params in vision_encoder (inject_adapter_in_model might not set requires_grad)
        for name, param in self.vision_encoder.named_parameters():
             if 'lora' in name:
                  param.requires_grad = True
                  
        # LoRA params in reward_head are handled by get_peft_model, should be trainable.
        # Verify by checking requires_grad if needed.

        # Freeze original_reward_model used for scoring target
        for param in self.original_reward_model.parameters(): param.requires_grad = False

        print("PEFTImageReward model configuration complete (Vision + MLP adapted).")
        print(f"Total Trainable Params: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")

    def forward(self, intermediate_image, timestep, input_ids, attention_mask):
        """ Forward pass using intermediate image, timestep, and text. Adapts Vision+MLP. """
        # 1. Process Intermediate Image using found processor
        device = next(self.parameters()).device # Use device where model parameters are
        processed_intermediate_image = self.vis_processor(intermediate_image.to(device))
        # print(f"Processed intermediate shape: {processed_intermediate_image.shape}") # Debug shape

        # 2. Get Image Features from PEFT-adapted Vision Encoder
        # No need for torch.no_grad() here as vision_encoder has trainable LoRA layers
        vision_outputs = self.vision_encoder(processed_intermediate_image)
        image_features = vision_outputs.pooler_output # Use pooled output [B, D_vis]

        # 3. Process Text (using frozen components)
        with torch.no_grad():
            text_output = self.text_encoder.to(device)(
                input_ids=input_ids.to(device), attention_mask=attention_mask.to(device), return_dict=True
            )
            text_features = text_output.last_hidden_state[:, 0, :] # Use [CLS] token [B, D_text_raw]
            projected_text_features = self.text_proj.to(device)(text_features) # [B, D_text_proj]

        # 4. Get Timestep Embedding (trainable)
        timestep_emb = self.timestep_embedding(timestep) # [B, D_ts]

        # 5. Fuse Features (trainable layers)
        combined_features = torch.cat([
            image_features.to(timestep_emb.device), # Ensure devices match for cat
            timestep_emb, 
            projected_text_features.to(timestep_emb.device)
        ], dim=1)
        fused_features = self.fusion_layer(combined_features) # [B, D_inter_fusion]
        
        # 6. Project to Reward Head Input Dimension (trainable layer)
        projected_features = self.fusion_to_reward_proj(fused_features) # [B, D_reward_in]

        # 7. Pass through Adapted Reward Head MLP (trainable adapters)
        reward_score = self.reward_head(projected_features)

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