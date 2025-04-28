import torch
import torch.nn as nn
import ImageReward as reward
from peft import LoraConfig, TaskType, inject_adapter_in_model, get_peft_model
from torchvision.transforms.functional import to_pil_image

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
        # --- Debug Text Encoder Config ---
        try:
            print("--- Debugging Text Encoder Config ---")
            if hasattr(self.text_encoder, 'config'):
                 print(self.text_encoder.config)
                 # Potentially try modifying the config here if a relevant flag is found
                 # if hasattr(self.text_encoder.config, 'is_decoder') and self.text_encoder.config.is_decoder:
                 #      print("Attempting to set is_decoder = False")
                 #      self.text_encoder.config.is_decoder = False
                 # --- EDIT: Modify config --- 
                 if hasattr(self.text_encoder.config, 'add_cross_attention') and self.text_encoder.config.add_cross_attention:
                      print("Attempting to set add_cross_attention = False")
                      self.text_encoder.config.add_cross_attention = False
                      print(f"Config updated: add_cross_attention = {self.text_encoder.config.add_cross_attention}")
                 # --- End EDIT --- 
            else:
                 print("Text encoder does not have a .config attribute.")
            print("--- End Text Encoder Config Debug ---")
        except Exception as e:
            print(f"Error inspecting text encoder config: {e}")
        # --- End Debug ---

        # Reward Head MLP (Frozen for now)
        if not hasattr(self.base_reward_model, 'mlp'):
             raise AttributeError("Cannot find 'mlp' reward head.")
        self.reward_head = self.base_reward_model.mlp
        try: first_layer_of_mlp = self.reward_head.layers[0]; self.reward_head_in_dim = first_layer_of_mlp.in_features
        except Exception: raise AttributeError("Cannot get input dim for 'mlp' reward head.")
        print(f"Identified reward head: 'mlp' (expects input dim: {self.reward_head_in_dim})")

        # --- Image Preprocessing Method (Crucial!) ---
        if not hasattr(self.base_reward_model, 'preprocess'):
             raise AttributeError("Cannot find 'preprocess' method on base_reward_model.")
        self.preprocess_image = self.base_reward_model.preprocess
        print("Found 'preprocess' method on base model.")
        # --- End Preprocessing --- 

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
        
        # MLP Head PEFT - REMOVED FOR NOW
        # print("--- Debugging MLP Reward Head Structure ---")
        # print(self.reward_head)
        # print("--- End MLP Debug ---")
        # mlp_target_modules = ["0", "2", "4", "6", "7"] # Based on debug output
        # print(f"Attempting to target modules in MLP: {mlp_target_modules}")
        # peft_config_mlp = LoraConfig(
        #     task_type=TaskType.SEQ_CLS, r=8, lora_alpha=16, lora_dropout=0.1,
        #     target_modules=mlp_target_modules 
        # )
        # print("Applying LoRA to MLP Reward Head...")
        # self.reward_head = get_peft_model(self.reward_head, peft_config_mlp)
        
        # --- Freeze Parameters --- 
        # Freeze base model EXCEPT adapted vision encoder layers' original weights (handled by PEFT? No, inject doesn't handle base freezing)
        for name, param in self.base_reward_model.named_parameters():
            param.requires_grad = False
            
        # Unfreeze custom layers and Vision Encoder LoRA adapters
        for param in self.timestep_embedding.parameters(): param.requires_grad = True
        for param in self.fusion_layer.parameters(): param.requires_grad = True
        for param in self.fusion_to_reward_proj.parameters(): param.requires_grad = True
        
        # Explicitly unfreeze LoRA params in vision_encoder (inject_adapter_in_model might not set requires_grad)
        for name, param in self.vision_encoder.named_parameters():
             if 'lora' in name:
                  param.requires_grad = True
                  
        # Ensure MLP head remains frozen for now
        for param in self.reward_head.parameters():
            param.requires_grad = False

        # Freeze original_reward_model used for scoring target
        for param in self.original_reward_model.parameters(): param.requires_grad = False

        print("PEFTImageReward model configuration complete (Vision Encoder adapted).") # Updated message
        print(f"Total Trainable Params: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")

    def forward(self, intermediate_image, timestep, input_ids, attention_mask):
        """ Forward pass using preprocessed intermediate image, timestep, and text. Adapts Vision Encoder. """
        device = next(self.parameters()).device 
        
        # --- Debug --- 
        print(f"[DEBUG] Intermediate image input shape: {intermediate_image.shape}")
        # --- End Debug --- 
        
        # Ensure intermediate_image is a single image tensor [C, H, W] (assuming batch_size=1)
        if intermediate_image.shape[0] != 1:
            raise ValueError(f"This forward pass currently assumes batch_size=1, but got batch size {intermediate_image.shape[0]}")
        
        img_tensor_cpu = intermediate_image.squeeze(0).cpu().to(torch.float32) # Remove batch dim, move to CPU, ensure float32
            
        try:
             # 1. Convert SINGLE tensor to PIL
             img_pil = to_pil_image(img_tensor_cpu)
             # --- Debug --- 
             print(f"[DEBUG] Type of img_pil: {type(img_pil)}")
             print(f"[DEBUG] Type of self.preprocess_image: {type(self.preprocess_image)}")
             # --- End Debug --- 

             # 2. Preprocess SINGLE PIL image using the model's method
             processed_intermediate_image = self.base_reward_model.preprocess(img_pil) # Try calling directly

             # 3. Ensure processed tensor is on the correct device and add batch dim back
             # Assuming preprocess returns [C, 224, 224]
             processed_intermediate_image = processed_intermediate_image.unsqueeze(0).to(device)
             print(f"Preprocessed intermediate image shape: {processed_intermediate_image.shape}") # Debug

        except Exception as e:
             print(f"ERROR during preprocessing: {e}")
             print("Check input/output of self.preprocess_image on single PIL.")
             raise

        # 4. Get Image Features from PEFT-adapted Vision Encoder
        vision_outputs = self.vision_encoder(processed_intermediate_image) # Input should be [1, C, 224, 224]
        image_features = vision_outputs[:, 0] # Use CLS token output [1, D_vis]

        # 3. Process Text (using frozen components)
        with torch.no_grad():
             # Assuming input_ids/attention_mask are already [1, SeqLen]
            text_output = self.text_encoder.to(device)(
                input_ids=input_ids.to(device), 
                attention_mask=attention_mask.to(device), 
                encoder_hidden_states=None, # Tell the encoder we are not providing cross-attention states
                return_dict=True
            )
            text_features = text_output.last_hidden_state[:, 0, :] # Use [CLS] token [1, D_text_raw]
            projected_text_features = self.text_proj.to(device)(text_features) # [1, D_text_proj]

        # 4. Get Timestep Embedding (trainable)
        timestep_emb = self.timestep_embedding(timestep.to(device)) # Ensure timestep is on device [1, D_ts]

        # 5. Fuse Features (trainable layers)
        # Ensure all features have batch dim 1
        combined_features = torch.cat([
            image_features.to(timestep_emb.device), 
            timestep_emb,
            projected_text_features.to(timestep_emb.device)
        ], dim=1) # Should be [1, D_vis + D_ts + D_text_proj]
        fused_features = self.fusion_layer(combined_features) # [1, D_inter_fusion]

        # 6. Project to Reward Head Input Dimension (trainable layer)
        projected_features = self.fusion_to_reward_proj(fused_features) # [1, D_reward_in]

        # 7. Pass through ORIGINAL Reward Head MLP (frozen)
        with torch.no_grad():
             reward_score = self.reward_head(projected_features) # [1, 1] or [1]

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