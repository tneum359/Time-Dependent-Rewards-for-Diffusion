import torch
from torch.utils.data import Dataset, DataLoader
from diffusers import FluxPipeline
import random
import os
from huggingface_hub import login
import traceback # For detailed error printing

class TimeDependentDataset(Dataset):
    def __init__(self,
                 prompts_file="prompts.txt",
                 model_name="flux1.dev",
                 image_size=1024,
                 # Default to GPU if available, otherwise CPU
                 device="cuda" if torch.cuda.is_available() else "cpu", 
                 hf_token=None):
        """
        Generates triplets of (final_image, intermediate_image, timestep) based on prompts
        using a single-pass approach with callbacks and latent decoding. Optimized for GPU.
        """
        super().__init__()
        self.image_size = image_size
        # Store the target device
        self.device = torch.device(device) 
        print(f"Dataset configured to use device: {self.device}")

        # --- Read Prompts ---
        self.prompts = []
        try:
            # Check paths for prompts file
            script_dir = os.path.dirname(__file__)
            paths_to_check = [
                prompts_file,
                os.path.join(script_dir, prompts_file),
                os.path.join(script_dir, '..', prompts_file)
            ]
            found_path = None
            for path in paths_to_check:
                 if os.path.exists(path):
                      found_path = path
                      break
            if found_path is None:
                 raise FileNotFoundError(f"Prompts file not found in checked paths: {paths_to_check}")
            prompts_file = found_path # Use the found path

            with open(prompts_file, 'r') as f:
                self.prompts = [line.strip() for line in f if line.strip()]
            if not self.prompts:
                 raise ValueError(f"No valid prompts found in {prompts_file}")
            print(f"Loaded {len(self.prompts)} prompts from {prompts_file}")
        except FileNotFoundError:
            print(f"Error: Prompts file not found.")
            raise
        except Exception as e:
            print(f"Error reading prompts file {prompts_file}: {e}")
            raise
        # --- End Read Prompts ---

        self.total_samples = len(self.prompts)

        # --- HF Login ---
        if hf_token:
            login(token=hf_token)
        elif os.environ.get("HF_TOKEN"):
            login(token=os.environ.get("HF_TOKEN"))
        else:
            print("Warning: No Hugging Face token provided.")
        # --- End HF Login ---
        
        # --- Load Flux Pipeline ---
        try:
            print(f"Loading Flux pipeline ({model_name}) to device: {self.device}...")
            # Determine dtype based on device
            self.pipeline_dtype = torch.bfloat16 if self.device.type == 'cuda' else torch.float32
            print(f"Using dtype: {self.pipeline_dtype}")
            
            # Load pipeline explicitly to the target device
            self.pipeline = FluxPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-dev", # Assuming model_name corresponds to this
                torch_dtype=self.pipeline_dtype,
                # variant="bf16" # Example: if specific variant exists for dtype
            ).to(self.device) # Explicitly move the whole pipeline object

            # Enable CPU offload AFTER moving to GPU if VRAM is a concern
            # This offloads components back to CPU as needed during inference
            self.pipeline.enable_model_cpu_offload()
            print("Flux pipeline loaded and configured with model CPU offload.")
            
            # Small test call to ensure pipeline is working (optional)
            # print("Performing a small test generation...")
            # _ = self.pipeline(prompt="test", num_inference_steps=2, height=64, width=64)
            # print("Test generation successful.")

        except Exception as e:
            print(f"Error loading Flux model to {self.device}: {e}")
            print(traceback.format_exc())
            raise
        # --- End Load Flux Pipeline ---
        
    def __len__(self):
        return self.total_samples 
    
    def __getitem__(self, idx):
        """ Generate triplet, ensuring tensors are handled correctly for the device. """
        prompt = self.prompts[idx % len(self.prompts)] 
        num_inference_steps = 30 
        random_step = random.randint(1, num_inference_steps - 1)
        # Create timestep tensor directly on the target device (minor optimization)
        captured_timestep = torch.tensor(random_step, device=self.device) 
        
        intermediate_latents = None
        
        # Callback remains the same (captures latents which will be on the pipeline's compute device)
        def capture_callback(pipe, step, timestep, callback_kwargs):
            nonlocal intermediate_latents
            if step == random_step:
                latents = callback_kwargs["latents"].detach().clone()
                # print(f"Captured raw intermediate latents at step {step}, shape: {latents.shape}, device: {latents.device}") # Debug
                intermediate_latents = latents
            return callback_kwargs
        
        final_image = None
        intermediate_image = None

        # --- Generate Image (Pipeline handles internal device placement) ---
        try:
            # print(f"Generating image for prompt idx {idx} (step {random_step} captured)...") # Verbose
            with torch.no_grad():
                 # Ensure pipeline is on the correct device before call (usually redundant if loaded correctly)
                 # self.pipeline.to(self.device) 
                 output = self.pipeline(
                    prompt=prompt, height=self.image_size, width=self.image_size,
                    guidance_scale=3.5, num_inference_steps=num_inference_steps,
                    max_sequence_length=512, output_type="pt",  
                    callback_on_step_end=capture_callback,
                    callback_on_step_end_tensor_inputs=["latents"]
                 )
            # final_image tensor will be on the device where the final VAE step ran (likely GPU or CPU if offloaded)
            final_image = output.images[0] 
            # print(f"  Final image generated (device: {final_image.device}).") # Debug

            # --- Decode Intermediate Latents ---
            if intermediate_latents is not None:
                # print(f"Processing intermediate latents (initial shape: {intermediate_latents.shape}, device: {intermediate_latents.device})") # Debug
                # Ensure latents are on the main compute device for processing before VAE decode
                latents_to_process = intermediate_latents.to(self.device) 

                # Step 1: Unpack/Reshape (Logic remains the same)
                latent_channels = self.pipeline.vae.config.latent_channels
                latent_height = self.image_size // 8; latent_width = self.image_size // 8
                expected_shape = (1, latent_channels, latent_height, latent_width)
                numel_expected = 1 * latent_channels * latent_height * latent_width
                if hasattr(self.pipeline, "_unpack_latents"):
                    try:
                        vae_scale_factor = 2 ** (len(self.pipeline.vae.config.block_out_channels) - 1)
                        # print(f"Attempting to unpack latents...") # Verbose
                        latents_to_process = self.pipeline._unpack_latents(latents_to_process, self.image_size, self.image_size, vae_scale_factor)
                    except Exception as unpack_e:
                        # print(f"WARNING: _unpack_latents failed: {unpack_e}. Reshaping.") # Verbose
                        if latents_to_process.numel() == numel_expected: latents_to_process = latents_to_process.reshape(expected_shape)
                        else: raise ValueError(f"Cannot unpack/reshape {latents_to_process.shape} to {expected_shape}.") from unpack_e
                elif latents_to_process.numel() == numel_expected:
                     # print(f"Attempting direct reshape...") # Verbose
                     latents_to_process = latents_to_process.reshape(expected_shape)
                else: raise ValueError(f"Cannot reshape {latents_to_process.shape} to {expected_shape}.")

                # Step 2: Inverse Scale & Shift (on target device)
                scaling_factor = self.pipeline.vae.config.scaling_factor
                shift_factor = getattr(self.pipeline.vae.config, "shift_factor", 0.0)
                # print(f"Applying inverse scale/shift...") # Verbose
                model_dtype = self.pipeline.vae.dtype # Should match pipeline_dtype unless VAE differs
                latents_to_process = latents_to_process.to(dtype=model_dtype) # Match VAE dtype
                latents_for_decode = latents_to_process / scaling_factor + shift_factor
                # print(f"Latents prepared for VAE (device: {latents_for_decode.device})") # Debug

                # Step 3: VAE Decode (VAE is part of pipeline, handles its device placement)
                # print("Decoding latents using VAE...") # Verbose
                with torch.no_grad():
                    # Ensure VAE is on the correct device if offloading might have moved it
                    # self.pipeline.vae.to(self.device) 
                    decoded_output = self.pipeline.vae.decode(latents_for_decode, return_dict=False)
                    decoded_image_tensor = decoded_output[0] 
                # print(f"Decoded VAE sample (device: {decoded_image_tensor.device})") # Debug

                # Step 4: Post-Processing (ImageProcessor likely runs on CPU or matches input device)
                # print("Post-processing decoded image...") # Verbose
                # Ensure tensor is on the correct device for processor if needed (often not necessary)
                # decoded_image_tensor = decoded_image_tensor.to(self.device) 
                intermediate_image = self.pipeline.image_processor.postprocess(decoded_image_tensor, output_type="pt") 
                if isinstance(intermediate_image, list): intermediate_image = intermediate_image[0]
                if intermediate_image.dim() == 4 and intermediate_image.shape[0] == 1: intermediate_image = intermediate_image.squeeze(0)
                # print(f"Decoded intermediate image (device: {intermediate_image.device})") # Debug
            
            else: 
                 print(f"Warning: Intermediate latents not captured for idx {idx}. Using final image.")
                 intermediate_image = final_image.clone() # Use final image tensor

        except Exception as e:
             print(f"ERROR during generation/decoding for prompt idx {idx} ('{prompt[:50]}...'): {e}")
             print(traceback.format_exc())
             if final_image is None: raise RuntimeError(f"Failed generation entirely for idx {idx}") from e
             else: intermediate_image = final_image.clone() 

        if final_image is None or intermediate_image is None:
             raise RuntimeError(f"Generation failed for idx {idx}")

        # Return the 4-tuple, including the prompt string
        return final_image.cpu(), intermediate_image.cpu(), captured_timestep.cpu(), prompt

def load_diffusion_dataloader(
    prompts_file="prompts.txt", batch_size=4, image_size=1024, 
    shuffle=True, num_workers=0, hf_token=None, 
    # Added device argument
    device="cuda" if torch.cuda.is_available() else "cpu"): 
    """Helper function to create dataset and dataloader."""
    print(f"Creating dataset for device '{device}' with prompts from: {prompts_file}")
    dataset = TimeDependentDataset(
        prompts_file=prompts_file, image_size=image_size, 
        hf_token=hf_token, device=device # Pass device to dataset
    )
    
    print(f"Creating DataLoader with batch_size={batch_size}, shuffle={shuffle}, num_workers={num_workers}")
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, # MUST be 0 for GPU pipeline objects
        pin_memory=False # Safer with num_workers=0 and complex objects
    )