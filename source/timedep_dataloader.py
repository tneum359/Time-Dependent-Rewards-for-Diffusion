import torch
from torch.utils.data import Dataset, DataLoader
from diffusers import FluxPipeline
import random
import os
from huggingface_hub import login

class TimeDependentDataset(Dataset):
    def __init__(self, model_name="flux1.dev", batch_size=4, image_size=1024, device="cuda" if torch.cuda.is_available() else "cpu", hf_token=None):
        """
        Simplified wrapper for Flux1.dev model that returns triplets of 
        (fully denoised image, intermediate denoised image, timestep).
        
        Args:
            model_name (str): Model identifier, "flux1.dev" uses the default Flux model
            batch_size (int): Number of samples to generate in each batch
            image_size (int): Size of generated images
            device (str): Device to use for computation
            hf_token (str): Hugging Face token for authentication
        """
        super().__init__()
        self.batch_size = batch_size
        self.image_size = image_size
        self.device = device
        
        # Check for HF token and authenticate if provided
        if hf_token:
            login(token=hf_token)
        elif os.environ.get("HF_TOKEN"):
            login(token=os.environ.get("HF_TOKEN"))
        else:
            print("Warning: No Hugging Face token provided. You may encounter authentication errors.")
            print("Set the HF_TOKEN environment variable or pass hf_token to the constructor.")
        
        # Load Flux pipeline
        try:
            self.pipeline = FluxPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-dev", 
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                token=hf_token
            )
            
            # Enable CPU offload to save VRAM
            self.pipeline.enable_model_cpu_offload()
        except Exception as e:
            print(f"Error loading Flux model: {e}")
            print("Make sure you have a valid Hugging Face token with access to this model.")
            raise
        
        # Total samples to generate for the dataset
        self.total_samples = 1000  # Can be adjusted as needed
        
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        """
        Generate a triplet of (fully denoised image, intermediate denoised image, timestep).
        Replicates Flux pipeline's final decoding steps for intermediate latents.
        """
        num_inference_steps = 30
        random_step = random.randint(1, num_inference_steps - 1)
        captured_timestep = torch.tensor(random_step)
        intermediate_latents = None

        # Define callback to capture raw latents
        def capture_callback(pipe, step, timestep, callback_kwargs):
            nonlocal intermediate_latents
            if step == random_step:
                latents = callback_kwargs["latents"].detach().clone()
                print(f"Captured raw intermediate latents at step {step}, shape: {latents.shape}")
                intermediate_latents = latents
            return callback_kwargs

        # Generate final image and capture intermediate latent
        with torch.no_grad():
            output = self.pipeline(
                prompt="", height=self.image_size, width=self.image_size,
                guidance_scale=3.5, num_inference_steps=num_inference_steps,
                max_sequence_length=512, output_type="pt",
                callback_on_step_end=capture_callback,
                callback_on_step_end_tensor_inputs=["latents"]
            )
        final_image = output.images[0]

        # Decode intermediate latents using Flux's exact procedure
        if intermediate_latents is not None:
            try:
                print(f"Processing intermediate latents (initial shape: {intermediate_latents.shape})")
                latents_to_process = intermediate_latents

                # --- Step 1: Unpack Latents ---
                # Check if the pipeline has the specific _unpack_latents method
                if hasattr(self.pipeline, "_unpack_latents"):
                    try:
                        # Calculate the VAE scale factor based on its architecture (depth)
                        vae_scale_factor = 2 ** (len(self.pipeline.vae.config.block_out_channels) - 1)
                        print(f"Attempting to unpack latents using _unpack_latents with calculated vae_scale_factor={vae_scale_factor}...")
                        latents_to_process = self.pipeline._unpack_latents(
                            latents_to_process,
                            self.image_size, # target height
                            self.image_size, # target width
                            vae_scale_factor
                        )
                        print(f"Unpacked latents shape: {latents_to_process.shape}")
                    except Exception as unpack_e:
                        print(f"WARNING: _unpack_latents failed: {unpack_e}. Attempting direct reshape as fallback.")
                        # Fallback: Try direct reshape if element count matches expected VAE input
                        latent_channels = self.pipeline.vae.config.latent_channels
                        latent_height = self.image_size // 8
                        latent_width = self.image_size // 8
                        expected_shape = (1, latent_channels, latent_height, latent_width)
                        if latents_to_process.numel() == (1 * latent_channels * latent_height * latent_width):
                             print(f"Attempting direct reshape to {expected_shape}")
                             latents_to_process = latents_to_process.reshape(expected_shape)
                        else:
                            raise ValueError(f"Latent shape {latents_to_process.shape} cannot be unpacked or reshaped.") from unpack_e
                else:
                    print("Pipeline does not have _unpack_latents. Attempting direct reshape based on element count.")
                    # If no unpack method, rely solely on reshape if elements match
                    latent_channels = self.pipeline.vae.config.latent_channels
                    latent_height = self.image_size // 8
                    latent_width = self.image_size // 8
                    expected_shape = (1, latent_channels, latent_height, latent_width)
                    if latents_to_process.numel() == (1 * latent_channels * latent_height * latent_width):
                         print(f"Attempting direct reshape to {expected_shape}")
                         latents_to_process = latents_to_process.reshape(expected_shape)
                    else:
                        raise ValueError(f"Latent shape {latents_to_process.shape} cannot be reshaped and _unpack_latents not found.")

                # --- Step 2: Inverse Scale & Shift ---
                scaling_factor = self.pipeline.vae.config.scaling_factor
                shift_factor = getattr(self.pipeline.vae.config, "shift_factor", 0.0) # Default to 0 if missing
                print(f"Applying inverse scale/shift: scaling_factor={scaling_factor}, shift_factor={shift_factor}")
                
                # Ensure correct dtype BEFORE scaling/shifting
                model_dtype = self.pipeline.vae.dtype
                print(f"Ensuring latents are dtype: {model_dtype}")
                latents_to_process = latents_to_process.to(dtype=model_dtype)

                # Apply the transformation: z = x / scale + shift
                latents_for_decode = latents_to_process / scaling_factor + shift_factor
                print(f"Latents prepared for VAE (shape: {latents_for_decode.shape}, dtype: {latents_for_decode.dtype})")

                # --- Step 3: VAE Decode ---
                print("Decoding latents using VAE...")
                with torch.no_grad():
                    # Decode expects input shape [B, C, H, W]
                    decoded_output = self.pipeline.vae.decode(latents_for_decode, return_dict=False)
                    decoded_image_tensor = decoded_output[0] # Extract sample tensor
                print(f"Decoded VAE sample shape: {decoded_image_tensor.shape}, dtype: {decoded_image_tensor.dtype}")

                # --- Step 4: Post-Processing ---
                print("Post-processing decoded image using ImageProcessor...")
                # Use the pipeline's image_processor for correct denormalization etc.
                intermediate_image = self.pipeline.image_processor.postprocess(
                    decoded_image_tensor,
                    output_type="pt" # Keep as tensor [B, C, H, W]
                )
                # Postprocess returns a list, take the first element
                if isinstance(intermediate_image, list):
                    intermediate_image = intermediate_image[0]

                # Remove batch dimension -> [C, H, W]
                if intermediate_image.dim() == 4 and intermediate_image.shape[0] == 1:
                     intermediate_image = intermediate_image.squeeze(0)

                print(f"Successfully decoded intermediate image (shape: {intermediate_image.shape}, dtype: {intermediate_image.dtype})")

            except Exception as e:
                import traceback
                print(f"Error decoding intermediate latents following Flux steps: {e}")
                print(traceback.format_exc())
                print("Using final image as fallback")
                intermediate_image = final_image.clone()
        else:
            print("Warning: Could not capture intermediate state, using final image as fallback")
            intermediate_image = final_image.clone()

        return final_image, intermediate_image, captured_timestep
    
    def get_dataloader(self, shuffle=True, num_workers=0):
        """Return a DataLoader using this dataset"""
        return DataLoader(
            self,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )

def load_diffusion_dataloader(batch_size=4, image_size=1024, shuffle=True, num_workers=0, hf_token=None):
    """Helper function to create and return a dataloader for Flux model"""
    dataset = TimeDependentDataset(
        batch_size=batch_size, 
        image_size=image_size, 
        hf_token=hf_token
    )
    return dataset.get_dataloader(shuffle=shuffle, num_workers=num_workers)