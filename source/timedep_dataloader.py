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
        
        Returns:
            tuple: (final_image, intermediate_image, timestep)
        """
        # Choose a random timestep between 1 and 29
        num_inference_steps = 30
        random_step = random.randint(1, num_inference_steps - 1)
        captured_timestep = torch.tensor(random_step)
        
        # Initialize variables
        intermediate_latents = None
        
        # Define callback function to capture results
        def capture_callback(pipe, step, timestep, callback_kwargs):
            nonlocal intermediate_latents
            
            # Capture at our target intermediate step
            if step == random_step:
                # Get raw latents from the scheduler output
                latents = callback_kwargs["latents"].detach().clone()
                print(f"Captured raw intermediate latents at step {step}, shape: {latents.shape}")
                intermediate_latents = latents
                
            return callback_kwargs
        
        # Generate image with callback
        with torch.no_grad():
            output = self.pipeline(
                prompt="",  # Empty prompt for random generation
                height=self.image_size,
                width=self.image_size,
                guidance_scale=3.5,
                num_inference_steps=num_inference_steps,
                max_sequence_length=512,
                output_type="pt",  # Get tensor output for final_image
                callback_on_step_end=capture_callback,
                callback_on_step_end_tensor_inputs=["latents"]
            )
        
        # Get the final image from the output
        final_image = output.images[0]
        
        # Convert intermediate latents to image using the correct Flux procedure
        if intermediate_latents is not None:
            try:
                print(f"Processing intermediate latents with shape: {intermediate_latents.shape}")
                
                # --- Step 1: Unpack Latents (Optional - Check if needed) ---
                # Based on previous errors, the shape [1, 4096, 64] might need reshaping first
                # Calculate expected VAE shape
                latent_channels = self.pipeline.vae.config.latent_channels # Usually 16 for Flux
                latent_height = self.image_size // 8
                latent_width = self.image_size // 8
                expected_shape = (1, latent_channels, latent_height, latent_width)
                numel_expected = 1 * latent_channels * latent_height * latent_width

                if intermediate_latents.numel() == numel_expected:
                     print(f"Reshaping intermediate latents from {intermediate_latents.shape} to {expected_shape}")
                     intermediate_latents = intermediate_latents.reshape(expected_shape)
                else:
                     # If unpacking is needed and available:
                     # if hasattr(self.pipeline, "_unpack_latents") and hasattr(self.pipeline, "vae_scale_factor"):
                     #      print("Using _unpack_latents...")
                     #      intermediate_latents = self.pipeline._unpack_latents(intermediate_latents, self.image_size, self.image_size, self.pipeline.vae_scale_factor)
                     # else:
                     raise ValueError(f"Latent shape {intermediate_latents.shape} cannot be reshaped to {expected_shape} and _unpack_latents is not available/suitable.")

                # --- Step 2: Inverse Scale & Shift ---
                scaling_factor = self.pipeline.vae.config.scaling_factor
                # Check if shift_factor exists, default to 0 if not (though Flux VAE should have it)
                shift_factor = getattr(self.pipeline.vae.config, "shift_factor", 0.0)
                print(f"Using scaling_factor: {scaling_factor}, shift_factor: {shift_factor}")
                
                # Ensure correct dtype BEFORE scaling/shifting
                model_dtype = self.pipeline.vae.dtype
                print(f"Ensuring latents are dtype: {model_dtype}")
                intermediate_latents = intermediate_latents.to(dtype=model_dtype)

                # Apply the inverse transformation: z = x / scale + shift
                latents_for_decode = intermediate_latents / scaling_factor + shift_factor
                print(f"Latents prepared for VAE shape: {latents_for_decode.shape}, dtype: {latents_for_decode.dtype}")

                # --- Step 3: VAE Decode ---
                print("Decoding latents using VAE...")
                # The VAE decoder might output a tuple or an object, grab the sample
                decoded_output = self.pipeline.vae.decode(latents_for_decode, return_dict=False)
                # Assuming the first element is the sample tensor
                decoded_image_tensor = decoded_output[0] 
                print(f"Decoded VAE sample shape: {decoded_image_tensor.shape}, dtype: {decoded_image_tensor.dtype}")

                # --- Step 4: Post-Processing ---
                print("Post-processing decoded image...")
                # Use the pipeline's image_processor, outputting tensors ('pt')
                intermediate_image = self.pipeline.image_processor.postprocess(
                    decoded_image_tensor, 
                    output_type="pt" # Keep as tensor
                ) 
                # image_processor.postprocess usually returns a list of images, take the first one
                if isinstance(intermediate_image, list):
                    intermediate_image = intermediate_image[0]
                
                # Final check for batch dimension (should be [C, H, W])
                if intermediate_image.dim() == 4 and intermediate_image.shape[0] == 1:
                     intermediate_image = intermediate_image.squeeze(0)
                
                print(f"Final intermediate image shape: {intermediate_image.shape}, dtype: {intermediate_image.dtype}")
                
            except Exception as e:
                import traceback
                print(f"Error decoding intermediate latents with Flux steps: {e}")
                print(traceback.format_exc())
                print("Using final image as fallback")
                intermediate_image = final_image.clone()
        else:
            # Fallback if we couldn't capture intermediate state
            print("Warning: Could not capture intermediate state, using final image instead")
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