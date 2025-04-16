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
                # Get raw latents
                latents = callback_kwargs["latents"].detach().clone()
                print(f"Captured latents at step {step}, shape: {latents.shape}")
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
                output_type="pt",  # Get tensor output directly
                callback_on_step_end=capture_callback,
                callback_on_step_end_tensor_inputs=["latents"]
            )
        
        # Get the final image from the output
        final_image = output.images[0]
        
        # Convert intermediate latents to image
        if intermediate_latents is not None:
            try:
                print(f"Processing intermediate latents with shape: {intermediate_latents.shape}")
                
                # Calculate expected VAE shape
                latent_channels = self.pipeline.vae.config.latent_channels # Usually 16 for Flux? Check config.
                latent_height = self.image_size // 8
                latent_width = self.image_size // 8
                expected_shape = (1, latent_channels, latent_height, latent_width)
                print(f"Expected VAE input shape: {expected_shape}")
                
                # Calculate total elements to confirm match
                numel_captured = intermediate_latents.numel()
                numel_expected = 1 * latent_channels * latent_height * latent_width
                
                if numel_captured != numel_expected:
                    raise ValueError(f"Element count mismatch: captured {numel_captured}, expected {numel_expected}")
                    
                # Reshape directly
                print(f"Attempting reshape from {intermediate_latents.shape} to {expected_shape}")
                reshaped_latents = intermediate_latents.reshape(expected_shape)
                print(f"Reshaped successfully to: {reshaped_latents.shape}")
                
                # Match the dtype to the VAE's parameters
                model_dtype = self.pipeline.vae.dtype
                print(f"Ensuring latents are dtype: {model_dtype}")
                reshaped_latents = reshaped_latents.to(dtype=model_dtype)
                
                # Apply the VAE's scaling factor
                scaling_factor = getattr(self.pipeline.vae.config, "scaling_factor", 0.18215) # Provide a default just in case
                print(f"Using scaling factor: {scaling_factor}")
                latents_for_decode = reshaped_latents / scaling_factor
                
                # Decode the latents directly using the VAE
                print("Decoding latents using VAE...")
                decoded = self.pipeline.vae.decode(latents_for_decode).sample
                print(f"Decoded sample shape: {decoded.shape}")
                
                # Post-process to image format
                intermediate_image = (decoded / 2 + 0.5).clamp(0, 1)
                
                # Adjust dimensions if needed (remove batch dim)
                if intermediate_image.dim() == 4:
                    intermediate_image = intermediate_image[0]
                    
                print(f"Final intermediate image shape: {intermediate_image.shape}")
                
            except Exception as e:
                import traceback
                print(f"Error decoding intermediate latents: {e}")
                print(traceback.format_exc())  # Print the full stack trace
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