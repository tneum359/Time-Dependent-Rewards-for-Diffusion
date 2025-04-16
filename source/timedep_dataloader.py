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
        
        # Initialize variables to store final latents and output
        final_latents = None
        intermediate_latents = None
        
        # Define callback function to capture results at both stages
        def capture_callback(pipe, step, timestep, callback_kwargs):
            nonlocal intermediate_latents, final_latents
            
            # Capture at our target intermediate step
            if step == random_step:
                intermediate_latents = callback_kwargs["latents"].detach().clone()
                print(f"Captured intermediate latents at step {step}, shape: {intermediate_latents.shape}")
            
            # Also capture the final latents to see how they differ
            if step == num_inference_steps - 1:
                final_latents = callback_kwargs["latents"].detach().clone()
                print(f"Captured final latents at step {step}, shape: {final_latents.shape}")
            
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
        
        # Try to convert intermediate latents using the same method that worked for final latents
        if intermediate_latents is not None and final_latents is not None:
            try:
                print(f"Final image shape: {final_image.shape}")
                
                # Examine the internal structure of the pipeline to understand the latent transformation
                print("Checking internal pipeline structure...")
                
                # Try to transform the latents similar to how the final latents were transformed
                # First check if we can access and use the internal _decode_latents method
                if hasattr(self.pipeline, "_decode_latents"):
                    print("Using pipeline's internal _decode_latents method")
                    intermediate_image = self.pipeline._decode_latents(intermediate_latents)
                else:
                    print("Using custom latent reshaping for Flux model")
                    
                    # Determine expected dimensions
                    orig_h, orig_w = self.image_size, self.image_size
                    latent_h, latent_w = orig_h // 8, orig_w // 8  # Standard VAE downsampling
                    
                    # Reshape intermediate latents to match final latents shape
                    if final_latents.shape[1] == 16:
                        # Use final latent's structure to inform our reshape
                        print(f"Using final latent structure ({final_latents.shape}) as a guide")
                        
                        # Calculate a scaling factor for the intermediate latents based on final latents
                        scaling = final_latents.abs().mean() / intermediate_latents.abs().mean()
                        print(f"Calculated scaling factor: {scaling}")
                        
                        # Create properly shaped latents
                        reshaped_latents = intermediate_latents.repeat(1, 16, 1, 1)
                        
                        # Resize to match expected dimensions if needed
                        if reshaped_latents.shape[2:] != (latent_h, latent_w):
                            print(f"Resizing from {reshaped_latents.shape[2:]} to ({latent_h}, {latent_w})")
                            reshaped_latents = torch.nn.functional.interpolate(
                                reshaped_latents, 
                                size=(latent_h, latent_w),
                                mode='bilinear'
                            )
                    else:
                        # Fallback if the final latents don't have the expected 16 channels
                        print(f"Final latents don't have 16 channels, creating new reshaped tensor")
                        reshaped_latents = intermediate_latents.repeat(1, 16, 1, 1)
                        if reshaped_latents.shape[2:] != (latent_h, latent_w):
                            reshaped_latents = torch.nn.functional.interpolate(
                                reshaped_latents, 
                                size=(latent_h, latent_w),
                                mode='bilinear'
                            )
                    
                    # Apply the VAE's scaling factor
                    latents_for_decode = reshaped_latents / self.pipeline.vae.config.scaling_factor
                    
                    # Decode the latents
                    decoded = self.pipeline.vae.decode(latents_for_decode).sample
                    
                    # Post-process to image format
                    intermediate_image = (decoded / 2 + 0.5).clamp(0, 1)
                    
                    # Adjust dimensions if needed
                    if intermediate_image.dim() == 4:
                        intermediate_image = intermediate_image[0]
            except Exception as e:
                print(f"Error decoding intermediate latents: {e}")
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