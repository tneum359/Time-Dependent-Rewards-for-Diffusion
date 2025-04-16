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
        # Choose a random timestep between 1 and 49
        num_inference_steps = 50
        random_step = random.randint(1, num_inference_steps - 1)
        captured_timestep = torch.tensor(random_step)
        
        # Override the scheduler step method to capture intermediate latents
        original_step = self.pipeline.scheduler.step
        intermediate_latents = None
        
        def wrapped_step_function(*args, **kwargs):
            nonlocal intermediate_latents
            # Get the step count by checking model_output shape or other means
            current_step = self.pipeline.scheduler._step_count
            
            # Call the original step function
            step_output = original_step(*args, **kwargs)
            
            # If we're at our target step, save the latents
            if current_step == random_step:
                # The prev_sample contains the denoised latents at this step
                intermediate_latents = step_output.prev_sample
                
            return step_output
        
        try:
            # Temporarily override the step function
            self.pipeline.scheduler.step = wrapped_step_function
            
            # Generate image
            with torch.no_grad():
                output = self.pipeline(
                    prompt="",  # Empty prompt for random generation
                    height=self.image_size,
                    width=self.image_size,
                    guidance_scale=3.5,
                    num_inference_steps=num_inference_steps,
                    max_sequence_length=512,
                    output_type="pt"  # Get tensor output directly
                )
        finally:
            # Restore the original step function
            self.pipeline.scheduler.step = original_step
        
        # Get the final image from the output
        final_image = output.images[0]
        
        # Convert intermediate latents to image
        if intermediate_latents is not None:
            # Use the VAE to decode the latents to image
            intermediate_image = self.pipeline.vae.decode(intermediate_latents / self.pipeline.vae.config.scaling_factor).sample
            # Convert to proper format (0-1 range)
            intermediate_image = (intermediate_image / 2 + 0.5).clamp(0, 1)
            # Adjust dimensions if needed (batch, channels, height, width) -> (channels, height, width)
            if intermediate_image.dim() == 4:
                intermediate_image = intermediate_image[0]
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

def load_diffusion_dataloader(batch_size=4, image_size=1024, shuffle=True, num_workers=2, hf_token=None):
    """Helper function to create and return a dataloader for Flux model"""
    dataset = TimeDependentDataset(
        batch_size=batch_size, 
        image_size=image_size, 
        hf_token=hf_token
    )
    return dataset.get_dataloader(shuffle=shuffle, num_workers=num_workers)