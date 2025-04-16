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
        # Store intermediate results
        intermediate_image = None
        captured_timestep = None
        
        # Set up callback to capture intermediate image at random timestep
        random_step = random.randint(1, 49)  # Random step between 1 and 49 (for 50 steps)
        
        def callback_fn(step, timestep, latents):
            nonlocal intermediate_image, captured_timestep
            if step == random_step:
                # Store timestep and latents for intermediate image
                captured_timestep = timestep
                # Convert latents to image using the same process as the pipeline
                with torch.no_grad():
                    intermediate_image = self.pipeline.decode_latents(latents)
        
        # Generate image with empty prompt using the callback
        with torch.no_grad():
            output = self.pipeline(
                prompt="",  # Empty prompt for random generation
                height=self.image_size,
                width=self.image_size,
                guidance_scale=3.5,
                num_inference_steps=50,
                max_sequence_length=512,
                callback=callback_fn,
                callback_steps=1
            )
        
        # Get the final image from the output
        final_image = output.images[0]
        
        # Convert PIL to tensor
        if not isinstance(final_image, torch.Tensor):
            final_image = torch.from_numpy(final_image).permute(2, 0, 1) / 255.0
            if intermediate_image is not None and not isinstance(intermediate_image, torch.Tensor):
                intermediate_image = torch.from_numpy(intermediate_image).permute(2, 0, 1) / 255.0
        
        return final_image, intermediate_image, captured_timestep
    
    def get_dataloader(self, shuffle=True, num_workers=2):
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