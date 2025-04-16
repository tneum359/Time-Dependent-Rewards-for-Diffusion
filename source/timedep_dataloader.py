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
                # Instead of saving the latents, let's directly render the image 
                # by calling the pipeline's decode_latents if available
                if hasattr(pipe, "decode_latents"):
                    try:
                        # Note: we're saving the current latents to decode rather than the actual decoded image
                        # This preserves the original latent data for potential use
                        intermediate_latents = callback_kwargs["latents"].detach().clone()
                        print(f"Successfully captured intermediate state at step {step}")
                    except Exception as e:
                        print(f"Error during capture: {e}")
                else:
                    # If no direct decode method is available, we'll still try to save latents for later
                    intermediate_latents = callback_kwargs["latents"].detach().clone()
                    print(f"Captured latents at step {step}, shape: {intermediate_latents.shape}")
                
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
        
        # Try to get intermediate image
        if intermediate_latents is not None:
            try:
                # Try some direct access methods first
                if hasattr(self.pipeline, "decode_latents"):
                    print("Using pipeline's decode_latents method")
                    intermediate_image = self.pipeline.decode_latents(intermediate_latents)
                elif hasattr(self.pipeline, "_decode_latents"):
                    print("Using pipeline's _decode_latents method")
                    intermediate_image = self.pipeline._decode_latents(intermediate_latents)
                # If we still can't decode, we'll try a different approach
                else:
                    print("No direct decode method found. Attempting to use internal Flux rendering...")
                    
                    # Try to peek at what object processes or stores the final latents
                    # Look for a "renderer" or similar component
                    if hasattr(self.pipeline, "renderer") and hasattr(self.pipeline.renderer, "render"):
                        print("Using pipeline's renderer")
                        intermediate_image = self.pipeline.renderer.render(intermediate_latents)
                    else:
                        # Last resort - use the same callback mechanism Flux uses to convert latents to images
                        # This is risky but might work
                        print("Using advanced technique to decode latents...")
                        
                        # Store the original post-processing function
                        original_post_process = None
                        if hasattr(self.pipeline, "postprocess"):
                            original_post_process = self.pipeline.postprocess
                        
                        # Create a simple workaround function
                        def capture_decode(images):
                            nonlocal intermediate_image
                            intermediate_image = images[0] if isinstance(images, list) else images
                            return images
                        
                        try:
                            # Temporarily override any post-processing
                            if hasattr(self.pipeline, "postprocess"):
                                self.pipeline.postprocess = capture_decode
                            
                            # Try to decode using any available internal method
                            self.pipeline._decode_latents(intermediate_latents)
                        finally:
                            # Restore original post-processing
                            if original_post_process:
                                self.pipeline.postprocess = original_post_process
            
            except Exception as e:
                import traceback
                print(f"All approaches to decode intermediate latents failed: {e}")
                print(traceback.format_exc())
                print("Using final image as fallback")
                intermediate_image = final_image.clone()
        else:
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