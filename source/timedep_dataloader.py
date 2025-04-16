import torch
from torch.utils.data import Dataset, DataLoader
from diffusers import FluxPipeline
import random
import os
from huggingface_hub import login
import traceback # For detailed error printing

class TimeDependentDataset(Dataset):
    def __init__(self, 
                 prompts_file="prompts.txt", # Added prompts file argument
                 model_name="flux1.dev", 
                 batch_size=4, # Batch size is handled by DataLoader, not dataset itself usually
                 image_size=1024, 
                 device="cuda" if torch.cuda.is_available() else "cpu", 
                 hf_token=None):
        """
        Generates triplets of (final_image, intermediate_image, timestep) based on prompts.
        
        Args:
            prompts_file (str): Path to a text file containing prompts (one per line).
            model_name (str): Model identifier.
            image_size (int): Size of generated images.
            device (str): Device for computation.
            hf_token (str): Hugging Face token.
        """
        super().__init__()
        # self.batch_size = batch_size # Store batch size for DataLoader, not used here
        self.image_size = image_size
        self.device = device # Store device if needed internally, though pipeline handles it

        # --- Read Prompts ---
        self.prompts = []
        try:
            with open(prompts_file, 'r') as f:
                self.prompts = [line.strip() for line in f if line.strip()]
            if not self.prompts:
                 raise ValueError(f"No valid prompts found in {prompts_file}")
            print(f"Loaded {len(self.prompts)} prompts from {prompts_file}")
        except FileNotFoundError:
            print(f"Error: Prompts file not found at {prompts_file}")
            raise
        except Exception as e:
            print(f"Error reading prompts file {prompts_file}: {e}")
            raise
        # --- End Read Prompts ---

        # --- Set Total Samples ---
        self.total_samples = len(self.prompts) # Dataset size is number of prompts
        # --- End Set Total Samples ---

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
            self.pipeline = FluxPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-dev", 
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                # token=hf_token # Use environment or login
            )
            self.pipeline.enable_model_cpu_offload() # Still enable offload
            print("Flux pipeline loaded successfully.")
        except Exception as e:
            print(f"Error loading Flux model: {e}")
            print(traceback.format_exc())
            raise
        # --- End Load Flux Pipeline ---
        
    def __len__(self):
        # The length of the dataset is the number of prompts
        return self.total_samples 
    
    def __getitem__(self, idx):
        """
        Generate a triplet for a specific prompt index.
        Uses the two-pass approach for reliable intermediate images.
        """
        # Get the prompt for this index
        prompt = self.prompts[idx % len(self.prompts)] # Use modulo for safety if idx goes out of bounds
        
        # Choose a random timestep between 1 and 29
        num_inference_steps = 30 # Example value, adjust if needed
        random_step = random.randint(1, num_inference_steps - 1)
        captured_timestep = torch.tensor(random_step)
        
        # Set a fixed seed based on index for reproducibility of this specific sample
        # Using idx ensures each prompt gets a consistent generation pair across epochs
        generator = torch.Generator(device="cpu").manual_seed(idx) 
        
        final_image = None
        intermediate_image = None

        try:
            # First, generate the final image with full steps
            print(f"Generating final image for prompt idx {idx}...")
            with torch.no_grad():
                final_output = self.pipeline(
                    prompt=prompt, # Use the specific prompt
                    height=self.image_size,
                    width=self.image_size,
                    guidance_scale=3.5,
                    num_inference_steps=num_inference_steps,
                    max_sequence_length=512,
                    output_type="pt",  
                    generator=generator.manual_seed(idx) # Re-seed generator
                )
            final_image = final_output.images[0]
            print(f"  Final image generated.")

            # Then generate the intermediate image by running fewer steps
            print(f"Generating intermediate image (step {random_step}) for prompt idx {idx}...")
            with torch.no_grad():
                intermediate_output = self.pipeline(
                    prompt=prompt, # Same prompt
                    height=self.image_size,
                    width=self.image_size,
                    guidance_scale=3.5,
                    num_inference_steps=random_step, # Stop at the target step
                    max_sequence_length=512,
                    output_type="pt",
                    generator=generator.manual_seed(idx) # Re-seed generator again for consistency
                )
            intermediate_image = intermediate_output.images[0]
            print(f"  Intermediate image generated.")

        except Exception as e:
             print(f"Error during generation for prompt idx {idx} ('{prompt[:50]}...'): {e}")
             print(traceback.format_exc())
             # Return dummy data or re-raise? Returning dummy might hide issues. Re-raising is better.
             raise RuntimeError(f"Failed generation for idx {idx}") from e


        # Ensure tensors are returned
        if final_image is None or intermediate_image is None:
             # This should ideally not happen if exceptions are raised properly
             raise RuntimeError(f"Generation failed to produce images for idx {idx}")

        return final_image, intermediate_image, captured_timestep
    
    # get_dataloader method is removed from Dataset, it belongs outside

def load_diffusion_dataloader(
    prompts_file="prompts.txt", # Pass prompts file path
    batch_size=4, 
    image_size=1024, 
    shuffle=True, # Usually True for training
    num_workers=0, # Keep 0 for Flux pipeline compatibility
    hf_token=None):
    """Helper function to create dataset and dataloader."""
    print(f"Creating dataset with prompts from: {prompts_file}")
    dataset = TimeDependentDataset(
        prompts_file=prompts_file, # Pass to dataset
        image_size=image_size, 
        hf_token=hf_token
        # batch_size is not needed for Dataset init
    )
    
    print(f"Creating DataLoader with batch_size={batch_size}, shuffle={shuffle}, num_workers={num_workers}")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers # Must be 0 if pipeline is not pickleable
    )