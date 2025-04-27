import torch
from torch.utils.data import Dataset, DataLoader
# from diffusers import FluxPipeline # Removed
import random
import os
# from huggingface_hub import login # Removed
import traceback # For detailed error printing

class TimeDependentDataset(Dataset):
    def __init__(self,
                 prompts_file="prompts.txt",
                 # model_name="flux1.dev", # Removed
                 image_size=1024, # Keep image_size if needed elsewhere, maybe for validation? Or remove? Let's keep for now.
                 # Default to GPU if available, otherwise CPU
                 # device="cuda" if torch.cuda.is_available() else "cpu", # Removed
                 # hf_token=None): # Removed
                ): # Simplified constructor
        """
        Loads prompts from a file. Does NOT generate images.
        """
        super().__init__()
        self.image_size = image_size # Still storing this, might be useful later
        # # Store the target device # Removed
        # self.device = torch.device(device)
        # print(f"Dataset configured to use device: {self.device}")

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

        # --- HF Login --- Removed ---
        # if hf_token:
        #     login(token=hf_token)
        # elif os.environ.get("HF_TOKEN"):
        # --- End HF Login ---
        
        # --- Load Flux Pipeline --- Removed ---
        # try:
        #     print(f"Loading Flux pipeline ({model_name}) to device: {self.device}...")
        #     # Determine dtype based on device
        #     self.pipeline_dtype = torch.bfloat16 if self.device.type == 'cuda' else torch.float32
        #     print(f"Using dtype: {self.pipeline_dtype}")
        #     
        #     # Load pipeline explicitly to the target device
        #     self.pipeline = FluxPipeline.from_pretrained(
        # --- End Load Flux Pipeline ---
        
    def __len__(self):
        return self.total_samples 
    
    def __getitem__(self, idx):
        """ Returns only the prompt string for the given index. """
        prompt = self.prompts[idx % len(self.prompts)]
        # --- All generation logic removed ---
        # num_inference_steps = 30
        # random_step = random.randint(1, num_inference_steps - 1)
        # # Create timestep tensor directly on the target device (minor optimization)
        # captured_timestep = torch.tensor(random_step, device=self.device)
        #
        # intermediate_latents = None
        #
        # # Callback remains the same (captures latents which will be on the pipeline's compute device)
        # def capture_callback(pipe, step, timestep, callback_kwargs):
        #             if intermediate_image.dim() == 4 and intermediate_image.shape[0] == 1: intermediate_image = intermediate_image.squeeze(0)
        #             # print(f"Decoded intermediate image (device: {intermediate_image.device})") # Debug
        #
        #         else:
        #              print(f"Warning: Intermediate latents not captured for idx {idx}. Using final image.")
        #              intermediate_image = final_image.clone() # Use final image tensor
        #
        #     except Exception as e:
        #          print(f"ERROR during generation/decoding for prompt idx {idx} ('{prompt[:50]}...'): {e}")
        # Return only the prompt string
        # return final_image.cpu(), intermediate_image.cpu(), captured_timestep.cpu(), prompt # Original return
        return prompt # Return only the prompt

def load_diffusion_dataloader(
    prompts_file="prompts.txt", batch_size=4, image_size=1024,
    shuffle=True, num_workers=0,
    # hf_token=None, # Removed hf_token
    # device="cuda" if torch.cuda.is_available() else "cpu"): # Removed device
    ): # Simplified signature
    """Helper function to create dataset and dataloader (loads only prompts)."""
    print(f"Creating dataset with prompts from: {prompts_file}") # Updated print
    dataset = TimeDependentDataset(
        prompts_file=prompts_file, image_size=image_size,
        # hf_token=hf_token, device=device # Removed args
    )
    
    print(f"Creating DataLoader with batch_size={batch_size}, shuffle={shuffle}, num_workers={num_workers}")
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, # Can potentially be > 0 now, but keep 0 for simplicity
        pin_memory=False 
    )