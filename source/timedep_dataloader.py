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
                 image_size=1024,
                 device="cuda" if torch.cuda.is_available() else "cpu",
                 hf_token=None):
        """
        Generates triplets of (final_image, intermediate_image, timestep) based on prompts
        using a single-pass approach with callbacks and latent decoding.
        
        Args:
            prompts_file (str): Path to a text file containing prompts (one per line).
            model_name (str): Model identifier.
            image_size (int): Size of generated images.
            device (str): Device for computation.
            hf_token (str): Hugging Face token.
        """
        super().__init__()
        self.image_size = image_size
        self.device = device

        # --- Read Prompts ---
        self.prompts = []
        try:
            # Ensure the path is correct, especially in Colab (e.g., might need '/content/prompts.txt')
            if not os.path.exists(prompts_file):
                 # Try looking in parent directory if in 'source' subdir
                 parent_dir_prompts = os.path.join(os.path.dirname(__file__), '..', prompts_file)
                 if os.path.exists(parent_dir_prompts):
                      prompts_file = parent_dir_prompts
                 else:
                      raise FileNotFoundError(f"Prompts file not found at {prompts_file} or {parent_dir_prompts}")
                      
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
            self.pipeline = FluxPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-dev",
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            )
            self.pipeline.enable_model_cpu_offload()
            print("Flux pipeline loaded successfully.")
        except Exception as e:
            print(f"Error loading Flux model: {e}")
            print(traceback.format_exc())
            raise
        # --- End Load Flux Pipeline ---
        
    def __len__(self):
        return self.total_samples 
    
    def __getitem__(self, idx):
        """
        Generate a triplet using a single pass with callback and latent decoding.
        """
        # Get the prompt for this index
        prompt = self.prompts[idx % len(self.prompts)] 

        # Choose a random timestep
        num_inference_steps = 30 # Adjust as needed
        random_step = random.randint(1, num_inference_steps - 1)
        captured_timestep = torch.tensor(random_step)
        
        # Initialize variable for captured intermediate latents
        intermediate_latents = None
        
        # Define callback function to capture raw latents
        def capture_callback(pipe, step, timestep, callback_kwargs):
            nonlocal intermediate_latents
            if step == random_step:
                # Capture the raw scheduler output latents
                latents = callback_kwargs["latents"].detach().clone()
                print(f"Captured raw intermediate latents at step {step}, shape: {latents.shape}")
                intermediate_latents = latents
            return callback_kwargs
        
        final_image = None
        intermediate_image = None

        # --- Generate Image using Single Pass with Callback ---
        try:
            print(f"Generating image for prompt idx {idx} (step {random_step} captured)...")
            with torch.no_grad():
                # Set seed for reproducibility *within* this generation if desired, 
                # but using idx ensures different prompts get different noise.
                # generator = torch.Generator(device="cpu").manual_seed(idx) # Optional per-item seed
                output = self.pipeline(
                    prompt=prompt, # Use the specific prompt
                    height=self.image_size,
                    width=self.image_size,
                    guidance_scale=3.5,
                    num_inference_steps=num_inference_steps,
                    max_sequence_length=512,
                    output_type="pt",  # Get final image as tensor
                    callback_on_step_end=capture_callback,
                    callback_on_step_end_tensor_inputs=["latents"]
                    # generator=generator # Optional per-item seed
                )
            final_image = output.images[0]
            print(f"  Final image generated.")

            # --- Decode Intermediate Latents (Single-Pass Logic) ---
            if intermediate_latents is not None:
                print(f"Processing intermediate latents (initial shape: {intermediate_latents.shape})")
                latents_to_process = intermediate_latents

                # --- Step 1: Unpack/Reshape Latents ---
                latent_channels = self.pipeline.vae.config.latent_channels
                latent_height = self.image_size // 8
                latent_width = self.image_size // 8
                expected_shape = (1, latent_channels, latent_height, latent_width)
                numel_expected = 1 * latent_channels * latent_height * latent_width

                # Prioritize unpacking if available
                if hasattr(self.pipeline, "_unpack_latents"):
                    try:
                        vae_scale_factor = 2 ** (len(self.pipeline.vae.config.block_out_channels) - 1)
                        print(f"Attempting to unpack latents using _unpack_latents (vae_scale_factor={vae_scale_factor})...")
                        latents_to_process = self.pipeline._unpack_latents(
                            latents_to_process, self.image_size, self.image_size, vae_scale_factor
                        )
                        print(f"Unpacked latents shape: {latents_to_process.shape}")
                    except Exception as unpack_e:
                        print(f"WARNING: _unpack_latents failed: {unpack_e}. Attempting direct reshape.")
                        if latents_to_process.numel() == numel_expected:
                             latents_to_process = latents_to_process.reshape(expected_shape)
                        else:
                             raise ValueError(f"Latent shape {latents_to_process.shape} cannot be unpacked or reshaped to {expected_shape}.") from unpack_e
                # Fallback to reshape if no unpack method
                elif latents_to_process.numel() == numel_expected:
                     print(f"No _unpack_latents method found. Attempting direct reshape to {expected_shape}")
                     latents_to_process = latents_to_process.reshape(expected_shape)
                else:
                     raise ValueError(f"Latent shape {latents_to_process.shape} cannot be reshaped to {expected_shape} and _unpack_latents not found.")

                # --- Step 2: Inverse Scale & Shift ---
                scaling_factor = self.pipeline.vae.config.scaling_factor
                shift_factor = getattr(self.pipeline.vae.config, "shift_factor", 0.0)
                print(f"Applying inverse scale/shift: scale={scaling_factor}, shift={shift_factor}")
                model_dtype = self.pipeline.vae.dtype
                print(f"Ensuring latents are dtype: {model_dtype}")
                latents_to_process = latents_to_process.to(dtype=model_dtype)
                latents_for_decode = latents_to_process / scaling_factor + shift_factor
                print(f"Latents prepared for VAE (shape: {latents_for_decode.shape}, dtype: {latents_for_decode.dtype})")

                # --- Step 3: VAE Decode ---
                print("Decoding latents using VAE...")
                with torch.no_grad():
                    decoded_output = self.pipeline.vae.decode(latents_for_decode, return_dict=False)
                    decoded_image_tensor = decoded_output[0] 
                print(f"Decoded VAE sample shape: {decoded_image_tensor.shape}, dtype: {decoded_image_tensor.dtype}")

                # --- Step 4: Post-Processing ---
                print("Post-processing decoded image using ImageProcessor...")
                intermediate_image = self.pipeline.image_processor.postprocess(
                    decoded_image_tensor, output_type="pt"
                ) 
                if isinstance(intermediate_image, list): intermediate_image = intermediate_image[0]
                if intermediate_image.dim() == 4 and intermediate_image.shape[0] == 1: intermediate_image = intermediate_image.squeeze(0)
                print(f"Successfully decoded intermediate image (shape: {intermediate_image.shape}, dtype: {intermediate_image.dtype})")
            
            else: # intermediate_latents is None
                 print("Warning: Intermediate latents were not captured. Using final image as fallback.")
                 intermediate_image = final_image.clone()

        except Exception as e:
             print(f"ERROR during generation/decoding for prompt idx {idx} ('{prompt[:50]}...'): {e}")
             print(traceback.format_exc())
             # If generation failed, final_image might be None. If decoding failed, use final.
             if final_image is None:
                  raise RuntimeError(f"Failed generation entirely for idx {idx}") from e
             else:
                  print("Using final image as fallback for intermediate image due to error.")
                  intermediate_image = final_image.clone() # Fallback

        # Final check
        if final_image is None or intermediate_image is None:
             raise RuntimeError(f"Generation failed to produce valid images for idx {idx}")

        return final_image, intermediate_image, captured_timestep

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
    )
    
    print(f"Creating DataLoader with batch_size={batch_size}, shuffle={shuffle}, num_workers={num_workers}")
    # Important: Ensure pin_memory is False if using CPU or if issues arise
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers, # Must be 0 if pipeline is not pickleable
        pin_memory=False # Set to False for safety with complex objects/CPU offload
    )