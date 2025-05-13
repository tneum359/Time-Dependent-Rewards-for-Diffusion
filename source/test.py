import torch
import matplotlib.pyplot as plt
from diffusers import FluxPipeline
import random
import os
import argparse
from huggingface_hub import login
from PIL import Image
import numpy as np
import traceback

def plot_images_side_by_side(image1, title1, image2, title2):
    """Plots two images side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Convert tensors to PIL Images if necessary
    if isinstance(image1, torch.Tensor):
        image1 = image1.cpu().permute(1, 2, 0).numpy()
        image1 = (image1 * 255).astype(np.uint8)
        image1 = Image.fromarray(image1)
        
    if isinstance(image2, torch.Tensor):
        image2 = image2.cpu().permute(1, 2, 0).numpy()
        image2 = (image2 * 255).astype(np.uint8)
        image2 = Image.fromarray(image2)

    axes[0].imshow(image1)
    axes[0].set_title(title1)
    axes[0].axis('off')
    
    axes[1].imshow(image2)
    axes[1].set_title(title2)
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()

def generate_and_decode_latent(hf_token=None, image_size=512, num_inference_steps=50, prompt=""):
    """
    Generates an image with FLUX.1-dev, captures an intermediate latent, 
    decodes it, and plots both the final image and the decoded latent.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() and device == "cuda" else torch.float32

    # --- Authentication ---
    if not hf_token:
        hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
    else:
        print("Warning: No Hugging Face token provided. You might encounter authentication errors.")
        print("Please set the HF_TOKEN environment variable or pass --token.")
        print("Get your token from: https://huggingface.co/settings/tokens")
        # Optionally, you could raise an error here or try to proceed without a token
        # For FLUX.1-dev, a token is usually required.

    # --- Load Pipeline ---
    print(f"Loading FLUX.1-dev pipeline (this might take a while)...")
    try:
        pipeline = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=torch_dtype,
            # token=hf_token # from_pretrained uses the global login
        )
        if device == "cuda":
            print("CUDA device detected. Enabling model CPU offload to potentially save memory.")
            pipeline.enable_model_cpu_offload() # Enable offload first
            pipeline.to(device) # Then move to device
        else: # If on CPU, or if CUDA is available but model is too large for VRAM
            print("CUDA not available or not primary. Enabling model CPU offload for the pipeline.")
            pipeline.enable_model_cpu_offload()

    except Exception as e:
        print(f"Error loading FLUX.1-dev pipeline: {e}")
        print("Make sure you have a valid Hugging Face token with access to this model.")
        print("And that 'diffusers', 'transformers', 'accelerate' are installed.")
        return

    # --- Prepare to capture intermediate latent ---
    intermediate_latent_image = None
    captured_timestep_value = None
    
    # Choose a random step to capture the latent (e.g., halfway through)
    # Ensure it's less than num_inference_steps
    capture_step = random.randint(1, num_inference_steps - 1) 
    print(f"Will attempt to capture latent at inference step: {capture_step}")

    def callback_fn(pipe, step, timestep, latents):
        nonlocal intermediate_latent_image, captured_timestep_value
        if step == capture_step:
            print(f"Callback triggered at step {step}, timestep {timestep}")
            # Decode the latents from this step
            # The latents need to be scaled before decoding if using VAE from pipeline
            # For Flux, pipeline.decode_latents should handle this.
            with torch.no_grad():
                # Ensure latents are on the correct device for decoding
                decoded_latents = pipeline.decode_latents(latents.to(pipeline.device, dtype=pipeline.text_encoder.dtype if hasattr(pipeline, 'text_encoder') else torch.float32)) # Use appropriate dtype
                # pipeline.decode_latents returns a list of PIL images
                if decoded_latents and len(decoded_latents) > 0:
                    intermediate_latent_image = decoded_latents[0] # Take the first image if batch > 1
                    captured_timestep_value = timestep.item() if torch.is_tensor(timestep) else timestep
                    print(f"Captured and decoded intermediate latent image at step {step}.")
                else:
                    print(f"Warning: pipeline.decode_latents did not return an image at step {step}.")


    # --- Generate Image ---
    print(f"Generating image with prompt: '{prompt}' (Size: {image_size}x{image_size}, Steps: {num_inference_steps})")
    try:
        with torch.no_grad():
            output = pipeline(
                prompt=prompt,
                height=image_size,
                width=image_size,
                num_inference_steps=num_inference_steps,
                guidance_scale=0.0, # For FLUX, guidance_scale 0 is common for unconditioned generation
                callback_on_step_end=callback_fn,
                callback_on_step_end_tensor_inputs=["latents"], # Pass latents to callback
            )
        final_image = output.images[0]
        print("Image generation complete.")
    except Exception as e:
        print(f"Error during image generation: {e}")
        traceback.print_exc()
        return

    # --- Plot Results ---
    if final_image and intermediate_latent_image:
        plot_images_side_by_side(
            final_image, 
            f"Final Image (Prompt: '{prompt}')",
            intermediate_latent_image,
            f"Decoded Latent (Step: {capture_step}, Timestep: {captured_timestep_value:.2f} approx.)"
        )
    elif final_image:
        print("Final image generated, but intermediate latent was not captured.")
        final_image.show(title="Final Image")
    else:
        print("Image generation failed or no images were produced.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test FLUX.1-dev latent decoding.")
    parser.add_argument("--token", type=str, default=os.environ.get("HF_TOKEN"), help="Hugging Face token for authentication.")
    parser.add_argument("--prompt", type=str, default="A photo of a cat wearing a small hat", help="Prompt for image generation.")
    parser.add_argument("--size", type=int, default=512, help="Image size (square).")
    parser.add_argument("--steps", type=int, default=20, help="Number of inference steps (e.g., 20-50).") # Reduced for quicker testing
    
    args = parser.parse_args()

    if not args.token:
        print("Error: Hugging Face token is required. Set HF_TOKEN or use --token argument.")
        parser.print_help()
    else:
        generate_and_decode_latent(
            hf_token=args.token,
            image_size=args.size,
            num_inference_steps=args.steps,
            prompt=args.prompt
        ) 