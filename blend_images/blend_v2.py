"""Blend multiple images using vision-language model → LLM → text-to-image pipeline.

This implementation uses open-weights models for each step:
1. LLaVA for image captioning (vision-language model)
2. GPT-2 for intelligent text generation to create artistic blend descriptions
3. Stable Diffusion for text-to-image generation (optimized for MacBook Air)

Run as a CLI:
    python -m blend_images.blend_v2 image1.jpg image2.png -o output.png

or import the `blend_images_v2` function programmatically.
"""
from __future__ import annotations

import argparse
import pathlib
from typing import List
import re
import gc

import torch
from transformers import LlavaProcessor, LlavaForConditionalGeneration, GPT2LMHeadModel, GPT2Tokenizer
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
from tqdm import tqdm


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def _get_default_device(user_choice: str | None = None) -> torch.device:
    """Select an appropriate torch.device based on availability and user input."""
    if user_choice:
        return torch.device(user_choice)

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


def _optimize_memory_for_device(device: torch.device):
    """Apply memory optimizations based on device type."""
    # Always run garbage collection first
    gc.collect()

    if device.type == "mps":
        # MPS-specific optimizations
        torch.mps.empty_cache()
        # Force additional garbage collection for MPS
        gc.collect()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
        gc.collect()
    else:
        # CPU optimizations
        gc.collect()


def _create_blend_prompt(captions: List[str], device: torch.device) -> str:
    """Create a text prompt that describes the blend of multiple images using an LLM."""
    print("  Loading text generation model (GPT-2)...")

    # Load GPT-2 model and tokenizer with memory optimizations
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained(
        "gpt2",
        torch_dtype=torch.float16 if device.type in ["cuda", "mps"] else torch.float32,
        low_cpu_mem_usage=True,
    )

    # For MPS, keep GPT-2 on CPU to save memory
    if device.type == "mps":
        model = model.to("cpu")
        model_device = "cpu"
    else:
        model = model.to(device)
        model_device = device
    model.eval()

    # Add pad token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create a prompt for the LLM to generate a blend description
    input_text = f"Given these image descriptions, create a creative artistic description of what a blended image would look like:\n\n"
    for i, caption in enumerate(captions, 1):
        input_text += f"Image {i}: {caption}\n"
    input_text += "\nArtistic blend description:"

    # Tokenize and generate
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    inputs = inputs.to(model_device)

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=inputs.shape[1] + 50,  # Generate 50 more tokens
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            attention_mask=torch.ones_like(inputs)
        )

    # Decode the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract just the blend description part
    blend_description = generated_text[len(input_text):].strip()

    # Clean up the generated text (remove any incomplete sentences)
    sentences = blend_description.split('.')
    if len(sentences) > 1:
        blend_description = '.'.join(sentences[:-1]) + '.'

    # Clean up model from memory
    del model, tokenizer, inputs, outputs
    _optimize_memory_for_device(device)

    return blend_description if blend_description else f"artistic blend of {len(captions)} images: " + ", ".join(captions)


# -----------------------------------------------------------------------------
# Core functionality
# -----------------------------------------------------------------------------

def blend_images_v2(
    image_paths: List[str | pathlib.Path],
    output_path: str | pathlib.Path = "blended_v2.png",
    size: int = 384,  # Reduced default size for MacBook Air
    device: str | torch.device | None = None,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 25,  # Reduced default steps for MacBook Air
) -> pathlib.Path:
    """Blend multiple images using vision-language model → LLM → text-to-image pipeline.

    Parameters
    ----------
    image_paths : list[str | pathlib.Path]
        Paths to images (≥2).
    output_path : str | pathlib.Path, default ``"blended_v2.png"``
        Where to save the blended output.
    size : int, default ``384``
        Output image size (square). Reduced default for MacBook Air.
    device : str | torch.device | None, default ``None``
        Torch device to run on. If *None*, auto-detect.
    guidance_scale : float, default ``7.5``
        Guidance scale for text-to-image generation.
    num_inference_steps : int, default ``25``
        Number of inference steps for text-to-image generation. Reduced default for MacBook Air.

    Returns
    -------
    pathlib.Path
        Path to the saved result.
    """
    if len(image_paths) < 2:
        raise ValueError("Provide at least two images to blend.")

    device = _get_default_device(str(device) if device is not None else None)
    output_path = pathlib.Path(output_path)

    print(f"Step 1: Loading vision-language model (LLaVA) on {device}...")
    # 1. Load LLaVA for image captioning - using smaller variant for MacBook Air
    processor = LlavaProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
    model = LlavaForConditionalGeneration.from_pretrained(
        "llava-hf/llava-1.5-7b-hf",
        torch_dtype=torch.float16 if device.type in ["cuda", "mps"] else torch.float32,
        device_map="auto" if device.type != "mps" else None,  # Auto device mapping for CUDA, manual for MPS
        low_cpu_mem_usage=True,
    )

    # For MPS, use CPU offloading to manage memory
    if device.type == "mps":
        # Don't move entire model to MPS, keep some on CPU
        model = model.to("cpu")  # Keep model on CPU initially
    else:
        model = model.to(device)
    model.eval()

    print("Step 2: Generating image descriptions...")
    # 2. Generate captions for each image
    captions = []
    for path in tqdm(image_paths, desc="Captioning images"):
        img = Image.open(path).convert("RGB")

        # Generate caption using LLaVA format
        prompt = "USER: <image>\nWhat's in this image?\nASSISTANT:"

        inputs = processor(text=prompt, images=img, return_tensors="pt")

        # Handle device placement for MPS memory optimization
        if device.type == "mps":
            # Keep inputs on CPU for MPS to avoid memory issues
            inputs = {k: v.to("cpu") if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            model_device = "cpu"
        else:
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            model_device = device

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=100, num_beams=5)
        caption = processor.decode(generated_ids[0], skip_special_tokens=True)
        captions.append(caption)
        print(f"  {pathlib.Path(path).name}: {caption}")

    # Clear LLaVA model from memory
    del model, processor
    # Force aggressive cleanup for MPS
    if device.type == "mps":
        torch.mps.empty_cache()
        gc.collect()
        torch.mps.empty_cache()
    _optimize_memory_for_device(device)

    print("Step 3: Creating blend description...")
    # 3. Create blend prompt using actual LLM text generation
    blend_prompt = _create_blend_prompt(captions, device)
    print(f"  Blend prompt: {blend_prompt}")

    print("Step 4: Loading Stable Diffusion (optimized for MacBook Air)...")
    # 4. Use a lighter, more efficient model
    model_id = "stabilityai/stable-diffusion-2-1-base"  # Lighter than SD 1.5
    
    # Load with optimizations
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device.type in ["cuda", "mps"] else torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
        low_cpu_mem_usage=True,
    )

    # Use faster scheduler for fewer steps
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    # MacBook Air specific optimizations
    if device.type == "mps":
        # Enable attention slicing for MPS
        pipe.enable_attention_slicing()
        # Enable model CPU offloading for very limited memory
        pipe.enable_model_cpu_offload()  # Enable CPU offloading for MPS
        # Enable VAE slicing to reduce memory usage
        pipe.enable_vae_slicing()
        print("  Enabled MPS optimizations: attention slicing, model CPU offload, VAE slicing")
    else:
        pipe = pipe.to(device)

        # Enable memory efficient attention if available
        if hasattr(pipe.unet, 'set_use_memory_efficient_attention_xformers'):
            try:
                pipe.enable_xformers_memory_efficient_attention()
            except:
                pass
    
    print(f"Generating blended image ({size}x{size}, {num_inference_steps} steps)...")
    # 5. Generate the blended image
    with torch.no_grad():
        result = pipe(
            prompt=blend_prompt,
            height=size,
            width=size,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=torch.Generator(device=device).manual_seed(42),  # For reproducibility
        )
    
    # Save the result
    blended_image = result.images[0]
    blended_image.save(output_path)
    
    # Clean up
    del pipe
    _optimize_memory_for_device(device)
    
    print(f"Saved blended image to {output_path}")
    return output_path


# -----------------------------------------------------------------------------
# CLI wrapper
# -----------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    """Define and parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="blend_images_v2",
        description="Blend multiple images using vision-language model → LLM → text-to-image pipeline.",
    )
    parser.add_argument("images", nargs="+", help="Paths to images to blend (≥2).")
    parser.add_argument("--output", "-o", default="blended_v2.png", help="Output filename.")
    parser.add_argument("--size", type=int, default=384, help="Output image size (square). Default optimized for MacBook Air.")
    parser.add_argument("--device", default=None, help="Torch device to run on (optional).")
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=7.5,
        help="Guidance scale for text-to-image generation.",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=25,
        help="Number of inference steps for text-to-image generation. Default optimized for MacBook Air.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry-point used by `python -m blend_images.blend_v2`."""
    args = _parse_args()
    blend_images_v2(
        image_paths=args.images,
        output_path=args.output,
        size=args.size,
        device=args.device,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
    )


if __name__ == "__main__":
    main() 