"""Blend multiple images using latent averaging with Stable Diffusion VAE.

Run as a CLI:
    python -m blend_images.blend image1.jpg image2.png -o output.png

or import the `blend_images` function programmatically.
"""
from __future__ import annotations

import argparse
import pathlib
from typing import List

import torch
from diffusers import AutoencoderKL
from PIL import Image
from torchvision import transforms
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


def _square_center_crop(img: Image.Image) -> Image.Image:
    """Center-crop the longest edge to produce a square image."""
    w, h = img.size
    if w == h:
        return img
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side))


# -----------------------------------------------------------------------------
# Core functionality
# -----------------------------------------------------------------------------

def blend_images(
    image_paths: List[str | pathlib.Path],
    output_path: str | pathlib.Path = "blended.png",
    size: int = 512,
    device: str | torch.device | None = None,
) -> pathlib.Path:
    """Blend multiple images via latent-space averaging.

    Parameters
    ----------
    image_paths : list[str | pathlib.Path]
        Paths to images (≥2).
    output_path : str | pathlib.Path, default ``"blended.png"``
        Where to save the blended output.
    size : int, default ``512``
        Images are resized to ``size × size`` before encoding.
    device : str | torch.device | None, default ``None``
        Torch device to run on. If *None*, auto-detect.

    Returns
    -------
    pathlib.Path
        Path to the saved result.
    """
    if len(image_paths) < 2:
        raise ValueError("Provide at least two images to blend.")

    device = _get_default_device(str(device) if device is not None else None)
    output_path = pathlib.Path(output_path)

    # 1. Load VAE (weights are automatically downloaded if not in cache)
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
    vae = vae.to(device)
    vae.eval()

    # 2. Pre-processing pipeline
    preprocess = transforms.Compose(
        [
            _square_center_crop,
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2.0) - 1.0),  # map [0,1] -> [-1,1]
        ]
    )

    latents_list: list[torch.Tensor] = []

    for path in tqdm(image_paths, desc="Encoding images"):
        img = Image.open(path).convert("RGB")
        img_t = preprocess(img).unsqueeze(0).to(device)

        with torch.no_grad():
            latent_dist = vae.encode(img_t).latent_dist
            latent = latent_dist.sample() * 0.18215  # scale like Stable Diffusion
        latents_list.append(latent)

    # 3. Average latents
    latents = torch.cat(latents_list, dim=0)
    mixed_latent = latents.mean(dim=0, keepdim=True)

    # 4. Decode
    with torch.no_grad():
        decoded = vae.decode(mixed_latent / 0.18215).sample  # undo scaling

    # 5. Post-process
    img = (decoded.clamp(-1, 1) + 1) / 2  # [-1,1] -> [0,1]
    img = img.mul(255).byte().cpu().permute(0, 2, 3, 1).numpy()[0]
    blended = Image.fromarray(img)

    blended.save(output_path)
    return output_path


# -----------------------------------------------------------------------------
# CLI wrapper
# -----------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:  # noqa: D401
    """Define and parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="blend_images",
        description="Blend multiple images using the open-weights Stable Diffusion VAE.",
    )
    parser.add_argument("images", nargs="+", help="Paths to images to blend (≥2).")
    parser.add_argument("--output", "-o", default="blended.png", help="Output filename.")
    parser.add_argument("--size", type=int, default=512, help="Resize resolution (square).")
    parser.add_argument("--device", default=None, help="Torch device to run on (optional).")
    return parser.parse_args()


def main() -> None:  # noqa: D401
    """Entry-point used by `python -m blend_images.blend`."""
    args = _parse_args()
    blend_images(
        image_paths=args.images,
        output_path=args.output,
        size=args.size,
        device=args.device,
    )
    print(f"Saved blended image to {args.output}")


if __name__ == "__main__":
    main() 