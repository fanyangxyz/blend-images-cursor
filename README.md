# Blend Images with Open Weights

This project demonstrates how to blend (merge) multiple images into a single output image using **open-source neural-network weights**.  
We leverage the *Stable Diffusion* VAE (Variational Auto-Encoder) from Hugging Face (`stabilityai/sd-vae-ft-mse`) to encode each input image into a latent space, average the latent representations, and decode the averaged latent back to pixel space.  The result is a visually plausible blend of the inputs.

> **Why the VAE?**  The VAE component of Stable Diffusion is small (~335 MB), publicly licensed, and can reconstruct high-quality 512×512 RGB images.  Averaging in latent space often yields smoother, more natural blends than simple pixel-space alpha blending.

---

## Quick start

```bash
# 1. Clone this repo (or copy the code)
# 2. Create a virtualenv (recommended)
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the blender
python -m blend_images.blend \
    path/to/photo1.jpg path/to/photo2.png path/to/photo3.jpeg \
    --output blended.png
```

The first run will automatically **download** the open weights (the VAE) from Hugging Face.

### CLI options

```
usage: blend.py [-h] [--output OUTPUT] [--size SIZE] [--device DEVICE] images [images ...]

Positional arguments:
  images                Paths of the images you want to blend (≥2).

Optional arguments:
  -h, --help            show this help message and exit
  --output OUTPUT, -o OUTPUT
                        Output filename (default: blended.png)
  --size SIZE           Square resolution the images will be resized to (default: 512)
  --device DEVICE       Torch device to run on (e.g. cuda, mps, cpu). Auto-detected by default.
```

---

## How it works

1. **Load** the open-weights VAE (`AutoencoderKL`) via the [🤗 Diffusers](https://github.com/huggingface/diffusers) library.
2. **Pre-process** each input image:
   * Convert to RGB, square-pad & resize.
   * Map pixel range from `[0, 255]` → `[-1, 1]` to match the VAE's expected scale.
3. **Encode** each image into latent space: `latent = encoder(image).latent_dist.sample()`
4. **Average** all latents (simple mean).
5. **Decode** the averaged latent back to pixel space.
6. **Post-process** & save the resulting image.

The only learnable part is the VAE, whose weights are freely available under a permissive license.

---

## Notes

* If you have an Apple Silicon Mac, the script will automatically use the *MPS* backend (macOS GPU) when available.
* For best quality, supply images with visually related content and similar aspect ratios. All images are center-cropped to a square before blending.
* You can experiment with different latent-space operations (e.g., weighted averages) to influence the blend.

---

## License

The code in this repository is licensed under the MIT License.  The VAE weights are licensed under the [CreativeML Open RAIL-M](https://raw.githubusercontent.com/Stability-AI/stablediffusion/main/LICENSE).  By downloading and using the weights, you agree to their license terms. 