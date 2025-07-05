# Blend Images with Open Weights

This project demonstrates **two different approaches** to blend (merge) multiple images into a single output image using **open-source neural-network weights**. Both methods leverage different aspects of modern AI models to create visually compelling blends.

---

## Setup

```bash
# 1. Clone this repo (or copy the code)
# 2. Create a virtualenv (recommended)
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

The first run of either method will automatically **download** the required open weights from Hugging Face.

---

## Method 1: Latent Space Blending (`blend.py`)

**Approach**: Direct latent space averaging using Stable Diffusion's VAE

This method encodes images into a latent space, averages the latent representations, and decodes back to pixel space.

### Quick start

```bash
python -m blend_images.blend \
    path/to/photo1.jpg path/to/photo2.png path/to/photo3.jpeg \
    --output blended.png \
    --blend-mode mean
```

### How it works

1. **Load** the open-weights VAE (`AutoencoderKL`) from `stabilityai/sd-vae-ft-mse`
2. **Pre-process** each input image: convert to RGB, square-pad & resize
3. **Encode** each image into latent space: `latent = encoder(image).latent_dist.sample()`
4. **Average** all latents (mean or max operation)
5. **Decode** the averaged latent back to pixel space
6. **Post-process** & save the resulting image

> **Why this works**: The VAE component (~335 MB) is small, publicly licensed, and averaging in latent space often yields smoother, more natural blends than simple pixel-space alpha blending.

### CLI options

```
usage: blend.py [-h] [--output OUTPUT] [--size SIZE] [--blend-mode {mean,max}] [--device DEVICE] images [images ...]

Positional arguments:
  images                Paths of the images you want to blend (≥2).

Optional arguments:
  -h, --help            show this help message and exit
  --output OUTPUT, -o OUTPUT
                        Output filename (default: blended.png)
  --size SIZE           Square resolution the images will be resized to (default: 512)
  --blend-mode {mean,max}
                        How to combine latents: mean (average) or max (element-wise maximum). (default: mean)
  --device DEVICE       Torch device to run on (e.g. cuda, mps, cpu). Auto-detected by default.
```

---

## Method 2: Semantic Text Blending (`blend_v2.py`)

**Approach**: Vision-language model → text processing → text-to-image generation

This method understands what's in the images through text descriptions, then generates a new image based on the combined concepts.

### Quick start

```bash
python -m blend_images.blend_v2 \
    path/to/photo1.jpg path/to/photo2.png path/to/photo3.jpeg \
    --output blended_v2.png
```

### How it works

1. **Vision-Language Model**: Uses LLaVA to generate text descriptions of each input image
2. **Text Blending**: Combines the descriptions by concatenating them into a unified prompt
3. **Text-to-Image Generation**: Uses Stable Diffusion to generate the final blended image from the combined text prompt

> **Why this works**: By working in semantic text space rather than pixel/latent space, this approach can create more creative and interpretive blends based on conceptual understanding of the image contents.

### CLI options

```
usage: blend_v2.py [-h] [--output OUTPUT] [--size SIZE] [--guidance-scale GUIDANCE_SCALE] 
                   [--num-inference-steps NUM_INFERENCE_STEPS] [--device DEVICE] 
                   images [images ...]

Positional arguments:
  images                Paths of the images you want to blend (≥2).

Optional arguments:
  -h, --help            show this help message and exit
  --output OUTPUT, -o OUTPUT
                        Output filename (default: blended_v2.png)
  --size SIZE           Output image size (square) (default: 384)
  --guidance-scale GUIDANCE_SCALE
                        Guidance scale for text-to-image generation (default: 7.5)
  --num-inference-steps NUM_INFERENCE_STEPS
                        Number of inference steps for text-to-image generation (default: 25)
  --device DEVICE       Torch device to run on (e.g. cuda, mps, cpu). Auto-detected by default.
```

---

## Comparison of Methods

| Aspect | Method 1 (Latent Space) | Method 2 (Semantic Text) |
|--------|------------------------|--------------------------|
| **Approach** | Direct latent averaging | Text-based conceptual blending |
| **Fidelity** | High pixel-level fidelity | Creative interpretation |
| **Speed** | Fast (~seconds) | Slower (~minutes) |
| **Memory** | Lower (~335 MB VAE) | Higher (~multiple models) |
| **Creativity** | Conservative blends | Artistic/interpretive blends |
| **Best for** | Preserving image details | Creative artistic fusion |

---

## Usage Notes

### Performance Optimization

* **Apple Silicon Mac**: Both methods automatically use the *MPS* backend (macOS GPU) when available
* **GPU Memory**: Method 2 uses more GPU memory due to multiple models; Method 1 is more memory-efficient
* **Speed**: Method 1 is significantly faster for quick blending tasks

### Best Practices

* **Image Quality**: For best results with either method, supply images with visually related content
* **Aspect Ratios**: All images are center-cropped to squares before processing
* **Method Selection**: 
  - Choose **Method 1** for preserving image details and fast processing
  - Choose **Method 2** for creative, artistic interpretations and conceptual blending

### Experimentation

* **Method 1**: Try different blend modes (`mean` vs `max`) or experiment with weighted averages
* **Method 2**: Adjust guidance scale and inference steps for varying creativity levels and quality

---

## License

The code in this repository is licensed under the MIT License.  The VAE weights are licensed under the [CreativeML Open RAIL-M](https://raw.githubusercontent.com/Stability-AI/stablediffusion/main/LICENSE).  By downloading and using the weights, you agree to their license terms. 