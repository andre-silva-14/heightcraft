---
sidebar_position: 1
---

# AI Upscaling

Heightcraft's **AI Upscaling** is a powerful feature that allows you to increase the resolution and fidelity of your heightmaps. It uses deep learning models trained on high-resolution terrain data to hallucinate plausible details when upscaling.

## How it Works

The upscaling pipeline performs two key actions:
1.  **Super-Resolution**: Increases the pixel count (resolution) by 2x, 3x, or 4x.
2.  **Bit Depth Expansion**: Intelligently interpolates values to convert 8-bit inputs into 16-bit or 32-bit outputs, reducing "stair-stepping" artifacts common in low-bit-depth terrains.

## Usage

To enable upscaling, use the `--upscale` flag.

### Basic Upscaling

```bash
heightcraft input.png --upscale --output_path output.png
```

### Custom Factor

By default, the upscale factor is 2. You can change it to 3 or 4:

```bash
heightcraft input.png --upscale --upscale_factor 4 --output_path output.png
```

### Increasing Bit Depth

A common workflow is to take a standard 8-bit image and upscale it to a 16-bit heightmap for game engines.

```bash
heightcraft input_8bit.png --upscale --bit_depth 16 --output_path output_16bit.png
```

## Training Custom Models

Heightcraft supports training custom upscaling models on your own datasets.

```bash
heightcraft --train --dataset_path /path/to/high_res_tiffs --epochs 50
```

This will generate a `trained_model.h5` file that you can use with the `--pretrained_model` flag.

```bash
heightcraft input.png --upscale --pretrained_model my_model.h5
```
