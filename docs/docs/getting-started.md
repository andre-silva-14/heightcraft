---
sidebar_position: 3
---

# Getting Started

This guide will walk you through the basic usage of Heightcraft.

## Basic Usage

The general syntax is:

```bash
heightcraft [INPUT_FILE] [OPTIONS]
```

### 1. Convert a 3D Mesh to Heightmap

```bash
heightcraft terrain.obj
```

### 2. Process LiDAR Data

```bash
heightcraft scan.laz --output_path dem.tiff --bit_depth 32
```

### 3. AI Upscaling (Image-to-Image)

Upscale a low-res heightmap and increase bit depth:

```bash
heightcraft low_res.png --upscale --upscale_factor 2 --bit_depth 16 --output_path high_res.png
```

## Options

| Option | Description | Default |
| :--- | :--- | :--- |
| `--output_path` | Path to save the generated heightmap. | `height_map.png` |
| `--max_resolution` | Defines the size of the longest edge, aspect ratio preserved (e.g., 2048). | `1024` |
| `--bit_depth` | Bit depth of the output (8, 16, 32). | `16` |
| `--split` | Splits the output into equal sized files (i.e. 9 provides a 3x3 grid) | `1` |
| `--num_threads` | Number of CPU threads to use (when in CPU mode) | `4` |
| `--use_gpu` | Enable GPU acceleration. | `False` |
| `--num_samples` | Number of samples to samples the object for heightmap generation. For higher resolutions, the number of samples needs to be exponentially higher to avoid gaps (Black Holes) | `100000` |
| `--large_model` | Enables chunk processing to avoid OutOfMemory errors. | `False` for 3D meshes. LiDAR data always processes in chunks. |
| `--chunk_size` | Defines the size of chunks | `1000000` |
| `--upscale` | Enable AI upscaling. | `False` |
| `--upscale_factor` | Factor to upscale by (2, 3, 4). | `2` |
| `--pretrained_model` | Filepath to the pretained model to use for AI upscaling. | `None` |
| `--sea_level` | Z-plane cut level. Values below this are flattened. Optionally exports water_mask.png. | `None` |
| `--slope_map` | Generate slope_map.png (white=steep, black=flat). | `False` |
| `--curvature_map` | Generate curvature_map.png (convex/concave). | `False` |

## Next Steps

Explore the deep dive guides to learn more about specific workflows:

- [Mesh Processing](./guides/mesh-processing)
- [LiDAR Processing](./guides/lidar-processing)
- [AI Upscaling](./guides/ai-upscaling)
- [Masking & Textures](./guides/masking-and-textures)
