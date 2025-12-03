<div align="center">
  <img src=".github/assets/banner.png" alt="Heightcraft Banner" width="60%" />
  
  **Heightmap Generation & AI Upscaling Tool**

  [![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![Documentation](https://img.shields.io/badge/docs-docusaurus-green)](./docs)
[![PyTest Coverage](https://img.shields.io/badge/coverage-90%25-green)](./tests)
</div>



**Heightcraft** is a high-performance CLI tool designed for **Game Engineers**, **GIS Professionals**, and **Researchers**. It automates the pipeline of converting 3D data into high-precision heightmaps, featuring accessible **AI Upscaling**.

---

## 🚀 Key Features

- **🤖 Heightmap AI Upscaling**: Transform low-res inputs into crisp, hi-res/high-detail heightmaps. Also supporting bit depth increase (8-bit → 16/32-bit) with AI hallucination.
- **📡 LiDAR Support**: Stream process massive `.las` and `.laz` point clouds into Digital Elevation Models (DEMs).
- **🏔️ Mesh to Heightmap**: Bake `.gltf`, `.glb`, `.obj`, `.stl`, and `.ply` meshes into heightmaps with automated chunking.
- **🌊 Feature Masks**: Optionally export water masks with sea level thresholds and texture masks for slope and curvature.
- **🎯 High Precision**: Native support for **32-bit Float TIFF**, **16-bit PNG**, and **RAW** formats.
- **⚡ Performance**: GPU acceleration and memory-efficient streaming for gigabyte-scale datasets.

## 📖 Documentation

Full documentation is available in the [Docs](https://andre-silva-14.github.io/heightcraft/).

- [**Getting Started**](https://andre-silva-14.github.io/heightcraft/docs/getting-started)
- [**LiDAR Processing**](https://andre-silva-14.github.io/heightcraft/docs/guides/lidar-processing)
- [**Mesh Processing**](https://andre-silva-14.github.io/heightcraft/docs/guides/mesh-processing)
- [**AI Upscaling Deep Dive**](https://andre-silva-14.github.io/heightcraft/docs/guides/ai-upscaling)

## 📦 Installation

```bash
pipx install heightcraft
```

## ⚡ Quick Start

**Upscale an image (8-bit → 16-bit and 3x upscale):**
```bash
heightcraft input.png --upscale --upscale-factor 3 --bit_depth 16 --output_path high_res.png
```

**Process LiDAR data into an high-res 32-bit heightmap:**
```bash
heightcraft scan.laz --bit_depth 32 --chunk_size 600000 --max_resolution 24576
```

**Convert a 3D Mesh to an upscaled heightmap:**
```bash
heightcraft terrain.obj --large-model --chunk_size 200000 --max_resolution 8192 --upscale
```

## Development

To install development dependencies:
```bash
pipx install uv
uv sync --group dev
uv run main.py
```

## 🧪 Testing

Run the test suite:

```bash
pytest tests
```
