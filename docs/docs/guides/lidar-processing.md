---
sidebar_position: 2
---

# LiDAR Processing

Heightcraft provides robust support for processing LiDAR point clouds (`.las`, `.laz`). It is designed to handle large datasets efficiently using a streaming, chunked pipeline.

## Features

- **Format Support**: `.las` and `.laz` (compressed LAS).
- **Streaming**: Processes files chunk-by-chunk to keep memory usage low.
- **Auto-Detection**: Automatically detects LiDAR files based on extension.
- **32-bit Output**: Supports 32-bit Float TIFF for maximum precision.

## Usage

Simply provide a `.las` or `.laz` file as input.

```bash
heightcraft scan.laz --output_path dem.tiff --bit_depth 32 --max_resolution 2048
```

## Memory Management

For extremely large files (multi-gigabyte), Heightcraft automatically uses a chunked approach. You can control the chunk size if needed, though the default is usually optimal.

```bash
# Process in chunks of 5 million points
heightcraft scan.laz --chunk_size 5000000
```

## Integration with Upscaling

You can combine LiDAR processing with AI upscaling to get ultra-high-resolution DEMs from lower-density point clouds.

```bash
heightcraft scan.laz --upscale --upscale_factor 2 --output_path high_res_dem.tiff
```
