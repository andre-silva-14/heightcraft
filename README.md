# 3D Model Height Map Generator

This project generates a height map from a 3D model file. It supports various 3D file formats and can utilize GPU acceleration for faster processing.

## Features

- Load and preprocess 3D models
- Calculate dynamic resolution based on model aspect ratio
- Sample points on the model surface using GPU or CPU
- Generate high-precision height maps with 16-bit greyscale (can be downsized to 8-bit)
- Save height maps in supported image formats

## Supported 3D Model Formats

- STL
- OBJ
- PLY
- GLB
- GLTF

## Installation

Ensure you have Python 3.x installed along with the required packages:

```bash
pip install trimesh numpy matplotlib torch
```

# Usage

Standard usage

    $ main.py path/to/model.obj

Customize output file (Default: height_map.png)

    $ main.py path/to/model.obj --output_path my_heightmap.png

Customize the longest dimention of the image resolution, mesh aspect ratio always persists (Default: 256)

    $ main.py path/to/model.obj --max_resolution 512

Select the bit depth of the outputed image (8 or 16) (Default: 16)

    $ main.py path/to/model.obj --bit_depth 8

Customize the points to sample from the 3D model's surface (Default: 10000)

    $ main.py path/to/model.obj --num_samples 20000
    
Customize CPU Threads (Default: 4)

    $ main.py path/to/model.obj --num_threads 8

Use GPU acceleration instead of CPU processing

    $ main.py path/to/model.obj --use_gpu

