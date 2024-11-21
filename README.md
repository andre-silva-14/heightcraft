# Mesh Heightmap

Generates a heightmap out of a 3D mesh object.

# Usage

Standard usage

    $ main.py path/to/model.obj

Customize output file (Default: height_map.png)

    $ main.py path/to/model.obj --output_path my_heightmap.png

Customize the longest dimention of the image resolution, mesh aspect ratio always persists (Default: 256)

    $ main.py path/to/model.obj --max_resolution 512

Customize the points to sample from the 3D model's surface (Default: 10000)

    $ main.py path/to/model.obj --num_samples 20000
    
Customize CPU Threads (Default: 4)

    $ main.py path/to/model.obj --num_threads 8

Use GPU acceleration instead of CPU processing

    $ main.py path/to/model.obj --use_gpu

