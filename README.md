# 🗺️ Heightcraft

Transform your 3D models into super detailed height maps with Heightcraft - a powerful and flexible height map generator that supports multiple formats and advanced processing options.

## ✨ Features

- 🎮 GPU-accelerated processing for lightning-fast generation
- 🎯 Smart resolution calculation based on model proportions
- 🎨 High-precision output (8-bit or 16-bit depth)
- 🧩 Split large maps into manageable pieces
- 🔍 AI-powered upscaling for enhanced detail
- 💾 Memory-efficient processing for large models
- 🚀 Multi-threaded CPU processing

## 🔧 Supported 3D Model Formats

- STL (Standard Triangle Language)
- OBJ (Wavefront Object)
- PLY (Polygon File Format)
- GLB (GL Transmission Binary Format)
- GLTF (GL Transmission Format)

## 📦 Installation

Ensure you have Python 3.x installed, then install the required packages:

```bash
pip install -r requirements.txt
```

## 🚀 Usage

### Basic Usage

Generate a height map with default settings:

```shellscript
python main.py path/to/model.obj
```

### Output Options

Customize the output file (Default: height_map.png):

```shellscript
python main.py path/to/model.obj --output_path my_heightmap.png
```

Set the maximum resolution (Default: 256):

```shellscript
python main.py path/to/model.obj --max_resolution 512
```

Choose bit depth - 8 or 16 (Default: 16):

```shellscript
python main.py path/to/model.obj --bit_depth 8
```

### Processing Options

Use GPU acceleration:

```shellscript
python main.py path/to/model.obj --use_gpu
```

Adjust sampling density (Default: 10000):

```shellscript
python main.py path/to/model.obj --num_samples 20000
```

Set CPU thread count (Default: 4):

```shellscript
python main.py path/to/model.obj --num_threads 8
```

### Advanced Features

Split output into multiple files:

```shellscript
python main.py path/to/model.obj --split 4  # Splits into 2x2 grid
python main.py path/to/model.obj --split 9  # Splits into 3x3 grid
```

Enable AI upscaling:

```shellscript
python main.py path/to/model.obj --upscale --upscale_factor 2
```

Use a pretrained upscaling model:

```shellscript
python main.py path/to/model.obj --upscale --pretrained_model path/to/model.h5
```

### Large Model Processing

Handle large models efficiently (default `chunk_size` is 1000000):

```shellscript
python main.py path/to/large_model.obj --large_model --chunk_size 1500000
```

## 🔄 Complete Example

Generate a high-resolution height map with GPU acceleration, upscaling, and splitting:

```shellscript
python main.py model.obj \
  --output_path detailed_map.png \
  --max_resolution 1024 \
  --use_gpu \
  --num_samples 50000 \
  --upscale \
  --pretrained_model path/to/model.h5 \
  --split 4
```

## 🤝 Contributing

Contributions are welcome! Feel free to submit issues and pull requests.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.
