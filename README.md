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
- 🔄 Intelligent caching system for faster repeated processing
- 📊 Memory usage monitoring and optimization
- 🛡️ Robust error handling and resource cleanup

## 🔧 Supported 3D Model Formats

- STL (Standard Triangle Language)
- OBJ (Wavefront Object)
- PLY (Polygon File Format)
- GLB (GL Transmission Binary Format)
- GLTF (GL Transmission Format)

## 📦 Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management. Here's how to get started:

1. Install uv using pipx (recommended):
```bash
pipx install uv
```

2. Create a virtual environment and install dependencies:
```bash
uv venv
source .venv/bin/activate  # On Unix/macOS
.venv\Scripts\activate     # On Windows
uv sync
```

For development setup:
```bash
uv sync --group dev
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

Adjust sampling density (Default: 100000):

```shellscript
python main.py path/to/model.obj --num_samples 200000
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

### Memory Management

Control memory usage (Default: 80% of available memory):

```shellscript
python main.py path/to/model.obj --max_memory 0.6
```

Specify cache directory for faster repeated processing (default: `.cache` in current directory):

```shellscript
python main.py path/to/model.obj --cache_dir ./cache
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
  --num_samples 1000000 \
  --upscale \
  --pretrained_model path/to/model.h5 \
  --split 4 \
  --max_memory 0.7 \
  --cache_dir ./cache
```


## Development

To install development dependencies:
```bash
uv pip install -e ".[dev]"
```


Running the legacy code:

```shellscript
python3 main.py --legacy path/to/model.obj
```

## 🧪 Testing

Run the test suite:

```shellscript
pytest tests
```


## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and pull requests. When contributing:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.