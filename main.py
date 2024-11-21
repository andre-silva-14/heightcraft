#! /usr/bin/env python3
from lib.model_loader import ModelLoader
from lib.resolution_calculator import ResolutionCalculator
from lib.height_map_generator import HeightMapGenerator
from lib.large_model_handler import LargeModelHandler
import argparse
import logging

def setup_logging():
    """Configures logging for the application."""
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )

def parse_arguments():
    """Parses and validates command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate a height map from a 3D model.")
    parser.add_argument("file_path", type=str, help="Path to the input 3D model file.")
    parser.add_argument("--output_path", type=str, default="height_map.png", help="Path to save the generated height map. Default: height_map.png")
    parser.add_argument("--max_resolution", type=int, default=256, help="Maximum resolution for the longest dimension. Default: 256.")
    parser.add_argument("--use_gpu", action="store_true", help="Enable GPU acceleration. Default: CPU only.")
    parser.add_argument("--num_samples", type=int, default=10000, help="Number of points to sample from the 3D model surface. Default: 10,000.")
    parser.add_argument("--num_threads", type=int, default=4, help="Number of threads for parallel processing on CPU. Default: 4.")
    parser.add_argument("--bit_depth", type=int, choices=[8, 16], default=16, help="Bit depth for the height map. Default: 16.")
    parser.add_argument("--large_model", action="store_true", help="Use memory-efficient techniques for large models.")
    parser.add_argument("--chunk_size", type=int, default=1000000, help="Chunk size for processing large models. Default: 1,000,000.")

    args = parser.parse_args()
    validate_arguments(args)
    return args

def validate_arguments(args):
    """Validates the parsed arguments."""
    if args.max_resolution <= 0:
        raise ValueError("Maximum resolution must be a positive integer.")
    if args.num_samples <= 0:
        raise ValueError("Number of samples must be a positive integer.")
    if args.num_threads < 1:
        raise ValueError("Number of threads must be at least 1.")
    if args.chunk_size <= 0:
        raise ValueError("Chunk size must be a positive integer.")

def main():
    """Main function to run the height map generation process."""
    setup_logging()
    args = parse_arguments()

    try:
        with resource_manager.gpu_session():
            if args.large_model:
                handler = LargeModelHandler(args.file_path, args.chunk_size)
                handler.load_model_info()
                min_coords, max_coords = handler.calculate_bounding_box()
                target_resolution = ResolutionCalculator.calculate_from_bounds(min_coords, max_coords, args.max_resolution)
                
                if args.use_gpu:
                    sampled_points = handler.sample_points_gpu(args.num_samples)
                else:
                    sampled_points = handler.sample_points_cpu(args.num_samples)
                
                height_map = HeightMapGenerator.generate_from_points(
                    sampled_points,
                    target_resolution=target_resolution,
                    bit_depth=args.bit_depth
                )
            else:
                mesh = ModelLoader.load(args.file_path)
                target_resolution = ResolutionCalculator.calculate(mesh, args.max_resolution)
                height_map = HeightMapGenerator.generate(
                    mesh,
                    target_resolution=target_resolution,
                    use_gpu=args.use_gpu,
                    num_samples=args.num_samples,
                    num_threads=args.num_threads,
                    bit_depth=args.bit_depth
                )
            
            HeightMapGenerator.save_height_map(height_map, args.output_path)
    except Exception as e:
        logging.error(f"Error: {e}")
        raise
    finally:
        # Ensure cleanup even if an exception occurs
        resource_manager.clear_gpu_memory()

if __name__ == "__main__":
    main()
