#! /usr/bin/env python3
import argparse
import logging
import math
import sys

from lib.height_map_generator import HeightMapGenerator
from lib.resolution_calculator import ResolutionCalculator


def setup_logging():
    """Configures logging for the application."""
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
    )


def validate_split(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(
            f"{value} is an invalid split value. Must be a positive integer."
        )
    if not math.isqrt(ivalue) ** 2 == ivalue and not any(
        ivalue % i == 0 for i in range(2, int(math.sqrt(ivalue)) + 1)
    ):
        raise argparse.ArgumentTypeError(
            f"{value} is an invalid split value. Must be able to form a grid (e.g., 4, 9, 12)."
        )
    return ivalue


def parse_arguments():
    """Parses and validates command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate a height map from a 3D model."
    )
    parser.add_argument("file_path", type=str, help="Path to the input 3D model file.")
    parser.add_argument(
        "--output_path",
        type=str,
        default="height_map.png",
        help="Path to save the generated height map. Default: height_map.png",
    )
    parser.add_argument(
        "--max_resolution",
        type=int,
        default=256,
        help="Maximum resolution for the longest dimension. Default: 256.",
    )
    parser.add_argument(
        "--use_gpu",
        action="store_true",
        help="Enable GPU acceleration. Default: CPU only.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100000,
        help="Number of points to sample from the 3D model surface. Default: 10,000.",
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=4,
        help="Number of threads for parallel processing on CPU. Default: 4.",
    )
    parser.add_argument(
        "--bit_depth",
        type=int,
        choices=[8, 16],
        default=16,
        help="Bit depth for the height map. Default: 16.",
    )
    parser.add_argument(
        "--large_model",
        action="store_true",
        help="Use memory-efficient techniques for large models.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=1000000,
        help="Chunk size for processing large models. Default: 1,000,000.",
    )
    parser.add_argument(
        "--split",
        type=validate_split,
        default=1,
        help="Number of files to split the output into (must form a grid)",
    )
    parser.add_argument(
        "--upscale",
        action="store_true",
        help="Enable upscaling of the generated height map.",
    )
    parser.add_argument(
        "--upscale_factor",
        type=int,
        default=2,
        help="Factor by which to upscale the height map. Default: 2.",
    )
    parser.add_argument(
        "--pretrained_model",
        type=str,
        help="Path to pretrained upscaling model (optional).",
    )

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
    if args.upscale and args.upscale_factor < 2:
        raise ValueError("Upscale factor must be at least 2.")


def main():
    """Main function to run the height map generation process."""
    setup_logging()
    args = parse_arguments()

    try:
        if args.large_model:
            from lib.large_model_handler import LargeModelHandler

            handler = LargeModelHandler(args.file_path, args.chunk_size)
            handler.load_model_info()
            min_coords, max_coords = handler.calculate_bounding_box()
            target_resolution = ResolutionCalculator.calculate_from_bounds(
                min_coords, max_coords, args.max_resolution
            )

            sampled_points = handler.sample_points(args.num_samples, args.use_gpu)

            height_map = HeightMapGenerator.generate_from_points(
                sampled_points,
                target_resolution=target_resolution,
                bit_depth=args.bit_depth,
            )
        else:
            from lib.model_loader import ModelLoader

            mesh = ModelLoader.load(args.file_path)
            target_resolution = ResolutionCalculator.calculate(
                mesh, args.max_resolution
            )
            height_map = HeightMapGenerator.generate(
                mesh,
                target_resolution=target_resolution,
                use_gpu=args.use_gpu,
                num_samples=args.num_samples,
                num_threads=args.num_threads,
                bit_depth=args.bit_depth,
            )

        if args.upscale:
            from lib.upscaler import HeightMapUpscaler, load_pretrained_model

            logging.info("Upscaling the generated height map...")
            upscaler = (
                load_pretrained_model(args.pretrained_model)
                if args.pretrained_model
                else HeightMapUpscaler()
            )
            height_map = upscaler.upscale(height_map, args.upscale_factor)
            logging.info("Upscaling completed.")

        HeightMapGenerator.save_height_map(
            height_map, args.output_path, args.bit_depth, args.split
        )
    except Exception as e:
        logging.error(f"Error: {e}")
        raise


def run_tests():
    """Run all unit tests."""
    pytest.main(["-v", "tests"])


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        run_tests()
    else:
        main()
