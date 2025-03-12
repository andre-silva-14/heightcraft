#! /usr/bin/env python3
import argparse
import gc
import logging
import math
import os
import sys

import numpy as np

from legacy.lib.height_map_generator import HeightMapConfig, HeightMapGenerator
from legacy.lib.resolution_calculator import ResolutionCalculator


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


def parse_arguments(raw_args):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate height maps from 3D models.")
    parser.add_argument("file_path", type=str, help="Path to the 3D model file.")
    parser.add_argument(
        "--output_path",
        type=str,
        default="height_map.png",
        help="Output path for the height map. Default: height_map.png",
    )
    parser.add_argument(
        "--max_resolution",
        type=int,
        default=256,
        help="Maximum resolution for the height map. Default: 256",
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
        help="Number of points to sample from the 3D model surface. Default: 100,000.",
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
        "--extreme_memory_saving",
        action="store_true",
        help="Use extreme memory-saving techniques for very large models. May be slower.",
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
        help="Enable AI upscaling of the height map.",
    )
    parser.add_argument(
        "--upscale_factor",
        type=int,
        default=2,
        choices=[2, 3, 4],
        help="Factor by which to upscale the height map. Default: 2.",
    )
    parser.add_argument(
        "--pretrained_model",
        type=str,
        help="Path to a pretrained upscaling model.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        help="Directory to use for caching. Default: .cache in current directory.",
    )
    parser.add_argument(
        "--max_memory",
        type=float,
        default=0.8,
        help="Maximum memory usage as a fraction of available memory. Default: 0.8 (80%).",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run tests instead of generating a height map.",
    )
    return parser.parse_args(raw_args)


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
    if not 0 < args.max_memory <= 1:
        raise ValueError("Maximum memory must be between 0 and 1.")


def main(raw_args = sys.argv[1:]):
    """Main function to run the height map generation process."""
    setup_logging()
    args = parse_arguments(raw_args)

    try:
        # Calculate target resolution
        if args.large_model:
            from legacy.lib.large_model_handler import LargeModelHandler

            # Adjust chunk size if extreme memory saving is enabled
            chunk_size = args.chunk_size
            if args.extreme_memory_saving:
                chunk_size = min(chunk_size, 100000)
                logging.info(
                    f"Extreme memory saving enabled, using chunk size: {chunk_size}"
                )

            with LargeModelHandler(
                args.file_path, chunk_size, max_memory=args.max_memory
            ) as handler:
                handler.load_model_info()
                min_coords, max_coords = handler.calculate_bounding_box()
                target_resolution = ResolutionCalculator.calculate_from_bounds(
                    min_coords, max_coords, args.max_resolution
                )

                # Create height map configuration
                config = HeightMapConfig(
                    target_resolution=target_resolution,
                    bit_depth=args.bit_depth,
                    num_samples=args.num_samples,
                    num_threads=args.num_threads,
                    use_gpu=args.use_gpu,
                    split=args.split,
                    cache_dir=args.cache_dir,
                    max_memory=args.max_memory,
                )

                # Generate height map
                with HeightMapGenerator(config) as generator:
                    # For GLTF scenes, we need to sample points directly
                    if handler.is_scene:
                        logging.info("Processing GLTF scene...")

                        # For extreme memory saving, process in batches
                        if args.extreme_memory_saving:
                            logging.info(
                                "Using extreme memory saving for point sampling"
                            )
                            # Reduce number of samples if needed
                            num_samples = min(args.num_samples, 10000000)
                            if num_samples < args.num_samples:
                                logging.warning(
                                    f"Reducing number of samples to {num_samples} for memory efficiency"
                                )

                            # Sample points in batches
                            batch_size = min(1000000, num_samples // 10)
                            num_batches = max(1, num_samples // batch_size)
                            logging.info(f"Sampling points in {num_batches} batches")

                            all_points = []
                            for i in range(num_batches):
                                logging.info(f"Sampling batch {i+1}/{num_batches}")
                                batch_points = handler.sample_points(
                                    batch_size, args.use_gpu
                                )
                                all_points.append(batch_points)
                                # Force garbage collection
                                gc.collect()

                            # Combine batches
                            points = np.vstack(all_points)
                            # If we need more points, duplicate some
                            if len(points) < num_samples:
                                additional = num_samples - len(points)
                                indices = np.random.choice(
                                    len(points), additional, replace=True
                                )
                                points = np.vstack([points, points[indices]])
                        else:
                            # Normal sampling
                            points = handler.sample_points(
                                args.num_samples, args.use_gpu
                            )

                        height_map = generator._generate_from_points(points)
                    else:
                        height_map = generator.generate(handler.concatenated_mesh)

                    if args.upscale:
                        from legacy.lib.upscaler import (
                            HeightMapUpscaler,
                            load_pretrained_model,
                        )

                        logging.info("Upscaling the generated height map...")
                        upscaler = (
                            load_pretrained_model(args.pretrained_model)
                            if args.pretrained_model
                            else HeightMapUpscaler()
                        )
                        height_map = upscaler.upscale(height_map, args.upscale_factor)
                        logging.info("Upscaling completed.")

                    generator.save_height_map(height_map, args.output_path)
        else:
            from legacy.lib.model_loader import ModelLoader

            mesh = ModelLoader.load(args.file_path)
            target_resolution = ResolutionCalculator.calculate(
                mesh, args.max_resolution
            )

            # Create height map configuration
            config = HeightMapConfig(
                target_resolution=target_resolution,
                bit_depth=args.bit_depth,
                num_samples=args.num_samples,
                num_threads=args.num_threads,
                use_gpu=args.use_gpu,
                split=args.split,
                cache_dir=args.cache_dir,
                max_memory=args.max_memory,
            )

            # Generate height map
            with HeightMapGenerator(config) as generator:
                height_map = generator.generate(mesh)

                if args.upscale:
                    from legacy.lib.upscaler import HeightMapUpscaler, load_pretrained_model

                    logging.info("Upscaling the generated height map...")
                    upscaler = (
                        load_pretrained_model(args.pretrained_model)
                        if args.pretrained_model
                        else HeightMapUpscaler()
                    )
                    height_map = upscaler.upscale(height_map, args.upscale_factor)
                    logging.info("Upscaling completed.")

                generator.save_height_map(height_map, args.output_path)

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
