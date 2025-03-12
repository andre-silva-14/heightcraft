"""
Argument parser for Heightcraft CLI.

This module provides functions for parsing and validating command-line arguments.
"""

import argparse
import math
from typing import Any, Dict, List, Optional


def validate_split(value: str) -> int:
    """
    Validate the split argument.
    
    Args:
        value: Split value as string
        
    Returns:
        Split value as integer
        
    Raises:
        argparse.ArgumentTypeError: If the value is invalid
    """
    try:
        ivalue = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value} is not an integer")
    
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(
            f"{value} is an invalid split value. Must be a positive integer."
        )
    
    # Check if the value can form a grid
    if not math.isqrt(ivalue) ** 2 == ivalue and not any(
        ivalue % i == 0 for i in range(2, int(math.sqrt(ivalue)) + 1)
    ):
        raise argparse.ArgumentTypeError(
            f"{value} is an invalid split value. Must be able to form a grid (e.g., 4, 9, 12)."
        )
    
    return ivalue


def create_parser() -> argparse.ArgumentParser:
    """
    Create the argument parser.
    
    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="Generate height maps from 3D models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Required arguments
    parser.add_argument(
        "file_path", 
        type=str, 
        help="Path to the 3D model file."
    )
    
    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--output_path",
        type=str,
        default="height_map.png",
        help="Output path for the height map.",
    )
    output_group.add_argument(
        "--max_resolution",
        type=int,
        default=256,
        help="Maximum resolution for the height map.",
    )
    output_group.add_argument(
        "--bit_depth",
        type=int,
        choices=[8, 16],
        default=16,
        help="Bit depth for the height map.",
    )
    output_group.add_argument(
        "--split",
        type=validate_split,
        default=1,
        help="Number of files to split the output into (must form a grid).",
    )
    
    # Processing options
    processing_group = parser.add_argument_group("Processing Options")
    processing_group.add_argument(
        "--use_gpu",
        action="store_true",
        help="Enable GPU acceleration.",
    )
    processing_group.add_argument(
        "--num_samples",
        type=int,
        default=100000,
        help="Number of points to sample from the 3D model surface.",
    )
    processing_group.add_argument(
        "--num_threads",
        type=int,
        default=4,
        help="Number of threads for parallel processing on CPU.",
    )
    
    # Large model options
    large_model_group = parser.add_argument_group("Large Model Options")
    large_model_group.add_argument(
        "--large_model",
        action="store_true",
        help="Use memory-efficient techniques for large models.",
    )
    large_model_group.add_argument(
        "--extreme_memory_saving",
        action="store_true",
        help="Use extreme memory-saving techniques for very large models. May be slower.",
    )
    large_model_group.add_argument(
        "--chunk_size",
        type=int,
        default=1000000,
        help="Chunk size for processing large models.",
    )
    large_model_group.add_argument(
        "--max_memory",
        type=float,
        default=0.8,
        help="Maximum memory usage as a fraction of available memory.",
    )
    large_model_group.add_argument(
        "--cache_dir",
        type=str,
        help="Directory to use for caching. Default: .cache in current directory.",
    )
    
    # Upscaling options
    upscale_group = parser.add_argument_group("Upscaling Options")
    upscale_group.add_argument(
        "--upscale",
        action="store_true",
        help="Enable AI upscaling of the height map.",
    )
    upscale_group.add_argument(
        "--upscale_factor",
        type=int,
        default=2,
        choices=[2, 3, 4],
        help="Factor by which to upscale the height map.",
    )
    upscale_group.add_argument(
        "--pretrained_model",
        type=str,
        help="Path to a pretrained upscaling model.",
    )
    
    # Miscellaneous options
    misc_group = parser.add_argument_group("Miscellaneous Options")
    misc_group.add_argument(
        "--test",
        action="store_true",
        help="Run tests instead of generating a height map.",
    )
    misc_group.add_argument(
        "--verbose", "-v",
        action="count",
        default=0,
        help="Increase verbosity level. Use -v for INFO, -vv for DEBUG.",
    )
    
    return parser


def parse_arguments(args: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Parse command-line arguments.
    
    Args:
        args: Command-line arguments (defaults to sys.argv[1:])
        
    Returns:
        Dictionary of parsed arguments
        
    Raises:
        SystemExit: If argument parsing fails
    """
    parser = create_parser()
    parsed_args = parser.parse_args(args)
    return vars(parsed_args)


def validate_arguments(args: Dict[str, Any]) -> None:
    """
    Validate parsed arguments.
    
    Args:
        args: Dictionary of parsed arguments
        
    Raises:
        ValueError: If validation fails
    """
    if args["max_resolution"] <= 0:
        raise ValueError("Maximum resolution must be a positive integer.")
    
    if args["num_samples"] <= 0:
        raise ValueError("Number of samples must be a positive integer.")
    
    if args["num_threads"] < 1:
        raise ValueError("Number of threads must be at least 1.")
    
    if args["chunk_size"] <= 0:
        raise ValueError("Chunk size must be a positive integer.")
    
    if args["upscale"] and args["upscale_factor"] < 2:
        raise ValueError("Upscale factor must be at least 2.")
    
    if not 0 < args["max_memory"] <= 1:
        raise ValueError("Maximum memory must be between 0 and 1.") 