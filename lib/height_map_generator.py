import gc
import logging
import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import trimesh

from .point_sampler import PointSampler


class HeightMapFormat(Enum):
    """Supported height map formats."""

    PNG = auto()
    TIFF = auto()
    JPEG = auto()
    UNKNOWN = auto()

    @classmethod
    def from_extension(cls, extension: str) -> "HeightMapFormat":
        """Convert file extension to format enum."""
        ext = extension.lower()
        if ext in [".png"]:
            return cls.PNG
        elif ext in [".tiff", ".tif"]:
            return cls.TIFF
        elif ext in [".jpg", ".jpeg"]:
            return cls.JPEG
        return cls.UNKNOWN


@dataclass
class HeightMapConfig:
    """Configuration for height map generation."""

    target_resolution: Tuple[int, int]
    bit_depth: int
    num_samples: int
    num_threads: int
    use_gpu: bool
    split: int = 1
    cache_dir: Optional[str] = None
    max_memory: float = 0.8


class HeightMapGenerator:
    """Generates height maps from 3D models with advanced features and optimizations."""

    def __init__(self, config: HeightMapConfig):
        """
        Initialize the height map generator.

        Args:
            config: Configuration for height map generation
        """
        self.config = config
        self._setup_cache()
        self._height_map_cache: Dict[str, np.ndarray] = {}
        self._setup_thread_pool()

    def _setup_cache(self) -> None:
        """Set up the cache directory."""
        if self.config.cache_dir:
            self.cache_dir = self.config.cache_dir
        else:
            self.cache_dir = os.path.join(os.getcwd(), ".cache")
        os.makedirs(self.cache_dir, exist_ok=True)

    def _setup_thread_pool(self) -> None:
        """Set up the thread pool for parallel processing."""
        self.thread_pool = ThreadPoolExecutor(max_workers=self.config.num_threads)

    def generate(self, mesh: trimesh.Trimesh) -> np.ndarray:
        """
        Generates a height map from a 3D model.

        Args:
            mesh: The 3D model mesh

        Returns:
            Generated height map
        """
        logging.info(
            f"Generating height map with resolution {self.config.target_resolution}..."
        )

        try:
            # Check cache first
            cache_key = self._get_cache_key(mesh)
            if cache_key in self._height_map_cache:
                logging.info("Using cached height map")
                return self._height_map_cache[cache_key]

            # Sample points from the mesh
            sampled_points = PointSampler.sample_points(
                mesh,
                self.config.num_samples,
                self.config.use_gpu,
                self.config.num_threads,
            )

            # Generate height map from points
            height_map = self._generate_from_points(sampled_points)

            # Cache the result
            self._height_map_cache[cache_key] = height_map

            return height_map

        except Exception as e:
            logging.error(f"Error generating height map: {e}")
            raise

    def _generate_from_points(self, sampled_points: np.ndarray) -> np.ndarray:
        """Generate height map from sampled points with optimizations."""
        logging.info(
            f"Generating height map from points with resolution {self.config.target_resolution}..."
        )

        # Extract 2D points and Z values
        points_2d = sampled_points[:, :2]
        z_values = sampled_points[:, 2]

        # Calculate bounds
        bounds = self._calculate_bounds(points_2d, z_values)

        # Create height map with parallel processing
        height_map = self._create_height_map(points_2d, z_values, bounds)

        # Apply bit depth conversion
        return self._convert_bit_depth(height_map)

    def _calculate_bounds(
        self, points_2d: np.ndarray, z_values: np.ndarray
    ) -> Dict[str, float]:
        """Calculate bounds for the height map."""
        return {
            "min_x": points_2d[:, 0].min(),
            "max_x": points_2d[:, 0].max(),
            "min_y": points_2d[:, 1].min(),
            "max_y": points_2d[:, 1].max(),
            "min_z": z_values.min(),
            "max_z": z_values.max(),
        }

    def _create_height_map(
        self, points_2d: np.ndarray, z_values: np.ndarray, bounds: Dict[str, float]
    ) -> np.ndarray:
        """Create height map using parallel processing."""
        width, height = self.config.target_resolution
        height_map = np.zeros((height, width), dtype=np.float32)

        # Calculate coordinates in parallel
        futures = []
        chunk_size = len(points_2d) // self.config.num_threads
        for i in range(0, len(points_2d), chunk_size):
            chunk = slice(i, min(i + chunk_size, len(points_2d)))
            futures.append(
                self.thread_pool.submit(
                    self._process_chunk,
                    points_2d[chunk],
                    z_values[chunk],
                    bounds,
                    width,
                    height,
                )
            )

        # Combine results
        for future in futures:
            chunk_coords, chunk_values = future.result()
            np.maximum.at(
                height_map, (chunk_coords[:, 1], chunk_coords[:, 0]), chunk_values
            )

        return height_map

    @staticmethod
    def _process_chunk(
        points: np.ndarray,
        z_values: np.ndarray,
        bounds: Dict[str, float],
        width: int,
        height: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Process a chunk of points."""
        x_coords = (
            (points[:, 0] - bounds["min_x"])
            / (bounds["max_x"] - bounds["min_x"])
            * (width - 1)
        ).astype(int)
        y_coords = (
            (points[:, 1] - bounds["min_y"])
            / (bounds["max_y"] - bounds["min_y"])
            * (height - 1)
        ).astype(int)
        z_normalized = (z_values - bounds["min_z"]) / (
            bounds["max_z"] - bounds["min_z"]
        )
        return np.column_stack((x_coords, y_coords)), z_normalized

    def _convert_bit_depth(self, height_map: np.ndarray) -> np.ndarray:
        """Convert height map to specified bit depth."""
        if self.config.bit_depth == 8:
            return (height_map * 255).astype(np.uint8)
        elif self.config.bit_depth == 16:
            return (height_map * 65535).astype(np.uint16)
        else:
            raise ValueError("Bit depth must be either 8 or 16.")

    def save_height_map(
        self, height_map: np.ndarray, output_path: str, is_part: bool = False
    ) -> None:
        """Save height map with format handling and splitting support."""
        if self.config.split == 1:
            self._save_single_height_map(height_map, output_path, is_part)
        else:
            self._save_split_height_maps(height_map, output_path)

    def _save_single_height_map(
        self, height_map: np.ndarray, output_path: str, is_part: bool
    ) -> None:
        """Save a single height map with format handling."""
        file_format = HeightMapFormat.from_extension(Path(output_path).suffix)

        if file_format == HeightMapFormat.UNKNOWN:
            logging.warning("Unsupported format, defaulting to PNG")
            file_format = HeightMapFormat.PNG
            output_path = str(Path(output_path).with_suffix(".png"))

        # Handle JPEG 16-bit conversion
        if file_format == HeightMapFormat.JPEG and self.config.bit_depth == 16:
            logging.warning(
                "JPEG format does not support 16-bit depth, converting to 8-bit"
            )
            height_map = self._convert_to_8bit(height_map)

        # Save the image
        plt.imsave(output_path, height_map, cmap="gray")

        status = "part" if is_part else "full"
        logging.info(f"Height map {status} saved to {output_path}")

    def _save_split_height_maps(self, height_map: np.ndarray, output_path: str) -> None:
        """Save split height maps with parallel processing."""
        height, width = height_map.shape
        grid_rows, grid_cols = self._get_optimal_grid(self.config.split, height, width)

        part_height = height // grid_rows
        part_width = width // grid_cols
        base_name, ext = os.path.splitext(output_path)

        futures = []
        for i in range(grid_rows):
            for j in range(grid_cols):
                start_y = i * part_height
                end_y = start_y + part_height
                start_x = j * part_width
                end_x = start_x + part_width

                part = height_map[start_y:end_y, start_x:end_x]
                part_path = f"{base_name}_part_{i}_{j}{ext}"

                futures.append(
                    self.thread_pool.submit(
                        self._save_single_height_map, part, part_path, True
                    )
                )

        # Wait for all saves to complete
        for future in futures:
            future.result()

    @staticmethod
    def _get_optimal_grid(split: int, height: int, width: int) -> Tuple[int, int]:
        """Calculate optimal grid dimensions for splitting."""
        aspect_ratio = width / height

        # Check if split is a perfect square
        sqrt_split = math.isqrt(split)
        if sqrt_split**2 == split:
            return sqrt_split, sqrt_split

        # Find the factor pair that's closest to the aspect ratio
        factors = [
            (i, split // i)
            for i in range(1, int(math.sqrt(split)) + 1)
            if split % i == 0
        ]
        optimal_factor = min(factors, key=lambda x: abs(x[1] / x[0] - aspect_ratio))

        return (
            (optimal_factor[0], optimal_factor[1])
            if aspect_ratio >= 1
            else (optimal_factor[1], optimal_factor[0])
        )

    @staticmethod
    def _convert_to_8bit(height_map: np.ndarray) -> np.ndarray:
        """Convert 16-bit height map to 8-bit."""
        normalized = (height_map - np.min(height_map)) / (
            np.max(height_map) - np.min(height_map)
        )
        return (normalized * 255).astype(np.uint8)

    def _get_cache_key(self, mesh: trimesh.Trimesh) -> str:
        """Generate a cache key for the mesh."""
        if mesh is None:
            return f"none_mesh_{self.config.target_resolution}_{self.config.bit_depth}"
        return f"{hash(str(mesh.vertices))}_{hash(str(mesh.faces))}_{self.config.target_resolution}_{self.config.bit_depth}"

    def cleanup(self) -> None:
        """Clean up resources."""
        self.thread_pool.shutdown(wait=True)
        self._height_map_cache.clear()
        gc.collect()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()
