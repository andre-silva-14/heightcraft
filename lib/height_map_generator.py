import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from typing import Tuple, Union
import trimesh
from .point_sampler import PointSampler
import os
import math

class HeightMapGenerator:
    @staticmethod
    def generate(mesh: trimesh.Trimesh, target_resolution: Tuple[int, int], use_gpu: bool, num_samples: int, num_threads: int, bit_depth: int = 16) -> np.ndarray:
        """
        Generates a height map (grayscale image) from a 3D model.
        
        Args:
            mesh (trimesh.Trimesh): The 3D model mesh.
            target_resolution (Tuple[int, int]): Target resolution (width, height) for the height map.
            use_gpu (bool): Whether to use GPU for point sampling.
            num_samples (int): Number of points to sample from the 3D model surface.
            num_threads (int): Number of threads for parallel processing on CPU.
            bit_depth (int): Bit depth for the height map (8 or 16).
        
        Returns:
            np.ndarray: Generated height map.
        """
        logging.info(f"Generating height map with resolution {target_resolution}...")

        sampled_points = PointSampler.sample_points(mesh, num_samples, use_gpu, num_threads)
        return HeightMapGenerator.generate_from_points(sampled_points, target_resolution, bit_depth)

    @staticmethod
    def generate_from_points(sampled_points: np.ndarray, target_resolution: Tuple[int, int], bit_depth: int = 16) -> np.ndarray:
        """
        Generates a height map (grayscale image) from sampled points.
        
        Args:
            sampled_points (np.ndarray): Sampled points from the 3D model surface.
            target_resolution (Tuple[int, int]): Target resolution (width, height) for the height map.
            bit_depth (int): Bit depth for the height map (8 or 16).
        
        Returns:
            np.ndarray: Generated height map.
        """
        logging.info(f"Generating height map from points with resolution {target_resolution}...")

        points_2d = sampled_points[:, :2]
        z_values = sampled_points[:, 2]

        min_x, max_x = points_2d[:, 0].min(), points_2d[:, 0].max()
        min_y, max_y = points_2d[:, 1].min(), points_2d[:, 1].max()
        min_z, max_z = z_values.min(), z_values.max()

        width, height = target_resolution
        height_map = np.zeros((height, width), dtype=np.float32)

        x_coords = ((points_2d[:, 0] - min_x) / (max_x - min_x) * (width - 1)).astype(int)
        y_coords = ((points_2d[:, 1] - min_y) / (max_y - min_y) * (height - 1)).astype(int)
        z_normalized = (z_values - min_z) / (max_z - min_z)

        np.maximum.at(height_map, (y_coords, x_coords), z_normalized)

        if bit_depth == 8:
            height_map = (height_map * 255).astype(np.uint8)
        elif bit_depth == 16:
            height_map = (height_map * 65535).astype(np.uint16)
        else:
            raise ValueError("Bit depth must be either 8 or 16.")

        logging.info("Height map generation complete.")
        return height_map

    @staticmethod
    def save_height_map(height_map: np.ndarray, output_path: str, bit_depth: int, split: int = 1, is_part: bool = False) -> None:
        """
        Saves the generated height map to the specified output path.
        
        Args:
            height_map (np.ndarray): The generated height map.
            output_path (str): Path to save the generated height map.
            bit_depth (int): The bit-depth used for the generated height map.
            split (int): Number of files to split the output into (must form a grid).
            is_part (bool): Whether the saved heightmap is parcial or full
        """
        if split == 1:
            file_extension = Path(output_path).suffix
            output_format = file_extension[1:].lower()

            if bit_depth == 16 and output_format in ['jpg', 'jpeg']:
                # Re-Normalize the height map to the range [0, 1]
                logging.warning(f"The choosen output format {output_format} does not support 16 bit-depth. Falling back to 8 bit-depth.")
                normalized_map = (height_map - np.min(height_map)) / (np.max(height_map) - np.min(height_map))
                height_map = (normalized_map * 255).astype(np.uint8)

            # Save the image
            if output_format in ['png', 'tiff', 'tif', 'jpg', 'jpeg']:
                plt.imsave(output_path, height_map, cmap="gray", format=output_format)
            else:
                # Default to PNG if the format is not recognized
                logging.warning(f"The choosen output format {output_format} is not supported. Falling back to PNG.")
                plt.imsave(output_path, height_map, cmap="gray", format='png')
            
            if is_part:
                logging.info(f"Height map part saved to {output_path}")
            else:
                logging.info(f"Height map saved to {output_path}")
        else:
            HeightMapGenerator._save_split_height_maps(height_map, output_path, bit_depth, split)

    @staticmethod
    def _save_split_height_maps(height_map: np.ndarray, output_path: str, bit_depth: int, split: int) -> None:
        """
        Splits the height map into a grid and saves each part as a separate file.
        
        Args:
            height_map (np.ndarray): The generated height map.
            output_path (str): Base path for saving the split height maps.
            bit_depth (int): The bit-depth used for the generated height map.
            split (int): Number of files to split the output into.
        """
        height, width = height_map.shape
        grid_rows, grid_cols = HeightMapGenerator._get_optimal_grid(split, height, width)
        
        part_height = height // grid_rows
        part_width = width // grid_cols

        base_name, ext = os.path.splitext(output_path)
        
        for i in range(grid_rows):
            for j in range(grid_cols):
                start_y = i * part_height
                end_y = start_y + part_height
                start_x = j * part_width
                end_x = start_x + part_width
                
                part = height_map[start_y:end_y, start_x:end_x]
                part_path = f"{base_name}_part_{i}_{j}{ext}"
                
                HeightMapGenerator.save_height_map(part_path, part, bit_depth, is_part=True)

    @staticmethod
    def _get_optimal_grid(split: int, height: int, width: int) -> Tuple[int, int]:
        """
        Determines the optimal grid layout based on the split value and image dimensions.
        
        Args:
            split (int): Number of files to split the output into.
            height (int): Height of the image.
            width (int): Width of the image.
        
        Returns:
            Tuple[int, int]: Number of rows and columns in the grid.
        """
        aspect_ratio = width / height
        
        # Check if split is a perfect square
        sqrt_split = math.isqrt(split)
        if sqrt_split ** 2 == split:
            return sqrt_split, sqrt_split
        
        # Find the factor pair that's closest to the aspect ratio
        factors = [(i, split // i) for i in range(1, int(math.sqrt(split)) + 1) if split % i == 0]
        optimal_factor = min(factors, key=lambda x: abs(x[1] / x[0] - aspect_ratio))
        
        # Determine which dimension (height or width) should have more divisions
        if aspect_ratio >= 1:
            return optimal_factor[0], optimal_factor[1]  # More columns than rows
        else:
            return optimal_factor[1], optimal_factor[0]  # More rows than columns
