import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import Tuple, Union
import trimesh
from lib.point_sampler import PointSampler

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
    def save_height_map(height_map: np.ndarray, output_path: str) -> None:
        """
        Saves the generated height map to the specified output path.
        
        Args:
            height_map (np.ndarray): The generated height map.
            output_path (str): Path to save the generated height map.
        """
        plt.imsave(output_path, height_map, cmap="gray")
        logging.info(f"Height map saved to {output_path}")
