"""
Height map service for working with height maps.

This module provides services for loading, saving, and manipulating height maps.
"""

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import ndimage

from heightcraft.core.exceptions import HeightMapServiceError
from heightcraft.domain.height_map import HeightMap
from heightcraft.domain.mesh import Mesh
from heightcraft.domain.point_cloud import PointCloud
from heightcraft.utils import converters


class HeightMapService:
    """Service for working with height maps."""
    
    def __init__(self, height_map_repository=None) -> None:
        """
        Initialize the height map service.
        
        Args:
            height_map_repository: Repository for loading and saving height maps.
        """
        self.height_map_repository = height_map_repository
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def load_height_map(self, file_path: str, resolution: float) -> HeightMap:
        """
        Load a height map from a file.
        
        Args:
            file_path: Path to the height map file.
            resolution: The resolution of the height map.
            
        Returns:
            A HeightMap object.
            
        Raises:
            HeightMapServiceError: If the height map cannot be loaded.
        """
        try:
            if self.height_map_repository:
                return self.height_map_repository.load(file_path, resolution)
            else:
                return HeightMap.from_file(file_path, resolution)
        except Exception as e:
            raise HeightMapServiceError(f"Failed to load height map from {file_path}: {str(e)}")
    
    def save_height_map(self, height_map: HeightMap, file_path: str) -> bool:
        """
        Save a height map to a file.
        
        Args:
            height_map: The height map to save.
            file_path: Path where to save the height map.
            
        Returns:
            True if the height map was saved successfully.
            
        Raises:
            HeightMapServiceError: If the height map cannot be saved.
        """
        try:
            if self.height_map_repository:
                return self.height_map_repository.save(height_map, file_path)
            else:
                return height_map.save(file_path)
        except Exception as e:
            raise HeightMapServiceError(f"Failed to save height map to {file_path}: {str(e)}")

    def split_height_map(self, height_map: HeightMap, split_count: int) -> List[HeightMap]:
        """
        Split a height map into multiple tiles.
        
        Args:
            height_map: The height map to split.
            split_count: The number of tiles to split into (must be a perfect square).
            
        Returns:
            List of height map tiles.
            
        Raises:
            HeightMapServiceError: If the height map cannot be split.
        """
        try:
            # Calculate grid size (tiles per side) from total split count
            grid_size = int(split_count ** 0.5)
            
            if grid_size * grid_size != split_count:
                raise HeightMapServiceError(f"Split count {split_count} must be a perfect square (e.g., 4, 9, 16)")
                
            return height_map.split(grid_size)
        except Exception as e:
            raise HeightMapServiceError(f"Failed to split height map: {str(e)}")

    def save_split_height_maps(self, split_maps: List[HeightMap], output_path: str) -> str:
        """
        Save split height maps to a directory.
        
        Args:
            split_maps: List of height map tiles.
            output_path: Base path for the output.
            
        Returns:
            Path to the directory containing the split maps.
            
        Raises:
            HeightMapServiceError: If the maps cannot be saved.
        """
        try:
            # Create output directory
            base_name = os.path.splitext(os.path.basename(output_path))[0]
            output_dir = os.path.join(os.path.dirname(os.path.abspath(output_path)), f"{base_name}_split")
            os.makedirs(output_dir, exist_ok=True)
            
            # Calculate grid dimensions
            num_tiles = len(split_maps)
            grid_size = int(num_tiles ** 0.5)
            
            # Save each tile
            for i, tile in enumerate(split_maps):
                row = i // grid_size
                col = i % grid_size
                
                # Construct tile path
                ext = os.path.splitext(output_path)[1]
                if not ext:
                    ext = ".png"
                
                tile_path = os.path.join(output_dir, f"{base_name}_{row}_{col}{ext}")
                
                # Save tile
                self.save_height_map(tile, tile_path)
            
            return output_dir
            
        except Exception as e:
            raise HeightMapServiceError(f"Failed to save split height maps: {str(e)}")
    
    def normalize_height_map(self, height_map: HeightMap) -> HeightMap:
        """
        Normalize the height map values to the range [0, 1].
        
        Args:
            height_map: The height map to normalize.
            
        Returns:
            A new normalized height map.
            
        Raises:
            HeightMapServiceError: If the height map cannot be normalized.
        """
        try:
            # Use converters for normalization
            normalized_data = converters.normalize_array(height_map.data, 0.0, 1.0)
            
            # Create a new height map with the normalized data
            return HeightMap(normalized_data, height_map.bit_depth)
        except Exception as e:
            raise HeightMapServiceError(f"Failed to normalize height map: {str(e)}")

    def generate_from_point_cloud(
        self, 
        point_cloud: PointCloud, 
        resolution: Tuple[int, int], 
        bit_depth: int = 16,
        num_threads: int = 4
    ) -> HeightMap:
        """
        Generate a height map from a point cloud.
        
        Args:
            point_cloud: The point cloud to generate from.
            resolution: Target resolution (width, height).
            bit_depth: Bit depth of the output height map (8 or 16).
            num_threads: Number of threads to use for processing.
            
        Returns:
            Generated height map.
            
        Raises:
            HeightMapServiceError: If generation fails.
        """
        try:
            width, height = resolution
            points = point_cloud.points
            
            if len(points) == 0:
                raise HeightMapServiceError("Cannot generate height map from empty point cloud")
                
            # Extract 2D points and Z values
            points_2d = points[:, :2]
            z_values = points[:, 2]
            bounds = point_cloud.bounds
            
            # Initialize height map
            height_map_data = np.zeros((height, width), dtype=np.float32)
            
            # Calculate coordinates in parallel
            futures = []
            chunk_size = max(1, len(points_2d) // num_threads)
            
            # Pre-calculate ranges to avoid redundant computation in chunks
            x_range = bounds["max_x"] - bounds["min_x"]
            y_range = bounds["max_y"] - bounds["min_y"]
            z_range = bounds["max_z"] - bounds["min_z"]
            
            # Handle zero ranges
            if x_range == 0: x_range = 1.0  # Avoid division by zero
            if y_range == 0: y_range = 1.0
            if z_range == 0: z_range = 1.0
            
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                for i in range(0, len(points_2d), chunk_size):
                    chunk = slice(i, min(i + chunk_size, len(points_2d)))
                    futures.append(
                        executor.submit(
                            self._process_chunk,
                            points_2d[chunk],
                            z_values[chunk],
                            width,
                            height,
                            bounds,
                            (x_range, y_range, z_range)
                        )
                    )
                
                # Combine results
                for future in futures:
                    chunk_coords, chunk_values = future.result()
                    if len(chunk_coords) > 0:
                        # Use maximum projection (highest point wins)
                        np.maximum.at(
                            height_map_data, (chunk_coords[:, 1], chunk_coords[:, 0]), chunk_values
                        )
            
            # Create height map object (data is already normalized Z values from _process_chunk)
            # But we might want to ensure it's in [0, 1] range if _process_chunk does that
            # _process_chunk normalizes Z to [0, 1] based on bounds
            
            return HeightMap(height_map_data, bit_depth)
            
        except Exception as e:
            raise HeightMapServiceError(f"Failed to generate height map from point cloud: {str(e)}")

    def _process_chunk(
        self, 
        points: np.ndarray, 
        z_values: np.ndarray, 
        width: int, 
        height: int, 
        bounds: Dict[str, float],
        ranges: Tuple[float, float, float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process a chunk of points for height map generation.
        
        Args:
            points: 2D points
            z_values: Z values
            width: Width of the height map
            height: Height of the height map
            bounds: Dictionary of bounds
            ranges: Tuple of (x_range, y_range, z_range)
            
        Returns:
            Tuple of coordinates and values
        """
        # Calculate pixel coordinates
        x_range, y_range, z_range = ranges
        
        # Check if ranges are effectively zero (handled by caller but double check for safety)
        if x_range <= 1e-9:
            x_coords = np.zeros(len(points), dtype=int)
        else:
            x_coords = (
                (points[:, 0] - bounds["min_x"])
                / x_range
                * (width - 1)
            ).astype(int)
        
        if y_range <= 1e-9:
            y_coords = np.zeros(len(points), dtype=int)
        else:
            y_coords = (
                (1.0 - (points[:, 1] - bounds["min_y"])
                / y_range)
                * (height - 1)
            ).astype(int)
        
        # Clip coordinates to be safe
        x_coords = np.clip(x_coords, 0, width - 1)
        y_coords = np.clip(y_coords, 0, height - 1)
        
        # Normalize Z values
        if z_range <= 1e-9:
            z_normalized = np.zeros(len(z_values), dtype=np.float32)
        else:
            z_normalized = (z_values - bounds["min_z"]) / z_range
        
        return np.column_stack((x_coords, y_coords)), z_normalized
    
    def resize_height_map(self, height_map: HeightMap, width: int, height: int) -> HeightMap:
        """
        Resize the height map.
        
        Args:
            height_map: The height map to resize.
            width: The new width.
            height: The new height.
            
        Returns:
            A new resized height map.
            
        Raises:
            HeightMapServiceError: If the height map cannot be resized.
        """
        try:
            return height_map.resize((width, height))
        except Exception as e:
            raise HeightMapServiceError(f"Failed to resize height map: {str(e)}")
    
    def crop_height_map(self, height_map: HeightMap, x_min: int, y_min: int, width: int, height: int) -> HeightMap:
        """
        Crop the height map.
        
        Args:
            height_map: The height map to crop.
            x_min: The minimum x-coordinate.
            y_min: The minimum y-coordinate.
            width: The width of the crop.
            height: The height of the crop.
            
        Returns:
            A new cropped height map.
            
        Raises:
            HeightMapServiceError: If the height map cannot be cropped.
        """
        try:
            start = (x_min, y_min)
            end = (x_min + width, y_min + height)
            return height_map.crop(start, end)
        except Exception as e:
            raise HeightMapServiceError(f"Failed to crop height map: {str(e)}")
    
    def convert_height_map_to_mesh(self, height_map: HeightMap) -> Mesh:
        """
        Convert a height map to a mesh.
        
        Args:
            height_map: The height map to convert.
            
        Returns:
            A Mesh object.
            
        Raises:
            HeightMapServiceError: If the height map cannot be converted.
        """
        try:
            return height_map.to_mesh()
        except Exception as e:
            raise HeightMapServiceError(f"Failed to convert height map to mesh: {str(e)}")
    
    def convert_height_map_to_point_cloud(self, height_map: HeightMap) -> PointCloud:
        """
        Convert a height map to a point cloud.
        
        Args:
            height_map: The height map to convert.
            
        Returns:
            A PointCloud object.
            
        Raises:
            HeightMapServiceError: If the height map cannot be converted.
        """
        try:
            return height_map.to_point_cloud()
        except Exception as e:
            raise HeightMapServiceError(f"Failed to convert height map to point cloud: {str(e)}")
    
    def get_height_map_info(self, height_map: HeightMap) -> Dict:
        """
        Get information about the height map.
        
        Args:
            height_map: The height map to get information for.
            
        Returns:
            A dictionary with height map information.
            
        Raises:
            HeightMapServiceError: If the information cannot be retrieved.
        """
        try:
            return height_map.to_dict()
        except Exception as e:
            raise HeightMapServiceError(f"Failed to get height map information: {str(e)}")
    
    def apply_gaussian_blur(self, height_map: HeightMap, sigma: float) -> HeightMap:
        """
        Apply a Gaussian blur to the height map.
        
        Args:
            height_map: The height map to blur.
            sigma: The standard deviation of the Gaussian kernel.
            
        Returns:
            A new blurred height map.
            
        Raises:
            HeightMapServiceError: If the blur cannot be applied.
        """
        if sigma <= 0:
            raise HeightMapServiceError(f"Sigma must be positive, got {sigma}")
        
        try:
            # Apply Gaussian blur to the data
            blurred_data = ndimage.gaussian_filter(height_map.data, sigma=sigma)
            
            # Create a new height map with the blurred data
            return HeightMap(blurred_data, height_map.bit_depth)
        except Exception as e:
            raise HeightMapServiceError(f"Failed to apply Gaussian blur: {str(e)}")
    
    def apply_median_filter(self, height_map: HeightMap, kernel_size: int) -> HeightMap:
        """
        Apply a median filter to the height map.
        
        Args:
            height_map: The height map to filter.
            kernel_size: The size of the kernel. Must be odd.
            
        Returns:
            A new filtered height map.
            
        Raises:
            HeightMapServiceError: If the filter cannot be applied.
        """
        if kernel_size <= 0:
            raise HeightMapServiceError(f"Kernel size must be positive, got {kernel_size}")
        
        if kernel_size % 2 == 0:
            raise HeightMapServiceError(f"Kernel size must be odd, got {kernel_size}")
        
        try:
            # Apply median filter to the data
            filtered_data = ndimage.median_filter(height_map.data, size=kernel_size)
            
            # Create a new height map with the filtered data
            return HeightMap(filtered_data, height_map.bit_depth)
        except Exception as e:
            raise HeightMapServiceError(f"Failed to apply median filter: {str(e)}")
    
    def equalize_histogram(self, height_map: HeightMap) -> HeightMap:
        """
        Equalize the histogram of the height map.
        
        Args:
            height_map: The height map to equalize.
            
        Returns:
            A new equalized height map.
            
        Raises:
            HeightMapServiceError: If the histogram cannot be equalized.
        """
        try:
            # Get the data and its range
            data = height_map.data
            min_val = np.min(data)
            max_val = np.max(data)
            
            if max_val == min_val:
                # If all values are the same, just return a copy
                return HeightMap(data.copy(), height_map.bit_depth)
            
            # Normalize to [0, 1] for processing
            norm_data = (data - min_val) / (max_val - min_val)
            
            # Calculate the histogram with high precision (65536 bins)
            # This preserves detail for 16-bit height maps
            hist, bins = np.histogram(norm_data.flatten(), 65536, [0.0, 1.0])
            
            # Calculate the cumulative distribution function
            cdf = hist.cumsum()
            
            # Normalize the CDF to [0, 1]
            cdf_normalized = cdf * 1.0 / cdf[-1]
            
            # Apply the equalization using linear interpolation
            # We use the upper edges of bins for mapping
            equalized_data = np.interp(norm_data.flatten(), bins[1:], cdf_normalized)
            
            # Reshape back to original shape
            equalized_data = equalized_data.reshape(data.shape)
            
            # Scale back to original range
            equalized_data = (equalized_data / equalized_data.max()) * (max_val - min_val) + min_val
            
            # Create a new height map with the equalized data
            return HeightMap(equalized_data.astype(np.float32), height_map.bit_depth)
        except Exception as e:
            raise HeightMapServiceError(f"Failed to equalize histogram: {str(e)}") 