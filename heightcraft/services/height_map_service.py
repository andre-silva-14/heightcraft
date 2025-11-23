"""
Height map service for working with height maps.

This module provides services for loading, saving, and manipulating height maps.
"""

import os
from typing import Dict, Optional, Tuple, Union

import numpy as np
from scipy import ndimage

from heightcraft.core.exceptions import HeightMapServiceError
from heightcraft.domain.height_map import HeightMap
from heightcraft.domain.mesh import Mesh
from heightcraft.domain.point_cloud import PointCloud


class HeightMapService:
    """Service for working with height maps."""
    
    def __init__(self, height_map_repository=None) -> None:
        """
        Initialize the height map service.
        
        Args:
            height_map_repository: Repository for loading and saving height maps.
        """
        self.height_map_repository = height_map_repository
    
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
            # Get the data and normalize it manually
            data = height_map.data
            min_val = np.min(data)
            max_val = np.max(data)
            
            # Avoid division by zero
            if min_val == max_val:
                normalized_data = np.zeros_like(data)
            else:
                normalized_data = (data - min_val) / (max_val - min_val)
            
            # Create a new height map with the normalized data
            return HeightMap(normalized_data, height_map.bit_depth)
        except Exception as e:
            raise HeightMapServiceError(f"Failed to normalize height map: {str(e)}")
    
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