"""
LiDAR repository for loading LiDAR data.

This module provides functionality for loading LiDAR data from .las and .laz files.
"""

import logging
import os
from typing import Optional

import numpy as np

from heightcraft.core.exceptions import RepositoryError
from heightcraft.domain.point_cloud import PointCloud

try:
    import laspy
except ImportError:
    laspy = None


class LidarRepository:
    """Repository for loading LiDAR data."""
    
    def __init__(self) -> None:
        """Initialize the LiDAR repository."""
        self.logger = logging.getLogger(self.__class__.__name__)
        
        if laspy is None:
            self.logger.warning("laspy is not installed. LiDAR support will be disabled.")
    
    def get_chunk_iterator(self, file_path: str, chunk_size: int):
        """
        Get an iterator over chunks of points from a LiDAR file.
        
        Args:
            file_path: Path to the LiDAR file.
            chunk_size: Number of points per chunk.
            
        Yields:
            PointCloud object for each chunk.
            
        Raises:
            RepositoryError: If the file cannot be loaded.
        """
        if laspy is None:
            raise RepositoryError("laspy is not installed. Please install 'laspy[lazrs]' to use LiDAR features.")
            
        if not os.path.exists(file_path):
            raise RepositoryError(f"File not found: {file_path}")
            
        try:
            self.logger.info(f"Opening LiDAR file for streaming: {file_path} (chunk size: {chunk_size})")
            
            with laspy.open(file_path) as fh:
                # Iterate over chunks
                for chunk in fh.chunk_iterator(chunk_size):
                    # Extract X, Y, Z coordinates
                    x = chunk.x
                    y = chunk.y
                    z = chunk.z
                    
                    # Stack into (N, 3) array
                    points = np.column_stack((x, y, z))
                    
                    yield PointCloud(points)
                    
        except Exception as e:
            raise RepositoryError(f"Failed to stream LiDAR file from {file_path}: {str(e)}")

    def get_bounds(self, file_path: str) -> dict:
        """
        Get the bounds of the LiDAR file without loading all points.
        
        Args:
            file_path: Path to the LiDAR file.
            
        Returns:
            Dictionary with bounds.
        """
        if laspy is None:
            raise RepositoryError("laspy is not installed.")
            
        if not os.path.exists(file_path):
            raise RepositoryError(f"File not found: {file_path}")
            
        try:
            with laspy.open(file_path) as fh:
                header = fh.header
                return {
                    "min_x": header.x_min,
                    "max_x": header.x_max,
                    "min_y": header.y_min,
                    "max_y": header.y_max,
                    "min_z": header.z_min,
                    "max_z": header.z_max,
                }
        except Exception as e:
            raise RepositoryError(f"Failed to get bounds from {file_path}: {str(e)}")

    def load(self, file_path: str) -> PointCloud:
        """
        Load a point cloud from a LiDAR file.
        
        Args:
            file_path: Path to the LiDAR file (.las or .laz).
            
        Returns:
            A PointCloud object.
            
        Raises:
            RepositoryError: If the file cannot be loaded or is invalid.
        """
        if laspy is None:
            raise RepositoryError("laspy is not installed. Please install 'laspy[lazrs]' to use LiDAR features.")
            
        if not os.path.exists(file_path):
            raise RepositoryError(f"File not found: {file_path}")
        
        try:
            self.logger.info(f"Loading LiDAR file: {file_path}")
            
            with laspy.open(file_path) as fh:
                las = fh.read()
                
                # Extract X, Y, Z coordinates
                # laspy provides scaled dimensions by default when accessing x, y, z
                x = las.x
                y = las.y
                z = las.z
                
                # Stack into (N, 3) array
                points = np.column_stack((x, y, z))
                
                self.logger.info(f"Loaded {len(points)} points from {file_path}")
                
                return PointCloud(points)
                
        except Exception as e:
            raise RepositoryError(f"Failed to load LiDAR file from {file_path}: {str(e)}")
