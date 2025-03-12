"""
Point cloud domain model for Heightcraft.

This module provides the PointCloud class which represents a collection
of 3D points sampled from a mesh.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from heightcraft.core.exceptions import SamplingError


class PointCloud:
    """
    Domain model representing a collection of 3D points.
    
    This class represents a point cloud, which is a collection of points
    in 3D space, typically sampled from a mesh surface.
    """
    
    def __init__(self, points: np.ndarray):
        """
        Initialize the point cloud.
        
        Args:
            points: The points as a numpy array of shape (N, 3)
            
        Raises:
            SamplingError: If the points array has an invalid shape
        """
        self._validate_points(points)
        self._points = points
        self._logger = logging.getLogger(self.__class__.__name__)
        
        # Cache for properties
        self._bounds = None
    
    @property
    def points(self) -> np.ndarray:
        """Get the points."""
        return self._points
    
    @property
    def size(self) -> int:
        """Get the number of points."""
        return len(self._points)
    
    @property
    def x(self) -> np.ndarray:
        """Get the X coordinates."""
        return self._points[:, 0]
    
    @property
    def y(self) -> np.ndarray:
        """Get the Y coordinates."""
        return self._points[:, 1]
    
    @property
    def z(self) -> np.ndarray:
        """Get the Z coordinates."""
        return self._points[:, 2]
    
    @property
    def bounds(self) -> Dict[str, float]:
        """
        Get the bounds of the point cloud.
        
        Returns:
            Dictionary with min_x, max_x, min_y, max_y, min_z, max_z
        """
        if self._bounds is None:
            self._bounds = {
                "min_x": np.min(self.x),
                "max_x": np.max(self.x),
                "min_y": np.min(self.y),
                "max_y": np.max(self.y),
                "min_z": np.min(self.z),
                "max_z": np.max(self.z),
            }
        return self._bounds
    
    def get_xy_points(self) -> np.ndarray:
        """
        Get the 2D points (X, Y coordinates).
        
        Returns:
            The points as a numpy array of shape (N, 2)
        """
        return self._points[:, :2]
    
    def normalize_z(self) -> np.ndarray:
        """
        Normalize the Z coordinates to the range [0, 1].
        
        Returns:
            Normalized Z coordinates
        """
        min_z = self.bounds["min_z"]
        max_z = self.bounds["max_z"]
        z_range = max_z - min_z
        
        if z_range == 0:
            return np.zeros_like(self.z)
        
        return (self.z - min_z) / z_range
    
    def get_aspect_ratio(self) -> float:
        """
        Calculate the aspect ratio of the point cloud (X/Y).
        
        Returns:
            Aspect ratio
        """
        x_range = self.bounds["max_x"] - self.bounds["min_x"]
        y_range = self.bounds["max_y"] - self.bounds["min_y"]
        
        return x_range / y_range if y_range != 0 else 1.0
    
    def subsample(self, count: int) -> "PointCloud":
        """
        Create a subsampled version of this point cloud.
        
        Args:
            count: Number of points to include in the subsample
            
        Returns:
            New PointCloud with the subsampled points
            
        Raises:
            SamplingError: If count is greater than the number of points
        """
        if count > self.size:
            raise SamplingError(f"Cannot subsample {count} points from a point cloud with {self.size} points")
        
        # Random sampling without replacement
        indices = np.random.choice(self.size, count, replace=False)
        return PointCloud(self._points[indices])
    
    def to_dict(self) -> Dict:
        """
        Convert the point cloud to a dictionary.
        
        Returns:
            Dictionary representation of the point cloud
        """
        return {
            "size": self.size,
            "bounds": self.bounds,
        }
    
    @staticmethod
    def _validate_points(points: np.ndarray) -> None:
        """
        Validate the points array.
        
        Args:
            points: The points to validate
            
        Raises:
            SamplingError: If the points array has an invalid shape
        """
        if not isinstance(points, np.ndarray):
            raise SamplingError(f"Expected numpy array, got {type(points)}")
        
        if len(points.shape) != 2:
            raise SamplingError(f"Expected 2D array, got {len(points.shape)}D")
        
        if points.shape[1] != 3:
            raise SamplingError(f"Expected points with 3 coordinates, got {points.shape[1]}")
        
        if len(points) == 0:
            raise SamplingError("Point cloud cannot be empty")
    
    @classmethod
    def merge(cls, point_clouds: List["PointCloud"]) -> "PointCloud":
        """
        Merge multiple point clouds.
        
        Args:
            point_clouds: List of point clouds to merge
            
        Returns:
            Merged point cloud
            
        Raises:
            SamplingError: If the list of point clouds is empty
        """
        if not point_clouds:
            raise SamplingError("Cannot merge an empty list of point clouds")
        
        # Stack all points
        all_points = np.vstack([pc.points for pc in point_clouds])
        return cls(all_points) 