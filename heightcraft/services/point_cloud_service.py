"""
Point cloud service for Heightcraft.

This module provides the PointCloudService class for handling point cloud operations.
"""

import logging
from typing import List, Optional, Tuple, Union, Dict

import numpy as np

from heightcraft.core.exceptions import SamplingError, PointCloudServiceError
from heightcraft.domain.point_cloud import PointCloud
from heightcraft.infrastructure.profiler import profiler


class PointCloudService:
    """Service class for handling point cloud operations."""

    def __init__(self):
        """Initialize the point cloud service."""
        self.logger = logging.getLogger(__name__)

    @profiler.profile()
    def create_point_cloud(self, points: np.ndarray) -> PointCloud:
        """
        Create a point cloud from numpy array of points.

        Args:
            points: Numpy array of shape (N, 3) containing point coordinates

        Returns:
            PointCloud: A new point cloud instance

        Raises:
            SamplingError: If points array is invalid
        """
        try:
            return PointCloud(points)
        except Exception as e:
            raise PointCloudServiceError(f"Failed to create point cloud: {e}")

    @profiler.profile()
    def filter_points(self, point_cloud: PointCloud, min_height: float = None, 
                     max_height: float = None) -> PointCloud:
        """
        Filter points based on height constraints.

        Args:
            point_cloud: Input point cloud
            min_height: Minimum height threshold (optional)
            max_height: Maximum height threshold (optional)

        Returns:
            PointCloud: Filtered point cloud

        Raises:
            PointCloudServiceError: If filtering fails
        """
        try:
            points = point_cloud.points
            mask = np.ones(len(points), dtype=bool)

            if min_height is not None:
                mask &= points[:, 2] >= min_height
            if max_height is not None:
                mask &= points[:, 2] <= max_height

            return PointCloud(points[mask])
        except Exception as e:
            raise PointCloudServiceError(f"Failed to filter points: {e}")

    @profiler.profile()
    def normalize_heights(self, point_cloud: PointCloud) -> PointCloud:
        """
        Normalize point heights to [0, 1] range.

        Args:
            point_cloud: Input point cloud

        Returns:
            PointCloud: Point cloud with normalized heights

        Raises:
            PointCloudServiceError: If normalization fails
        """
        try:
            normalized_z = point_cloud.normalize_z()
            points = point_cloud.points.copy()
            points[:, 2] = normalized_z
            return PointCloud(points)
        except Exception as e:
            raise PointCloudServiceError(f"Failed to normalize heights: {e}")

    @profiler.profile()
    def compute_bounds(self, point_cloud: PointCloud) -> Dict[str, float]:
        """
        Compute the bounding box of the point cloud.

        Args:
            point_cloud: Input point cloud

        Returns:
            Dict[str, float]: Dictionary with min_x, max_x, min_y, max_y, min_z, max_z

        Raises:
            PointCloudServiceError: If bound computation fails
        """
        try:
            return point_cloud.bounds
        except Exception as e:
            raise PointCloudServiceError(f"Failed to compute bounds: {e}") 