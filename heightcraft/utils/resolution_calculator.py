"""
Resolution calculator utility for calculating optimal resolutions.

This module provides functionality for calculating optimal resolutions for
height maps based on mesh dimensions and desired point counts.
"""

import numpy as np
from typing import Optional, Tuple

from heightcraft.core.exceptions import CalculationError
from heightcraft.domain.mesh import Mesh


class ResolutionCalculator:
    """Utility for calculating optimal resolutions."""
    
    def __init__(self) -> None:
        """Initialize the resolution calculator."""
        pass
    
    def calculate_optimal_resolution(self, mesh: Mesh, target_width: int, target_height: int) -> float:
        """
        Calculate the optimal resolution for a height map based on mesh dimensions and target dimensions.
        
        Args:
            mesh: The mesh to calculate the resolution for.
            target_width: The target width of the height map.
            target_height: The target height of the height map.
            
        Returns:
            The optimal resolution.
            
        Raises:
            CalculationError: If the resolution cannot be calculated.
        """
        if target_width <= 0 or target_height <= 0:
            raise CalculationError(f"Target dimensions must be positive, got {target_width}x{target_height}")
        
        try:
            # Get the mesh bounds
            bounds = mesh.bounds
            # bounds is a 2x3 array where bounds[0] is min and bounds[1] is max
            min_x, min_y, _ = bounds[0]
            max_x, max_y, _ = bounds[1]
            width = max_x - min_x
            height = max_y - min_y
            
            # Calculate the aspect ratio of the mesh
            aspect_ratio = width / height if height > 0 else 1.0
            
            # Calculate the aspect ratio of the target
            target_aspect_ratio = target_width / target_height if target_height > 0 else 1.0
            
            # Calculate the resolution based on the aspect ratio
            if aspect_ratio >= target_aspect_ratio:
                # Mesh is wider than the target, so use width to determine resolution
                resolution = width / target_width
            else:
                # Mesh is taller than the target, so use height to determine resolution
                resolution = height / target_height
            
            return resolution
        
        except Exception as e:
            raise CalculationError(f"Failed to calculate optimal resolution: {str(e)}")
    
    def calculate_dimensions_from_resolution(self, mesh: Mesh, resolution: float) -> Tuple[int, int]:
        """
        Calculate the dimensions of a height map based on mesh dimensions and resolution.
        
        Args:
            mesh: The mesh to calculate the dimensions for.
            resolution: The resolution of the height map.
            
        Returns:
            A tuple of (width, height).
            
        Raises:
            CalculationError: If the dimensions cannot be calculated.
        """
        if resolution <= 0:
            raise CalculationError(f"Resolution must be positive, got {resolution}")
        
        try:
            # Get the mesh bounds
            bounds = mesh.bounds
            # bounds is a 2x3 array where bounds[0] is min and bounds[1] is max
            min_x, min_y, _ = bounds[0]
            max_x, max_y, _ = bounds[1]
            width = max_x - min_x
            height = max_y - min_y
            
            # Calculate the number of pixels needed
            num_pixels_x = int(width / resolution)
            num_pixels_y = int(height / resolution)
            
            # Ensure minimum size
            num_pixels_x = max(num_pixels_x, 1)
            num_pixels_y = max(num_pixels_y, 1)
            
            return num_pixels_x, num_pixels_y
        
        except Exception as e:
            raise CalculationError(f"Failed to calculate dimensions from resolution: {str(e)}")
    
    def calculate_resolution_from_point_count(self, mesh: Mesh, target_points: int) -> float:
        """
        Calculate the resolution of a height map based on mesh dimensions and target point count.
        
        Args:
            mesh: The mesh to calculate the resolution for.
            target_points: The target number of points.
            
        Returns:
            The optimal resolution.
            
        Raises:
            CalculationError: If the resolution cannot be calculated.
        """
        if target_points <= 0:
            raise CalculationError(f"Target point count must be positive, got {target_points}")
        
        try:
            # Get the mesh bounds
            bounds = mesh.bounds
            # bounds is a 2x3 array where bounds[0] is min and bounds[1] is max
            min_x, min_y, _ = bounds[0]
            max_x, max_y, _ = bounds[1]
            width = max_x - min_x
            height = max_y - min_y
            
            # Calculate the area of the mesh in the XY plane
            area = width * height
            
            # Calculate the resolution based on the area and target point count
            # Each point will occupy approximately (resolution)^2 area
            # So area / target_points = resolution^2
            resolution = np.sqrt(area / target_points)
            
            return resolution
            
        except Exception as e:
            raise CalculationError(f"Failed to calculate resolution from point count: {str(e)}")
    
    def estimate_point_count(self, mesh: Mesh, resolution: float) -> int:
        """
        Estimate the number of points in a height map based on mesh dimensions and resolution.
        
        Args:
            mesh: The mesh to estimate the point count for.
            resolution: The resolution of the height map.
            
        Returns:
            The estimated number of points.
            
        Raises:
            CalculationError: If the point count cannot be estimated.
        """
        if resolution <= 0:
            raise CalculationError(f"Resolution must be positive, got {resolution}")
        
        try:
            # Get the mesh bounds
            bounds = mesh.bounds
            # bounds is a 2x3 array where bounds[0] is min and bounds[1] is max
            min_x, min_y, _ = bounds[0]
            max_x, max_y, _ = bounds[1]
            width = max_x - min_x
            height = max_y - min_y
            
            # Calculate the number of points
            # Each point occupies approximately (resolution)^2 area
            point_count = int((width / resolution) * (height / resolution))
            
            # Ensure minimum size
            point_count = max(point_count, 1)
            
            return point_count
        
        except Exception as e:
            raise CalculationError(f"Failed to estimate point count: {str(e)}") 