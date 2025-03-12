"""
Point sampler utility for sampling points from meshes.

This module provides functionality for sampling points from meshes using
various algorithms and optimizations.
"""

import numpy as np
import trimesh
from typing import Optional, Union, List

from heightcraft.core.exceptions import SamplingError
from heightcraft.domain.mesh import Mesh
from heightcraft.domain.point_cloud import PointCloud


class PointSampler:
    """Utility for sampling points from meshes."""
    
    def __init__(self) -> None:
        """Initialize the point sampler."""
        pass
    
    def sample_points(
        self, 
        mesh: Mesh, 
        num_points: int, 
        use_gpu: bool = False, 
        num_threads: int = 1,
        seed: Optional[int] = None
    ) -> PointCloud:
        """
        Sample points from a mesh.
        
        Args:
            mesh: The mesh to sample points from.
            num_points: The number of points to sample.
            use_gpu: Whether to use GPU acceleration if available.
            num_threads: Number of threads to use for CPU sampling.
            seed: Random seed for reproducible sampling.
            
        Returns:
            A PointCloud object.
            
        Raises:
            SamplingError: If the points cannot be sampled.
        """
        # Input validation
        if num_points <= 0:
            raise SamplingError(f"Number of points must be positive, got {num_points}")
        
        if num_threads <= 0:
            raise SamplingError(f"Number of threads must be positive, got {num_threads}")
        
        # Set seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        try:
            # Create a trimesh object
            trimesh_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)
            
            # Check if the mesh has any valid faces
            if trimesh_mesh.area <= 0:
                raise SamplingError("Mesh has zero surface area, cannot sample points")
            
            # Sample points
            if use_gpu:
                # Placeholder for GPU sampling
                # This would need implementation with GPU libraries like PyTorch or TensorFlow
                # For now, fall back to CPU sampling
                points = self._sample_points_cpu(trimesh_mesh, num_points, num_threads)
            else:
                # Use CPU sampling
                points = self._sample_points_cpu(trimesh_mesh, num_points, num_threads)
            
            # Create a point cloud
            return PointCloud(points)
        
        except Exception as e:
            if "zero surface area" in str(e).lower() or "zero sampling area" in str(e).lower():
                raise SamplingError(f"Mesh has zero surface area, cannot sample points: {str(e)}")
            raise SamplingError(f"Failed to sample points from mesh: {str(e)}")
    
    def _sample_points_cpu(
        self, 
        trimesh_mesh: trimesh.Trimesh, 
        num_points: int, 
        num_threads: int
    ) -> np.ndarray:
        """
        Sample points from a mesh using CPU.
        
        Args:
            trimesh_mesh: The mesh to sample points from.
            num_points: The number of points to sample.
            num_threads: Number of threads to use for sampling.
            
        Returns:
            A numpy array of sampled points.
            
        Raises:
            SamplingError: If the points cannot be sampled.
        """
        try:
            # Sample points
            points, _ = trimesh.sample.sample_surface(trimesh_mesh, num_points)
            return points
        except Exception as e:
            raise SamplingError(f"Failed to sample points from mesh using CPU: {str(e)}")
    
    def merge_point_clouds(self, point_clouds: List[PointCloud]) -> PointCloud:
        """
        Merge multiple point clouds into one.
        
        Args:
            point_clouds: List of point clouds to merge.
            
        Returns:
            A merged PointCloud object.
            
        Raises:
            SamplingError: If the point clouds cannot be merged.
        """
        if not point_clouds:
            raise SamplingError("No point clouds to merge")
        
        try:
            # Special case for a single point cloud
            if len(point_clouds) == 1:
                return point_clouds[0]
            
            # Merge all point clouds
            all_points = np.vstack([pc.points for pc in point_clouds])
            
            # Create a new point cloud
            return PointCloud(all_points)
        
        except Exception as e:
            raise SamplingError(f"Failed to merge point clouds: {str(e)}")
    
    def subsample_point_cloud(self, point_cloud: PointCloud, num_points: int) -> PointCloud:
        """
        Subsample a point cloud to a specific number of points.
        
        Args:
            point_cloud: The point cloud to subsample.
            num_points: The number of points to sample.
            
        Returns:
            A subsampled PointCloud object.
            
        Raises:
            SamplingError: If the point cloud cannot be subsampled.
        """
        if num_points <= 0:
            raise SamplingError(f"Number of points must be positive, got {num_points}")
        
        if num_points > point_cloud.size:
            raise SamplingError(f"Cannot subsample to {num_points} points, point cloud only has {point_cloud.size} points")
        
        try:
            # Subsample the point cloud
            indices = np.random.choice(point_cloud.size, num_points, replace=False)
            subsampled_points = point_cloud.points[indices]
            
            # Create a new point cloud
            return PointCloud(subsampled_points)
        
        except Exception as e:
            raise SamplingError(f"Failed to subsample point cloud: {str(e)}")

def is_gpu_supported() -> bool:
    """
    Check if GPU sampling is supported.
    
    Returns:
        True if GPU sampling is supported, False otherwise.
    """
    try:
        # Try to import the necessary CUDA modules
        import trimesh.ray.ray_pyembree
        return True
    except ImportError:
        return False 