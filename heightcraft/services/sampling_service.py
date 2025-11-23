"""
Sampling service for Heightcraft.

This module provides the SamplingService class for sampling points from meshes.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple, Union

import numpy as np

from heightcraft.core.config import SamplingConfig
from heightcraft.core.exceptions import SamplingError
from heightcraft.domain.mesh import Mesh
from heightcraft.domain.point_cloud import PointCloud
from heightcraft.infrastructure.gpu_manager import gpu_manager
from heightcraft.infrastructure.profiler import profiler


class SamplingService:
    """
    Service for sampling points from meshes.
    
    This class provides methods for sampling points from meshes using
    different strategies (uniform, weighted, etc.) and hardware (CPU, GPU).
    """
    
    def __init__(self, config: SamplingConfig):
        """
        Initialize the sampling service.
        
        Args:
            config: Sampling configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @profiler.profile()
    def sample_points(self, mesh: Mesh, num_samples: int, use_gpu: bool, num_threads: int = 1) -> PointCloud:
        """
        Sample points from a mesh with explicit parameters.
        
        Args:
            mesh: The mesh to sample from
            num_samples: Number of points to sample
            use_gpu: Whether to use GPU
            num_threads: Number of threads for CPU sampling
            
        Returns:
            PointCloud containing sampled points
            
        Raises:
            SamplingError: If sampling fails
        """
        try:
            if use_gpu and gpu_manager.has_gpu:
                points = self._sample_points_gpu(mesh, num_samples)
            elif num_threads > 1:
                points = self.sample_with_threads(mesh, num_samples, num_threads)
            else:
                points = self._sample_points_cpu(mesh, num_samples)
                
            return PointCloud(points)
        except Exception as e:
            raise SamplingError(f"Failed to sample points: {str(e)}")

    @profiler.profile()
    def sample_from_mesh(self, mesh: Mesh) -> np.ndarray:
        """
        Sample points from a mesh.
        
        This method delegates to the appropriate sampling method based on
        the configuration (GPU or CPU).
        
        Args:
            mesh: The mesh to sample from
            
        Returns:
            Sampled points as a numpy array
            
        Raises:
            SamplingError: If point sampling fails
        """
        try:
            num_samples = self.config.num_samples
            
            if self.config.use_gpu and gpu_manager.has_gpu:
                self.logger.info(f"Using GPU for sampling {num_samples} points")
                return self._sample_points_gpu(mesh, num_samples)
            else:
                self.logger.info(f"Using CPU for sampling {num_samples} points")
                return self._sample_points_cpu(mesh, num_samples)
        except Exception as e:
            raise SamplingError(f"Failed to sample points from mesh: {str(e)}")
    
    @profiler.profile()
    def _sample_points_cpu(self, mesh: Mesh, num_samples: int) -> np.ndarray:
        """
        Sample points using CPU.
        
        Args:
            mesh: The mesh to sample from
            num_samples: Number of points to sample
            
        Returns:
            Sampled points as a numpy array
            
        Raises:
            SamplingError: If point sampling fails
        """
        try:
            # Get the underlying trimesh object
            trimesh_obj = mesh.mesh
            
            # Validate mesh
            if trimesh_obj.area <= 0:
                raise SamplingError("Mesh has zero surface area, cannot sample points")
            
            if not trimesh_obj.is_watertight:
                self.logger.warning("Mesh is not watertight, sampling may be unreliable")
            
            if hasattr(trimesh_obj, 'is_degenerate') and trimesh_obj.is_degenerate.any():
                self.logger.warning("Mesh has degenerate faces, sampling may be unreliable")
            
            # Log mesh info
            self.logger.info(f"Sampling {num_samples} points from mesh with {len(trimesh_obj.vertices)} vertices and {len(trimesh_obj.faces)} faces")
            self.logger.info(f"Mesh surface area: {trimesh_obj.area}")
            
            # Sample points uniformly from the mesh surface
            points = trimesh_obj.sample(num_samples)
            
            if points is None or len(points) == 0:
                raise SamplingError("No points were sampled from the mesh")
            
            if len(points) != num_samples:
                self.logger.warning(f"Requested {num_samples} points but got {len(points)} points")
            
            return points
        except Exception as e:
            raise SamplingError(f"Failed to sample points on CPU: {str(e)}")
    
    @profiler.profile()
    def _sample_points_gpu(self, mesh: Mesh, num_samples: int) -> np.ndarray:
        """
        Sample points using GPU.
        
        Args:
            mesh: The mesh to sample from
            num_samples: Number of points to sample
            
        Returns:
            Sampled points as a numpy array
            
        Raises:
            SamplingError: If point sampling fails
        """
        try:
            # Check if torch is available
            try:
                import torch
            except ImportError:
                self.logger.warning("PyTorch not available, falling back to CPU sampling")
                return self._sample_points_cpu(mesh, num_samples)
            
            self.logger.info(f"Sampling {num_samples} points using GPU")
            
            # Get mesh data
            vertices = mesh.vertices
            faces = mesh.faces
            
            # Move data to GPU
            vertices_gpu = torch.tensor(vertices, dtype=torch.float32, device="cuda")
            faces_gpu = torch.tensor(faces, dtype=torch.int64, device="cuda")
            
            # Get vertices for each face
            v0 = vertices_gpu[faces_gpu[:, 0]]
            v1 = vertices_gpu[faces_gpu[:, 1]]
            v2 = vertices_gpu[faces_gpu[:, 2]]
            
            # Calculate face areas
            face_areas = 0.5 * torch.norm(torch.cross(v1 - v0, v2 - v0, dim=1), dim=1)
            
            # Handle case where some face areas might be close to zero
            epsilon = 1e-10
            face_areas = torch.clamp(face_areas, min=epsilon)
            
            # Calculate face probabilities based on area
            face_probs = face_areas / torch.sum(face_areas)
            
            # Sample faces based on area
            face_indices = torch.multinomial(face_probs, num_samples, replacement=True)
            
            # Sample points within each face
            r1 = torch.sqrt(torch.rand(num_samples, device="cuda"))
            r2 = torch.rand(num_samples, device="cuda")
            
            # Barycentric coordinates
            a = 1.0 - r1
            b = r1 * (1.0 - r2)
            c = r1 * r2
            
            # Get sampled faces
            sampled_v0 = v0[face_indices]
            sampled_v1 = v1[face_indices]
            sampled_v2 = v2[face_indices]
            
            # Calculate points using barycentric coordinates
            points = (
                a.unsqueeze(1) * sampled_v0 +
                b.unsqueeze(1) * sampled_v1 +
                c.unsqueeze(1) * sampled_v2
            )
            
            # Move back to CPU
            points_cpu = points.cpu().numpy()
            
            return points_cpu
        except Exception as e:
            self.logger.warning(f"GPU sampling failed: {str(e)}, falling back to CPU")
            return self._sample_points_cpu(mesh, num_samples)
    
    @profiler.profile()
    def sample_with_threads(self, mesh: Mesh, num_samples: int, num_threads: int) -> np.ndarray:
        """
        Sample points using multiple CPU threads.
        
        Args:
            mesh: The mesh to sample from
            num_samples: Number of points to sample
            num_threads: Number of threads to use
            
        Returns:
            Sampled points as a numpy array
            
        Raises:
            SamplingError: If point sampling fails
        """
        try:
            self.logger.info(f"Sampling {num_samples} points using {num_threads} threads")
            
            # Validate inputs
            if num_threads <= 0:
                raise SamplingError(f"Invalid number of threads: {num_threads}")
            if num_samples <= 0:
                raise SamplingError(f"Invalid number of samples: {num_samples}")
                
            # Validate mesh
            trimesh_obj = mesh.mesh
            if trimesh_obj.area <= 0:
                raise SamplingError("Mesh has zero surface area, cannot sample points")
            
            # Calculate samples per thread
            samples_per_thread = max(1, num_samples // num_threads)
            remaining_samples = num_samples % num_threads
            
            self.logger.info(f"Sampling {samples_per_thread} points per thread with {remaining_samples} remaining")
            
            # Initialize thread pool with a timeout
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                # Submit sampling tasks
                futures = []
                for i in range(num_threads):
                    thread_samples = samples_per_thread + (1 if i < remaining_samples else 0)
                    if thread_samples > 0:
                        futures.append(executor.submit(
                            self._sample_points_cpu, mesh, thread_samples
                        ))
                
                # Collect results with timeout
                all_points = []
                timeout_per_future = 30  # seconds
                
                for future in as_completed(futures, timeout=timeout_per_future * len(futures)):
                    try:
                        points = future.result(timeout=timeout_per_future)
                        if points is not None and len(points) > 0:
                            all_points.append(points)
                        else:
                            self.logger.warning("Thread returned no points")
                    except TimeoutError:
                        raise SamplingError("Point sampling timed out")
                    except Exception as e:
                        raise SamplingError(f"Thread failed to sample points: {str(e)}")
            
            # Check if we got any points
            if not all_points:
                raise SamplingError("No points were sampled from any thread")
            
            # Combine results
            combined_points = np.vstack(all_points)
            
            # Log results
            self.logger.info(f"Successfully sampled {len(combined_points)} points")
            
            return combined_points
        except Exception as e:
            raise SamplingError(f"Failed to sample points with threads: {str(e)}")
    
    @profiler.profile()
    def subsample_point_cloud(self, point_cloud: PointCloud, num_samples: int) -> PointCloud:
        """
        Subsample a point cloud.
        
        Args:
            point_cloud: The point cloud to subsample
            num_samples: Number of points to include in the subsample
            
        Returns:
            Subsampled point cloud
            
        Raises:
            SamplingError: If subsampling fails
        """
        try:
            if num_samples >= point_cloud.size:
                return point_cloud
            
            self.logger.info(f"Subsampling point cloud from {point_cloud.size} to {num_samples} points")
            
            return point_cloud.subsample(num_samples)
            
        except Exception as e:
            raise SamplingError(f"Failed to subsample point cloud: {e}")
    
    @profiler.profile()
    def merge_point_clouds(self, point_clouds: list[PointCloud]) -> PointCloud:
        """
        Merge multiple point clouds.
        
        Args:
            point_clouds: List of point clouds to merge
            
        Returns:
            Merged point cloud
            
        Raises:
            SamplingError: If merging fails
        """
        try:
            total_points = sum(pc.size for pc in point_clouds)
            self.logger.info(f"Merging {len(point_clouds)} point clouds with {total_points} total points")
            
            return PointCloud.merge(point_clouds)
            
        except Exception as e:
            raise SamplingError(f"Failed to merge point clouds: {e}") 