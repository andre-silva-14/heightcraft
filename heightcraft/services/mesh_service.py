"""
Mesh service for working with 3D meshes.

This module provides services for manipulating, analyzing, and validating 3D meshes.
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import os
import tempfile
import trimesh

from heightcraft.core.exceptions import MeshServiceError, MeshValidationError
from heightcraft.domain.mesh import Mesh
from heightcraft.domain.point_cloud import PointCloud
from heightcraft.domain.height_map import HeightMap
from heightcraft.infrastructure.profiler import profiler
from heightcraft.utils.validators import validate_mesh
from heightcraft.infrastructure.model_repository import ModelRepository
from heightcraft.core.config import SamplingConfig
from heightcraft.core.config import ModelConfig
from heightcraft.services.sampling_service import SamplingService


class MeshService:
    """Service for working with 3D meshes."""
    
    def __init__(self) -> None:
        """Initialize the mesh service."""
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @profiler.profile()
    def prepare_mesh(self, mesh: Mesh, center: bool = True, align: bool = True) -> Mesh:
        """
        Prepare a mesh for processing by centering and/or aligning it.
        
        Args:
            mesh: The mesh to prepare
            center: Whether to center the mesh
            align: Whether to align the mesh to the XY plane
            
        Returns:
            Prepared mesh
            
        Raises:
            MeshServiceError: If the mesh cannot be prepared
        """
        try:
            if center:
                self.logger.info("Centering mesh")
                mesh.center()
            
            if align:
                self.logger.info("Aligning mesh to XY plane")
                mesh.align_to_xy()
            
            return mesh
        except Exception as e:
            raise MeshServiceError(f"Failed to prepare mesh: {str(e)}")
    
    @profiler.profile()
    def center_mesh(self, mesh: Mesh) -> Mesh:
        """
        Center a mesh at the origin.
        
        Args:
            mesh: The mesh to center
            
        Returns:
            Centered mesh
            
        Raises:
            MeshServiceError: If the mesh cannot be centered
        """
        try:
            self.logger.info("Centering mesh")
            mesh.center()
            return mesh
        except Exception as e:
            raise MeshServiceError(f"Failed to center mesh: {str(e)}")
    
    @profiler.profile()
    def align_mesh_to_xy(self, mesh: Mesh) -> Mesh:
        """
        Align a mesh to the XY plane.
        
        Args:
            mesh: The mesh to align
            
        Returns:
            Aligned mesh
            
        Raises:
            MeshServiceError: If the mesh cannot be aligned
        """
        try:
            self.logger.info("Aligning mesh to XY plane")
            mesh.align_to_xy()
            return mesh
        except Exception as e:
            raise MeshServiceError(f"Failed to align mesh: {str(e)}")
    
    @profiler.profile()
    def validate_mesh(self, mesh: Mesh, strict: bool = False) -> None:
        """
        Validate a mesh for processing.
        
        Args:
            mesh: The mesh to validate
            strict: Whether to perform strict validation
            
        Raises:
            MeshValidationError: If the mesh fails validation
        """
        # Use centralized validation from utils.validators
        validate_mesh(mesh, strict)
    
    @profiler.profile()
    def get_bounds(self, mesh: Mesh) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the bounds of a mesh.
        
        Args:
            mesh: The mesh
            
        Returns:
            Tuple of (min_coords, max_coords)
            
        Raises:
            MeshServiceError: If the bounds cannot be calculated
        """
        try:
            # Get the bounds from the mesh
            return mesh.bounds
        except Exception as e:
            raise MeshServiceError(f"Failed to get mesh bounds: {str(e)}")
    
    @profiler.profile()
    def get_aspect_ratio(self, mesh: Mesh) -> float:
        """
        Get the aspect ratio of a mesh (X/Y).
        
        Args:
            mesh: The mesh
            
        Returns:
            Aspect ratio
            
        Raises:
            MeshServiceError: If the aspect ratio cannot be calculated
        """
        try:
            return mesh.get_aspect_ratio()
        except Exception as e:
            raise MeshServiceError(f"Failed to calculate mesh aspect ratio: {str(e)}")
    
    @profiler.profile()
    def calculate_target_resolution(
        self, 
        mesh: Mesh, 
        max_resolution: int
    ) -> Tuple[int, int]:
        """
        Calculate the target resolution for a height map based on mesh proportions.
        
        Args:
            mesh: The mesh
            max_resolution: Maximum resolution
            
        Returns:
            Tuple of (width, height)
            
        Raises:
            MeshServiceError: If the resolution cannot be calculated
        """
        try:
            # Get aspect ratio
            aspect_ratio = self.get_aspect_ratio(mesh)
            
            # Calculate resolution
            if aspect_ratio >= 1.0:
                # Wider than tall
                width = max_resolution
                height = int(max_resolution / aspect_ratio)
            else:
                # Taller than wide
                height = max_resolution
                width = int(max_resolution * aspect_ratio)
            
            # Ensure minimum size
            width = max(width, 32)
            height = max(height, 32)
            
            self.logger.info(f"Calculated target resolution: {width}x{height}")
            
            return width, height
        except Exception as e:
            raise MeshServiceError(f"Failed to calculate target resolution: {str(e)}")
    
    @profiler.profile()
    def combine_meshes(self, meshes: list[Mesh]) -> Mesh:
        """
        Combine multiple meshes into one.
        
        Args:
            meshes: List of meshes to combine
            
        Returns:
            Combined mesh
            
        Raises:
            MeshServiceError: If the meshes cannot be combined
        """
        if not meshes:
            raise MeshServiceError("No meshes provided")
        
        if len(meshes) == 1:
            return meshes[0]
        
        try:
            import trimesh
            # Extract the underlying trimesh objects and concatenate them
            combined = trimesh.util.concatenate([m.mesh for m in meshes])
            return Mesh(combined)
        except Exception as e:
            raise MeshServiceError(f"Failed to combine meshes: {str(e)}")
    
    @profiler.profile()
    def mesh_to_point_cloud(self, mesh: Mesh, num_points: int) -> PointCloud:
        """
        Convert a mesh to a point cloud.

        Args:
            mesh: The mesh to convert
            num_points: Number of points to sample from the mesh

        Returns:
            A point cloud representing the mesh

        Raises:
            MeshServiceError: If the mesh cannot be converted
        """
        try:
            # Create sampling configuration
            sampling_config = SamplingConfig(
                num_samples=num_points,
                use_gpu=False,
                num_threads=4
            )
            
            # Initialize the sampling service
            sampling_service = SamplingService(sampling_config)
            
            # Sample points from the mesh
            sampled_points = sampling_service.sample_from_mesh(mesh)
            
            # Create a point cloud from the sampled points
            point_cloud = PointCloud(sampled_points)
            
            return point_cloud
        except Exception as e:
            raise MeshServiceError(f"Failed to convert mesh to point cloud: {str(e)}")
    
    @profiler.profile()
    def subdivide_mesh(self, mesh: Mesh, max_edge_length: float) -> Mesh:
        """
        Subdivide a mesh to ensure no edge is longer than max_edge_length.
        
        Args:
            mesh: The mesh to subdivide
            max_edge_length: Maximum allowed edge length
            
        Returns:
            Subdivided mesh
            
        Raises:
            MeshServiceError: If the subdivision fails
        """
        try:
            # Get the trimesh object
            tm = mesh.mesh
            
            # Get current edge lengths
            edges = tm.edges_unique
            edge_lengths = tm.edges_unique_length
            max_current = edge_lengths.max()
            
            if max_current <= max_edge_length:
                self.logger.info("Mesh does not need subdivision")
                return mesh
            
            # Calculate subdivision iterations needed
            import math
            iterations = math.ceil(math.log2(max_current / max_edge_length))
            iterations = min(iterations, 5)  # Limit to avoid excessive subdivisions
            
            self.logger.info(f"Subdividing mesh with {iterations} iterations")
            
            # Subdivide
            subdivided = tm.subdivide(iterations)
            
            return Mesh(subdivided)
        except Exception as e:
            raise MeshServiceError(f"Failed to subdivide mesh: {str(e)}")
    
    @profiler.profile()
    def save_mesh(self, mesh: Mesh, file_path: str) -> bool:
        """
        Save a mesh to a file.

        Args:
            mesh: The mesh to save
            file_path: Path to save the mesh to

        Returns:
            True if the mesh was saved successfully

        Raises:
            MeshServiceError: If the mesh cannot be saved
        """
        try:
            # Create a model repository
            repository = ModelRepository()
            
            # Save the mesh
            return repository.save(mesh, file_path)
        except Exception as e:
            raise MeshServiceError(f"Failed to save mesh to {file_path}: {str(e)}")
    
    @profiler.profile()
    def load_mesh(self, file_path: str) -> Mesh:
        """
        Load a mesh from a file.

        Args:
            file_path: Path to the mesh file

        Returns:
            The loaded mesh

        Raises:
            MeshServiceError: If the mesh cannot be loaded
        """
        try:
            # Create a model repository
            repository = ModelRepository()
            
            # Load the mesh
            return repository.load(file_path)
        except Exception as e:
            raise MeshServiceError(f"Failed to load mesh from {file_path}: {str(e)}")
    
    @profiler.profile()
    def convert_mesh_to_height_map(self, mesh: Mesh, resolution: float) -> HeightMap:
        """
        Convert a mesh to a height map.
        
        Args:
            mesh: The mesh to convert
            resolution: Resolution of the height map
            
        Returns:
            A height map representing the mesh
            
        Raises:
            MeshServiceError: If the mesh cannot be converted
        """
        try:
            # Make sure the mesh is centered and aligned to the XY plane
            self.prepare_mesh(mesh)
            
            # Calculate the bounds of the mesh
            bounds = self.get_bounds(mesh)
            min_bound, max_bound = bounds
            
            # Calculate the dimensions of the height map
            width = int((max_bound[0] - min_bound[0]) / resolution) + 1
            height = int((max_bound[1] - min_bound[1]) / resolution) + 1
            
            # Create a height map
            height_map_data = np.zeros((height, width), dtype=np.float32)
            
            # Project the mesh vertices onto the XY plane
            vertices = mesh.vertices
            faces = mesh.faces
            
            # For each face, find the corresponding height map pixels and set their values
            for face in faces:
                v1, v2, v3 = vertices[face]
                
                # Convert vertex positions to pixel coordinates
                x1, y1 = int((v1[0] - min_bound[0]) / resolution), int((v1[1] - min_bound[1]) / resolution)
                x2, y2 = int((v2[0] - min_bound[0]) / resolution), int((v2[1] - min_bound[1]) / resolution)
                x3, y3 = int((v3[0] - min_bound[0]) / resolution), int((v3[1] - min_bound[1]) / resolution)
                
                # Set the height values
                height_map_data[y1, x1] = max(height_map_data[y1, x1], v1[2])
                height_map_data[y2, x2] = max(height_map_data[y2, x2], v2[2])
                height_map_data[y3, x3] = max(height_map_data[y3, x3], v3[2])
            
            # Create a height map object
            height_map = HeightMap(height_map_data, 16)  # Use 16-bit precision instead of 32-bit
            
            return height_map
        except Exception as e:
            raise MeshServiceError(f"Failed to convert mesh to height map: {str(e)}") 