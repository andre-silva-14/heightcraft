"""
Mesh processor for generating and processing meshes.

This module provides functionality for processing meshes and converting them
to other formats.
"""

import os
from typing import List, Optional, Tuple, Dict, Union

from heightcraft.core.exceptions import ProcessingError
from heightcraft.domain.mesh import Mesh
from heightcraft.domain.height_map import HeightMap
from heightcraft.domain.point_cloud import PointCloud
from heightcraft.services.mesh_service import MeshService
from heightcraft.services.height_map_service import HeightMapService
from heightcraft.processors.height_map_processor import HeightMapConfig


class MeshProcessor:
    """Processor for generating and processing meshes."""
    
    def __init__(
        self,
        mesh_service: MeshService,
        height_map_service: HeightMapService
    ) -> None:
        """
        Initialize the mesh processor.
        
        Args:
            mesh_service: Service for working with meshes.
            height_map_service: Service for working with height maps.
        """
        self.mesh_service = mesh_service
        self.height_map_service = height_map_service
        self._temp_files = []
    
    def process_mesh(
        self,
        mesh: Mesh,
        center: bool = False,
        align_to_xy: bool = False
    ) -> Mesh:
        """
        Process a mesh.
        
        Args:
            mesh: The mesh to process.
            center: Whether to center the mesh.
            align_to_xy: Whether to align the mesh to the XY plane.
            
        Returns:
            The processed mesh.
            
        Raises:
            ProcessingError: If the mesh cannot be processed.
        """
        try:
            processed_mesh = mesh
            
            if center:
                processed_mesh = self.mesh_service.center_mesh(processed_mesh)
            
            if align_to_xy:
                processed_mesh = self.mesh_service.align_mesh_to_xy(processed_mesh)
            
            return processed_mesh
        
        except Exception as e:
            raise ProcessingError(f"Failed to process mesh: {str(e)}")
    
    def sample_points_from_mesh(
        self,
        mesh: Mesh,
        num_points: int
    ) -> PointCloud:
        """
        Sample points from a mesh.
        
        Args:
            mesh: The mesh to sample points from.
            num_points: The number of points to sample.
            
        Returns:
            The sampled point cloud.
            
        Raises:
            ProcessingError: If the points cannot be sampled.
        """
        try:
            return self.mesh_service.sample_points_from_mesh(mesh, num_points)
        
        except Exception as e:
            raise ProcessingError(f"Failed to sample points from mesh: {str(e)}")
    
    def convert_mesh_to_height_map(
        self,
        mesh: Mesh,
        resolution: float,
        normalize: bool = True
    ) -> HeightMap:
        """
        Convert a mesh to a height map.
        
        Args:
            mesh: The mesh to convert.
            resolution: The resolution of the height map.
            normalize: Whether to normalize the height map.
            
        Returns:
            The converted height map.
            
        Raises:
            ProcessingError: If the mesh cannot be converted.
        """
        try:
            height_map = self.mesh_service.convert_mesh_to_height_map(mesh, resolution)
            
            if normalize:
                height_map = self.height_map_service.normalize_height_map(height_map)
            
            return height_map
        
        except Exception as e:
            raise ProcessingError(f"Failed to convert mesh to height map: {str(e)}")
    
    def process_mesh_file(
        self,
        input_file: str,
        output_file: str,
        center: bool = False,
        align_to_xy: bool = False
    ) -> bool:
        """
        Process a mesh file.
        
        Args:
            input_file: Path to the input mesh file.
            output_file: Path where to save the processed mesh.
            center: Whether to center the mesh.
            align_to_xy: Whether to align the mesh to the XY plane.
            
        Returns:
            True if the mesh was processed and saved successfully.
            
        Raises:
            ProcessingError: If the mesh cannot be processed or saved.
        """
        try:
            # Load the mesh
            mesh = self.mesh_service.load_mesh(input_file)
            
            # Process the mesh
            processed_mesh = self.process_mesh(mesh, center, align_to_xy)
            
            # Save the mesh
            return self.mesh_service.save_mesh(processed_mesh, output_file)
        
        except Exception as e:
            raise ProcessingError(f"Failed to process mesh file: {str(e)}")
    
    def process_mesh_to_height_map(
        self,
        mesh: Mesh,
        resolution: float,
        center: bool = False,
        align_to_xy: bool = False,
        normalize: bool = True,
        apply_blur: bool = False,
        blur_sigma: float = 1.0
    ) -> HeightMap:
        """
        Process a mesh to a height map.
        
        Args:
            mesh: The mesh to process.
            resolution: The resolution of the height map.
            center: Whether to center the mesh.
            align_to_xy: Whether to align the mesh to the XY plane.
            normalize: Whether to normalize the height map.
            apply_blur: Whether to apply a Gaussian blur to the height map.
            blur_sigma: Sigma for the Gaussian blur.
            
        Returns:
            The processed height map.
            
        Raises:
            ProcessingError: If the mesh cannot be processed.
        """
        try:
            # Process the mesh
            processed_mesh = self.process_mesh(mesh, center, align_to_xy)
            
            # Convert the mesh to a height map
            height_map = self.convert_mesh_to_height_map(processed_mesh, resolution, normalize)
            
            # Apply blur if requested
            if apply_blur:
                height_map = self.height_map_service.apply_gaussian_blur(height_map, blur_sigma)
            
            return height_map
        
        except Exception as e:
            raise ProcessingError(f"Failed to process mesh to height map: {str(e)}")
    
    def process_mesh_file_to_height_map(
        self,
        input_file: str,
        output_file: str,
        resolution: float = 0.1,
        center: bool = False,
        align_to_xy: bool = False,
        normalize: bool = True,
        apply_blur: bool = False,
        blur_sigma: float = 1.0
    ) -> bool:
        """
        Process a mesh file to a height map file.
        
        Args:
            input_file: Path to the input mesh file.
            output_file: Path where to save the height map.
            resolution: The resolution of the height map.
            center: Whether to center the mesh.
            align_to_xy: Whether to align the mesh to the XY plane.
            normalize: Whether to normalize the height map.
            apply_blur: Whether to apply a Gaussian blur to the height map.
            blur_sigma: Sigma for the Gaussian blur.
            
        Returns:
            True if the mesh was processed and saved successfully.
            
        Raises:
            ProcessingError: If the mesh cannot be processed or saved.
        """
        try:
            # Load the mesh
            mesh = self.mesh_service.load_mesh(input_file)
            
            # Process the mesh to a height map
            height_map = self.process_mesh_to_height_map(
                mesh,
                resolution,
                center,
                align_to_xy,
                normalize,
                apply_blur,
                blur_sigma
            )
            
            # Save the height map
            return self.height_map_service.save_height_map(height_map, output_file)
        
        except Exception as e:
            raise ProcessingError(f"Failed to process mesh file to height map: {str(e)}")
    
    def process_height_map_to_mesh(
        self,
        height_map_file: str,
        output_file: str,
        resolution: float = 1.0,
        resize: Optional[Tuple[int, int]] = None,
        apply_filter: bool = False,
        filter_size: int = 3,
        center: bool = False
    ) -> bool:
        """
        Process a height map file to a mesh.
        
        Args:
            height_map_file: Path to the height map file.
            output_file: Path where to save the mesh.
            resolution: The resolution of the height map.
            resize: Target dimensions (width, height) to resize the height map to.
            apply_filter: Whether to apply a median filter to the height map.
            filter_size: Size of the median filter kernel. Must be odd.
            center: Whether to center the mesh after conversion.
            
        Returns:
            True if the height map was processed and saved successfully.
            
        Raises:
            ProcessingError: If the height map cannot be processed or saved.
        """
        try:
            # Load the height map
            height_map = self.height_map_service.load_height_map(height_map_file, resolution)
            
            # Resize the height map if needed
            if resize is not None:
                height_map = self.height_map_service.resize_height_map(height_map, resize[0], resize[1])
            
            # Apply filter if requested
            if apply_filter:
                height_map = self.height_map_service.apply_median_filter(height_map, filter_size)
            
            # Convert the height map to a mesh
            mesh = self.height_map_service.convert_height_map_to_mesh(height_map)
            
            # Center the mesh if requested
            if center:
                mesh = self.mesh_service.center_mesh(mesh)
            
            # Save the mesh
            return self.mesh_service.save_mesh(mesh, output_file)
        
        except Exception as e:
            raise ProcessingError(f"Failed to process height map to mesh: {str(e)}")
    
    def cleanup(self) -> None:
        """
        Clean up temporary files.
        
        This method removes all temporary files created by the processor.
        """
        for file_path in self._temp_files:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception:
                    pass
        
        self._temp_files = []
    
    def __enter__(self) -> 'MeshProcessor':
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.cleanup() 