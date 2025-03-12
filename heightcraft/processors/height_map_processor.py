"""
Height map processor for generating and processing height maps.

This module provides functionality for generating and processing height maps from meshes.
"""

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Union

from heightcraft.core.exceptions import ProcessingError
from heightcraft.domain.mesh import Mesh
from heightcraft.domain.height_map import HeightMap
from heightcraft.services.mesh_service import MeshService
from heightcraft.services.height_map_service import HeightMapService


@dataclass
class HeightMapConfig:
    """Configuration for height map generation and processing."""
    
    resolution: float = 0.1
    """Resolution of the height map."""
    
    width: int = 256
    """Width of the height map in pixels."""
    
    height: int = 256
    """Height of the height map in pixels."""
    
    normalize: bool = True
    """Whether to normalize the height map."""
    
    bit_depth: int = 16
    """Bit depth of the height map (8 or 16)."""
    
    center: bool = False
    """Whether to center the mesh before generating the height map."""
    
    align: bool = False
    """Whether to align the mesh to the XY plane before generating the height map."""
    
    flip_y: bool = False
    """Whether to flip the Y axis of the height map."""


class HeightMapProcessor:
    """Processor for generating and processing height maps."""
    
    def __init__(
        self,
        mesh_service: MeshService,
        height_map_service: HeightMapService
    ) -> None:
        """
        Initialize the height map processor.
        
        Args:
            mesh_service: Service for working with meshes.
            height_map_service: Service for working with height maps.
        """
        self.mesh_service = mesh_service
        self.height_map_service = height_map_service
        self._temp_files = []
    
    def generate_height_map(
        self,
        mesh: Mesh,
        config: HeightMapConfig
    ) -> HeightMap:
        """
        Generate a height map from a mesh.
        
        Args:
            mesh: The mesh to generate the height map from.
            config: Configuration for height map generation.
            
        Returns:
            The generated height map.
            
        Raises:
            ProcessingError: If the height map cannot be generated.
        """
        try:
            # Validate config
            if config.bit_depth not in [8, 16]:
                raise ProcessingError(f"Bit depth must be 8 or 16, got {config.bit_depth}")
            
            # Preprocess the mesh
            if config.center:
                mesh = self.mesh_service.center_mesh(mesh)
            
            if config.align:
                mesh = self.mesh_service.align_mesh_to_xy(mesh)
            
            # Generate the height map
            height_map = self.mesh_service.convert_mesh_to_height_map(mesh, config.resolution)
            
            # Postprocess the height map
            if config.normalize:
                height_map = self.height_map_service.normalize_height_map(height_map)
            
            # Resize the height map if needed
            if height_map.width != config.width or height_map.height != config.height:
                height_map = self.height_map_service.resize_height_map(height_map, config.width, config.height)
            
            return height_map
        
        except Exception as e:
            raise ProcessingError(f"Failed to generate height map: {str(e)}")
    
    def save_height_map(
        self,
        height_map: HeightMap,
        output_file: str
    ) -> bool:
        """
        Save a height map to a file.
        
        Args:
            height_map: The height map to save.
            output_file: Path where to save the height map.
            
        Returns:
            True if the height map was saved successfully.
            
        Raises:
            ProcessingError: If the height map cannot be saved.
        """
        try:
            return self.height_map_service.save_height_map(height_map, output_file)
        
        except Exception as e:
            raise ProcessingError(f"Failed to save height map: {str(e)}")
    
    def save_split_height_maps(
        self,
        height_maps: Dict[str, HeightMap],
        output_directory: str
    ) -> bool:
        """
        Save split height maps to files.
        
        Args:
            height_maps: Dictionary of height maps to save (key is the name).
            output_directory: Directory where to save the height maps.
            
        Returns:
            True if all height maps were saved successfully.
            
        Raises:
            ProcessingError: If any height map cannot be saved.
        """
        try:
            # Create the output directory if it doesn't exist
            os.makedirs(output_directory, exist_ok=True)
            
            # Save each height map
            for name, height_map in height_maps.items():
                output_file = os.path.join(output_directory, f"{name}.png")
                self.height_map_service.save_height_map(height_map, output_file)
            
            return True
        
        except Exception as e:
            raise ProcessingError(f"Failed to save split height maps: {str(e)}")
    
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
    
    def __enter__(self) -> 'HeightMapProcessor':
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.cleanup()
    
    def process_mesh_file_to_height_map(
        self,
        mesh_file: str,
        output_file: str,
        config: Optional[HeightMapConfig] = None
    ) -> bool:
        """
        Process a mesh file to a height map.
        
        Args:
            mesh_file: Path to the mesh file.
            output_file: Path where to save the height map.
            config: Configuration for height map generation. If None, default configuration is used.
            
        Returns:
            True if the height map was generated and saved successfully.
            
        Raises:
            ProcessingError: If the height map cannot be generated or saved.
        """
        try:
            # Use default config if not provided
            if config is None:
                config = HeightMapConfig()
            
            # Load the mesh
            mesh = self.mesh_service.load_mesh(mesh_file)
            
            # Generate the height map
            height_map = self.generate_height_map(mesh, config)
            
            # Save the height map
            return self.save_height_map(height_map, output_file)
        
        except Exception as e:
            raise ProcessingError(f"Failed to process mesh file to height map: {str(e)}")
    
    def process_mesh_file_to_multiple_views(
        self,
        mesh_file: str,
        output_directory: str,
        views: List[str] = ["top", "front", "side"],
        config: Optional[HeightMapConfig] = None
    ) -> bool:
        """
        Process a mesh file to multiple height map views.
        
        Args:
            mesh_file: Path to the mesh file.
            output_directory: Directory where to save the height maps.
            views: List of views to generate (e.g. ["top", "front", "side"]).
            config: Configuration for height map generation. If None, default configuration is used.
            
        Returns:
            True if all height maps were generated and saved successfully.
            
        Raises:
            ProcessingError: If any height map cannot be generated or saved.
        """
        try:
            # Use default config if not provided
            if config is None:
                config = HeightMapConfig()
            
            # Load the mesh
            mesh = self.mesh_service.load_mesh(mesh_file)
            
            # Center the mesh
            mesh = self.mesh_service.center_mesh(mesh)
            
            # Generate height maps for each view
            height_maps = {}
            
            for view in views:
                # Generate the height map
                view_mesh = mesh  # Start with the centered mesh
                
                # For non-top views, we need to rotate the mesh
                if view == "front":
                    # Rotate around X axis by 90 degrees
                    pass  # This would actually require a rotate method in mesh
                elif view == "side":
                    # Rotate around Y axis by 90 degrees
                    pass  # This would actually require a rotate method in mesh
                
                # Generate the height map for this view
                height_map = self.generate_height_map(view_mesh, config)
                
                # Add to the dictionary
                height_maps[view] = height_map
            
            # Save the height maps
            return self.save_split_height_maps(height_maps, output_directory)
        
        except Exception as e:
            raise ProcessingError(f"Failed to process mesh file to multiple views: {str(e)}")
    
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
            True if the mesh was generated and saved successfully.
            
        Raises:
            ProcessingError: If the mesh cannot be generated or saved.
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