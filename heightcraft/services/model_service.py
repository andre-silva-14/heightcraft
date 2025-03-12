"""
Model service for Heightcraft.

This module provides the ModelService class for loading and converting 3D models
from different file formats.
"""

import logging
import os
from typing import Optional, Tuple, Union

import numpy as np
import trimesh

from heightcraft.core.config import ModelConfig
from heightcraft.core.exceptions import ModelLoadError
from heightcraft.domain.mesh import Mesh
from heightcraft.infrastructure.cache_manager import CacheManager
from heightcraft.infrastructure.profiler import profiler


class ModelService:
    """
    Service for loading and converting 3D models from different file formats.
    
    This class focuses on file I/O operations and format conversion, leaving
    mesh-specific operations to the MeshService.
    """
    
    def __init__(self, cache_manager: Optional[CacheManager] = None):
        """
        Initialize the model service.
        
        Args:
            cache_manager: Cache manager for caching models
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.cache_manager = cache_manager or CacheManager()
    
    @profiler.profile()
    def load_model(self, file_path: str, use_cache: bool = True) -> Mesh:
        """
        Load a 3D model from a file.
        
        Args:
            file_path: Path to the model file
            use_cache: Whether to use the cache
            
        Returns:
            Loaded mesh
            
        Raises:
            ModelLoadError: If the model cannot be loaded
        """
        if not os.path.exists(file_path):
            raise ModelLoadError(f"Model file not found: {file_path}")
        
        # Try to get from cache first
        if use_cache:
            cached_mesh = self._get_from_cache(file_path)
            if cached_mesh is not None:
                self.logger.info(f"Loaded model from cache: {file_path}")
                return cached_mesh
        
        # Load from file
        try:
            self.logger.info(f"Loading model from file: {file_path}")
            
            # Determine file format from extension
            ext = os.path.splitext(file_path)[1].lower()
            
            # Load based on format
            if ext in ['.stl', '.obj', '.ply', '.glb', '.gltf']:
                # Use trimesh for standard formats
                mesh_data = trimesh.load(file_path)
                
                # Handle different return types from trimesh.load
                if isinstance(mesh_data, trimesh.Scene):
                    self.logger.info("Loaded a scene with multiple meshes")
                    # For scenes, we need to extract the mesh(es)
                    meshes = [Mesh(m) for m in mesh_data.geometry.values()]
                    if len(meshes) == 1:
                        mesh = meshes[0]
                    else:
                        # For multiple meshes, create a single combined mesh
                        combined = trimesh.util.concatenate(
                            [m.mesh for m in meshes]
                        )
                        mesh = Mesh(combined)
                else:
                    mesh = Mesh(mesh_data)
            else:
                raise ModelLoadError(f"Unsupported file format: {ext}")
            
            # Cache the result
            if use_cache:
                self._add_to_cache(file_path, mesh)
            
            return mesh
        except Exception as e:
            raise ModelLoadError(f"Failed to load model from {file_path}: {str(e)}")
    
    def _get_from_cache(self, file_path: str) -> Optional[Mesh]:
        """
        Get a model from the cache.
        
        Args:
            file_path: Path to the model file
            
        Returns:
            Mesh from cache, or None if not found
        """
        try:
            if self.cache_manager.has(file_path):
                return self.cache_manager.get(file_path)
        except Exception as e:
            self.logger.warning(f"Error getting model from cache: {str(e)}")
        
        return None
    
    def _add_to_cache(self, file_path: str, mesh: Mesh) -> None:
        """
        Add a model to the cache.
        
        Args:
            file_path: Path to the model file
            mesh: Mesh to cache
        """
        try:
            self.cache_manager.set(file_path, mesh)
        except Exception as e:
            self.logger.warning(f"Error adding model to cache: {str(e)}")
    
    def get_supported_formats(self) -> list[str]:
        """
        Get list of supported file formats.
        
        Returns:
            List of supported file extensions
        """
        return ['.stl', '.obj', '.ply', '.glb', '.gltf']
    
    def convert_model_format(self, mesh: Mesh, output_format: str, output_path: str) -> str:
        """
        Convert a mesh to a different format and save it.
        
        Args:
            mesh: Mesh to convert
            output_format: Output format extension (e.g., '.obj')
            output_path: Output file path
            
        Returns:
            Path to the converted file
            
        Raises:
            ModelLoadError: If the conversion fails
        """
        if output_format not in self.get_supported_formats():
            raise ModelLoadError(f"Unsupported output format: {output_format}")
        
        try:
            # Make sure path ends with the correct extension
            if not output_path.lower().endswith(output_format):
                output_path = f"{os.path.splitext(output_path)[0]}{output_format}"
            
            # Export mesh
            mesh.mesh.export(output_path)
            
            return output_path
        except Exception as e:
            raise ModelLoadError(f"Failed to convert model to {output_format}: {str(e)}") 