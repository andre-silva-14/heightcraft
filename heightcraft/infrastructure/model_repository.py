"""
Model repository for loading and saving 3D models.

This module provides functionality for loading and saving 3D models
in various formats using the trimesh library.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import trimesh

from heightcraft.core.exceptions import RepositoryError
from heightcraft.domain.mesh import Mesh


class ModelRepository:
    """Repository for loading and saving 3D models."""
    
    def __init__(self) -> None:
        """Initialize the model repository."""
        self._supported_formats = [
            '.obj', '.stl', '.ply', '.glb', '.gltf', '.off', '.collada', '.dae', '.3ds'
        ]
    
    def load(self, file_path: str) -> Mesh:
        """
        Load a 3D model from a file.
        
        Args:
            file_path: Path to the 3D model file.
            
        Returns:
            A Mesh object.
            
        Raises:
            RepositoryError: If the file cannot be loaded or is invalid.
        """
        if not os.path.exists(file_path):
            raise RepositoryError(f"File not found: {file_path}")
        
        try:
            # Load the model
            loaded = trimesh.load(
                file_path,
                force='mesh',
                process=True,
                maintain_order=True,
                skip_materials=True,
                skip_texture=True
            )
            
            # Handle both single meshes and scenes with meshes
            if isinstance(loaded, trimesh.Scene):
                # Match legacy implementation using dump with concatenate=True
                mesh = loaded.dump(concatenate=True)
            else:
                mesh = loaded
            
            # Create a domain model mesh
            return Mesh(mesh)
        
        except Exception as e:
            raise RepositoryError(f"Failed to load model from {file_path}: {str(e)}")
    
    def save(self, mesh: Mesh, file_path: str) -> bool:
        """
        Save a 3D model to a file.
        
        Args:
            mesh: The mesh to save.
            file_path: Path where to save the 3D model.
            
        Returns:
            True if the mesh was saved successfully.
            
        Raises:
            RepositoryError: If the mesh cannot be saved.
        """
        # Check if the file extension is supported
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext not in self._supported_formats:
            raise RepositoryError(f"Unsupported file format: {file_ext}")
        
        try:
            # Get the underlying trimesh mesh
            trimesh_mesh = mesh.mesh
            
            # Save the mesh
            trimesh_mesh.export(file_path)
            
            return True
        
        except Exception as e:
            raise RepositoryError(f"Failed to save model to {file_path}: {str(e)}")
    
    def supported_formats(self) -> List[str]:
        """
        Get a list of supported file formats.
        
        Returns:
            A list of supported file extensions.
        """
        return self._supported_formats 