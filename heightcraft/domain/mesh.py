"""
Mesh domain model for Heightcraft.

This module provides the Mesh class which encapsulates a 3D mesh and
provides domain-specific behavior and validation.
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import trimesh

from heightcraft.core.exceptions import MeshValidationError


class Mesh:
    """
    Domain model representing a 3D mesh.
    
    This class wraps the trimesh.Trimesh class and adds domain-specific
    behavior and validation.
    """
    
    def __init__(self, mesh: trimesh.Trimesh):
        """
        Initialize the mesh.
        
        Args:
            mesh: The trimesh.Trimesh object to wrap
            
        Raises:
            MeshValidationError: If the mesh fails validation
        """
        self._validate_mesh(mesh)
        self._mesh = mesh
        self._logger = logging.getLogger(self.__class__.__name__)
    
    @property
    def mesh(self) -> trimesh.Trimesh:
        """Get the underlying trimesh.Trimesh object."""
        return self._mesh
    
    @property
    def vertices(self) -> np.ndarray:
        """Get the mesh vertices."""
        return self._mesh.vertices
    
    @property
    def faces(self) -> np.ndarray:
        """Get the mesh faces."""
        return self._mesh.faces
    
    @property
    def bounds(self) -> np.ndarray:
        """Get the mesh bounds."""
        return self._mesh.bounds
    
    @property
    def extents(self) -> np.ndarray:
        """Get the mesh extents."""
        return self._mesh.extents
    
    @property
    def is_watertight(self) -> bool:
        """Check if the mesh is watertight."""
        return self._mesh.is_watertight
    
    @property
    def is_winding_consistent(self) -> bool:
        """Check if the mesh has consistent face winding."""
        return self._mesh.is_winding_consistent
    
    @property
    def vertex_count(self) -> int:
        """Get the number of vertices."""
        return len(self._mesh.vertices)
    
    @property
    def face_count(self) -> int:
        """Get the number of faces."""
        return len(self._mesh.faces)
    
    @property
    def has_degenerate_faces(self) -> bool:
        """Check if the mesh has degenerate faces."""
        return hasattr(self._mesh, 'is_degenerate') and self._mesh.is_degenerate.any()
    
    def center(self) -> None:
        """Center the mesh at the origin."""
        centroid = np.mean(self._mesh.vertices, axis=0)
        self._mesh.apply_translation(-centroid)
        self._logger.info(f"Mesh centered by translating by {-centroid}")
    
    def align_to_xy(self) -> None:
        """Align the mesh to the XY plane with Z pointing up."""
        self._logger.info("Aligning mesh to XY plane")
        # Ensure Z is up
        self._mesh.apply_transform(trimesh.transformations.rotation_matrix(
            np.pi / 2, [1, 0, 0]))
    
    def get_aspect_ratio(self) -> float:
        """
        Calculate the aspect ratio of the mesh (X/Y).
        
        Returns:
            Aspect ratio
        """
        extents = self._mesh.extents
        return extents[0] / extents[1] if extents[1] != 0 else 1.0
    
    def to_dict(self) -> Dict:
        """
        Convert the mesh to a dictionary.
        
        Returns:
            Dictionary representation of the mesh
        """
        return {
            "vertex_count": self.vertex_count,
            "face_count": self.face_count,
            "bounds": self.bounds.tolist(),
            "extents": self.extents.tolist(),
            "is_watertight": self.is_watertight,
            "is_winding_consistent": self.is_winding_consistent,
        }
    
    @staticmethod
    def _validate_mesh(mesh: trimesh.Trimesh) -> None:
        """
        Validate the mesh.
        
        Args:
            mesh: The mesh to validate
            
        Raises:
            MeshValidationError: If the mesh fails validation
        """
        if not isinstance(mesh, trimesh.Trimesh):
            raise MeshValidationError(f"Expected trimesh.Trimesh, got {type(mesh)}")
        
        try:
            # Use centralized validation
            from heightcraft.utils.validators import validate_mesh
            validate_mesh(mesh)
        except ImportError:
            # Fallback if validators module can't be imported
            if len(mesh.vertices) == 0:
                raise MeshValidationError("Mesh has no vertices")
            
            if len(mesh.faces) == 0:
                raise MeshValidationError("Mesh has no faces")
    
    @classmethod
    def from_file(cls, file_path: str) -> "Mesh":
        """
        Load a mesh from a file.
        
        This method is deprecated. Use ModelService instead.
        
        Args:
            file_path: Path to the mesh file
            
        Returns:
            Loaded mesh
            
        Raises:
            MeshValidationError: If the mesh fails validation
        """
        warnings.warn(
            "Mesh.from_file is deprecated. Use ModelService.load_model instead.",
            DeprecationWarning,
            stacklevel=2
        )
        
        try:
            # Load the mesh
            loaded = trimesh.load(file_path)
            
            # Handle scenes
            if isinstance(loaded, trimesh.Scene):
                # Extract a single mesh from the scene
                mesh = loaded.dump(concatenate=True)
            else:
                mesh = loaded
            
            return cls(mesh)
        except Exception as e:
            raise MeshValidationError(f"Failed to load mesh from {file_path}: {e}") 