"""
Repository for HeightMap domain objects.

This module provides a repository for loading and saving HeightMap objects.
"""

import os
import numpy as np
from typing import Optional

from heightcraft.core.exceptions import HeightMapValidationError
from heightcraft.domain.height_map import HeightMap
from heightcraft.infrastructure.resource_manager import ResourceManager


class HeightMapRepository:
    """Repository for HeightMap domain objects."""
    
    def __init__(self, resource_manager: Optional[ResourceManager] = None) -> None:
        """
        Initialize the height map repository.
        
        Args:
            resource_manager: Resource manager to use for file operations.
                If None, a new ResourceManager instance will be created.
        """
        self.resource_manager = resource_manager or ResourceManager()
    
    def load(self, file_path: str, bit_depth: int = 8) -> HeightMap:
        """
        Load a height map from a file.
        
        Args:
            file_path: Path to the height map file.
            bit_depth: Bit depth of the height map (8 or 16).
            
        Returns:
            A HeightMap object.
            
        Raises:
            HeightMapValidationError: If the height map cannot be loaded.
        """
        try:
            # If the file is a relative path, resolve it against the base directory
            if not os.path.isabs(file_path):
                file_path = self.resource_manager.get_resource_path(file_path)
            
            # Load the data from the file
            data = np.load(file_path)
            
            # Create and return a new HeightMap
            return HeightMap(data, bit_depth)
        except Exception as e:
            raise HeightMapValidationError(f"Failed to load height map from {file_path}: {str(e)}")
    
    def save(self, height_map: HeightMap, file_path: str) -> bool:
        """
        Save a height map to a file.
        
        Args:
            height_map: Height map to save.
            file_path: Path to save the height map to.
            
        Returns:
            True if the height map was saved successfully.
            
        Raises:
            HeightMapValidationError: If the height map cannot be saved.
        """
        try:
            # If the file is a relative path, resolve it against the base directory
            if not os.path.isabs(file_path):
                file_path = self.resource_manager.get_resource_path(file_path)
            
            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            
            # Save the data
            np.save(file_path, height_map.data)
            
            return True
        except Exception as e:
            raise HeightMapValidationError(f"Failed to save height map to {file_path}: {str(e)}")
    
    def list_height_maps(self, directory: str = "") -> list:
        """
        List all height maps in a directory.
        
        Args:
            directory: Directory to list height maps from, relative to the base directory.
            
        Returns:
            List of height map filenames.
        """
        try:
            return self.resource_manager.list_resources("*.npy", directory)
        except Exception as e:
            return [] 