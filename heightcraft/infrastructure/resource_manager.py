"""
Resource manager for handling files, directories, and hardware resources.

This module provides functionality for managing application resources,
including loading, saving, and listing files and directories, as well as
managing hardware resources like GPU memory.
"""

import os
import glob
import json
import shutil
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from heightcraft.core.exceptions import ResourceError


class ResourceManager:
    """Manager for handling application resources including file and hardware resources."""
    
    _instance = None
    
    @classmethod
    def get_instance(cls, base_directory: Optional[str] = None) -> 'ResourceManager':
        """
        Get the singleton instance of ResourceManager.
        
        Args:
            base_directory: Base directory for resources. If None, the current working directory is used.
            
        Returns:
            ResourceManager instance
        """
        if cls._instance is None:
            cls._instance = cls(base_directory)
        elif base_directory is not None and cls._instance.base_directory != base_directory:
            logging.warning(f"ResourceManager already initialized with a different base directory: "
                           f"{cls._instance.base_directory}. Ignoring new value: {base_directory}.")
        return cls._instance
    
    def __init__(self, base_directory: Optional[str] = None) -> None:
        """
        Initialize the resource manager.
        
        Args:
            base_directory: Base directory for resources. If None, the current working directory is used.
        """
        self.base_directory = base_directory or os.getcwd()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def get_resource_path(self, resource_name: str) -> str:
        """
        Get the absolute path to a resource.
        
        Args:
            resource_name: Name of the resource (file or directory).
            
        Returns:
            Absolute path to the resource.
        """
        # Handle path separators consistently
        resource_name = resource_name.replace('/', os.path.sep).replace('\\', os.path.sep)
        
        # Join the base directory with the resource name
        return os.path.join(self.base_directory, resource_name)
    
    def resource_exists(self, resource_name: str) -> bool:
        """
        Check if a resource exists.
        
        Args:
            resource_name: Name of the resource (file or directory).
            
        Returns:
            True if the resource exists, False otherwise.
        """
        resource_path = self.get_resource_path(resource_name)
        return os.path.exists(resource_path)
    
    def load_resource(self, resource_name: str) -> str:
        """
        Load a text resource.
        
        Args:
            resource_name: Name of the resource.
            
        Returns:
            Content of the resource as a string.
            
        Raises:
            ResourceError: If the resource cannot be loaded.
        """
        resource_path = self.get_resource_path(resource_name)
        
        if not os.path.exists(resource_path):
            raise ResourceError(f"Resource not found: {resource_name}")
        
        if not os.path.isfile(resource_path):
            raise ResourceError(f"Resource is not a file: {resource_name}")
        
        try:
            with open(resource_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            raise ResourceError(f"Failed to load resource {resource_name}: {str(e)}")
    
    def load_json_resource(self, resource_name: str) -> Dict[str, Any]:
        """
        Load a JSON resource.
        
        Args:
            resource_name: Name of the resource.
            
        Returns:
            Content of the resource as a dictionary.
            
        Raises:
            ResourceError: If the resource cannot be loaded or is not valid JSON.
        """
        try:
            content = self.load_resource(resource_name)
            return json.loads(content)
        except json.JSONDecodeError as e:
            raise ResourceError(f"Resource {resource_name} is not valid JSON: {str(e)}")
        except ResourceError:
            raise
        except Exception as e:
            raise ResourceError(f"Failed to load JSON resource {resource_name}: {str(e)}")
    
    def save_resource(self, resource_name: str, content: str) -> bool:
        """
        Save a text resource.
        
        Args:
            resource_name: Name of the resource.
            content: Content to save.
            
        Returns:
            True if the resource was saved successfully.
            
        Raises:
            ResourceError: If the resource cannot be saved.
        """
        resource_path = self.get_resource_path(resource_name)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(resource_path), exist_ok=True)
        
        try:
            with open(resource_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        except Exception as e:
            raise ResourceError(f"Failed to save resource {resource_name}: {str(e)}")
    
    def list_resources(self, pattern: str = "*", directory: str = "") -> List[str]:
        """
        List resources matching a pattern.
        
        Args:
            pattern: Glob pattern to match.
            directory: Directory to search in, relative to the base directory.
                      If empty, the base directory is used.
            
        Returns:
            List of resources matching the pattern.
            
        Raises:
            ResourceError: If the directory cannot be accessed.
        """
        directory_path = self.get_resource_path(directory)
        
        if not os.path.exists(directory_path):
            raise ResourceError(f"Directory not found: {directory}")
        
        if not os.path.isdir(directory_path):
            raise ResourceError(f"Not a directory: {directory}")
        
        try:
            # Get full paths
            full_paths = glob.glob(os.path.join(directory_path, pattern))
            
            # Convert to paths relative to base directory
            relative_paths = [os.path.relpath(p, self.base_directory) for p in full_paths]
            
            return relative_paths
        except Exception as e:
            raise ResourceError(f"Failed to list resources in {directory}: {str(e)}")
    
    def delete_resource(self, resource_name: str) -> bool:
        """
        Delete a resource.
        
        Args:
            resource_name: Name of the resource.
            
        Returns:
            True if the resource was deleted successfully.
            
        Raises:
            ResourceError: If the resource cannot be deleted.
        """
        resource_path = self.get_resource_path(resource_name)
        
        if not os.path.exists(resource_path):
            # Already gone, consider it a success
            return True
        
        try:
            if os.path.isdir(resource_path):
                shutil.rmtree(resource_path)
            else:
                os.remove(resource_path)
            return True
        except Exception as e:
            raise ResourceError(f"Failed to delete resource {resource_name}: {str(e)}")
    
    def create_directory(self, directory_name: str) -> bool:
        """
        Create a directory.
        
        Args:
            directory_name: Name of the directory to create.
            
        Returns:
            True if the directory was created successfully.
            
        Raises:
            ResourceError: If the directory cannot be created.
        """
        directory_path = self.get_resource_path(directory_name)
        
        try:
            os.makedirs(directory_path, exist_ok=True)
            return True
        except Exception as e:
            raise ResourceError(f"Failed to create directory {directory_name}: {str(e)}")
    
    def get_directory_size(self, directory_name: str = "") -> int:
        """
        Get the size of a directory in bytes.
        
        Args:
            directory_name: Name of the directory, relative to the base directory.
                           If empty, the base directory is used.
            
        Returns:
            Size of the directory in bytes.
            
        Raises:
            ResourceError: If the directory cannot be accessed.
        """
        directory_path = self.get_resource_path(directory_name)
        
        if not os.path.exists(directory_path):
            raise ResourceError(f"Directory not found: {directory_name}")
        
        if not os.path.isdir(directory_path):
            raise ResourceError(f"Not a directory: {directory_name}")
        
        try:
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(directory_path):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    if os.path.exists(fp):  # Skip if file was removed
                        total_size += os.path.getsize(fp)
            return total_size
        except Exception as e:
            raise ResourceError(f"Failed to get directory size for {directory_name}: {str(e)}")
    
    def create_gpu_manager(self):
        """
        Create a GPU manager instance.
        
        This is a factory method that delegates to the GPUManager.
        
        Returns:
            GPU manager instance
        """
        from heightcraft.infrastructure.gpu_manager import gpu_manager 