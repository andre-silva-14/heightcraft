"""
File storage for Heightcraft.

This module provides a FileStorage class for handling file operations,
including loading, saving, and caching files.
"""

import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from heightcraft.core.exceptions import FileError


class FileStorage:
    """
    File storage for handling file operations.
    
    This class provides methods for loading, saving, and caching files.
    """
    
    def __init__(self, base_dir: Optional[str] = None):
        """
        Initialize the file storage.
        
        Args:
            base_dir: Base directory for file operations (defaults to current working directory)
        """
        self.base_dir = base_dir or os.getcwd()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def ensure_directory(self, directory: str) -> str:
        """
        Ensure a directory exists, creating it if necessary.
        
        Args:
            directory: Directory path (absolute or relative to base_dir)
            
        Returns:
            Absolute path to the directory
            
        Raises:
            FileError: If the directory cannot be created
        """
        try:
            # Get absolute path
            directory = self._get_absolute_path(directory)
            
            # Create directory if it doesn't exist
            os.makedirs(directory, exist_ok=True)
            
            return directory
        except Exception as e:
            raise FileError(f"Failed to ensure directory exists: {e}")
    
    def file_exists(self, file_path: str) -> bool:
        """
        Check if a file exists.
        
        Args:
            file_path: File path (absolute or relative to base_dir)
            
        Returns:
            True if the file exists, False otherwise
        """
        return os.path.isfile(self._get_absolute_path(file_path))
    
    def directory_exists(self, directory: str) -> bool:
        """
        Check if a directory exists.
        
        Args:
            directory: Directory path (absolute or relative to base_dir)
            
        Returns:
            True if the directory exists, False otherwise
        """
        return os.path.isdir(self._get_absolute_path(directory))
    
    def list_files(self, directory: str, pattern: Optional[str] = None) -> List[str]:
        """
        List files in a directory.
        
        Args:
            directory: Directory path (absolute or relative to base_dir)
            pattern: Glob pattern for filtering files
            
        Returns:
            List of file paths
            
        Raises:
            FileError: If the directory does not exist
        """
        try:
            # Get absolute path
            directory = self._get_absolute_path(directory)
            
            # Check if directory exists
            if not os.path.isdir(directory):
                raise FileError(f"Directory does not exist: {directory}")
            
            # List files
            if pattern:
                return list(str(p) for p in Path(directory).glob(pattern))
            else:
                return list(str(p) for p in Path(directory).iterdir() if p.is_file())
        except Exception as e:
            if isinstance(e, FileError):
                raise
            raise FileError(f"Failed to list files: {e}")
    
    def delete_file(self, file_path: str) -> None:
        """
        Delete a file.
        
        Args:
            file_path: File path (absolute or relative to base_dir)
            
        Raises:
            FileError: If the file cannot be deleted
        """
        try:
            # Get absolute path
            file_path = self._get_absolute_path(file_path)
            
            # Check if file exists
            if not os.path.isfile(file_path):
                self.logger.warning(f"File does not exist: {file_path}")
                return
            
            # Delete file
            os.remove(file_path)
            self.logger.debug(f"Deleted file: {file_path}")
        except Exception as e:
            raise FileError(f"Failed to delete file: {e}")
    
    def clear_directory(self, directory: str) -> None:
        """
        Clear a directory (remove all files and subdirectories).
        
        Args:
            directory: Directory path (absolute or relative to base_dir)
            
        Raises:
            FileError: If the directory cannot be cleared
        """
        try:
            # Get absolute path
            directory = self._get_absolute_path(directory)
            
            # Check if directory exists
            if not os.path.isdir(directory):
                self.logger.warning(f"Directory does not exist: {directory}")
                return
            
            # Clear directory
            for item in os.listdir(directory):
                item_path = os.path.join(directory, item)
                if os.path.isfile(item_path):
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
            
            self.logger.debug(f"Cleared directory: {directory}")
        except Exception as e:
            raise FileError(f"Failed to clear directory: {e}")
    
    def get_file_size(self, file_path: str) -> int:
        """
        Get the size of a file in bytes.
        
        Args:
            file_path: File path (absolute or relative to base_dir)
            
        Returns:
            File size in bytes
            
        Raises:
            FileError: If the file size cannot be determined
        """
        try:
            # Get absolute path
            file_path = self._get_absolute_path(file_path)
            
            # Check if file exists
            if not os.path.isfile(file_path):
                raise FileError(f"File does not exist: {file_path}")
            
            # Get file size
            return os.path.getsize(file_path)
        except Exception as e:
            if isinstance(e, FileError):
                raise
            raise FileError(f"Failed to get file size: {e}")
    
    def get_file_extension(self, file_path: str) -> str:
        """
        Get the extension of a file.
        
        Args:
            file_path: File path (absolute or relative to base_dir)
            
        Returns:
            File extension (including the dot)
        """
        return os.path.splitext(file_path)[1]
    
    def join_paths(self, *paths: str) -> str:
        """
        Join paths.
        
        Args:
            *paths: Paths to join
            
        Returns:
            Joined path
        """
        return os.path.join(*paths)
    
    def _get_absolute_path(self, path: str) -> str:
        """
        Get the absolute path.
        
        Args:
            path: Path (absolute or relative to base_dir)
            
        Returns:
            Absolute path
        """
        if os.path.isabs(path):
            return path
        return os.path.join(self.base_dir, path) 