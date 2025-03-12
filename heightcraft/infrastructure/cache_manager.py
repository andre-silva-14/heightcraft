"""
Cache manager for Heightcraft.

This module provides a CacheManager class for caching data to improve performance.
"""

import hashlib
import json
import logging
import os
import pickle
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from heightcraft.core.exceptions import CacheError
from heightcraft.infrastructure.file_storage import FileStorage


class CacheManager:
    """
    Cache manager for caching data to improve performance.
    
    This class provides methods for caching data to disk to avoid
    recomputing expensive operations.
    """
    
    def __init__(self, cache_dir: Optional[str] = None, max_age: int = 86400):
        """
        Initialize the cache manager.
        
        Args:
            cache_dir: Directory for storing cache files (defaults to .cache in current working directory)
            max_age: Maximum age of cache entries in seconds (defaults to 1 day)
        """
        self.cache_dir = cache_dir or os.path.join(os.getcwd(), ".cache")
        self.max_age = max_age
        self.logger = logging.getLogger(self.__class__.__name__)
        self.file_storage = FileStorage()
        
        # Create cache directory
        self._ensure_cache_directory()
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a value from the cache.
        
        Args:
            key: Cache key
            default: Default value to return if the key is not in the cache
            
        Returns:
            Cached value or default
        """
        cache_path = self._get_cache_path(key)
        
        try:
            # Check if cache file exists
            if not self.file_storage.file_exists(cache_path):
                return default
            
            # Check if cache file is too old
            if self._is_cache_expired(cache_path):
                self.logger.debug(f"Cache expired for key: {key}")
                return default
            
            # Load cache
            with open(cache_path, "rb") as f:
                self.logger.debug(f"Cache hit for key: {key}")
                return pickle.load(f)
                
        except Exception as e:
            self.logger.warning(f"Failed to get cache entry: {e}")
            return default
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            
        Raises:
            CacheError: If the value cannot be cached
        """
        cache_path = self._get_cache_path(key)
        
        try:
            # Ensure cache directory exists
            self._ensure_cache_directory()
            
            # Save cache
            with open(cache_path, "wb") as f:
                pickle.dump(value, f)
                
            self.logger.debug(f"Cached value for key: {key}")
                
        except Exception as e:
            raise CacheError(f"Failed to set cache entry: {e}")
    
    def delete(self, key: str) -> None:
        """
        Delete a value from the cache.
        
        Args:
            key: Cache key
        """
        cache_path = self._get_cache_path(key)
        
        try:
            # Delete cache file
            self.file_storage.delete_file(cache_path)
            self.logger.debug(f"Deleted cache entry for key: {key}")
                
        except Exception as e:
            self.logger.warning(f"Failed to delete cache entry: {e}")
    
    def clear(self) -> None:
        """
        Clear all cache entries.
        
        Raises:
            CacheError: If the cache cannot be cleared
        """
        try:
            # Clear cache directory
            self.file_storage.clear_directory(self.cache_dir)
            self.logger.debug("Cleared cache")
                
        except Exception as e:
            raise CacheError(f"Failed to clear cache: {e}")
    
    def get_numpy_array(self, key: str, default: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """
        Get a numpy array from the cache.
        
        This method is optimized for storing and retrieving numpy arrays.
        
        Args:
            key: Cache key
            default: Default value to return if the key is not in the cache
            
        Returns:
            Cached numpy array or default
        """
        cache_path = self._get_cache_path(key, extension=".npy")
        
        try:
            # Check if cache file exists
            if not self.file_storage.file_exists(cache_path):
                return default
            
            # Check if cache file is too old
            if self._is_cache_expired(cache_path):
                self.logger.debug(f"Cache expired for key: {key}")
                return default
            
            # Load cache
            self.logger.debug(f"Cache hit for key: {key}")
            return np.load(cache_path)
                
        except Exception as e:
            self.logger.warning(f"Failed to get cache entry: {e}")
            return default
    
    def set_numpy_array(self, key: str, value: np.ndarray) -> None:
        """
        Set a numpy array in the cache.
        
        This method is optimized for storing and retrieving numpy arrays.
        
        Args:
            key: Cache key
            value: Numpy array to cache
            
        Raises:
            CacheError: If the array cannot be cached
        """
        cache_path = self._get_cache_path(key, extension=".npy")
        
        try:
            # Ensure cache directory exists
            self._ensure_cache_directory()
            
            # Save cache
            np.save(cache_path, value)
            self.logger.debug(f"Cached numpy array for key: {key}")
                
        except Exception as e:
            raise CacheError(f"Failed to set cache entry: {e}")
    
    @contextmanager
    def cache_computation(self, key: str, force_recompute: bool = False):
        """
        Cache a computation.
        
        This context manager caches the result of a computation.
        
        Args:
            key: Cache key
            force_recompute: Whether to force recomputation even if the cache is valid
            
        Yields:
            A function to set the result
            
        Example:
            ```python
            with cache_manager.cache_computation("my_key") as set_result:
                # Expensive computation
                result = compute_expensive_result()
                set_result(result)
            ```
        """
        if not force_recompute:
            # Try to get from cache
            result = self.get(key)
            if result is not None:
                yield lambda _: None
                return
        
        result = None
        
        def set_result(value):
            nonlocal result
            result = value
        
        yield set_result
        
        if result is not None:
            self.set(key, result)
    
    def _get_cache_path(self, key: str, extension: str = ".pickle") -> str:
        """
        Get the cache file path for a key.
        
        Args:
            key: Cache key
            extension: File extension
            
        Returns:
            Cache file path
        """
        # Create a hash of the key
        key_hash = hashlib.md5(key.encode()).hexdigest()
        
        # Create a valid filename
        filename = f"{key_hash}{extension}"
        
        return os.path.join(self.cache_dir, filename)
    
    def _ensure_cache_directory(self) -> None:
        """
        Ensure the cache directory exists.
        
        Raises:
            CacheError: If the cache directory cannot be created
        """
        try:
            self.file_storage.ensure_directory(self.cache_dir)
        except Exception as e:
            raise CacheError(f"Failed to ensure cache directory exists: {e}")
    
    def _is_cache_expired(self, cache_path: str) -> bool:
        """
        Check if a cache entry is expired.
        
        Args:
            cache_path: Path to the cache file
            
        Returns:
            True if the cache entry is expired, False otherwise
        """
        try:
            # Get file modification time
            mtime = os.path.getmtime(cache_path)
            
            # Check if file is too old
            return time.time() - mtime > self.max_age
                
        except Exception:
            # If we can't check the mtime, assume it's expired
            return True 