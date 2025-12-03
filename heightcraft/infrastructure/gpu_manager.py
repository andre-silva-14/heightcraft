"""
GPU resource management for Heightcraft.

This module provides a robust GPU resource manager that ensures proper allocation,
tracking, and cleanup of GPU resources.
"""

import logging
import weakref
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Union

# Import torch conditionally
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Import exceptions
from heightcraft.core.exceptions import GPUError


class GPUManager:
    """
    Manager for GPU resources and operations.
    
    This class manages GPU resources and provides utilities for GPU-accelerated operations.
    """
    
    _instance = None
    
    @classmethod
    def get_instance(cls):
        """
        Get or create the singleton instance of the GPU manager.
        
        Returns:
            The singleton instance
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        """Initialize the GPU manager."""
        import logging
        
        # Set defaults for all attributes
        self.torch_available = False
        self.has_gpu = False
        self.gpu_tensors = weakref.WeakSet()
        
        # Initialize GPU-related resources
        try:
            # Import necessary libraries
            import torch
            
            # Check GPU availability
            self.torch_available = torch.cuda.is_available()
            self.has_gpu = self.torch_available
            
            if self.has_gpu:
                logging.info("GPU support enabled. Checking devices...")
                
                # Get device information
                device_count = torch.cuda.device_count()
                devices = []
                
                for i in range(device_count):
                    name = torch.cuda.get_device_name(i)
                    memory = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
                    devices.append((name, memory))
                
                logging.info(f"GPU support enabled. {device_count} device(s) found.")
                for i, (name, memory) in enumerate(devices):
                    logging.info(f"  Device {i}: {name} ({memory:.2f} GB)")
            else:
                logging.warning("No GPU support detected.")
                
        except ImportError:
            logging.warning("GPU libraries not available. Using CPU only.")
    
    def allocate_tensor(self, data, dtype=None):
        """
        Allocate a tensor on the GPU and track it for automatic cleanup.
        
        Args:
            data: Data to be allocated on GPU
            dtype: Data type for the tensor
            
        Returns:
            PyTorch tensor on GPU
            
        Raises:
            RuntimeError: If GPU is not available
        """
        import logging
        
        try:
            # Import torch
            import torch
            
            # Check if GPU is actually available
            if not self.has_gpu or not torch.cuda.is_available():
                logging.warning("GPU not available for tensor allocation, using CPU")
                return torch.tensor(data, dtype=dtype)
            
            # Create tensor on GPU
            tensor = torch.tensor(data, device="cuda", dtype=dtype)
            
            # Track tensor for cleanup
            self.gpu_tensors.add(tensor)
            
            return tensor
            
        except Exception as e:
            logging.error(f"Error allocating GPU tensor: {e}")
            # Fallback to CPU
            import torch
            return torch.tensor(data, dtype=dtype)
    
    @contextmanager
    def session(self):
        """
        Context manager for GPU operations to ensure proper cleanup.
        """
        try:
            yield
        finally:
            self.clear_memory()
    
    def clear_memory(self):
        """Clear GPU memory to prevent leaks."""
        import logging
        
        try:
            # Clear PyTorch memory if available
            if self.torch_available:
                import torch
                
                # Release tracked tensors
                for tensor in self.gpu_tensors:
                    del tensor
                self.gpu_tensors.clear()
                
                # Force CUDA memory cleanup
                torch.cuda.empty_cache()
                logging.info("GPU memory cleared")
            
            # Force garbage collection
            import gc
            gc.collect()
            
        except Exception as e:
            logging.error(f"Error clearing GPU memory: {e}")

    def get_free_memory(self) -> int:
        """
        Get free GPU memory in bytes.
        
        Returns:
            Free memory in bytes
        """
        if not self.has_gpu:
            return 0
        
        try:
            # Force garbage collection to get accurate memory stats
            self._collect_garbage()
            
            import torch
            # Get memory statistics
            device = torch.cuda.current_device()
            return torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
        except Exception as e:
            # Use logging instead of self.logger as it might not be initialized
            logging.warning(f"Could not get free GPU memory: {e}")
            return 0
    
    def _collect_garbage(self) -> None:
        """Force garbage collection for better memory stats."""
        try:
            # PyTorch-specific garbage collection
            if self.has_gpu:
                import torch
                torch.cuda.empty_cache()
            
            # Python garbage collection
            import gc
            gc.collect()
        except Exception as e:
            logging.warning(f"Error during garbage collection: {e}")
    
    @contextmanager
    def use_gpu_tensor(self, tensor):
        """
        Context manager for safely using a GPU tensor.
        
        Args:
            tensor: PyTorch tensor to track
            
        Yields:
            The tensor for use in a with statement
            
        Example:
            ```
            with gpu_manager.use_gpu_tensor(torch.zeros(1000, 1000, device='cuda')) as tensor:
                # Use tensor here
            # Tensor is automatically freed when exiting the context
            ```
        """
        try:
            self.register_tensor(tensor)
            yield tensor
        finally:
            self.unregister_tensor(tensor)
    
    def register_tensor(self, tensor) -> None:
        """
        Register a GPU tensor for tracking.
        
        Args:
            tensor: PyTorch tensor to track
        """
        if self.has_gpu and tensor is not None and hasattr(tensor, 'device'):
            if tensor.device.type == 'cuda':
                self.gpu_tensors.add(tensor)
    
    def unregister_tensor(self, tensor) -> None:
        """
        Unregister a GPU tensor from tracking.
        
        Args:
            tensor: PyTorch tensor to unregister
        """
        if tensor in self.gpu_tensors:
            self.gpu_tensors.discard(tensor)

# Factory function that gets the instance from ResourceManager
def get_gpu_manager():
    """
    Get the GPUManager instance.
    
    This function is the recommended way to get the GPUManager instance.
    It delegates to ResourceManager for better integration with the rest of the application.
    
    Returns:
        GPUManager instance
    """
    # Import here to avoid circular imports
    from heightcraft.infrastructure.resource_manager import ResourceManager
    resource_manager = ResourceManager.get_instance()
    return resource_manager.create_gpu_manager()


# Global instance for backward compatibility
# New code should use get_gpu_manager() instead
gpu_manager = GPUManager.get_instance()
 