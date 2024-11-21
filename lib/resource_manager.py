import torch
import logging
from contextlib import contextmanager

class ResourceManager:
    def __init__(self):
        self.gpu_tensors = []

    def allocate_gpu_tensor(self, *args, **kwargs):
        """Allocate a GPU tensor and keep track of it."""
        tensor = torch.tensor(*args, **kwargs, device='cuda')
        self.gpu_tensors.append(tensor)
        return tensor

    def clear_gpu_memory(self):
        """Clear all tracked GPU tensors."""
        for tensor in self.gpu_tensors:
            del tensor
        self.gpu_tensors.clear()
        torch.cuda.empty_cache()
        logging.info("GPU memory cleared")

    @contextmanager
    def gpu_session(self):
        """Context manager for GPU sessions to ensure cleanup."""
        try:
            yield
        finally:
            self.clear_gpu_memory()

    def __del__(self):
        """Destructor to ensure cleanup if the object is deleted."""
        self.clear_gpu_memory()

resource_manager = ResourceManager()

