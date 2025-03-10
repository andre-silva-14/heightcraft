import logging
from contextlib import contextmanager

# Import torch conditionally
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class ResourceManager:
    def __init__(self):
        self.gpu_tensors = []
        self.has_gpu = (
            TORCH_AVAILABLE and hasattr(torch, "cuda") and torch.cuda.is_available()
        )

    def allocate_gpu_tensor(self, *args, **kwargs):
        """Allocate a GPU tensor and keep track of it."""
        if not self.has_gpu:
            raise RuntimeError("GPU not available")

        tensor = torch.tensor(*args, **kwargs, device="cuda")
        self.gpu_tensors.append(tensor)
        return tensor

    def clear_gpu_memory(self):
        """Clear all tracked GPU tensors."""
        if not self.has_gpu:
            return

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
        try:
            self.clear_gpu_memory()
        except Exception:
            # Ignore errors during cleanup in destructor
            pass


resource_manager = ResourceManager()
