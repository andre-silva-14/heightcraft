import pytest
import torch
from lib.resource_manager import ResourceManager

@pytest.fixture
def resource_manager():
    return ResourceManager()

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_allocate_gpu_tensor(resource_manager):
    tensor = resource_manager.allocate_gpu_tensor([1, 2, 3], dtype=torch.float32)
    assert tensor.device.type == 'cuda'
    assert len(resource_manager.gpu_tensors) == 1

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_clear_gpu_memory(resource_manager):
    resource_manager.allocate_gpu_tensor([1, 2, 3], dtype=torch.float32)
    resource_manager.allocate_gpu_tensor([4, 5, 6], dtype=torch.float32)
    assert len(resource_manager.gpu_tensors) == 2
    resource_manager.clear_gpu_memory()
    assert len(resource_manager.gpu_tensors) == 0

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_session(resource_manager):
    with resource_manager.gpu_session():
        tensor1 = resource_manager.allocate_gpu_tensor([1, 2, 3], dtype=torch.float32)
        tensor2 = resource_manager.allocate_gpu_tensor([4, 5, 6], dtype=torch.float32)
        assert len(resource_manager.gpu_tensors) == 2
    assert len(resource_manager.gpu_tensors) == 0

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_session_with_exception(resource_manager):
    try:
        with resource_manager.gpu_session():
            tensor = resource_manager.allocate_gpu_tensor([1, 2, 3], dtype=torch.float32)
            raise Exception("Test exception")
    except Exception:
        pass
    assert len(resource_manager.gpu_tensors) == 0
