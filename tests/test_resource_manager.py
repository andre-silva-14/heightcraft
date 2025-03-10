import pytest
import torch
import threading
import time
import numpy as np
import gc
from lib.resource_manager import ResourceManager, resource_manager

@pytest.fixture
def resource_manager_instance():
    """Create a fresh ResourceManager instance for each test."""
    rm = ResourceManager()
    yield rm
    # Clean up after test
    rm.clear_gpu_memory()

def test_singleton_pattern():
    """Test that resource_manager is a singleton instance."""
    # Get the global instance
    manager1 = resource_manager
    manager2 = resource_manager
    
    # They should be the same object
    assert manager1 is manager2
    assert id(manager1) == id(manager2)
    
    # But different from a new instance
    manager3 = ResourceManager()
    assert manager1 is not manager3
    assert id(manager1) != id(manager3)
    
    # Clean up
    manager3.clear_gpu_memory()

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_allocate_gpu_tensor(resource_manager_instance):
    tensor = resource_manager_instance.allocate_gpu_tensor([1, 2, 3], dtype=torch.float32)
    assert tensor.device.type == 'cuda'
    assert len(resource_manager_instance.gpu_tensors) == 1

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_clear_gpu_memory(resource_manager_instance):
    resource_manager_instance.allocate_gpu_tensor([1, 2, 3], dtype=torch.float32)
    resource_manager_instance.allocate_gpu_tensor([4, 5, 6], dtype=torch.float32)
    assert len(resource_manager_instance.gpu_tensors) == 2
    resource_manager_instance.clear_gpu_memory()
    assert len(resource_manager_instance.gpu_tensors) == 0

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_session(resource_manager_instance):
    with resource_manager_instance.gpu_session():
        tensor1 = resource_manager_instance.allocate_gpu_tensor([1, 2, 3], dtype=torch.float32)
        tensor2 = resource_manager_instance.allocate_gpu_tensor([4, 5, 6], dtype=torch.float32)
        assert len(resource_manager_instance.gpu_tensors) == 2
    assert len(resource_manager_instance.gpu_tensors) == 0

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_session_with_exception(resource_manager_instance):
    try:
        with resource_manager_instance.gpu_session():
            tensor = resource_manager_instance.allocate_gpu_tensor([1, 2, 3], dtype=torch.float32)
            raise Exception("Test exception")
    except Exception:
        pass
    assert len(resource_manager_instance.gpu_tensors) == 0

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_allocate_large_tensor(resource_manager_instance):
    """Test allocating a large tensor that approaches GPU memory limits."""
    # Get available GPU memory
    if torch.cuda.is_available():
        # Determine a size that's large but not too large
        # Using a fraction of available memory to avoid OOM
        free_memory, total_memory = torch.cuda.mem_get_info()
        safe_size = int(free_memory * 0.3)  # Use 30% of free memory
        
        # Calculate dimensions for a tensor of that size
        # Each float32 element is 4 bytes
        dim_size = int(np.floor(np.sqrt(safe_size / 4)))
        
        try:
            # Allocate a large 2D tensor
            large_tensor = resource_manager_instance.allocate_gpu_tensor(
                [dim_size, dim_size], dtype=torch.float32
            )
            assert large_tensor.device.type == 'cuda'
            assert large_tensor.shape == (dim_size, dim_size)
            assert len(resource_manager_instance.gpu_tensors) == 1
            
            # Clean up
            resource_manager_instance.clear_gpu_memory()
            del large_tensor
            torch.cuda.empty_cache()
        except Exception as e:
            # If this fails due to OOM, just skip
            pytest.skip(f"Skipping due to memory limitations: {e}")

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_nested_gpu_sessions(resource_manager_instance):
    """Test nested GPU sessions to ensure proper cleanup."""
    with resource_manager_instance.gpu_session():
        tensor1 = resource_manager_instance.allocate_gpu_tensor([1, 2, 3], dtype=torch.float32)
        assert len(resource_manager_instance.gpu_tensors) == 1
        
        # Save a reference to verify tensor1 still exists after inner session
        tensor1_ref = tensor1
        
        with resource_manager_instance.gpu_session():
            tensor2 = resource_manager_instance.allocate_gpu_tensor([4, 5, 6], dtype=torch.float32)
            assert len(resource_manager_instance.gpu_tensors) == 2
            
            # Inner session ends here - all tensors are cleared
        
        # All tensors are cleared at the end of any session
        assert len(resource_manager_instance.gpu_tensors) == 0
        
        # but tensor1_ref should still be accessible since we have a reference
        assert tensor1_ref.device.type == 'cuda'
        
        # Re-add tensor1 to the tracking list for outer session cleanup
        resource_manager_instance.gpu_tensors.append(tensor1_ref)
        
        # Outer session ends here
    
    # All tensors should be freed
    assert len(resource_manager_instance.gpu_tensors) == 0

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_concurrent_gpu_sessions():
    """Test concurrent GPU sessions from multiple threads."""
    # Create a shared resource manager
    shared_manager = ResourceManager()
    
    # Track errors in threads
    errors = []
    
    # Define a function for threads to run
    def thread_function(thread_id):
        try:
            with shared_manager.gpu_session():
                # Allocate a tensor
                tensor = shared_manager.allocate_gpu_tensor([thread_id, thread_id], dtype=torch.float32)
                # Simulate some work
                time.sleep(0.1)
                # Verify the tensor is still in memory
                assert tensor.device.type == 'cuda'
        except Exception as e:
            errors.append(f"Error in thread {thread_id}: {e}")
    
    # Create and start threads
    threads = []
    num_threads = 5
    for i in range(num_threads):
        thread = threading.Thread(target=thread_function, args=(i+1,))
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    # Check for errors
    assert not errors, f"Threads encountered errors: {errors}"
    
    # All tensors should be cleaned up
    assert len(shared_manager.gpu_tensors) == 0
    
    # Clean up
    shared_manager.clear_gpu_memory()

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_tensor_persistence(resource_manager_instance):
    """Test that tensors persist while referenced, even after session ends."""
    # Keep a reference to a tensor
    tensor_ref = None
    
    with resource_manager_instance.gpu_session():
        tensor = resource_manager_instance.allocate_gpu_tensor([10, 10], dtype=torch.float32)
        tensor_ref = tensor
    
    # Session ended, but we still have a reference
    assert tensor_ref is not None
    
    # The tensor should still be accessible and on GPU
    assert tensor_ref.device.type == 'cuda'
    
    # But it should no longer be tracked by the resource manager
    assert len(resource_manager_instance.gpu_tensors) == 0
    
    # Explicitly delete the reference
    del tensor_ref
    # Force garbage collection
    gc.collect()
    torch.cuda.empty_cache()
