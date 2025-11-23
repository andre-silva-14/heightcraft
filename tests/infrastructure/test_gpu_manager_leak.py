"""
Test for GPUManager memory leak.
"""

import unittest
import weakref
import gc
from unittest.mock import MagicMock, patch
import sys

# Mock torch before importing GPUManager if needed, but we can just rely on the internal logic
# GPUManager imports torch inside methods or conditionally.

from heightcraft.infrastructure.gpu_manager import GPUManager

class TestGPUManagerLeak(unittest.TestCase):
    """Test for GPUManager memory leak."""
    
    def setUp(self):
        # Reset singleton
        GPUManager._instance = None
        self.gpu_manager = GPUManager.get_instance()
        
    def test_allocate_tensor_leak(self):
        """Test that allocate_tensor doesn't leak."""
        # This test requires us to simulate the lifecycle
        
        class DummyTensor:
            pass
            
        tensor = DummyTensor()
        
        # Add to manager
        self.gpu_manager.gpu_tensors.add(tensor)
        
        # Verify it's there
        self.assertEqual(len(self.gpu_manager.gpu_tensors), 1)
        
        # Delete ref
        del tensor
        gc.collect()
        
        # Verify it's gone
        self.assertEqual(len(self.gpu_manager.gpu_tensors), 0)

if __name__ == '__main__':
    unittest.main()
