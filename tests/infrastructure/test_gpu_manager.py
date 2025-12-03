import pytest
import sys
from unittest.mock import MagicMock, patch

from heightcraft.infrastructure.gpu_manager import GPUManager

class TestGPUManager:
    @pytest.fixture
    def mock_torch(self):
        with patch.dict(sys.modules, {'torch': MagicMock()}):
            yield sys.modules['torch']

    @pytest.fixture
    def configured_mock_torch(self, mock_torch):
        # Configure torch mock to behave like a real GPU environment
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1
        mock_torch.cuda.get_device_name.return_value = "Test GPU"
        
        mock_props = MagicMock()
        mock_props.total_memory = 1024**3
        mock_torch.cuda.get_device_properties.return_value = mock_props
        
        mock_torch.cuda.memory_reserved.return_value = 1000
        mock_torch.cuda.memory_allocated.return_value = 400
        mock_torch.cuda.current_device.return_value = 0
        
        return mock_torch

    def test_init_no_torch(self):
        with patch.dict(sys.modules, {'torch': None}):
            pass

    def test_init_cpu_only(self, mock_torch):
        mock_torch.cuda.is_available.return_value = False
        manager = GPUManager()
        manager.__init__()
        assert not manager.has_gpu

    def test_init_gpu_available(self, configured_mock_torch):
        with patch('logging.info'):
            manager = GPUManager()
            manager.__init__()
            assert manager.has_gpu

    def test_allocate_tensor_cpu(self, mock_torch):
        mock_torch.cuda.is_available.return_value = False
        manager = GPUManager()
        manager.__init__()
        
        data = [1, 2, 3]
        manager.allocate_tensor(data)
        
        mock_torch.tensor.assert_called_with(data, dtype=None)

    def test_allocate_tensor_gpu(self, configured_mock_torch):
        manager = GPUManager()
        
        with patch('logging.info'), patch('logging.warning'), patch('logging.error'):
            manager.__init__()
            
            data = [1, 2, 3]
            manager.allocate_tensor(data)
        
        configured_mock_torch.tensor.assert_called_with(data, device="cuda", dtype=None)
        assert len(manager.gpu_tensors) == 1

    def test_allocate_tensor_error_fallback(self, configured_mock_torch):
        manager = GPUManager()
        
        with patch('logging.info'), patch('logging.warning'), patch('logging.error'):
            manager.__init__()
            
            # First call raises exception
            configured_mock_torch.tensor.side_effect = [Exception("GPU Error"), MagicMock()]
            
            data = [1, 2, 3]
            manager.allocate_tensor(data)
        
        # Should have tried GPU then CPU
        assert configured_mock_torch.tensor.call_count == 2

    def test_clear_memory(self, configured_mock_torch):
        manager = GPUManager()
        
        with patch('logging.info'), patch('logging.warning'), patch('logging.error'):
            manager.__init__()
            
            # Add a mock tensor
            tensor = MagicMock()
            manager.gpu_tensors.add(tensor)
            
            manager.clear_memory()
        
        assert len(manager.gpu_tensors) == 0
        configured_mock_torch.cuda.empty_cache.assert_called()

    def test_get_free_memory(self, configured_mock_torch):
        manager = GPUManager()
        
        with patch('logging.info'), patch('logging.warning'), patch('logging.error'):
            manager.__init__()
            assert manager.get_free_memory() == 600

    def test_get_free_memory_no_gpu(self, mock_torch):
        mock_torch.cuda.is_available.return_value = False
        manager = GPUManager()
        with patch('logging.info'), patch('logging.warning'), patch('logging.error'):
            manager.__init__()
            assert manager.get_free_memory() == 0

    def test_use_gpu_tensor(self, configured_mock_torch):
        manager = GPUManager()
        
        with patch('logging.info'), patch('logging.warning'), patch('logging.error'):
            manager.__init__()
            
            tensor = MagicMock()
            tensor.device.type = 'cuda'
            
            with manager.use_gpu_tensor(tensor) as t:
                assert t == tensor
                assert tensor in manager.gpu_tensors
                
            assert tensor not in manager.gpu_tensors

    def test_register_unregister_tensor(self, configured_mock_torch):
        manager = GPUManager()
        
        with patch('logging.info'), patch('logging.warning'), patch('logging.error'):
            manager.__init__()
            
            tensor = MagicMock()
            tensor.device.type = 'cuda'
            
            manager.register_tensor(tensor)
            assert tensor in manager.gpu_tensors
            
            manager.unregister_tensor(tensor)
            assert tensor not in manager.gpu_tensors
