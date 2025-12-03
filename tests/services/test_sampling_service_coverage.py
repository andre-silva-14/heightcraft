"""
Tests for SamplingService coverage improvement.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from heightcraft.services.sampling_service import SamplingService
from heightcraft.core.config import SamplingConfig
from heightcraft.core.exceptions import SamplingError
from heightcraft.domain.mesh import Mesh
from tests.base_test_case import BaseTestCase

class TestSamplingServiceCoverage(BaseTestCase):
    """Tests for SamplingService coverage."""

    def setUp(self):
        super().setUp()
        self.config = SamplingConfig(
            num_samples=100,
            use_gpu=True,
            num_threads=4
        )
        self.sampling_service = SamplingService(self.config)
        self.mock_mesh = Mock(spec=Mesh)
        self.mock_mesh.vertices = np.array([[0,0,0], [1,0,0], [0,1,0]])
        self.mock_mesh.faces = np.array([[0,1,2]])

    @patch('heightcraft.services.sampling_service.gpu_manager')
    @patch.dict('sys.modules', {'torch': MagicMock()})
    def test_sample_points_gpu_logic(self, mock_gpu_manager):
        # Test the internal logic of _sample_points_gpu by mocking torch
        import sys
        mock_torch = sys.modules['torch']
        
        # Setup torch mocks
        mock_torch.tensor.return_value = MagicMock()
        mock_torch.float32 = 'float32'
        mock_torch.int64 = 'int64'
        mock_torch.norm.return_value = MagicMock()
        mock_torch.cross.return_value = MagicMock()
        mock_torch.sum.return_value = MagicMock()
        mock_torch.multinomial.return_value = MagicMock()
        mock_torch.sqrt.return_value = MagicMock()
        mock_torch.rand.return_value = MagicMock()
        mock_torch.clamp.return_value = MagicMock()
        
        # Mock tensor operations
        mock_tensor = MagicMock()
        mock_torch.tensor.return_value = mock_tensor
        mock_tensor.__getitem__.return_value = mock_tensor
        mock_tensor.__sub__.return_value = mock_tensor
        mock_tensor.__rsub__.return_value = mock_tensor
        mock_tensor.__mul__.return_value = mock_tensor
        mock_tensor.__add__.return_value = mock_tensor
        mock_tensor.unsqueeze.return_value = mock_tensor
        
        # Ensure torch functions return the mock tensor
        mock_torch.sqrt.return_value = mock_tensor
        mock_torch.rand.return_value = mock_tensor
        mock_torch.clamp.return_value = mock_tensor
        
        # Mock result
        mock_tensor.cpu.return_value.numpy.return_value = np.zeros((100, 3))
        
        # Call method
        result = self.sampling_service._sample_points_gpu(self.mock_mesh, 100)
        
        # Verify
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (100, 3))

    def test_sample_with_threads_timeout(self):
        # Mock ThreadPoolExecutor to simulate timeout
        # We need to patch where it is imported
        with patch('heightcraft.services.sampling_service.ThreadPoolExecutor') as MockExecutor:
            mock_executor = MockExecutor.return_value
            mock_executor.__enter__.return_value = mock_executor
            
            mock_future = Mock()
            mock_future.result.side_effect = TimeoutError()
            mock_executor.submit.return_value = mock_future
            
            # We need as_completed to yield our mock future
            with patch('heightcraft.services.sampling_service.as_completed', return_value=[mock_future]):
                # Mock mesh validation
                self.mock_mesh.mesh.area = 1.0
                
                with self.assertRaises(SamplingError) as cm:
                    self.sampling_service.sample_with_threads(self.mock_mesh, 100, 2)
                
                self.assertIn("timed out", str(cm.exception))

    def test_sample_with_threads_invalid_args(self):
        with self.assertRaises(SamplingError):
            self.sampling_service.sample_with_threads(self.mock_mesh, 100, 0)
        
        with self.assertRaises(SamplingError):
            self.sampling_service.sample_with_threads(self.mock_mesh, 0, 4)

    def test_sample_points_cpu_fallback(self):
        # Force CPU fallback by raising exception in GPU sampling logic (e.g. torch import or usage)
        # We patch torch to raise ImportError to simulate fallback
        with patch.dict('sys.modules', {'torch': None}):
            with patch.object(self.sampling_service, '_sample_points_cpu', return_value=np.zeros((10, 3))) as mock_cpu:
                # Ensure has_gpu is True to attempt GPU first
                with patch('heightcraft.services.sampling_service.gpu_manager') as mock_gpu:
                    mock_gpu.has_gpu = True
                    
                    # This will try to import torch, fail, and fallback
                    result = self.sampling_service.sample_points(self.mock_mesh, 10, use_gpu=True)
                    
                    mock_cpu.assert_called_once()
                    self.assertEqual(len(result.points), 10)
