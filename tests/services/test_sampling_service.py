"""
Tests for the SamplingService.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock

import numpy as np
import trimesh

from heightcraft.core.config import SamplingConfig
from heightcraft.core.exceptions import SamplingError
from heightcraft.domain.mesh import Mesh
from heightcraft.domain.point_cloud import PointCloud
from heightcraft.services.sampling_service import SamplingService
from tests.base_test_case import BaseTestCase


class TestSamplingService(BaseTestCase):
    """Tests for the SamplingService."""
    
    def setUp(self) -> None:
        """Set up test environment."""
        super().setUp()
        
        # Create a sampling configuration
        self.config = SamplingConfig(
            num_samples=1000,
            use_gpu=False,
            num_threads=4
        )
        
        # Create an instance of SamplingService
        self.sampling_service = SamplingService(self.config)
        
        # Create a mock trimesh
        self.mock_trimesh = Mock(spec=trimesh.Trimesh)
        self.mock_trimesh.vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        self.mock_trimesh.faces = np.array([
            [0, 1, 2],
            [0, 2, 3],
            [0, 3, 1],
            [1, 3, 2]
        ])
        self.mock_trimesh.sample = Mock(return_value=np.array([
            [0.1, 0.1, 0.1],
            [0.2, 0.2, 0.2],
            [0.3, 0.3, 0.3]
        ]))
        # Add missing properties required by SamplingService
        self.mock_trimesh.area = 1.0  # Non-zero area
        self.mock_trimesh.is_watertight = True
        self.mock_trimesh.is_degenerate = np.array([False, False, False, False])  # One value per face
        
        # Create a test mesh
        self.test_mesh = Mock(spec=Mesh)
        self.test_mesh.mesh = self.mock_trimesh
        self.test_mesh.vertices = self.mock_trimesh.vertices
        self.test_mesh.faces = self.mock_trimesh.faces
    
    def test_sample_from_mesh_cpu(self) -> None:
        """Test sampling points from a mesh using CPU."""
        # Call the method
        result = self.sampling_service.sample_from_mesh(self.test_mesh)
        
        # Check that the trimesh sample method was called
        self.mock_trimesh.sample.assert_called_once_with(1000)
        
        # Check that the result is a numpy array
        self.assertIsInstance(result, np.ndarray)
        
        # Check that the result has the expected shape
        self.assertEqual(result.shape, (3, 3))
    
    @patch('heightcraft.services.sampling_service.gpu_manager')
    def test_sample_from_mesh_gpu(self, mock_gpu_manager) -> None:
        """Test sampling points from a mesh using GPU."""
        # Set up mocks
        mock_gpu_manager.has_gpu = True
        
        # Create a sampling configuration with GPU enabled
        config = SamplingConfig(
            num_samples=1000,
            use_gpu=True,
            num_threads=4
        )
        
        # Create an instance of SamplingService
        sampling_service = SamplingService(config)
        
        # Mock the _sample_points_gpu method
        sampling_service._sample_points_gpu = Mock(return_value=np.array([
            [0.1, 0.1, 0.1],
            [0.2, 0.2, 0.2],
            [0.3, 0.3, 0.3]
        ]))
        
        # Call the method
        result = sampling_service.sample_from_mesh(self.test_mesh)
        
        # Check that the GPU sampling method was called
        sampling_service._sample_points_gpu.assert_called_once_with(self.test_mesh, 1000)
        
        # Check that the result is a numpy array
        self.assertIsInstance(result, np.ndarray)
        
        # Check that the result has the expected shape
        self.assertEqual(result.shape, (3, 3))
    
    @patch('heightcraft.services.sampling_service.gpu_manager')
    def test_sample_from_mesh_gpu_fallback(self, mock_gpu_manager) -> None:
        """Test sampling points from a mesh with GPU fallback to CPU."""
        # Set up mocks
        mock_gpu_manager.has_gpu = False
        
        # Create a sampling configuration with GPU enabled
        config = SamplingConfig(
            num_samples=1000,
            use_gpu=True,
            num_threads=4
        )
        
        # Create an instance of SamplingService
        sampling_service = SamplingService(config)
        
        # Call the method
        result = sampling_service.sample_from_mesh(self.test_mesh)
        
        # Check that the trimesh sample method was called (fallback to CPU)
        self.mock_trimesh.sample.assert_called_once_with(1000)
        
        # Check that the result is a numpy array
        self.assertIsInstance(result, np.ndarray)
        
        # Check that the result has the expected shape
        self.assertEqual(result.shape, (3, 3))
    
    def test_sample_with_threads(self) -> None:
        """Test sampling points using multiple threads."""
        # Mock the _sample_points_cpu method
        self.sampling_service._sample_points_cpu = Mock(return_value=np.array([
            [0.1, 0.1, 0.1],
            [0.2, 0.2, 0.2],
            [0.3, 0.3, 0.3]
        ]))
        
        # Call the method
        result = self.sampling_service.sample_with_threads(self.test_mesh, 1000, 4)
        
        # Check that the CPU sampling method was called multiple times
        self.assertEqual(self.sampling_service._sample_points_cpu.call_count, 4)
        
        # Check that the result is a numpy array
        self.assertIsInstance(result, np.ndarray)
    
    def test_sample_from_mesh_error(self) -> None:
        """Test handling errors when sampling points."""
        # Set up mock to raise an exception
        self.mock_trimesh.sample.side_effect = Exception("Sampling error")
        
        # Call the method and check for exception
        with self.assertRaises(SamplingError):
            self.sampling_service.sample_from_mesh(self.test_mesh)

    def test_sample_points_deterministic(self) -> None:
        """Test that sampling is deterministic with the same seed."""
        # Setup mock to return different values on subsequent calls unless seeded
        # Note: Since we are mocking trimesh, we can't easily test the actual numpy random seeding effect
        # on trimesh.sample. However, we can verify that our service sets the seed.
        
        with patch('numpy.random.seed') as mock_seed:
            self.sampling_service.sample_points(self.test_mesh, 100, use_gpu=False, seed=42)
            mock_seed.assert_called_with(42)

    def test_sample_extreme_num_points(self) -> None:
        """Test sampling an extreme number of points."""
        # Test with a very large number of points
        large_num = 10000
        self.sampling_service.sample_points(self.test_mesh, large_num, use_gpu=False)
        self.mock_trimesh.sample.assert_called_with(large_num)
        
        # Test with a very small number of points
        small_num = 1
        self.sampling_service.sample_points(self.test_mesh, small_num, use_gpu=False)
        self.mock_trimesh.sample.assert_called_with(small_num)

    def test_sample_zero_area_mesh(self) -> None:
        """Test sampling from a mesh with zero area faces."""
        # Create a mesh with zero area
        self.mock_trimesh.area = 0.0
        
        # Try to sample points
        with self.assertRaises(SamplingError):
            self.sampling_service.sample_points(self.test_mesh, 10, use_gpu=False)

    def test_sample_with_threads_worker_exception(self) -> None:
        """Test handling of exceptions in worker threads."""
        # Mock _sample_points_cpu to raise an exception
        self.sampling_service._sample_points_cpu = Mock(side_effect=Exception("Worker failed"))
        
        with self.assertRaises(SamplingError):
            self.sampling_service.sample_with_threads(self.test_mesh, 1000, 4)

    def test_sample_with_threads_empty_result(self) -> None:
        """Test handling of empty results from threads."""
        # Mock _sample_points_cpu to return empty array
        self.sampling_service._sample_points_cpu = Mock(return_value=np.array([]))
        
        with self.assertRaises(SamplingError):
            self.sampling_service.sample_with_threads(self.test_mesh, 1000, 4)

    @patch.dict('sys.modules', {'torch': None})
    def test_gpu_sampling_no_torch(self) -> None:
        """Test GPU sampling fallback when torch is not available."""
        # Ensure has_gpu is True so it tries to enter the GPU block
        with patch('heightcraft.services.sampling_service.gpu_manager') as mock_gpu:
            mock_gpu.has_gpu = True
            
            # Mock CPU sampling to verify fallback
            self.sampling_service._sample_points_cpu = Mock(return_value=np.array([[1, 2, 3]]))
            
            # This should trigger the ImportError inside _sample_points_gpu
            result = self.sampling_service._sample_points_gpu(self.test_mesh, 10)
            
            self.sampling_service._sample_points_cpu.assert_called()
            np.testing.assert_array_equal(result, np.array([[1, 2, 3]])) 