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