"""
Unit tests for LargeModelProcessor.
"""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, Mock, PropertyMock, patch

import numpy as np
import trimesh

from heightcraft.core.config import ApplicationConfig, ModelConfig, SamplingConfig, HeightMapConfig, OutputConfig, ProcessingMode
from heightcraft.domain.mesh import Mesh
from heightcraft.domain.point_cloud import PointCloud
from heightcraft.processors.large_model_processor import LargeModelProcessor
from heightcraft.services.mesh_service import MeshService
from heightcraft.services.model_service import ModelService
from heightcraft.services.height_map_service import HeightMapService
from heightcraft.services.sampling_service import SamplingService

class TestLargeModelProcessor(unittest.TestCase):
    """Test cases for LargeModelProcessor."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary directory
        self.test_dir = tempfile.mkdtemp()
        self.test_mesh_file = os.path.join(self.test_dir, "test_large_model.obj")
        
        # Create a dummy mesh file
        with open(self.test_mesh_file, "w") as f:
            f.write("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3")
            
        # Configure application
        model_config = ModelConfig(
            file_path=self.test_mesh_file,
            mode=ProcessingMode.LARGE,
            chunk_size=100
        )
        sampling_config = SamplingConfig(
            num_samples=100,
            use_gpu=False
        )
        height_map_config = HeightMapConfig(
            max_resolution=128,
            bit_depth=16
        )
        output_config = OutputConfig(
            output_path=os.path.join(self.test_dir, "output.png")
        )
        
        self.config = ApplicationConfig(
            model_config=model_config,
            sampling_config=sampling_config,
            height_map_config=height_map_config,
            output_config=output_config
        )
        
        # Patch services at the class level (where they are imported in LargeModelProcessor)
        # We need to patch the classes so that when LargeModelProcessor instantiates them, it gets our mocks
        self.mesh_service_patcher = patch('heightcraft.processors.large_model_processor.MeshService')
        self.model_service_patcher = patch('heightcraft.processors.large_model_processor.ModelService')
        self.height_map_service_patcher = patch('heightcraft.processors.large_model_processor.HeightMapService')
        self.sampling_service_patcher = patch('heightcraft.processors.large_model_processor.SamplingService')
        
        self.MockMeshService = self.mesh_service_patcher.start()
        self.MockModelService = self.model_service_patcher.start()
        self.MockHeightMapService = self.height_map_service_patcher.start()
        self.MockSamplingService = self.sampling_service_patcher.start()
        
        # Setup mock instances
        self.mesh_service = self.MockMeshService.return_value
        self.model_service = self.MockModelService.return_value
        self.height_map_service = self.MockHeightMapService.return_value
        self.sampling_service = self.MockSamplingService.return_value
        
        # Initialize processor
        self.processor = LargeModelProcessor(self.config)
        
        # Mock internal methods to avoid complex logic in unit tests
        # We want to test the orchestration, not the trimesh logic (which is tested in integration)
        # But for some tests we might want to let it run.
        
    def tearDown(self):
        """Clean up."""
        self.mesh_service_patcher.stop()
        self.model_service_patcher.stop()
        self.height_map_service_patcher.stop()
        self.sampling_service_patcher.stop()
        
        import shutil
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_initialization(self):
        """Test processor initialization."""
        self.assertIsInstance(self.processor, LargeModelProcessor)
        self.assertEqual(self.processor.chunk_size, 100)
        
    @patch('heightcraft.processors.large_model_processor.trimesh.load')
    def test_load_model(self, mock_load):
        """Test loading model."""
        # Mock trimesh load
        mock_mesh = MagicMock()
        mock_mesh.vertices = np.zeros((10, 3))
        mock_mesh.faces = np.zeros((5, 3))
        mock_load.return_value = mock_mesh
        
        self.processor.load_model()
        
        mock_load.assert_called_once_with(self.test_mesh_file, process=False)
        self.assertTrue(len(self.processor.chunks) > 0)

    def test_sample_points(self):
        """Test sampling points."""
        # Setup processor state
        self.processor.chunks = [{'vertices': 0, 'vertex_count': 1, 'faces': np.array([[0, 1, 2]])}]
        self.processor.vertex_buffer = [np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])]
        
        # Mock sampling service response
        mock_points = np.array([[0.1, 0.1, 0]])
        mock_point_cloud = PointCloud(mock_points)
        self.sampling_service.sample_points.return_value = mock_point_cloud
        
        # Run sampling
        points = self.processor.sample_points()
        
        # Verify
        self.assertTrue(len(points) > 0)
        self.sampling_service.sample_points.assert_called()

    def test_generate_height_map(self):
        """Test height map generation."""
        # Setup processor state
        self.processor.points = np.random.rand(100, 3)
        
        # Mock height map service response
        mock_height_map_obj = MagicMock()
        mock_height_map_obj.data = np.zeros((128, 128))
        self.height_map_service.generate_from_point_cloud.return_value = mock_height_map_obj
        
        # Run generation
        height_map = self.processor.generate_height_map()
        
        # Verify
        self.assertIsNotNone(height_map)
        self.height_map_service.generate_from_point_cloud.assert_called()

    def test_save_height_map(self):
        """Test saving height map."""
        # Setup processor state
        self.processor.height_map = np.zeros((128, 128))
        
        # Mock save
        self.height_map_service.save_height_map.return_value = "output.png"
        
        # Run save
        path = self.processor.save_height_map()
        
        # Verify
        self.assertEqual(path, "output.png")
        self.height_map_service.save_height_map.assert_called()

if __name__ == '__main__':
    unittest.main()