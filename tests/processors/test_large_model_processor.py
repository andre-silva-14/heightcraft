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
from multiprocessing.pool import ThreadPool
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
        self.processor.chunks = [{'vertices': 0, 'vertex_count': 1, 'faces': np.array([[0, 1, 2]]), 'vertex_offset': 0}]
        self.processor.vertex_buffer = [np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])]
        
        # Mock sampling service response
        mock_points = np.array([[0.1, 0.1, 0]])
        mock_point_cloud = PointCloud(mock_points)
        self.sampling_service.sample_points.return_value = mock_point_cloud
        
        # Run sampling
        points_generator = self.processor.sample_points()
        points = list(points_generator)
        
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
        # Verify
        self.assertIsNotNone(height_map)
        # generate_from_point_cloud is no longer called in LargeModelProcessor
        # Instead it calls create_height_map_buffer and update_height_map_buffer
        self.height_map_service.create_height_map_buffer.assert_called()
        self.height_map_service.update_height_map_buffer.assert_called()

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

    @patch('heightcraft.processors.large_model_processor.as_completed')
    @patch('heightcraft.processors.large_model_processor.ThreadPoolExecutor')
    @patch('heightcraft.processors.large_model_processor.trimesh.load')
    def test_large_model_processor_threading_gpu(self, mock_load, MockThreadPool, mock_as_completed):
        """Test that LargeModelProcessor uses max_workers=1 when use_gpu is True."""
        # Setup config with GPU enabled
        model_config = ModelConfig(file_path="dummy.obj", mode=ProcessingMode.LARGE, chunk_size=100)
        sampling_config = SamplingConfig(num_samples=100, use_gpu=True, num_threads=4) # Request 4 threads
        height_map_config = HeightMapConfig(max_resolution=128)
        output_config = OutputConfig(output_path="out.png")
        
        config = ApplicationConfig(
            model_config=model_config,
            sampling_config=sampling_config,
            height_map_config=height_map_config,
            output_config=output_config
        )
        
        processor = LargeModelProcessor(config)
        
        # Mock internal state to simulate having chunks
        processor.chunks = [{'vertices': 0, 'vertex_count': 1, 'faces': np.array([[0, 1, 2]]), 'vertex_offset': 0}]
        processor.vertex_buffer = [np.array([[0,0,0], [1,0,0], [0,1,0]])] # Valid vertices for faces [0, 1, 2]
        
        # Mock sampling service to avoid actual work
        processor.sampling_service = MagicMock()
        mock_points = MagicMock()
        mock_points.points = np.array([[0.1, 0.1, 0.1]])
        mock_points.size = 1
        processor.sampling_service.sample_points.return_value = mock_points

        # Configure as_completed to yield the mock future
        mock_future = MagicMock()
        mock_future.result.return_value = MagicMock(points=np.array([[0.1, 0.1, 0.1]]), size=1)
        MockThreadPool.return_value.submit.return_value = mock_future
        mock_as_completed.return_value = [mock_future]

        # Call the method that uses ThreadPool
        # We need to mock _sample_points_from_chunks or _sample_points_from_scene
        # Let's test _sample_points_from_chunks
        # Consume the generator to trigger execution
        list(processor._sample_points_from_chunks(100, True))
        
        # Verify ThreadPool was initialized with max_workers=1
        # The method creates a ThreadPool instance. We check the call args.
        # Note: ThreadPool might be called multiple times (e.g. in load_model), 
        # but we are interested in the call inside _sample_points_from_chunks.
        
        # Filter calls to find the one with max_workers=1
        found_sequential_call = False
        for call in MockThreadPool.call_args_list:
            if call.kwargs.get('max_workers') == 1:
                found_sequential_call = True
                break
        
        self.assertTrue(found_sequential_call, "ThreadPool should be initialized with max_workers=1 when use_gpu is True")

    @patch('heightcraft.processors.large_model_processor.as_completed')
    @patch('heightcraft.processors.large_model_processor.ThreadPoolExecutor')
    @patch('heightcraft.processors.large_model_processor.trimesh.load')
    def test_large_model_processor_threading_cpu(self, mock_load, MockThreadPool, mock_as_completed):
        """Test that LargeModelProcessor uses configured threads when use_gpu is False."""
        # Setup config with GPU disabled
        model_config = ModelConfig(file_path="dummy.obj", mode=ProcessingMode.LARGE, chunk_size=100)
        sampling_config = SamplingConfig(num_samples=100, use_gpu=False, num_threads=4) # Request 4 threads
        height_map_config = HeightMapConfig(max_resolution=128)
        output_config = OutputConfig(output_path="out.png")
        
        config = ApplicationConfig(
            model_config=model_config,
            sampling_config=sampling_config,
            height_map_config=height_map_config,
            output_config=output_config
        )
        
        processor = LargeModelProcessor(config)
        
        # Mock internal state
        processor.chunks = [{'vertices': 0, 'vertex_count': 1, 'faces': np.array([[0, 1, 2]]), 'vertex_offset': 0}]
        processor.vertex_buffer = [np.array([[0,0,0], [1,0,0], [0,1,0]])]
        processor.sampling_service = MagicMock()
        mock_points = MagicMock()
        mock_points.points = np.array([[0.1, 0.1, 0.1]])
        mock_points.size = 1
        processor.sampling_service.sample_points.return_value = mock_points

        # Configure as_completed to yield the mock future
        mock_future = MagicMock()
        mock_future.result.return_value = MagicMock(points=np.array([[0.1, 0.1, 0.1]]), size=1)
        MockThreadPool.return_value.submit.return_value = mock_future
        mock_as_completed.return_value = [mock_future]

        # Call the method
        # Consume the generator to trigger execution
        list(processor._sample_points_from_chunks(100, False))
        
        # Verify ThreadPool was initialized with max_workers=4
        found_parallel_call = False
        for call in MockThreadPool.call_args_list:
            if call.kwargs.get('max_workers') == 4:
                found_parallel_call = True
                break
        
        self.assertTrue(found_parallel_call, "ThreadPool should be initialized with max_workers=4 when use_gpu is False")

if __name__ == '__main__':
    unittest.main()