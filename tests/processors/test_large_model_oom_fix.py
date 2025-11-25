
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from heightcraft.core.config import ApplicationConfig, ModelConfig, SamplingConfig, HeightMapConfig, OutputConfig, ProcessingMode
from heightcraft.processors.large_model_processor import LargeModelProcessor

class TestLargeModelOOMFix(unittest.TestCase):
    def test_config_use_gpu_flag(self):
        """Test that ApplicationConfig correctly reads use_gpu flag."""
        # Case 1: use_gpu=True
        args_gpu = {
            'file_path': 'dummy.obj',
            'use_gpu': True,
            'no_gpu': False # This key shouldn't matter with the fix, but simulating CLI args
        }
        config_gpu = ApplicationConfig.from_dict(args_gpu)
        self.assertTrue(config_gpu.sampling_config.use_gpu, "use_gpu should be True when passed as True")

        # Case 2: use_gpu=False
        args_cpu = {
            'file_path': 'dummy.obj',
            'use_gpu': False
        }
        config_cpu = ApplicationConfig.from_dict(args_cpu)
        self.assertFalse(config_cpu.sampling_config.use_gpu, "use_gpu should be False when passed as False")
        
        # Case 3: Default (missing key) -> False (as per get('use_gpu', False))
        args_default = {
            'file_path': 'dummy.obj'
        }
        config_default = ApplicationConfig.from_dict(args_default)
        self.assertFalse(config_default.sampling_config.use_gpu, "use_gpu should default to False")

    @patch('heightcraft.processors.large_model_processor.ThreadPool')
    @patch('heightcraft.processors.large_model_processor.trimesh.load')
    def test_large_model_processor_threading_gpu(self, mock_load, MockThreadPool):
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
        processor.chunks = [{'vertices': 0, 'vertex_count': 1, 'faces': np.array([[0, 1, 2]])}]
        processor.vertex_buffer = [np.array([[0,0,0], [1,0,0], [0,1,0]])] # Valid vertices for faces [0, 1, 2]
        
        # Mock sampling service to avoid actual work
        processor.sampling_service = MagicMock()
        processor.sampling_service.sample_points.return_value = MagicMock(points=np.array([[0.1, 0.1, 0.1]]))

        # Call the method that uses ThreadPool
        # We need to mock _sample_points_from_chunks or _sample_points_from_scene
        # Let's test _sample_points_from_chunks
        processor._sample_points_from_chunks(100, True)
        
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

    @patch('heightcraft.processors.large_model_processor.ThreadPool')
    @patch('heightcraft.processors.large_model_processor.trimesh.load')
    def test_large_model_processor_threading_cpu(self, mock_load, MockThreadPool):
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
        processor.chunks = [{'vertices': 0, 'vertex_count': 1, 'faces': np.array([[0, 1, 2]])}]
        processor.vertex_buffer = [np.array([[0,0,0], [1,0,0], [0,1,0]])]
        processor.sampling_service = MagicMock()
        processor.sampling_service.sample_points.return_value = MagicMock(points=np.array([[0.1, 0.1, 0.1]]))

        # Call the method
        processor._sample_points_from_chunks(100, False)
        
        # Verify ThreadPool was initialized with max_workers=4
        found_parallel_call = False
        for call in MockThreadPool.call_args_list:
            if call.kwargs.get('max_workers') == 4:
                found_parallel_call = True
                break
        
        self.assertTrue(found_parallel_call, "ThreadPool should be initialized with max_workers=4 when use_gpu is False")

if __name__ == '__main__':
    unittest.main()
