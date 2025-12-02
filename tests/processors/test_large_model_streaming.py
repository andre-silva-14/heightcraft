
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import trimesh
import os
import logging
import sys

from heightcraft.core.config import ApplicationConfig, ModelConfig, SamplingConfig
from heightcraft.processors.large_model_processor import LargeModelProcessor
from heightcraft.domain.point_cloud import PointCloud

class TestLargeModelStreaming(unittest.TestCase):
    """Test cases for LargeModelProcessor streaming functionality."""
    def setUp(self):
        # Configure logging to stderr to avoid unittest capture
        logging.basicConfig(stream=sys.stderr, level=logging.DEBUG, force=True)
        self.test_file = "test_model_oom.glb"
        # Create a simple box mesh
        mesh = trimesh.creation.box()
        mesh.export(self.test_file)
        
        # Config
        config_dict = {
            'file_path': self.test_file,
            'chunk_size': 100000,
            'num_samples': 100,
            'num_threads': 1,
            'large_model': True
        }
        self.config = ApplicationConfig.from_dict(config_dict)
        
    def tearDown(self):
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
            
    @patch('heightcraft.domain.point_cloud.PointCloud.merge')
    def test_incremental_processing_avoids_merge(self, mock_merge):
        """
        Test that LargeModelProcessor processes chunks incrementally and avoids 
        merging all points into a single massive PointCloud (which causes OOM).
        """
        # Setup processor
        processor = LargeModelProcessor(self.config)
        processor.load_model()
        
        # Mock SamplingService.sample_points to return a dummy PointCloud
        # We need to make sure the mock is used.
        # Since ThreadPool might be tricky with mocks, we'll patch the class method if needed,
        # but let's try patching the instance method first.
        
        with patch.object(processor.sampling_service, 'sample_points') as mock_sample:
            dummy_points = np.zeros((100, 3))
            mock_sample.return_value = PointCloud(dummy_points)
            
            # We also need to mock generate_height_map or the internal method that consumes points
            # because if we change sample_points to NOT return points, generate_height_map will fail.
            # But for now, let's just see if we can run sample_points and check for merge.
            
            # Note: The current implementation of sample_points calls _sample_points_from_chunks
            # which returns a merged array.
            
            points = processor.sample_points()
            
            # Consume the generator
            points_list = list(points)
            total_points = sum(len(chunk) for chunk in points_list)
            
            self.assertEqual(total_points, 100, "Should return 100 points total")
            
            # This assertion confirms that merge was NOT called
            mock_merge.assert_not_called()

if __name__ == "__main__":
    unittest.main()
