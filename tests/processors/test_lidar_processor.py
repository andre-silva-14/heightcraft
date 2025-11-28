"""
Test LiDAR processor.
"""

import os
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from heightcraft.core.config import ApplicationConfig, ModelConfig, ProcessingMode
from heightcraft.processors.lidar_processor import LidarProcessor
from heightcraft.domain.point_cloud import PointCloud

class TestLidarProcessor(unittest.TestCase):
    
    def setUp(self):
        self.config = MagicMock(spec=ApplicationConfig)
        self.config.model_config = MagicMock(spec=ModelConfig)
        self.config.model_config.file_path = "test.las"
        self.config.model_config.mode = ProcessingMode.LIDAR
        
        self.config.height_map_config = MagicMock()
        self.config.height_map_config.max_resolution = 100
        self.config.height_map_config.bit_depth = 16
        
        self.config.sampling_config = MagicMock()
        self.config.sampling_config.num_threads = 1
        
        self.config.output_config = MagicMock()
        self.config.output_config.output_path = "output.png"
        
        self.config.upscale_config = MagicMock()
        self.config.upscale_config.enabled = False
        
    @patch("heightcraft.processors.lidar_processor.LidarRepository")
    @patch("heightcraft.processors.lidar_processor.HeightMapService")
    def test_process(self, mock_service_cls, mock_repo_cls):
        # Setup mocks
        mock_repo = mock_repo_cls.return_value
        mock_service = mock_service_cls.return_value
        
        # Mock bounds
        mock_repo.get_bounds.return_value = {
            "min_x": 0.0, "max_x": 100.0,
            "min_y": 0.0, "max_y": 100.0,
            "min_z": 0.0, "max_z": 50.0
        }
        
        # Mock chunk iterator
        points = np.random.rand(100, 3)
        # Ensure points are within bounds
        points[:, 0] *= 100
        points[:, 1] *= 100
        points[:, 2] *= 50
        
        mock_pc = PointCloud(points)
        # Mock iterator to yield one chunk
        mock_repo.get_chunk_iterator.return_value = iter([mock_pc])
        
        # Mock height map generation (not used in streaming, but service is instantiated)
        # In streaming mode, we create HeightMap directly in processor, 
        # but we still use service.save_height_map
        
        # Initialize processor
        processor = LidarProcessor(self.config)
        
        # Run process
        result = processor.process()
        
        # Verify calls
        mock_repo.get_bounds.assert_called_once_with("test.las")
        mock_repo.get_chunk_iterator.assert_called_once()
        mock_service.save_height_map.assert_called_once()
        
        self.assertEqual(result, "output.png")
