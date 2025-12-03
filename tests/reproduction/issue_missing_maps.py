
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import os
from heightcraft.core.config import ApplicationConfig, HeightMapConfig, OutputConfig, ModelConfig, SamplingConfig, UpscaleConfig
from heightcraft.processors.large_model_processor import LargeModelProcessor
from heightcraft.processors.lidar_processor import LidarProcessor
from heightcraft.domain.height_map import HeightMap

class TestMissingMaps(unittest.TestCase):
    def setUp(self):
        self.config = ApplicationConfig(
            model_config=ModelConfig(file_path="dummy.obj", chunk_size=100),
            height_map_config=HeightMapConfig(
                max_resolution=100,
                sea_level=0.5,
                slope_map=True,
                curvature_map=True
            ),
            output_config=OutputConfig(output_path="output.png"),
            sampling_config=SamplingConfig(num_samples=1000),
            upscale_config=UpscaleConfig()
        )

    def test_large_model_processor_saves_maps(self):
        processor = LargeModelProcessor(self.config)
        processor.height_map = np.zeros((100, 100), dtype=np.float32)
        processor.bounds = {"min_z": 0, "max_z": 100}
        
        # Mock the services
        processor.height_map_service = MagicMock()
        processor.height_map_service.apply_sea_level.return_value = (MagicMock(), MagicMock())
        processor.height_map_service.generate_slope_map.return_value = MagicMock()
        processor.height_map_service.generate_curvature_map.return_value = MagicMock()
        
        processor.save_height_map()
        
        # Check if methods were called
        processor.height_map_service.apply_sea_level.assert_called()
        processor.height_map_service.generate_slope_map.assert_called()
        processor.height_map_service.generate_curvature_map.assert_called()

    def test_lidar_processor_saves_maps(self):
        processor = LidarProcessor(self.config)
        processor.height_map_obj = HeightMap(np.zeros((100, 100), dtype=np.float32), 16)
        processor.bounds = {"min_z": 0, "max_z": 100}
        
        # Mock the services
        processor.height_map_service = MagicMock()
        processor.height_map_service.apply_sea_level.return_value = (MagicMock(), MagicMock())
        processor.height_map_service.generate_slope_map.return_value = MagicMock()
        processor.height_map_service.generate_curvature_map.return_value = MagicMock()
        
        processor.save_height_map()
        
        # Check if methods were called
        processor.height_map_service.apply_sea_level.assert_called()
        processor.height_map_service.generate_slope_map.assert_called()
        processor.height_map_service.generate_curvature_map.assert_called()

if __name__ == '__main__':
    unittest.main()
