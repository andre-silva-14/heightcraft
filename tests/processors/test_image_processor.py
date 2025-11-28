"""
Test Image processor.
"""

import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from heightcraft.core.config import ApplicationConfig, ModelConfig, ProcessingMode
from heightcraft.processors.image_processor import ImageProcessor
from heightcraft.domain.height_map import HeightMap

class TestImageProcessor(unittest.TestCase):
    
    def setUp(self):
        self.config = MagicMock(spec=ApplicationConfig)
        self.config.model_config = MagicMock(spec=ModelConfig)
        self.config.model_config.file_path = "test.png"
        self.config.model_config.mode = ProcessingMode.IMAGE
        
        self.config.height_map_config = MagicMock()
        self.config.height_map_config.bit_depth = 16
        
        self.config.output_config = MagicMock()
        self.config.output_config.output_path = "output.png"
        
        self.config.sampling_config = MagicMock()
        self.config.sampling_config.num_threads = 1
        
        self.config.upscale_config = MagicMock()
        self.config.upscale_config.enabled = True
        self.config.upscale_config.upscale_factor = 2
        
    @patch("heightcraft.processors.image_processor.HeightMap")
    @patch("heightcraft.processors.image_processor.HeightMapService")
    @patch("heightcraft.processors.image_processor.UpscalingService")
    def test_process_upscaling_enabled(self, mock_upscale_cls, mock_service_cls, mock_hm_cls):
        # Setup mocks
        mock_upscale_service = mock_upscale_cls.return_value
        mock_height_map_service = mock_service_cls.return_value
        
        # Mock input height map
        mock_input_hm = MagicMock(spec=HeightMap)
        mock_input_hm.data = np.zeros((10, 10))
        mock_input_hm.bit_depth = 8
        mock_hm_cls.from_file.return_value = mock_input_hm
        
        # Mock upscaled height map
        mock_upscaled_hm = MagicMock(spec=HeightMap)
        mock_upscaled_hm.data = np.zeros((20, 20))
        mock_upscaled_hm.bit_depth = 8
        mock_upscale_service.upscale.return_value = mock_upscaled_hm
        
        # Mock converted height map
        mock_converted_hm = MagicMock(spec=HeightMap)
        mock_converted_hm.data = np.zeros((20, 20))
        mock_converted_hm.bit_depth = 16
        mock_upscaled_hm.convert_to_16bit.return_value = mock_converted_hm
        
        # Initialize processor
        processor = ImageProcessor(self.config)
        
        # Run process
        result = processor.process()
        
        # Verify calls
        mock_hm_cls.from_file.assert_called_once_with("test.png", bit_depth=16)
        mock_upscale_service.upscale.assert_called_once_with(mock_input_hm)
        mock_upscaled_hm.convert_to_16bit.assert_called_once()
        mock_height_map_service.save_height_map.assert_called_once_with(mock_converted_hm, "output.png")
        
        self.assertEqual(result, "output.png")

    @patch("heightcraft.processors.image_processor.HeightMap")
    @patch("heightcraft.processors.image_processor.HeightMapService")
    @patch("heightcraft.processors.image_processor.UpscalingService")
    def test_process_upscaling_disabled(self, mock_upscale_cls, mock_service_cls, mock_hm_cls):
        # Disable upscaling
        self.config.upscale_config.enabled = False
        
        # Initialize processor
        processor = ImageProcessor(self.config)
        
        # Run process
        result = processor.process()
        
        # Verify calls
        mock_hm_cls.from_file.assert_not_called()
        mock_upscale_cls.return_value.upscale.assert_not_called()
        mock_service_cls.return_value.save_height_map.assert_not_called()
        
        self.assertEqual(result, "")

    @patch("heightcraft.processors.image_processor.HeightMap")
    @patch("heightcraft.processors.image_processor.HeightMapService")
    @patch("heightcraft.processors.image_processor.UpscalingService")
    def test_process_bit_depth_conversion(self, mock_upscale_cls, mock_service_cls, mock_hm_cls):
        # Setup mocks
        mock_upscale_service = mock_upscale_cls.return_value
        
        # Mock input height map (8-bit)
        mock_input_hm = MagicMock(spec=HeightMap)
        mock_input_hm.bit_depth = 8
        mock_hm_cls.from_file.return_value = mock_input_hm
        
        # Mock upscaled height map (still 8-bit)
        mock_upscaled_hm = MagicMock(spec=HeightMap)
        mock_upscaled_hm.bit_depth = 8
        mock_upscale_service.upscale.return_value = mock_upscaled_hm
        
        # Config target is 16-bit
        self.config.height_map_config.bit_depth = 16
        
        # Initialize processor
        processor = ImageProcessor(self.config)
        
        # Run process
        processor.process()
        
        # Verify conversion called
        mock_upscaled_hm.convert_to_16bit.assert_called_once()
