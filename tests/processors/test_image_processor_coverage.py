"""
Tests for ImageProcessor coverage improvement.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from heightcraft.processors.image_processor import ImageProcessor
from heightcraft.core.config import ApplicationConfig, ModelConfig, ProcessingMode
from heightcraft.core.exceptions import ProcessingError
from heightcraft.domain.height_map import HeightMap
from tests.base_test_case import BaseTestCase

class TestImageProcessorCoverage(BaseTestCase):
    """Tests for ImageProcessor coverage."""

    def setUp(self):
        super().setUp()
        self.config = MagicMock()
        self.config.model_config = MagicMock(spec=ModelConfig)
        self.config.model_config.file_path = "test.png"
        self.config.model_config.mode = ProcessingMode.IMAGE
        
        self.config.height_map_config = MagicMock()
        self.config.height_map_config.bit_depth = 16
        
        self.config.output_config = MagicMock()
        self.config.output_config.output_path = "output.png"
        
        self.config.upscale_config = MagicMock()
        self.config.upscale_config.enabled = True
        self.config.upscale_config.upscale_factor = 2

        self.processor = ImageProcessor(self.config)

    def test_load_model_no_path(self):
        self.config.model_config.file_path = None
        with self.assertRaises(ProcessingError):
            self.processor.load_model()

    @patch("heightcraft.processors.image_processor.HeightMap")
    def test_load_model_error(self, mock_hm_cls):
        mock_hm_cls.from_file.side_effect = Exception("Load error")
        with self.assertRaises(ProcessingError):
            self.processor.load_model()

    def test_upscale_height_map_no_map(self):
        self.processor.height_map_obj = None
        # Should log warning but not crash
        self.processor.upscale_height_map()

    def test_upscale_height_map_error(self):
        self.processor.height_map_obj = Mock(spec=HeightMap)
        with patch.object(self.processor.upscaling_service, 'upscale', side_effect=Exception("Upscale error")):
            with self.assertRaises(ProcessingError):
                self.processor.upscale_height_map()

    def test_save_height_map_no_map(self):
        self.processor.height_map_obj = None
        with self.assertRaises(ProcessingError):
            self.processor.save_height_map()

    @patch("heightcraft.processors.image_processor.HeightMap")
    @patch("heightcraft.processors.image_processor.HeightMapService")
    @patch("heightcraft.processors.image_processor.UpscalingService")
    def test_process_bit_depth_conversion_32(self, mock_upscale_cls, mock_service_cls, mock_hm_cls):
        # Setup mocks
        mock_upscale_service = mock_upscale_cls.return_value
        
        # Mock input height map
        mock_input_hm = MagicMock(spec=HeightMap)
        mock_input_hm.bit_depth = 8
        mock_hm_cls.from_file.return_value = mock_input_hm
        
        # Mock upscaled height map
        mock_upscaled_hm = MagicMock(spec=HeightMap)
        mock_upscaled_hm.bit_depth = 8
        mock_upscale_service.upscale.return_value = mock_upscaled_hm
        
        # Config target is 32-bit
        self.config.height_map_config.bit_depth = 32
        
        # Initialize processor
        processor = ImageProcessor(self.config)
        
        # Run process
        processor.process()
        
        # Verify 32-bit conversion (via constructor in this case as per implementation)
        # The implementation does: self.height_map_obj = HeightMap(self.height_map_obj.data, bit_depth=32)
        # So we check if HeightMap was called with bit_depth=32
        # Note: HeightMap is mocked, so we check the mock calls.
        # It's called once for from_file, and then again for 32-bit conversion
        
        # Filter calls to constructor (not from_file)
        constructor_calls = [call for call in mock_hm_cls.mock_calls if call[0] == '']
        # Expecting call with (data, bit_depth=32)
        # Since we can't easily check data equality on mock, we check kwargs
        found_32bit_call = False
        for call in constructor_calls:
            if call.kwargs.get('bit_depth') == 32:
                found_32bit_call = True
                break
        self.assertTrue(found_32bit_call)

    @patch("heightcraft.processors.image_processor.HeightMap")
    def test_process_error(self, mock_hm_cls):
        mock_hm_cls.from_file.side_effect = Exception("Process error")
        with self.assertRaises(ProcessingError):
            self.processor.process()

    def test_unused_methods(self):
        # Coverage for unused methods
        self.assertEqual(len(self.processor.sample_points()), 0)
        self.assertIsNone(self.processor.generate_height_map())
