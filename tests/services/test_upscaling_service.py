"""
Tests for the UpscalingService.
"""

import os
import unittest
from unittest.mock import Mock, patch, MagicMock

import numpy as np

from heightcraft.core.exceptions import UpscalingError
from heightcraft.domain.height_map import HeightMap
from heightcraft.services.upscaling_service import UpscalingService
from heightcraft.core.config import UpscaleConfig
from tests.base_test_case import BaseTestCase


class TestUpscalingService(BaseTestCase):
    """Tests for the UpscalingService."""
    
    def setUp(self) -> None:
        """Set up test environment."""
        super().setUp()
        
        # Create mock dependencies
        self.cache_manager = Mock()
        self.height_map_service = Mock()
        self.file_storage = Mock()
        
        # Create a config
        self.config = UpscaleConfig()
        
        # Create an instance of UpscalingService
        self.upscaling_service = UpscalingService(
            config=self.config,
            cache_manager=self.cache_manager,
            height_map_service=self.height_map_service,
            file_storage=self.file_storage
        )
        
        # Create a test height map
        self.data = np.array([
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8]
        ], dtype=np.float32)
        self.bit_depth = 8  # Using bit_depth instead of resolution
        self.test_height_map = HeightMap(self.data, self.bit_depth)
        
        # Test file paths
        self.test_input_file = self.get_temp_path("test_input.png")
        self.test_output_file = self.get_temp_path("test_output.png")
    
    @patch('heightcraft.services.upscaling_service.TF_AVAILABLE', True)
    @patch('heightcraft.services.upscaling_service.UpscalingService._upscale_with_interpolation')
    def test_upscale_height_map(self, mock_upscale_with_interpolation) -> None:
        """Test upscaling a height map."""
        # Create a mock upscaled data that is an actual numpy array
        mock_upscaled_data = np.ones((6, 6))
        mock_upscale_with_interpolation.return_value = mock_upscaled_data

        # Create a test height map with a real numpy array
        test_data = np.ones((3, 3))
        test_height_map = HeightMap(test_data, bit_depth=8)

        # Create a new config with pretrained_model set to False
        from dataclasses import replace
        mock_config = replace(self.config, pretrained_model=False)
        
        # Create a new upscaling service with the mock config
        upscaling_service = UpscalingService(mock_config, self.file_storage)
        
        # Call the method
        result = upscaling_service.upscale(test_height_map, scale_factor=2)

        # Check that the result is a height map
        self.assertIsInstance(result, HeightMap)
        
        # Check dimensions
        self.assertEqual(result.shape, (6, 6))
        
        # Check that the interpolation method was called once
        mock_upscale_with_interpolation.assert_called_once()
    
    @patch('heightcraft.services.upscaling_service.TF_AVAILABLE', True)
    @patch('heightcraft.services.upscaling_service.UpscalingService._upscale_with_interpolation')
    def test_upscale_height_map_with_dimensions(self, mock_upscale_with_interpolation) -> None:
        """Test upscaling a height map with specific dimensions."""
        # Create a mock upscaled data that is an actual numpy array
        mock_upscaled_data = np.ones((8, 8))
        mock_upscale_with_interpolation.return_value = mock_upscaled_data

        # Create a test height map with a real numpy array
        test_data = np.ones((4, 4))
        test_height_map = HeightMap(test_data, bit_depth=8)

        # Create a new config with pretrained_model set to False
        from dataclasses import replace
        mock_config = replace(self.config, pretrained_model=False)
        
        # Create a new upscaling service with the mock config
        upscaling_service = UpscalingService(mock_config, self.file_storage)
        
        # Call the method
        result = upscaling_service.upscale(test_height_map, scale_factor=2)

        # Check that the result is a height map
        self.assertIsInstance(result, HeightMap)
        
        # Check dimensions
        self.assertEqual(result.shape, (8, 8))
        
        # Check that the interpolation method was called once
        mock_upscale_with_interpolation.assert_called_once()
    
    def test_upscale_height_map_with_invalid_scale_factor(self) -> None:
        """Test upscaling a height map with an invalid scale factor."""
        # Call the method with a negative scale factor
        with self.assertRaises(UpscalingError):
            self.upscaling_service.upscale(self.test_height_map, scale_factor=-1)
        
        # Call the method with a zero scale factor
        with self.assertRaises(UpscalingError):
            self.upscaling_service.upscale(self.test_height_map, scale_factor=0)
    
    def test_upscale_height_map_with_invalid_dimensions(self) -> None:
        """Test upscaling a height map with invalid dimensions."""
        # Call the method with invalid scale factor (not 2, 3, or 4)
        with self.assertRaises(UpscalingError):
            self.upscaling_service.upscale(self.test_height_map, scale_factor=5)
    
    @patch('heightcraft.services.upscaling_service.TF_AVAILABLE', False)
    def test_upscale_without_tensorflow(self) -> None:
        """Test upscaling when TensorFlow is not available."""
        # Should work using interpolation instead of neural network
        result = self.upscaling_service.upscale(self.test_height_map, scale_factor=2)
        self.assertIsInstance(result, HeightMap)
    
    @patch('heightcraft.services.upscaling_service.TF_AVAILABLE', True)
    def test_upscale_from_file(self) -> None:
        """Test upscaling a height map from a file."""
        # Create a mock for the UpscalingService class
        with patch('heightcraft.services.upscaling_service.UpscalingService.upscale_file', return_value=True) as mock_upscale_file:
            # Call the method and check the result
            result = self.upscaling_service.upscale_file(self.test_input_file, self.test_output_file, scale_factor=2)
            
            # Check that the result is True (as we mocked the method to return True)
            self.assertTrue(result)
            
            # Verify that the upscale_file method was called
            mock_upscale_file.assert_called_once()
            
            # Check that the call arguments included the correct file paths and scale factor
            args, kwargs = mock_upscale_file.call_args
            self.assertEqual(args[0], self.test_input_file)
            self.assertEqual(args[1], self.test_output_file)
            self.assertEqual(kwargs.get('scale_factor'), 2)
    
    @patch('heightcraft.services.upscaling_service.TF_AVAILABLE', True)
    @patch('heightcraft.services.upscaling_service.tf')
    def test_upscale_from_file_with_error(self, mock_tf) -> None:
        """Test upscaling a height map file with an error."""
        # Mock file loading to raise an exception
        self.height_map_service.load_height_map.side_effect = Exception("Load error")
        
        # Call the method
        with self.assertRaises(UpscalingError):
            self.upscaling_service.upscale_file(self.test_input_file, self.test_output_file, scale_factor=2) 