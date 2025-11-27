"""
Tests for specific bug fixes in services.
"""

import unittest
import numpy as np
from unittest.mock import MagicMock, patch

from heightcraft.services.upscaling_service import UpscalingService
from heightcraft.services.height_map_service import HeightMapService
from heightcraft.domain.height_map import HeightMap

class TestFixes(unittest.TestCase):
    """Test cases for bug fixes."""
    
    def setUp(self):
        self.upscaling_service = UpscalingService()
        self.height_map_service = HeightMapService()
        
    def test_upscaling_division_by_zero(self):
        """Test that upscaling handles flat height maps (max == min) without division by zero."""
        # Create a flat height map
        data = np.ones((10, 10)) * 5.0
        
        # Mock model
        self.upscaling_service.model = MagicMock()
        # The model returns a tensor
        import torch
        self.upscaling_service.model.return_value = torch.zeros((1, 1, 10, 10))
        
        # We need to mock parameters().device
        mock_param = MagicMock()
        mock_param.device = 'cpu'
        self.upscaling_service.model.parameters.return_value = iter([mock_param])

        # Call the method
        # It should not raise ZeroDivisionError or produce NaNs
        try:
            result = self.upscaling_service._upscale_with_model(data, 2, False)
            self.assertFalse(np.any(np.isnan(result)))
        except Exception as e:
            self.fail(f"Upscaling raised exception on flat data: {e}")

    def test_histogram_equalization_precision(self):
        """Test that histogram equalization runs on 16-bit data without error."""
        # Create a gradient height map
        data = np.linspace(0, 1000, 100).reshape(10, 10).astype(np.float32)
        height_map = HeightMap(data, 16)
        
        # Run equalization
        result = self.height_map_service.equalize_histogram(height_map)
        
        # Check result is valid
        self.assertIsNotNone(result)
        self.assertEqual(result.data.shape, data.shape)
        self.assertFalse(np.any(np.isnan(result.data)))
        
        # Check that we didn't lose too much precision (hard to test exactly, but we can check it's not just 256 values)
        # If it was 8-bit, unique values would be <= 256.
        # With 16-bit float input, we expect more unique values if the input had them.
        # Our input has 100 unique values.
        # If we had 10000 unique values, 8-bit binning would reduce it.
        
        # Let's try with more unique values
        data_large = np.linspace(0, 1, 1000).reshape(10, 100).astype(np.float32)
        height_map_large = HeightMap(data_large, 16)
        result_large = self.height_map_service.equalize_histogram(height_map_large)
        
        unique_input = len(np.unique(data_large))
        unique_output = len(np.unique(result_large.data))
        
        # In 8-bit quantization, unique_output would be at most 256.
        # In our high-precision version, it should be close to unique_input (1000).
        self.assertGreater(unique_output, 256, "Histogram equalization should preserve > 8-bit precision")

if __name__ == '__main__':
    unittest.main()
