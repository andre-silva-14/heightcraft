import numpy as np
import pytest
from heightcraft.core.exceptions import ValidationError
from heightcraft.utils import converters

class TestConverters:
    
    def test_normalize_array(self):
        # Test basic normalization
        arr = np.array([0, 50, 100], dtype=np.float32)
        norm = converters.normalize_array(arr, 0.0, 1.0)
        np.testing.assert_array_equal(norm, np.array([0.0, 0.5, 1.0], dtype=np.float32))
        
        # Test with custom source range
        norm = converters.normalize_array(arr, 0.0, 1.0, source_min=0, source_max=200)
        np.testing.assert_array_equal(norm, np.array([0.0, 0.25, 0.5], dtype=np.float32))
        
        # Test constant array
        arr_const = np.array([10, 10, 10], dtype=np.float32)
        norm = converters.normalize_array(arr_const, 0.0, 1.0)
        np.testing.assert_array_equal(norm, np.array([0.5, 0.5, 0.5], dtype=np.float32))

    def test_convert_dtype(self):
        # Test float to uint8
        arr = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        conv = converters.convert_dtype(arr, np.uint8, normalize=True)
        np.testing.assert_array_equal(conv, np.array([0, 127, 255], dtype=np.uint8))
        
        # Test no normalization
        arr = np.array([0, 128, 255], dtype=np.float32)
        conv = converters.convert_dtype(arr, np.uint8, normalize=False)
        np.testing.assert_array_equal(conv, np.array([0, 128, 255], dtype=np.uint8))

    def test_bit_depth_to_dtype(self):
        assert converters.bit_depth_to_dtype(8) == np.dtype(np.uint8)
        assert converters.bit_depth_to_dtype(16) == np.dtype(np.uint16)
        assert converters.bit_depth_to_dtype(32) == np.dtype(np.float32)
        
        with pytest.raises(ValidationError):
            converters.bit_depth_to_dtype(24)

    def test_convert_bit_depth(self):
        arr = np.array([0.0, 1.0], dtype=np.float32)
        
        # To 8-bit
        conv8 = converters.convert_bit_depth(arr, 8)
        np.testing.assert_array_equal(conv8, np.array([0, 255], dtype=np.uint8))
        
        # To 16-bit
        conv16 = converters.convert_bit_depth(arr, 16)
        np.testing.assert_array_equal(conv16, np.array([0, 65535], dtype=np.uint16))

    def test_int_to_uint(self):
        # Test int8 to uint8
        arr = np.array([-128, 0, 127], dtype=np.int8)
        conv = converters.int_to_uint(arr)
        # -128 -> 0, 127 -> 255
        np.testing.assert_array_equal(conv, np.array([0, 128, 255], dtype=np.uint8))
        
        # Test already uint
        arr_uint = np.array([0, 255], dtype=np.uint8)
        assert converters.int_to_uint(arr_uint) is arr_uint

    def test_ensure_range(self):
        arr = np.array([-1.0, 0.5, 2.0])
        clipped = converters.ensure_range(arr, 0.0, 1.0)
        np.testing.assert_array_equal(clipped, np.array([0.0, 0.5, 1.0]))

    def test_ensure_shape(self):
        arr = np.zeros((10, 10))
        
        # Same shape
        assert converters.ensure_shape(arr, (10, 10)) is arr
        
        # Crop
        cropped = converters.ensure_shape(arr, (5, 5))
        assert cropped.shape == (5, 5)
        
        # Pad
        padded = converters.ensure_shape(arr, (15, 15))
        assert padded.shape == (15, 15)
        
        # Mixed
        mixed = converters.ensure_shape(arr, (5, 15))
        assert mixed.shape == (5, 15)
