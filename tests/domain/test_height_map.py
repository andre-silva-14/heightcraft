"""
Tests for the HeightMap domain model.
"""

import os
import unittest
from unittest.mock import Mock, patch

import numpy as np

from heightcraft.core.exceptions import HeightMapValidationError
from heightcraft.domain.height_map import HeightMap
from tests.base_test_case import BaseTestCase


class TestHeightMap(BaseTestCase):
    """Tests for the HeightMap domain model."""
    
    def setUp(self) -> None:
        """Set up test environment."""
        super().setUp()
        
        # Create a simple test height map
        self.data = np.array([
            [0.0, 0.5, 1.0],
            [0.5, 0.5, 0.5],
            [1.0, 0.5, 0.0]
        ], dtype=np.float32)
        self.bit_depth = 8  # Use 8-bit depth for tests
        self.height_map = HeightMap(self.data, self.bit_depth)
    
    def test_initialization(self) -> None:
        """Test initialization."""
        # Test valid inputs
        height_map = HeightMap(self.data, bit_depth=8)
        self.assertIsInstance(height_map, HeightMap)
        self.assertIs(height_map.data, self.data)
        self.assertEqual(height_map.bit_depth, 8)
        
        # Test with 16-bit depth
        height_map = HeightMap(self.data, bit_depth=16)
        self.assertIsInstance(height_map, HeightMap)
        self.assertEqual(height_map.bit_depth, 16)
        
        # Test with invalid data type
        with self.assertRaises(HeightMapValidationError):
            HeightMap("not an array", bit_depth=8)
        
        # Test with 1D array
        with self.assertRaises(HeightMapValidationError):
            HeightMap(np.array([0, 1, 2]), bit_depth=8)
        
        # Test with 3D array
        with self.assertRaises(HeightMapValidationError):
            HeightMap(np.zeros((3, 3, 3)), bit_depth=8)
            
        # Test with invalid bit depth
        with self.assertRaises(HeightMapValidationError):
            HeightMap(self.data, bit_depth=10)
    
    def test_properties(self) -> None:
        """Test property accessors."""
        # Test height and width
        self.assertEqual(self.height_map.height, 3)
        self.assertEqual(self.height_map.width, 3)
        
        # Test min and max
        self.assertEqual(self.height_map.min, 0.0)
        self.assertEqual(self.height_map.max, 1.0)
        
        # Test aspect_ratio
        self.assertEqual(self.height_map.aspect_ratio, 1.0)
    
    def test_data_normalization(self) -> None:
        """Test that data is normalized to [0, 1]."""
        # Create a height map with values outside [0, 1]
        data = np.array([
            [-1.0, 0.0, 1.0],
            [2.0, 3.0, 4.0]
        ], dtype=np.float32)
        
        # Check that it's normalized
        height_map = HeightMap(data, bit_depth=8)
        self.assertEqual(height_map.min, 0.0)
        self.assertEqual(height_map.max, 1.0)
        
        # Check specific values
        np.testing.assert_array_almost_equal(
            height_map.data,
            np.array([
                [0.0, 0.2, 0.4],
                [0.6, 0.8, 1.0]
            ], dtype=np.float32),
            decimal=5
        )
    
    def test_to_mesh(self) -> None:
        """Test converting to a mesh."""
        # Convert to mesh
        mesh = self.height_map.to_mesh()
        
        # Check that we got a Mesh object
        self.assertIsInstance(mesh, object)
        
        # Check that the mesh has the expected dimensions
        self.assertEqual(mesh.vertex_count, 9)  # 3x3 grid
        self.assertEqual(mesh.face_count, 8)    # 2x2 grid of quads = 8 triangles
    
    def test_to_point_cloud(self) -> None:
        """Test converting to a point cloud."""
        # Convert to a point cloud
        point_cloud = self.height_map.to_point_cloud()
        
        # Check that we got a NumPy array
        self.assertIsInstance(point_cloud, np.ndarray)
        
        # Check that the array has the expected shape
        self.assertEqual(point_cloud.shape, (9, 3))
        
        # Test some specific points for correctness
        # Bottom-left corner
        np.testing.assert_array_almost_equal(
            point_cloud[0],
            np.array([0, 0, self.data[0, 0]])
        )
        
        # Top-right corner
        np.testing.assert_array_almost_equal(
            point_cloud[8],
            np.array([2, 2, self.data[2, 2]])
        )
    
    def test_resize(self) -> None:
        """Test resizing a height map."""
        # Resize the height map to a new shape
        new_shape = (5, 5)
        resized = self.height_map.resize(new_shape)
        
        # Check that we got a HeightMap object
        self.assertIsInstance(resized, HeightMap)
        
        # Check that the shape is as expected
        self.assertEqual(resized.height, 5)
        self.assertEqual(resized.width, 5)
        
        # Check that the bit depth is preserved
        self.assertEqual(resized.bit_depth, self.height_map.bit_depth)
        
        # Check specific corner values
        self.assertEqual(resized.data[0, 0], self.height_map.data[0, 0])
        self.assertEqual(resized.data[4, 4], self.height_map.data[2, 2])
    
    def test_crop(self) -> None:
        """Test cropping a height map."""
        # Crop the height map
        cropped = self.height_map.crop((1, 1), (3, 3))
        
        # Check that we got a HeightMap object
        self.assertIsInstance(cropped, HeightMap)
        
        # Check that the shape is as expected
        self.assertEqual(cropped.height, 2)
        self.assertEqual(cropped.width, 2)
        
        # Check that the bit depth is preserved
        self.assertEqual(cropped.bit_depth, self.height_map.bit_depth)
        
        # Check that the data is as expected
        np.testing.assert_array_equal(
            cropped.data,
            self.height_map.data[1:3, 1:3]
        )
    
    def test_to_dict(self) -> None:
        """Test converting the height map to a dictionary."""
        # Convert to a dictionary
        height_map_dict = self.height_map.to_dict()
        
        # Check that it's a dictionary
        self.assertIsInstance(height_map_dict, dict)
        
        # Check that it has the expected keys
        expected_keys = [
            "width",
            "height",
            "aspect_ratio",
            "min",
            "max",
            "bit_depth"
        ]
        for key in expected_keys:
            self.assertIn(key, height_map_dict)
        
        # Check that the values match the height map
        self.assertEqual(height_map_dict["width"], self.height_map.width)
        self.assertEqual(height_map_dict["height"], self.height_map.height)
        self.assertEqual(height_map_dict["aspect_ratio"], self.height_map.aspect_ratio)
        self.assertEqual(height_map_dict["min"], self.height_map.min)
        self.assertEqual(height_map_dict["max"], self.height_map.max)
        self.assertEqual(height_map_dict["bit_depth"], self.height_map.bit_depth)
    
    @patch("numpy.load")
    def test_from_file(self, mock_load) -> None:
        """Test loading a height map from a file."""
        # Set up the mock
        mock_load.return_value = self.data
        
        # Create a test filepath
        file_path = self.get_temp_path("test_height_map.npy")
        
        # Test with bit depth provided
        with self.assertWarns(DeprecationWarning):
            height_map = HeightMap.from_file(file_path, bit_depth=8)
        
        # Check the result
        self.assertIsInstance(height_map, HeightMap)
        self.assertEqual(height_map.bit_depth, 8)
        
        # Check that the mock was called
        mock_load.assert_called_with(file_path)
    
    @patch("numpy.load")
    def test_from_file_with_missing_bit_depth(self, mock_load) -> None:
        """Test loading a height map with missing bit depth."""
        # Set up the mock
        mock_load.return_value = self.data
        
        # Create a test filepath
        file_path = self.get_temp_path("test_height_map.npy")
        
        # Test without bit depth
        with self.assertRaises(HeightMapValidationError):
            with self.assertWarns(DeprecationWarning):
                HeightMap.from_file(file_path)
    
    @patch("numpy.save")
    def test_save(self, mock_save) -> None:
        """Test saving a height map to a file."""
        # Create a test filepath
        file_path = self.get_temp_path("test_height_map.npy")
        
        # Test the method
        self.height_map.save(file_path)
        
        # Check that numpy.save was called with the correct arguments
        mock_save.assert_called_once()
        self.assertEqual(mock_save.call_args[0][0], file_path)
        np.testing.assert_array_equal(mock_save.call_args[0][1], self.height_map.data) 