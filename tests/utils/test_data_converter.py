"""
Tests for the DataConverter utility.
"""

import os
import unittest
from unittest.mock import Mock, patch

import numpy as np

from heightcraft.core.exceptions import ConversionError
from heightcraft.utils.data_converter import DataConverter
from tests.base_test_case import BaseTestCase


class TestDataConverter(BaseTestCase):
    """Tests for the DataConverter utility."""
    
    def setUp(self) -> None:
        """Set up test environment."""
        super().setUp()
        
        # Create an instance of DataConverter
        self.data_converter = DataConverter()
        
        # Create test data
        self.test_array = np.array([
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8]
        ], dtype=np.float32)
    
    def test_normalize_array(self) -> None:
        """Test normalizing a numpy array."""
        # Call the method
        normalized = self.data_converter.normalize_array(self.test_array)
        
        # Check that we got a numpy array
        self.assertIsInstance(normalized, np.ndarray)
        
        # Check that the values are normalized to the range [0, 1]
        self.assertAlmostEqual(np.min(normalized), 0.0)
        self.assertAlmostEqual(np.max(normalized), 1.0)
        
        # Check that the dimensions are the same
        self.assertEqual(normalized.shape, self.test_array.shape)
        
        # Test with invalid array
        with self.assertRaises(ConversionError):
            self.data_converter.normalize_array("not_an_array")
        
        # Test with empty array
        with self.assertRaises(ConversionError):
            self.data_converter.normalize_array(np.array([]))
        
        # Test with array that has same min and max values
        with self.assertRaises(ConversionError):
            self.data_converter.normalize_array(np.ones((3, 3)))
    
    def test_array_to_image(self) -> None:
        """Test converting a numpy array to an image."""
        # Mock PIL's Image
        with patch("PIL.Image.fromarray") as mock_from_array:
            mock_image = Mock()
            mock_from_array.return_value = mock_image
            
            # Call the method
            image = self.data_converter.array_to_image(self.test_array)
            
            # Check that the method was called with the correct arguments
            mock_from_array.assert_called_once()
            
            # Check that we got an image
            self.assertEqual(image, mock_image)
            
            # Test with invalid array
            with self.assertRaises(ConversionError):
                self.data_converter.array_to_image("not_an_array")
            
            # Test with 3D array
            with self.assertRaises(ConversionError):
                self.data_converter.array_to_image(np.zeros((2, 2, 2)))
    
    def test_image_to_array(self) -> None:
        """Test converting an image to a numpy array."""
        # Mock image with proper data
        mock_image = Mock()
        mock_image.size = (3, 3)
        # Set up pixel data as a list of RGB tuples to simulate color image
        mock_image.getdata.return_value = [(r, g, b) for r, g, b in zip(range(9), range(9), range(9))]
        
        # Set up the mode property to simulate a color image
        mock_image.mode = "RGB"

        # Patch the isinstance check to always return True for our mock
        with patch('heightcraft.utils.data_converter.isinstance', return_value=True):
            # Patch the validation method to do nothing
            with patch.object(self.data_converter, '_validate_image'):
                # Call the method
                array = self.data_converter.image_to_array(mock_image)

                # Check the result
                self.assertIsInstance(array, np.ndarray)
                self.assertEqual(array.shape, (3, 3, 3))  # 3x3 RGB image
    
    def test_resize_array(self) -> None:
        """Test resizing a numpy array."""
        # Call the method
        new_shape = (6, 6)
        resized = self.data_converter.resize_array(self.test_array, new_shape)
        
        # Check that we got a numpy array
        self.assertIsInstance(resized, np.ndarray)
        
        # Check that the dimensions are correct
        self.assertEqual(resized.shape, new_shape)
        
        # Test with invalid array
        with self.assertRaises(ConversionError):
            self.data_converter.resize_array("not_an_array", new_shape)
        
        # Test with invalid shape
        with self.assertRaises(ConversionError):
            self.data_converter.resize_array(self.test_array, "not_a_shape")
        
        # Test with negative dimensions
        with self.assertRaises(ConversionError):
            self.data_converter.resize_array(self.test_array, (-1, 6))
        with self.assertRaises(ConversionError):
            self.data_converter.resize_array(self.test_array, (6, -1))
    
    def test_crop_array(self) -> None:
        """Test cropping a numpy array."""
        # Call the method
        x_min, y_min = 1, 1
        width, height = 1, 1
        cropped = self.data_converter.crop_array(self.test_array, x_min, y_min, width, height)
        
        # Check that we got a numpy array
        self.assertIsInstance(cropped, np.ndarray)
        
        # Check that the dimensions are correct
        self.assertEqual(cropped.shape, (height, width))
        
        # Check that the values are correct
        self.assertEqual(cropped[0, 0], 4)
        
        # Test with invalid array
        with self.assertRaises(ConversionError):
            self.data_converter.crop_array("not_an_array", x_min, y_min, width, height)
        
        # Test with invalid parameters
        with self.assertRaises(ConversionError):
            self.data_converter.crop_array(self.test_array, -1, 0, 1, 1)
        with self.assertRaises(ConversionError):
            self.data_converter.crop_array(self.test_array, 0, -1, 1, 1)
        with self.assertRaises(ConversionError):
            self.data_converter.crop_array(self.test_array, 0, 0, 0, 1)
        with self.assertRaises(ConversionError):
            self.data_converter.crop_array(self.test_array, 0, 0, 1, 0)
        with self.assertRaises(ConversionError):
            self.data_converter.crop_array(self.test_array, 0, 0, 4, 1)
        with self.assertRaises(ConversionError):
            self.data_converter.crop_array(self.test_array, 0, 0, 1, 4)
    
    def test_apply_gaussian_blur(self) -> None:
        """Test applying a Gaussian blur to a numpy array."""
        # Call the method
        sigma = 1.0
        blurred = self.data_converter.apply_gaussian_blur(self.test_array, sigma)
        
        # Check that we got a numpy array
        self.assertIsInstance(blurred, np.ndarray)
        
        # Check that the dimensions are the same
        self.assertEqual(blurred.shape, self.test_array.shape)
        
        # Test with invalid array
        with self.assertRaises(ConversionError):
            self.data_converter.apply_gaussian_blur("not_an_array", sigma)
        
        # Test with invalid sigma
        with self.assertRaises(ConversionError):
            self.data_converter.apply_gaussian_blur(self.test_array, -1.0)
    
    def test_apply_median_filter(self) -> None:
        """Test applying a median filter to a numpy array."""
        # Call the method
        kernel_size = 3
        filtered = self.data_converter.apply_median_filter(self.test_array, kernel_size)
        
        # Check that we got a numpy array
        self.assertIsInstance(filtered, np.ndarray)
        
        # Check that the dimensions are the same
        self.assertEqual(filtered.shape, self.test_array.shape)
        
        # Test with invalid array
        with self.assertRaises(ConversionError):
            self.data_converter.apply_median_filter("not_an_array", kernel_size)
        
        # Test with even kernel size
        with self.assertRaises(ConversionError):
            self.data_converter.apply_median_filter(self.test_array, 2)
        
        # Test with invalid kernel size
        with self.assertRaises(ConversionError):
            self.data_converter.apply_median_filter(self.test_array, -1)
    
    def test_equalize_histogram(self) -> None:
        """Test equalizing the histogram of a numpy array."""
        # Call the method
        equalized = self.data_converter.equalize_histogram(self.test_array)
        
        # Check that we got a numpy array
        self.assertIsInstance(equalized, np.ndarray)
        
        # Check that the dimensions are the same
        self.assertEqual(equalized.shape, self.test_array.shape)
        
        # Test with invalid array
        with self.assertRaises(ConversionError):
            self.data_converter.equalize_histogram("not_an_array")
    
    def test_convert_mesh_to_point_cloud(self) -> None:
        """Test converting a mesh to a point cloud."""
        # Mock mesh and point cloud
        mock_mesh = Mock()
        mock_mesh.vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])  # Add attributes needed by validation
        mock_mesh.faces = np.array([[0, 1, 2]])
        mock_point_cloud = Mock()
        mock_mesh.to_point_cloud.return_value = mock_point_cloud

        # Patch the isinstance check to always return True for our mock
        with patch('heightcraft.utils.data_converter.isinstance', return_value=True):
            # Patch the validation method to do nothing
            with patch.object(self.data_converter, '_validate_mesh'):
                # Call the method
                point_cloud = self.data_converter.convert_mesh_to_point_cloud(mock_mesh)

                # Check the result
                self.assertEqual(point_cloud, mock_point_cloud)

    def test_convert_height_map_to_mesh(self) -> None:
        """Test converting a height map to a mesh."""
        # Mock height map and mesh
        mock_height_map = Mock()
        mock_height_map.width = 10  # Add attributes needed by validation
        mock_height_map.height = 10
        mock_mesh = Mock()
        mock_height_map.to_mesh.return_value = mock_mesh

        # Patch the isinstance check to always return True for our mock
        with patch('heightcraft.utils.data_converter.isinstance', return_value=True):
            # Patch the validation method to do nothing
            with patch.object(self.data_converter, '_validate_height_map'):
                # Call the method
                mesh = self.data_converter.convert_height_map_to_mesh(mock_height_map)

                # Check the result
                self.assertEqual(mesh, mock_mesh) 