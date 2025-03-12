"""
Tests for the HeightMapService.
"""

import unittest
from unittest.mock import Mock, patch

import numpy as np

from heightcraft.core.exceptions import HeightMapServiceError
from heightcraft.domain.height_map import HeightMap
from heightcraft.domain.point_cloud import PointCloud
from heightcraft.services.height_map_service import HeightMapService
from tests.base_test_case import BaseTestCase


class TestHeightMapService(BaseTestCase):
    """Tests for the HeightMapService."""
    
    def setUp(self) -> None:
        """Set up test environment."""
        super().setUp()
        
        # Create mock dependencies
        self.resource_manager = Mock()
        self.repository = Mock()
        
        # Create an instance of HeightMapService
        self.height_map_service = HeightMapService(self.repository)
        
        # Create a test height map
        self.data = np.array([
            [0, 0.1, 0.2],
            [0.3, 0.4, 0.5],
            [0.6, 0.7, 0.8]
        ], dtype=np.float32)
        self.bit_depth = 8
        self.test_height_map = HeightMap(self.data, bit_depth=self.bit_depth)
        
        # Create test file paths
        self.test_file = "/tmp/test.png"
    
    def test_load_height_map(self) -> None:
        """Test loading a height map."""
        # Set up mock
        self.repository.load.return_value = self.test_height_map
        
        # Call the method
        file_path = self.get_temp_path("test_height_map.png")
        height_map = self.height_map_service.load_height_map(file_path, self.bit_depth)
        
        # Check that the repository method was called
        self.repository.load.assert_called_once_with(file_path, self.bit_depth)
        
        # Check the returned height map
        self.assertEqual(height_map, self.test_height_map)
        
        # Test with repository error
        self.repository.load.side_effect = Exception("Repository error")
        with self.assertRaises(HeightMapServiceError):
            self.height_map_service.load_height_map(file_path, self.bit_depth)
    
    def test_save_height_map(self) -> None:
        """Test saving a height map."""
        # Set up mock
        self.repository.save.return_value = True
        
        # Call the method
        file_path = self.get_temp_path("test_height_map.png")
        result = self.height_map_service.save_height_map(self.test_height_map, file_path)
        
        # Check that the repository method was called
        self.repository.save.assert_called_once_with(self.test_height_map, file_path)
        
        # Check the result
        self.assertTrue(result)
        
        # Test with repository error
        self.repository.save.side_effect = Exception("Repository error")
        with self.assertRaises(HeightMapServiceError):
            self.height_map_service.save_height_map(self.test_height_map, file_path)
    
    def test_normalize_height_map(self) -> None:
        """Test normalizing a height map."""
        # Mock the HeightMap behavior since normalize() doesn't exist
        # Create a normalized version of the data
        min_val = self.data.min()
        max_val = self.data.max()
        normalized_data = (self.data - min_val) / (max_val - min_val)
        normalized_height_map = HeightMap(normalized_data, bit_depth=self.bit_depth)
        
        # Mock the behavior
        self.repository.save.return_value = True
        
        # Patch the HeightMap creation
        with patch('heightcraft.domain.height_map.HeightMap') as mock_height_map_class:
            mock_height_map_class.return_value = normalized_height_map
            
            # Call the method
            normalized = self.height_map_service.normalize_height_map(self.test_height_map)
            
            # Check the result - don't compare objects directly, check attributes
            self.assertEqual(normalized.bit_depth, normalized_height_map.bit_depth)
            self.assertEqual(normalized.width, normalized_height_map.width)
            self.assertEqual(normalized.height, normalized_height_map.height)
    
    def test_resize_height_map(self) -> None:
        """Test resizing a height map."""
        # Create a resized version of the data
        new_width = 6
        new_height = 6
        resized_data = np.zeros((new_height, new_width), dtype=np.float32)
        resized_height_map = HeightMap(resized_data, bit_depth=self.bit_depth)
        
        # Mock the behavior by patching HeightMap.resize
        with patch.object(HeightMap, 'resize') as mock_resize:
            mock_resize.return_value = resized_height_map
            
            # Call the method
            resized = self.height_map_service.resize_height_map(self.test_height_map, new_width, new_height)
            
            # Check that resize was called with the correct arguments
            mock_resize.assert_called_once_with((new_width, new_height))
            
            # Check the result
            self.assertEqual(resized, resized_height_map)
    
    def test_crop_height_map(self) -> None:
        """Test cropping a height map."""
        # Create a cropped version of the data
        x_min, y_min = 1, 1
        width, height = 1, 1
        cropped_data = self.data[y_min:y_min+height, x_min:x_min+width].copy()
        cropped_height_map = HeightMap(cropped_data, bit_depth=self.bit_depth)
        
        # Mock the behavior by patching HeightMap.crop
        with patch.object(HeightMap, 'crop') as mock_crop:
            mock_crop.return_value = cropped_height_map
            
            # Call the method
            cropped = self.height_map_service.crop_height_map(self.test_height_map, x_min, y_min, width, height)
            
            # Check that crop was called with the correct arguments
            mock_crop.assert_called_once_with((x_min, y_min), (x_min+width, y_min+height))
            
            # Check the result
            self.assertEqual(cropped, cropped_height_map)
        
        # Test with invalid parameters
        with self.assertRaises(HeightMapServiceError):
            self.height_map_service.crop_height_map(self.test_height_map, -1, 0, 1, 1)
        with self.assertRaises(HeightMapServiceError):
            self.height_map_service.crop_height_map(self.test_height_map, 0, -1, 1, 1)
        with self.assertRaises(HeightMapServiceError):
            self.height_map_service.crop_height_map(self.test_height_map, 0, 0, 0, 1)
        with self.assertRaises(HeightMapServiceError):
            self.height_map_service.crop_height_map(self.test_height_map, 0, 0, 1, 0)
    
    def test_convert_height_map_to_mesh(self) -> None:
        """Test converting a height map to a mesh."""
        # Call the method
        mesh = self.height_map_service.convert_height_map_to_mesh(self.test_height_map)
        
        # Check that we got a mesh object
        self.assertIsNotNone(mesh)
        
        # Check mesh properties
        self.assertEqual(mesh.vertices.shape[0], self.test_height_map.width * self.test_height_map.height)
        self.assertEqual(mesh.faces.shape[0], 2 * (self.test_height_map.width - 1) * (self.test_height_map.height - 1))
    
    def test_convert_height_map_to_point_cloud(self) -> None:
        """Test converting a height map to a point cloud."""
        # Mock the point cloud creation with a specific size
        mock_point_cloud = Mock(spec=PointCloud)
        mock_point_cloud.size = 9  # Match the expected size of 3x3 height map
        
        # Patch the convert_to_point_cloud method
        with patch.object(self.test_height_map, 'to_point_cloud', return_value=mock_point_cloud):
            # Call the method
            point_cloud = self.height_map_service.convert_height_map_to_point_cloud(self.test_height_map)
            
            # Check that we got a point cloud object
            self.assertIsNotNone(point_cloud)
            
            # Check point cloud size matches what we expect
            self.assertEqual(point_cloud.size, self.test_height_map.width * self.test_height_map.height)
    
    def test_get_height_map_info(self) -> None:
        """Test getting height map information."""
        # Call the method
        info = self.height_map_service.get_height_map_info(self.test_height_map)
        
        # Check that we got a dictionary
        self.assertIsInstance(info, dict)
        
        # Check that it has the expected keys
        expected_keys = ["width", "height", "aspect_ratio", "bit_depth", "min", "max"]
        for key in expected_keys:
            self.assertIn(key, info)
        
        # Check values
        self.assertEqual(info["width"], self.test_height_map.width)
        self.assertEqual(info["height"], self.test_height_map.height)
        self.assertEqual(info["aspect_ratio"], self.test_height_map.aspect_ratio)
        self.assertEqual(info["bit_depth"], self.test_height_map.bit_depth)
        self.assertEqual(info["min"], self.test_height_map.min_height)
        self.assertEqual(info["max"], self.test_height_map.max_height)
    
    def test_apply_gaussian_blur(self) -> None:
        """Test applying a Gaussian blur to a height map."""
        # Call the method
        sigma = 1.0
        blurred = self.height_map_service.apply_gaussian_blur(self.test_height_map, sigma)
        
        # Check that we got a new HeightMap
        self.assertIsInstance(blurred, HeightMap)
        
        # Check that the dimensions are the same
        self.assertEqual(blurred.width, self.test_height_map.width)
        self.assertEqual(blurred.height, self.test_height_map.height)
        
        # Test with invalid sigma
        with self.assertRaises(HeightMapServiceError):
            self.height_map_service.apply_gaussian_blur(self.test_height_map, -1.0)
    
    def test_apply_median_filter(self) -> None:
        """Test applying a median filter to a height map."""
        # Call the method
        kernel_size = 3
        filtered = self.height_map_service.apply_median_filter(self.test_height_map, kernel_size)
        
        # Check that we got a new HeightMap
        self.assertIsInstance(filtered, HeightMap)
        
        # Check that the dimensions are the same
        self.assertEqual(filtered.width, self.test_height_map.width)
        self.assertEqual(filtered.height, self.test_height_map.height)
        
        # Test with even kernel size
        with self.assertRaises(HeightMapServiceError):
            self.height_map_service.apply_median_filter(self.test_height_map, 2)
        
        # Test with invalid kernel size
        with self.assertRaises(HeightMapServiceError):
            self.height_map_service.apply_median_filter(self.test_height_map, -1)
    
    def test_equalize_histogram(self) -> None:
        """Test equalizing the histogram of a height map."""
        # Call the method
        equalized = self.height_map_service.equalize_histogram(self.test_height_map)
        
        # Check that we got a new HeightMap
        self.assertIsInstance(equalized, HeightMap)
        
        # Check that the dimensions are the same
        self.assertEqual(equalized.width, self.test_height_map.width)
        self.assertEqual(equalized.height, self.test_height_map.height) 