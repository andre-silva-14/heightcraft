"""
Tests for the InputValidator utility.
"""

import os
import unittest
from unittest.mock import Mock, patch

import numpy as np

from heightcraft.core.exceptions import ValidationError
from heightcraft.utils.input_validator import InputValidator
from tests.base_test_case import BaseTestCase


class TestInputValidator(BaseTestCase):
    """Tests for the InputValidator utility."""
    
    def setUp(self) -> None:
        """Set up test environment."""
        super().setUp()
        
        # Create an instance of InputValidator
        self.validator = InputValidator()
        
        # Create test data
        self.test_array = np.array([
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8]
        ], dtype=np.float32)
        
        # Create a test file path
        self.test_file = self.get_temp_path("test_file.txt")
        with open(self.test_file, "w") as f:
            f.write("Test content")
    
    def test_validate_file_exists(self) -> None:
        """Test validating that a file exists."""
        # Test with existing file
        result = self.validator.validate_file_exists(self.test_file)
        self.assertTrue(result)
        
        # Test with non-existent file
        with self.assertRaises(ValidationError):
            self.validator.validate_file_exists("non_existent_file.txt")
    
    def test_validate_file_extension(self) -> None:
        """Test validating a file extension."""
        # Test with valid extension
        result = self.validator.validate_file_extension(self.test_file, [".txt"])
        self.assertTrue(result)
        
        # Test with invalid extension
        with self.assertRaises(ValidationError):
            self.validator.validate_file_extension(self.test_file, [".png", ".jpg"])
    
    def test_validate_directory_exists(self) -> None:
        """Test validating that a directory exists."""
        # Test with existing directory
        result = self.validator.validate_directory_exists(os.path.dirname(self.test_file))
        self.assertTrue(result)
        
        # Test with non-existent directory
        with self.assertRaises(ValidationError):
            self.validator.validate_directory_exists("non_existent_directory")
    
    def test_validate_numpy_array(self) -> None:
        """Test validating a numpy array."""
        # Test with valid array
        result = self.validator.validate_numpy_array(self.test_array)
        self.assertTrue(result)
        
        # Test with non-array
        with self.assertRaises(ValidationError):
            self.validator.validate_numpy_array("not_an_array")
    
    def test_validate_numpy_array_dimensions(self) -> None:
        """Test validating numpy array dimensions."""
        # Test with valid 2D array
        result = self.validator.validate_numpy_array_dimensions(self.test_array, 2)
        self.assertTrue(result)
        
        # Test with 1D array but expecting 2D
        with self.assertRaises(ValidationError):
            self.validator.validate_numpy_array_dimensions(np.array([1, 2, 3]), 2)
        
        # Test with 2D array but expecting 1D
        with self.assertRaises(ValidationError):
            self.validator.validate_numpy_array_dimensions(self.test_array, 1)
    
    def test_validate_numpy_array_shape(self) -> None:
        """Test validating numpy array shape."""
        # Test with valid shape
        result = self.validator.validate_numpy_array_shape(self.test_array, (3, 3))
        self.assertTrue(result)
        
        # Test with invalid shape
        with self.assertRaises(ValidationError):
            self.validator.validate_numpy_array_shape(self.test_array, (4, 4))
    
    def test_validate_positive_number(self) -> None:
        """Test validating a positive number."""
        # Test with positive number
        result = self.validator.validate_positive_number(5)
        self.assertTrue(result)
        
        # Test with zero
        with self.assertRaises(ValidationError):
            self.validator.validate_positive_number(0)
        
        # Test with negative number
        with self.assertRaises(ValidationError):
            self.validator.validate_positive_number(-5)
        
        # Test with non-number
        with self.assertRaises(ValidationError):
            self.validator.validate_positive_number("not_a_number")
    
    def test_validate_non_negative_number(self) -> None:
        """Test validating a non-negative number."""
        # Test with positive number
        result = self.validator.validate_non_negative_number(5)
        self.assertTrue(result)
        
        # Test with zero
        result = self.validator.validate_non_negative_number(0)
        self.assertTrue(result)
        
        # Test with negative number
        with self.assertRaises(ValidationError):
            self.validator.validate_non_negative_number(-5)
        
        # Test with non-number
        with self.assertRaises(ValidationError):
            self.validator.validate_non_negative_number("not_a_number")
    
    def test_validate_number_range(self) -> None:
        """Test validating a number in a range."""
        # Test with number in range
        result = self.validator.validate_number_range(5, 0, 10)
        self.assertTrue(result)
        
        # Test with number at lower bound
        result = self.validator.validate_number_range(0, 0, 10)
        self.assertTrue(result)
        
        # Test with number at upper bound
        result = self.validator.validate_number_range(10, 0, 10)
        self.assertTrue(result)
        
        # Test with number below range
        with self.assertRaises(ValidationError):
            self.validator.validate_number_range(-1, 0, 10)
        
        # Test with number above range
        with self.assertRaises(ValidationError):
            self.validator.validate_number_range(11, 0, 10)
        
        # Test with non-number
        with self.assertRaises(ValidationError):
            self.validator.validate_number_range("not_a_number", 0, 10)
    
    def test_validate_string_not_empty(self) -> None:
        """Test validating a non-empty string."""
        # Test with non-empty string
        result = self.validator.validate_string_not_empty("test")
        self.assertTrue(result)
        
        # Test with empty string
        with self.assertRaises(ValidationError):
            self.validator.validate_string_not_empty("")
        
        # Test with non-string
        with self.assertRaises(ValidationError):
            self.validator.validate_string_not_empty(5)
    
    def test_validate_odd_number(self) -> None:
        """Test validating an odd number."""
        # Test with odd number
        result = self.validator.validate_odd_number(3)
        self.assertTrue(result)
        
        # Test with even number
        with self.assertRaises(ValidationError):
            self.validator.validate_odd_number(2)
        
        # Test with non-integer
        with self.assertRaises(ValidationError):
            self.validator.validate_odd_number(3.5)
        
        # Test with non-number
        with self.assertRaises(ValidationError):
            self.validator.validate_odd_number("not_a_number")
    
    def test_validate_mesh(self) -> None:
        """Test validating a mesh object."""
        # Mock mesh object
        mock_mesh = Mock()
        mock_mesh.vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        mock_mesh.faces = np.array([[0, 1, 2]])
        
        # Test with valid mesh
        result = self.validator.validate_mesh(mock_mesh)
        self.assertTrue(result)
        
        # Test with invalid mesh (no vertices)
        mock_mesh = Mock()
        mock_mesh.vertices = np.array([])
        mock_mesh.faces = np.array([[0, 1, 2]])
        
        with self.assertRaises(ValidationError):
            self.validator.validate_mesh(mock_mesh)
        
        # Test with invalid mesh (no faces)
        mock_mesh = Mock()
        mock_mesh.vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        mock_mesh.faces = np.array([])
        
        with self.assertRaises(ValidationError):
            self.validator.validate_mesh(mock_mesh)
        
        # Test with non-mesh
        with self.assertRaises(ValidationError):
            self.validator.validate_mesh("not_a_mesh")
    
    def test_validate_height_map(self) -> None:
        """Test validating a height map object."""
        # Mock height map object
        mock_height_map = Mock()
        mock_height_map.data = self.test_array
        mock_height_map.width = 3
        mock_height_map.height = 3
        mock_height_map.resolution = 1.0
        
        # Test with valid height map
        result = self.validator.validate_height_map(mock_height_map)
        self.assertTrue(result)
        
        # Test with invalid height map (no data)
        mock_height_map = Mock()
        mock_height_map.data = None
        mock_height_map.width = 3
        mock_height_map.height = 3
        mock_height_map.resolution = 1.0
        
        with self.assertRaises(ValidationError):
            self.validator.validate_height_map(mock_height_map)
        
        # Test with invalid height map (negative resolution)
        mock_height_map = Mock()
        mock_height_map.data = self.test_array
        mock_height_map.width = 3
        mock_height_map.height = 3
        mock_height_map.resolution = -1.0
        
        with self.assertRaises(ValidationError):
            self.validator.validate_height_map(mock_height_map)
        
        # Test with non-height map
        with self.assertRaises(ValidationError):
            self.validator.validate_height_map("not_a_height_map")