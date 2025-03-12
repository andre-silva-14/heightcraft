"""
Input validator utility for validating input parameters.

This module provides functionality for validating various types of input parameters,
including files, directories, numbers, and domain objects.
"""

import os
import numpy as np
from typing import List, Optional, Union, Any, Type

from heightcraft.core.exceptions import ValidationError


class InputValidator:
    """Utility for validating inputs."""
    
    def __init__(self) -> None:
        """Initialize the input validator."""
        pass
    
    def validate_file_exists(self, file_path: str) -> bool:
        """
        Validate that a file exists.
        
        Args:
            file_path: Path to the file.
            
        Returns:
            True if validation passes.
            
        Raises:
            ValidationError: If validation fails.
        """
        if not os.path.exists(file_path):
            raise ValidationError(f"File does not exist: {file_path}")
        
        if not os.path.isfile(file_path):
            raise ValidationError(f"Path is not a file: {file_path}")
        
        return True
    
    def validate_file_extension(self, file_path: str, allowed_extensions: List[str]) -> bool:
        """
        Validate that a file has an allowed extension.
        
        Args:
            file_path: Path to the file.
            allowed_extensions: List of allowed extensions (including the dot, e.g. [".txt", ".png"]).
            
        Returns:
            True if validation passes.
            
        Raises:
            ValidationError: If validation fails.
        """
        extension = os.path.splitext(file_path)[1].lower()
        
        if extension not in allowed_extensions:
            raise ValidationError(f"File extension not allowed: {extension}, allowed: {allowed_extensions}")
        
        return True
    
    def validate_directory_exists(self, directory_path: str) -> bool:
        """
        Validate that a directory exists.
        
        Args:
            directory_path: Path to the directory.
            
        Returns:
            True if validation passes.
            
        Raises:
            ValidationError: If validation fails.
        """
        if not os.path.exists(directory_path):
            raise ValidationError(f"Directory does not exist: {directory_path}")
        
        if not os.path.isdir(directory_path):
            raise ValidationError(f"Path is not a directory: {directory_path}")
        
        return True
    
    def validate_numpy_array(self, array: Any) -> bool:
        """
        Validate that an object is a numpy array.
        
        Args:
            array: Object to validate.
            
        Returns:
            True if validation passes.
            
        Raises:
            ValidationError: If validation fails.
        """
        if not isinstance(array, np.ndarray):
            raise ValidationError(f"Object is not a numpy array: {type(array)}")
        
        return True
    
    def validate_numpy_array_dimensions(self, array: np.ndarray, dims: int) -> bool:
        """
        Validate that a numpy array has a specific number of dimensions.
        
        Args:
            array: Array to validate.
            dims: Expected number of dimensions.
            
        Returns:
            True if validation passes.
            
        Raises:
            ValidationError: If validation fails.
        """
        self.validate_numpy_array(array)
        
        if array.ndim != dims:
            raise ValidationError(f"Array has {array.ndim} dimensions, expected {dims}")
        
        return True
    
    def validate_numpy_array_shape(self, array: np.ndarray, shape: tuple) -> bool:
        """
        Validate that a numpy array has a specific shape.
        
        Args:
            array: Array to validate.
            shape: Expected shape.
            
        Returns:
            True if validation passes.
            
        Raises:
            ValidationError: If validation fails.
        """
        self.validate_numpy_array(array)
        
        if array.shape != shape:
            raise ValidationError(f"Array has shape {array.shape}, expected {shape}")
        
        return True
    
    def validate_positive_number(self, value: Union[int, float]) -> bool:
        """
        Validate that a number is positive (> 0).
        
        Args:
            value: Number to validate.
            
        Returns:
            True if validation passes.
            
        Raises:
            ValidationError: If validation fails.
        """
        if not isinstance(value, (int, float)):
            raise ValidationError(f"Value is not a number: {type(value)}")
        
        if value <= 0:
            raise ValidationError(f"Value is not positive: {value}")
        
        return True
    
    def validate_non_negative_number(self, value: Union[int, float]) -> bool:
        """
        Validate that a number is non-negative (>= 0).
        
        Args:
            value: Number to validate.
            
        Returns:
            True if validation passes.
            
        Raises:
            ValidationError: If validation fails.
        """
        if not isinstance(value, (int, float)):
            raise ValidationError(f"Value is not a number: {type(value)}")
        
        if value < 0:
            raise ValidationError(f"Value is not non-negative: {value}")
        
        return True
    
    def validate_number_range(self, value: Union[int, float], min_value: Union[int, float], max_value: Union[int, float]) -> bool:
        """
        Validate that a number is within a specific range.
        
        Args:
            value: Number to validate.
            min_value: Minimum value (inclusive).
            max_value: Maximum value (inclusive).
            
        Returns:
            True if validation passes.
            
        Raises:
            ValidationError: If validation fails.
        """
        if not isinstance(value, (int, float)):
            raise ValidationError(f"Value is not a number: {type(value)}")
        
        if value < min_value or value > max_value:
            raise ValidationError(f"Value {value} is not in range [{min_value}, {max_value}]")
        
        return True
    
    def validate_string_not_empty(self, value: str) -> bool:
        """
        Validate that a string is not empty.
        
        Args:
            value: String to validate.
            
        Returns:
            True if validation passes.
            
        Raises:
            ValidationError: If validation fails.
        """
        if not isinstance(value, str):
            raise ValidationError(f"Value is not a string: {type(value)}")
        
        if value == "":
            raise ValidationError("String is empty")
        
        return True
    
    def validate_odd_number(self, value: int) -> bool:
        """
        Validate that a number is odd.
        
        Args:
            value: Number to validate.
            
        Returns:
            True if validation passes.
            
        Raises:
            ValidationError: If validation fails.
        """
        if not isinstance(value, int):
            raise ValidationError(f"Value is not an integer: {type(value)}")
        
        if value % 2 == 0:
            raise ValidationError(f"Value is not odd: {value}")
        
        return True
    
    def validate_mesh(self, mesh) -> bool:
        """
        Validate that an object is a valid mesh.
        
        Args:
            mesh: Mesh to validate.
            
        Returns:
            True if validation passes.
            
        Raises:
            ValidationError: If validation fails.
        """
        try:
            # Check that the mesh has the expected attributes
            for attr in ["vertices", "faces"]:
                if not hasattr(mesh, attr):
                    raise ValidationError(f"Mesh does not have '{attr}' attribute")
            
            # Check that vertices and faces are numpy arrays
            self.validate_numpy_array(mesh.vertices)
            self.validate_numpy_array(mesh.faces)
            
            # Check that vertices and faces have the expected shapes
            if mesh.vertices.shape[0] == 0 or mesh.vertices.shape[1] != 3:
                raise ValidationError(f"Vertices have invalid shape: {mesh.vertices.shape}")
            
            if mesh.faces.shape[0] == 0 or mesh.faces.shape[1] != 3:
                raise ValidationError(f"Faces have invalid shape: {mesh.faces.shape}")
            
            return True
        
        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            raise ValidationError(f"Invalid mesh: {str(e)}")
    
    def validate_height_map(self, height_map) -> bool:
        """
        Validate that an object is a valid height map.
        
        Args:
            height_map: Height map to validate.
            
        Returns:
            True if validation passes.
            
        Raises:
            ValidationError: If validation fails.
        """
        try:
            # Check that the height map has the expected attributes
            for attr in ["data", "resolution", "width", "height"]:
                if not hasattr(height_map, attr):
                    raise ValidationError(f"Height map does not have '{attr}' attribute")
            
            # Check that data is a numpy array
            self.validate_numpy_array(height_map.data)
            
            # Check that resolution is positive
            self.validate_positive_number(height_map.resolution)
            
            # Check that width and height are positive
            self.validate_positive_number(height_map.width)
            self.validate_positive_number(height_map.height)
            
            # Check that the data has the expected shape
            if height_map.data.shape != (height_map.height, height_map.width):
                raise ValidationError(f"Data has shape {height_map.data.shape}, expected ({height_map.height}, {height_map.width})")
            
            return True
        
        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            raise ValidationError(f"Invalid height map: {str(e)}") 