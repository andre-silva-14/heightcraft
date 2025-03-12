"""
Validators for Heightcraft.

This module provides validators for input validation.
"""

import math
import os
import re
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from heightcraft.core.exceptions import ValidationError, MeshValidationError


def validate_positive_integer(value: Any, name: str = "value") -> int:
    """
    Validate that a value is a positive integer.
    
    Args:
        value: Value to validate
        name: Name of the value (for error messages)
        
    Returns:
        Validated value as an integer
        
    Raises:
        ValidationError: If the value is not a positive integer
    """
    try:
        int_value = int(value)
        if int_value <= 0:
            raise ValidationError(f"{name} must be a positive integer, got {value}")
        return int_value
    except (ValueError, TypeError):
        raise ValidationError(f"{name} must be an integer, got {value} ({type(value).__name__})")


def validate_non_negative_integer(value: Any, name: str = "value") -> int:
    """
    Validate that a value is a non-negative integer.
    
    Args:
        value: Value to validate
        name: Name of the value (for error messages)
        
    Returns:
        Validated value as an integer
        
    Raises:
        ValidationError: If the value is not a non-negative integer
    """
    try:
        int_value = int(value)
        if int_value < 0:
            raise ValidationError(f"{name} must be a non-negative integer, got {value}")
        return int_value
    except (ValueError, TypeError):
        raise ValidationError(f"{name} must be an integer, got {value} ({type(value).__name__})")


def validate_range(value: Any, min_value: float, max_value: float, name: str = "value") -> float:
    """
    Validate that a value is within a range.
    
    Args:
        value: Value to validate
        min_value: Minimum allowed value (inclusive)
        max_value: Maximum allowed value (inclusive)
        name: Name of the value (for error messages)
        
    Returns:
        Validated value as a float
        
    Raises:
        ValidationError: If the value is outside the range
    """
    try:
        float_value = float(value)
        if float_value < min_value or float_value > max_value:
            raise ValidationError(f"{name} must be between {min_value} and {max_value}, got {value}")
        return float_value
    except (ValueError, TypeError):
        raise ValidationError(f"{name} must be a number, got {value} ({type(value).__name__})")


def validate_choice(value: Any, choices: List[Any], name: str = "value") -> Any:
    """
    Validate that a value is one of a set of choices.
    
    Args:
        value: Value to validate
        choices: List of valid choices
        name: Name of the value (for error messages)
        
    Returns:
        Validated value
        
    Raises:
        ValidationError: If the value is not one of the choices
    """
    if value not in choices:
        raise ValidationError(f"{name} must be one of {choices}, got {value}")
    return value


def validate_file_exists(path: str) -> str:
    """
    Validate that a file exists.
    
    Args:
        path: Path to validate
        
    Returns:
        Validated path
        
    Raises:
        ValidationError: If the file doesn't exist
    """
    if not os.path.exists(path):
        raise ValidationError(f"File does not exist: {path}")
    if not os.path.isfile(path):
        raise ValidationError(f"Path is not a file: {path}")
    return path


def validate_directory_exists(path: str) -> str:
    """
    Validate that a directory exists.
    
    Args:
        path: Path to validate
        
    Returns:
        Validated path
        
    Raises:
        ValidationError: If the directory does not exist
    """
    if not os.path.isdir(path):
        raise ValidationError(f"Directory does not exist: {path}")
    return path


def validate_file_extension(path: str, allowed_extensions: List[str]) -> str:
    """
    Validate that a file has an allowed extension.
    
    Args:
        path: Path to validate
        allowed_extensions: List of allowed extensions (with or without the dot)
        
    Returns:
        Validated path
        
    Raises:
        ValidationError: If the file does not have an allowed extension
    """
    # Normalize extensions (ensure they start with a dot)
    normalized_extensions = [ext if ext.startswith(".") else f".{ext}" for ext in allowed_extensions]
    
    # Get the file extension
    extension = os.path.splitext(path)[1].lower()
    
    if extension not in normalized_extensions:
        raise ValidationError(f"File must have one of these extensions: {normalized_extensions}, got {extension}")
    
    return path


def validate_split_value(value: Any) -> int:
    """
    Validate a split value (must be a positive integer that can form a grid).
    
    Args:
        value: Value to validate
        
    Returns:
        Validated value as an integer
        
    Raises:
        ValidationError: If the value is not valid
    """
    try:
        int_value = int(value)
        if int_value <= 0:
            raise ValidationError(f"Split must be a positive integer, got {value}")
        
        # Check if the value can form a grid
        if math.isqrt(int_value) ** 2 == int_value:
            # Perfect square (e.g., 4, 9, 16)
            return int_value
        
        # Check if the value can form a rectangular grid
        for i in range(2, int(math.sqrt(int_value)) + 1):
            if int_value % i == 0:
                # Can form a grid with dimensions i x (int_value/i)
                return int_value
        
        raise ValidationError(f"Split must be able to form a grid (e.g., 4, 9, 12), got {value}")
        
    except (ValueError, TypeError):
        raise ValidationError(f"Split must be an integer, got {value} ({type(value).__name__})")


def validate_bit_depth(value: Any) -> int:
    """
    Validate a bit depth value (must be 8 or 16).
    
    Args:
        value: Value to validate
        
    Returns:
        Validated value as an integer
        
    Raises:
        ValidationError: If the value is not valid
    """
    try:
        int_value = int(value)
        if int_value not in [8, 16]:
            raise ValidationError(f"Bit depth must be 8 or 16, got {value}")
        return int_value
    except (ValueError, TypeError):
        raise ValidationError(f"Bit depth must be an integer, got {value} ({type(value).__name__})")


def validate_upscale_factor(value: Any) -> int:
    """
    Validate an upscale factor (must be 2, 3, or 4).
    
    Args:
        value: Value to validate
        
    Returns:
        Validated value as an integer
        
    Raises:
        ValidationError: If the value is not valid
    """
    try:
        int_value = int(value)
        if int_value not in [2, 3, 4]:
            raise ValidationError(f"Upscale factor must be 2, 3, or 4, got {value}")
        return int_value
    except (ValueError, TypeError):
        raise ValidationError(f"Upscale factor must be an integer, got {value} ({type(value).__name__})")


def validate_max_memory(value: Any) -> float:
    """
    Validate a max memory value (must be between 0 and 1).
    
    Args:
        value: Value to validate
        
    Returns:
        Validated value as a float
        
    Raises:
        ValidationError: If the value is not valid
    """
    return validate_range(value, 0.0, 1.0, "max_memory")


def validate_mesh(mesh: Any) -> None:
    """
    Validate that a mesh object is valid.
    
    Args:
        mesh: Mesh object to validate (should be a trimesh.Trimesh)
        
    Raises:
        MeshValidationError: If the mesh is not valid
    """
    if mesh is None:
        raise MeshValidationError("Mesh cannot be None")
    
    # Check if it's a trimesh.Trimesh object
    if not hasattr(mesh, 'vertices') or not hasattr(mesh, 'faces'):
        raise MeshValidationError("Mesh must be a trimesh.Trimesh object")
    
    # Check if the mesh has vertices
    if len(mesh.vertices) == 0:
        raise MeshValidationError("Mesh has no vertices")
    
    # Check if the mesh has faces
    if len(mesh.faces) == 0:
        raise MeshValidationError("Mesh has no faces")
    
    # Optional: Check for manifoldness, watertightness, etc.
    # These are good properties for meshes to have, but might be too strict
    # for some use cases.
    #
    # if not mesh.is_watertight:
    #     raise MeshValidationError("Mesh is not watertight")
    #
    # if not mesh.is_winding_consistent:
    #     raise MeshValidationError("Mesh has inconsistent winding") 