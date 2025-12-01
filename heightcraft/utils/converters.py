"""
Converters for Heightcraft.

This module provides converters for data conversion between different formats.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from heightcraft.core.exceptions import ValidationError


def normalize_array(
    array: np.ndarray,
    target_min: float = 0.0,
    target_max: float = 1.0,
    source_min: Optional[float] = None,
    source_max: Optional[float] = None
) -> np.ndarray:
    """
    Normalize a numpy array to a target range.
    
    Args:
        array: Array to normalize
        target_min: Target minimum value
        target_max: Target maximum value
        source_min: Source minimum value (defaults to array.min())
        source_max: Source maximum value (defaults to array.max())
        
    Returns:
        Normalized array
        
    Raises:
        ValidationError: If the source range is invalid
    """
    # Get source range
    if source_min is None:
        source_min = np.min(array)
    if source_max is None:
        source_max = np.max(array)
    
    # Validate source range
    if source_min == source_max:
        return np.full_like(array, (target_min + target_max) / 2)
    
    # Normalize
    # Cast to float to avoid integer overflow
    normalized = (array.astype(float) - float(source_min)) / (float(source_max) - float(source_min))
    
    # Scale to target range
    return normalized * (target_max - target_min) + target_min


def convert_dtype(
    array: np.ndarray,
    dtype: Union[str, np.dtype],
    normalize: bool = True
) -> np.ndarray:
    """
    Convert a numpy array to a different data type.
    
    Args:
        array: Array to convert
        dtype: Target data type
        normalize: Whether to normalize the array before conversion
        
    Returns:
        Converted array
    """
    # Get target dtype info
    target_dtype = np.dtype(dtype)
    
    # Handle conversion
    if normalize and np.issubdtype(target_dtype, np.integer):
        # Get target range
        info = np.iinfo(target_dtype)
        target_min = info.min
        target_max = info.max
        
        # Normalize
        normalized = normalize_array(array, target_min, target_max)
        
        # Convert
        return normalized.astype(target_dtype)
    else:
        # Direct conversion
        return array.astype(target_dtype)


def bit_depth_to_dtype(bit_depth: int) -> np.dtype:
    """
    Convert a bit depth to a numpy data type.
    
    Args:
        bit_depth: Bit depth (8 or 16)
        
    Returns:
        Numpy data type
        
    Raises:
        ValidationError: If the bit depth is not supported
    """
    if bit_depth == 8:
        return np.dtype(np.uint8)
    elif bit_depth == 16:
        return np.dtype(np.uint16)
    elif bit_depth == 32:
        return np.dtype(np.float32)
    else:
        raise ValidationError(f"Unsupported bit depth: {bit_depth}, must be 8, 16, or 32")


def convert_bit_depth(
    array: np.ndarray,
    bit_depth: int,
    normalize: bool = True
) -> np.ndarray:
    """
    Convert a numpy array to a specific bit depth.
    
    Args:
        array: Array to convert
        bit_depth: Target bit depth (8 or 16)
        normalize: Whether to normalize the array before conversion
        
    Returns:
        Converted array
        
    Raises:
        ValidationError: If the bit depth is not supported
    """
    dtype = bit_depth_to_dtype(bit_depth)
    return convert_dtype(array, dtype, normalize)


def int_to_uint(array: np.ndarray) -> np.ndarray:
    """
    Convert a signed integer array to an unsigned integer array.
    
    Args:
        array: Array to convert
        
    Returns:
        Converted array
    """
    # Get source dtype info
    source_dtype = array.dtype
    
    # Check if conversion is needed
    if not np.issubdtype(source_dtype, np.integer) or not np.issubdtype(source_dtype, np.signedinteger):
        return array
    
    # Get corresponding unsigned dtype
    if source_dtype == np.int8:
        target_dtype = np.uint8
    elif source_dtype == np.int16:
        target_dtype = np.uint16
    elif source_dtype == np.int32:
        target_dtype = np.uint32
    elif source_dtype == np.int64:
        target_dtype = np.uint64
    else:
        # Default to uint32 for other types
        target_dtype = np.uint32
    
    # Normalize
    info = np.iinfo(source_dtype)
    normalized = normalize_array(array, 0, 1, info.min, info.max)
    
    # Convert
    info = np.iinfo(target_dtype)
    return (normalized * info.max).astype(target_dtype)


def ensure_range(
    array: np.ndarray,
    min_value: float = 0.0,
    max_value: float = 1.0
) -> np.ndarray:
    """
    Ensure an array is within a specific range.
    
    Args:
        array: Array to check
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        
    Returns:
        Array with values clipped to the specified range
    """
    return np.clip(array, min_value, max_value)


def ensure_shape(
    array: np.ndarray,
    shape: Tuple[int, ...],
    mode: str = "constant"
) -> np.ndarray:
    """
    Ensure an array has a specific shape, padding or cropping as needed.
    
    Args:
        array: Array to reshape
        shape: Target shape
        mode: Padding mode (constant, edge, linear_ramp, etc.)
        
    Returns:
        Reshaped array
    """
    # Check if reshaping is needed
    if array.shape == shape:
        return array
    
    # Calculate padding or cropping for each dimension
    pad_width = []
    for i, (current, target) in enumerate(zip(array.shape, shape)):
        if current < target:
            # Padding needed
            pad_before = (target - current) // 2
            pad_after = target - current - pad_before
            pad_width.append((pad_before, pad_after))
        elif current > target:
            # Cropping needed
            crop_before = (current - target) // 2
            crop_after = current - target - crop_before
            pad_width.append((-crop_before, -crop_after))
        else:
            # No change needed
            pad_width.append((0, 0))
    
    # Add padding for remaining dimensions
    for i in range(len(array.shape), len(shape)):
        pad_width.append((0, shape[i]))
    
    # Handle negative padding (cropping)
    slices = []
    for i, (pad_before, pad_after) in enumerate(pad_width):
        if pad_before < 0 or pad_after < 0:
            # Cropping
            start = abs(pad_before) if pad_before < 0 else 0
            end = array.shape[i] - abs(pad_after) if pad_after < 0 else array.shape[i]
            slices.append(slice(start, end))
        else:
            # No cropping
            slices.append(slice(None))
    
    # Crop if needed
    if any(pad_before < 0 or pad_after < 0 for pad_before, pad_after in pad_width):
        array = array[tuple(slices)]
    
    # Pad if needed
    pad_width = [(max(0, pad_before), max(0, pad_after)) for pad_before, pad_after in pad_width]
    if any(pad_before > 0 or pad_after > 0 for pad_before, pad_after in pad_width):
        array = np.pad(array, pad_width, mode=mode)
    
    return array 