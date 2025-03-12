"""
Data converter utility for converting between different data formats.

This module provides functionality for converting between different data formats,
such as numpy arrays, images, and domain objects.
"""

import numpy as np
from PIL import Image
from scipy import ndimage
from typing import Optional, Tuple, Union, Any

from heightcraft.core.exceptions import ConversionError
from heightcraft.domain.mesh import Mesh
from heightcraft.domain.height_map import HeightMap
from heightcraft.domain.point_cloud import PointCloud


class DataConverter:
    """Utility for converting between different data formats."""
    
    def __init__(self) -> None:
        """Initialize the data converter."""
        pass
    
    def normalize_array(self, array: np.ndarray) -> np.ndarray:
        """
        Normalize an array to the range [0, 1].
        
        Args:
            array: The array to normalize.
            
        Returns:
            The normalized array.
            
        Raises:
            ConversionError: If the array cannot be normalized.
        """
        try:
            # Check that the array is a numpy array
            if not isinstance(array, np.ndarray):
                raise ConversionError(f"Expected numpy array, got {type(array)}")
            
            # Check that the array is not empty
            if array.size == 0:
                raise ConversionError("Array is empty")
            
            # Get the minimum and maximum values
            min_val = np.min(array)
            max_val = np.max(array)
            
            # Check if the array is constant
            if max_val == min_val:
                raise ConversionError("Array has constant values, cannot normalize")
            
            # Normalize the array
            normalized = (array - min_val) / (max_val - min_val)
            
            return normalized
        
        except Exception as e:
            if isinstance(e, ConversionError):
                raise
            raise ConversionError(f"Failed to normalize array: {str(e)}")
    
    def array_to_image(self, array: np.ndarray, mode: str = "L") -> Image.Image:
        """
        Convert a numpy array to a PIL Image.
        
        Args:
            array: The array to convert.
            mode: The mode of the image (e.g. "L" for grayscale, "RGB" for color).
            
        Returns:
            The converted image.
            
        Raises:
            ConversionError: If the array cannot be converted.
        """
        try:
            # Check that the array is a numpy array
            if not isinstance(array, np.ndarray):
                raise ConversionError(f"Expected numpy array, got {type(array)}")
            
            # Check that the array is 2D or 3D
            if array.ndim not in [2, 3]:
                raise ConversionError(f"Array must be 2D or 3D, got {array.ndim}D")
            
            # If 3D, check that it has 3 or 4 channels (RGB or RGBA)
            if array.ndim == 3 and array.shape[2] not in [3, 4]:
                raise ConversionError(f"3D array must have 3 or 4 channels, got {array.shape[2]}")
            
            # Convert the array to an image
            image = Image.fromarray(array, mode=mode)
            
            return image
        
        except Exception as e:
            if isinstance(e, ConversionError):
                raise
            raise ConversionError(f"Failed to convert array to image: {str(e)}")
    
    def image_to_array(self, image: Image.Image) -> np.ndarray:
        """
        Convert a PIL Image to a numpy array.
        
        Args:
            image: The image to convert.
            
        Returns:
            The converted array.
            
        Raises:
            ConversionError: If the image cannot be converted.
        """
        try:
            # Check that the image is a PIL Image
            if not isinstance(image, Image.Image):
                raise ConversionError(f"Expected PIL Image, got {type(image)}")
            
            # Get the image size
            width, height = image.size
            
            # Convert the image to a numpy array
            data = list(image.getdata())
            
            # Reshape the data to match the image dimensions
            if isinstance(data[0], tuple):
                # Color image
                num_channels = len(data[0])
                array = np.array(data).reshape((height, width, num_channels))
            else:
                # Grayscale image
                array = np.array(data).reshape((height, width))
            
            return array
        
        except Exception as e:
            if isinstance(e, ConversionError):
                raise
            raise ConversionError(f"Failed to convert image to array: {str(e)}")
    
    def resize_array(self, array: np.ndarray, new_shape: Tuple[int, int]) -> np.ndarray:
        """
        Resize a 2D array to a new shape.
        
        Args:
            array: The array to resize.
            new_shape: The new shape (height, width).
            
        Returns:
            The resized array.
            
        Raises:
            ConversionError: If the array cannot be resized.
        """
        try:
            # Check that the array is a numpy array
            if not isinstance(array, np.ndarray):
                raise ConversionError(f"Expected numpy array, got {type(array)}")
            
            # Check that the array is 2D
            if array.ndim != 2:
                raise ConversionError(f"Array must be 2D, got {array.ndim}D")
            
            # Check that the new shape is valid
            if not isinstance(new_shape, tuple) or len(new_shape) != 2:
                raise ConversionError(f"Expected new_shape to be a tuple of length 2, got {new_shape}")
            
            # Check that the new dimensions are positive
            new_height, new_width = new_shape
            if new_width <= 0 or new_height <= 0:
                raise ConversionError(f"New dimensions must be positive, got {new_width}x{new_height}")
            
            # Resize the array
            zoom_factors = (new_height / array.shape[0], new_width / array.shape[1])
            resized = ndimage.zoom(array, zoom_factors, order=1)
            
            return resized
        
        except Exception as e:
            if isinstance(e, ConversionError):
                raise
            raise ConversionError(f"Failed to resize array: {str(e)}")
    
    def crop_array(self, array: np.ndarray, x_min: int, y_min: int, width: int, height: int) -> np.ndarray:
        """
        Crop a 2D array.
        
        Args:
            array: The array to crop.
            x_min: The minimum x-coordinate.
            y_min: The minimum y-coordinate.
            width: The width of the crop.
            height: The height of the crop.
            
        Returns:
            The cropped array.
            
        Raises:
            ConversionError: If the array cannot be cropped.
        """
        try:
            # Check that the array is a numpy array
            if not isinstance(array, np.ndarray):
                raise ConversionError(f"Expected numpy array, got {type(array)}")
            
            # Check that the array is 2D
            if array.ndim != 2:
                raise ConversionError(f"Array must be 2D, got {array.ndim}D")
            
            # Check that the crop parameters are valid
            if x_min < 0 or y_min < 0:
                raise ConversionError(f"Crop origin must be non-negative, got ({x_min}, {y_min})")
            
            if width <= 0 or height <= 0:
                raise ConversionError(f"Crop dimensions must be positive, got {width}x{height}")
            
            if x_min + width > array.shape[1] or y_min + height > array.shape[0]:
                raise ConversionError(f"Crop region ({x_min}, {y_min}, {width}x{height}) is outside array bounds {array.shape}")
            
            # Crop the array
            cropped = array[y_min:y_min+height, x_min:x_min+width]
            
            return cropped
        
        except Exception as e:
            if isinstance(e, ConversionError):
                raise
            raise ConversionError(f"Failed to crop array: {str(e)}")
    
    def apply_gaussian_blur(self, array: np.ndarray, sigma: float) -> np.ndarray:
        """
        Apply a Gaussian blur to a 2D array.
        
        Args:
            array: The array to blur.
            sigma: The standard deviation of the Gaussian kernel.
            
        Returns:
            The blurred array.
            
        Raises:
            ConversionError: If the blur cannot be applied.
        """
        try:
            # Check that the array is a numpy array
            if not isinstance(array, np.ndarray):
                raise ConversionError(f"Expected numpy array, got {type(array)}")
            
            # Check that sigma is positive
            if sigma <= 0:
                raise ConversionError(f"Sigma must be positive, got {sigma}")
            
            # Apply Gaussian blur
            blurred = ndimage.gaussian_filter(array, sigma=sigma)
            
            return blurred
        
        except Exception as e:
            if isinstance(e, ConversionError):
                raise
            raise ConversionError(f"Failed to apply Gaussian blur: {str(e)}")
    
    def apply_median_filter(self, array: np.ndarray, kernel_size: int) -> np.ndarray:
        """
        Apply a median filter to a 2D array.
        
        Args:
            array: The array to filter.
            kernel_size: The size of the kernel. Must be odd.
            
        Returns:
            The filtered array.
            
        Raises:
            ConversionError: If the filter cannot be applied.
        """
        try:
            # Check that the array is a numpy array
            if not isinstance(array, np.ndarray):
                raise ConversionError(f"Expected numpy array, got {type(array)}")
            
            # Check that kernel_size is positive
            if kernel_size <= 0:
                raise ConversionError(f"Kernel size must be positive, got {kernel_size}")
            
            # Check that kernel_size is odd
            if kernel_size % 2 == 0:
                raise ConversionError(f"Kernel size must be odd, got {kernel_size}")
            
            # Apply median filter
            filtered = ndimage.median_filter(array, size=kernel_size)
            
            return filtered
        
        except Exception as e:
            if isinstance(e, ConversionError):
                raise
            raise ConversionError(f"Failed to apply median filter: {str(e)}")
    
    def equalize_histogram(self, array: np.ndarray) -> np.ndarray:
        """
        Equalize the histogram of a 2D array.
        
        Args:
            array: The array to equalize.
            
        Returns:
            The equalized array.
            
        Raises:
            ConversionError: If the histogram cannot be equalized.
        """
        try:
            # Check that the array is a numpy array
            if not isinstance(array, np.ndarray):
                raise ConversionError(f"Expected numpy array, got {type(array)}")
            
            # Get the data and its range
            min_val = np.min(array)
            max_val = np.max(array)
            
            if max_val == min_val:
                # If all values are the same, just return a copy
                return array.copy()
            
            # Normalize to [0, 1] for processing
            norm_data = (array - min_val) / (max_val - min_val)
            
            # Convert to 8-bit for histogram equalization
            data_8bit = (norm_data * 255).astype(np.uint8)
            
            # Calculate the histogram
            hist, bins = np.histogram(data_8bit.flatten(), 256, [0, 256])
            
            # Calculate the cumulative distribution function
            cdf = hist.cumsum()
            
            # Normalize the CDF
            cdf_normalized = cdf * float(norm_data.max()) / cdf[-1]
            
            # Apply the equalization
            equalized_data = np.interp(data_8bit.flatten(), range(256), cdf_normalized)
            
            # Reshape back to original shape
            equalized_data = equalized_data.reshape(array.shape)
            
            # Scale back to original range
            equalized_data = (equalized_data / equalized_data.max()) * (max_val - min_val) + min_val
            
            return equalized_data.astype(array.dtype)
        
        except Exception as e:
            if isinstance(e, ConversionError):
                raise
            raise ConversionError(f"Failed to equalize histogram: {str(e)}")
    
    def convert_mesh_to_point_cloud(self, mesh: Mesh) -> PointCloud:
        """
        Convert a mesh to a point cloud.
        
        Args:
            mesh: The mesh to convert.
            
        Returns:
            The converted point cloud.
            
        Raises:
            ConversionError: If the mesh cannot be converted.
        """
        try:
            # Check that the mesh is a Mesh object
            if not isinstance(mesh, Mesh):
                raise ConversionError(f"Expected Mesh object, got {type(mesh)}")
            
            # Convert the mesh to a point cloud
            return mesh.to_point_cloud()
        
        except Exception as e:
            if isinstance(e, ConversionError):
                raise
            raise ConversionError(f"Failed to convert mesh to point cloud: {str(e)}")
    
    def convert_height_map_to_mesh(self, height_map: HeightMap) -> Mesh:
        """
        Convert a height map to a mesh.
        
        Args:
            height_map: The height map to convert.
            
        Returns:
            The converted mesh.
            
        Raises:
            ConversionError: If the height map cannot be converted.
        """
        try:
            # Check that the height map is a HeightMap object
            if not isinstance(height_map, HeightMap):
                raise ConversionError(f"Expected HeightMap object, got {type(height_map)}")
            
            # Convert the height map to a mesh
            return height_map.to_mesh()
        
        except Exception as e:
            if isinstance(e, ConversionError):
                raise
            raise ConversionError(f"Failed to convert height map to mesh: {str(e)}")
    
    def _validate_array(self, array: np.ndarray) -> None:
        """
        Validate a numpy array.
        
        Args:
            array: The array to validate
            
        Raises:
            ConversionError: If the array is invalid
        """
        if not isinstance(array, np.ndarray):
            raise ConversionError(f"Expected np.ndarray, got {type(array)}")
            
    def _validate_image(self, image) -> None:
        """
        Validate an image.
        
        Args:
            image: The image to validate
            
        Raises:
            ConversionError: If the image is invalid
        """
        # Check for PIL Image type (dynamically since we don't want to import PIL here)
        if not hasattr(image, 'size') or not hasattr(image, 'getdata'):
            raise ConversionError(f"Expected PIL.Image.Image, got {type(image)}")
            
    def _validate_mesh(self, mesh) -> None:
        """
        Validate a mesh.
        
        Args:
            mesh: The mesh to validate
            
        Raises:
            ConversionError: If the mesh is invalid
        """
        # Check if it's a trimesh.Trimesh (dynamically since we don't want to import trimesh here)
        if not hasattr(mesh, 'vertices') or not hasattr(mesh, 'faces'):
            raise ConversionError(f"Expected trimesh.Trimesh, got {type(mesh)}")
            
    def _validate_height_map(self, height_map) -> None:
        """
        Validate a height map.
        
        Args:
            height_map: The height map to validate
            
        Raises:
            ConversionError: If the height map is invalid
        """
        # Check for HeightMap class (dynamically since we handle circular imports)
        if not hasattr(height_map, 'width') or not hasattr(height_map, 'height'):
            raise ConversionError(f"Expected HeightMap, got {type(height_map)}") 