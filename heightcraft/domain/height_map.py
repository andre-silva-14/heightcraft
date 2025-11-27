"""
Height map domain model for Heightcraft.

This module provides the HeightMap class which represents a height map
generated from a point cloud.
"""

import logging
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

from heightcraft.core.config import OutputFormat
from heightcraft.core.exceptions import HeightMapGenerationError, HeightMapValidationError
from heightcraft.domain.point_cloud import PointCloud


class HeightMap:
    """
    Domain model representing a height map.
    
    A height map is a 2D image where the intensity (gray level) of each pixel
    represents the height at that point.
    """
    
    def __init__(self, data: np.ndarray, bit_depth: int = 16):
        """
        Initialize the height map.
        
        Args:
            data: The height map data as a 2D numpy array
            bit_depth: The bit depth of the height map (8 or 16)
            
        Raises:
            HeightMapGenerationError: If the data is invalid or bit depth is unsupported
        """
        self._validate_data(data)
        self._validate_bit_depth(bit_depth)
        
        # Normalize data to [0, 1] range
        if data.min() < 0 or data.max() > 1:
            min_val = data.min()
            max_val = data.max()
            # Avoid division by zero
            if min_val != max_val:
                data = (data - min_val) / (max_val - min_val)
            else:
                data = np.zeros_like(data)  # All values will be zero
        
        self._data = data
        self._bit_depth = bit_depth
        self._logger = logging.getLogger(self.__class__.__name__)
    
    @property
    def data(self) -> np.ndarray:
        """Get the height map data."""
        return self._data
    
    @property
    def width(self) -> int:
        """Get the width of the height map in pixels."""
        return self._data.shape[1]
    
    @property
    def height(self) -> int:
        """Get the height of the height map in pixels."""
        return self._data.shape[0]
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Get the shape (height, width) of the height map."""
        return self._data.shape
    
    @property
    def bit_depth(self) -> int:
        """Get the bit depth of the height map."""
        return self._bit_depth
    
    @property
    def max_value(self) -> int:
        """Get the maximum value based on bit depth."""
        return (2 ** self._bit_depth) - 1
    
    @property
    def min_height(self) -> float:
        """Get the minimum height value in the height map."""
        return float(np.min(self._data))
    
    @property
    def max_height(self) -> float:
        """Get the maximum height value in the height map."""
        return float(np.max(self._data))
    
    @property
    def min(self) -> float:
        """Get the minimum height value in the height map."""
        return self.min_height
    
    @property
    def max(self) -> float:
        """Get the maximum height value in the height map."""
        return self.max_height
    
    @property
    def aspect_ratio(self) -> float:
        """Get the aspect ratio (width / height) of the height map."""
        return self.width / self.height
    
    def save(self, 
             output_path: str, 
             format: Optional[OutputFormat] = None,
             colormap: str = "gray") -> bool:
        """
        Save the height map to a file.
        
        Args:
            output_path: The path to save the height map to
            format: The output format to use (default is determined from file extension)
            colormap: The colormap to use for visualization (only for image formats)
            
        Returns:
            True if the save was successful
            
        Raises:
            HeightMapGenerationError: If there was an error saving the height map
        """
        try:
            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            # Determine format
            if format is None:
                ext = os.path.splitext(output_path)[1].lower()
                if ext in ['.png']:
                    format = OutputFormat.PNG
                elif ext in ['.jpg', '.jpeg']:
                    format = OutputFormat.JPEG
                elif ext in ['.tif', '.tiff']:
                    format = OutputFormat.TIFF
            
            # If format is not specified and extension is unknown, or if it's not an image format,
            # fall back to numpy save (legacy behavior or for .npy files)
            if format is None and not output_path.endswith('.npy'):
                # Default to numpy save for backward compatibility if no format matches
                np.save(output_path, self._data)
                return True
                
            # Handle image formats
            if format in [OutputFormat.PNG, OutputFormat.JPEG, OutputFormat.TIFF]:
                from PIL import Image
                
                # Prepare data for PIL
                if self._bit_depth == 16:
                    # For 16-bit, we need to ensure data is uint16
                    if self._data.dtype != np.uint16:
                        # Normalize to 0-1 then scale to 65535
                        data_norm = (self._data - self.min_height) / (self.max_height - self.min_height) if self.max_height > self.min_height else np.zeros_like(self._data)
                        img_data = (data_norm * 65535).astype(np.uint16)
                    else:
                        img_data = self._data
                    
                    # Create 16-bit image using the newer Pillow API
                    # Convert to 'I' (32-bit integer) mode and then save as 16-bit
                    img = Image.fromarray(img_data.astype(np.uint16))
                else:
                    # For 8-bit, ensure data is uint8
                    if self._data.dtype != np.uint8:
                        # Normalize to 0-1 then scale to 255
                        data_norm = (self._data - self.min_height) / (self.max_height - self.min_height) if self.max_height > self.min_height else np.zeros_like(self._data)
                        img_data = (data_norm * 255).astype(np.uint8)
                    else:
                        img_data = self._data
                    
                    # Create 8-bit image
                    img = Image.fromarray(img_data, mode='L')
                
                # Save using PIL
                img.save(output_path)
                return True
            
            # Fallback to numpy save
            np.save(output_path, self._data)
            return True
            
        except Exception as e:
            raise HeightMapGenerationError(f"Error saving height map to {output_path}: {str(e)}")
    
    def to_mesh(self) -> Any:
        """
        Convert the height map to a mesh.
        
        Returns:
            A Mesh object
        
        Raises:
            HeightMapGenerationError: If there was an error creating the mesh
        """
        try:
            from heightcraft.domain.mesh import Mesh
            import trimesh
            
            # Create a grid of vertices
            height, width = self.shape
            y, x = np.mgrid[0:height, 0:width]
            
            # Combine x, y, and height values to form vertices
            vertices = np.column_stack([x.flatten(), y.flatten(), self._data.flatten()])
            
            # Create quad faces from the grid
            faces = []
            for i in range(height - 1):
                for j in range(width - 1):
                    # Get the indices of the four corners of this grid cell
                    top_left = i * width + j
                    top_right = i * width + (j + 1)
                    bottom_left = (i + 1) * width + j
                    bottom_right = (i + 1) * width + (j + 1)
                    
                    # Create two triangular faces (a quad split into two triangles)
                    faces.append([top_left, bottom_left, bottom_right])
                    faces.append([top_left, bottom_right, top_right])
            
            # Convert faces to a numpy array
            faces = np.array(faces)
            
            # Create a trimesh object
            trimesh_obj = trimesh.Trimesh(vertices=vertices, faces=faces)
            
            # Return a Mesh domain object
            return Mesh(trimesh_obj)
            
        except Exception as e:
            raise HeightMapGenerationError(f"Error creating mesh from height map: {str(e)}")
    
    def to_point_cloud(self) -> np.ndarray:
        """
        Convert the height map to a point cloud.
        
        Returns:
            A numpy array of 3D points
        
        Raises:
            HeightMapGenerationError: If there was an error creating the point cloud
        """
        try:
            # Create a grid of x and y coordinates
            height, width = self.shape
            y, x = np.mgrid[0:height, 0:width]
            
            # Extract the height values
            z = self._data
            
            # Combine x, y, and z into a single array of points
            points = np.column_stack([x.flatten(), y.flatten(), z.flatten()])
            
            return points
            
        except Exception as e:
            raise HeightMapGenerationError(f"Error creating point cloud from height map: {str(e)}")
    
    def resize(self, new_shape: Tuple[int, int]) -> "HeightMap":
        """
        Resize the height map to a new shape.
        
        Args:
            new_shape: The new shape (height, width) for the height map
            
        Returns:
            A new HeightMap with the specified shape
            
        Raises:
            HeightMapGenerationError: If there was an error resizing the height map
        """
        try:
            # Use scipy's zoom function to resize the height map
            zoom_factors = (new_shape[0] / self.height, new_shape[1] / self.width)
            resized_data = ndimage.zoom(self._data, zoom_factors, order=1)
            
            # Create and return a new HeightMap with the resized data
            return HeightMap(resized_data, self._bit_depth)
            
        except Exception as e:
            raise HeightMapGenerationError(f"Error resizing height map: {str(e)}")
    
    def crop(self, start: Tuple[int, int], end: Tuple[int, int]) -> "HeightMap":
        """
        Crop the height map to a specified region.
        
        Args:
            start: The starting coordinates (y, x) for the crop
            end: The ending coordinates (y, x) for the crop (exclusive)
            
        Returns:
            A new HeightMap containing only the specified region
            
        Raises:
            HeightMapGenerationError: If there was an error cropping the height map
        """
        try:
            # Extract the region
            cropped_data = self._data[start[0]:end[0], start[1]:end[1]]
            
            # Create and return a new HeightMap with the cropped data
            return HeightMap(cropped_data, self._bit_depth)
            
        except Exception as e:
            raise HeightMapGenerationError(f"Error cropping height map: {str(e)}")
    
    def split(self, grid_size: int) -> List["HeightMap"]:
        """
        Split the height map into a grid of smaller height maps.
        
        Args:
            grid_size: The number of tiles to split into (must be a perfect square or allow
                      rectangular division)
            
        Returns:
            A list of HeightMap objects representing the tiles
            
        Raises:
            HeightMapGenerationError: If there was an error splitting the height map
        """
        if grid_size <= 0:
            raise HeightMapGenerationError(f"Grid size must be positive, got {grid_size}")
        
        if grid_size == 1:
            return [self]
        
        result = []
        
        # Calculate tile dimensions
        tile_height = self.height // grid_size
        tile_width = self.width // grid_size
        
        # Create tiles
        for i in range(grid_size):
            for j in range(grid_size):
                # Calculate tile boundaries
                y_start = i * tile_height
                y_end = (i + 1) * tile_height if i < grid_size - 1 else self.height
                x_start = j * tile_width
                x_end = (j + 1) * tile_width if j < grid_size - 1 else self.width
                
                # Extract tile
                tile_data = self._data[y_start:y_end, x_start:x_end]
                
                # Create new height map for tile
                tile = HeightMap(tile_data, self._bit_depth)
                result.append(tile)
        
        return result
    
    def convert_to_8bit(self) -> "HeightMap":
        """
        Convert the height map to 8-bit precision.
        
        Returns:
            A new HeightMap with 8-bit precision
            
        Raises:
            HeightMapGenerationError: If there was an error converting the height map
        """
        if self._bit_depth == 8:
            return self
        
        # Normalize to 0-1 range
        normalized = (self._data - np.min(self._data)) / (np.max(self._data) - np.min(self._data))
        
        # Convert to 8-bit
        data_8bit = (normalized * 255).astype(np.uint8)
        return HeightMap(data_8bit, 8)
    
    def convert_to_16bit(self) -> "HeightMap":
        """
        Convert the height map to 16-bit precision.
        
        Returns:
            A new HeightMap with 16-bit precision
            
        Raises:
            HeightMapGenerationError: If there was an error converting the height map
        """
        if self._bit_depth == 16:
            return self
        
        # Normalize to 0-1 range
        normalized = self._data.astype(float) / 255
        
        # Convert to 16-bit
        data_16bit = (normalized * 65535).astype(np.uint16)
        return HeightMap(data_16bit, 16)
    
    def to_dict(self) -> Dict:
        """
        Convert the height map to a dictionary representation.
        
        Returns:
            A dictionary containing the height map's attributes
        """
        return {
            "width": self.width,
            "height": self.height,
            "aspect_ratio": self.aspect_ratio,
            "min": self.min_height,
            "max": self.max_height,
            "bit_depth": self.bit_depth
        }
    
    @staticmethod
    def _validate_data(data: np.ndarray) -> None:
        """
        Validate the height map data.
        
        Args:
            data: The data to validate
            
        Raises:
            HeightMapValidationError: If the data is invalid
        """
        if not isinstance(data, np.ndarray):
            raise HeightMapValidationError(f"Expected numpy array, got {type(data)}")
        
        if len(data.shape) != 2:
            raise HeightMapValidationError(f"Expected 2D array, got {len(data.shape)}D")
    
    @staticmethod
    def _validate_bit_depth(bit_depth: int) -> None:
        """
        Validate the bit depth.
        
        Args:
            bit_depth: The bit depth to validate
            
        Raises:
            HeightMapValidationError: If the bit depth is invalid
        """
        if bit_depth not in [8, 16]:
            raise HeightMapValidationError(f"Bit depth must be 8 or 16, got {bit_depth}")
    
    @classmethod
    def from_file(cls, file_path: str, bit_depth: Optional[int] = None) -> "HeightMap":
        """
        Load a height map from a file.
        
        Args:
            file_path: The path to the file
            bit_depth: The bit depth to use (required for compatibility)
            
        Returns:
            A new HeightMap object
            
        Raises:
            HeightMapValidationError: If there was an error loading the height map or bit_depth is missing
        """
        try:
            warnings.warn(
                "HeightMap.from_file is deprecated and will be removed in a future version.",
                DeprecationWarning,
                stacklevel=2
            )
            
            if bit_depth is None:
                raise HeightMapValidationError("bit_depth is required for compatibility")
            
            # Load the data
            data = np.load(file_path)
            
            # Create and return a new HeightMap
            return cls(data, bit_depth)
            
        except Exception as e:
            if isinstance(e, HeightMapValidationError):
                raise
            raise HeightMapValidationError(f"Error loading height map from {file_path}: {str(e)}")
    
    @classmethod
    def from_point_cloud(
        cls, 
        point_cloud: PointCloud, 
        resolution: Tuple[int, int],
        bit_depth: int = 16
    ) -> "HeightMap":
        """
        Create a height map from a point cloud.
        
        Args:
            point_cloud: The point cloud to convert
            resolution: The resolution (height, width) of the height map
            bit_depth: The bit depth of the height map (8 or 16)
            
        Returns:
            A new HeightMap object
            
        Raises:
            HeightMapGenerationError: If there was an error creating the height map
        """
        width, height = resolution
        
        # Initialize height map
        height_map = np.zeros((height, width), dtype=np.float32)
        
        # Get 2D points and Z values
        points_2d = point_cloud.get_xy_points()
        z_values = point_cloud.normalize_z()
        
        # Get bounds
        bounds = point_cloud.bounds
        
        # Calculate pixel coordinates
        x_coords = (
            (points_2d[:, 0] - bounds["min_x"])
            / (bounds["max_x"] - bounds["min_x"])
            * (width - 1)
        ).astype(int)
        
        y_coords = (
            (points_2d[:, 1] - bounds["min_y"])
            / (bounds["max_y"] - bounds["min_y"])
            * (height - 1)
        ).astype(int)
        
        # Assign Z values to pixels (maximum value at each coordinate)
        np.maximum.at(height_map, (y_coords, x_coords), z_values)
        
        # Convert to the proper bit depth
        if bit_depth == 8:
            data = (height_map * 255).astype(np.uint8)
        else:  # bit_depth == 16
            data = (height_map * 65535).astype(np.uint16)
        
        return cls(data, bit_depth) 