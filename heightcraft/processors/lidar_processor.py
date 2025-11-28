"""
LiDAR processor for Heightcraft.

This module provides a processor for handling LiDAR data.
"""

import logging
import os

from typing import Optional
import numpy as np

from heightcraft.core.config import ApplicationConfig
from heightcraft.core.exceptions import ProcessingError
from heightcraft.domain.point_cloud import PointCloud
from heightcraft.infrastructure.lidar_repository import LidarRepository
from heightcraft.processors.base_processor import BaseProcessor
from heightcraft.services.height_map_service import HeightMapService


class LidarProcessor(BaseProcessor):
    """Processor for LiDAR data."""
    
    def __init__(self, config: ApplicationConfig):
        """
        Initialize the LiDAR processor.
        
        Args:
            config: Application configuration
        """
        super().__init__(config)
        self.lidar_repository = LidarRepository()
        self.height_map_service = HeightMapService()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def load_model(self) -> None:
        """
        Load the LiDAR data.
        
        For streaming, we just get the bounds here.
        """
        file_path = self.config.model_config.file_path
        if not file_path:
            raise ProcessingError("No file path provided")
            
        self.logger.info(f"Getting bounds for LiDAR file: {file_path}")
        self.bounds = self.lidar_repository.get_bounds(file_path)
        self.logger.info(f"Bounds: {self.bounds}")

    def sample_points(self) -> np.ndarray:
        """
        Return the points loaded from LiDAR.
        
        For streaming, this is not used in the main loop, but we implement it for compatibility.
        """
        return np.array([]) # Dummy return

    def generate_height_map(self) -> np.ndarray:
        """
        Generate height map from points.
        
        For streaming, this is not used in the main loop.
        """
        return np.array([]) # Dummy return

    def save_height_map(self, output_path: Optional[str] = None) -> str:
        """Save the height map."""
        if self.height_map_obj is None:
            raise ProcessingError("Height map not generated")
            
        path = output_path or self.config.output_config.output_path
        self.height_map_service.save_height_map(self.height_map_obj, path)
        return path

    def upscale_height_map(self) -> None:
        """Upscale the height map."""
        if not self.config.upscale_config.enabled:
            return
            
        if self.height_map_obj is None:
            self.logger.warning("No height map to upscale")
            return
            
        self.logger.info(f"Upscaling height map by factor {self.config.upscale_config.upscale_factor}")
        
        try:
            # Import locally to avoid circular imports
            from heightcraft.services.upscaling_service import UpscalingService
            
            upscaling_service = UpscalingService(self.config.upscale_config)
            self.height_map_obj = upscaling_service.upscale(self.height_map_obj)
            self.height_map = self.height_map_obj.data
            
        except Exception as e:
            self.logger.error(f"Upscaling failed: {e}")
            raise ProcessingError(f"Upscaling failed: {e}")

    def process(self) -> str:
        """
        Process the LiDAR data and generate a height map using streaming.
        
        Returns:
            Path to the generated height map
            
        Raises:
            ProcessingError: If processing fails
        """
        try:
            file_path = self.config.model_config.file_path
            if not file_path:
                raise ProcessingError("No file path provided")
                
            self.logger.info(f"Processing LiDAR file: {file_path}")
            
            # 1. Get Bounds
            self.load_model()
            
            # 2. Determine Resolution
            width = self.config.height_map_config.max_resolution
            height = self.config.height_map_config.max_resolution
            
            # Adjust aspect ratio
            x_range = self.bounds["max_x"] - self.bounds["min_x"]
            y_range = self.bounds["max_y"] - self.bounds["min_y"]
            aspect_ratio = x_range / y_range if y_range > 0 else 1.0
            
            if aspect_ratio > 1:
                height = int(width / aspect_ratio)
            else:
                width = int(height * aspect_ratio)
            
            self.logger.info(f"Target resolution: {width}x{height}")
            
            # 3. Initialize Height Map Grid
            # We use float32 for accumulation
            height_map_data = np.full((height, width), -np.inf, dtype=np.float32)
            
            # 4. Stream Chunks
            chunk_size = self.config.model_config.chunk_size
            
            # Pre-calculate ranges
            z_range = self.bounds["max_z"] - self.bounds["min_z"]
            if z_range == 0: z_range = 1.0
            
            # We need to manually normalize Z values relative to the GLOBAL bounds
            # The HeightMapService.generate_from_point_cloud normalizes based on the chunk's bounds,
            # which is wrong for streaming. We need to project points manually or use a modified service method.
            # Let's do it manually here for efficiency.
            
            for i, chunk_pc in enumerate(self.lidar_repository.get_chunk_iterator(file_path, chunk_size)):
                self.logger.debug(f"Processing chunk {i+1} ({chunk_pc.size} points)")
                
                points_2d = chunk_pc.get_xy_points()
                z_values = chunk_pc.z
                
                # Normalize Z globally
                z_normalized = (z_values - self.bounds["min_z"]) / z_range
                
                # Calculate pixel coordinates
                x_coords = (
                    (points_2d[:, 0] - self.bounds["min_x"])
                    / x_range
                    * (width - 1)
                ).astype(int)
                
                y_coords = (
                    (points_2d[:, 1] - self.bounds["min_y"])
                    / y_range
                    * (height - 1)
                ).astype(int)
                
                # Clip coordinates
                x_coords = np.clip(x_coords, 0, width - 1)
                y_coords = np.clip(y_coords, 0, height - 1)
                
                # Update grid (max projection)
                np.maximum.at(height_map_data, (y_coords, x_coords), z_normalized)
            
            # Replace -inf with 0 (or min value)
            height_map_data[height_map_data == -np.inf] = 0.0
            
            # 5. Create Height Map Object
            from heightcraft.domain.height_map import HeightMap
            
            # Convert to target bit depth
            bit_depth = self.config.height_map_config.bit_depth
            
            if bit_depth == 8:
                data = (height_map_data * 255).astype(np.uint8)
            elif bit_depth == 16:
                data = (height_map_data * 65535).astype(np.uint16)
            else: # 32-bit
                data = height_map_data
                
            self.height_map_obj = HeightMap(data, bit_depth)
            self.height_map = self.height_map_obj.data
            
            # 6. Upscale (if enabled)
            if self.config.upscale_config.enabled:
                self.upscale_height_map()
            
            # 7. Save
            output_path = self.save_height_map()
            
            self.logger.info(f"Saved height map to {output_path}")
            return output_path
            
        except Exception as e:
            raise ProcessingError(f"Failed to process LiDAR data: {str(e)}")
