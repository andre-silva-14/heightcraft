"""
Image processor for Heightcraft.

This module provides a processor for handling image-to-image operations,
primarily for upscaling existing height maps.
"""

import logging
import os
from typing import Optional
import numpy as np

from heightcraft.core.config import ApplicationConfig
from heightcraft.core.exceptions import ProcessingError
from heightcraft.processors.base_processor import BaseProcessor
from heightcraft.services.height_map_service import HeightMapService
from heightcraft.services.upscaling_service import UpscalingService
from heightcraft.domain.height_map import HeightMap


class ImageProcessor(BaseProcessor):
    """Processor for image data (height map to height map)."""
    
    def __init__(self, config: ApplicationConfig):
        """
        Initialize the image processor.
        
        Args:
            config: Application configuration
        """
        super().__init__(config)
        self.height_map_service = HeightMapService()
        self.upscaling_service = UpscalingService(config.upscale_config)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def load_model(self) -> None:
        """
        Load the input image as a height map.
        """
        file_path = self.config.model_config.file_path
        if not file_path:
            raise ProcessingError("No file path provided")
            
        self.logger.info(f"Loading input image: {file_path}")
        try:
            # We don't need resolution here as we load the image as-is
            # But load_height_map might expect it or use it for validation
            # Let's use HeightMap.from_file directly or via service if available
            # Service load_height_map takes resolution but it might be optional or ignored for from_file
            # Actually HeightMapService.load_height_map implementation:
            # def load_height_map(self, file_path: str, resolution: Union[float, Tuple[int, int]] = 1.0) -> HeightMap:
            # It calls HeightMap.from_file.
            
            # We need to know the bit depth of the input to load it correctly?
            # HeightMap.from_file takes bit_depth arg.
            # But for images (PIL), it might auto-detect.
            # Let's try loading without enforcing bit depth first, or assume 8/16 based on config?
            # No, config.height_map_config.bit_depth is for OUTPUT.
            # We should let HeightMap.from_file detect it or default to something safe.
            # However, HeightMap.from_file requires bit_depth for compatibility (raises error if None).
            # Let's try to infer it or pass 16 as a safe default for loading?
            # Actually, for image inputs, we want to preserve the input bit depth if possible, 
            # or load it into a HeightMap which will normalize it anyway.
            
            # Let's use a safe default of 16 for loading, as it can hold 8-bit too.
            self.height_map_obj = HeightMap.from_file(file_path, bit_depth=16)
            self.height_map = self.height_map_obj.data
            
        except Exception as e:
            raise ProcessingError(f"Failed to load input image: {e}")

    def sample_points(self) -> np.ndarray:
        """
        Not used for image processing.
        """
        return np.array([])

    def generate_height_map(self) -> np.ndarray:
        """
        Not used for image processing.
        """
        return self.height_map

    def upscale_height_map(self) -> None:
        """Upscale the height map."""
        if not self.config.upscale_config.enabled:
            return
            
        if self.height_map_obj is None:
            self.logger.warning("No height map to upscale")
            return
            
        self.logger.info(f"Upscaling height map by factor {self.config.upscale_config.upscale_factor}")
        
        try:
            self.height_map_obj = self.upscaling_service.upscale(self.height_map_obj)
            self.height_map = self.height_map_obj.data
            
        except Exception as e:
            self.logger.error(f"Upscaling failed: {e}")
            raise ProcessingError(f"Upscaling failed: {e}")

    def save_height_map(self, output_path: Optional[str] = None) -> str:
        """Save the height map."""
        if self.height_map_obj is None:
            raise ProcessingError("Height map not generated")
            
        path = output_path or self.config.output_config.output_path
        self.height_map_service.save_height_map(self.height_map_obj, path)
        return path

    def process(self) -> str:
        """
        Process the image.
        
        Returns:
            Path to the generated height map
            
        Raises:
            ProcessingError: If processing fails
        """
        # Check if upscaling is enabled
        if not self.config.upscale_config.enabled:
            self.logger.warning("Image provided but upscaling is NOT enabled. Nothing to do.")
            self.logger.warning("Use --upscale to enable AI upscaling for images.")
            return ""
            
        try:
            # 1. Load Image
            self.load_model()
            
            # 2. Upscale
            self.upscale_height_map()
            
            # 3. Convert to target bit depth
            target_bit_depth = self.config.height_map_config.bit_depth
            current_bit_depth = self.height_map_obj.bit_depth
            
            if target_bit_depth != current_bit_depth:
                self.logger.info(f"Converting bit depth: {current_bit_depth}-bit -> {target_bit_depth}-bit")
                if target_bit_depth == 8:
                    self.height_map_obj = self.height_map_obj.convert_to_8bit()
                elif target_bit_depth == 16:
                    self.height_map_obj = self.height_map_obj.convert_to_16bit()
                # For 32-bit, HeightMap constructor handles it if we re-create, 
                # but we don't have a convert_to_32bit method.
                # However, HeightMap.save handles 32-bit saving if the object says it's 32-bit?
                # Or we can manually create a new HeightMap with 32-bit depth.
                elif target_bit_depth == 32:
                     self.height_map_obj = HeightMap(self.height_map_obj.data, bit_depth=32)
            
            # 4. Save
            output_path = self.config.output_config.output_path
            self.height_map_service.save_height_map(self.height_map_obj, output_path)
            
            self.logger.info(f"Saved processed image to {output_path}")
            return output_path
            
        except Exception as e:
            raise ProcessingError(f"Failed to process image: {str(e)}")
