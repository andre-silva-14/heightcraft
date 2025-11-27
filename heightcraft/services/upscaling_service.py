"""
Upscaling service for Heightcraft.

This module provides the UpscalingService class for enhancing height maps using AI upscaling.
"""

import logging
import os
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from heightcraft.core.config import UpscaleConfig
from heightcraft.core.exceptions import UpscalingError
from heightcraft.domain.height_map import HeightMap
from heightcraft.infrastructure.cache_manager import CacheManager
from heightcraft.infrastructure.file_storage import FileStorage
from heightcraft.infrastructure.gpu_manager import gpu_manager
from heightcraft.infrastructure.profiler import profiler
from heightcraft.models.upscaler import UpscalerModel


class UpscalingService:
    """
    Service for enhancing height maps using AI upscaling.
    
    This class provides methods for upscaling height maps using AI models.
    """
    
    def __init__(
        self, 
        config: Optional[UpscaleConfig] = None,
        cache_manager: Optional[CacheManager] = None,
        height_map_service: Optional['HeightMapService'] = None,
        file_storage: Optional[FileStorage] = None
    ):
        """
        Initialize the upscaling service.
        
        Args:
            config: Upscaling configuration
            cache_manager: Cache manager for caching upscaled height maps
            height_map_service: Service for height map I/O
            file_storage: File storage for loading pretrained models
        """
        self.config = config or UpscaleConfig()
        self.cache_manager = cache_manager or CacheManager()
        
        # Use HeightMapService for height map I/O
        if height_map_service:
            self.height_map_service = height_map_service
        else:
            # Import locally to avoid circular imports
            from heightcraft.services.height_map_service import HeightMapService
            self.height_map_service = HeightMapService()
            
        self.file_storage = file_storage or FileStorage()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model = None

    @profiler.profile()
    def upscale_file(
        self,
        input_file: str,
        output_file: str,
        scale_factor: Optional[int] = None,
        use_gpu: bool = True
    ) -> bool:
        """
        Upscale a height map from a file and save the result.

        Args:
            input_file: Path to the input height map file
            output_file: Path to save the upscaled height map
            scale_factor: Scale factor (2, 3, or 4) (overrides config)
            use_gpu: Whether to use GPU for upscaling

        Returns:
            True if the file was upscaled and saved successfully

        Raises:
            UpscalingError: If upscaling fails
        """
        try:
            # Load the height map using HeightMapService
            # We use a default resolution of 1.0 as we are primarily operating on the image data
            height_map = self.height_map_service.load_height_map(input_file, resolution=1.0)
            
            # Upscale the height map
            upscaled = self.upscale(height_map, scale_factor, use_gpu)
            
            # Save the upscaled height map using HeightMapService
            return self.height_map_service.save_height_map(upscaled, output_file)
        except Exception as e:
            raise UpscalingError(f"Failed to upscale file {input_file}: {e}")
    
    @profiler.profile()
    def upscale(
        self, 
        height_map: HeightMap,
        scale_factor: Optional[int] = None,
        use_gpu: bool = True
    ) -> HeightMap:
        """
        Upscale a height map.
        
        Args:
            height_map: The height map to upscale
            scale_factor: Scale factor (2, 3, or 4) (overrides config)
            use_gpu: Whether to use GPU for upscaling
            
        Returns:
            Upscaled height map
            
        Raises:
            UpscalingError: If upscaling fails
        """
        try:
            # Check if upscaling is enabled
            if not self.config.enabled and scale_factor is None:
                self.logger.info("Upscaling is disabled")
                return height_map
            
            # Use scale factor from arguments or config
            scale_factor = scale_factor if scale_factor is not None else self.config.upscale_factor
            
            # Validate scale factor
            if scale_factor not in [2, 3, 4]:
                raise UpscalingError(f"Invalid scale factor: {scale_factor}, must be 2, 3, or 4")
            
            self.logger.info(f"Upscaling height map by factor {scale_factor}")
            
            # Choose upscaling method
            if self.config.pretrained_model:
                # Use pretrained model
                upscaled_data = self._upscale_with_model(height_map.data, scale_factor, use_gpu)
            else:
                # Use bicubic interpolation
                upscaled_data = self._upscale_with_interpolation(height_map.data, scale_factor)
            
            # Create new height map
            upscaled_height_map = HeightMap(upscaled_data, height_map.bit_depth)
            
            self.logger.info(
                f"Upscaling complete: {height_map.width}x{height_map.height} -> "
                f"{upscaled_height_map.width}x{upscaled_height_map.height}"
            )
            
            return upscaled_height_map
            
        except Exception as e:
            raise UpscalingError(f"Failed to upscale height map: {e}")
    
    @profiler.profile()
    def _upscale_with_model(
        self, 
        data: np.ndarray,
        scale_factor: int,
        use_gpu: bool
    ) -> np.ndarray:
        """
        Upscale data using a pretrained model.
        
        Args:
            data: Height map data
            scale_factor: Scale factor (2, 3, or 4)
            use_gpu: Whether to use GPU for upscaling
            
        Returns:
            Upscaled data
            
        Raises:
            UpscalingError: If upscaling fails
        """
        try:
            # Load the model if not already loaded
            if self.model is None:
                self._load_model(use_gpu)
            
            # Normalize data to 0-1 range
            original_min = np.min(data)
            original_max = np.max(data)
            
            if original_max == original_min:
                data_normalized = np.zeros_like(data)
            else:
                data_normalized = (data - original_min) / (original_max - original_min)
            
            # Convert to tensor
            # Add batch and channel dimensions: (1, 1, H, W)
            tensor = torch.from_numpy(data_normalized).float().unsqueeze(0).unsqueeze(0)
            
            # Move to device
            device = next(self.model.parameters()).device
            tensor = tensor.to(device)
            
            # Run prediction
            with torch.no_grad():
                upscaled_tensor = self.model(tensor)
            
            # Move back to CPU and numpy
            upscaled_data = upscaled_tensor.cpu().numpy()[0, 0, :, :]
            
            # Restore original range
            upscaled_data = upscaled_data * (original_max - original_min) + original_min
            
            # Convert to original data type
            upscaled_data = upscaled_data.astype(data.dtype)
            
            return upscaled_data
            
        except Exception as e:
            raise UpscalingError(f"Failed to upscale with model: {e}")
    
    @profiler.profile()
    def _upscale_with_interpolation(self, data: np.ndarray, scale_factor: int) -> np.ndarray:
        """
        Upscale data using bicubic interpolation.
        
        Args:
            data: Height map data
            scale_factor: Scale factor (2, 3, or 4)
            
        Returns:
            Upscaled data
            
        Raises:
            UpscalingError: If upscaling fails
        """
        try:
            # Convert to tensor
            # Add batch and channel dimensions: (1, 1, H, W)
            tensor = torch.from_numpy(data).float().unsqueeze(0).unsqueeze(0)
            
            # Run resize operation
            upscaled_tensor = F.interpolate(
                tensor,
                scale_factor=scale_factor,
                mode='bicubic',
                align_corners=False
            )
            
            # Move back to numpy and remove dimensions
            upscaled_data = upscaled_tensor.numpy()[0, 0, :, :]
            
            # Convert to original data type
            upscaled_data = upscaled_data.astype(data.dtype)
            
            return upscaled_data
            
        except Exception as e:
            raise UpscalingError(f"Failed to upscale with interpolation: {e}")
    
    def _load_model(self, use_gpu: bool) -> None:
        """
        Load a pretrained upscaling model.
        
        Args:
            use_gpu: Whether to use GPU for the model
            
        Raises:
            UpscalingError: If the model cannot be loaded
        """
        try:
            self.logger.info(f"Loading pretrained model from {self.config.pretrained_model}")
            
            # Check if model file exists
            if not self.file_storage.file_exists(self.config.pretrained_model):
                raise UpscalingError(f"Pretrained model not found: {self.config.pretrained_model}")
            
            # Determine device
            device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
            self.logger.info(f"Using {device} for model inference")
            
            # Initialize model
            # We assume scale factor 2 for the default model architecture
            self.model = UpscalerModel(scale_factor=2)
            
            # Load weights
            state_dict = torch.load(self.config.pretrained_model, map_location=device)
            self.model.load_state_dict(state_dict)
            
            # Move model to device
            self.model.to(device)
            self.model.eval()
            
            self.logger.info("Pretrained model loaded successfully")
            
        except Exception as e:
            raise UpscalingError(f"Failed to load pretrained model: {e}")
    
    @classmethod
    def create_default_model(cls, output_path: str = "upscaler_model.pt") -> str:
        """
        Create a default upscaling model.
        
        Args:
            output_path: Path to save the model
            
        Returns:
            Path to the saved model
            
        Raises:
            UpscalingError: If the model cannot be created
        """
        try:
            logging.info("Creating default upscaling model")
            
            # Create model
            model = UpscalerModel(scale_factor=2)
            
            # Save model state dict
            torch.save(model.state_dict(), output_path)
            
            logging.info(f"Default upscaling model saved to {output_path}")
            return output_path
            
        except Exception as e:
            raise UpscalingError(f"Failed to create default model: {e}")
 