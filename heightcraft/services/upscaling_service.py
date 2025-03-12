"""
Upscaling service for Heightcraft.

This module provides the UpscalingService class for enhancing height maps using AI upscaling.
"""

import logging
import os
from typing import Optional, Tuple, Union

import numpy as np

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

from heightcraft.core.config import UpscaleConfig
from heightcraft.core.exceptions import UpscalingError
from heightcraft.domain.height_map import HeightMap
from heightcraft.infrastructure.cache_manager import CacheManager
from heightcraft.infrastructure.file_storage import FileStorage
from heightcraft.infrastructure.gpu_manager import gpu_manager
from heightcraft.infrastructure.profiler import profiler


class UpscalingService:
    """
    Service for enhancing height maps using AI upscaling.
    
    This class provides methods for upscaling height maps using AI models.
    """
    
    def __init__(
        self, 
        config: Optional[UpscaleConfig] = None,
        cache_manager: Optional[CacheManager] = None,
        file_storage: Optional[FileStorage] = None
    ):
        """
        Initialize the upscaling service.
        
        Args:
            config: Upscaling configuration
            cache_manager: Cache manager for caching upscaled height maps
            file_storage: File storage for loading pretrained models
        """
        self.config = config or UpscaleConfig()
        self.cache_manager = cache_manager or CacheManager()
        self.file_storage = file_storage or FileStorage()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model = None
    
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
        if not TF_AVAILABLE:
            raise UpscalingError("TensorFlow is not available for model-based upscaling")
        
        try:
            # Load the model if not already loaded
            if self.model is None:
                self._load_model(use_gpu)
            
            # Normalize data to 0-1 range
            original_min = np.min(data)
            original_max = np.max(data)
            data_normalized = (data - original_min) / (original_max - original_min)
            
            # Add batch and channel dimensions
            data_input = data_normalized.reshape(1, *data.shape, 1)
            
            # Run prediction
            with tf.device("/GPU:0" if use_gpu and gpu_manager.has_gpu else "/CPU:0"):
                upscaled_data = self.model.predict(data_input)[0, :, :, 0]
            
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
            # Calculate new dimensions
            new_height = data.shape[0] * scale_factor
            new_width = data.shape[1] * scale_factor
            
            # Use TensorFlow for interpolation if available
            if TF_AVAILABLE:
                # Add batch and channel dimensions
                data_input = data.reshape(1, *data.shape, 1)
                
                # Run resize operation
                resized = tf.image.resize(
                    data_input,
                    [new_height, new_width],
                    method=tf.image.ResizeMethod.BICUBIC
                )
                
                # Remove batch and channel dimensions
                upscaled_data = resized[0, :, :, 0].numpy()
            else:
                # Use scipy for interpolation
                from scipy.ndimage import zoom
                
                # Apply zoom
                upscaled_data = zoom(data, scale_factor, order=3)
            
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
        if not TF_AVAILABLE:
            raise UpscalingError("TensorFlow is not available for model loading")
        
        try:
            self.logger.info(f"Loading pretrained model from {self.config.pretrained_model}")
            
            # Check if model file exists
            if not self.file_storage.file_exists(self.config.pretrained_model):
                raise UpscalingError(f"Pretrained model not found: {self.config.pretrained_model}")
            
            # Configure TensorFlow
            if not use_gpu or not gpu_manager.has_gpu:
                self.logger.info("Using CPU for model inference")
                tf.config.set_visible_devices([], 'GPU')
            else:
                self.logger.info("Using GPU for model inference")
            
            # Load the model
            self.model = tf.keras.models.load_model(self.config.pretrained_model)
            
            self.logger.info("Pretrained model loaded successfully")
            
        except Exception as e:
            raise UpscalingError(f"Failed to load pretrained model: {e}")
    
    @classmethod
    def create_default_model(cls, output_path: str = "upscaler_model.h5") -> str:
        """
        Create a default upscaling model.
        
        Args:
            output_path: Path to save the model
            
        Returns:
            Path to the saved model
            
        Raises:
            UpscalingError: If the model cannot be created
        """
        if not TF_AVAILABLE:
            raise UpscalingError("TensorFlow is not available for model creation")
        
        try:
            logging.info("Creating default upscaling model")
            
            # Define a simple EDSR-like model
            input_shape = (None, None, 1)
            
            # Input layer
            inputs = tf.keras.layers.Input(shape=input_shape)
            x = inputs
            
            # Feature extraction
            x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)
            
            # Residual blocks
            for _ in range(4):
                skip = x
                x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)
                x = tf.keras.layers.Conv2D(64, 3, padding='same')(x)
                x = tf.keras.layers.Add()([x, skip])
            
            # Upscaling blocks
            x = tf.keras.layers.Conv2D(256, 3, padding='same')(x)
            x = tf.keras.layers.Lambda(lambda x: tf.nn.depth_to_space(x, 2))(x)
            
            # Output layer
            outputs = tf.keras.layers.Conv2D(1, 3, padding='same')(x)
            
            # Create model
            model = tf.keras.Model(inputs=inputs, outputs=outputs)
            
            # Compile model
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                loss='mse'
            )
            
            # Save model
            model.save(output_path)
            
            logging.info(f"Default upscaling model saved to {output_path}")
            return output_path
            
        except Exception as e:
            raise UpscalingError(f"Failed to create default model: {e}")
    
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
            # Load the height map
            height_map = self.file_storage.load_height_map(input_file)
            
            # Upscale the height map
            upscaled = self.upscale(height_map, scale_factor, use_gpu)
            
            # Save the upscaled height map
            return self.file_storage.save_height_map(upscaled, output_file)
        except Exception as e:
            raise UpscalingError(f"Failed to upscale file {input_file}: {e}") 