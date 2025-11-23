"""
Training service for Heightcraft.

This module provides the TrainingService class for training AI upscaling models.
"""

import logging
import os
import glob
from typing import Optional, Tuple, List, Dict

import numpy as np

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

from heightcraft.core.config import UpscaleConfig
from heightcraft.core.exceptions import HeightcraftError
from heightcraft.infrastructure.file_storage import FileStorage
from heightcraft.infrastructure.gpu_manager import gpu_manager
from heightcraft.services.upscaling_service import UpscalingService
from heightcraft.infrastructure.profiler import profiler


class TrainingError(HeightcraftError):
    """Exception raised for training errors."""
    pass


class TrainingService:
    """
    Service for training AI upscaling models.
    
    This class handles dataset preparation and model training.
    """
    
    def __init__(
        self, 
        config: Optional[UpscaleConfig] = None,
        file_storage: Optional[FileStorage] = None
    ):
        """
        Initialize the training service.
        
        Args:
            config: Upscaling configuration
            file_storage: File storage for loading/saving files
        """
        self.config = config or UpscaleConfig()
        self.file_storage = file_storage or FileStorage()
        self.logger = logging.getLogger(self.__class__.__name__)
        
    @profiler.profile()
    def train_model(
        self,
        dataset_path: str,
        output_model_path: str,
        epochs: int = 10,
        batch_size: int = 16,
        learning_rate: float = 1e-4,
        validation_split: float = 0.2
    ) -> str:
        """
        Train an upscaling model.
        
        Args:
            dataset_path: Path to directory containing high-res height maps
            output_model_path: Path to save the trained model
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            validation_split: Fraction of data to use for validation
            
        Returns:
            Path to the saved model
            
        Raises:
            TrainingError: If training fails
        """
        if not TF_AVAILABLE:
            raise TrainingError("TensorFlow is not available for training")
            
        try:
            self.logger.info(f"Starting training with dataset: {dataset_path}")
            
            # 1. Load dataset
            image_paths = self._get_image_paths(dataset_path)
            if not image_paths:
                raise TrainingError(f"No images found in {dataset_path}")
                
            self.logger.info(f"Found {len(image_paths)} images")
            
            # 2. Create TensorFlow dataset
            dataset = self._create_dataset(image_paths, batch_size, validation_split)
            train_ds = dataset['train']
            val_ds = dataset['val']
            
            # 3. Create or load model
            model = self._get_model(output_model_path)
            
            # 4. Compile model
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                loss='mse',
                metrics=['mae']
            )
            
            # 5. Train model
            self.logger.info(f"Training for {epochs} epochs...")
            
            # Setup callbacks
            callbacks = [
                tf.keras.callbacks.ModelCheckpoint(
                    output_model_path,
                    save_best_only=True,
                    monitor='val_loss'
                ),
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                )
            ]
            
            history = model.fit(
                train_ds,
                epochs=epochs,
                validation_data=val_ds,
                callbacks=callbacks
            )
            
            self.logger.info("Training complete")
            
            # Save final model if not saved by checkpoint
            if not os.path.exists(output_model_path):
                model.save(output_model_path)
                
            return output_model_path
            
        except Exception as e:
            raise TrainingError(f"Training failed: {e}")

    def _get_image_paths(self, dataset_path: str) -> List[str]:
        """Get list of image paths from directory."""
        extensions = ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff']
        paths = []
        for ext in extensions:
            paths.extend(glob.glob(os.path.join(dataset_path, ext)))
            paths.extend(glob.glob(os.path.join(dataset_path, "**", ext), recursive=True))
        return sorted(list(set(paths)))

    def _create_dataset(self, image_paths: List[str], batch_size: int, validation_split: float) -> Dict:
        """Create TensorFlow dataset from image paths."""
        
        # Split paths
        np.random.shuffle(image_paths)
        split_idx = int(len(image_paths) * (1 - validation_split))
        train_paths = image_paths[:split_idx]
        val_paths = image_paths[split_idx:]
        
        def process_path(file_path):
            # Load image
            img = tf.io.read_file(file_path)
            img = tf.image.decode_image(img, channels=1, expand_animations=False)
            img = tf.image.convert_image_dtype(img, tf.float32)
            
            # Ensure strictly 1 channel (grayscale)
            if img.shape[-1] != 1:
                 img = tf.image.rgb_to_grayscale(img)

            # Random crop to fixed size for training (e.g., 128x128)
            # We need fixed size patches for batching
            img = tf.image.resize_with_crop_or_pad(img, 128, 128)
            
            # Create low-res input (downsample by factor 2)
            # We use area interpolation for downsampling to simulate averaging
            lr_img = tf.image.resize(img, [64, 64], method='area')
            
            return lr_img, img

        def configure_for_performance(ds):
            ds = ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
            ds = ds.cache()
            ds = ds.shuffle(buffer_size=1000)
            ds = ds.batch(batch_size)
            ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
            return ds

        train_ds = tf.data.Dataset.from_tensor_slices(train_paths)
        train_ds = configure_for_performance(train_ds)
        
        val_ds = tf.data.Dataset.from_tensor_slices(val_paths)
        val_ds = val_ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
        val_ds = val_ds.batch(batch_size)
        val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
        
        return {'train': train_ds, 'val': val_ds}

    def _get_model(self, model_path: str) -> tf.keras.Model:
        """Get model instance (load existing or create new)."""
        if os.path.exists(model_path):
            self.logger.info(f"Resuming training from {model_path}")
            return tf.keras.models.load_model(model_path)
        else:
            self.logger.info("Creating new model")
            # We reuse the architecture from UpscalingService
            # But we need to instantiate it directly here or call the helper
            # Since UpscalingService.create_default_model saves to disk, we can use a temp file
            # Or better, we can extract the model creation logic.
            # For now, let's just recreate the simple architecture here to avoid I/O
            
            inputs = tf.keras.layers.Input(shape=(None, None, 1))
            x = inputs
            x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)
            for _ in range(4):
                skip = x
                x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)
                x = tf.keras.layers.Conv2D(64, 3, padding='same')(x)
                x = tf.keras.layers.Add()([x, skip])
            
            x = tf.keras.layers.Conv2D(256, 3, padding='same')(x)
            x = tf.keras.layers.Lambda(lambda x: tf.nn.depth_to_space(x, 2))(x)
            outputs = tf.keras.layers.Conv2D(1, 3, padding='same')(x)
            
            return tf.keras.Model(inputs=inputs, outputs=outputs)
