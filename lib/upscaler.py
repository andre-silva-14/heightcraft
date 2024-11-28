import logging

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.models import Sequential


class HeightMapUpscaler:
    def __init__(self):
        self.model = self._build_srcnn_model()

    def _build_srcnn_model(self):
        model = Sequential()
        model.add(
            Conv2D(
                64,
                kernel_size=(9, 9),
                activation="relu",
                padding="same",
                input_shape=(None, None, 1),
            )
        )
        model.add(Conv2D(32, kernel_size=(1, 1), activation="relu", padding="same"))
        model.add(Conv2D(1, kernel_size=(5, 5), activation="linear", padding="same"))
        model.compile(optimizer="adam", loss="mse")
        return model

    def train(self, low_res_maps, high_res_maps, epochs=100, batch_size=32):
        """
        Deprecated function. No longer in use.
        """
        logging.info("Training the upscaling model...")
        self.model.fit(
            low_res_maps,
            high_res_maps,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
        )
        logging.info("Training completed.")

    def upscale(self, height_map: np.ndarray, scale_factor: int = 2) -> np.ndarray:
        """
        Upscale the input height map using the SRCNN model.

        Args:
            height_map (np.ndarray): Input height map to upscale.
            scale_factor (int): Factor by which to upscale the image. Default is 2.

        Returns:
            np.ndarray: Upscaled height map with enhanced details.
        """
        logging.info(f"Upscaling height map by a factor of {scale_factor}...")

        # Store the original min and max values
        original_min = np.min(height_map)
        original_max = np.max(height_map)

        # Normalize the height map to [0, 1] range
        height_map_normalized = (height_map - original_min) / (
            original_max - original_min
        )

        # Resize the image using bicubic interpolation
        height, width = height_map_normalized.shape
        resized = tf.image.resize(
            height_map_normalized[..., np.newaxis],
            (height * scale_factor, width * scale_factor),
            method="bicubic",
        )

        # Apply the SRCNN model for enhancement
        enhanced = self.model.predict(resized)

        # Ensure the enhanced image is in the range [0, 1]
        enhanced = np.clip(enhanced, 0, 1)

        # Denormalize the result back to the original range
        result = enhanced.squeeze() * (original_max - original_min) + original_min

        logging.info("Upscaling completed.")
        return result.astype(height_map.dtype)


def load_pretrained_model(model_path: str) -> HeightMapUpscaler:
    """
    Load a pretrained HeightMapUpscaler model.

    Args:
        model_path (str): Path to the pretrained model weights.

    Returns:
        HeightMapUpscaler: Upscaler with loaded pretrained weights.
    """
    upscaler = HeightMapUpscaler()
    if model_path and os.path.exists(model_path):
        upscaler.model.load_weights(model_path)
        logging.info(f"Loaded pretrained model from {model_path}")
    else:
        logging.warning(
            "No pretrained model found. Using untrained model for basic upscaling."
        )
    return upscaler
