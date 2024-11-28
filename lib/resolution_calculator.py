import logging
from typing import Tuple

import numpy as np
from trimesh import Trimesh


class ResolutionCalculator:
    @staticmethod
    def calculate(mesh: Trimesh, max_resolution: int) -> Tuple[int, int]:
        """
        Calculates the dynamic resolution based on the model's aspect ratio.

        Args:
            mesh (Trimesh): The 3D model mesh.
            max_resolution (int): Maximum resolution for the longest dimension.

        Returns:
            Tuple[int, int]: Width and height of the calculated resolution.

        Raises:
            ValueError: If the model bounding box has zero width or height.
        """
        logging.info("Calculating dynamic resolution...")

        # Get the bounding box from the mesh
        bounds = mesh.bounds
        min_coords, max_coords = bounds[0], bounds[1]

        return ResolutionCalculator.calculate_from_bounds(
            min_coords, max_coords, max_resolution
        )

    @staticmethod
    def calculate_from_bounds(
        min_coords: np.ndarray, max_coords: np.ndarray, max_resolution: int
    ) -> Tuple[int, int]:
        """
        Calculates the dynamic resolution based on the model's bounding box.

        Args:
            min_coords (np.ndarray): Minimum coordinates of the bounding box.
            max_coords (np.ndarray): Maximum coordinates of the bounding box.
            max_resolution (int): Maximum resolution for the longest dimension.

        Returns:
            Tuple[int, int]: Width and height of the calculated resolution.

        Raises:
            ValueError: If the model bounding box has zero width or height.
        """
        logging.info("Calculating dynamic resolution from bounding box...")

        # Calculate the width and height of the bounding box
        model_width = max_coords[0] - min_coords[0]
        model_height = max_coords[1] - min_coords[1]

        if model_width <= 0 or model_height <= 0:
            raise ValueError("Model bounding box has zero or negative width or height.")

        # Calculate the aspect ratio
        aspect_ratio = model_width / model_height

        # If the width is greater than the height (model is wider)
        if aspect_ratio >= 1.0:
            width = max_resolution
            height = max(1, int(width / aspect_ratio))
        else:  # If the height is greater than the width (model is taller)
            height = max_resolution
            width = max(1, int(height * aspect_ratio))

        logging.info(f"Dynamic resolution determined: {width}x{height}")
        return width, height
