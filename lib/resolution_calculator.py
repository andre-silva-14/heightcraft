import trimesh
import logging
from typing import Tuple

class ResolutionCalculator:
    @staticmethod
    def calculate(mesh: trimesh.Trimesh, max_resolution: int) -> Tuple[int, int]:
        """
        Calculates the dynamic resolution based on the model's aspect ratio.
        
        Args:
            mesh (trimesh.Trimesh): The 3D model mesh.
            max_resolution (int): Maximum resolution for the longest dimension.
        
        Returns:
            Tuple[int, int]: Width and height of the calculated resolution.
        
        Raises:
            ValueError: If the model bounding box has zero width or height.
        """
        logging.info("Calculating dynamic resolution...")
        
        bounds = mesh.bounds
        model_width = bounds[1][0] - bounds[0][0]  # X range
        model_height = bounds[1][1] - bounds[0][1]  # Y range

        if model_width <= 0 or model_height <= 0:
            raise ValueError("Model bounding box has zero or negative width or height.")

        aspect_ratio = model_width / model_height

        if aspect_ratio >= 1.0:
            width = max_resolution
            height = max(1, int(width / aspect_ratio))
        else:
            height = max_resolution
            width = max(1, int(height * aspect_ratio))

        logging.info(f"Dynamic resolution determined: {width}x{height}")
        return width, height
