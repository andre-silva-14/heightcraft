import trimesh
import numpy as np
from time import time
import logging
from typing import Tuple

SUPPORTED_FORMATS = ['.stl', '.obj', '.ply', '.glb', '.gltf']

class ModelLoader:
    @staticmethod
    def load(file_path: str) -> trimesh.Trimesh:
        """
        Loads and preprocesses the 3D model from the given file path.
        
        Supported formats: STL, OBJ, PLY, GLB, GLTF
        
        Args:
            file_path (str): Path to the 3D model file.
        
        Returns:
            trimesh.Trimesh: Loaded and preprocessed 3D model.
        
        Raises:
            ValueError: If the file format is not supported or loading fails.
        """
        if not any(file_path.lower().endswith(fmt) for fmt in SUPPORTED_FORMATS):
            raise ValueError(f"Unsupported file format. Supported formats: {', '.join(SUPPORTED_FORMATS)}")

        try:
            start_time = time()
            logging.info(f"Loading 3D model from {file_path}...")
            mesh = trimesh.load_mesh(file_path)
            logging.info(f"Model loaded successfully in {time() - start_time:.2f} seconds.")
        except Exception as e:
            raise ValueError(f"Failed to load the 3D model: {e}")

        ModelLoader._center_model(mesh)
        ModelLoader._align_model(mesh)
        
        return mesh

    @staticmethod
    def _center_model(mesh: trimesh.Trimesh) -> None:
        """Centers the model at the origin."""
        logging.info("Centering the model at the origin...")
        mesh.apply_translation(-mesh.centroid)

    @staticmethod
    def _align_model(mesh: trimesh.Trimesh) -> None:
        """Aligns the model so that its largest inertia axis is aligned with Z-axis."""
        logging.info("Aligning the model using principal inertia axes...")
        try:
            _, rotation = mesh.principal_inertia_axes()
            mesh.apply_transform(rotation)
            logging.info("Model aligned successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to calculate principal inertia axes: {e}")
