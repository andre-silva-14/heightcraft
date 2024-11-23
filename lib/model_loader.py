import logging
from time import time
from typing import Tuple

import numpy as np
import trimesh

SUPPORTED_FORMATS = [".stl", ".obj", ".ply", ".glb", ".gltf"]


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
            raise ValueError(
                f"Unsupported file format. Supported formats: {', '.join(SUPPORTED_FORMATS)}"
            )

        try:
            start_time = time()
            logging.info(f"Loading 3D model from {file_path}...")
            model = trimesh.load_mesh(file_path)
            logging.info(
                f"Model loaded successfully in {time() - start_time:.2f} seconds."
            )
            return ModelLoader._process_scene(model)
        except Exception as e:
            raise ValueError(f"Failed to load the 3D model: {e}")

    @staticmethod
    def _process_scene(scene):
        if isinstance(scene, trimesh.Scene):
            # If it's a scene, we need to combine all meshes
            meshes = []
            for name, geometry in scene.geometry.items():
                if isinstance(geometry, trimesh.Trimesh):
                    meshes.append(geometry)
            if not meshes:
                raise ValueError("No valid meshes found in the scene")
            combined_mesh = trimesh.util.concatenate(meshes)
        elif isinstance(scene, trimesh.Trimesh):
            # If it's already a single mesh, use it directly
            combined_mesh = scene
        else:
            raise ValueError(f"Unsupported object type: {type(scene)}")

        ModelLoader._center_model(combined_mesh)
        ModelLoader._align_model(combined_mesh)

        return combined_mesh

    @staticmethod
    def _center_model(mesh: trimesh.Trimesh) -> None:
        """Centers the model at the origin."""
        logging.info("Centering the model at the origin...")
        if len(mesh.vertices) > 0:
            centroid = np.mean(mesh.vertices, axis=0)
            mesh.apply_translation(-centroid)
        else:
            logging.warning("Cannot center an empty mesh.")

    @staticmethod
    def _align_model(mesh: trimesh.Trimesh) -> None:
        """Aligns the model so that its largest inertia axis is aligned with Z-axis."""
        logging.info("Aligning the model using principal inertia axes...")
        try:
            if len(mesh.vertices) > 0 and len(mesh.faces) > 0:
                inertia = mesh.moment_inertia
                if np.any(inertia):
                    _, rotation = np.linalg.eigh(inertia)
                    rotation_matrix = np.eye(4)
                    rotation_matrix[:3, :3] = rotation.T
                    mesh.apply_transform(rotation_matrix)
                    logging.info("Model aligned successfully.")
                else:
                    logging.warning("Inertia tensor is zero. Cannot align the model.")
            else:
                logging.warning("Not enough vertices or faces to align the model.")
        except Exception as e:
            raise RuntimeError(f"Failed to calculate principal inertia axes: {e}")
