import gc
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import psutil
import trimesh
from torch import Tensor as TorchTensor
from trimesh import Scene, Trimesh

from .model_loader import ModelLoader
from .resource_manager import resource_manager


class LargeModelHandler:
    """Handles large 3D models with memory-efficient processing and streaming capabilities."""

    def __init__(
        self,
        file_path: str,
        chunk_size: int = 1000000,
        max_memory: float = 0.8,
        cache_dir: Optional[str] = None,
        num_threads: int = 4,
    ):
        """
        Initialize the LargeModelHandler.

        Args:
            file_path: Path to the 3D model file
            chunk_size: Size of chunks for streaming operations
            max_memory: Maximum memory usage as fraction of available memory
            cache_dir: Directory for caching temporary data
            num_threads: Number of threads for parallel processing
        """
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.max_memory = max_memory
        self.cache_dir = cache_dir or os.path.join(os.path.dirname(file_path), ".cache")
        self.num_threads = num_threads
        self._setup_cache()
        self._check_memory_limits()

        # Model data
        self.scene: Optional[Scene] = None
        self.concatenated_mesh: Optional[Trimesh] = None
        self.total_vertices: int = 0
        self.total_faces: int = 0
        self.is_scene: bool = False
        self._vertex_cache: Dict[str, np.ndarray] = {}
        self._face_cache: Dict[str, np.ndarray] = {}

    def _setup_cache(self) -> None:
        """Set up the cache directory and structure."""
        os.makedirs(self.cache_dir, exist_ok=True)
        self.vertex_cache_path = os.path.join(self.cache_dir, "vertices")
        self.face_cache_path = os.path.join(self.cache_dir, "faces")
        os.makedirs(self.vertex_cache_path, exist_ok=True)
        os.makedirs(self.face_cache_path, exist_ok=True)

    def _check_memory_limits(self) -> None:
        """Check if there's enough memory available for processing."""
        available_memory = psutil.virtual_memory().available
        required_memory = self.chunk_size * 3 * 4  # Rough estimate for vertices
        if required_memory > available_memory * self.max_memory:
            raise MemoryError(
                f"Not enough memory available. Required: {required_memory/1024/1024:.2f}MB, "
                f"Available: {available_memory/1024/1024:.2f}MB"
            )

    def load_model_info(self) -> None:
        """
        Load and validate model information with proper error handling.
        """
        try:
            logging.info(f"Loading 3D model from {self.file_path}...")
            # First, try to load without processing to check if it's a scene
            temp_load = trimesh.load(self.file_path, process=False)
            self._validate_model(temp_load)

            if isinstance(temp_load, Scene):
                logging.info(
                    "Detected a GLTF scene with multiple meshes, processing scene..."
                )
                self.is_scene = True
                self.scene = trimesh.load(self.file_path, force="scene")
                self._process_scene()
            else:
                self._validate_mesh(temp_load)
                self.concatenated_mesh = temp_load

            self._update_model_stats()
            logging.info(
                f"Processed model has {self.total_vertices} vertices and {self.total_faces} faces"
            )

        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            raise

    def _validate_model(self, model: Union[Scene, Trimesh]) -> None:
        """Validate the loaded model."""
        if not isinstance(model, (Scene, Trimesh)):
            raise ValueError(f"Unsupported model type: {type(model)}")
        if isinstance(model, Trimesh):
            self._validate_mesh(model)

    def _validate_mesh(self, mesh: Trimesh) -> None:
        """Validate mesh geometry and properties."""
        if len(mesh.vertices) == 0:
            raise ValueError("Mesh has no vertices")
        if len(mesh.faces) == 0:
            raise ValueError("Mesh has no faces")
        if not mesh.is_watertight:
            logging.warning("Mesh is not watertight, which may affect sampling quality")
        if not mesh.is_winding_consistent:
            logging.warning("Mesh has inconsistent face winding")

    def _process_scene(self) -> None:
        """Process a GLTF scene with streaming support and parallel processing."""
        all_vertices: List[np.ndarray] = []
        all_faces: List[np.ndarray] = []
        total_vertices = 0

        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = []
            for node_name in self.scene.graph.nodes_geometry:
                transform, geometry_name = self.scene.graph[node_name]
                mesh = self.scene.geometry[geometry_name]
                futures.append(
                    executor.submit(
                        self._process_mesh_node,
                        mesh,
                        transform,
                        total_vertices,
                        node_name,
                    )
                )
                total_vertices += len(mesh.vertices)

            for future in futures:
                vertices, faces = future.result()
                all_vertices.append(vertices)
                all_faces.append(faces)

        # Concatenate results with memory-efficient streaming
        self._concatenate_results(all_vertices, all_faces)

    def _process_mesh_node(
        self, mesh: Trimesh, transform: np.ndarray, vertex_offset: int, node_name: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Process a single mesh node with chunked transformation."""
        vertices = []
        faces = []

        # Process vertices in chunks
        vertex_chunks = np.array_split(
            mesh.vertices, max(1, len(mesh.vertices) // self.chunk_size)
        )
        for i, chunk in enumerate(vertex_chunks):
            transformed_chunk = trimesh.transform_points(chunk, transform)
            vertices.append(transformed_chunk)

            # Cache the chunk if it's large enough
            if len(chunk) > self.chunk_size // 2:
                cache_path = os.path.join(
                    self.vertex_cache_path, f"{node_name}_{i}.npy"
                )
                np.save(cache_path, transformed_chunk)
                self._vertex_cache[cache_path] = transformed_chunk

        # Process faces
        face_chunks = np.array_split(
            mesh.faces, max(1, len(mesh.faces) // self.chunk_size)
        )
        for i, chunk in enumerate(face_chunks):
            offset_chunk = chunk + vertex_offset
            faces.append(offset_chunk)

            # Cache the chunk if it's large enough
            if len(chunk) > self.chunk_size // 2:
                cache_path = os.path.join(self.face_cache_path, f"{node_name}_{i}.npy")
                np.save(cache_path, offset_chunk)
                self._face_cache[cache_path] = offset_chunk

        return np.vstack(vertices), np.vstack(faces)

    def _concatenate_results(
        self, vertices_list: List[np.ndarray], faces_list: List[np.ndarray]
    ) -> None:
        """Concatenate results with memory-efficient streaming."""
        # Create a new mesh with streaming support
        self.concatenated_mesh = Trimesh(
            vertices=np.vstack(vertices_list),
            faces=np.vstack(faces_list),
            process=False,  # Prevent automatic processing
        )

    def stream_vertices(self) -> Iterator[np.ndarray]:
        """Stream vertices in chunks with caching support."""
        if self.concatenated_mesh is None:
            raise RuntimeError("Model not loaded. Call load_model_info() first.")

        for i in range(0, self.total_vertices, self.chunk_size):
            cache_key = f"vertices_{i}"
            if cache_key in self._vertex_cache:
                yield self._vertex_cache[cache_key]
            else:
                chunk = self.concatenated_mesh.vertices[i : i + self.chunk_size]
                self._vertex_cache[cache_key] = chunk
                yield chunk

    def stream_faces(self) -> Iterator[np.ndarray]:
        """Stream faces in chunks with caching support."""
        if self.concatenated_mesh is None:
            raise RuntimeError("Model not loaded. Call load_model_info() first.")

        for i in range(0, self.total_faces, self.chunk_size):
            cache_key = f"faces_{i}"
            if cache_key in self._face_cache:
                yield self._face_cache[cache_key]
            else:
                chunk = self.concatenated_mesh.faces[i : i + self.chunk_size]
                self._face_cache[cache_key] = chunk
                yield chunk

    def calculate_bounding_box(self) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate the bounding box using streamed vertices with parallel processing."""
        min_coords = np.array([np.inf, np.inf, np.inf])
        max_coords = np.array([-np.inf, -np.inf, -np.inf])

        # For scenes, use the scene's bounds directly if available
        if self.is_scene and hasattr(self.scene, "bounds"):
            scene_bounds = self.scene.bounds
            if scene_bounds is not None and scene_bounds.shape == (2, 3):
                logging.info("Using scene bounds directly")
                return scene_bounds[0], scene_bounds[1]

        # For concatenated meshes, use the mesh's bounds directly if available
        if not self.is_scene and self.concatenated_mesh is not None:
            mesh_bounds = self.concatenated_mesh.bounds
            if mesh_bounds is not None and mesh_bounds.shape == (2, 3):
                logging.info("Using mesh bounds directly")
                return mesh_bounds[0], mesh_bounds[1]

        # Fall back to calculating bounds from streamed vertices
        logging.info("Calculating bounds from streamed vertices")
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = []
            for chunk in self.stream_vertices():
                futures.append(executor.submit(self._calculate_chunk_bounds, chunk))

            for future in futures:
                chunk_min, chunk_max = future.result()
                min_coords = np.minimum(min_coords, chunk_min)
                max_coords = np.maximum(max_coords, chunk_max)

        # Validate the bounds
        if np.any(np.isinf(min_coords)) or np.any(np.isinf(max_coords)):
            logging.warning("Invalid bounds detected, using default bounds")
            return np.array([0, 0, 0]), np.array([1, 1, 1])

        logging.info(f"Calculated bounds: min={min_coords}, max={max_coords}")
        return min_coords, max_coords

    @staticmethod
    def _calculate_chunk_bounds(chunk: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate bounds for a chunk of vertices."""
        return np.min(chunk, axis=0), np.max(chunk, axis=0)

    def sample_points(self, num_samples: int, use_gpu: bool) -> np.ndarray:
        """Sample points from the model surface with proper resource management."""
        if num_samples <= 0:
            raise ValueError("Number of samples must be positive")

        if not use_gpu:
            return self._sample_points_cpu(num_samples)

        from torch.cuda import is_available as cuda_is_available

        if use_gpu and cuda_is_available():
            return self._sample_points_gpu(num_samples)
        else:
            logging.warning(
                "GPU Acceleration was requested but CUDA is not available. "
                "Falling back to CPU processing."
            )
            return self._sample_points_cpu(num_samples)

    def _sample_points_cpu(self, num_samples: int) -> np.ndarray:
        """Sample points using CPU with parallel processing and memory optimization."""
        if self.is_scene:
            return self._sample_points_from_scene_cpu(num_samples)

        # For regular meshes, use the existing approach
        samples = []
        samples_per_face = max(1, num_samples // self.total_faces)

        # Calculate batch size based on available memory
        batch_size = min(100000, max(1000, self.chunk_size // 10))
        num_batches = max(1, self.total_faces // batch_size)
        logging.info(f"Processing in {num_batches} batches of {batch_size} faces each")

        face_batches = []
        face_count = 0
        current_batch = []

        # Create batches of faces
        for face_chunk in self.stream_faces():
            current_batch.append(face_chunk)
            face_count += len(face_chunk)

            if face_count >= batch_size:
                if current_batch:
                    face_batches.append(np.vstack(current_batch))
                current_batch = []
                face_count = 0

        # Add the last batch if it's not empty
        if current_batch:
            face_batches.append(np.vstack(current_batch))

        # Process each batch
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            for batch_idx, face_batch in enumerate(face_batches):
                logging.info(f"Processing batch {batch_idx+1}/{len(face_batches)}")
                batch_samples = samples_per_face * len(face_batch)

                # Adjust the last batch to ensure we get exactly num_samples points
                if batch_idx == len(face_batches) - 1:
                    total_sampled = sum(len(s) for s in samples)
                    batch_samples = max(1, num_samples - total_sampled)

                # Sample points from this batch
                try:
                    batch_result = self._sample_chunk_cpu(face_batch, batch_samples)
                    samples.append(batch_result)

                    # Check memory usage and clear cache if needed
                    if psutil.virtual_memory().percent > 90:
                        logging.warning("High memory usage detected, clearing caches")
                        gc.collect()
                except Exception as e:
                    logging.error(f"Error sampling batch {batch_idx}: {e}")
                    # If we have some samples, continue with what we have
                    if samples:
                        continue
                    else:
                        raise

        # Combine all samples
        result = np.vstack(samples)

        # If we have too many samples, subsample
        if len(result) > num_samples:
            indices = np.random.choice(len(result), num_samples, replace=False)
            result = result[indices]
        # If we have too few samples, duplicate some
        elif len(result) < num_samples:
            additional = num_samples - len(result)
            indices = np.random.choice(len(result), additional, replace=True)
            result = np.vstack([result, result[indices]])

        return result

    def _sample_points_from_scene_cpu(self, num_samples: int) -> np.ndarray:
        """Sample points from a scene using CPU with memory optimization."""
        logging.info(
            f"Sampling {num_samples} points from scene with {len(self.scene.geometry)} geometries"
        )

        # Allocate samples proportionally to mesh surface area
        mesh_areas = {}
        total_area = 0

        for name, mesh in self.scene.geometry.items():
            if hasattr(mesh, "area"):
                area = mesh.area
                if area > 0:
                    mesh_areas[name] = area
                    total_area += area

        if total_area <= 0:
            logging.warning("No valid meshes with positive area found in scene")
            # Fall back to equal distribution
            mesh_areas = {name: 1 for name in self.scene.geometry}
            total_area = len(mesh_areas)

        # Calculate samples per mesh based on area
        samples_per_mesh = {}
        remaining_samples = num_samples

        for name, area in mesh_areas.items():
            mesh_samples = max(10, int((area / total_area) * num_samples))
            samples_per_mesh[name] = mesh_samples
            remaining_samples -= mesh_samples

        # Adjust to ensure we get exactly num_samples
        if remaining_samples > 0:
            # Distribute remaining samples
            for name in sorted(mesh_areas.keys()):
                samples_per_mesh[name] += 1
                remaining_samples -= 1
                if remaining_samples <= 0:
                    break

        # Sample points from each mesh
        all_samples = []

        for node_name in self.scene.graph.nodes_geometry:
            transform, geometry_name = self.scene.graph[node_name]

            if geometry_name not in samples_per_mesh:
                continue

            mesh = self.scene.geometry[geometry_name]
            mesh_samples = samples_per_mesh[geometry_name]

            if mesh_samples <= 0:
                continue

            try:
                # Sample points from this mesh
                points, _ = trimesh.sample.sample_surface_even(mesh, mesh_samples)

                # Transform points to scene coordinates
                if transform is not None:
                    points = trimesh.transformations.transform_points(points, transform)

                all_samples.append(points)

                # Check memory usage and clear cache if needed
                if psutil.virtual_memory().percent > 90:
                    logging.warning("High memory usage detected, clearing caches")
                    gc.collect()
            except Exception as e:
                logging.error(f"Error sampling mesh {geometry_name}: {e}")
                # Continue with other meshes

        if not all_samples:
            raise ValueError("Failed to sample any points from the scene")

        # Combine all samples
        result = np.vstack(all_samples)

        # If we have too many samples, subsample
        if len(result) > num_samples:
            indices = np.random.choice(len(result), num_samples, replace=False)
            result = result[indices]
        # If we have too few samples, duplicate some
        elif len(result) < num_samples:
            additional = num_samples - len(result)
            indices = np.random.choice(len(result), additional, replace=True)
            result = np.vstack([result, result[indices]])

        return result

    def _sample_chunk_cpu(self, faces: np.ndarray, num_samples: int) -> np.ndarray:
        """Sample points from a chunk of faces using CPU."""
        return trimesh.sample.sample_surface_even(
            Trimesh(vertices=self.concatenated_mesh.vertices, faces=faces),
            num_samples,
        )

    def _sample_points_gpu(self, num_samples: int) -> np.ndarray:
        """Sample points using GPU with proper memory management."""
        import torch

        samples = []
        batch_size = min(self.chunk_size, num_samples)

        with resource_manager.gpu_session():
            for i in range(0, num_samples, batch_size):
                current_batch = min(batch_size, num_samples - i)
                face_chunk = self._get_face_chunk(current_batch)

                # Process batch on GPU
                batch_samples = self._process_gpu_batch(face_chunk)
                samples.append(batch_samples.cpu().numpy())

                # Clear GPU memory after each batch
                torch.cuda.empty_cache()

        return np.vstack(samples)

    def _get_face_chunk(self, size: int) -> np.ndarray:
        """Get a chunk of faces for sampling."""
        if self.concatenated_mesh is None:
            raise RuntimeError("Model not loaded")
        return self.concatenated_mesh.faces[:size]

    def _process_gpu_batch(self, faces: np.ndarray) -> TorchTensor:
        """Process a batch of faces on the GPU."""
        import torch

        vertices = resource_manager.allocate_gpu_tensor(
            self.concatenated_mesh.vertices, dtype=torch.float32
        )
        faces = resource_manager.allocate_gpu_tensor(faces, dtype=torch.long)

        # Compute face areas
        v0, v1, v2 = (
            vertices[faces[:, 0]],
            vertices[faces[:, 1]],
            vertices[faces[:, 2]],
        )
        face_areas = 0.5 * torch.norm(torch.cross(v1 - v0, v2 - v0), dim=1)
        face_probs = face_areas / torch.sum(face_areas)

        # Sample faces
        face_indices = torch.multinomial(face_probs, len(faces), replacement=True)
        r1, r2 = torch.sqrt(torch.rand(len(faces), device="cuda")), torch.rand(
            len(faces), device="cuda"
        )

        # Generate points
        return (
            (1 - r1.unsqueeze(1)) * vertices[faces[face_indices, 0]]
            + (r1 * (1 - r2)).unsqueeze(1) * vertices[faces[face_indices, 1]]
            + (r1 * r2).unsqueeze(1) * vertices[faces[face_indices, 2]]
        )

    def _update_model_stats(self) -> None:
        """Update model statistics."""
        if self.concatenated_mesh is None:
            raise RuntimeError("Model not loaded")
        self.total_vertices = len(self.concatenated_mesh.vertices)
        self.total_faces = len(self.concatenated_mesh.faces)

    def cleanup(self) -> None:
        """Clean up resources and temporary files."""
        # Clear caches
        self._vertex_cache.clear()
        self._face_cache.clear()

        # Clear GPU memory
        if hasattr(self, "resource_manager"):
            self.resource_manager.clear_gpu_memory()

        # Remove temporary files
        if os.path.exists(self.vertex_cache_path):
            for file in os.listdir(self.vertex_cache_path):
                os.remove(os.path.join(self.vertex_cache_path, file))
        if os.path.exists(self.face_cache_path):
            for file in os.listdir(self.face_cache_path):
                os.remove(os.path.join(self.face_cache_path, file))

        # Clear references
        self.scene = None
        self.concatenated_mesh = None
        gc.collect()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()
