import trimesh
import numpy as np
import torch
from typing import Iterator, Tuple
import logging
from .resource_manager import resource_manager

class LargeModelHandler:
    def __init__(self, file_path: str, chunk_size: int = 1000000):
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.mesh = None
        self.total_vertices = 0
        self.total_faces = 0

    def load_model_info(self):
        """Load basic information about the model without loading the entire mesh."""
        logging.info(f"Loading model info from {self.file_path}")
        self.mesh = trimesh.load(self.file_path, process=False)
        self.total_vertices = len(self.mesh.vertices)
        self.total_faces = len(self.mesh.faces)
        logging.info(f"Model has {self.total_vertices} vertices and {self.total_faces} faces")

    def stream_vertices(self) -> Iterator[np.ndarray]:
        """Stream vertices in chunks to reduce memory usage."""
        for i in range(0, self.total_vertices, self.chunk_size):
            yield self.mesh.vertices[i:i+self.chunk_size]

    def stream_faces(self) -> Iterator[np.ndarray]:
        """Stream faces in chunks to reduce memory usage."""
        for i in range(0, self.total_faces, self.chunk_size):
            yield self.mesh.faces[i:i+self.chunk_size]

    def calculate_bounding_box(self) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate the bounding box of the model using streamed vertices."""
        min_coords = np.array([np.inf, np.inf, np.inf])
        max_coords = np.array([-np.inf, -np.inf, -np.inf])

        for chunk in self.stream_vertices():
            min_coords = np.minimum(min_coords, np.min(chunk, axis=0))
            max_coords = np.maximum(max_coords, np.max(chunk, axis=0))

        return min_coords, max_coords

    def sample_points_cpu(self, num_samples: int) -> np.ndarray:
        """Sample points from the model surface using CPU, processing in chunks."""
        samples = []
        samples_per_face = num_samples // self.total_faces
        remaining_samples = num_samples % self.total_faces

        for face_chunk in self.stream_faces():
            chunk_samples = trimesh.sample.sample_surface_even(trimesh.Trimesh(vertices=self.mesh.vertices, faces=face_chunk), 
                                                               samples_per_face * len(face_chunk))
            samples.append(chunk_samples)

        # Handle remaining samples
        if remaining_samples > 0:
            extra_samples = trimesh.sample.sample_surface_even(self.mesh, remaining_samples)
            samples.append(extra_samples)

        return np.vstack(samples)

    def sample_points_gpu(self, num_samples: int) -> np.ndarray:
        """Sample points from the model surface using GPU, processing in chunks."""
        samples = []
        samples_per_face = num_samples // self.total_faces
        remaining_samples = num_samples % self.total_faces

        with resource_manager.gpu_session():
            vertices = resource_manager.allocate_gpu_tensor(self.mesh.vertices, dtype=torch.float32)
            for face_chunk in self.stream_faces():
                faces = resource_manager.allocate_gpu_tensor(face_chunk, dtype=torch.long)

                # Compute face areas
                v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
                face_areas = 0.5 * torch.norm(torch.cross(v1 - v0, v2 - v0), dim=1)

                # Sample faces based on their areas
                face_probs = face_areas / torch.sum(face_areas)
                face_indices = torch.multinomial(face_probs, samples_per_face * len(face_chunk), replacement=True)

                # Generate random barycentric coordinates
                r1, r2 = torch.sqrt(torch.rand(len(face_indices), device='cuda')), torch.rand(len(face_indices), device='cuda')
                sample_points = (
                    (1 - r1.unsqueeze(1)) * vertices[faces[face_indices, 0]] +
                    (r1 * (1 - r2)).unsqueeze(1) * vertices[faces[face_indices, 1]] +
                    (r1 * r2).unsqueeze(1) * vertices[faces[face_indices, 2]]
                )

                samples.append(sample_points.cpu().numpy())

            # Handle remaining samples
            if remaining_samples > 0:
                # Use the last face chunk for simplicity
                extra_indices = torch.multinomial(face_probs, remaining_samples, replacement=True)
                r1, r2 = torch.sqrt(torch.rand(remaining_samples, device='cuda')), torch.rand(remaining_samples, device='cuda')
                extra_points = (
                    (1 - r1.unsqueeze(1)) * vertices[faces[extra_indices, 0]] +
                    (r1 * (1 - r2)).unsqueeze(1) * vertices[faces[extra_indices, 1]] +
                    (r1 * r2).unsqueeze(1) * vertices[faces[extra_indices, 2]]
                )
                samples.append(extra_points.cpu().numpy())

        return np.vstack(samples)
