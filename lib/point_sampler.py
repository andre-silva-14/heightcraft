import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Union

import numpy as np
import trimesh


class PointSampler:
    @staticmethod
    def sample_points(
        mesh: trimesh.Trimesh, num_samples: int, use_gpu: bool, num_threads: int
    ) -> np.ndarray:
        """
        Samples points from the surface of a 3D model.

        Args:
            mesh (trimesh.Trimesh): The 3D model mesh.
            num_samples (int): Number of points to sample.
            use_gpu (bool): Whether to use GPU for sampling.
            num_threads (int): Number of threads for CPU sampling.

        Returns:
            np.ndarray: Sampled points.
        """
        if num_samples <= 0:
            raise ValueError("Number of samples must be a positive integer.")
        if num_threads <= 0:
            raise ValueError("Number of threads must be a positive integer.")

        if not use_gpu:
            return PointSampler._sample_points_on_cpu(mesh, num_samples, num_threads)

        from torch.cuda import is_available as cuda_is_available

        if use_gpu and cuda_is_available():
            return PointSampler._sample_points_on_gpu(mesh, num_samples)
        else:
            logging.warning(
                f"GPU Acceleration was requested but CUDA is not available. Falling back to CPU processing."
            )
            return PointSampler._sample_points_on_cpu(mesh, num_samples, num_threads)

    @staticmethod
    def _sample_points_on_gpu(mesh: trimesh.Trimesh, num_samples: int) -> np.ndarray:
        """Samples points uniformly from the surface of a 3D model on the GPU."""
        logging.info(f"Sampling {num_samples} points from the 3D model using GPU...")

        import torch

        from .resource_manager import resource_manager

        with resource_manager.gpu_session():
            vertices = resource_manager.allocate_gpu_tensor(
                mesh.vertices, dtype=torch.float32
            )
            faces = resource_manager.allocate_gpu_tensor(mesh.faces, dtype=torch.long)

            v0, v1, v2 = (
                vertices[faces[:, 0]],
                vertices[faces[:, 1]],
                vertices[faces[:, 2]],
            )
            face_areas = 0.5 * torch.norm(torch.linalg.cross(v1 - v0, v2 - v0, dim=1), dim=1)
            
            # Handle case where some face areas might be close to zero
            epsilon = 1e-10
            face_areas = torch.clamp(face_areas, min=epsilon)
            face_probs = face_areas / torch.sum(face_areas)

            face_indices = torch.multinomial(face_probs, num_samples, replacement=True)
            r1, r2 = torch.sqrt(torch.rand(num_samples, device="cuda")), torch.rand(
                num_samples, device="cuda"
            )
            u, v, w = 1 - r1, r1 * (1 - r2), r1 * r2

            sampled_faces = faces[face_indices]
            sampled_points = (
                u.unsqueeze(1) * vertices[sampled_faces[:, 0]]
                + v.unsqueeze(1) * vertices[sampled_faces[:, 1]]
                + w.unsqueeze(1) * vertices[sampled_faces[:, 2]]
            )

            logging.info(f"Sampling complete: {num_samples} points generated.")
            return sampled_points.cpu().numpy()

    @staticmethod
    def _sample_points_on_cpu(
        mesh: trimesh.Trimesh, num_samples: int, num_threads: int
    ) -> np.ndarray:
        """Samples points uniformly from the surface of a 3D model on the CPU using multiple threads."""
        logging.info(
            f"Sampling {num_samples} points from the 3D model using CPU with {num_threads} threads..."
        )

        samples_per_thread = num_samples // num_threads
        remaining_samples = num_samples % num_threads

        def sample_chunk(chunk_size):
            return mesh.sample(chunk_size)

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(sample_chunk, samples_per_thread)
                for _ in range(num_threads)
            ]
            if remaining_samples > 0:
                futures.append(executor.submit(sample_chunk, remaining_samples))

            sampled_points = np.vstack(
                [future.result() for future in as_completed(futures)]
            )

        logging.info(f"Sampling complete: {num_samples} points generated.")
        return sampled_points
