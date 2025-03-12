"""
Standard processor for Heightcraft.

This module provides a standard processor implementation for regular-sized 3D models.
It implements the BaseProcessor interface and provides concrete implementations
for loading models, sampling points, generating height maps, and saving results.
"""

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import trimesh

from heightcraft.core.config import ApplicationConfig, OutputFormat
from heightcraft.core.exceptions import (
    HeightMapGenerationError,
    ModelLoadError,
    ProcessingError,
    SamplingError,
)
from heightcraft.processors.base_processor import BaseProcessor


class StandardProcessor(BaseProcessor):
    """
    Standard processor for regular-sized 3D models.
    
    This processor implements the BaseProcessor interface and provides concrete
    implementations for loading models, sampling points, generating height maps,
    and saving results.
    """
    
    def __init__(self, config):
        """
        Initialize the processor.
        
        Args:
            config: Application configuration
        """
        super().__init__(config)
        
        # Add debug logging
        logging.debug(f"Initializing StandardProcessor with config: {config}")
        
        # Initialize GPU manager if GPU is enabled
        use_gpu = config.sampling_config.use_gpu
        
        if use_gpu:
            from heightcraft.infrastructure.gpu_manager import GPUManager
            self.gpu_manager = GPUManager.get_instance()
            if self.gpu_manager is None:
                logging.warning("GPU requested but not available. Falling back to CPU.")
        else:
            logging.info("GPU acceleration not requested, using CPU only")
            self.gpu_manager = None
        
        # Initialize caching system
        self.cache_manager = None
        if self.model_config.cache_dir:
            from heightcraft.infrastructure.cache_manager import CacheManager
            self.cache_manager = CacheManager(self.model_config.cache_dir)
        
        # Initialize additional attributes
        self.mesh = None
        self.points = None
        self.height_map = None
        self.bounds = {}
        
        self.thread_pool = ThreadPoolExecutor(max_workers=self.sampling_config.num_threads)
    
    def load_model(self) -> None:
        """
        Load the 3D model.
        
        Raises:
            ModelLoadError: If the model cannot be loaded
        """
        try:
            self.logger.info(f"Loading 3D model from {self.model_config.file_path}")
            self.mesh = trimesh.load(self.model_config.file_path)
            
            # Validate mesh
            if not isinstance(self.mesh, trimesh.Trimesh):
                if isinstance(self.mesh, trimesh.Scene):
                    self.logger.info("Converting scene to mesh")
                    self.mesh = self._process_scene(self.mesh)
                else:
                    raise ModelLoadError(f"Unsupported model type: {type(self.mesh)}")
            
            # Validate mesh geometry
            if len(self.mesh.vertices) == 0:
                raise ModelLoadError("Mesh has no vertices")
            if len(self.mesh.faces) == 0:
                raise ModelLoadError("Mesh has no faces")
            
            # Log warnings for potential issues
            if not self.mesh.is_watertight:
                self.logger.warning("Mesh is not watertight, which may affect sampling quality")
            if not self.mesh.is_winding_consistent:
                self.logger.warning("Mesh has inconsistent face winding")
            
            # Center and align the mesh
            self._center_model()
            self._align_model()
            
            self.logger.info(f"Model loaded with {len(self.mesh.vertices)} vertices and {len(self.mesh.faces)} faces")
            
        except Exception as e:
            raise ModelLoadError(f"Failed to load model: {e}")

    def _process_scene(self, scene: trimesh.Scene) -> trimesh.Trimesh:
        """Process a scene by combining all meshes and centering/aligning."""
        self.logger.info("Processing scene by combining all meshes and centering/aligning")

        if not isinstance(scene, trimesh.Scene):
            raise ValueError(f"Unsupported object type: {type(scene)}")
        
        meshes = []
        for name, geometry in scene.geometry.items():
            if isinstance(geometry, trimesh.Trimesh):
                geometry_transformation = scene.graph[name][0]
                geometry.apply_transform(geometry_transformation)
                meshes.append(geometry)
        if not meshes:
            raise ValueError("No valid meshes found in the scene")
        return trimesh.util.concatenate(meshes)
    
    
    def _center_model(self) -> None:
        """Center the model at the origin."""
        self.logger.info("Centering the model at the origin")
        centroid = np.mean(self.mesh.vertices, axis=0)
        self.mesh.apply_translation(-centroid)
    
    def _align_model(self) -> None:
        """Align the model to the XY plane."""
        self.logger.info("Aligning the model to the XY plane")
        
        # FIXED: Ensure Z is up with correct orientation
        # First, determine which axis should be up based on model extents
        extents = self.mesh.extents
        
        # Use the smallest dimension as the up axis
        up_axis = np.argmin(extents)
        
        # Create rotation matrix to align that axis with Z (up)
        if up_axis == 0:  # X is smallest
            rotation = trimesh.transformations.rotation_matrix(np.pi/2, [0, 1, 0])
        elif up_axis == 1:  # Y is smallest
            rotation = trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0])
        else:  # Z is already up
            rotation = np.eye(4)
        
        # Apply the transformation
        self.mesh.apply_transform(rotation)
    
    def sample_points(self) -> np.ndarray:
        """
        Sample points from the 3D model.
        
        Returns:
            Sampled points
            
        Raises:
            SamplingError: If point sampling fails
        """
        try:
            num_samples = self.sampling_config.num_samples
            use_gpu = self.sampling_config.use_gpu
            num_threads = self.sampling_config.num_threads
            
            self.logger.info(f"Sampling {num_samples} points (use_gpu={use_gpu}, threads={num_threads})")
            
            # Use the appropriate sampling method based on configuration
            if use_gpu and hasattr(self, 'gpu_manager') and self.gpu_manager is not None:
                self.logger.info("Using GPU acceleration for point sampling")
                points = self._sample_points_on_gpu(num_samples)
            else:
                if use_gpu:
                    self.logger.warning("GPU sampling requested but unavailable. Using CPU instead.")
                self.logger.info("Using CPU for point sampling")
                points = self._sample_points_on_cpu(num_samples, num_threads)
            
            return points
        except Exception as e:
            self.logger.error(f"Point sampling failed: {e}")
            raise SamplingError(f"Failed to sample points: {e}")
    
    def _sample_points_on_cpu(self, num_samples: int, num_threads: int) -> np.ndarray:
        """
        Sample points using CPU with multithreading.
        
        Args:
            num_samples: Number of points to sample
            num_threads: Number of threads to use
            
        Returns:
            Sampled points
        """
        self.logger.info(f"Sampling points using CPU with {num_threads} threads")
        
        # Calculate samples per thread
        samples_per_thread = num_samples // num_threads
        remaining_samples = num_samples % num_threads
        
        # Define sampling function
        def sample_chunk(chunk_size):
            return self.mesh.sample(chunk_size)
        
        # Sample points in parallel
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(sample_chunk, samples_per_thread)
                for _ in range(num_threads)
            ]
            
            # Handle remaining samples
            if remaining_samples > 0:
                futures.append(executor.submit(sample_chunk, remaining_samples))
            
            # Collect results
            sampled_points = np.vstack([future.result() for future in futures])
        
        self.logger.info(f"Sampling complete: {len(sampled_points)} points generated")
        return sampled_points
    
    def _sample_points_on_gpu(self, num_samples: int) -> np.ndarray:
        """
        Sample points using GPU with a self-contained implementation.
        
        Args:
            num_samples: Number of points to sample
            
        Returns:
            Sampled points
        """
        self.logger.info("Sampling points using GPU (self-contained implementation)")
        
        # Import PyTorch first to check availability
        try:
            import torch
            if not torch.cuda.is_available():
                self.logger.warning("PyTorch reports CUDA is not available. Falling back to CPU sampling.")
                return self._sample_points_on_cpu(num_samples, self.sampling_config.num_threads)
            
            # Log detailed GPU information
            device_count = torch.cuda.device_count()
            self.logger.info(f"PyTorch reports {device_count} CUDA device(s) available")
            for i in range(device_count):
                self.logger.info(f"Device {i}: {torch.cuda.get_device_name(i)}")
            
            self.logger.info(f"Current device: {torch.cuda.current_device()}")
        except ImportError:
            self.logger.error("PyTorch not installed. Falling back to CPU sampling.")
            return self._sample_points_on_cpu(num_samples, self.sampling_config.num_threads)
        
        try:
            # Explicitly set device to cuda:0
            device = torch.device("cuda:0")
            self.logger.info(f"Using device: {device}")
            
            # Ensure mesh data is in the right format
            vertices = np.ascontiguousarray(self.mesh.vertices, dtype=np.float32)
            faces = np.ascontiguousarray(self.mesh.faces, dtype=np.int64)
            
            self.logger.info(f"Creating vertex tensor of shape {vertices.shape} on {device}")
            vertices_tensor = torch.tensor(vertices, device=device)
            
            self.logger.info(f"Creating faces tensor of shape {faces.shape} on {device}")
            faces_tensor = torch.tensor(faces, device=device)
            
            # Get vertices for each face
            self.logger.info("Extracting face vertices")
            v0 = vertices_tensor[faces_tensor[:, 0]]
            v1 = vertices_tensor[faces_tensor[:, 1]]
            v2 = vertices_tensor[faces_tensor[:, 2]]
            
            # Calculate face areas
            self.logger.info("Calculating face areas")
            cross_product = torch.cross(v1 - v0, v2 - v0, dim=1)
            face_areas = 0.5 * torch.norm(cross_product, dim=1)
            
            # Handle case where some face areas might be close to zero
            face_areas = torch.clamp(face_areas, min=1e-10)
            
            # Calculate face probabilities
            total_area = torch.sum(face_areas)
            self.logger.info(f"Total mesh area: {total_area.item()}")
            face_probs = face_areas / total_area
            
            # Sample faces based on area
            self.logger.info(f"Sampling {num_samples} faces")
            face_indices = torch.multinomial(face_probs, num_samples, replacement=True)
            
            # Generate random points within each sampled face
            self.logger.info("Generating barycentric coordinates")
            rand_tensor = torch.rand(num_samples, 2, device=device)
            r1 = torch.sqrt(rand_tensor[:, 0])
            r2 = rand_tensor[:, 1]
            
            # Barycentric coordinates
            u = 1.0 - r1
            v = r1 * (1.0 - r2)
            w = r1 * r2
            
            # Get vertices of sampled faces
            self.logger.info("Computing final point positions")
            sampled_v0 = vertices_tensor[faces_tensor[face_indices, 0]]
            sampled_v1 = vertices_tensor[faces_tensor[face_indices, 1]]
            sampled_v2 = vertices_tensor[faces_tensor[face_indices, 2]]
            
            # Compute points using barycentric coordinates
            u_expanded = u.unsqueeze(1)
            v_expanded = v.unsqueeze(1)
            w_expanded = w.unsqueeze(1)
            
            sampled_points = (
                u_expanded * sampled_v0 + 
                v_expanded * sampled_v1 + 
                w_expanded * sampled_v2
            )
            
            # Move result back to CPU and convert to numpy
            self.logger.info("Transferring results back to CPU")
            result = sampled_points.cpu().numpy()
            
            # Explicit cleanup
            self.logger.info("Cleaning up GPU memory")
            del vertices_tensor, faces_tensor, v0, v1, v2
            del cross_product, face_areas, face_probs, face_indices
            del rand_tensor, r1, r2, u, v, w
            del sampled_v0, sampled_v1, sampled_v2, u_expanded, v_expanded, w_expanded, sampled_points
            torch.cuda.empty_cache()
            
            self.logger.info(f"GPU sampling complete: {len(result)} points generated")
            return result
            
        except Exception as e:
            self.logger.error(f"GPU sampling failed with error: {e}")
            # Add stack trace for debugging
            import traceback
            self.logger.error(f"Stack trace: {traceback.format_exc()}")
            
            # Clean up GPU memory
            try:
                torch.cuda.empty_cache()
            except:
                pass
            
            self.logger.info("Falling back to CPU sampling")
            return self._sample_points_on_cpu(num_samples, self.sampling_config.num_threads)
    
    def generate_height_map(self) -> np.ndarray:
        """
        Generate a height map from sampled points.
        
        Returns:
            Generated height map
            
        Raises:
            HeightMapGenerationError: If height map generation fails
        """
        try:
            if self.points is None:
                raise HeightMapGenerationError("No points available. Call sample_points() first.")
            
            # Extract 2D points and Z values
            points_2d = self.points[:, :2]
            z_values = self.points[:, 2]
            
            # Calculate bounds
            self.bounds = self._calculate_bounds(points_2d, z_values)
            
            # Create height map with parallel processing
            height_map = self._create_height_map(points_2d, z_values, self.bounds)
            
            # Apply bit depth conversion
            return self._convert_bit_depth(height_map)
            
        except Exception as e:
            raise HeightMapGenerationError(f"Failed to generate height map: {e}")
    
    def _calculate_bounds(self, points_2d: np.ndarray, z_values: np.ndarray) -> Dict[str, float]:
        """
        Calculate bounds for the height map.
        
        Args:
            points_2d: 2D points
            z_values: Z values
            
        Returns:
            Dictionary of bounds
        """
        return {
            "min_x": points_2d[:, 0].min(),
            "max_x": points_2d[:, 0].max(),
            "min_y": points_2d[:, 1].min(),
            "max_y": points_2d[:, 1].max(),
            "min_z": z_values.min(),
            "max_z": z_values.max(),
        }
    
    def _create_height_map(self, points_2d: np.ndarray, z_values: np.ndarray, bounds: Dict[str, float]) -> np.ndarray:
        """
        Create height map using parallel processing.
        
        Args:
            points_2d: 2D points
            z_values: Z values
            bounds: Dictionary of bounds
            
        Returns:
            Height map
        """
        # Get target resolution
        width, height = self._calculate_target_resolution()
        
        # Initialize height map
        height_map = np.zeros((height, width), dtype=np.float32)
        
        # Calculate coordinates in parallel
        futures = []
        chunk_size = len(points_2d) // self.sampling_config.num_threads
        
        for i in range(0, len(points_2d), chunk_size):
            chunk = slice(i, min(i + chunk_size, len(points_2d)))
            futures.append(
                self.thread_pool.submit(
                    self._process_chunk,
                    points_2d[chunk],
                    z_values[chunk],
                    width,
                    height,
                    bounds,
                )
            )
        
        # Combine results
        for future in futures:
            chunk_coords, chunk_values = future.result()
            np.maximum.at(
                height_map, (chunk_coords[:, 1], chunk_coords[:, 0]), chunk_values
            )
        
        return height_map
    
    def _process_chunk(
        self, points: np.ndarray, z_values: np.ndarray, width: int, height: int, bounds: Dict[str, float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process a chunk of points.
        
        Args:
            points: 2D points
            z_values: Z values
            width: Width of the height map
            height: Height of the height map
            bounds: Dictionary of bounds
            
        Returns:
            Tuple of coordinates and values
        """
        # Calculate pixel coordinates
        x_coords = (
            (points[:, 0] - bounds["min_x"])
            / (bounds["max_x"] - bounds["min_x"])
            * (width - 1)
        ).astype(int)
        
        y_coords = (
            (1.0 - (points[:, 1] - bounds["min_y"])
            / (bounds["max_y"] - bounds["min_y"]))
            * (height - 1)
        ).astype(int)
        
        # Normalize Z values
        z_normalized = (z_values - bounds["min_z"]) / (
            bounds["max_z"] - bounds["min_z"]
        )
        
        return np.column_stack((x_coords, y_coords)), z_normalized
    
    def _calculate_target_resolution(self) -> Tuple[int, int]:
        """
        Calculate target resolution based on model proportions.
        
        Returns:
            Tuple of width and height
        """
        max_resolution = self.height_map_config.max_resolution
        
        # Calculate aspect ratio
        x_range = self.bounds["max_x"] - self.bounds["min_x"]
        y_range = self.bounds["max_y"] - self.bounds["min_y"]
        
        aspect_ratio = x_range / y_range if y_range != 0 else 1.0
        
        # Calculate resolution
        if aspect_ratio >= 1.0:
            # Wider than tall
            width = max_resolution
            height = int(max_resolution / aspect_ratio)
        else:
            # Taller than wide
            height = max_resolution
            width = int(max_resolution * aspect_ratio)
        
        # Ensure minimum size
        width = max(width, 32)
        height = max(height, 32)
        
        return width, height
    
    def _convert_bit_depth(self, height_map: np.ndarray) -> np.ndarray:
        """
        Convert height map to specified bit depth.
        
        Args:
            height_map: Height map
            
        Returns:
            Converted height map
        """
        if self.height_map_config.bit_depth == 8:
            return (height_map * 255).astype(np.uint8)
        elif self.height_map_config.bit_depth == 16:
            return (height_map * 65535).astype(np.uint16)
        else:
            raise ValueError("Bit depth must be either 8 or 16")
    
    def save_height_map(self, output_path: Optional[str] = None) -> str:
        """
        Save the height map to disk.
        
        Args:
            output_path: Path to save the height map (defaults to config.output_config.output_path)
            
        Returns:
            Path to the saved height map
            
        Raises:
            ProcessingError: If the height map cannot be saved
        """
        try:
            if self.height_map is None:
                raise ProcessingError("No height map available. Call generate_height_map() first.")
            
            # Use provided output path or default from config
            output_path = output_path or self.output_config.output_path
            
            # Handle splitting
            if self.height_map_config.split == 1:
                return self._save_single_height_map(output_path)
            else:
                return self._save_split_height_maps(output_path)
                
        except Exception as e:
            raise ProcessingError(f"Failed to save height map: {e}")
    
    def _save_single_height_map(self, output_path: str) -> str:
        """
        Save a single height map.
        
        Args:
            output_path: Path to save the height map
            
        Returns:
            Path to the saved height map
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Get format
        file_format = self.output_config.format
        
        # Handle JPEG 16-bit conversion
        if file_format == OutputFormat.JPEG and self.height_map_config.bit_depth == 16:
            self.logger.warning(
                "JPEG format does not support 16-bit depth, converting to 8-bit"
            )
            height_map = self._convert_to_8bit(self.height_map)
        else:
            height_map = self.height_map
        
        # Save the image
        plt.imsave(output_path, height_map, cmap="gray")
        
        return output_path
    
    def _save_split_height_maps(self, output_path: str) -> str:
        """
        Save split height maps.
        
        Args:
            output_path: Base path for split height maps
            
        Returns:
            Directory containing split height maps
        """
        # Create output directory
        base_name = Path(output_path).stem
        output_dir = os.path.join(os.path.dirname(output_path), f"{base_name}_split")
        os.makedirs(output_dir, exist_ok=True)
        
        # Calculate grid dimensions
        split = self.height_map_config.split
        grid_size = int(split ** 0.5)
        
        # If not a perfect square, find the best grid
        if grid_size ** 2 != split:
            for i in range(2, int(split ** 0.5) + 1):
                if split % i == 0:
                    grid_size = i
                    break
        
        rows = split // grid_size
        cols = grid_size
        
        # Get height map dimensions
        height, width = self.height_map.shape
        
        # Calculate tile dimensions
        tile_height = height // rows
        tile_width = width // cols
        
        # Save each tile
        for i in range(rows):
            for j in range(cols):
                # Extract tile
                y_start = i * tile_height
                y_end = (i + 1) * tile_height if i < rows - 1 else height
                x_start = j * tile_width
                x_end = (j + 1) * tile_width if j < cols - 1 else width
                
                tile = self.height_map[y_start:y_end, x_start:x_end]
                
                # Save tile
                tile_path = os.path.join(output_dir, f"{base_name}_{i}_{j}{Path(output_path).suffix}")
                plt.imsave(tile_path, tile, cmap="gray")
                
                self.logger.info(f"Saved tile {i}_{j} to {tile_path}")
        
        return output_dir
    
    def _convert_to_8bit(self, height_map: np.ndarray) -> np.ndarray:
        """
        Convert 16-bit height map to 8-bit.
        
        Args:
            height_map: 16-bit height map
            
        Returns:
            8-bit height map
        """
        # Normalize to 0-1 range
        normalized = (height_map - np.min(height_map)) / (
            np.max(height_map) - np.min(height_map)
        )
        
        # Convert to 8-bit
        return (normalized * 255).astype(np.uint8)
    
    def cleanup(self) -> None:
        """Clean up resources."""
        super().cleanup()
        
        # Shut down thread pool
        self.thread_pool.shutdown(wait=True)
        
        # Force garbage collection
        import gc
        gc.collect() 