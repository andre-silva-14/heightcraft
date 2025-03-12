"""
Large model processor for Heightcraft.

This module provides the LargeModelProcessor class for processing large 3D models
with memory-efficient techniques.
"""

import gc
import logging
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import trimesh

from heightcraft.core.config import ApplicationConfig, ModelConfig, SamplingConfig
from heightcraft.core.exceptions import (
    HeightMapGenerationError,
    ModelLoadError,
    ProcessingError,
    SamplingError,
)
from heightcraft.domain.height_map import HeightMap
from heightcraft.domain.mesh import Mesh
from heightcraft.domain.point_cloud import PointCloud
from heightcraft.infrastructure.cache_manager import CacheManager
from heightcraft.infrastructure.gpu_manager import gpu_manager
from heightcraft.infrastructure.profiler import profiler
from heightcraft.processors.base_processor import BaseProcessor
from heightcraft.services.mesh_service import MeshService
from heightcraft.services.model_service import ModelService
from heightcraft.services.height_map_service import HeightMapService
from heightcraft.services.sampling_service import SamplingService
from heightcraft.services.point_cloud_service import PointCloudService
from heightcraft.utils.threading import ThreadPool


class LargeModelProcessor(BaseProcessor):
    """
    Processor for large 3D models with memory-efficient techniques.
    
    This processor implements memory-efficient techniques for loading,
    sampling, and processing large 3D models that may not fit in memory.
    """
    
    def __init__(self, config: ApplicationConfig):
        """
        Initialize the processor.
        
        Args:
            config: Application configuration
        """
        super().__init__(config)
        
        # Get services from config
        self.mesh_service = config.get_service(MeshService)
        self.model_service = config.get_service(ModelService)
        self.height_map_service = config.get_service(HeightMapService)
        self.sampling_service = config.get_service(SamplingService)
        
        # Initialize cache manager
        self.cache_manager = None
        if self.model_config.cache_dir:
            self.cache_manager = CacheManager(self.model_config.cache_dir)
        
        # Initialize other required attributes
        self.mesh = None
        self.points = None
        self.height_map = None
        self.bounds = {}
        self.chunks = []
        self._temp_files = []  # List to track temporary files
        
        # Logging
        self.logger.info(f"Initialized LargeModelProcessor for large model processing")
        
        # Use the chunk size from the configuration
        self.chunk_size = self.model_config.chunk_size
        self.logger.info(f"Using chunk size: {self.chunk_size}")
        
        # Initialize state
        self.is_scene = False
        self.vertex_buffer = None
        self.face_buffer = None
    
    @profiler.profile()
    def load_model(self) -> None:
        """
        Load a 3D model with memory-efficient techniques.
        
        Raises:
            ModelLoadError: If the model cannot be loaded
        """
        try:
            self.logger.info(f"Loading large model from {self.model_config.file_path}")
            
            # Load the model
            raw_mesh = trimesh.load(self.model_config.file_path, process=False)
            
            # Check if it's a scene
            if isinstance(raw_mesh, trimesh.Scene):
                self.is_scene = True
                self.logger.info("Detected a scene with multiple meshes")
                self._process_scene(raw_mesh)
            else:
                self.is_scene = False
                self.logger.info("Detected a single mesh")
                self._process_single_mesh(raw_mesh)
            
            # Force garbage collection
            gc.collect()
            
            self.logger.info("Model loading complete")
            
        except Exception as e:
            raise ModelLoadError(f"Failed to load model: {e}")
    
    @profiler.profile()
    def _process_scene(self, scene: trimesh.Scene) -> None:
        """
        Process a scene with multiple meshes.
        
        Args:
            scene: The scene to process
            
        Raises:
            ModelLoadError: If the scene cannot be processed
        """
        # Clear existing state
        self.chunks = []
        self.vertex_buffer = None
        self.face_buffer = None
        
        try:
            # Get geometry count
            geometry_count = len(scene.geometry)
            self.logger.info(f"Processing scene with {geometry_count} geometries")
            
            # Initialize vertex buffer
            self.vertex_buffer = []
            vertex_offset = 0
            
            # Process each geometry in chunks
            with ThreadPool(max_workers=self.sampling_config.num_threads) as pool:
                for node_name in scene.graph.nodes_geometry:
                    # Get geometry
                    transform, geometry_name = scene.graph[node_name]
                    mesh = scene.geometry[geometry_name]
                    
                    self.logger.info(f"Processing node {node_name} with {len(mesh.vertices)} vertices")
                    
                    # Process vertices in chunks
                    vertex_chunks = np.array_split(
                        mesh.vertices,
                        max(1, len(mesh.vertices) // self.chunk_size)
                    )
                    
                    # Transform and store vertices
                    for chunk in vertex_chunks:
                        transformed_chunk = trimesh.transform_points(chunk, transform)
                        self.vertex_buffer.append(transformed_chunk)
                    
                    # Process faces with offset
                    face_chunks = np.array_split(
                        mesh.faces,
                        max(1, len(mesh.faces) // self.chunk_size)
                    )
                    
                    for chunk in face_chunks:
                        offset_chunk = chunk + vertex_offset
                        self.chunks.append({
                            "vertices": len(self.vertex_buffer) - len(vertex_chunks),
                            "vertex_count": len(vertex_chunks),
                            "faces": offset_chunk
                        })
                    
                    # Update vertex offset
                    vertex_offset += len(mesh.vertices)
            
            # Create a mesh from the first chunks for validation and bounds
            sample_data = {"vertices": np.vstack(self.vertex_buffer[:min(10, len(self.vertex_buffer))]), 
                           "faces": np.vstack([chunk["faces"] for chunk in self.chunks[:min(10, len(self.chunks))]])}
            self.mesh = Mesh(trimesh.Trimesh(**sample_data))
            
            # Center and align
            self._center_and_align()
            
            self.logger.info(f"Scene processing complete: {vertex_offset} vertices, {len(self.chunks)} chunks")
            
        except Exception as e:
            raise ModelLoadError(f"Failed to process scene: {e}")
    
    @profiler.profile()
    def _process_single_mesh(self, mesh: trimesh.Trimesh) -> None:
        """
        Process a single large mesh.
        
        Args:
            mesh: The mesh to process
            
        Raises:
            ModelLoadError: If the mesh cannot be processed
        """
        # Clear existing state
        self.chunks = []
        self.vertex_buffer = None
        self.face_buffer = None
        
        try:
            # Process in chunks
            vertex_chunks = np.array_split(
                mesh.vertices,
                max(1, len(mesh.vertices) // self.chunk_size)
            )
            
            face_chunks = np.array_split(
                mesh.faces,
                max(1, len(mesh.faces) // self.chunk_size)
            )
            
            # Store vertices
            self.vertex_buffer = vertex_chunks
            
            # Store face chunks
            for i, chunk in enumerate(face_chunks):
                self.chunks.append({
                    "vertices": 0,
                    "vertex_count": len(vertex_chunks),
                    "faces": chunk
                })
            
            # Create a mesh from the first chunks for validation and bounds
            sample_data = {"vertices": np.vstack(self.vertex_buffer[:min(10, len(self.vertex_buffer))]), 
                           "faces": np.vstack([chunk["faces"] for chunk in self.chunks[:min(10, len(self.chunks))]])}
            self.mesh = Mesh(trimesh.Trimesh(**sample_data))
            
            # Center and align
            self._center_and_align()
            
            self.logger.info(f"Mesh processing complete: {len(mesh.vertices)} vertices, {len(self.chunks)} chunks")
            
        except Exception as e:
            raise ModelLoadError(f"Failed to process mesh: {e}")
    
    @profiler.profile()
    def _center_and_align(self) -> None:
        """
        Center and align the model.
        
        This operation affects all vertices in the buffer.
        """
        # Calculate centroid from first vertex chunk
        if self.vertex_buffer and len(self.vertex_buffer) > 0:
            sample_vertices = self.vertex_buffer[0]
            centroid = np.mean(sample_vertices, axis=0)
            
            # Center each vertex chunk
            self.logger.info(f"Centering model by translating by {-centroid}")
            for i in range(len(self.vertex_buffer)):
                self.vertex_buffer[i] = self.vertex_buffer[i] - centroid
            
            # Align to XY plane
            self.logger.info("Aligning model to XY plane")
            rotation = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
            for i in range(len(self.vertex_buffer)):
                self.vertex_buffer[i] = trimesh.transform_points(self.vertex_buffer[i], rotation)
    
    @profiler.profile()
    def sample_points(self) -> np.ndarray:
        """
        Sample points from the model with memory-efficient techniques.
        
        Returns:
            Sampled points
            
        Raises:
            SamplingError: If point sampling fails
        """
        try:
            num_samples = self.sampling_config.num_samples
            use_gpu = self.sampling_config.use_gpu
            
            self.logger.info(f"Sampling {num_samples} points from the large model")
            
            # Handle normal sampling
            if self.is_scene:
                return self._sample_points_from_scene(num_samples, use_gpu)
            else:
                return self._sample_points_from_chunks(num_samples, use_gpu)
            
        except Exception as e:
            raise SamplingError(f"Failed to sample points: {e}")
    
    @profiler.profile()
    def _sample_points_from_scene(self, num_samples: int, use_gpu: bool) -> np.ndarray:
        """
        Sample points from a scene.
        
        Args:
            num_samples: Number of points to sample
            use_gpu: Whether to use GPU for sampling
            
        Returns:
            Sampled points
            
        Raises:
            SamplingError: If point sampling fails
        """
        # Calculate area for each geometry to distribute samples proportionally
        total_faces = sum(len(chunk["faces"]) for chunk in self.chunks)
        
        # Distribute samples proportionally to face count
        samples_per_chunk = []
        for chunk in self.chunks:
            face_ratio = len(chunk["faces"]) / total_faces
            chunk_samples = max(1, int(num_samples * face_ratio))
            samples_per_chunk.append(chunk_samples)
        
        # Adjust to ensure we get exactly num_samples
        total_assigned = sum(samples_per_chunk)
        if total_assigned < num_samples:
            # Add remaining samples to the largest chunk
            largest_idx = samples_per_chunk.index(max(samples_per_chunk))
            samples_per_chunk[largest_idx] += num_samples - total_assigned
        elif total_assigned > num_samples:
            # Remove excess samples from the largest chunk
            largest_idx = samples_per_chunk.index(max(samples_per_chunk))
            samples_per_chunk[largest_idx] -= total_assigned - num_samples
        
        # Sample points from each chunk
        point_clouds = []
        
        with ThreadPool(max_workers=self.sampling_config.num_threads) as pool:
            for i, (chunk, chunk_samples) in enumerate(zip(self.chunks, samples_per_chunk)):
                if chunk_samples <= 0:
                    continue
                
                self.logger.debug(f"Sampling {chunk_samples} points from chunk {i+1}/{len(self.chunks)}")
                
                # Create a mesh for this chunk
                vertices = np.vstack(self.vertex_buffer[chunk["vertices"]:chunk["vertices"] + chunk["vertex_count"]])
                faces = chunk["faces"]
                chunk_mesh = Mesh(trimesh.Trimesh(vertices=vertices, faces=faces))
                
                # Sample points
                chunk_points = self.sampling_service.sample_points(
                    chunk_mesh, chunk_samples, use_gpu, 
                    self.sampling_config.num_threads
                )
                
                point_clouds.append(chunk_points)
        
        # Merge results
        return PointCloud.merge(point_clouds).points
    
    @profiler.profile()
    def _sample_points_from_chunks(self, num_samples: int, use_gpu: bool) -> np.ndarray:
        """
        Sample points from mesh chunks.
        
        Args:
            num_samples: Number of points to sample
            use_gpu: Whether to use GPU for sampling
            
        Returns:
            Sampled points
            
        Raises:
            SamplingError: If point sampling fails
        """
        # Create a single mesh from all chunks
        vertices = np.vstack(self.vertex_buffer)
        faces = np.vstack([chunk["faces"] for chunk in self.chunks])
        
        # Create mesh
        mesh = Mesh(trimesh.Trimesh(vertices=vertices, faces=faces))
        
        # Sample points
        points = self.sampling_service.sample_points(
            mesh, num_samples, use_gpu, self.sampling_config.num_threads
        )
        
        return points.points
    
    @profiler.profile()
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
            
            # Create point cloud
            point_cloud = PointCloud(self.points)
            
            # Calculate target resolution
            width, height = self._calculate_target_resolution(point_cloud)
            resolution = (width, height)
            
            # Generate height map
            height_map = self.height_map_service.generate_height_map(
                point_cloud, resolution, self.height_map_config.bit_depth
            )
            
            return height_map.data
            
        except Exception as e:
            raise HeightMapGenerationError(f"Failed to generate height map: {e}")
    
    @profiler.profile()
    def _calculate_target_resolution(self, point_cloud: PointCloud) -> Tuple[int, int]:
        """
        Calculate target resolution based on point cloud proportions.
        
        Args:
            point_cloud: The point cloud
            
        Returns:
            Tuple of (width, height)
        """
        # Get max resolution from config
        max_resolution = self.height_map_config.max_resolution
        
        # Calculate aspect ratio
        aspect_ratio = point_cloud.get_aspect_ratio()
        
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
        
        self.logger.info(f"Calculated target resolution: {width}x{height}")
        
        return width, height
    
    @profiler.profile()
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
            
            # Create height map domain object
            height_map = HeightMap(self.height_map, self.height_map_config.bit_depth)
            
            # Handle splitting
            if self.height_map_config.split > 1:
                # Split height map
                split_maps = self.height_map_service.split_height_map(
                    height_map, self.height_map_config.split
                )
                
                # Save split maps
                output_dir = self.height_map_service.save_split_height_maps(
                    split_maps, output_path
                )
                
                self.logger.info(f"Split height maps saved to {output_dir}")
                return output_dir
            else:
                # Save single height map
                saved_path = self.height_map_service.save_height_map(
                    height_map, output_path
                )
                
                self.logger.info(f"Height map saved to {saved_path}")
                return saved_path
            
        except Exception as e:
            raise ProcessingError(f"Failed to save height map: {e}")
    
    @profiler.profile()
    def cleanup(self) -> None:
        """Clean up resources."""
        super().cleanup()
        
        # Clear vertex buffer
        if self.vertex_buffer:
            self.vertex_buffer.clear()
            self.vertex_buffer = None
        
        # Clear chunks
        self.chunks.clear()
        
        # Force garbage collection
        gc.collect()
    
    @profiler.profile()
    def load_model_info(self, file_path: str) -> Dict:
        """
        Load model information without loading the entire model.
        
        Args:
            file_path: Path to the model file
            
        Returns:
            Dictionary with model information
            
        Raises:
            ProcessingError: If model information cannot be loaded
        """
        try:
            # Load the mesh
            mesh = self.mesh_service.load_mesh(file_path)
            
            # Get model info
            return {
                "vertex_count": mesh.vertex_count,
                "face_count": mesh.face_count,
                "is_watertight": mesh.is_watertight,
                "has_degenerate_faces": mesh.has_degenerate_faces,
                "is_winding_consistent": mesh.is_winding_consistent,
                "bounds": mesh.bounds,
                "file_path": file_path
            }
        except Exception as e:
            raise ProcessingError(f"Failed to load model info: {str(e)}")
            
    @profiler.profile()
    def align_model(self, mesh: Mesh) -> Mesh:
        """
        Align a mesh to the XY plane.
        
        Args:
            mesh: The mesh to align
            
        Returns:
            Aligned mesh
            
        Raises:
            ProcessingError: If the mesh cannot be aligned
        """
        try:
            return self.mesh_service.align_mesh_to_xy(mesh)
        except Exception as e:
            raise ProcessingError(f"Failed to align model: {str(e)}")
            
    @profiler.profile()
    def center_model(self, mesh: Mesh) -> Mesh:
        """
        Center a mesh at the origin.
        
        Args:
            mesh: The mesh to center
            
        Returns:
            Centered mesh
            
        Raises:
            ProcessingError: If the mesh cannot be centered
        """
        try:
            return self.mesh_service.center_mesh(mesh)
        except Exception as e:
            raise ProcessingError(f"Failed to center model: {str(e)}")
            
    @profiler.profile()
    def calculate_bounding_box(self, mesh: Mesh) -> np.ndarray:
        """
        Calculate the bounding box of a mesh.
        
        Args:
            mesh: The mesh to calculate the bounding box for
            
        Returns:
            Bounding box as a numpy array [[min_x, min_y, min_z], [max_x, max_y, max_z]]
            
        Raises:
            ProcessingError: If the bounding box cannot be calculated
        """
        try:
            return self.mesh_service.get_bounds(mesh)
        except Exception as e:
            raise ProcessingError(f"Failed to calculate bounding box: {str(e)}")
            
    @profiler.profile()
    def sample_points(self, mesh: Mesh, num_points: int) -> PointCloud:
        """
        Sample points from a mesh.
        
        Args:
            mesh: The mesh to sample points from
            num_points: Number of points to sample
            
        Returns:
            Point cloud with sampled points
            
        Raises:
            ProcessingError: If points cannot be sampled
        """
        try:
            return self.mesh_service.mesh_to_point_cloud(mesh, num_points)
        except Exception as e:
            raise ProcessingError(f"Failed to sample points: {str(e)}") 