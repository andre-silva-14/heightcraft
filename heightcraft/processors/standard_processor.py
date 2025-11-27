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
from heightcraft.domain.height_map import HeightMap
from heightcraft.domain.point_cloud import PointCloud
from heightcraft.services.height_map_service import HeightMapService
from heightcraft.services.mesh_service import MeshService
from heightcraft.services.sampling_service import SamplingService


class StandardProcessor(BaseProcessor):
    """
    Standard processor for regular-sized 3D models.
    
    This processor implements the BaseProcessor interface and provides concrete
    implementations for loading models, sampling points, generating height maps,
    and saving results.
    """
    
    def __init__(self, config: ApplicationConfig):
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
        
        # Initialize services
        self.height_map_service = HeightMapService()
        self.mesh_service = MeshService()
        self.sampling_service = SamplingService(config.sampling_config)
        
        # Lazy import to avoid TensorFlow overhead/crashes
        from heightcraft.services.upscaling_service import UpscalingService
        self.upscaling_service = UpscalingService(
            config=config.upscale_config,
            cache_manager=self.cache_manager,
            height_map_service=self.height_map_service
        )
    
    def load_model(self) -> None:
        """
        Load the 3D model.
        
        Raises:
            ModelLoadError: If the model cannot be loaded
        """
        try:
            self.logger.info(f"Loading 3D model from {self.model_config.file_path}")
            
            # Load mesh using service
            self.mesh = self.mesh_service.load_mesh(self.model_config.file_path)
            
            # Center and align using service
            self.mesh = self.mesh_service.prepare_mesh(self.mesh, center=True, align=True)
            
            self.logger.info(f"Model loaded with {len(self.mesh.vertices)} vertices and {len(self.mesh.faces)} faces")
            
        except Exception as e:
            raise ModelLoadError(f"Failed to load model: {e}")
    
    def sample_points(self) -> np.ndarray:
        """
        Sample points from the 3D model.
        
        Returns:
            Sampled points
            
        Raises:
            SamplingError: If point sampling fails
        """
        try:
            self.logger.info("Sampling points using SamplingService")
            
            # Use SamplingService to sample points
            # It handles CPU/GPU switching and threading based on config
            self.points = self.sampling_service.sample_from_mesh(self.mesh)
            
            return self.points
        except Exception as e:
            self.logger.error(f"Point sampling failed: {e}")
            raise SamplingError(f"Failed to sample points: {e}")
    
    def _calculate_target_resolution(self) -> Tuple[int, int]:
        """
        Calculate target resolution based on model proportions.
        
        Returns:
            Tuple of width and height
        """
        # Use ResolutionCalculator
        from heightcraft.utils.resolution_calculator import ResolutionCalculator
        calculator = ResolutionCalculator()
        
        width, height = calculator.calculate_resolution_from_bounds(
            self.bounds,
            max_resolution=self.height_map_config.max_resolution
        )
        
        # Ensure minimum size
        width = max(width, 32)
        height = max(height, 32)
        
        return width, height
            
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
            
            self.logger.info("Generating height map using HeightMapService")
            
            # Calculate target resolution using ResolutionCalculator
            width, height = self._calculate_target_resolution()
            
            # Create PointCloud domain object
            point_cloud = PointCloud(self.points)
            
            # Generate height map using HeightMapService
            height_map_obj = self.height_map_service.generate_from_point_cloud(
                point_cloud,
                (width, height),
                bit_depth=self.height_map_config.bit_depth,
                num_threads=self.sampling_config.num_threads
            )
            
            # Store the raw data for internal use (StandardProcessor exposes .height_map as ndarray)
            # But we also need to store the object for save_height_map if we want to avoid recreating it
            # However, save_height_map currently recreates it from self.height_map (ndarray)
            # We should probably update save_height_map to use the object if available
            # For now, let's just set self.height_map to the data to maintain backward compatibility
            self.height_map = height_map_obj.data
            
            # Return the data
            return self.height_map
            
        except Exception as e:
            raise HeightMapGenerationError(f"Failed to generate height map: {e}")

    def upscale_height_map(self) -> None:
        """
        Upscale the generated height map.
        
        Raises:
            ProcessingError: If upscaling fails
        """
        try:
            if self.height_map is None:
                raise ProcessingError("No height map available. Call generate_height_map() first.")
            
            self.logger.info("Upscaling height map using UpscalingService")
            
            # Create HeightMap domain object
            height_map_obj = HeightMap(self.height_map, self.height_map_config.bit_depth)
            
            # Upscale using service
            upscaled_map = self.upscaling_service.upscale(
                height_map_obj,
                scale_factor=self.config.upscale_config.upscale_factor,
                use_gpu=self.config.sampling_config.use_gpu
            )
            
            # Update internal state
            self.height_map = upscaled_map.data
            
            self.logger.info(f"Height map upscaled to {self.height_map.shape}")
            
        except Exception as e:
            raise ProcessingError(f"Failed to upscale height map: {e}")
    
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
            
            # Create HeightMap domain object
            height_map = HeightMap(self.height_map, self.height_map_config.bit_depth)
            
            # Handle splitting
            # Handle splitting
            if self.height_map_config.split > 1:
                # Split height map using service (handles grid size calculation)
                split_maps = self.height_map_service.split_height_map(height_map, self.height_map_config.split)
                
                # Save split maps
                return self.height_map_service.save_split_height_maps(
                    split_maps, output_path
                )
            else:
                # Save single height map
                self.height_map_service.save_height_map(height_map, output_path)
                return output_path
                
        except Exception as e:
            raise ProcessingError(f"Failed to save height map: {e}")
    
    def cleanup(self) -> None:
        """Clean up resources."""
        super().cleanup()
        
        # Force garbage collection
        import gc
        gc.collect() 