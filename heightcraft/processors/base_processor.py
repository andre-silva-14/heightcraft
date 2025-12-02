"""
Base processor for Heightcraft.

This module defines the base processor class that all processing strategies must implement.
It follows the Strategy pattern to allow for different processing approaches.
"""

import abc
import logging
from typing import Dict, Optional, Tuple, Union, Generator, Any

import numpy as np
import trimesh

from heightcraft.core.config import ApplicationConfig, ModelConfig, SamplingConfig
from heightcraft.core.exceptions import ProcessingError


class BaseProcessor(abc.ABC):
    """
    Abstract base class for all processors.
    
    This class defines the interface that all processing strategies must implement.
    It follows the Strategy pattern to allow for different processing approaches.
    """
    
    def __init__(self, config: ApplicationConfig):
        """
        Initialize the processor.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.model_config = config.model_config
        self.sampling_config = config.sampling_config
        self.height_map_config = config.height_map_config
        self.output_config = config.output_config
        self.upscale_config = config.upscale_config
        
        # Initialize state
        self.mesh: Optional[trimesh.Trimesh] = None
        self.points: Optional[np.ndarray] = None
        self.height_map: Optional[np.ndarray] = None
        self.bounds: Dict[str, float] = {}
        
        # Set up logging
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abc.abstractmethod
    def load_model(self) -> None:
        """
        Load the 3D model.
        
        This method must be implemented by all subclasses.
        
        Raises:
            ProcessingError: If the model cannot be loaded
        """
        pass
    
    @abc.abstractmethod
    def sample_points(self) -> Union[np.ndarray, Generator[np.ndarray, None, None]]:
        """
        Sample points from the 3D model.
        
        This method must be implemented by all subclasses.
        
        Returns:
            Sampled points (array or generator yielding arrays)
            
        Raises:
            ProcessingError: If point sampling fails
        """
        pass
    
    @abc.abstractmethod
    def generate_height_map(self) -> np.ndarray:
        """
        Generate a height map from sampled points.
        
        This method must be implemented by all subclasses.
        
        Returns:
            Generated height map
            
        Raises:
            ProcessingError: If height map generation fails
        """
        pass
    
    @abc.abstractmethod
    def save_height_map(self, output_path: Optional[str] = None) -> str:
        """
        Save the height map to disk.
        
        This method must be implemented by all subclasses.
        
        Args:
            output_path: Path to save the height map (defaults to config.output_config.output_path)
            
        Returns:
            Path to the saved height map
            
        Raises:
            ProcessingError: If the height map cannot be saved
        """
        pass
    
    def process(self) -> str:
        """
        Run the full processing pipeline.
        
        Returns:
            Path to the saved height map
            
        Raises:
            ProcessingError: If any step fails
        """
        try:
            self.logger.info("Starting processing pipeline")
            
            # 1. Load model
            self.logger.info("Loading 3D model")
            self.load_model()
            
            # 2. Sample points
            self.logger.info(f"Sampling {self.sampling_config.num_samples} points")
            self.points = self.sample_points()
            
            # 3. Generate height map
            self.logger.info("Generating height map")
            self.height_map = self.generate_height_map()
            
            # 4. Upscale height map (if enabled)
            if self.config.upscale_config.enabled:
                self.logger.info("Upscaling enabled, performing upscaling step")
                self.upscale_height_map()
            
            # 5. Save height map
            output_path = self.save_height_map()
            
            self.logger.info(f"Processing complete. Result saved to {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Processing failed: {e}")
            self.cleanup()
            raise ProcessingError(f"Processing failed: {e}")

    @abc.abstractmethod
    def upscale_height_map(self) -> None:
        """
        Upscale the generated height map.
        
        This method must be implemented by all subclasses.
        
        Raises:
            ProcessingError: If upscaling fails
        """
        pass

    
    def cleanup(self) -> None:
        """
        Clean up resources.
        
        This method should be called when the processor is no longer needed.
        """
        self.mesh = None
        self.points = None
        self.height_map = None
        self.bounds.clear()
        
        # Force garbage collection to release memory
        import gc
        gc.collect()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup() 