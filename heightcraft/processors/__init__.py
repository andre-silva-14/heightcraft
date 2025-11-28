"""
Processor factory for Heightcraft.

This module provides a factory for creating processors based on the application configuration.
It follows the Factory pattern to create the appropriate processor for the given configuration.
"""

from typing import Type

from heightcraft.core.config import ApplicationConfig, ProcessingMode
from heightcraft.processors.base_processor import BaseProcessor


def create_processor(config: ApplicationConfig) -> BaseProcessor:
    """
    Create a processor based on the application configuration.
    
    This function follows the Factory pattern to create the appropriate processor
    for the given configuration.
    
    Args:
        config: Application configuration
        
    Returns:
        Appropriate processor for the configuration
        
    Raises:
        ImportError: If the required processor cannot be imported
    """
    # Determine processor type based on processing mode
    if config.model_config.mode == ProcessingMode.STANDARD:
        from heightcraft.processors.standard_processor import StandardProcessor
        return StandardProcessor(config)
    elif config.model_config.mode == ProcessingMode.LARGE:
        from heightcraft.processors.large_model_processor import LargeModelProcessor
        return LargeModelProcessor(config)
    elif config.model_config.mode == ProcessingMode.LIDAR:
        from heightcraft.processors.lidar_processor import LidarProcessor
        return LidarProcessor(config)
    else:
        # Default to standard processor
        from heightcraft.processors.standard_processor import StandardProcessor
        return StandardProcessor(config) 