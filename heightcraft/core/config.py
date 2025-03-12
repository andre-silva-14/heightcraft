"""
Configuration classes for Heightcraft.

This module provides configuration classes for the application, using dataclasses
for clarity, type validation, and immutability.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union


class OutputFormat(Enum):
    """Supported output formats for height maps."""
    
    PNG = auto()
    TIFF = auto()
    JPEG = auto()
    
    @classmethod
    def from_extension(cls, extension: str) -> "OutputFormat":
        """Convert file extension to format enum."""
        ext = extension.lower()
        if ext in [".png"]:
            return cls.PNG
        elif ext in [".tiff", ".tif"]:
            return cls.TIFF
        elif ext in [".jpg", ".jpeg"]:
            return cls.JPEG
        return cls.PNG  # Default to PNG


class ProcessingMode(Enum):
    """Processing modes for different model sizes."""
    
    STANDARD = auto()  # Standard processing for regular models
    LARGE = auto()     # Memory-efficient processing for large models


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for 3D model processing."""
    
    file_path: str
    mode: ProcessingMode = ProcessingMode.STANDARD
    chunk_size: int = 1000000
    max_memory: float = 0.8
    cache_dir: Optional[str] = None
    num_threads: int = 4
    
    def __post_init__(self):
        """Validate configuration values."""
        if self.chunk_size <= 0:
            raise ValueError("Chunk size must be a positive integer")
        if not 0 < self.max_memory <= 1:
            raise ValueError("Max memory must be between 0 and 1")
        if self.num_threads <= 0:
            raise ValueError("Number of threads must be a positive integer")


@dataclass(frozen=True)
class SamplingConfig:
    """Configuration for point sampling."""
    
    num_samples: int
    use_gpu: bool = False
    num_threads: int = 4
    
    def __post_init__(self):
        """Validate configuration values."""
        if self.num_samples <= 0:
            raise ValueError("Number of samples must be a positive integer")
        if self.num_threads <= 0:
            raise ValueError("Number of threads must be a positive integer")


@dataclass(frozen=True)
class HeightMapConfig:
    """Configuration for height map generation."""
    
    max_resolution: int
    bit_depth: int = 16
    split: int = 1
    optimize_grid: bool = True
    
    def __post_init__(self):
        """Validate configuration values."""
        if self.max_resolution <= 0:
            raise ValueError("Maximum resolution must be a positive integer")
        if self.bit_depth not in [8, 16]:
            raise ValueError("Bit depth must be either 8 or 16")
        if self.split <= 0:
            raise ValueError("Split value must be a positive integer")


@dataclass(frozen=True)
class UpscaleConfig:
    """Configuration for upscaling."""
    
    enabled: bool = False
    upscale_factor: int = 2
    pretrained_model: Optional[str] = None
    
    def __post_init__(self):
        """Validate configuration values."""
        if self.enabled and self.upscale_factor not in [2, 3, 4]:
            raise ValueError("Upscale factor must be 2, 3, or 4")


@dataclass(frozen=True)
class OutputConfig:
    """Configuration for output options."""
    
    output_path: str
    format: OutputFormat = field(default=OutputFormat.PNG)
    
    def __post_init__(self):
        """Set format based on output path extension."""
        extension = Path(self.output_path).suffix
        object.__setattr__(self, 'format', OutputFormat.from_extension(extension))


@dataclass(frozen=True)
class ApplicationConfig:
    """Main application configuration combining all sub-configurations."""
    
    model_config: ModelConfig
    sampling_config: SamplingConfig
    height_map_config: HeightMapConfig
    output_config: OutputConfig
    upscale_config: UpscaleConfig = field(default_factory=lambda: UpscaleConfig())
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> "ApplicationConfig":
        """Create configuration from a dictionary."""
        # Determine processing mode
        processing_mode = ProcessingMode.STANDARD
        if config_dict.get("large_model", False):
            processing_mode = ProcessingMode.LARGE
        
        # Create model configuration
        model_config = ModelConfig(
            file_path=config_dict["file_path"],
            mode=processing_mode,
            chunk_size=config_dict.get("chunk_size", 1000000),
            max_memory=config_dict.get("max_memory", 0.8),
            cache_dir=config_dict.get("cache_dir"),
            num_threads=config_dict.get("num_threads", 4)
        )
        
        # Create sampling configuration
        sampling_config = SamplingConfig(
            num_samples=config_dict.get("num_samples", 100000),
            use_gpu=config_dict.get("use_gpu", False),
            num_threads=config_dict.get("num_threads", 4)
        )
        
        # Create height map configuration
        height_map_config = HeightMapConfig(
            max_resolution=config_dict.get("max_resolution", 256),
            bit_depth=config_dict.get("bit_depth", 16),
            split=config_dict.get("split", 1)
        )
        
        # Create output configuration
        output_config = OutputConfig(
            output_path=config_dict.get("output_path", "height_map.png")
        )
        
        # Create upscale configuration
        upscale_config = UpscaleConfig(
            enabled=config_dict.get("upscale", False),
            upscale_factor=config_dict.get("upscale_factor", 2),
            pretrained_model=config_dict.get("pretrained_model")
        )
        
        return cls(
            model_config=model_config,
            sampling_config=sampling_config,
            height_map_config=height_map_config,
            output_config=output_config,
            upscale_config=upscale_config
        )
        
    @classmethod
    def from_args(cls, args) -> "ApplicationConfig":
        """Create configuration from command-line arguments."""
        # Convert args to dict
        config_dict = vars(args)
        return cls.from_dict(config_dict) 