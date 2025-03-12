"""
Heightcraft exception hierarchy.

This module defines all custom exceptions used throughout the Heightcraft application.
Properly structured exceptions help with error handling, debugging, and user experience.
"""

class HeightcraftError(Exception):
    """Base exception for all Heightcraft-specific errors."""
    
    def __init__(self, message="An error occurred in Heightcraft", *args, **kwargs):
        self.message = message
        super().__init__(self.message, *args, **kwargs)


class ConfigurationError(HeightcraftError):
    """Raised when there is a configuration problem."""
    
    def __init__(self, message="Invalid configuration", *args, **kwargs):
        super().__init__(message, *args, **kwargs)


class ResourceError(HeightcraftError):
    """Base class for resource-related errors."""
    pass


class GPUError(ResourceError):
    """Raised when there is a problem with GPU operations."""
    
    def __init__(self, message="GPU operation failed", *args, **kwargs):
        super().__init__(message, *args, **kwargs)


class MemoryError(ResourceError):
    """Raised when there is insufficient memory."""
    
    def __init__(self, message="Insufficient memory", *args, **kwargs):
        super().__init__(message, *args, **kwargs)


class ProcessingError(HeightcraftError):
    """Base class for processing-related errors."""
    pass


class ModelLoadError(ProcessingError):
    """Raised when there is a problem loading a 3D model."""
    
    def __init__(self, message="Failed to load 3D model", *args, **kwargs):
        super().__init__(message, *args, **kwargs)


class MeshValidationError(ProcessingError):
    """Raised when a mesh fails validation."""
    
    def __init__(self, message="Mesh validation failed", *args, **kwargs):
        super().__init__(message, *args, **kwargs)


class SamplingError(ProcessingError):
    """Raised when point sampling fails."""
    
    def __init__(self, message="Point sampling failed", *args, **kwargs):
        super().__init__(message, *args, **kwargs)


class HeightMapGenerationError(ProcessingError):
    """Raised when height map generation fails."""
    
    def __init__(self, message="Height map generation failed", *args, **kwargs):
        super().__init__(message, *args, **kwargs)


class UpscalingError(ProcessingError):
    """Raised when upscaling fails."""
    
    def __init__(self, message="Upscaling failed", *args, **kwargs):
        super().__init__(message, *args, **kwargs)


class FileError(HeightcraftError):
    """Raised when there is a problem with file operations."""
    
    def __init__(self, message="File operation failed", *args, **kwargs):
        super().__init__(message, *args, **kwargs)


class CacheError(HeightcraftError):
    """Raised when there is a problem with the cache."""
    
    def __init__(self, message="Cache operation failed", *args, **kwargs):
        super().__init__(message, *args, **kwargs)


class ThreadingError(HeightcraftError):
    """Raised when there is a problem with threading."""
    
    def __init__(self, message="Threading operation failed", *args, **kwargs):
        super().__init__(message, *args, **kwargs)


class ValidationError(HeightcraftError):
    """Raised when input validation fails."""
    
    def __init__(self, message="Input validation failed", *args, **kwargs):
        super().__init__(message, *args, **kwargs)


# Domain model exceptions
class DomainError(HeightcraftError):
    """Base class for domain model related errors."""
    pass


class MeshError(DomainError):
    """Raised when there is a problem with a mesh."""
    
    def __init__(self, message="Mesh operation failed", *args, **kwargs):
        super().__init__(message, *args, **kwargs)


class HeightMapError(DomainError):
    """Raised when there is a problem with a height map."""
    
    def __init__(self, message="Height map operation failed", *args, **kwargs):
        super().__init__(message, *args, **kwargs)


class HeightMapValidationError(HeightMapError):
    """Raised when a height map fails validation."""
    
    def __init__(self, message="Height map validation failed", *args, **kwargs):
        super().__init__(message, *args, **kwargs)


class PointCloudError(DomainError):
    """Raised when there is a problem with a point cloud."""
    
    def __init__(self, message="Point cloud operation failed", *args, **kwargs):
        super().__init__(message, *args, **kwargs)


# Service layer exceptions
class ServiceError(HeightcraftError):
    """Base class for service layer related errors."""
    pass


class MeshServiceError(ServiceError):
    """Raised when there is a problem with the mesh service."""
    
    def __init__(self, message="Mesh service operation failed", *args, **kwargs):
        super().__init__(message, *args, **kwargs)


class HeightMapServiceError(ServiceError):
    """Raised when there is a problem with the height map service."""
    
    def __init__(self, message="Height map service operation failed", *args, **kwargs):
        super().__init__(message, *args, **kwargs)


class PointCloudServiceError(ServiceError):
    """Raised when there is a problem with the point cloud service."""
    
    def __init__(self, message="Point cloud service operation failed", *args, **kwargs):
        super().__init__(message, *args, **kwargs)


# Infrastructure layer exceptions
class InfrastructureError(HeightcraftError):
    """Base class for infrastructure related errors."""
    pass


class RepositoryError(InfrastructureError):
    """Raised when there is a problem with a repository."""
    
    def __init__(self, message="Repository operation failed", *args, **kwargs):
        super().__init__(message, *args, **kwargs)


# Utility exceptions
class UtilityError(HeightcraftError):
    """Base class for utility related errors."""
    pass


class ConversionError(UtilityError):
    """Raised when there is a problem with data conversion."""
    
    def __init__(self, message="Data conversion failed", *args, **kwargs):
        super().__init__(message, *args, **kwargs)


class CalculationError(UtilityError):
    """Raised when there is a problem with a calculation."""
    
    def __init__(self, message="Calculation failed", *args, **kwargs):
        super().__init__(message, *args, **kwargs) 