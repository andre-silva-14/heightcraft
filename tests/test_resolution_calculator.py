import pytest
import numpy as np
import trimesh
from lib.resolution_calculator import ResolutionCalculator

@pytest.fixture
def sample_mesh():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 1]])
    faces = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]])
    return trimesh.Trimesh(vertices=vertices, faces=faces)

@pytest.fixture
def non_uniform_x_dominant():
    """Create a mesh with X as the dominant dimension."""
    # X=4, Y=1, Z=2
    vertices = np.array([
        [0, 0, 0], [4, 0, 0], [4, 1, 0], [0, 1, 0],  # Bottom face
        [0, 0, 2], [4, 0, 2], [4, 1, 2], [0, 1, 2],  # Top face
    ])
    faces = np.array([
        [0, 1, 2], [0, 2, 3],  # Bottom face
        [4, 5, 6], [4, 6, 7],  # Top face
        [0, 4, 5], [0, 5, 1],  # Front face
        [3, 2, 6], [3, 6, 7],  # Back face
        [0, 3, 7], [0, 7, 4],  # Left face
        [1, 5, 6], [1, 6, 2],  # Right face
    ])
    return trimesh.Trimesh(vertices=vertices, faces=faces)

@pytest.fixture
def non_uniform_y_dominant():
    """Create a mesh with Y as the dominant dimension."""
    # X=1, Y=4, Z=2
    vertices = np.array([
        [0, 0, 0], [1, 0, 0], [1, 4, 0], [0, 4, 0],  # Bottom face
        [0, 0, 2], [1, 0, 2], [1, 4, 2], [0, 4, 2],  # Top face
    ])
    faces = np.array([
        [0, 1, 2], [0, 2, 3],  # Bottom face
        [4, 5, 6], [4, 6, 7],  # Top face
        [0, 4, 5], [0, 5, 1],  # Front face
        [3, 2, 6], [3, 6, 7],  # Back face
        [0, 3, 7], [0, 7, 4],  # Left face
        [1, 5, 6], [1, 6, 2],  # Right face
    ])
    return trimesh.Trimesh(vertices=vertices, faces=faces)

@pytest.fixture
def non_uniform_z_dominant():
    """Create a mesh with Z as the dominant dimension."""
    # X=1, Y=2, Z=4
    vertices = np.array([
        [0, 0, 0], [1, 0, 0], [1, 2, 0], [0, 2, 0],  # Bottom face
        [0, 0, 4], [1, 0, 4], [1, 2, 4], [0, 2, 4],  # Top face
    ])
    faces = np.array([
        [0, 1, 2], [0, 2, 3],  # Bottom face
        [4, 5, 6], [4, 6, 7],  # Top face
        [0, 4, 5], [0, 5, 1],  # Front face
        [3, 2, 6], [3, 6, 7],  # Back face
        [0, 3, 7], [0, 7, 4],  # Left face
        [1, 5, 6], [1, 6, 2],  # Right face
    ])
    return trimesh.Trimesh(vertices=vertices, faces=faces)

def test_calculate_resolution(sample_mesh):
    max_resolution = 100
    width, height = ResolutionCalculator.calculate(sample_mesh, max_resolution)
    assert width == 50
    assert height == 100

def test_calculate_resolution_square_model():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    max_resolution = 100
    width, height = ResolutionCalculator.calculate(mesh, max_resolution)
    assert width == 100
    assert height == 100

def test_calculate_resolution_zero_dimensions():
    vertices = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    faces = np.array([[0, 1, 2]])
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    max_resolution = 100
    with pytest.raises(ValueError, match="Model bounding box has zero or negative width or height"):
        ResolutionCalculator.calculate(mesh, max_resolution)

def test_calculate_resolution_x_dominant(non_uniform_x_dominant):
    """Test resolution calculation with X as dominant dimension."""
    max_resolution = 1000
    width, height = ResolutionCalculator.calculate(non_uniform_x_dominant, max_resolution)
    
    # X=4, Y=1, so aspect ratio is 4:1
    # Expected width=1000, height=250
    assert width == 1000
    assert height == 250
    
    # Verify aspect ratio is maintained
    assert abs((width / height) - 4.0) < 0.01

def test_calculate_resolution_y_dominant(non_uniform_y_dominant):
    """Test resolution calculation with Y as dominant dimension."""
    max_resolution = 1000
    width, height = ResolutionCalculator.calculate(non_uniform_y_dominant, max_resolution)
    
    # X=1, Y=4, so aspect ratio is 1:4
    # Expected width=250, height=1000
    assert width == 250
    assert height == 1000
    
    # Verify aspect ratio is maintained
    assert abs((height / width) - 4.0) < 0.01

def test_calculate_resolution_z_ignored(non_uniform_z_dominant):
    """Test that Z dimension is ignored in resolution calculation."""
    max_resolution = 1000
    width, height = ResolutionCalculator.calculate(non_uniform_z_dominant, max_resolution)
    
    # X=1, Y=2, Z=4, but Z should be ignored
    # So aspect ratio is 1:2
    # Expected width=500, height=1000
    assert width == 500
    assert height == 1000
    
    # Verify aspect ratio is maintained
    assert abs((height / width) - 2.0) < 0.01

def test_calculate_resolution_with_centered_model():
    """Test resolution calculation with a centered model."""
    # Create a model centered at origin
    vertices = np.array([
        [-2, -1, 0], [2, -1, 0], [2, 1, 0], [-2, 1, 0]
    ])
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    max_resolution = 1000
    width, height = ResolutionCalculator.calculate(mesh, max_resolution)
    
    # Width is 4, height is 2, aspect ratio is 2:1
    assert width == 1000
    assert height == 500
    
    # Verify aspect ratio is maintained
    assert abs((width / height) - 2.0) < 0.01

def test_calculate_from_bounds():
    min_coords = np.array([0, 0, 0])
    max_coords = np.array([2, 1, 1])
    max_resolution = 100
    width, height = ResolutionCalculator.calculate_from_bounds(min_coords, max_coords, max_resolution)
    assert width == 100
    assert height == 50

def test_calculate_from_bounds_zero_dimensions():
    min_coords = np.array([0, 0, 0])
    max_coords = np.array([0, 1, 1])
    max_resolution = 100
    with pytest.raises(ValueError, match="Model bounding box has zero or negative width or height"):
        ResolutionCalculator.calculate_from_bounds(min_coords, max_coords, max_resolution)

def test_calculate_from_bounds_with_negative_dimensions():
    """Test calculation with negative dimensions (should raise an error)."""
    min_coords = np.array([1, 0, 0])  # Min > Max for X
    max_coords = np.array([0, 1, 1])
    max_resolution = 100
    with pytest.raises(ValueError, match="Model bounding box has zero or negative width or height"):
        ResolutionCalculator.calculate_from_bounds(min_coords, max_coords, max_resolution)

def test_calculate_from_bounds_ignores_z():
    """Test that Z dimension is ignored in bounds-based resolution calculation."""
    # Different Z dimensions shouldn't affect resolution
    min_coords1 = np.array([0, 0, 0])
    max_coords1 = np.array([2, 1, 1])
    
    min_coords2 = np.array([0, 0, 0])
    max_coords2 = np.array([2, 1, 10])  # Much larger Z
    
    max_resolution = 100
    
    width1, height1 = ResolutionCalculator.calculate_from_bounds(min_coords1, max_coords1, max_resolution)
    width2, height2 = ResolutionCalculator.calculate_from_bounds(min_coords2, max_coords2, max_resolution)
    
    # Results should be identical regardless of Z difference
    assert width1 == width2
    assert height1 == height2

def test_calculate_from_bounds_extreme_aspect_ratio():
    """Test resolution calculation with extreme aspect ratios."""
    # Very wide, short model (100:1 aspect ratio)
    min_coords = np.array([0, 0, 0])
    max_coords = np.array([100, 1, 10])
    max_resolution = 1000
    
    width, height = ResolutionCalculator.calculate_from_bounds(min_coords, max_coords, max_resolution)
    
    # Expected width=1000, height=10
    assert width == 1000
    assert height == 10
    
    # Verify aspect ratio is maintained
    assert abs((width / height) - 100.0) < 0.1

def test_calculate_with_aligned_model(non_uniform_y_dominant):
    """Test resolution calculation after model alignment."""
    # Initialize with Y as dominant dimension
    # X=1, Y=4, Z=2
    mesh = non_uniform_y_dominant.copy()
    
    # Apply a rotation to "align" the model
    # This simulates what happens in LargeModelHandler._align_model
    # Swap Y and Z axes
    rotation = np.eye(4)
    rotation[1, 1] = 0
    rotation[2, 2] = 0
    rotation[1, 2] = 1
    rotation[2, 1] = 1
    mesh.apply_transform(rotation)
    
    # After rotation, dimensions should be X=1, Y=2, Z=4
    max_resolution = 1000
    width, height = ResolutionCalculator.calculate(mesh, max_resolution)
    
    # Since Z is now 4 (was Y), it should be ignored
    # X=1, Y=2, aspect ratio 1:2
    # Expected width=500, height=1000
    assert width == 500
    assert height == 1000
    
    # Verify aspect ratio
    assert abs((height / width) - 2.0) < 0.01
