import pytest
import numpy as np
import trimesh
from lib.resolution_calculator import ResolutionCalculator

@pytest.fixture
def sample_mesh():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 1]])
    faces = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]])
    return trimesh.Trimesh(vertices=vertices, faces=faces)

def test_calculate_resolution(sample_mesh):
    max_resolution = 100
    width, height = ResolutionCalculator.calculate(sample_mesh, max_resolution)
    assert width == 100
    assert height == 50

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
