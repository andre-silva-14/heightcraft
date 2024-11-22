import pytest
import numpy as np
import trimesh
import os
from lib.height_map_generator import HeightMapGenerator

@pytest.fixture
def sample_mesh():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    faces = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]])
    return trimesh.Trimesh(vertices=vertices, faces=faces)

def test_generate_height_map(sample_mesh):
    target_resolution = (100, 100)
    height_map = HeightMapGenerator.generate(sample_mesh, target_resolution, use_gpu=False, num_samples=10000, num_threads=4, bit_depth=16)
    assert height_map.shape == target_resolution
    assert height_map.dtype == np.uint16
    assert np.min(height_map) >= 0
    assert np.max(height_map) <= 65535

def test_generate_height_map_8bit(sample_mesh):
    target_resolution = (100, 100)
    height_map = HeightMapGenerator.generate(sample_mesh, target_resolution, use_gpu=False, num_samples=10000, num_threads=4, bit_depth=8)
    assert height_map.shape == target_resolution
    assert height_map.dtype == np.uint8
    assert np.min(height_map) >= 0
    assert np.max(height_map) <= 255

def test_generate_from_points():
    points = np.array([[0, 0, 0], [1, 0, 0.5], [0, 1, 1], [1, 1, 0.75]])
    target_resolution = (10, 10)
    height_map = HeightMapGenerator.generate_from_points(points, target_resolution, bit_depth=16)
    assert height_map.shape == target_resolution
    assert height_map.dtype == np.uint16
    assert np.min(height_map) >= 0
    assert np.max(height_map) <= 65535

def test_generate_invalid_bit_depth(sample_mesh):
    target_resolution = (100, 100)
    with pytest.raises(ValueError, match="Bit depth must be either 8 or 16"):
        HeightMapGenerator.generate(sample_mesh, target_resolution, use_gpu=False, num_samples=10000, num_threads=4, bit_depth=32)

def test_save_height_map(tmp_path):
    height_map = np.random.randint(0, 65536, size=(100, 100), dtype=np.uint16)
    output_path = tmp_path / "test_height_map.png"
    HeightMapGenerator.save_height_map(height_map, str(output_path))
    assert output_path.exists()

def test_save_split_height_maps(tmp_path):
    height_map = np.random.randint(0, 65536, size=(100, 100), dtype=np.uint16)
    output_path = tmp_path / "test_height_map.png"
    HeightMapGenerator.save_height_map(height_map, str(output_path), split=4)
    
    assert (tmp_path / "test_height_map_part_0_0.png").exists()
    assert (tmp_path / "test_height_map_part_0_1.png").exists()
    assert (tmp_path / "test_height_map_part_1_0.png").exists()
    assert (tmp_path / "test_height_map_part_1_1.png").exists()

def test_get_optimal_grid():
    assert HeightMapGenerator._get_optimal_grid(4, 100, 100) == (2, 2)
    assert HeightMapGenerator._get_optimal_grid(9, 100, 100) == (3, 3)
    assert HeightMapGenerator._get_optimal_grid(8, 100, 200) == (2, 4)
    assert HeightMapGenerator._get_optimal_grid(8, 200, 100) == (4, 2)
    assert HeightMapGenerator._get_optimal_grid(6, 100, 200) == (2, 3)
    assert HeightMapGenerator._get_optimal_grid(6, 200, 100) == (3, 2)
