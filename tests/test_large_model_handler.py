import torch
import pytest
import numpy as np
import trimesh
from unittest.mock import patch, MagicMock
from lib.large_model_handler import LargeModelHandler

@pytest.fixture
def mock_trimesh_load():
    with patch('trimesh.load') as mock_load:
        mock_mesh = MagicMock(spec=trimesh.Trimesh)
        mock_mesh.vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        mock_mesh.faces = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]])
        mock_load.return_value = mock_mesh
        yield mock_load

def test_load_model_info(mock_trimesh_load):
    handler = LargeModelHandler('test_model.stl')
    handler.load_model_info()
    assert handler.total_vertices == 4
    assert handler.total_faces == 4

def test_stream_vertices(mock_trimesh_load):
    handler = LargeModelHandler('test_model.stl', chunk_size=2)
    handler.load_model_info()
    vertices = list(handler.stream_vertices())
    assert len(vertices) == 2
    assert np.array_equal(vertices[0], np.array([[0, 0, 0], [1, 0, 0]]))
    assert np.array_equal(vertices[1], np.array([[0, 1, 0], [0, 0, 1]]))

def test_stream_faces(mock_trimesh_load):
    handler = LargeModelHandler('test_model.stl', chunk_size=2)
    handler.load_model_info()
    faces = list(handler.stream_faces())
    assert len(faces) == 2
    assert np.array_equal(faces[0], np.array([[0, 1, 2], [0, 2, 3]]))
    assert np.array_equal(faces[1], np.array([[0, 3, 1], [1, 3, 2]]))

def test_calculate_bounding_box(mock_trimesh_load):
    handler = LargeModelHandler('test_model.stl')
    handler.load_model_info()
    min_coords, max_coords = handler.calculate_bounding_box()
    assert np.array_equal(min_coords, [0, 0, 0])
    assert np.array_equal(max_coords, [1, 1, 1])

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_sample_points_gpu(mock_trimesh_load):
    handler = LargeModelHandler('test_model.stl')
    handler.load_model_info()
    num_samples = 1000
    points = handler.sample_points_gpu(num_samples)
    assert points.shape == (num_samples, 3)
    assert np.all(points >= 0) and np.all(points <= 1)

def test_sample_points_cpu(mock_trimesh_load):
    handler = LargeModelHandler('test_model.stl')
    handler.load_model_info()
    num_samples = 1000
    points = handler.sample_points_cpu(num_samples)
    assert points.shape == (num_samples, 3)
    assert np.all(points >= 0) and np.all(points <= 1)
