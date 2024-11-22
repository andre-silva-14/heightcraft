import pytest
import numpy as np
import trimesh
from unittest.mock import patch, MagicMock
from lib.model_loader import ModelLoader

@pytest.fixture
def mock_trimesh_load():
    mock_mesh = MagicMock(spec=trimesh.Trimesh)
    mock_mesh.vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    mock_mesh.faces = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]])
    mock_mesh.centroid = np.array([0.25, 0.25, 0.25])
    mock_mesh.moment_inertia = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
    
    with patch('trimesh.load_mesh', return_value=mock_mesh) as mock_load:
        yield mock_load

def test_load_valid_file(mock_trimesh_load):
    file_path = 'test_model.stl'
    mesh = ModelLoader.load(file_path)
    mock_trimesh_load.assert_called_once_with(file_path)
    assert isinstance(mesh, MagicMock)
    assert mesh.apply_translation.called
    assert mesh.apply_transform.called

def test_load_unsupported_format():
    with pytest.raises(ValueError, match="Unsupported file format"):
        ModelLoader.load('test_model.unsupported')

def test_load_file_error(mock_trimesh_load):
    mock_trimesh_load.side_effect = Exception("File not found")
    with pytest.raises(ValueError, match="Failed to load the 3D model"):
        ModelLoader.load('nonexistent_file.stl')

def test_center_model():
    mesh = trimesh.Trimesh(vertices=np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]]))
    ModelLoader._center_model(mesh)
    assert np.allclose(mesh.centroid, [0, 0, 0], atol=1e-6)

def test_align_model():
    mesh = trimesh.Trimesh(vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]))
    mesh.moment_inertia = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
    ModelLoader._align_model(mesh)
    # Check if the largest inertia axis is aligned with Z-axis
    assert mesh.apply_transform.called
