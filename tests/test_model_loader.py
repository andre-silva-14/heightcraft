import pytest
import numpy as np
import trimesh
from unittest.mock import patch, MagicMock
from lib.model_loader import ModelLoader

@pytest.fixture
def mock_trimesh_load():
    with patch('trimesh.load_mesh') as mock_load:
        mock_mesh = MagicMock(spec=trimesh.Trimesh)
        mock_mesh.vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        mock_mesh.faces = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]])
        mock_load.return_value = mock_mesh
        yield mock_load

def test_load_valid_file(mock_trimesh_load):
    file_path = 'test_model.stl'
    mesh = ModelLoader.load(file_path)
    mock_trimesh_load.assert_called_once_with(file_path)
    assert isinstance(mesh, trimesh.Trimesh)

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
    assert np.allclose(mesh.centroid, [0, 0, 0])

def test_align_model():
    mesh = trimesh.Trimesh(vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]))
    ModelLoader._align_model(mesh)
    # Check if the largest inertia axis is aligned with Z-axis
    inertia_tensor = mesh.moment_inertia
    assert np.argmax(np.diag(inertia_tensor)) == 2  # Z-axis should have the largest moment of inertia
