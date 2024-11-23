from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import trimesh

from lib.model_loader import ModelLoader


@pytest.fixture
def mock_trimesh_load():
    mock_mesh = MagicMock(spec=trimesh.Trimesh)
    mock_mesh.vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    mock_mesh.faces = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]])
    mock_mesh.centroid = np.array([0.25, 0.25, 0.25])
    mock_mesh.moment_inertia = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])

    with patch("trimesh.load_mesh", return_value=mock_mesh) as mock_load:
        yield mock_load


def test_load_valid_file(mock_trimesh_load):
    file_path = "test_model.stl"
    mesh = ModelLoader.load(file_path)
    mock_trimesh_load.assert_called_once_with(file_path)
    assert isinstance(mesh, MagicMock)
    assert mesh.apply_translation.called
    assert mesh.apply_transform.called


def test_load_unsupported_format():
    with pytest.raises(ValueError, match="Unsupported file format"):
        ModelLoader.load("test_model.unsupported")


def test_load_file_error(mock_trimesh_load):
    mock_trimesh_load.side_effect = Exception("File not found")
    with pytest.raises(ValueError, match="Failed to load the 3D model"):
        ModelLoader.load("nonexistent_file.stl")


def test_center_model():
    mesh = MagicMock(spec=trimesh.Trimesh)
    mesh.vertices = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
    mesh.centroid = np.array([2, 2, 2])

    ModelLoader._center_model(mesh)

    mesh.apply_translation.assert_called_once()
    translation_arg = mesh.apply_translation.call_args[0][0]
    assert np.allclose(translation_arg, [-2, -2, -2], atol=1e-6)


def test_align_model():
    mesh = MagicMock(spec=trimesh.Trimesh)
    mesh.vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    mesh.faces = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]])
    mesh.moment_inertia = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])

    ModelLoader._align_model(mesh)

    mesh.apply_transform.assert_called_once()
    transform_arg = mesh.apply_transform.call_args[0][0]
    assert transform_arg.shape == (4, 4)
    assert np.allclose(np.linalg.det(transform_arg[:3, :3]), 1, atol=1e-6)


def test_load_scene_with_multiple_meshes():
    # Create mock meshes
    mock_mesh1 = trimesh.Trimesh(
        vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]], faces=[[0, 1, 2]]
    )
    mock_mesh2 = trimesh.Trimesh(
        vertices=[[0, 0, 1], [1, 0, 1], [0, 1, 1]], faces=[[0, 1, 2]]
    )

    # Create a mock Scene
    mock_scene = trimesh.Scene()
    mock_scene.add_geometry(mock_mesh1)
    mock_scene.add_geometry(mock_mesh2)

    # Certer and align the mock meshes

    ModelLoader._center_model(mock_mesh1)
    ModelLoader._align_model(mock_mesh1)
    ModelLoader._center_model(mock_mesh2)
    ModelLoader._align_model(mock_mesh2)

    with patch("trimesh.load_mesh", return_value=mock_scene):
        result = ModelLoader.load("test_scene.gltf")

        # Assert that the result is a Trimesh object
        assert isinstance(result, trimesh.Trimesh)

        # Assert that the vertices of the resulting mesh are correct
        expected_vertices = np.vstack([mock_mesh1.vertices, mock_mesh2.vertices])
        np.testing.assert_array_almost_equal(result.vertices, expected_vertices)

        # Assert that the faces of the resulting mesh are correct
        expected_faces = np.vstack(
            [mock_mesh1.faces, mock_mesh2.faces + len(mock_mesh1.vertices)]
        )
        np.testing.assert_array_equal(result.faces, expected_faces)

        # Assert that the model was centered
        assert np.allclose(result.centroid, [0, 0, 0], atol=1e-6)

        # Perform an inertia check
        inertia_tensor = result.moment_inertia
        assert inertia_tensor is not None, "Inertia tensor is None"

        # If the inertia tensor is non-zero, we check the alignment
        if np.linalg.norm(inertia_tensor) > 0:
            inertia_matrix = np.array(inertia_tensor)
            # Ensure the principal axes are aligned (identity matrix)
            inertia_matrix = np.dot(inertia_matrix, inertia_matrix.T)
            assert np.allclose(inertia_matrix, np.eye(3), atol=1e-6)
        else:
            # Log a message or skip the alignment check when the inertia tensor is zero
            print("Warning: Inertia tensor is zero, skipping alignment check.")
