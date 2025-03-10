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


@pytest.fixture
def real_mesh_x_smallest():
    """Create a real mesh with X as the smallest dimension."""
    # X=0.2, Y=2, Z=1
    vertices = np.array([
        [0, 0, 0], [0.2, 0, 0], [0.2, 2, 0], [0, 2, 0],
        [0, 0, 1], [0.2, 0, 1], [0.2, 2, 1], [0, 2, 1],
    ])
    faces = np.array([
        [0, 1, 2], [0, 2, 3],
        [4, 5, 6], [4, 6, 7],
        [0, 4, 5], [0, 5, 1],
        [3, 2, 6], [3, 6, 7],
        [0, 3, 7], [0, 7, 4],
        [1, 5, 6], [1, 6, 2],
    ])
    return trimesh.Trimesh(vertices=vertices, faces=faces)


@pytest.fixture
def real_mesh_y_smallest():
    """Create a real mesh with Y as the smallest dimension."""
    # X=2, Y=0.2, Z=1
    vertices = np.array([
        [0, 0, 0], [2, 0, 0], [2, 0.2, 0], [0, 0.2, 0],
        [0, 0, 1], [2, 0, 1], [2, 0.2, 1], [0, 0.2, 1],
    ])
    faces = np.array([
        [0, 1, 2], [0, 2, 3],
        [4, 5, 6], [4, 6, 7],
        [0, 4, 5], [0, 5, 1],
        [3, 2, 6], [3, 6, 7],
        [0, 3, 7], [0, 7, 4],
        [1, 5, 6], [1, 6, 2],
    ])
    return trimesh.Trimesh(vertices=vertices, faces=faces)


@pytest.fixture
def real_mesh_z_smallest():
    """Create a real mesh with Z as the smallest dimension."""
    # X=2, Y=1, Z=0.2
    vertices = np.array([
        [0, 0, 0], [2, 0, 0], [2, 1, 0], [0, 1, 0],
        [0, 0, 0.2], [2, 0, 0.2], [2, 1, 0.2], [0, 1, 0.2],
    ])
    faces = np.array([
        [0, 1, 2], [0, 2, 3],
        [4, 5, 6], [4, 6, 7],
        [0, 4, 5], [0, 5, 1],
        [3, 2, 6], [3, 6, 7],
        [0, 3, 7], [0, 7, 4],
        [1, 5, 6], [1, 6, 2],
    ])
    return trimesh.Trimesh(vertices=vertices, faces=faces)


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


def test_align_model_x_smallest(real_mesh_x_smallest):
    """Test alignment when X is the smallest dimension."""
    # Get initial extents
    initial_extents = real_mesh_x_smallest.bounding_box.extents
    assert np.argmin(initial_extents) == 0  # X is smallest
    
    # Apply alignment
    ModelLoader._align_model(real_mesh_x_smallest)
    
    # After alignment, Z should be the smallest dimension
    aligned_extents = real_mesh_x_smallest.bounding_box.extents
    assert np.argmin(aligned_extents) == 2  # Z should now be smallest
    assert np.isclose(aligned_extents[2], 0.2, atol=1e-6)  # Z should have the original X value


def test_align_model_y_smallest(real_mesh_y_smallest):
    """Test alignment when Y is the smallest dimension."""
    # Get initial extents
    initial_extents = real_mesh_y_smallest.bounding_box.extents
    assert np.argmin(initial_extents) == 1  # Y is smallest
    
    # Apply alignment
    ModelLoader._align_model(real_mesh_y_smallest)
    
    # After alignment, Z should be the smallest dimension
    aligned_extents = real_mesh_y_smallest.bounding_box.extents
    assert np.argmin(aligned_extents) == 2  # Z should now be smallest
    assert np.isclose(aligned_extents[2], 0.2, atol=1e-6)  # Z should have the original Y value


def test_align_model_z_already_smallest(real_mesh_z_smallest):
    """Test alignment when Z is already the smallest dimension."""
    # Get initial extents
    initial_extents = real_mesh_z_smallest.bounding_box.extents
    assert np.argmin(initial_extents) == 2  # Z is smallest
    
    # Apply alignment
    ModelLoader._align_model(real_mesh_z_smallest)
    
    # After alignment, Z should still be the smallest dimension (no change)
    aligned_extents = real_mesh_z_smallest.bounding_box.extents
    assert np.argmin(aligned_extents) == 2  # Z remains smallest
    assert np.isclose(aligned_extents[2], 0.2, atol=1e-6)  # Z value unchanged


def test_align_model_preserves_volume():
    """Test that alignment preserves the volume of the mesh."""
    # Create a mesh with different dimensions
    vertices = np.array([
        [0, 0, 0], [1, 0, 0], [1, 2, 0], [0, 2, 0],
        [0, 0, 0.5], [1, 0, 0.5], [1, 2, 0.5], [0, 2, 0.5],
    ])
    faces = np.array([
        [0, 1, 2], [0, 2, 3],
        [4, 5, 6], [4, 6, 7],
        [0, 4, 5], [0, 5, 1],
        [3, 2, 6], [3, 6, 7],
        [0, 3, 7], [0, 7, 4],
        [1, 5, 6], [1, 6, 2],
    ])
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    # Get initial volume
    initial_volume = mesh.volume
    
    # Apply alignment
    ModelLoader._align_model(mesh)
    
    # Check volume after alignment
    aligned_volume = mesh.volume
    
    # Volumes should be very close (allowing for floating point precision)
    assert np.isclose(initial_volume, aligned_volume, rtol=1e-5)


def test_align_model_with_empty_mesh():
    """Test alignment with an empty mesh."""
    mesh = trimesh.Trimesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3)))
    
    # This should not raise an error
    ModelLoader._align_model(mesh)


def test_center_model_with_empty_mesh():
    """Test centering with an empty mesh."""
    mesh = trimesh.Trimesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3)))
    
    # This should not raise an error
    ModelLoader._center_model(mesh)


def test_process_scene():
    """Test the _process_scene static method."""
    # Create a simple scene with a single mesh
    mesh = trimesh.Trimesh(
        vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
        faces=np.array([[0, 1, 2]])
    )
    
    scene = trimesh.Scene()
    scene.add_geometry(mesh)
    
    # Process the scene
    processed_mesh = ModelLoader._process_scene(scene)
    
    # Check that the result is a trimesh
    assert isinstance(processed_mesh, trimesh.Trimesh)
    
    # Check that it has the expected number of vertices and faces
    assert len(processed_mesh.vertices) == 3
    assert len(processed_mesh.faces) == 1


def test_process_scene_with_transformations():
    """Test _process_scene with transformations in the scene graph."""
    # Create two meshes with different transformations
    mesh1 = trimesh.Trimesh(
        vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
        faces=np.array([[0, 1, 2]])
    )

    mesh2 = trimesh.Trimesh(
        vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
        faces=np.array([[0, 1, 2]])
    )

    scene = trimesh.Scene()

    # Add first mesh at origin
    scene.add_geometry(mesh1)

    # Add second mesh translated by [2, 0, 0]
    transform = np.eye(4)
    transform[:3, 3] = [2, 0, 0]
    scene.add_geometry(mesh2, transform=transform)

    # Process the scene
    processed_mesh = ModelLoader._process_scene(scene)

    # Check that the result has combined vertices (should have 6 vertices)
    assert len(processed_mesh.vertices) == 6
    assert len(processed_mesh.faces) == 2

    # Note: The processed mesh might be centered and aligned,
    # so we can't directly check for the exact vertices.
    # Instead, check that the mesh contains points that are 
    # appropriately spaced relative to each other.
    
    # Get the unique x-coordinates
    x_coords = np.unique(processed_mesh.vertices[:, 0])
    assert len(x_coords) >= 4  # Should have at least 4 unique x-coordinates
    
    # The mesh should have a reasonable bounding box
    bounds = processed_mesh.bounds
    assert bounds[1][0] - bounds[0][0] > 0  # X dimension
    assert bounds[1][1] - bounds[0][1] > 0  # Y dimension


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

    # Center and align the mock meshes
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

        # Verify Z is the smallest dimension
        extents = result.bounding_box.extents
        assert np.argmin(extents) == 2


def test_load_real_scene_with_alignment():
    """Test loading and aligning a real scene with different orientations."""
    # Create fixtures for different orientations
    # We'll create these directly rather than using fixtures to avoid direct fixture calls
    
    # X-dominant mesh
    x_mesh = trimesh.Trimesh(
        vertices=np.array([
            [0, 0, 0], [4, 0, 0], [4, 1, 0], [0, 1, 0],
            [0, 0, 2], [4, 0, 2], [4, 1, 2], [0, 1, 2],
        ]),
        faces=np.array([
            [0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7],
            [0, 4, 5], [0, 5, 1], [3, 2, 6], [3, 6, 7],
            [0, 3, 7], [0, 7, 4], [1, 5, 6], [1, 6, 2],
        ])
    )
    
    # Y-dominant mesh
    y_mesh = trimesh.Trimesh(
        vertices=np.array([
            [0, 0, 0], [1, 0, 0], [1, 4, 0], [0, 4, 0],
            [0, 0, 2], [1, 0, 2], [1, 4, 2], [0, 4, 2],
        ]),
        faces=np.array([
            [0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7],
            [0, 4, 5], [0, 5, 1], [3, 2, 6], [3, 6, 7],
            [0, 3, 7], [0, 7, 4], [1, 5, 6], [1, 6, 2],
        ])
    )
    
    # Z-dominant mesh
    z_mesh = trimesh.Trimesh(
        vertices=np.array([
            [0, 0, 0], [1, 0, 0], [1, 2, 0], [0, 2, 0],
            [0, 0, 4], [1, 0, 4], [1, 2, 4], [0, 2, 4],
        ]),
        faces=np.array([
            [0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7],
            [0, 4, 5], [0, 5, 1], [3, 2, 6], [3, 6, 7],
            [0, 3, 7], [0, 7, 4], [1, 5, 6], [1, 6, 2],
        ])
    )
    
    # Test each mesh
    for mesh, name, original_dominant_dim in [
        (x_mesh, "x-dominant", 0),
        (y_mesh, "y-dominant", 1),
        (z_mesh, "z-dominant", 2),
    ]:
        # Center and align the mesh
        aligned_mesh = mesh.copy()
        ModelLoader._center_model(aligned_mesh)
        ModelLoader._align_model(aligned_mesh)
        
        # After alignment, Z should be the smallest dimension
        extents = aligned_mesh.bounding_box.extents
        smallest_dim = np.argmin(extents)
        assert smallest_dim == 2, f"Failed for {name}, smallest dimension is {smallest_dim}"
        
        # Check that the mesh is centered
        centroid = np.mean(aligned_mesh.vertices, axis=0)
        assert np.allclose(centroid, [0, 0, 0], atol=1e-6)


def test_load_scene_with_empty_meshes():
    """Test loading a scene with empty meshes."""
    # Create a valid mesh and an empty mesh
    valid_mesh = trimesh.Trimesh(
        vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
        faces=np.array([[0, 1, 2]])
    )
    
    empty_mesh = trimesh.Trimesh(
        vertices=np.zeros((0, 3)),
        faces=np.zeros((0, 3))
    )
    
    # Create scene
    scene = trimesh.Scene()
    scene.add_geometry(valid_mesh)
    scene.add_geometry(empty_mesh)
    
    with patch("trimesh.load_mesh", return_value=scene):
        # This should not raise an error
        result = ModelLoader.load("test_scene_with_empty.gltf")
        
        # The result should only contain the valid mesh
        assert len(result.vertices) == 3
        assert len(result.faces) == 1
