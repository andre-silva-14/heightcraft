import os

import numpy as np
import pytest
import trimesh

from lib.large_model_handler import LargeModelHandler


@pytest.fixture
def sample_mesh():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    faces = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]])
    return trimesh.Trimesh(vertices=vertices, faces=faces)


def test_load_model_info(tmp_path, sample_mesh):
    # Save sample mesh to a temporary file
    mesh_path = tmp_path / "test_mesh.stl"
    sample_mesh.export(mesh_path)

    # When exporting to STL, trimesh duplicates vertices for each face
    # So we can't directly compare vertex counts
    with LargeModelHandler(str(mesh_path), chunk_size=1000) as handler:
        handler.load_model_info()
        # For STL files, each face gets its own vertices (no sharing)
        # So for 4 faces, we expect 12 vertices (3 per face)
        assert handler.total_vertices == 12
        assert handler.total_faces == 4
        assert not handler.is_scene


def test_calculate_bounding_box(tmp_path, sample_mesh):
    mesh_path = tmp_path / "test_mesh.stl"
    sample_mesh.export(mesh_path)

    with LargeModelHandler(str(mesh_path), chunk_size=1000) as handler:
        handler.load_model_info()
        min_coords, max_coords = handler.calculate_bounding_box()
        assert np.allclose(min_coords, [0, 0, 0])
        assert np.allclose(max_coords, [1, 1, 1])


def test_sample_points(tmp_path, sample_mesh):
    mesh_path = tmp_path / "test_mesh.stl"
    sample_mesh.export(mesh_path)

    with LargeModelHandler(str(mesh_path), chunk_size=1000) as handler:
        handler.load_model_info()

        # Mock the _sample_points_cpu method to return a simple array
        def mock_sample(num_samples):
            return np.random.random((num_samples, 3))

        handler._sample_points_cpu = mock_sample
        points = handler.sample_points(num_samples=1000, use_gpu=False)
        assert len(points) == 1000
        assert points.shape[1] == 3


def test_stream_vertices(tmp_path, sample_mesh):
    mesh_path = tmp_path / "test_mesh.stl"
    sample_mesh.export(mesh_path)

    with LargeModelHandler(str(mesh_path), chunk_size=1000) as handler:
        handler.load_model_info()
        vertices = list(handler.stream_vertices())
        assert len(vertices) > 0
        # Combine all chunks
        all_vertices = np.vstack(vertices)
        # For STL files, we expect 12 vertices (3 per face, 4 faces)
        assert all_vertices.shape == (12, 3)
        # Check that vertices are within the bounding box of the original mesh
        min_coords = np.min(sample_mesh.vertices, axis=0)
        max_coords = np.max(sample_mesh.vertices, axis=0)
        assert np.all(all_vertices >= min_coords)
        assert np.all(all_vertices <= max_coords)


def test_stream_faces(tmp_path, sample_mesh):
    mesh_path = tmp_path / "test_mesh.stl"
    sample_mesh.export(mesh_path)

    # Load the exported mesh to compare correctly
    exported_mesh = trimesh.load(str(mesh_path))

    with LargeModelHandler(str(mesh_path), chunk_size=1000) as handler:
        handler.load_model_info()
        faces = list(handler.stream_faces())
        assert len(faces) > 0
        # For faces, we need to check that they reference valid vertices
        all_faces = np.vstack(faces)
        assert all_faces.shape[1] == 3
        assert np.max(all_faces) < handler.total_vertices


def test_cleanup(tmp_path, sample_mesh):
    mesh_path = tmp_path / "test_mesh.stl"
    sample_mesh.export(mesh_path)

    handler = LargeModelHandler(str(mesh_path), chunk_size=1000)
    handler.load_model_info()
    handler.cleanup()

    # Verify resources are cleaned up
    assert handler.scene is None
    assert handler.concatenated_mesh is None
    assert not handler._vertex_cache
    assert not handler._face_cache


def test_context_manager(tmp_path, sample_mesh):
    mesh_path = tmp_path / "test_mesh.stl"
    sample_mesh.export(mesh_path)

    with LargeModelHandler(str(mesh_path), chunk_size=1000) as handler:
        handler.load_model_info()

        # Mock the _sample_points_cpu method to return a simple array
        def mock_sample(num_samples):
            return np.random.random((num_samples, 3))

        handler._sample_points_cpu = mock_sample
        points = handler.sample_points(num_samples=1000, use_gpu=False)
        assert len(points) == 1000

    # Verify cleanup after context manager
    assert handler.scene is None
    assert handler.concatenated_mesh is None
    assert not handler._vertex_cache
    assert not handler._face_cache


def test_invalid_file():
    with pytest.raises(Exception):
        with LargeModelHandler("nonexistent.stl", chunk_size=1000) as handler:
            handler.load_model_info()


def test_memory_limits():
    with pytest.raises(MemoryError):
        LargeModelHandler("test.stl", chunk_size=1000000000, max_memory=0.1)
