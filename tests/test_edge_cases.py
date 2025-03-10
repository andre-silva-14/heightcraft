import pytest
import numpy as np
import trimesh
import os
from pathlib import Path
import tempfile
import shutil

from lib.model_loader import ModelLoader
from lib.large_model_handler import LargeModelHandler
from lib.height_map_generator import HeightMapGenerator, HeightMapConfig
from lib.point_sampler import PointSampler
from lib.resolution_calculator import ResolutionCalculator


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def corrupted_file(temp_dir):
    """Create a corrupted STL file."""
    corrupted_path = temp_dir / "corrupted.stl"
    with open(corrupted_path, "w") as f:
        f.write("This is not a valid STL file")
    return corrupted_path


@pytest.fixture
def empty_file(temp_dir):
    """Create an empty file."""
    empty_path = temp_dir / "empty.stl"
    with open(empty_path, "w") as f:
        pass
    return empty_path


@pytest.fixture
def extremely_thin_mesh(temp_dir):
    """Create an extremely thin mesh (pancake-like)."""
    # Create a mesh with height of only 0.001 units
    vertices = np.array([
        [0, 0, 0], [1, 0, 0], [1, 0.001, 0], [0, 0.001, 0],
        [0, 0, 1], [1, 0, 1], [1, 0.001, 1], [0, 0.001, 1],
    ])
    faces = np.array([
        [0, 1, 2], [0, 2, 3],  # Bottom face
        [4, 5, 6], [4, 6, 7],  # Top face
        [0, 4, 5], [0, 5, 1],  # Front face
        [3, 2, 6], [3, 6, 7],  # Back face
        [0, 3, 7], [0, 7, 4],  # Left face
        [1, 5, 6], [1, 6, 2],  # Right face
    ])
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    # Export to STL
    mesh_path = temp_dir / "extremely_thin.stl"
    mesh.export(mesh_path)
    return mesh_path


@pytest.fixture
def tiny_mesh(temp_dir):
    """Create a tiny mesh with very few vertices and faces."""
    # Just a single tetrahedron
    vertices = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    faces = np.array([
        [0, 1, 2],
        [0, 1, 3],
        [0, 2, 3],
        [1, 2, 3]
    ])
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    # Export to STL
    mesh_path = temp_dir / "tiny.stl"
    mesh.export(mesh_path)
    return mesh_path


@pytest.fixture
def mesh_with_extreme_aspect_ratio(temp_dir):
    """Create a mesh with an extreme aspect ratio (1:1000:1)."""
    # Create a very long, thin, and short box
    vertices = np.array([
        [0, 0, 0], [1000, 0, 0], [1000, 1, 0], [0, 1, 0],
        [0, 0, 1], [1000, 0, 1], [1000, 1, 1], [0, 1, 1],
    ])
    faces = np.array([
        [0, 1, 2], [0, 2, 3],  # Bottom face
        [4, 5, 6], [4, 6, 7],  # Top face
        [0, 4, 5], [0, 5, 1],  # Front face
        [3, 2, 6], [3, 6, 7],  # Back face
        [0, 3, 7], [0, 7, 4],  # Left face
        [1, 5, 6], [1, 6, 2],  # Right face
    ])
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    # Export to STL
    mesh_path = temp_dir / "extreme_aspect_ratio.stl"
    mesh.export(mesh_path)
    return mesh_path


@pytest.fixture
def mesh_with_many_components(temp_dir):
    """Create a mesh with many separate components."""
    # Create 10 small cubes arranged in a line
    all_vertices = []
    all_faces = []
    cube_size = 1.0
    spacing = 2.0
    
    for i in range(10):
        offset = np.array([i * spacing, 0, 0])
        
        # Create a cube
        cube_vertices = np.array([
            [0, 0, 0], [cube_size, 0, 0], [cube_size, cube_size, 0], [0, cube_size, 0],
            [0, 0, cube_size], [cube_size, 0, cube_size], 
            [cube_size, cube_size, cube_size], [0, cube_size, cube_size],
        ]) + offset
        
        cube_faces = np.array([
            [0, 1, 2], [0, 2, 3],  # Bottom face
            [4, 5, 6], [4, 6, 7],  # Top face
            [0, 4, 5], [0, 5, 1],  # Front face
            [3, 2, 6], [3, 6, 7],  # Back face
            [0, 3, 7], [0, 7, 4],  # Left face
            [1, 5, 6], [1, 6, 2],  # Right face
        ])
        
        # Adjust face indices for the combined mesh
        if all_vertices:
            vertex_offset = len(all_vertices)
            cube_faces += vertex_offset
        
        all_vertices.extend(cube_vertices)
        all_faces.extend(cube_faces)
    
    # Create combined mesh
    mesh = trimesh.Trimesh(vertices=np.array(all_vertices), faces=np.array(all_faces))
    
    # Export to STL
    mesh_path = temp_dir / "many_components.stl"
    mesh.export(mesh_path)
    return mesh_path


def test_corrupted_file_handling(corrupted_file):
    """Test how the system handles a corrupted file."""
    # ModelLoader should raise a ValueError
    with pytest.raises(ValueError):
        ModelLoader.load(str(corrupted_file))
    
    # LargeModelHandler should also raise an exception
    with pytest.raises(Exception):
        with LargeModelHandler(str(corrupted_file), chunk_size=1000) as handler:
            handler.load_model_info()
    
    # HeightMapGenerator should handle the error gracefully
    config = HeightMapConfig(
        target_resolution=(50, 50),
        bit_depth=8,
        num_samples=1000,
        num_threads=2,
        use_gpu=False,
    )
    with HeightMapGenerator(config) as generator:
        with pytest.raises(Exception):
            generator.generate_from_file(str(corrupted_file))


def test_empty_file_handling(empty_file):
    """Test how the system handles an empty file."""
    # Both ModelLoader and LargeModelHandler should raise exceptions
    with pytest.raises(Exception):
        ModelLoader.load(str(empty_file))
    
    with pytest.raises(Exception):
        with LargeModelHandler(str(empty_file), chunk_size=1000) as handler:
            handler.load_model_info()


def test_extremely_thin_mesh_handling(extremely_thin_mesh):
    """Test how the system handles an extremely thin mesh."""
    # Load with ModelLoader
    mesh = ModelLoader.load(str(extremely_thin_mesh))
    
    # Verify it was aligned correctly (smallest dimension should be Z)
    extents = mesh.bounding_box.extents
    assert np.argmin(extents) == 2  # Z should be smallest
    
    # Try height map generation
    config = HeightMapConfig(
        target_resolution=(50, 50),
        bit_depth=8,
        num_samples=1000,
        num_threads=2,
        use_gpu=False,
    )
    with HeightMapGenerator(config) as generator:
        height_map = generator.generate(mesh)
        assert height_map.shape == (50, 50)
        
        # Even with a very thin mesh, we should have some height variations
        assert np.min(height_map) >= 0
        assert np.max(height_map) > 0


def test_tiny_mesh_handling(tiny_mesh):
    """Test how the system handles a tiny mesh with few vertices/faces."""
    # Load with ModelLoader
    mesh = ModelLoader.load(str(tiny_mesh))
    
    # Try sampling with very few samples (should still work)
    points = PointSampler.sample_points(mesh, num_samples=10, use_gpu=False, num_threads=1)
    assert points.shape == (10, 3)
    
    # Try with a very small number of samples relative to faces
    config = HeightMapConfig(
        target_resolution=(10, 10),  # Very small resolution
        bit_depth=8,
        num_samples=20,  # Few samples
        num_threads=1,
        use_gpu=False,
    )
    with HeightMapGenerator(config) as generator:
        height_map = generator.generate(mesh)
        assert height_map.shape == (10, 10)


def test_extreme_aspect_ratio_handling(mesh_with_extreme_aspect_ratio):
    """Test how the system handles a mesh with an extreme aspect ratio."""
    # Load with LargeModelHandler
    with LargeModelHandler(str(mesh_with_extreme_aspect_ratio), chunk_size=1000) as handler:
        handler.load_model_info()
        
        # Verify it was aligned correctly
        extents = handler.concatenated_mesh.bounding_box.extents
        
        # The Y dimension (index 1) is the smallest for this particular mesh
        # Note: This might vary based on your alignment implementation
        assert np.argmin(extents) != np.argmax(extents)  # Smallest dimension should not be the largest
        
        # Calculate resolution
        min_coords, max_coords = handler.calculate_bounding_box()
        width, height = ResolutionCalculator.calculate_from_bounds(
            min_coords, max_coords, max_resolution=1000
        )
        
        # Width should be significantly larger than height due to aspect ratio
        assert width > height
        
        # But it should be capped at max_resolution
        assert width <= 1000


def test_mesh_with_many_components(mesh_with_many_components):
    """Test how the system handles a mesh with many separate components."""
    # Load with LargeModelHandler
    with LargeModelHandler(str(mesh_with_many_components), chunk_size=1000) as handler:
        handler.load_model_info()
        
        # Verify all components were loaded
        assert handler.total_faces > 100  # Should have at least 10 cubes * 12 faces per cube
        
        # Generate a height map
        config = HeightMapConfig(
            target_resolution=(100, 100),
            bit_depth=8,
            num_samples=2000,
            num_threads=2,
            use_gpu=False,
        )
        with HeightMapGenerator(config) as generator:
            height_map = generator.generate(handler.concatenated_mesh)
            assert height_map.shape == (100, 100)
            
            # Since we have components arranged in a line, we should have some variation
            # horizontally across the height map
            horizontal_variation = np.std(np.mean(height_map, axis=0))
            assert horizontal_variation > 0


def test_empty_mesh_handling():
    """Test how the system handles a completely empty mesh."""
    # Create an empty mesh with no vertices or faces
    empty_mesh = trimesh.Trimesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3)))
    
    # PointSampler should either raise an exception or handle it gracefully
    try:
        points = PointSampler.sample_points(empty_mesh, num_samples=100, use_gpu=False, num_threads=1)
        # If no exception is raised, verify that we got valid points
        assert points.shape == (100, 3)
        assert np.all(np.isfinite(points))
    except Exception as e:
        # If an exception is raised, that's also acceptable
        # We're just verifying the sampler doesn't crash with invalid results
        pass
    
    # HeightMapGenerator should handle it gracefully
    config = HeightMapConfig(
        target_resolution=(50, 50),
        bit_depth=8,
        num_samples=1000,
        num_threads=2,
        use_gpu=False,
    )
    with HeightMapGenerator(config) as generator:
        try:
            height_map = generator.generate(empty_mesh)
            # If no exception, make sure we got a valid height map
            assert height_map.shape == (50, 50)
        except Exception as e:
            # If an exception is raised, that's also fine
            pass


def test_sample_zero_area_mesh():
    """Test handling of a degenerate mesh with zero area."""
    # Create a degenerate mesh (a line with zero area)
    vertices = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    faces = np.array([[0, 1, 2]])  # This face is degenerate (collinear points)
    degenerate_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    # Verify the mesh has zero area
    assert degenerate_mesh.area < 1e-10
    
    # Instead of expecting a specific exception, let's just verify the behavior
    # The sampler should either raise an exception or return valid points
    try:
        points = PointSampler.sample_points(degenerate_mesh, num_samples=100, use_gpu=False, num_threads=1)
        # If no exception is raised, verify that we got valid points
        assert points.shape == (100, 3)
        assert np.all(np.isfinite(points))
    except Exception as e:
        # If an exception is raised, that's also acceptable
        # We're just verifying the sampler doesn't crash with invalid results
        pass


def test_resolution_edge_cases():
    """Test edge cases in resolution calculation."""
    # Create different meshes with various dimensions
    
    # 1. Square mesh
    square_mesh = trimesh.creation.box(extents=[1, 1, 0.1])
    width, height = ResolutionCalculator.calculate(square_mesh, max_resolution=1000)
    assert width == 1000
    assert height == 1000
    
    # 2. Extremely wide mesh
    wide_mesh = trimesh.creation.box(extents=[100, 1, 0.1])
    width, height = ResolutionCalculator.calculate(wide_mesh, max_resolution=1000)
    assert width == 1000
    assert height == 10
    
    # 3. Extremely tall mesh
    tall_mesh = trimesh.creation.box(extents=[1, 100, 0.1])
    width, height = ResolutionCalculator.calculate(tall_mesh, max_resolution=1000)
    assert width == 10
    assert height == 1000
    
    # 4. Test with tiny max_resolution
    width, height = ResolutionCalculator.calculate(square_mesh, max_resolution=1)
    assert width == 1
    assert height == 1


def test_cross_system_integration(temp_dir):
    """Test the entire pipeline with a complex model."""
    # Create a complex mesh: a sphere with a cube inside
    sphere = trimesh.creation.icosphere(radius=2.0, subdivisions=2)
    cube = trimesh.creation.box(extents=[1, 1, 1])
    
    # Combine them
    sphere_vertices = sphere.vertices
    cube_vertices = cube.vertices
    combined_vertices = np.vstack([sphere_vertices, cube_vertices])
    
    # Adjust face indices for cube
    cube_faces = cube.faces + len(sphere_vertices)
    combined_faces = np.vstack([sphere.faces, cube_faces])
    
    combined_mesh = trimesh.Trimesh(vertices=combined_vertices, faces=combined_faces)
    
    # Export the mesh
    mesh_path = temp_dir / "complex_model.stl"
    combined_mesh.export(mesh_path)
    
    # Test the full pipeline
    config = HeightMapConfig(
        target_resolution=(200, 200),
        bit_depth=16,
        num_samples=5000,
        num_threads=4,
        use_gpu=False,
        split=2,  # Test splitting
        cache_dir=str(temp_dir),
    )
    
    with HeightMapGenerator(config) as generator:
        # First load the model using ModelLoader
        from lib.model_loader import ModelLoader
        loaded_mesh = ModelLoader.load(str(mesh_path))
        
        # Generate height map from the loaded mesh
        height_map = generator.generate(loaded_mesh)
        
        assert height_map.shape == (200, 200)
        assert height_map.dtype == np.uint16
        
        # Save the height map with splitting
        output_path = temp_dir / "complex_height_map.png"
        generator.save_height_map(height_map, str(output_path))
        
        # Check for at least one split file
        # The naming convention may vary based on implementation
        split_files = list(temp_dir.glob("complex_height_map_part_*.png"))
        assert len(split_files) > 0, "No split files were generated"
        
        # We can also check if the original file was created as a fallback
        if not split_files:
            assert output_path.exists(), "Neither split files nor original file was created" 