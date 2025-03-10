import os

import numpy as np
import pytest
import trimesh
from unittest.mock import patch, MagicMock

from lib.height_map_generator import HeightMapConfig, HeightMapGenerator
from lib.model_loader import ModelLoader
from lib.large_model_handler import LargeModelHandler


@pytest.fixture
def sample_mesh():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    faces = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]])
    return trimesh.Trimesh(vertices=vertices, faces=faces)


@pytest.fixture
def non_uniform_mesh():
    """Create a non-uniform mesh for testing alignment."""
    # Create a mesh with X=2, Y=0.2, Z=1 (Y is smallest)
    vertices = np.array([
        [0, 0, 0], [2, 0, 0], [2, 0.2, 0], [0, 0.2, 0],  # Bottom face
        [0, 0, 1], [2, 0, 1], [2, 0.2, 1], [0, 0.2, 1],  # Top face
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
def height_map_config():
    return HeightMapConfig(
        target_resolution=(100, 100),
        bit_depth=16,
        num_samples=10000,
        num_threads=4,
        use_gpu=False,
        split=1,
        cache_dir=None,
        max_memory=0.8,
    )


def test_generate_height_map(sample_mesh, height_map_config):
    with HeightMapGenerator(height_map_config) as generator:
        height_map = generator.generate(sample_mesh)
        assert height_map.shape == height_map_config.target_resolution
        assert height_map.dtype == np.uint16
        assert np.min(height_map) >= 0
        assert np.max(height_map) <= 65535


def test_generate_height_map_with_non_uniform_mesh(non_uniform_mesh, height_map_config):
    """Test generating a height map with a non-uniform mesh."""
    # First, check the original dimensions
    original_extents = non_uniform_mesh.bounding_box.extents
    assert original_extents[1] < original_extents[0]  # Y < X
    assert original_extents[1] < original_extents[2]  # Y < Z
    
    # Make a copy to avoid modifying the fixture
    mesh_copy = non_uniform_mesh.copy()
    
    # Center and align the mesh as ModelLoader would do
    ModelLoader._center_model(mesh_copy)
    ModelLoader._align_model(mesh_copy)
    
    # After alignment, Z should be the smallest dimension
    aligned_extents = mesh_copy.bounding_box.extents
    assert np.argmin(aligned_extents) == 2  # Z is now smallest
    
    # Now generate a height map
    with HeightMapGenerator(height_map_config) as generator:
        height_map = generator.generate(mesh_copy)
        
        # The height map dimensions should match the target resolution
        assert height_map.shape == height_map_config.target_resolution
        
        # Check that the height map has reasonable values
        assert np.min(height_map) >= 0
        assert np.max(height_map) <= 65535
        
        # Since we've centered and aligned the mesh, the height map should have values
        # mostly in the center, with higher concentration where the mesh is taller
        # This is a heuristic test, not exact
        center_x, center_y = height_map.shape[1] // 2, height_map.shape[0] // 2
        center_region = height_map[
            center_y - 10:center_y + 10, 
            center_x - 10:center_x + 10
        ]
        # Center region should have non-zero values
        assert np.mean(center_region) > 0


def test_generate_height_map_8bit(sample_mesh):
    config = HeightMapConfig(
        target_resolution=(100, 100),
        bit_depth=8,
        num_samples=10000,
        num_threads=4,
        use_gpu=False,
        split=1,
        cache_dir=None,
        max_memory=0.8,
    )
    with HeightMapGenerator(config) as generator:
        height_map = generator.generate(sample_mesh)
        assert height_map.shape == config.target_resolution
        assert height_map.dtype == np.uint8
        assert np.min(height_map) >= 0
        assert np.max(height_map) <= 255


def test_generate_from_points():
    points = np.array([[0, 0, 0], [1, 0, 0.5], [0, 1, 1], [1, 1, 0.75]])
    target_resolution = (10, 10)
    config = HeightMapConfig(
        target_resolution=target_resolution,
        bit_depth=16,
        num_samples=1000,
        num_threads=4,
        use_gpu=False,
    )
    generator = HeightMapGenerator(config)
    height_map = generator._generate_from_points(points)
    assert height_map.shape == target_resolution
    assert height_map.dtype == np.uint16
    assert np.min(height_map) >= 0
    assert np.max(height_map) <= 65535


def test_generate_from_points_with_different_orientations():
    """Test height map generation with points in different orientations."""
    # X-Y plane points (Z is height)
    points_xy = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],  # Base points (z=0)
        [0.25, 0.25, 1], [0.75, 0.25, 1], [0.25, 0.75, 1], [0.75, 0.75, 1]  # Peak points (z=1)
    ])
    
    # X-Z plane points (Y is height) - shouldn't work well without alignment
    points_xz = np.array([
        [0, 0, 0], [1, 0, 0], [0, 0, 1], [1, 0, 1],  # Base points (y=0)
        [0.25, 1, 0.25], [0.75, 1, 0.25], [0.25, 1, 0.75], [0.75, 1, 0.75]  # Peak points (y=1)
    ])
    
    # Create a configuration
    target_resolution = (20, 20)
    config = HeightMapConfig(
        target_resolution=target_resolution,
        bit_depth=16,
        num_samples=1000,
        num_threads=4,
        use_gpu=False,
    )
    
    # Generate height maps
    generator = HeightMapGenerator(config)
    height_map_xy = generator._generate_from_points(points_xy)
    height_map_xz = generator._generate_from_points(points_xz)
    
    # Both should have the correct shape and data type
    assert height_map_xy.shape == target_resolution
    assert height_map_xz.shape == target_resolution
    
    # The XY points should produce a more distinct height map 
    # (since Z is properly used as height)
    non_zero_xy = np.count_nonzero(height_map_xy)
    non_zero_xz = np.count_nonzero(height_map_xz)
    
    # This is just a heuristic - the actual values depend on interpolation
    # XY points should have a non-zero area (reflecting the peak)
    assert non_zero_xy > 0
    
    # XZ points might also have non-zero values due to interpolation,
    # but the pattern would be different - but this test can't precisely
    # quantify that difference


def test_generate_with_model_loader_integration(tmp_path):
    """Test the integration between ModelLoader alignment and HeightMapGenerator."""
    # Create a mesh with a non-uniform orientation
    # X=2, Y=0.2, Z=1 (Y is smallest)
    vertices = np.array([
        [0, 0, 0], [2, 0, 0], [2, 0.2, 0], [0, 0.2, 0],
        [0, 0, 1], [2, 0, 1], [2, 0.2, 1], [0, 0.2, 1],
    ])
    faces = np.array([
        [0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7],
        [0, 4, 5], [0, 5, 1], [3, 2, 6], [3, 6, 7],
        [0, 3, 7], [0, 7, 4], [1, 5, 6], [1, 6, 2],
    ])
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    # Export the mesh to a temporary file
    mesh_path = tmp_path / "test_align.stl"
    mesh.export(mesh_path)

    # Create a mock for ModelLoader.load that will return a properly aligned mesh
    def mock_load(file_path):
        loaded_mesh = mesh.copy()
        ModelLoader._center_model(loaded_mesh)
        ModelLoader._align_model(loaded_mesh)
        return loaded_mesh

    with patch("lib.model_loader.ModelLoader.load", side_effect=mock_load):
        # Create configuration
        config = HeightMapConfig(
            target_resolution=(50, 50),
            bit_depth=16,
            num_samples=5000,
            num_threads=4,
            use_gpu=False,
            split=1,
            cache_dir=None,
            max_memory=0.8,
        )

        # Initialize generator
        with HeightMapGenerator(config) as generator:
            # This will use the patched ModelLoader.load
            loaded_mesh = ModelLoader.load(str(mesh_path))
            height_map = generator.generate(loaded_mesh)

            # Verify the result
            assert height_map.shape == (50, 50)
            assert height_map.dtype == np.uint16
            assert np.min(height_map) >= 0


def test_generate_with_large_model_handler_integration(tmp_path):
    """Test integration between LargeModelHandler and HeightMapGenerator."""
    # Create a simple mesh
    mesh = trimesh.Trimesh(
        vertices=np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 0.5], [1, 0, 0.5], [1, 1, 0.5], [0, 1, 0.5],
        ]),
        faces=np.array([
            [0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7],
            [0, 4, 5], [0, 5, 1], [3, 2, 6], [3, 6, 7],
            [0, 3, 7], [0, 7, 4], [1, 5, 6], [1, 6, 2],
        ])
    )

    # Export the mesh to a temporary file
    mesh_path = tmp_path / "test_large.stl"
    mesh.export(mesh_path)

    # Mock the load_model_info method to avoid actual file loading
    original_load_model_info = LargeModelHandler.load_model_info

    def mock_load_model_info(self):
        self.concatenated_mesh = mesh.copy()
        ModelLoader._center_model(self.concatenated_mesh)
        ModelLoader._align_model(self.concatenated_mesh)
        self.total_vertices = len(self.concatenated_mesh.vertices)
        self.total_faces = len(self.concatenated_mesh.faces)

    # Apply the patch
    LargeModelHandler.load_model_info = mock_load_model_info

    try:
        # Create configuration
        config = HeightMapConfig(
            target_resolution=(50, 50),
            bit_depth=16,
            num_samples=5000,
            num_threads=4,
            use_gpu=False,
            split=1,
            cache_dir=None,
            max_memory=0.8,
        )

        # Load the mesh first with LargeModelHandler
        with LargeModelHandler(str(mesh_path), chunk_size=1000) as handler:
            handler.load_model_info()
            aligned_mesh = handler.concatenated_mesh
            
            # Then use HeightMapGenerator
            with HeightMapGenerator(config) as generator:
                # Generate from the aligned mesh
                height_map = generator.generate(aligned_mesh)
                
                # Verify the result
                assert height_map.shape == (50, 50)
                assert height_map.dtype == np.uint16
                assert np.min(height_map) >= 0
                
    finally:
        # Restore the original method
        LargeModelHandler.load_model_info = original_load_model_info


def test_generate_invalid_bit_depth(sample_mesh):
    config = HeightMapConfig(
        target_resolution=(100, 100),
        bit_depth=32,  # Invalid bit depth
        num_samples=10000,
        num_threads=4,
        use_gpu=False,
        split=1,
        cache_dir=None,
        max_memory=0.8,
    )
    with pytest.raises(ValueError, match="Bit depth must be either 8 or 16"):
        with HeightMapGenerator(config) as generator:
            generator.generate(sample_mesh)


def test_save_height_map(tmp_path, height_map_config):
    height_map = np.random.randint(0, 65536, size=(100, 100), dtype=np.uint16)
    output_path = tmp_path / "test_height_map.png"

    with HeightMapGenerator(height_map_config) as generator:
        generator.save_height_map(height_map, str(output_path))
        assert output_path.exists()


def test_save_split_height_maps(tmp_path):
    config = HeightMapConfig(
        target_resolution=(100, 100),
        bit_depth=16,
        num_samples=10000,
        num_threads=4,
        use_gpu=False,
        split=4,
        cache_dir=None,
        max_memory=0.8,
    )
    height_map = np.random.randint(0, 65536, size=(100, 100), dtype=np.uint16)
    output_path = tmp_path / "test_height_map.png"

    with HeightMapGenerator(config) as generator:
        generator.save_height_map(height_map, str(output_path))

        assert (tmp_path / "test_height_map_part_0_0.png").exists()
        assert (tmp_path / "test_height_map_part_0_1.png").exists()
        assert (tmp_path / "test_height_map_part_1_0.png").exists()
        assert (tmp_path / "test_height_map_part_1_1.png").exists()


def test_get_optimal_grid():
    config = HeightMapConfig(
        target_resolution=(100, 100),
        bit_depth=16,
        num_samples=10000,
        num_threads=4,
        use_gpu=False,
        split=1,
        cache_dir=None,
        max_memory=0.8,
    )
    with HeightMapGenerator(config) as generator:
        assert generator._get_optimal_grid(4, 100, 100) == (2, 2)
        assert generator._get_optimal_grid(9, 100, 100) == (3, 3)
        assert generator._get_optimal_grid(8, 100, 200) == (2, 4)
        assert generator._get_optimal_grid(8, 200, 100) == (4, 2)
        assert generator._get_optimal_grid(6, 100, 200) == (2, 3)
        assert generator._get_optimal_grid(6, 200, 100) == (3, 2)


def test_cache_functionality(tmp_path, sample_mesh):
    config = HeightMapConfig(
        target_resolution=(100, 100),
        bit_depth=16,
        num_samples=10000,
        num_threads=4,
        use_gpu=False,
        split=1,
        cache_dir=str(tmp_path),
        max_memory=0.8,
    )

    # Create a generator with a modified _get_cache_key method for testing
    class TestGenerator(HeightMapGenerator):
        def _get_cache_key(self, mesh):
            return "test_cache_key"

    # First generation
    with TestGenerator(config) as generator:
        height_map1 = generator.generate(sample_mesh)
        # Save to in-memory cache
        generator._height_map_cache["test_cache_key"] = height_map1

    # Second generation should use cache
    with TestGenerator(config) as generator:
        # Manually set the cache
        generator._height_map_cache["test_cache_key"] = height_map1
        height_map2 = generator.generate(sample_mesh)
        assert np.array_equal(height_map1, height_map2)


def test_cleanup(tmp_path):
    config = HeightMapConfig(
        target_resolution=(100, 100),
        bit_depth=16,
        num_samples=10000,
        num_threads=4,
        use_gpu=False,
        split=1,
        cache_dir=str(tmp_path),
        max_memory=0.8,
    )

    generator = HeightMapGenerator(config)
    generator.cleanup()

    # Verify thread pool is shutdown
    assert generator.thread_pool._shutdown
    # Verify cache is cleared
    assert not generator._height_map_cache


def test_end_to_end_integration_with_different_orientations(tmp_path):
    """End-to-end test of the full workflow with differently oriented meshes."""
    # Create three meshes with different orientations
    # 1. X-dominant mesh
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

    # 2. Y-dominant mesh
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

    # 3. Z-dominant mesh
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

    # Export all meshes
    x_path = tmp_path / "x_dominant.stl"
    y_path = tmp_path / "y_dominant.stl"
    z_path = tmp_path / "z_dominant.stl"

    x_mesh.export(x_path)
    y_mesh.export(y_path)
    z_mesh.export(z_path)

    # Create a configuration with small parameters for faster testing
    config = HeightMapConfig(
        target_resolution=(50, 50),
        bit_depth=8,
        num_samples=5000,
        num_threads=2,
        use_gpu=False,
        split=1,
        cache_dir=None,
        max_memory=0.8,
    )

    # Generate height maps for all three meshes
    with HeightMapGenerator(config) as generator:
        # Process each mesh individually and verify proper alignment
        for mesh_path, expected_dominant_dim in [
            (x_path, 0),  # X dominant
            (y_path, 1),  # Y dominant
            (z_path, 2),  # Z dominant
        ]:
            # Load and align the mesh
            mesh = ModelLoader.load(str(mesh_path))
            
            # After alignment, Z should always be the smallest dimension
            extents = mesh.bounding_box.extents
            assert np.argmin(extents) == 2
            
            # Generate the height map
            height_map = generator.generate(mesh)
            
            # Verify the result
            assert height_map.shape == (50, 50)
            assert height_map.dtype == np.uint8
            assert np.min(height_map) >= 0
            assert np.max(height_map) <= 255
