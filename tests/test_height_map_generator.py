import os

import numpy as np
import pytest
import trimesh

from lib.height_map_generator import HeightMapConfig, HeightMapGenerator


@pytest.fixture
def sample_mesh():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    faces = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]])
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
