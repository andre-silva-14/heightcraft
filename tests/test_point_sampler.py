import pytest
import numpy as np
import trimesh
import torch
from lib.point_sampler import PointSampler

@pytest.fixture
def sample_mesh():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    faces = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]])
    return trimesh.Trimesh(vertices=vertices, faces=faces)

def test_sample_points_cpu(sample_mesh):
    num_samples = 1000
    num_threads = 4
    points = PointSampler.sample_points(sample_mesh, num_samples, use_gpu=False, num_threads=num_threads)
    assert points.shape == (num_samples, 3)
    assert np.all(points >= 0) and np.all(points <= 1)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_sample_points_gpu(sample_mesh):
    num_samples = 1000
    points = PointSampler.sample_points(sample_mesh, num_samples, use_gpu=True, num_threads=1)
    assert points.shape == (num_samples, 3)
    assert np.all(points >= 0) and np.all(points <= 1)

def test_sample_points_cpu_multithreading(sample_mesh):
    num_samples = 1000
    num_threads_list = [1, 2, 4]
    for num_threads in num_threads_list:
        points = PointSampler.sample_points(sample_mesh, num_samples, use_gpu=False, num_threads=num_threads)
        assert points.shape == (num_samples, 3)
        assert np.all(points >= 0) and np.all(points <= 1)

def test_sample_points_invalid_num_samples(sample_mesh):
    with pytest.raises(ValueError, match="Number of samples must be a positive integer"):
        PointSampler.sample_points(sample_mesh, num_samples=0, use_gpu=False, num_threads=1)

def test_sample_points_invalid_num_threads(sample_mesh):
    with pytest.raises(ValueError, match="Number of threads must be a positive integer"):
        PointSampler.sample_points(sample_mesh, num_samples=1000, use_gpu=False, num_threads=0)
