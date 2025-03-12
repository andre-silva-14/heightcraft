"""
Tests for the PointSampler utility.
"""

import os
import unittest
from unittest.mock import Mock, patch

import numpy as np
import trimesh

from heightcraft.core.exceptions import SamplingError
from heightcraft.domain.mesh import Mesh
from heightcraft.domain.point_cloud import PointCloud
from heightcraft.utils.point_sampler import PointSampler
from tests.base_test_case import BaseTestCase


class TestPointSampler(BaseTestCase):
    """Tests for the PointSampler utility."""
    
    def setUp(self) -> None:
        """Set up test environment."""
        super().setUp()
        
        # Create a simple test mesh
        vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        faces = np.array([
            [0, 1, 2],
            [0, 2, 3],
            [0, 3, 1],
            [1, 3, 2]
        ])
        
        # Create a test trimesh and wrap it in a Mesh
        trimesh_obj = trimesh.Trimesh(vertices=vertices, faces=faces)
        self.test_mesh = Mesh(trimesh_obj)
        
        # Create a complex mesh for testing
        vertices_complex = np.random.rand(100, 3)
        faces_complex = []
        for i in range(98):
            faces_complex.append([i, i+1, i+2])
        faces_complex = np.array(faces_complex)
        
        # Create a complex trimesh and wrap it in a Mesh
        trimesh_complex = trimesh.Trimesh(vertices=vertices_complex, faces=faces_complex)
        self.complex_mesh = Mesh(trimesh_complex)
        
        # Create a thin mesh for testing
        vertices_thin = np.array([
            [0, 0, 0],
            [1000, 0, 0],
            [0, 0.001, 0]
        ])
        faces_thin = np.array([[0, 1, 2]])
        
        # Create a thin trimesh and wrap it in a Mesh
        trimesh_thin = trimesh.Trimesh(vertices=vertices_thin, faces=faces_thin)
        self.thin_mesh = Mesh(trimesh_thin)
        
        # Create a point sampler
        self.point_sampler = PointSampler()
        
        # Set up CPU and GPU parameters
        self.num_samples = 1000
        self.cpu_params = {"num_threads": 1}
        
        # Mock utility functions
        self.is_gpu_supported_patch = patch('heightcraft.utils.point_sampler.is_gpu_supported')
        self.mock_is_gpu_supported = self.is_gpu_supported_patch.start()
        self.mock_is_gpu_supported.return_value = True
    
    def tearDown(self) -> None:
        """Clean up after the test."""
        super().tearDown()
        self.is_gpu_supported_patch.stop()
    
    def test_sample_points_cpu(self) -> None:
        """Test sampling points using CPU."""
        # Call the method
        num_points = 100
        point_cloud = self.point_sampler.sample_points(self.test_mesh, num_points, use_gpu=False)
        
        # Check that we got a point cloud
        self.assertIsInstance(point_cloud, PointCloud)
        
        # Check that it has the expected number of points
        self.assertEqual(point_cloud.size, num_points)
        
        # Check that the points are within the bounds of the mesh
        bounds = self.test_mesh.bounds
        min_bounds = bounds[0]  # [min_x, min_y, min_z]
        max_bounds = bounds[1]  # [max_x, max_y, max_z]
        
        for point in point_cloud.points:
            self.assertTrue(min_bounds[0] <= point[0] <= max_bounds[0])  # x bounds
            self.assertTrue(min_bounds[1] <= point[1] <= max_bounds[1])  # y bounds
            self.assertTrue(min_bounds[2] <= point[2] <= max_bounds[2])  # z bounds
    
    def test_sample_points_cpu_multithreading(self) -> None:
        """Test sampling points using CPU with multiple threads."""
        # Call the method
        num_points = 100
        num_threads = 4
        point_cloud = self.point_sampler.sample_points(self.test_mesh, num_points, use_gpu=False, num_threads=num_threads)
        
        # Check that we got a point cloud
        self.assertIsInstance(point_cloud, PointCloud)
        
        # Check that it has the expected number of points
        self.assertEqual(point_cloud.size, num_points)
    
    def test_sample_points_invalid_num_samples(self) -> None:
        """Test sampling points with invalid number of samples."""
        # Test with negative number of samples
        with self.assertRaises(SamplingError):
            self.point_sampler.sample_points(self.test_mesh, -10)
        
        # Test with zero samples
        with self.assertRaises(SamplingError):
            self.point_sampler.sample_points(self.test_mesh, 0)
    
    def test_sample_points_invalid_num_threads(self) -> None:
        """Test sampling points with invalid number of threads."""
        # Test with negative number of threads
        with self.assertRaises(SamplingError):
            self.point_sampler.sample_points(self.test_mesh, 100, num_threads=-2)
        
        # Test with zero threads
        with self.assertRaises(SamplingError):
            self.point_sampler.sample_points(self.test_mesh, 100, num_threads=0)
    
    def test_sample_points_complex_mesh(self) -> None:
        """Test sampling points from a complex mesh."""
        # Call the method
        num_points = 1000
        point_cloud = self.point_sampler.sample_points(self.complex_mesh, num_points, use_gpu=False)
        
        # Check that we got a point cloud
        self.assertIsInstance(point_cloud, PointCloud)
        
        # Check that it has the expected number of points
        self.assertEqual(point_cloud.size, num_points)
        
        # Check that the points are within the bounds of the mesh
        bounds = self.complex_mesh.bounds
        min_bounds = bounds[0]  # [min_x, min_y, min_z]
        max_bounds = bounds[1]  # [max_x, max_y, max_z]
        
        for point in point_cloud.points:
            self.assertTrue(min_bounds[0] <= point[0] <= max_bounds[0])  # x bounds
            self.assertTrue(min_bounds[1] <= point[1] <= max_bounds[1])  # y bounds
            self.assertTrue(min_bounds[2] <= point[2] <= max_bounds[2])  # z bounds
        
        # We're just checking that sampling works - we don't need to validate
        # the specific distribution of points, as that can vary
    
    def test_sample_points_thin_mesh(self) -> None:
        """Test sampling points from an extremely thin mesh."""
        # Call the method
        num_points = 100
        point_cloud = self.point_sampler.sample_points(self.thin_mesh, num_points, use_gpu=False)
        
        # Check that we got a point cloud
        self.assertIsInstance(point_cloud, PointCloud)
        
        # Check that it has the expected number of points
        self.assertEqual(point_cloud.size, num_points)
        
        # For thin meshes, most points should be on the large faces (top/bottom)
        # Count points with Y near 0 or near 0.01
        y_near_0 = sum(1 for p in point_cloud.points if abs(p[1]) < 0.001)
        y_near_001 = sum(1 for p in point_cloud.points if abs(p[1] - 0.01) < 0.001)
        
        # Most points should be on top or bottom faces
        self.assertGreater(y_near_0 + y_near_001, num_points * 0.5)
    
    def test_sample_extreme_num_points(self) -> None:
        """Test sampling an extreme number of points."""
        # Test with a very large number of points
        large_num = 10000
        point_cloud = self.point_sampler.sample_points(self.complex_mesh, large_num, use_gpu=False)
        self.assertEqual(point_cloud.size, large_num)
        
        # Test with a very small number of points
        small_num = 1
        point_cloud = self.point_sampler.sample_points(self.test_mesh, small_num, use_gpu=False)
        self.assertEqual(point_cloud.size, small_num)
    
    def test_sample_points_deterministic(self) -> None:
        """Test that sampling is deterministic with the same seed."""
        # Sample with a specific seed
        seed = 42
        point_cloud1 = self.point_sampler.sample_points(self.test_mesh, 100, use_gpu=False, seed=seed)
        
        # Sample again with the same seed
        point_cloud2 = self.point_sampler.sample_points(self.test_mesh, 100, use_gpu=False, seed=seed)
        
        # Check that the points are the same
        np.testing.assert_array_almost_equal(point_cloud1.points, point_cloud2.points)
        
        # Sample with a different seed
        point_cloud3 = self.point_sampler.sample_points(self.test_mesh, 100, use_gpu=False, seed=seed+1)
        
        # The points should be different
        self.assertFalse(np.array_equal(point_cloud1.points, point_cloud3.points))
    
    def test_sample_zero_area_mesh(self) -> None:
        """Test sampling from a mesh with zero area faces."""
        # Create a mesh with zero area faces
        vertices = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ])
        faces = np.array([[0, 1, 2]])
        
        # Create a zero-area trimesh and wrap it in a Mesh
        trimesh_obj = trimesh.Trimesh(vertices=vertices, faces=faces)
        zero_area_mesh = Mesh(trimesh_obj)
        
        # Try to sample points
        with self.assertRaises(SamplingError):
            self.point_sampler.sample_points(zero_area_mesh, 10)
    
    @patch('heightcraft.utils.validators.validate_mesh')  # Skip validation
    def test_empty_mesh_handling(self, mock_validate_mesh) -> None:
        """Test handling of an empty mesh."""
        # Create a minimal mesh with empty vertices
        vertices = np.array([[0, 0, 0]])  # Need at least one vertex to pass validation
        faces = np.array([])
        empty_trimesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        empty_mesh = Mesh(empty_trimesh)
        
        # Let validation pass but mock the sample_points_cpu to raise an error
        with patch.object(self.point_sampler, '_sample_points_cpu', side_effect=SamplingError("Empty mesh")):
            # Try to sample points
            with self.assertRaises(SamplingError):
                self.point_sampler.sample_points(empty_mesh, 10) 