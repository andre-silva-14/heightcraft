"""
Tests for the PointCloud domain model.
"""

import os
import unittest
from unittest.mock import Mock, patch

import numpy as np

from heightcraft.core.exceptions import SamplingError
from heightcraft.domain.point_cloud import PointCloud
from tests.base_test_case import BaseTestCase


class TestPointCloud(BaseTestCase):
    """Tests for the PointCloud domain model."""
    
    def setUp(self) -> None:
        """Set up test environment."""
        super().setUp()
        
        # Create a simple test point cloud
        self.points = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 1]
        ])
        self.point_cloud = PointCloud(self.points)
    
    def test_initialization(self) -> None:
        """Test initialization."""
        # Test with valid points
        point_cloud = PointCloud(self.points)
        self.assertIsInstance(point_cloud, PointCloud)
        
        # Test with invalid points (not a numpy array)
        with self.assertRaises(SamplingError):
            PointCloud([[0, 0, 0], [1, 1, 1]])
        
        # Test with invalid points (wrong shape)
        with self.assertRaises(SamplingError):
            PointCloud(np.array([[0, 0], [1, 1]]))
        
        # Test with invalid points (empty)
        with self.assertRaises(SamplingError):
            PointCloud(np.array([]))
    
    def test_properties(self) -> None:
        """Test properties."""
        # Test points property
        np.testing.assert_array_equal(self.point_cloud.points, self.points)
        
        # Test size property
        self.assertEqual(self.point_cloud.size, len(self.points))
        
        # Test x property
        np.testing.assert_array_equal(self.point_cloud.x, self.points[:, 0])
        
        # Test y property
        np.testing.assert_array_equal(self.point_cloud.y, self.points[:, 1])
        
        # Test z property
        np.testing.assert_array_equal(self.point_cloud.z, self.points[:, 2])
        
        # Test bounds property
        bounds = self.point_cloud.bounds
        self.assertEqual(bounds["min_x"], 0)
        self.assertEqual(bounds["max_x"], 1)
        self.assertEqual(bounds["min_y"], 0)
        self.assertEqual(bounds["max_y"], 1)
        self.assertEqual(bounds["min_z"], 0)
        self.assertEqual(bounds["max_z"], 1)
    
    def test_get_xy_points(self) -> None:
        """Test getting the XY points."""
        # Get XY points
        xy_points = self.point_cloud.get_xy_points()
        
        # Check that we got the expected shape
        self.assertEqual(xy_points.shape, (5, 2))
        
        # Check that we got the expected values
        np.testing.assert_array_equal(xy_points, self.points[:, :2])
    
    def test_normalize_z(self) -> None:
        """Test normalizing the Z coordinates."""
        # Normalize Z
        normalized_z = self.point_cloud.normalize_z()
        
        # Check that we got the expected shape
        self.assertEqual(normalized_z.shape, (5,))
        
        # Check that values are in the range [0, 1]
        self.assertTrue(np.all(normalized_z >= 0))
        self.assertTrue(np.all(normalized_z <= 1))
        
        # Check specific values
        self.assertEqual(normalized_z[0], 0)  # min z
        self.assertEqual(normalized_z[3], 1)  # max z
    
    def test_get_aspect_ratio(self) -> None:
        """Test getting the aspect ratio."""
        # Get aspect ratio
        aspect_ratio = self.point_cloud.get_aspect_ratio()
        
        # Check that it's a float
        self.assertIsInstance(aspect_ratio, float)
        
        # Check that it's positive
        self.assertGreater(aspect_ratio, 0)
        
        # For our test data, X and Y ranges are the same, so aspect ratio should be 1
        self.assertEqual(aspect_ratio, 1.0)
    
    def test_subsample(self) -> None:
        """Test subsampling the point cloud."""
        # Subsample to 3 points
        subsampled = self.point_cloud.subsample(3)
        
        # Check that we got the expected size
        self.assertEqual(subsampled.size, 3)
        
        # Check that the points are from the original point cloud
        for point in subsampled.points:
            self.assertTrue(any(np.array_equal(point, original_point) for original_point in self.points))
        
        # Test with too many points
        with self.assertRaises(SamplingError):
            self.point_cloud.subsample(10)
    
    def test_to_dict(self) -> None:
        """Test converting the point cloud to a dictionary."""
        # Convert to dictionary
        point_cloud_dict = self.point_cloud.to_dict()
        
        # Check that it has the expected keys
        expected_keys = ["size", "bounds"]
        for key in expected_keys:
            self.assertIn(key, point_cloud_dict)
        
        # Check values
        self.assertEqual(point_cloud_dict["size"], self.point_cloud.size)
        self.assertEqual(point_cloud_dict["bounds"], self.point_cloud.bounds)
    
    def test_merge(self) -> None:
        """Test merging point clouds."""
        # Create a second point cloud
        points2 = np.array([
            [2, 2, 2],
            [3, 3, 3]
        ])
        point_cloud2 = PointCloud(points2)
        
        # Merge point clouds
        merged = PointCloud.merge([self.point_cloud, point_cloud2])
        
        # Check that we got the expected size
        self.assertEqual(merged.size, self.point_cloud.size + point_cloud2.size)
        
        # Check that the merged point cloud contains all points from both point clouds
        for point in self.point_cloud.points:
            self.assertTrue(any(np.array_equal(point, merged_point) for merged_point in merged.points))
        for point in point_cloud2.points:
            self.assertTrue(any(np.array_equal(point, merged_point) for merged_point in merged.points))
        
        # Test with an empty list
        with self.assertRaises(SamplingError):
            PointCloud.merge([]) 