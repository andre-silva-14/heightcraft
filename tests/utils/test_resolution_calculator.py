"""
Tests for the ResolutionCalculator utility.
"""

import os
import unittest
from unittest.mock import Mock, patch

import numpy as np
import trimesh

from heightcraft.core.exceptions import CalculationError
from heightcraft.domain.mesh import Mesh
from heightcraft.utils.resolution_calculator import ResolutionCalculator
from tests.base_test_case import BaseTestCase


class TestResolutionCalculator(BaseTestCase):
    """Tests for the ResolutionCalculator utility."""
    
    def setUp(self) -> None:
        """Set up test environment."""
        super().setUp()
        
        # Create an instance of ResolutionCalculator
        self.calculator = ResolutionCalculator()
        
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
        
        # Create a trimesh and pass it to Mesh constructor
        trimesh_obj = trimesh.Trimesh(vertices=vertices, faces=faces)
        self.test_mesh = Mesh(trimesh_obj)
        
        # Create a non-uniform test mesh
        vertices_non_uniform = np.array([
            [0, 0, 0], [2, 0, 0], [2, 0.2, 0], [0, 0.2, 0],  # Bottom face
            [0, 0, 1], [2, 0, 1], [2, 0.2, 1], [0, 0.2, 1],  # Top face
        ])
        faces_non_uniform = np.array([
            [0, 1, 2], [0, 2, 3],  # Bottom face
            [4, 5, 6], [4, 6, 7],  # Top face
            [0, 4, 5], [0, 5, 1],  # Front face
            [3, 2, 6], [3, 6, 7],  # Back face
            [0, 3, 7], [0, 7, 4],  # Left face
            [1, 5, 6], [1, 6, 2],  # Right face
        ])
        
        # Create a non-uniform trimesh and pass it to Mesh constructor
        trimesh_non_uniform = trimesh.Trimesh(vertices=vertices_non_uniform, faces=faces_non_uniform)
        self.non_uniform_mesh = Mesh(trimesh_non_uniform)
    
    def test_calculate_optimal_resolution(self) -> None:
        """Test calculating the optimal resolution."""
        # Call the method
        target_width = 256
        target_height = 256
        resolution = self.calculator.calculate_optimal_resolution(self.test_mesh, target_width, target_height)
        
        # Check that we got a positive float
        self.assertIsInstance(resolution, float)
        self.assertGreater(resolution, 0)
        
        # For our test mesh, the X and Y dimensions are the same (1x1),
        # so the resolution should be 1 / target_width
        expected_resolution = 1.0 / target_width
        self.assertAlmostEqual(resolution, expected_resolution)
    
    def test_calculate_optimal_resolution_non_uniform(self) -> None:
        """Test calculating the optimal resolution for a non-uniform mesh."""
        # Call the method
        target_width = 256
        target_height = 256
        resolution = self.calculator.calculate_optimal_resolution(self.non_uniform_mesh, target_width, target_height)
        
        # Check that we got a positive float
        self.assertIsInstance(resolution, float)
        self.assertGreater(resolution, 0)
        
        # For our non-uniform mesh, the X dimension is 2 and the Y dimension is 0.2,
        # so the resolution should be based on the larger dimension (X)
        expected_resolution = 2.0 / target_width
        self.assertAlmostEqual(resolution, expected_resolution)
    
    def test_calculate_optimal_resolution_with_invalid_target_dimensions(self) -> None:
        """Test calculating the optimal resolution with invalid target dimensions."""
        # Call the method with negative width
        with self.assertRaises(CalculationError):
            self.calculator.calculate_optimal_resolution(self.test_mesh, -1, 256)
        
        # Call the method with negative height
        with self.assertRaises(CalculationError):
            self.calculator.calculate_optimal_resolution(self.test_mesh, 256, -1)
        
        # Call the method with zero width
        with self.assertRaises(CalculationError):
            self.calculator.calculate_optimal_resolution(self.test_mesh, 0, 256)
        
        # Call the method with zero height
        with self.assertRaises(CalculationError):
            self.calculator.calculate_optimal_resolution(self.test_mesh, 256, 0)
    
    def test_calculate_dimensions_from_resolution(self) -> None:
        """Test calculating dimensions from a resolution."""
        # Call the method
        resolution = 0.01
        width, height = self.calculator.calculate_dimensions_from_resolution(self.test_mesh, resolution)
        
        # Check that we got positive integers
        self.assertIsInstance(width, int)
        self.assertIsInstance(height, int)
        self.assertGreater(width, 0)
        self.assertGreater(height, 0)
        
        # For our test mesh, the X and Y dimensions are the same (1x1),
        # so the width and height should be 1 / resolution
        expected_dimension = int(1.0 / resolution)
        self.assertEqual(width, expected_dimension)
        self.assertEqual(height, expected_dimension)
    
    def test_calculate_dimensions_from_resolution_non_uniform(self) -> None:
        """Test calculating dimensions from a resolution for a non-uniform mesh."""
        # Call the method
        resolution = 0.01
        width, height = self.calculator.calculate_dimensions_from_resolution(self.non_uniform_mesh, resolution)
        
        # Check that we got positive integers
        self.assertIsInstance(width, int)
        self.assertIsInstance(height, int)
        self.assertGreater(width, 0)
        self.assertGreater(height, 0)
        
        # For our non-uniform mesh, the X dimension is 2 and the Y dimension is 0.2
        expected_width = int(2.0 / resolution)
        expected_height = int(0.2 / resolution)
        self.assertEqual(width, expected_width)
        self.assertEqual(height, expected_height)
    
    def test_calculate_dimensions_from_resolution_with_invalid_resolution(self) -> None:
        """Test calculating dimensions from an invalid resolution."""
        # Call the method with negative resolution
        with self.assertRaises(CalculationError):
            self.calculator.calculate_dimensions_from_resolution(self.test_mesh, -0.01)
        
        # Call the method with zero resolution
        with self.assertRaises(CalculationError):
            self.calculator.calculate_dimensions_from_resolution(self.test_mesh, 0)
    
    def test_calculate_resolution_from_point_count(self) -> None:
        """Test calculating resolution from a target point count."""
        # Call the method
        target_points = 10000
        resolution = self.calculator.calculate_resolution_from_point_count(self.test_mesh, target_points)
        
        # Check that we got a positive float
        self.assertIsInstance(resolution, float)
        self.assertGreater(resolution, 0)
        
        # For our test mesh, the X and Y dimensions are the same (1x1),
        # so the resolution should be sqrt(1 / target_points)
        expected_resolution = np.sqrt(1.0 / target_points)
        self.assertAlmostEqual(resolution, expected_resolution)
    
    def test_calculate_resolution_from_point_count_non_uniform(self) -> None:
        """Test calculating resolution from a target point count for a non-uniform mesh."""
        # Call the method
        target_points = 10000
        resolution = self.calculator.calculate_resolution_from_point_count(self.non_uniform_mesh, target_points)
        
        # Check that we got a positive float
        self.assertIsInstance(resolution, float)
        self.assertGreater(resolution, 0)
        
        # For our non-uniform mesh, the X dimension is 2 and the Y dimension is 0.2,
        # so the area is 0.4 and the resolution should be sqrt(0.4 / target_points)
        expected_resolution = np.sqrt(0.4 / target_points)
        self.assertAlmostEqual(resolution, expected_resolution)
    
    def test_calculate_resolution_from_point_count_with_invalid_target_points(self) -> None:
        """Test calculating resolution from an invalid target point count."""
        # Call the method with negative target points
        with self.assertRaises(CalculationError):
            self.calculator.calculate_resolution_from_point_count(self.test_mesh, -1)
        
        # Call the method with zero target points
        with self.assertRaises(CalculationError):
            self.calculator.calculate_resolution_from_point_count(self.test_mesh, 0)
    
    def test_estimate_point_count(self) -> None:
        """Test estimating the point count from a resolution."""
        # Call the method
        resolution = 0.01
        point_count = self.calculator.estimate_point_count(self.test_mesh, resolution)
        
        # Check that we got a positive integer
        self.assertIsInstance(point_count, int)
        self.assertGreater(point_count, 0)
        
        # For our test mesh, the X and Y dimensions are the same (1x1),
        # so the point count should be (1 / resolution)^2
        expected_point_count = int((1.0 / resolution) ** 2)
        self.assertEqual(point_count, expected_point_count)
    
    def test_estimate_point_count_non_uniform(self) -> None:
        """Test estimating the point count from a resolution for a non-uniform mesh."""
        # Call the method
        resolution = 0.01
        point_count = self.calculator.estimate_point_count(self.non_uniform_mesh, resolution)
        
        # Check that we got a positive integer
        self.assertIsInstance(point_count, int)
        self.assertGreater(point_count, 0)
        
        # For our non-uniform mesh, the X dimension is 2 and the Y dimension is 0.2,
        # so the point count should be (2 / resolution) * (0.2 / resolution)
        expected_point_count = int((2.0 / resolution) * (0.2 / resolution))
        self.assertEqual(point_count, expected_point_count)
    
    def test_estimate_point_count_with_invalid_resolution(self) -> None:
        """Test estimating the point count from an invalid resolution."""
        # Call the method with negative resolution
        with self.assertRaises(CalculationError):
            self.calculator.estimate_point_count(self.test_mesh, -0.01)
        
        # Call the method with zero resolution
        with self.assertRaises(CalculationError):
            self.calculator.estimate_point_count(self.test_mesh, 0) 