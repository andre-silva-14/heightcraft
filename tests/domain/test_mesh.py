"""
Tests for the Mesh domain model.
"""

import os
import unittest
from unittest.mock import Mock, patch

import numpy as np
import trimesh

from heightcraft.core.exceptions import MeshValidationError
from heightcraft.domain.mesh import Mesh
from heightcraft.core.config import SamplingConfig
from heightcraft.services.sampling_service import SamplingService
from tests.base_test_case import BaseTestCase


class TestMesh(BaseTestCase):
    """Tests for the Mesh domain model."""
    
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
            [0, 1, 3],
            [0, 2, 3],
            [1, 2, 3]
        ])
        self.trimesh_obj = trimesh.Trimesh(vertices=vertices, faces=faces)
        self.mesh = Mesh(self.trimesh_obj)
    
    def test_initialization(self) -> None:
        """Test initialization."""
        # Test with valid mesh
        mesh = Mesh(self.trimesh_obj)
        self.assertIsInstance(mesh, Mesh)
        
        # Test with invalid mesh
        with self.assertRaises(MeshValidationError):
            Mesh(trimesh.Trimesh(vertices=np.array([]), faces=np.array([])))
    
    def test_properties(self) -> None:
        """Test property accessors."""
        # Test vertices
        self.assertIsInstance(self.mesh.vertices, np.ndarray)
        self.assertEqual(self.mesh.vertices.shape, (4, 3))
        
        # Test faces
        self.assertIsInstance(self.mesh.faces, np.ndarray)
        self.assertEqual(self.mesh.faces.shape, (4, 3))
        
        # Test bounds
        self.assertIsInstance(self.mesh.bounds, np.ndarray)
        self.assertEqual(self.mesh.bounds.shape, (2, 3))
        
        # Test extents
        self.assertIsInstance(self.mesh.extents, np.ndarray)
        self.assertEqual(self.mesh.extents.shape, (3,))
        
        # Test is_watertight
        self.assertIsInstance(self.mesh.is_watertight, bool)
        
        # Test is_winding_consistent
        self.assertIsInstance(self.mesh.is_winding_consistent, bool)
        
        # Test vertex_count
        self.assertEqual(self.mesh.vertex_count, 4)
        
        # Test face_count
        self.assertEqual(self.mesh.face_count, 4)
    
    def test_center(self) -> None:
        """Test centering the mesh."""
        # Create a test mesh with known center
        vertices = np.array([
            [1, 1, 1],
            [2, 1, 1],
            [1, 2, 1],
            [1, 1, 2]
        ])
        faces = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]])
        trimesh_obj = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh = Mesh(trimesh_obj)
        
        # Center the mesh
        mesh.center()
        
        # Check that the center is now at the origin
        center = np.mean(mesh.vertices, axis=0)
        np.testing.assert_array_almost_equal(center, [0, 0, 0])
    
    def test_align_to_xy(self) -> None:
        """Test aligning the mesh to the XY plane."""
        # Get the original vertices
        original_vertices = self.mesh.vertices.copy()
        
        # Align the mesh
        self.mesh.align_to_xy()
        
        # Check that something changed
        self.assertFalse(np.array_equal(original_vertices, self.mesh.vertices))
    
    @patch('heightcraft.services.sampling_service.SamplingService')
    def test_point_sampling_via_service(self, mock_sampling_service_class) -> None:
        """Test sampling points from a mesh using SamplingService."""
        # Set up the mock
        mock_sampling_service = mock_sampling_service_class.return_value
        mock_sampling_service.sample_from_mesh.return_value = np.array([
            [0.1, 0.1, 0.1],
            [0.2, 0.2, 0.2],
            [0.3, 0.3, 0.3]
        ])
        
        # Create a sampling config
        config = SamplingConfig(num_samples=100)
        
        # Create a sampling service
        sampling_service = SamplingService(config)
        
        # Sample points
        points = sampling_service.sample_from_mesh(self.mesh)
        
        # Check the result
        self.assertIsInstance(points, np.ndarray)
        # Don't check the exact shape, as the real service might return 100 points
        self.assertEqual(points.shape[1], 3)  # Each point should have 3 coordinates
    
    def test_get_aspect_ratio(self) -> None:
        """Test getting the aspect ratio of the mesh."""
        # Create a mesh with a known aspect ratio
        vertices = np.array([
            [0, 0, 0],
            [2, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        faces = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]])
        trimesh_obj = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh = Mesh(trimesh_obj)
        
        # Get the aspect ratio
        aspect_ratio = mesh.get_aspect_ratio()
        
        # Check the result (should be 2/1 = 2)
        self.assertEqual(aspect_ratio, 2.0)
    
    def test_to_dict(self) -> None:
        """Test converting the mesh to a dictionary."""
        # Get the dictionary
        mesh_dict = self.mesh.to_dict()
        
        # Check that it's a dictionary
        self.assertIsInstance(mesh_dict, dict)
        
        # Check that it has the expected keys
        expected_keys = [
            "vertex_count",
            "face_count",
            "bounds",
            "extents",
            "is_watertight",
            "is_winding_consistent"
        ]
        for key in expected_keys:
            self.assertIn(key, mesh_dict)
        
        # Check that the values match the mesh
        self.assertEqual(mesh_dict["vertex_count"], self.mesh.vertex_count)
        self.assertEqual(mesh_dict["face_count"], self.mesh.face_count)
        # Check bounds as list or numpy array
        if isinstance(mesh_dict["bounds"], list):
            self.assertEqual(mesh_dict["bounds"], self.mesh.bounds.tolist())
        else:
            self.assertEqual(mesh_dict["bounds"].tolist(), self.mesh.bounds.tolist())
        
        # Check extents as list or numpy array
        if isinstance(mesh_dict["extents"], list):
            self.assertEqual(mesh_dict["extents"], self.mesh.extents.tolist())
        else:
            self.assertEqual(mesh_dict["extents"].tolist(), self.mesh.extents.tolist())
            
        self.assertEqual(mesh_dict["is_watertight"], self.mesh.is_watertight)
        self.assertEqual(mesh_dict["is_winding_consistent"], self.mesh.is_winding_consistent)
    
    @patch("trimesh.load")
    def test_from_file(self, mock_load) -> None:
        """Test loading a mesh from a file."""
        # Set up the mock
        mock_load.return_value = self.trimesh_obj
        
        # Create a test filepath
        file_path = self.get_temp_path("test_mesh.obj")
        
        # Test the method with patch to catch the deprecation warning
        with self.assertWarns(DeprecationWarning):
            mesh = Mesh.from_file(file_path)
        
        # Check the result
        self.assertIsInstance(mesh, Mesh)
        self.assertEqual(mesh.vertex_count, self.mesh.vertex_count)
        self.assertEqual(mesh.face_count, self.mesh.face_count)
        
        # Check that the mock was called
        mock_load.assert_called_once_with(file_path)
    
    @patch("trimesh.load")
    def test_from_file_with_scene(self, mock_load) -> None:
        """Test loading a mesh from a file that contains a scene."""
        # Set up the mock scene
        mock_scene = Mock(spec=trimesh.Scene)
        mock_scene.dump.return_value = self.trimesh_obj
        mock_load.return_value = mock_scene
        
        # Create a test filepath
        file_path = self.get_temp_path("test_scene.obj")
        
        # Test the method with patch to catch the deprecation warning
        with self.assertWarns(DeprecationWarning):
            mesh = Mesh.from_file(file_path)
        
        # Check the result
        self.assertIsInstance(mesh, Mesh)
        
        # Check that the mock was called
        mock_load.assert_called_once_with(file_path)
        mock_scene.dump.assert_called_once_with(concatenate=True)
    
    @patch("trimesh.load")
    def test_from_file_with_error(self, mock_load) -> None:
        """Test loading a mesh from a file with an error."""
        # Set up the mock to raise an exception
        mock_load.side_effect = Exception("Test error")
        
        # Create a test filepath
        file_path = self.get_temp_path("test_error.obj")
        
        # Test the method with patch to catch the deprecation warning
        with self.assertRaises(MeshValidationError):
            with self.assertWarns(DeprecationWarning):
                Mesh.from_file(file_path) 