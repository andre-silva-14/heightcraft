"""
Tests for the ModelRepository.
"""

import os
import unittest
from unittest.mock import Mock, patch, MagicMock

import numpy as np
import trimesh

from heightcraft.core.exceptions import RepositoryError
from heightcraft.domain.mesh import Mesh
from heightcraft.infrastructure.model_repository import ModelRepository
from tests.base_test_case import BaseTestCase


class TestModelRepository(BaseTestCase):
    """Tests for the ModelRepository."""
    
    def setUp(self) -> None:
        """Set up test environment."""
        super().setUp()
        
        # Create an instance of ModelRepository
        self.repository = ModelRepository()
        
        # Create test data
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

        # Create a trimesh object for testing
        self.test_trimesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        
        # Use the temp directory from BaseTestCase and test file
        self.test_file = os.path.join(self.temp_dir.name, "test_model.obj")
        
        # Save the test mesh to the file
        self.test_trimesh.export(self.test_file)
        
        # Create a Mesh from the trimesh
        self.test_mesh = Mesh(self.test_trimesh)
    
    @patch('trimesh.load')
    def test_load_mesh(self, mock_load) -> None:
        """Test loading a mesh from a file."""
        # Set up mock
        mock_load.return_value = self.test_trimesh
        
        # Call the method
        mesh = self.repository.load(self.test_file)
        
        # Check that trimesh.load was called with the correct arguments
        mock_load.assert_called_once_with(
            self.test_file,
            force='mesh',
            process=True,
            maintain_order=True,
            skip_materials=True,
            skip_texture=True
        )
        
        # Check the returned mesh
        self.assertIsInstance(mesh, Mesh)
        np.testing.assert_array_equal(mesh.vertices, self.test_mesh.vertices)
        np.testing.assert_array_equal(mesh.faces, self.test_mesh.faces)
    
    @patch('trimesh.load')
    @patch('heightcraft.domain.mesh.Mesh._validate_mesh')  # Patch the mesh validation directly
    def test_load_mesh_with_scene(self, mock_validate_mesh, mock_load) -> None:
        """Test loading a mesh from a file that contains a scene."""
        # Create a mock scene with a single geometry
        mock_scene = MagicMock()
        # Make sure the mesh in geometry is actually a trimesh.Trimesh
        mock_scene.geometry = {'mesh_0': self.test_trimesh}
        mock_load.return_value = mock_scene
        
        # Call the method
        mesh = self.repository.load(self.test_file)
        
        # Check that we got a mesh
        self.assertIsNotNone(mesh)
    
    @patch('trimesh.load')
    @patch('heightcraft.domain.mesh.Mesh._validate_mesh')  # Patch the mesh validation directly
    def test_load_mesh_with_multiple_geometries(self, mock_validate_mesh, mock_load) -> None:
        """Test loading a mesh from a file with multiple geometries."""
        # Create a mock scene with multiple geometries
        mesh1 = self.test_trimesh
        mesh2 = trimesh.Trimesh(vertices=np.array([[1, 1, 1], [2, 1, 1], [1, 2, 1], [1, 1, 2]]),
                                faces=np.array([[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]]))
        mock_scene = MagicMock()
        mock_scene.geometry = {'mesh_0': mesh1, 'mesh_1': mesh2}
        mock_load.return_value = mock_scene
        
        # Call the method
        mesh = self.repository.load(self.test_file)
        
        # Check that we got a mesh
        self.assertIsNotNone(mesh)
    
    @patch('trimesh.load')
    def test_load_mesh_with_empty_scene(self, mock_load) -> None:
        """Test loading a mesh from a file with an empty scene."""
        # Create a mock scene with no geometries
        mock_scene = MagicMock()
        mock_scene.geometry = {}
        mock_load.return_value = mock_scene
        
        # Call the method
        with self.assertRaises(RepositoryError):
            self.repository.load(self.test_file)
    
    @patch('trimesh.load')
    def test_load_mesh_with_error(self, mock_load) -> None:
        """Test loading a mesh from a file with an error."""
        # Set up mock to raise an exception
        mock_load.side_effect = Exception("Failed to load mesh")
        
        # Call the method
        with self.assertRaises(RepositoryError):
            self.repository.load(self.test_file)
    
    def test_load_non_existent_file(self) -> None:
        """Test loading a mesh from a non-existent file."""
        # Call the method with a non-existent file
        with self.assertRaises(RepositoryError):
            self.repository.load("non_existent_file.obj")
    
    def test_save_mesh(self) -> None:
        """Test saving a mesh to a file."""
        # Patch trimesh's export method
        with patch.object(trimesh.Trimesh, 'export') as mock_export:
            mock_export.return_value = True
            
            # Call the method
            result = self.repository.save(self.test_mesh, self.test_file)
            
            # Check that trimesh's export method was called
            mock_export.assert_called_once()
            
            # Check the result
            self.assertTrue(result)
    
    def test_save_mesh_with_error(self) -> None:
        """Test saving a mesh with an error."""
        # Patch trimesh's export method to raise an exception
        with patch.object(trimesh.Trimesh, 'export') as mock_export:
            mock_export.side_effect = Exception("Export error")
            
            # Call the method
            with self.assertRaises(RepositoryError):
                self.repository.save(self.test_mesh, self.test_file)
    
    def test_save_unsupported_format(self) -> None:
        """Test saving a mesh to a file with an unsupported format."""
        # Call the method with an unsupported file format
        with self.assertRaises(RepositoryError):
            self.repository.save(self.test_mesh, "model.unsupported")
    
    def test_supported_formats(self) -> None:
        """Test the supported formats method."""
        # Call the method
        formats = self.repository.supported_formats()
        
        # Check that we got a list of supported formats
        self.assertIsInstance(formats, list)
        self.assertGreater(len(formats), 0)
        
        # Some common formats should be supported
        common_formats = ['.obj', '.stl', '.ply']
        for fmt in common_formats:
            self.assertIn(fmt, formats) 