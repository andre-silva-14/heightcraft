"""
Tests for the MeshService.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock

import numpy as np
import trimesh

from heightcraft.core.exceptions import MeshServiceError, MeshValidationError
from heightcraft.domain.mesh import Mesh
from heightcraft.domain.point_cloud import PointCloud
from heightcraft.services.mesh_service import MeshService
from heightcraft.utils.point_sampler import PointSampler
from tests.base_test_case import BaseTestCase


class TestMeshService(BaseTestCase):
    """Tests for the MeshService."""
    
    def setUp(self) -> None:
        """Set up test environment."""
        super().setUp()
        
        # Create an instance of MeshService
        self.mesh_service = MeshService()
        
        # Create a mock trimesh
        self.mock_trimesh = Mock(spec=trimesh.Trimesh)
        self.mock_trimesh.vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        self.mock_trimesh.faces = np.array([
            [0, 1, 2],
            [0, 2, 3],
            [0, 3, 1],
            [1, 3, 2]
        ])
        self.mock_trimesh.bounds = np.array([
            [0, 0, 0],
            [1, 1, 1]
        ])
        self.mock_trimesh.extents = np.array([1, 1, 1])
        self.mock_trimesh.is_watertight = True
        self.mock_trimesh.is_winding_consistent = True
        
        # Create a test mesh
        self.test_mesh = Mock(spec=Mesh)
        self.test_mesh.mesh = self.mock_trimesh
        self.test_mesh.vertices = self.mock_trimesh.vertices
        self.test_mesh.faces = self.mock_trimesh.faces
        self.test_mesh.bounds = self.mock_trimesh.bounds
        self.test_mesh.extents = self.mock_trimesh.extents
        self.test_mesh.is_watertight = True
        self.test_mesh.is_winding_consistent = True
        self.test_mesh.vertex_count = len(self.mock_trimesh.vertices)
        self.test_mesh.face_count = len(self.mock_trimesh.faces)
        self.test_mesh.get_aspect_ratio.return_value = 1.0
    
    def test_prepare_mesh(self) -> None:
        """Test preparing a mesh."""
        # Call the method
        result = self.mesh_service.prepare_mesh(self.test_mesh)
        
        # Check that center and align were called
        self.test_mesh.center.assert_called_once()
        self.test_mesh.align_to_xy.assert_called_once()
        
        # Check that the result is the same mesh
        self.assertEqual(result, self.test_mesh)
    
    def test_prepare_mesh_center_only(self) -> None:
        """Test preparing a mesh with center only."""
        # Call the method
        result = self.mesh_service.prepare_mesh(self.test_mesh, center=True, align=False)
        
        # Check that center was called but not align
        self.test_mesh.center.assert_called_once()
        self.test_mesh.align_to_xy.assert_not_called()
        
        # Check that the result is the same mesh
        self.assertEqual(result, self.test_mesh)
    
    def test_prepare_mesh_align_only(self) -> None:
        """Test preparing a mesh with align only."""
        # Call the method
        result = self.mesh_service.prepare_mesh(self.test_mesh, center=False, align=True)
        
        # Check that align was called but not center
        self.test_mesh.center.assert_not_called()
        self.test_mesh.align_to_xy.assert_called_once()
        
        # Check that the result is the same mesh
        self.assertEqual(result, self.test_mesh)
    
    @patch('heightcraft.services.mesh_service.validate_mesh')
    def test_validate_mesh(self, mock_validate_mesh) -> None:
        """Test validating a mesh."""
        # Call the method
        self.mesh_service.validate_mesh(self.test_mesh)
        
        # Check that validate_mesh was called
        mock_validate_mesh.assert_called_once_with(self.test_mesh, False)
    
    @patch('heightcraft.services.mesh_service.validate_mesh')
    def test_validate_mesh_strict(self, mock_validate_mesh) -> None:
        """Test validating a mesh with strict validation."""
        # Call the method
        self.mesh_service.validate_mesh(self.test_mesh, strict=True)
        
        # Check that validate_mesh was called with strict=True
        mock_validate_mesh.assert_called_once_with(self.test_mesh, True)
    
    def test_get_bounds(self) -> None:
        """Test getting mesh bounds."""
        # Call the method
        result = self.mesh_service.get_bounds(self.test_mesh)
        
        # Check the result
        np.testing.assert_array_equal(result, self.test_mesh.bounds)
    
    def test_get_aspect_ratio(self) -> None:
        """Test getting mesh aspect ratio."""
        # Call the method
        result = self.mesh_service.get_aspect_ratio(self.test_mesh)
        
        # Check the result
        self.assertEqual(result, 1.0)
        self.test_mesh.get_aspect_ratio.assert_called_once()
    
    def test_calculate_target_resolution(self) -> None:
        """Test calculating target resolution."""
        # Set up mock
        self.test_mesh.get_aspect_ratio.return_value = 2.0
        
        # Call the method
        result = self.mesh_service.calculate_target_resolution(self.test_mesh, 256)
        
        # Check the result
        self.assertEqual(result, (256, 128))
    
    def test_calculate_target_resolution_tall(self) -> None:
        """Test calculating target resolution for a tall mesh."""
        # Set up mock
        self.test_mesh.get_aspect_ratio.return_value = 0.5
        
        # Call the method
        result = self.mesh_service.calculate_target_resolution(self.test_mesh, 256)
        
        # Check the result
        self.assertEqual(result, (128, 256))
    
    def test_combine_meshes(self) -> None:
        """Test combining meshes."""
        # Set up mocks
        mesh1 = Mock(spec=Mesh)
        mesh1.mesh = Mock(spec=trimesh.Trimesh)
        mesh2 = Mock(spec=Mesh)
        mesh2.mesh = Mock(spec=trimesh.Trimesh)
        
        with patch('trimesh.util.concatenate') as mock_concatenate:
            # Set up mock return value
            mock_concatenate.return_value = self.mock_trimesh
            
            # Call the method
            result = self.mesh_service.combine_meshes([mesh1, mesh2])
            
            # Check that concatenate was called
            mock_concatenate.assert_called_once()
            
            # Check that the result is a Mesh
            self.assertIsInstance(result, Mesh)
    
    def test_combine_meshes_single(self) -> None:
        """Test combining a single mesh."""
        # Call the method
        result = self.mesh_service.combine_meshes([self.test_mesh])
        
        # Check that the result is the same mesh
        self.assertEqual(result, self.test_mesh)
    
    def test_combine_meshes_empty(self) -> None:
        """Test combining no meshes."""
        # Call the method and check for exception
        with self.assertRaises(MeshServiceError):
            self.mesh_service.combine_meshes([])
    
    @patch('heightcraft.services.sampling_service.SamplingService.sample_from_mesh')
    @patch('heightcraft.domain.point_cloud.PointCloud._validate_points')  # Skip validation
    def test_mesh_to_point_cloud(self, mock_validate_points, mock_sample_from_mesh) -> None:
        """Test converting a mesh to a point cloud."""
        # Create points for testing
        test_points = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
        
        # Set up mock to return test points 
        mock_sample_from_mesh.return_value = test_points

        # Call the method
        result = self.mesh_service.mesh_to_point_cloud(self.test_mesh, num_points=1000)

        # Verify the result
        self.assertIsInstance(result, PointCloud)
        mock_sample_from_mesh.assert_called_once_with(self.test_mesh)
    
    def test_subdivide_mesh(self) -> None:
        """Test subdividing a mesh."""
        # Set up mocks
        self.mock_trimesh.edges_unique = np.array([[0, 1], [1, 2]])
        self.mock_trimesh.edges_unique_length = np.array([1.0, 2.0])
        self.mock_trimesh.subdivide.return_value = self.mock_trimesh
        
        # Call the method
        result = self.mesh_service.subdivide_mesh(self.test_mesh, 0.5)
        
        # Check that subdivide was called
        self.mock_trimesh.subdivide.assert_called_once()
        
        # Check that the result is a Mesh
        self.assertIsInstance(result, Mesh)
    
    def test_subdivide_mesh_no_subdivision_needed(self) -> None:
        """Test subdividing a mesh that doesn't need subdivision."""
        # Set up mocks
        self.mock_trimesh.edges_unique = np.array([[0, 1], [1, 2]])
        self.mock_trimesh.edges_unique_length = np.array([0.1, 0.2])
        
        # Call the method
        result = self.mesh_service.subdivide_mesh(self.test_mesh, 0.5)
        
        # Check that subdivide was not called
        self.mock_trimesh.subdivide.assert_not_called()
        
        # Check that the result is the same mesh
        self.assertEqual(result, self.test_mesh) 