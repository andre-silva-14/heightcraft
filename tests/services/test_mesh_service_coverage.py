"""
Tests for MeshService coverage improvement.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import trimesh

from heightcraft.services.mesh_service import MeshService
from heightcraft.domain.mesh import Mesh
from heightcraft.domain.point_cloud import PointCloud
from heightcraft.domain.height_map import HeightMap
from heightcraft.core.exceptions import MeshServiceError
from tests.base_test_case import BaseTestCase

class TestMeshServiceCoverage(BaseTestCase):
    """Tests for MeshService coverage."""

    def setUp(self) -> None:
        super().setUp()
        self.mesh_service = MeshService()
        self.mock_mesh = Mock(spec=Mesh)
        self.mock_mesh.mesh = Mock(spec=trimesh.Trimesh)
        self.mock_mesh.bounds = np.array([[0, 0, 0], [1, 1, 1]])

    def test_prepare_mesh_error(self):
        self.mock_mesh.center.side_effect = Exception("Center error")
        with self.assertRaises(MeshServiceError):
            self.mesh_service.prepare_mesh(self.mock_mesh)

    def test_center_mesh_error(self):
        self.mock_mesh.center.side_effect = Exception("Center error")
        with self.assertRaises(MeshServiceError):
            self.mesh_service.center_mesh(self.mock_mesh)

    def test_align_mesh_to_xy_error(self):
        self.mock_mesh.align_to_xy.side_effect = Exception("Align error")
        with self.assertRaises(MeshServiceError):
            self.mesh_service.align_mesh_to_xy(self.mock_mesh)

    def test_get_bounds_error(self):
        # Accessing bounds property raises exception
        type(self.mock_mesh).bounds = PropertyMock(side_effect=Exception("Bounds error"))
        # We need to recreate the mock because we messed with the property
        mock_mesh_error = Mock(spec=Mesh)
        # Define bounds as a property that raises
        p = PropertyMock(side_effect=Exception("Bounds error"))
        type(mock_mesh_error).bounds = p
        
        with self.assertRaises(MeshServiceError):
            self.mesh_service.get_bounds(mock_mesh_error)

    def test_get_aspect_ratio_error(self):
        self.mock_mesh.get_aspect_ratio.side_effect = Exception("Aspect ratio error")
        with self.assertRaises(MeshServiceError):
            self.mesh_service.get_aspect_ratio(self.mock_mesh)

    def test_calculate_target_resolution_error(self):
        with patch('heightcraft.utils.resolution_calculator.ResolutionCalculator') as MockCalc:
            MockCalc.return_value.calculate_resolution_from_bounds.side_effect = Exception("Calc error")
            with self.assertRaises(MeshServiceError):
                self.mesh_service.calculate_target_resolution(self.mock_mesh, 100)

    def test_combine_meshes_error(self):
        with patch('trimesh.util.concatenate', side_effect=Exception("Combine error")):
            with self.assertRaises(MeshServiceError):
                self.mesh_service.combine_meshes([self.mock_mesh, self.mock_mesh])

    def test_mesh_to_point_cloud_error(self):
        with patch('heightcraft.services.sampling_service.SamplingService') as MockService:
            MockService.return_value.sample_from_mesh.side_effect = Exception("Sample error")
            with self.assertRaises(MeshServiceError):
                self.mesh_service.mesh_to_point_cloud(self.mock_mesh, 100)

    def test_subdivide_mesh_error(self):
        # Mock trimesh object behavior
        self.mock_mesh.mesh.edges_unique_length = np.array([10.0])
        self.mock_mesh.mesh.subdivide.side_effect = Exception("Subdivide error")
        
        with self.assertRaises(MeshServiceError):
            self.mesh_service.subdivide_mesh(self.mock_mesh, 1.0)

    @patch('heightcraft.services.mesh_service.ModelRepository')
    def test_save_mesh_success(self, MockRepo):
        MockRepo.return_value.save.return_value = True
        result = self.mesh_service.save_mesh(self.mock_mesh, "test.obj")
        self.assertTrue(result)
        MockRepo.return_value.save.assert_called_once_with(self.mock_mesh, "test.obj")

    @patch('heightcraft.services.mesh_service.ModelRepository')
    def test_save_mesh_error(self, MockRepo):
        MockRepo.return_value.save.side_effect = Exception("Save error")
        with self.assertRaises(MeshServiceError):
            self.mesh_service.save_mesh(self.mock_mesh, "test.obj")

    @patch('heightcraft.services.mesh_service.ModelRepository')
    def test_load_mesh_success(self, MockRepo):
        MockRepo.return_value.load.return_value = self.mock_mesh
        result = self.mesh_service.load_mesh("test.obj")
        self.assertEqual(result, self.mock_mesh)
        MockRepo.return_value.load.assert_called_once_with("test.obj")

    @patch('heightcraft.services.mesh_service.ModelRepository')
    def test_load_mesh_error(self, MockRepo):
        MockRepo.return_value.load.side_effect = Exception("Load error")
        with self.assertRaises(MeshServiceError):
            self.mesh_service.load_mesh("test.obj")

    @patch('heightcraft.services.height_map_service.HeightMapService')
    @patch('heightcraft.services.mesh_service.SamplingService')
    def test_convert_mesh_to_height_map_success(self, MockSamplingService, MockHMService):
        # Setup mocks
        mock_points = np.array([[0,0,0], [1,1,1]])
        MockSamplingService.return_value.sample_from_mesh.return_value = mock_points
        
        mock_hm = Mock(spec=HeightMap)
        MockHMService.return_value.generate_from_point_cloud.return_value = mock_hm
        
        # Call method
        result = self.mesh_service.convert_mesh_to_height_map(self.mock_mesh, resolution=0.1)
        
        # Verify
        self.assertEqual(result, mock_hm)
        MockSamplingService.return_value.sample_from_mesh.assert_called_once()
        MockHMService.return_value.generate_from_point_cloud.assert_called_once()

    def test_convert_mesh_to_height_map_error(self):
        self.mock_mesh.center.side_effect = Exception("Prep error")
        with self.assertRaises(MeshServiceError):
            self.mesh_service.convert_mesh_to_height_map(self.mock_mesh, resolution=0.1)

from unittest.mock import PropertyMock
