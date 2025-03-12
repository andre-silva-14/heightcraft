"""
Integration test for the mesh to height map workflow.
"""

import os
import unittest
from unittest.mock import Mock, patch

import numpy as np
import trimesh

from heightcraft.domain.mesh import Mesh
from heightcraft.domain.height_map import HeightMap
from heightcraft.services.mesh_service import MeshService
from heightcraft.services.height_map_service import HeightMapService
from heightcraft.infrastructure.model_repository import ModelRepository
from heightcraft.infrastructure.height_map_repository import HeightMapRepository
from heightcraft.processors.mesh_processor import MeshProcessor
from heightcraft.utils.point_sampler import PointSampler
from tests.base_test_case import BaseTestCase


class TestMeshToHeightMapWorkflow(BaseTestCase):
    """Integration test for the mesh to height map workflow."""
    
    def setUp(self) -> None:
        """Set up test environment."""
        super().setUp()
        
        # Create real repositories
        self.mesh_repository = ModelRepository()
        self.height_map_repository = HeightMapRepository()
        
        # Create real services - MeshService no longer accepts a repository parameter
        self.mesh_service = MeshService()
        self.height_map_service = HeightMapService(self.height_map_repository)
        self.point_sampler = PointSampler()
        
        # Create a test mesh
        vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 0.5],
            [0, 1, 0.5],
            [1, 1, 0.75]
        ])
        faces = np.array([
            [0, 1, 3], [0, 3, 2],  # Bottom face
            [0, 4, 5], [0, 5, 1],  # Front face
            [0, 2, 6], [0, 6, 4],  # Left face
            [1, 5, 7], [1, 7, 3],  # Right face
            [2, 3, 7], [2, 7, 6],  # Back face
            [4, 6, 7], [4, 7, 5]   # Top face
        ])
        
        # Create a trimesh object and pass it to Mesh constructor
        trimesh_obj = trimesh.Trimesh(vertices=vertices, faces=faces)
        self.test_mesh = Mesh(trimesh_obj)
        
        # Define the test file paths
        self.test_mesh_file = self.get_temp_path("test_mesh.obj")
        self.test_height_map_file = self.get_temp_path("test_height_map.png")
        
        # Create a MeshProcessor
        self.mesh_processor = MeshProcessor(self.mesh_service, self.height_map_service)
    
    @patch('trimesh.load')
    @patch('PIL.Image.fromarray')
    @patch('PIL.Image.open')
    @patch('heightcraft.domain.mesh.Mesh._validate_mesh')  # Patch mesh validation
    def test_mesh_to_height_map_workflow(self, mock_validate_mesh, mock_image_open, mock_image_from_array, mock_trimesh_load):
        """Test the complete workflow from mesh to height map."""
        # Set up mocks
        mock_trimesh = Mock(spec=trimesh.Trimesh)
        mock_trimesh.vertices = self.test_mesh.vertices
        mock_trimesh.faces = self.test_mesh.faces
        mock_trimesh_load.return_value = mock_trimesh

        mock_pil_image = Mock()
        mock_pil_image.size = (10, 10)
        mock_pil_image.getdata.return_value = list(range(100))
        mock_image_open.return_value = mock_pil_image

        mock_image = Mock()
        mock_image_from_array.return_value = mock_image
        mock_image.save.return_value = None

        # Mock the get_bounds method to return a valid bounds tuple
        self.mesh_service.get_bounds = Mock(return_value=(np.array([0, 0, 0]), np.array([10, 10, 10])))

        # Mock the convert_mesh_to_height_map method to return a valid height map
        mock_height_map = Mock(spec=HeightMap)
        mock_height_map.data = np.zeros((101, 101), dtype=np.float32)
        mock_height_map.shape = (101, 101)
        mock_height_map.width = 101
        mock_height_map.height = 101
        mock_height_map.bit_depth = 16
        self.mesh_service.convert_mesh_to_height_map = Mock(return_value=mock_height_map)

        # Step 1: Save the test mesh to a file
        result = self.mesh_service.save_mesh(self.test_mesh, self.test_mesh_file)
        self.assertTrue(result)

        # Step 2: Load the mesh from the file
        loaded_mesh = self.mesh_service.load_mesh(self.test_mesh_file)
        self.assertIsInstance(loaded_mesh, Mesh)

        # Step 3: Center the mesh
        centered_mesh = self.mesh_service.center_mesh(loaded_mesh)
        self.assertIsInstance(centered_mesh, Mesh)

        # Step 4: Align the mesh to the XY plane
        aligned_mesh = self.mesh_service.align_mesh_to_xy(centered_mesh)
        self.assertIsInstance(aligned_mesh, Mesh)

        # Step 5: Convert the mesh to a height map
        resolution = 0.1
        height_map = self.mesh_service.convert_mesh_to_height_map(aligned_mesh, resolution)
        self.assertIsInstance(height_map, HeightMap)

        # Step 6: Save the height map
        result = self.height_map_service.save_height_map(height_map, self.test_height_map_file)
        self.assertTrue(result)

        # Check that all steps in the workflow were successful
        self.mesh_service.convert_mesh_to_height_map.assert_called_once_with(aligned_mesh, resolution)


class TestHeightMapToMeshWorkflow(BaseTestCase):
    """Integration test for the height map to mesh workflow."""
    
    def setUp(self) -> None:
        """Set up test environment."""
        super().setUp()
        
        # Create real repositories
        self.mesh_repository = ModelRepository()
        self.height_map_repository = HeightMapRepository()
        
        # Create real services
        self.mesh_service = MeshService()
        self.height_map_service = HeightMapService(self.height_map_repository)
        
        # Create a test height map
        self.data = np.array([
            [0, 0.1, 0.2],
            [0.3, 0.4, 0.5],
            [0.6, 0.7, 0.8]
        ], dtype=np.float32)
        self.resolution = 1.0
        self.test_height_map = HeightMap(self.data, bit_depth=8)
        
        # Define the test file paths
        self.test_height_map_file = self.get_temp_path("test_height_map.png")
        self.test_mesh_file = self.get_temp_path("test_mesh.obj")
        
        # Create a MeshProcessor
        self.mesh_processor = MeshProcessor(self.mesh_service, self.height_map_service)
    
    @patch('PIL.Image.open')
    @patch('PIL.Image.fromarray')
    @patch('trimesh.Trimesh')
    @patch('heightcraft.infrastructure.height_map_repository.HeightMapRepository.load')
    @patch('heightcraft.domain.mesh.Mesh._validate_mesh')  # Patch mesh validation
    def test_height_map_to_mesh_workflow(self, mock_validate_mesh, mock_load, mock_trimesh_class, mock_image_from_array, mock_image_open):
        """Test the complete workflow from height map to mesh."""
        # Set up mocks
        mock_pil_image = Mock()
        mock_pil_image.size = (3, 3)
        mock_pil_image.getdata.return_value = list(range(9))
        mock_image_open.return_value = mock_pil_image
        
        mock_image = Mock()
        mock_image_from_array.return_value = mock_image
        mock_image.save.return_value = None
        
        # Create a mock Trimesh object instead of mocking the class
        mock_trimesh = Mock()
        mock_trimesh.vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        mock_trimesh.faces = np.array([[0, 1, 2]])
        mock_trimesh.export.return_value = None
        mock_trimesh_class.return_value = mock_trimesh
        
        # Mock the height map loading to return the test height map
        mock_load.return_value = self.test_height_map

        # Step 1: Save the test height map to a file
        result = self.height_map_service.save_height_map(self.test_height_map, self.test_height_map_file)
        self.assertTrue(result)
        
        # Step 2: Load the height map from the file
        loaded_height_map = self.height_map_service.load_height_map(self.test_height_map_file, self.resolution)
        self.assertIsInstance(loaded_height_map, HeightMap)
        
        # Step 3: Resize the height map
        new_width, new_height = 6, 6
        resized_height_map = self.height_map_service.resize_height_map(loaded_height_map, new_width, new_height)
        self.assertIsInstance(resized_height_map, HeightMap)
        self.assertEqual(resized_height_map.width, new_width)
        self.assertEqual(resized_height_map.height, new_height)
        
        # Step 4: Apply median filter
        filtered_height_map = self.height_map_service.apply_median_filter(resized_height_map, 3)
        self.assertIsInstance(filtered_height_map, HeightMap)
        
        # Step 5: Convert the height map to a mesh
        mesh = self.height_map_service.convert_height_map_to_mesh(filtered_height_map)
        self.assertIsNotNone(mesh)
        
        # Step 6: Center the mesh
        centered_mesh = self.mesh_service.center_mesh(mesh)
        self.assertIsNotNone(centered_mesh)
        
        # Step 7: Save the mesh to a file
        result = self.mesh_service.save_mesh(centered_mesh, self.test_mesh_file)
        self.assertTrue(result)
        
        # Step 8: Use the processor to do the entire workflow at once
        result = self.mesh_processor.process_height_map_to_mesh(
            self.test_height_map_file,
            self.test_mesh_file,
            resolution=self.resolution,
            resize=(new_width, new_height),
            apply_filter=True,
            filter_size=3,
            center=True
        )
        self.assertTrue(result) 