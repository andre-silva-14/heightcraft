"""
Tests for the LargeModelProcessor.
"""

import os
import unittest
from unittest.mock import Mock, patch, MagicMock, PropertyMock

import numpy as np
import trimesh

from heightcraft.core.config import ApplicationConfig
from heightcraft.core.exceptions import ProcessingError
from heightcraft.domain.mesh import Mesh
from heightcraft.domain.point_cloud import PointCloud
from heightcraft.processors.large_model_processor import LargeModelProcessor
from heightcraft.services.mesh_service import MeshService
from heightcraft.services.model_service import ModelService
from heightcraft.services.height_map_service import HeightMapService
from heightcraft.services.sampling_service import SamplingService
from tests.base_test_case import BaseTestCase


class TestLargeModelProcessor(BaseTestCase):
    """Tests for the LargeModelProcessor."""
    
    def setUp(self) -> None:
        """Set up test environment."""
        super().setUp()
        
        # Create mock services
        self.mesh_service = Mock()
        self.model_service = Mock()
        self.height_map_service = Mock()
        self.sampling_service = Mock()
        
        # Create a mock config
        self.config = Mock()
        # Add the get_service method to the config mock
        self.config.get_service = Mock(side_effect=lambda service_class: {
            MeshService: self.mesh_service,
            ModelService: self.model_service,
            HeightMapService: self.height_map_service,
            SamplingService: self.sampling_service
        }.get(service_class))
        
        # Add model_config with chunk_size and cache_dir
        self.config.model_config = Mock()
        self.config.model_config.chunk_size = 1000
        self.config.model_config.cache_dir = self.get_temp_path("cache")
        
        # Add other config properties needed by the processor
        self.config.sampling_config = Mock()
        self.config.height_map_config = Mock()
        self.config.output_config = Mock()
        self.config.upscale_config = Mock()
        
        # Create an instance of LargeModelProcessor with the mock config
        self.processor = LargeModelProcessor(self.config)
        
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
            [0, 0, 0], [2, 0, 0], [2, 0, 1], [0, 0, 1],  # Bottom face
            [0, 0.2, 0], [2, 0.2, 0], [2, 0.2, 1], [0, 0.2, 1],  # Top face
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
        
        # Test file paths
        self.test_mesh_file = self.get_temp_path("test_large_model.obj")
        
        # Save the test mesh to a file
        trimesh_obj.export(self.test_mesh_file)
    
    def test_load_model_info(self) -> None:
        """Test loading model information."""
        # Set up a proper mock mesh with property return values
        mock_mesh = Mock(spec=Mesh)
        # Configure property mocks
        type(mock_mesh).vertex_count = PropertyMock(return_value=4)
        type(mock_mesh).face_count = PropertyMock(return_value=4)
        type(mock_mesh).is_watertight = PropertyMock(return_value=True)
        type(mock_mesh).has_degenerate_faces = PropertyMock(return_value=False)
        type(mock_mesh).is_winding_consistent = PropertyMock(return_value=True)
        type(mock_mesh).bounds = PropertyMock(return_value=np.array([[0, 0, 0], [1, 1, 1]]))

        # Use the mock instead of self.test_mesh
        self.mesh_service.load_mesh.return_value = mock_mesh
        
        # Call the method
        model_info = self.processor.load_model_info(self.test_mesh_file)
        
        # Check that the mesh service's load method was called
        self.mesh_service.load_mesh.assert_called_once_with(self.test_mesh_file)
        
        # Check the returned info
        self.assertIsNotNone(model_info)
        self.assertEqual(model_info["vertex_count"], 4)
        self.assertEqual(model_info["face_count"], 4)
        self.assertTrue(model_info["is_watertight"])
        self.assertFalse(model_info["has_degenerate_faces"])
        self.assertTrue(model_info["is_winding_consistent"])
        self.assertIn("bounds", model_info)
        self.assertEqual(model_info["file_path"], self.test_mesh_file)
        
        # Test with repository error
        self.mesh_service.load_mesh.side_effect = Exception("Repository error")
        with self.assertRaises(ProcessingError):
            self.processor.load_model_info(self.test_mesh_file)
    
    def test_load_model_info_with_non_uniform_mesh(self) -> None:
        """Test loading information for a non-uniform mesh."""
        # Set up a proper mock mesh with property return values
        mock_mesh = Mock(spec=Mesh)
        # Configure property mocks
        type(mock_mesh).vertex_count = PropertyMock(return_value=8)
        type(mock_mesh).face_count = PropertyMock(return_value=12)
        type(mock_mesh).is_watertight = PropertyMock(return_value=True)
        type(mock_mesh).has_degenerate_faces = PropertyMock(return_value=False)
        type(mock_mesh).is_winding_consistent = PropertyMock(return_value=True)
        type(mock_mesh).bounds = PropertyMock(return_value=np.array([[0, 0, 0], [2, 0.2, 1]]))

        # Use the mock instead of self.non_uniform_mesh
        self.mesh_service.load_mesh.return_value = mock_mesh
        
        # Call the method
        model_info = self.processor.load_model_info(self.test_mesh_file)
        
        # Check the returned info
        self.assertIsNotNone(model_info)
        self.assertEqual(model_info["vertex_count"], 8)
        self.assertEqual(model_info["face_count"], 12)
        self.assertTrue(model_info["is_watertight"])
        self.assertFalse(model_info["has_degenerate_faces"])
        self.assertTrue(model_info["is_winding_consistent"])
        
        # Check the bounds
        np.testing.assert_array_equal(model_info["bounds"], np.array([[0, 0, 0], [2, 0.2, 1]]))
    
    def test_center_model(self) -> None:
        """Test centering a model."""
        # Set up mock
        self.mesh_service.center_mesh.return_value = self.test_mesh
        
        # Call the method
        centered_mesh = self.processor.center_model(self.test_mesh)
        
        # Check that the mesh service's center method was called
        self.mesh_service.center_mesh.assert_called_once_with(self.test_mesh)
        
        # Check the returned mesh
        self.assertEqual(centered_mesh, self.test_mesh)
    
    def test_align_model(self) -> None:
        """Test aligning a model."""
        # Set up mock
        self.mesh_service.align_mesh_to_xy.return_value = self.test_mesh
        
        # Call the method
        aligned_mesh = self.processor.align_model(self.test_mesh)
        
        # Check that the mesh service's align method was called
        self.mesh_service.align_mesh_to_xy.assert_called_once_with(self.test_mesh)
        
        # Check the returned mesh
        self.assertEqual(aligned_mesh, self.test_mesh)
    
    def test_sample_points(self) -> None:
        """Test sampling points from a model."""
        # Create a mock point cloud
        points = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 1]
        ])
        mock_point_cloud = PointCloud(points)
        
        # Set up mock
        self.mesh_service.mesh_to_point_cloud.return_value = mock_point_cloud
        
        # Call the method
        point_cloud = self.processor.sample_points(self.test_mesh, 5)
        
        # Check that the mesh service's mesh_to_point_cloud method was called
        self.mesh_service.mesh_to_point_cloud.assert_called_once_with(self.test_mesh, 5)
        
        # Check the returned point cloud
        self.assertEqual(point_cloud, mock_point_cloud)
    
    def test_cleanup(self) -> None:
        """Test cleanup method."""
        # Create a temporary file
        tmp_file = self.get_temp_path("tmp_file.txt")
        with open(tmp_file, "w") as f:
            f.write("test")
        
        # Create a new list for temp files and replace the processor's list with it
        temp_files = [tmp_file]
        
        # Use patch to replace the _temp_files attribute
        with patch.object(self.processor, '_temp_files', temp_files):
            # Make sure there's one file in the temp_files list
            self.assertEqual(len(self.processor._temp_files), 1)
            
            # Mock the vertex_buffer and chunks attributes
            self.processor.vertex_buffer = [np.array([[0, 0, 0]])]
            self.processor.chunks = [{"vertices": 0, "faces": np.array([[0, 1, 2]])}]
            
            # Call cleanup
            self.processor.cleanup()
            
            # Check that vertex_buffer and chunks are cleared
            self.assertIsNone(self.processor.vertex_buffer)
            self.assertEqual(len(self.processor.chunks), 0)
            
            # Note: LargeModelProcessor doesn't clear _temp_files in its cleanup method
            # It only clears vertex_buffer and chunks
    
    def test_context_manager(self) -> None:
        """Test using the processor as a context manager."""
        # Create a temporary file
        tmp_file = self.get_temp_path("context_file.txt")
        with open(tmp_file, "w") as f:
            f.write("test")
        
        # Set up for context manager
        with patch.object(self.processor, 'cleanup') as mock_cleanup:
            with self.processor:
                # Add file to processor's temp files
                self.processor._temp_files.append(tmp_file)
            
            # Check that cleanup was called
            mock_cleanup.assert_called_once()
    
    def test_invalid_file(self) -> None:
        """Test handling of invalid file."""
        # Set up mock to raise an exception
        self.mesh_service.load_mesh.side_effect = Exception("Invalid file")
        
        # Call the method
        with self.assertRaises(ProcessingError):
            self.processor.load_model_info("invalid_file.obj")
    
    def test_calculate_bounding_box(self) -> None:
        """Test calculating the bounding box of a model."""
        # Set up mock bounds return value
        bounds_return = np.array([[0, 0, 0], [1, 1, 1]])
        self.mesh_service.get_bounds.return_value = bounds_return
        
        # Call the method
        bounds = self.processor.calculate_bounding_box(self.test_mesh)
        
        # Check the returned bounds
        self.assertIsNotNone(bounds)
        np.testing.assert_array_equal(bounds, bounds_return)
        self.mesh_service.get_bounds.assert_called_once_with(self.test_mesh)
    
    def test_calculate_bounding_box_after_alignment(self) -> None:
        """Test calculating the bounding box after alignment."""
        # Set up mocks
        self.mesh_service.align_mesh_to_xy.return_value = self.non_uniform_mesh
        bounds_return = np.array([[0, 0, 0], [2, 0.2, 1]])
        self.mesh_service.get_bounds.return_value = bounds_return
        
        # First align the mesh
        aligned_mesh = self.processor.align_model(self.test_mesh)
        
        # Then calculate the bounding box
        bounds = self.processor.calculate_bounding_box(aligned_mesh)
        
        # Check the returned bounds
        self.assertIsNotNone(bounds)
        np.testing.assert_array_equal(bounds, bounds_return)
        self.mesh_service.get_bounds.assert_called_once_with(self.non_uniform_mesh) 