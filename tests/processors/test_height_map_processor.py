"""
Tests for the HeightMapProcessor.
"""

import os
import unittest
from unittest.mock import Mock, patch, MagicMock

import numpy as np
import trimesh

from heightcraft.core.exceptions import ProcessingError
from heightcraft.domain.mesh import Mesh
from heightcraft.domain.height_map import HeightMap
from heightcraft.processors.height_map_processor import HeightMapProcessor, HeightMapConfig
from tests.base_test_case import BaseTestCase


class TestHeightMapProcessor(BaseTestCase):
    """Tests for the HeightMapProcessor."""
    
    def setUp(self) -> None:
        """Set up test environment."""
        super().setUp()
        
        # Create mock services
        self.mesh_service = Mock()
        self.height_map_service = Mock()
        
        # Create an instance of HeightMapProcessor
        self.processor = HeightMapProcessor(self.mesh_service, self.height_map_service)
        
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
        
        # Create a test height map
        self.data = np.array([
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8]
        ], dtype=np.float32)
        self.bit_depth = 8
        self.test_height_map = HeightMap(self.data, self.bit_depth)
        
        # Default height map config
        self.config = HeightMapConfig(
            resolution=0.1,
            width=256,
            height=256,
            normalize=True,
            bit_depth=16,
            flip_y=False
        )
        
        # Test file paths
        self.test_mesh_file = self.get_temp_path("test_height_map_model.obj")
        self.test_height_map_file = self.get_temp_path("test_height_map.png")
    
    def test_generate_height_map(self) -> None:
        """Test generating a height map from a mesh."""
        # Set up mocks
        self.mesh_service.convert_mesh_to_height_map.return_value = self.test_height_map
        self.height_map_service.normalize_height_map.return_value = self.test_height_map
        self.height_map_service.resize_height_map.return_value = self.test_height_map
        
        # Call the method
        height_map = self.processor.generate_height_map(self.test_mesh, self.config)
        
        # Check that the appropriate service methods were called
        self.mesh_service.convert_mesh_to_height_map.assert_called_once_with(self.test_mesh, self.config.resolution)
        self.height_map_service.normalize_height_map.assert_called_once()
        
        # Check the returned height map
        self.assertEqual(height_map, self.test_height_map)
    
    def test_generate_height_map_with_non_uniform_mesh(self) -> None:
        """Test generating a height map from a non-uniform mesh."""
        # Set up mocks
        self.mesh_service.center_mesh.return_value = self.non_uniform_mesh
        self.mesh_service.align_mesh_to_xy.return_value = self.non_uniform_mesh
        self.mesh_service.convert_mesh_to_height_map.return_value = self.test_height_map
        self.height_map_service.normalize_height_map.return_value = self.test_height_map
        self.height_map_service.resize_height_map.return_value = self.test_height_map
        
        # Call the method with center and align options
        config = HeightMapConfig(
            resolution=0.1,
            width=256,
            height=256,
            normalize=True,
            bit_depth=16,
            center=True,
            align=True
        )
        height_map = self.processor.generate_height_map(self.non_uniform_mesh, config)
        
        # Check that the appropriate service methods were called
        self.mesh_service.center_mesh.assert_called_once_with(self.non_uniform_mesh)
        self.mesh_service.align_mesh_to_xy.assert_called_once_with(self.non_uniform_mesh)
        self.mesh_service.convert_mesh_to_height_map.assert_called_once()
        
        # Check the returned height map
        self.assertEqual(height_map, self.test_height_map)
    
    def test_generate_height_map_8bit(self) -> None:
        """Test generating an 8-bit height map."""
        # Set up mocks
        self.mesh_service.convert_mesh_to_height_map.return_value = self.test_height_map
        self.height_map_service.normalize_height_map.return_value = self.test_height_map
        
        # Create mock for resize_height_map
        mock_resized = Mock(spec=HeightMap)
        self.height_map_service.resize_height_map.return_value = mock_resized

        # Create 8-bit config
        config = HeightMapConfig(
            resolution=0.1,
            width=256,
            height=256,
            normalize=True,
            bit_depth=8
        )

        # Call the method
        height_map = self.processor.generate_height_map(self.test_mesh, config)

        # Check that the expected methods were called
        self.mesh_service.convert_mesh_to_height_map.assert_called_once()
        self.height_map_service.normalize_height_map.assert_called_once()
        self.height_map_service.resize_height_map.assert_called_once()
        
        # Check that we got the expected result (the mock resized height map)
        self.assertIs(height_map, mock_resized)
    
    def test_generate_invalid_bit_depth(self) -> None:
        """Test generating a height map with invalid bit depth."""
        # Create invalid config
        config = HeightMapConfig(
            resolution=0.1,
            width=256,
            height=256,
            normalize=True,
            bit_depth=12  # Invalid, should be 8 or 16
        )
        
        # Call the method
        with self.assertRaises(ProcessingError):
            self.processor.generate_height_map(self.test_mesh, config)
    
    def test_save_height_map(self) -> None:
        """Test saving a height map to a file."""
        # Set up mock
        self.height_map_service.save_height_map.return_value = True
        
        # Call the method
        result = self.processor.save_height_map(self.test_height_map, self.test_height_map_file)
        
        # Check that the height map service's save method was called
        self.height_map_service.save_height_map.assert_called_once_with(self.test_height_map, self.test_height_map_file)
        
        # Check the result
        self.assertTrue(result)
        
        # Test with service error
        self.height_map_service.save_height_map.side_effect = Exception("Save error")
        with self.assertRaises(ProcessingError):
            self.processor.save_height_map(self.test_height_map, self.test_height_map_file)
    
    def test_save_split_height_maps(self) -> None:
        """Test saving split height maps."""
        # Create test height maps for splitting
        height_maps = {
            "top": self.test_height_map,
            "front": self.test_height_map,
            "side": self.test_height_map
        }
        
        # Set up mock
        self.height_map_service.save_height_map.return_value = True
        
        # Call the method
        result = self.processor.save_split_height_maps(height_maps, self.get_temp_path(""))
        
        # Check that the height map service's save method was called for each height map
        self.assertEqual(self.height_map_service.save_height_map.call_count, 3)
        
        # Check the result
        self.assertTrue(result)
    
    def test_cleanup(self) -> None:
        """Test cleanup method."""
        # Create a temporary file
        tmp_file = self.get_temp_path("tmp_file.txt")
        with open(tmp_file, "w") as f:
            f.write("test")
        
        # Add file to processor's temp files
        self.processor._temp_files.append(tmp_file)
        
        # Call cleanup
        self.processor.cleanup()
        
        # Check that the file was removed
        self.assertFalse(os.path.exists(tmp_file))
        
        # Check that temp_files list is empty
        self.assertEqual(len(self.processor._temp_files), 0)
    
    def test_process_mesh_file_to_height_map(self) -> None:
        """Test processing a mesh file to a height map."""
        # Set up mocks
        self.mesh_service.load_mesh.return_value = self.test_mesh
        self.mesh_service.center_mesh.return_value = self.test_mesh
        self.mesh_service.align_mesh_to_xy.return_value = self.test_mesh
        self.mesh_service.convert_mesh_to_height_map.return_value = self.test_height_map
        self.height_map_service.normalize_height_map.return_value = self.test_height_map
        self.height_map_service.save_height_map.return_value = True
        
        # Call the method
        result = self.processor.process_mesh_file_to_height_map(
            self.test_mesh_file,
            self.test_height_map_file,
            config=self.config
        )
        
        # Check the result
        self.assertTrue(result)
        
        # Check that the mesh service's methods were called
        self.mesh_service.load_mesh.assert_called_once_with(self.test_mesh_file)
        
        # Check that the height map was saved
        self.height_map_service.save_height_map.assert_called_once()
    
    def test_process_mesh_file_to_multiple_views(self) -> None:
        """Test processing a mesh file to multiple height map views."""
        # Set up mocks
        self.mesh_service.load_mesh.return_value = self.test_mesh
        self.mesh_service.center_mesh.return_value = self.test_mesh
        self.mesh_service.convert_mesh_to_height_map.return_value = self.test_height_map
        self.height_map_service.normalize_height_map.return_value = self.test_height_map
        self.height_map_service.save_height_map.return_value = True
        
        # Call the method for multiple views
        result = self.processor.process_mesh_file_to_multiple_views(
            self.test_mesh_file,
            self.get_temp_path(""),
            views=["top", "front", "side"],
            config=self.config
        )
        
        # Check the result
        self.assertTrue(result)
        
        # Check that the mesh service's load method was called
        self.mesh_service.load_mesh.assert_called_once_with(self.test_mesh_file)
        
        # Check that height maps were saved for each view (3 views)
        self.assertEqual(self.height_map_service.save_height_map.call_count, 3) 