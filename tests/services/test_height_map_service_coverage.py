"""
Tests for HeightMapService coverage improvement.
"""

import unittest
import os
import tempfile
import numpy as np
from unittest.mock import Mock, patch, MagicMock, PropertyMock

from heightcraft.services.height_map_service import HeightMapService
from heightcraft.domain.height_map import HeightMap
from heightcraft.domain.point_cloud import PointCloud
from heightcraft.core.exceptions import HeightMapServiceError
from tests.base_test_case import BaseTestCase

class TestHeightMapServiceCoverage(BaseTestCase):
    """Tests for HeightMapService coverage."""

    def setUp(self):
        super().setUp()
        self.service = HeightMapService()
        self.data = np.array([[0.0, 0.5], [0.5, 1.0]], dtype=np.float32)
        self.height_map = HeightMap(self.data, bit_depth=16)

    def test_split_height_map(self):
        # 4x4 map
        hm = HeightMap(np.zeros((4, 4)), bit_depth=16)
        tiles = self.service.split_height_map(hm, 4)
        self.assertEqual(len(tiles), 4)
        
        with self.assertRaises(HeightMapServiceError):
            self.service.split_height_map(hm, 3) # Not a square

    def test_save_split_height_maps(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test.png")
            tiles = [self.height_map] * 4
            
            # Mock save_height_map to avoid actual file I/O and just check logic
            with patch.object(self.service, 'save_height_map') as mock_save:
                output_dir = self.service.save_split_height_maps(tiles, output_path)
                self.assertEqual(mock_save.call_count, 4)
                self.assertTrue(output_dir.endswith("test_split"))

    def test_generate_slope_map(self):
        slope = self.service.generate_slope_map(self.height_map)
        self.assertIsInstance(slope, HeightMap)
        self.assertEqual(slope.bit_depth, 8)

    def test_generate_curvature_map(self):
        curv = self.service.generate_curvature_map(self.height_map)
        self.assertIsInstance(curv, HeightMap)
        self.assertEqual(curv.bit_depth, 8)

    def test_apply_sea_level(self):
        hm, mask = self.service.apply_sea_level(self.height_map, 0.25)
        self.assertIsInstance(hm, HeightMap)
        self.assertIsInstance(mask, HeightMap)
        # Check that values below 0.25 are set to 0.25
        self.assertTrue(np.all(hm.data >= 0.25))
        # Check mask
        self.assertEqual(mask.data[0, 0], 1.0) # 0.0 < 0.25
        self.assertEqual(mask.data[1, 1], 0.0) # 1.0 >= 0.25

    def test_generate_from_point_cloud_threading(self):
        mock_pc = Mock(spec=PointCloud)
        mock_pc.points = np.array([[0,0,0], [1,1,1]])
        mock_pc.bounds = {"min_x": 0, "max_x": 1, "min_y": 0, "max_y": 1, "min_z": 0, "max_z": 1}
        
        # Test with multiple threads
        hm = self.service.generate_from_point_cloud(mock_pc, (2, 2), num_threads=2)
        self.assertIsInstance(hm, HeightMap)
        self.assertEqual(hm.width, 2)
        self.assertEqual(hm.height, 2)

    def test_generate_from_point_cloud_empty(self):
        mock_pc = Mock(spec=PointCloud)
        mock_pc.points = np.array([])
        with self.assertRaises(HeightMapServiceError):
            self.service.generate_from_point_cloud(mock_pc, (10, 10))

    def test_update_height_map_buffer_zero_range(self):
        # Test handling of zero ranges in bounds
        buffer = np.zeros((2, 2))
        points = np.array([[0,0,0]])
        bounds = {"min_x": 0, "max_x": 0, "min_y": 0, "max_y": 0, "min_z": 0, "max_z": 0}
        
        self.service.update_height_map_buffer(buffer, points, bounds, 2, 2)
        # Should not crash

    def test_error_handling(self):
        # Test various error paths
        # We need to ensure the mocks are actually called and raise exceptions
        
        # Mock numpy.gradient for slope map error
        with patch('numpy.gradient', side_effect=Exception("Error")):
            with self.assertRaises(HeightMapServiceError):
                self.service.generate_slope_map(self.height_map)
            
        # Mock ndimage.laplace for curvature map error
        with patch('scipy.ndimage.laplace', side_effect=Exception("Error")):
            with self.assertRaises(HeightMapServiceError):
                self.service.generate_curvature_map(self.height_map)
            
        # For apply_sea_level, we can pass an invalid height map (e.g. data property raises)
        mock_hm = Mock()
        type(mock_hm).data = PropertyMock(side_effect=Exception("Error"))
        with self.assertRaises(HeightMapServiceError):
            self.service.apply_sea_level(mock_hm, 0)
            
        # For save_split_height_maps, mock os.makedirs
        with patch('os.makedirs', side_effect=Exception("Error")):
            with self.assertRaises(HeightMapServiceError):
                self.service.save_split_height_maps([self.height_map], "path/test.png")
