"""
Tests for HeightMap coverage improvement.
"""

import unittest
import os
import tempfile
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from heightcraft.domain.height_map import HeightMap
from heightcraft.domain.point_cloud import PointCloud
from heightcraft.core.config import OutputFormat
from heightcraft.core.exceptions import HeightMapGenerationError, HeightMapValidationError
from tests.base_test_case import BaseTestCase

class TestHeightMapCoverage(BaseTestCase):
    """Tests for HeightMap coverage."""

    def setUp(self):
        super().setUp()
        self.data = np.array([[0.0, 0.5], [0.5, 1.0]], dtype=np.float32)
        self.height_map = HeightMap(self.data, bit_depth=16)

    def test_save_formats(self):
        # Test saving with different formats
        with tempfile.TemporaryDirectory() as tmpdir:
            # PNG
            png_path = os.path.join(tmpdir, "test.png")
            with patch('PIL.Image.fromarray') as mock_pil:
                mock_img = Mock()
                mock_pil.return_value = mock_img
                self.height_map.save(png_path, format=OutputFormat.PNG)
                mock_img.save.assert_called_with(png_path)

            # JPEG
            jpg_path = os.path.join(tmpdir, "test.jpg")
            with patch('PIL.Image.fromarray') as mock_pil:
                mock_img = Mock()
                mock_pil.return_value = mock_img
                self.height_map.save(jpg_path, format=OutputFormat.JPEG)
                mock_img.save.assert_called_with(jpg_path)

            # TIFF 32-bit
            tiff_path = os.path.join(tmpdir, "test.tiff")
            hm_32 = HeightMap(self.data, bit_depth=32)
            with patch.dict('sys.modules', {'tifffile': Mock()}):
                import sys
                mock_tifffile = sys.modules['tifffile']
                hm_32.save(tiff_path, format=OutputFormat.TIFF)
                mock_tifffile.imwrite.assert_called()

            # RAW
            raw_path = os.path.join(tmpdir, "test.raw")
            self.height_map.save(raw_path, format=OutputFormat.RAW)
            self.assertTrue(os.path.exists(raw_path))

    def test_save_bit_depths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # 8-bit
            hm_8 = HeightMap(self.data, bit_depth=8)
            path_8 = os.path.join(tmpdir, "test_8.png")
            with patch('PIL.Image.fromarray') as mock_pil:
                mock_img = Mock()
                mock_pil.return_value = mock_img
                hm_8.save(path_8)
                # Verify mode 'L' for 8-bit
                mock_pil.assert_called_with(unittest.mock.ANY, mode='L')

            # 32-bit as PNG (fallback)
            hm_32 = HeightMap(self.data, bit_depth=32)
            path_32 = os.path.join(tmpdir, "test_32.png")
            with patch('PIL.Image.fromarray') as mock_pil:
                mock_img = Mock()
                mock_pil.return_value = mock_img
                with self.assertWarns(UserWarning):
                    hm_32.save(path_32, format=OutputFormat.PNG)

    def test_split(self):
        # 4x4 map
        data = np.zeros((4, 4))
        hm = HeightMap(data)
        
        # Split into 4 (2x2 grid)
        # The split method takes grid_size as number of tiles per side? 
        # Let's check implementation. split(grid_size) -> grid_size*grid_size tiles?
        # Or grid_size is total tiles?
        # Looking at implementation: split(grid_size) loops i in range(grid_size), j in range(grid_size)
        # So it produces grid_size * grid_size tiles.
        
        tiles = hm.split(2) # 2x2 = 4 tiles
        self.assertEqual(len(tiles), 4)
        self.assertEqual(tiles[0].width, 2)
        self.assertEqual(tiles[0].height, 2)
        
        # Split into 1
        tiles = hm.split(1)
        self.assertEqual(len(tiles), 1)
        self.assertEqual(tiles[0], hm)
        
        # Invalid split
        with self.assertRaises(HeightMapGenerationError):
            hm.split(0)

    def test_convert_bit_depth(self):
        # 8 -> 8
        # Ensure input is already uint8 for 8-bit map to test identity return
        data_8 = (self.data * 255).astype(np.uint8)
        hm_8 = HeightMap(data_8, bit_depth=8)
        self.assertIs(hm_8.convert_to_8bit(), hm_8)
        
        # 16 -> 16
        data_16 = (self.data * 65535).astype(np.uint16)
        hm_16 = HeightMap(data_16, bit_depth=16)
        self.assertIs(hm_16.convert_to_16bit(), hm_16)
        
        # 16 -> 8
        hm_16_to_8 = hm_16.convert_to_8bit()
        self.assertEqual(hm_16_to_8.bit_depth, 8)
        # HeightMap stores data as normalized floats internally
        self.assertTrue(np.issubdtype(hm_16_to_8.data.dtype, np.floating))
        
        # 8 -> 16
        hm_8_to_16 = hm_8.convert_to_16bit()
        self.assertEqual(hm_8_to_16.bit_depth, 16)
        self.assertTrue(np.issubdtype(hm_8_to_16.data.dtype, np.floating))

    def test_from_point_cloud(self):
        mock_pc = Mock(spec=PointCloud)
        # 4 points forming a square
        mock_pc.get_xy_points.return_value = np.array([[0,0], [1,0], [0,1], [1,1]])
        mock_pc.normalize_z.return_value = np.array([0.0, 0.5, 0.5, 1.0])
        mock_pc.bounds = {"min_x": 0, "max_x": 1, "min_y": 0, "max_y": 1}
        
        # 8-bit
        hm_8 = HeightMap.from_point_cloud(mock_pc, (2, 2), bit_depth=8)
        self.assertEqual(hm_8.bit_depth, 8)
        self.assertTrue(np.issubdtype(hm_8.data.dtype, np.floating))
        
        # 16-bit
        hm_16 = HeightMap.from_point_cloud(mock_pc, (2, 2), bit_depth=16)
        self.assertEqual(hm_16.bit_depth, 16)
        self.assertTrue(np.issubdtype(hm_16.data.dtype, np.floating))
        
        # 32-bit
        hm_32 = HeightMap.from_point_cloud(mock_pc, (2, 2), bit_depth=32)
        self.assertEqual(hm_32.bit_depth, 32)
        self.assertTrue(np.issubdtype(hm_32.data.dtype, np.floating))

    def test_save_error(self):
        with patch('os.makedirs', side_effect=Exception("Dir error")):
            with self.assertRaises(HeightMapGenerationError):
                self.height_map.save("some/path/test.png")

    def test_to_mesh_error(self):
         # Mock trimesh.Trimesh to raise exception
         # We need to patch where it is used or imported
         with patch('trimesh.Trimesh', side_effect=Exception("Mesh error")):
             with self.assertRaises(HeightMapGenerationError):
                 self.height_map.to_mesh()

    def test_to_point_cloud_error(self):
        # Force error by making shape invalid for mgrid or similar?
        # Or mock numpy.mgrid
        with patch('numpy.mgrid', side_effect=Exception("PC error")):
            with self.assertRaises(HeightMapGenerationError):
                self.height_map.to_point_cloud()

    def test_resize_error(self):
        with patch('scipy.ndimage.zoom', side_effect=Exception("Resize error")):
            with self.assertRaises(HeightMapGenerationError):
                self.height_map.resize((10, 10))

    def test_crop_error(self):
        # Invalid crop causing index error
        with self.assertRaises(HeightMapGenerationError):
            # Pass invalid types to cause exception in slicing
            self.height_map.crop(None, None)
