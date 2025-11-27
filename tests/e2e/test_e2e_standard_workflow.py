"""End-to-end tests for standard processing workflow."""
import os
import tempfile
import unittest
from pathlib import Path

import trimesh
from PIL import Image

from heightcraft.main import main as heightcraft_main


class TestE2EStandardWorkflow(unittest.TestCase):
    """End-to-end tests for standard height map generation workflow."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)
        
        # Create a simple test mesh
        self.mesh_file = self.test_dir / "test_cube.obj"
        mesh = trimesh.creation.box(extents=(10, 10, 5))
        mesh.export(str(self.mesh_file))
        
    def tearDown(self):
        """Tear down test fixtures."""
        self.temp_dir.cleanup()
        
    def test_basic_height_map_generation(self):
        """Test basic height map generation with default settings."""
        output_file = self.test_dir / "output.png"
        
        # Run heightcraft
        args = [str(self.mesh_file), "--output_path", str(output_file)]
        exit_code = heightcraft_main(args)
        
        # Verify success
        self.assertEqual(exit_code, 0)
        self.assertTrue(output_file.exists())
        
        # Verify output is a valid image
        img = Image.open(output_file)
        self.assertIsNotNone(img)
        self.assertGreater(img.width, 0)
        self.assertGreater(img.height, 0)
        
    def test_8bit_vs_16bit_output(self):
        """Test that 8-bit and 16-bit outputs are generated correctly."""
        output_8bit = self.test_dir / "output_8bit.png"
        output_16bit = self.test_dir / "output_16bit.png"
        
        # Generate 8-bit height map
        args_8bit = [
            str(self.mesh_file),
            "--output_path", str(output_8bit),
            "--bit_depth", "8"
        ]
        exit_code = heightcraft_main(args_8bit)
        self.assertEqual(exit_code, 0)
        self.assertTrue(output_8bit.exists())
        
        # Generate 16-bit height map
        args_16bit = [
            str(self.mesh_file),
            "--output_path", str(output_16bit),
            "--bit_depth", "16"
        ]
        exit_code = heightcraft_main(args_16bit)
        self.assertEqual(exit_code, 0)
        self.assertTrue(output_16bit.exists())
        
        # Verify images can be loaded
        img_8bit = Image.open(output_8bit)
        img_16bit = Image.open(output_16bit)
        
        self.assertIsNotNone(img_8bit)
        self.assertIsNotNone(img_16bit)
        
    def test_custom_resolution(self):
        """Test height map generation with custom resolution."""
        output_file = self.test_dir / "output_512.png"
        
        # Run with custom resolution
        args = [
            str(self.mesh_file),
            "--output_path", str(output_file),
            "--max_resolution", "512"
        ]
        exit_code = heightcraft_main(args)
        
        # Verify success
        self.assertEqual(exit_code, 0)
        self.assertTrue(output_file.exists())
        
        # Verify resolution
        img = Image.open(output_file)
        # Resolution should be <= 512 (smart resolution calculation may produce smaller)
        self.assertLessEqual(max(img.width, img.height), 512)
        
    def test_custom_sampling(self):
        """Test height map generation with custom sampling parameters."""
        output_file = self.test_dir / "output_sampled.png"
        
        # Run with custom sampling
        args = [
            str(self.mesh_file),
            "--output_path", str(output_file),
            "--num_samples", "50000",
            "--num_threads", "2"
        ]
        exit_code = heightcraft_main(args)
        
        # Verify success
        self.assertEqual(exit_code, 0)
        self.assertTrue(output_file.exists())
