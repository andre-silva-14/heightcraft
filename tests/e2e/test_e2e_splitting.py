"""End-to-end tests for splitting functionality."""
import os
import tempfile
import unittest
from pathlib import Path

import trimesh

from heightcraft.main import main as heightcraft_main


class TestE2ESplitting(unittest.TestCase):
    """End-to-end tests for height map splitting functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)
        
        # Create a test mesh
        self.mesh_file = self.test_dir / "test_mesh.obj"
        mesh = trimesh.creation.box(extents=(20, 20, 10))
        mesh.export(str(self.mesh_file))
        
    def tearDown(self):
        """Tear down test fixtures."""
        self.temp_dir.cleanup()
        
    def test_split_2x2(self):
        """Test splitting into 2x2 grid (4 tiles)."""
        output_file = self.test_dir / "heightmap.png"
        
        # Run with splitting
        args = [
            str(self.mesh_file),
            "--output_path", str(output_file),
            "--split", "4"
        ]
        exit_code = heightcraft_main(args)
        
        # Verify success
        self.assertEqual(exit_code, 0)
        
        # The service creates a _split subdirectory
        output_dir = self.test_dir / "heightmap_split"
        self.assertTrue(output_dir.exists())
        
        # Verify 4 tiles were created
        tile_files = list(output_dir.glob("*.png"))
        self.assertEqual(len(tile_files), 4, f"Expected 4 tiles, found {len(tile_files)}")
        
    def test_split_3x3(self):
        """Test splitting into 3x3 grid (9 tiles)."""
        output_file = self.test_dir / "heightmap_3x3.png"
        
        # Run with splitting
        args = [
            str(self.mesh_file),
            "--output_path", str(output_file),
            "--split", "9"
        ]
        exit_code = heightcraft_main(args)
        
        # Verify success
        self.assertEqual(exit_code, 0)
        
        # The service creates a _split subdirectory
        output_dir = self.test_dir / "heightmap_3x3_split"
        
        # Verify 9 tiles were created
        tile_files = list(output_dir.glob("*.png"))
        self.assertEqual(len(tile_files), 9, f"Expected 9 tiles, found {len(tile_files)}")
        
    def test_split_with_custom_resolution(self):
        """Test splitting with custom resolution."""
        output_file = self.test_dir / "heightmap_custom.png"
        
        # Run with splitting and custom resolution
        args = [
            str(self.mesh_file),
            "--output_path", str(output_file),
            "--split", "4",
            "--max_resolution", "512"
        ]
        exit_code = heightcraft_main(args)
        
        # Verify success
        self.assertEqual(exit_code, 0)
        
        # The service creates a _split subdirectory
        output_dir = self.test_dir / "heightmap_custom_split"
        
        # Verify tiles were created
        tile_files = list(output_dir.glob("*.png"))
        self.assertEqual(len(tile_files), 4)
