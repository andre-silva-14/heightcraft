"""End-to-end tests for large model processing."""
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import trimesh

from heightcraft.main import main as heightcraft_main


class TestE2ELargeModel(unittest.TestCase):
    """End-to-end tests for large model processing workflow."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)
        
    def tearDown(self):
        """Tear down test fixtures."""
        self.temp_dir.cleanup()
        
    def test_large_model_processing(self):
        """Test large model processing with chunking."""
        # Create a mesh with many vertices (simulating a large model)
        mesh_file = self.test_dir / "large_mesh.obj"
        
        # Create a sphere with high subdivision (more vertices)
        mesh = trimesh.creation.icosphere(subdivisions=5)
        mesh.export(str(mesh_file))
        
        output_file = self.test_dir / "large_output.png"
        
        # Run with large model mode
        args = [
            str(mesh_file),
            "--output_path", str(output_file),
            "--large_model",
            "--chunk_size", "10000"
        ]
        exit_code = heightcraft_main(args)
        
        # Verify success
        self.assertEqual(exit_code, 0)
        self.assertTrue(output_file.exists())
        
    def test_large_model_with_memory_limit(self):
        """Test large model processing with memory limit."""
        mesh_file = self.test_dir / "mesh.obj"
        mesh = trimesh.creation.icosphere(subdivisions=4)
        mesh.export(str(mesh_file))
        
        output_file = self.test_dir / "output.png"
        
        # Run with memory limit
        args = [
            str(mesh_file),
            "--output_path", str(output_file),
            "--large_model",
            "--max_memory", "0.5"
        ]
        exit_code = heightcraft_main(args)
        
        # Verify success
        self.assertEqual(exit_code, 0)
        self.assertTrue(output_file.exists())
        
    def test_large_model_with_caching(self):
        """Test large model processing with caching."""
        mesh_file = self.test_dir / "mesh.obj"
        mesh = trimesh.creation.icosphere(subdivisions=4)
        mesh.export(str(mesh_file))
        
        output_file = self.test_dir / "output.png"
        cache_dir = self.test_dir / "cache"
        
        # Run with caching
        args = [
            str(mesh_file),
            "--output_path", str(output_file),
            "--large_model",
            "--cache_dir", str(cache_dir)
        ]
        exit_code = heightcraft_main(args)
        
        # Verify success
        self.assertEqual(exit_code, 0)
        self.assertTrue(output_file.exists())
        # Cache directory may or may not be created depending on whether caching was triggered
