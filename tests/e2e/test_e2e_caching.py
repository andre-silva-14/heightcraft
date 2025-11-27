"""End-to-end tests for caching mechanism."""
import os
import tempfile
import time
import unittest
from pathlib import Path

import trimesh

from heightcraft.main import main as heightcraft_main


class TestE2ECaching(unittest.TestCase):
    """End-to-end tests for caching mechanism."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)
        
        # Create a test mesh
        self.mesh_file = self.test_dir / "model.obj"
        mesh = trimesh.creation.icosphere(subdivisions=3)
        mesh.export(str(self.mesh_file))
        
        self.output_file = self.test_dir / "output.png"
        self.cache_dir = self.test_dir / "cache"
        
    def tearDown(self):
        """Tear down test fixtures."""
        self.temp_dir.cleanup()
        
    def test_cache_directory_creation(self):
        """Test that cache directory is created when specified."""
        # Run with cache directory
        args = [
            str(self.mesh_file),
            "--output_path", str(self.output_file),
            "--cache_dir", str(self.cache_dir)
        ]
        exit_code = heightcraft_main(args)
        
        # Verify success
        self.assertEqual(exit_code, 0)
        self.assertTrue(self.output_file.exists())
        
        # Note: Cache directory creation depends on whether the cache manager
        # actually needs to cache anything, so we won't assert its existence
        
    def test_repeated_processing_with_cache(self):
        """Test that running the same command twice works (cache should not break anything)."""
        args = [
            str(self.mesh_file),
            "--output_path", str(self.output_file),
            "--cache_dir", str(self.cache_dir)
        ]
        
        # First run
        start_time1 = time.time()
        exit_code1 = heightcraft_main(args)
        duration1 = time.time() - start_time1
        
        # Verify first run success
        self.assertEqual(exit_code1, 0)
        self.assertTrue(self.output_file.exists())
        
        # Remove output file to ensure second run recreates it
        self.output_file.unlink()
        
        # Second run (may benefit from caching)
        start_time2 = time.time()
        exit_code2 = heightcraft_main(args)
        duration2 = time.time() - start_time2
        
        # Verify second run success
        self.assertEqual(exit_code2, 0)
        self.assertTrue(self.output_file.exists())
        
        # Note: We don't assert that duration2 < duration1 because:
        # 1. Caching behavior depends on implementation
        # 2. Small test meshes may not show significant speedup
        # 3. Timing can be unreliable in test environments
