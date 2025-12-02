
import os
import sys
import logging
import numpy as np
import trimesh
import unittest
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from heightcraft.main import main
from heightcraft.processors.large_model_processor import LargeModelProcessor

class TestReproduction(unittest.TestCase):
    def setUp(self):
        # Create a dummy GLTF file
        self.test_file = "test_model.glb"
        mesh = trimesh.creation.box()
        # Make it have enough vertices to be interesting but not too huge
        mesh = mesh.subdivide()
        mesh = mesh.subdivide()
        mesh.export(self.test_file)
        
    def tearDown(self):
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
            
    def test_large_model_execution(self):
        # Arguments matching the user's report, but maybe scaled down slightly to avoid crashing the test runner if it is OOM
        # User: --max_resolution 8192 --num_samples 300000000 --chunk_size 100000 --large_model -vv
        
        # We'll use a smaller sample count initially to see if it runs at all, 
        # then we can try to increase it if it passes.
        # But the user says it fails.
        
        # Let's try to capture stdout/stderr to see what happens
        
        args = [
            self.test_file,
            "--max_resolution", "1024", # Smaller resolution for test
            "--num_samples", "100000", # Smaller samples for test
            "--chunk_size", "10000",
            "--large_model",
            "-vv"
        ]
        
        # Run main
        try:
            exit_code = main(args)
            self.assertEqual(exit_code, 0, "Main returned non-zero exit code")
        except SystemExit as e:
            self.assertEqual(e.code, 0, "SystemExit with non-zero code")
            
if __name__ == "__main__":
    unittest.main()
