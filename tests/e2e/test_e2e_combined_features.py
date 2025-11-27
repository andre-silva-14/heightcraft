"""End-to-end tests for combined features (README complete example)."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import trimesh

from heightcraft.main import main as heightcraft_main


class TestE2ECombinedFeatures(unittest.TestCase):
    """End-to-end tests for combined features from README example."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)
        
        # Create a test mesh
        self.mesh_file = self.test_dir / "model.obj"
        mesh = trimesh.creation.box(extents=(15, 15, 8))
        mesh.export(str(self.mesh_file))
        
    def tearDown(self):
        """Tear down test fixtures."""
        self.temp_dir.cleanup()
        
    def test_complete_example_without_upscaling(self):
        """Test the complete README example (without upscaling which requires TensorFlow).
        
        From README:
        python main.py model.obj \\
          --output_path detailed_map.png \\
          --max_resolution 1024 \\
          --use_gpu \\
          --num_samples 1000000 \\
          --split 4 \\
          --max_memory 0.7 \\
          --cache_dir ./cache
        """
        output_file = self.test_dir / "detailed_map.png"
        cache_dir = self.test_dir / "cache"
        
        # Run with combined features (skip GPU and reduce samples for testing)
        args = [
            str(self.mesh_file),
            "--output_path", str(output_file),
            "--max_resolution", "256",  # Reduced for testing
            "--num_samples", "10000",  # Reduced for testing
            "--split", "4",
            "--max_memory", "0.7",
            "--cache_dir", str(cache_dir)
        ]
        exit_code = heightcraft_main(args)
        
        # Verify success
        self.assertEqual(exit_code, 0)
        
        # Verify splitting worked - service creates a _split subdirectory
        output_dir = self.test_dir / "detailed_map_split"
        tile_files = list(output_dir.glob("*.png"))
        self.assertEqual(len(tile_files), 4, f"Expected 4 split files, found {len(tile_files)}")
        
    @patch('heightcraft.services.upscaling_service.UpscalingService.upscale')
    def test_upscaling_integration(self, mock_upscale):
        """Test that upscaling flag integrates correctly (mocked to avoid TensorFlow dependency)."""
        from heightcraft.domain.height_map import HeightMap
        import numpy as np
        
        # Mock the upscale to return a simple upscaled map
        def mock_upscale_func(height_map, scale_factor, use_gpu):
            data = height_map.data
            upscaled_data = np.repeat(np.repeat(data, scale_factor, axis=0), scale_factor, axis=1)
            return HeightMap(upscaled_data, height_map.bit_depth)
        
        mock_upscale.side_effect = mock_upscale_func
        
        output_file = self.test_dir / "upscaled_output.png"
        
        # Run with upscaling
        args = [
            str(self.mesh_file),
            "--output_path", str(output_file),
            "--upscale",
            "--upscale_factor", "2"
        ]
        exit_code = heightcraft_main(args)
        
        # Verify success
        self.assertEqual(exit_code, 0)
        self.assertTrue(output_file.exists())
        
        # Verify upscale was called
        mock_upscale.assert_called_once()
