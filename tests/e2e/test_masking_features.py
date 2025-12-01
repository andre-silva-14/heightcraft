import os
import tempfile
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from heightcraft.core.config import ApplicationConfig
from heightcraft.processors.standard_processor import StandardProcessor

class TestMaskingFeaturesE2E:
    
    @patch('heightcraft.services.mesh_service.MeshService.load_mesh')
    @patch('heightcraft.services.mesh_service.MeshService.prepare_mesh')
    @patch('heightcraft.services.sampling_service.SamplingService.sample_from_mesh')
    def test_masking_and_texture_generation(self, mock_sample, mock_prepare, mock_load):
        # Setup mocks
        mock_mesh = MagicMock()
        mock_mesh.vertices = np.array([[0,0,0], [1,1,1]])
        mock_mesh.faces = np.array([[0,1,0]])
        mock_load.return_value = mock_mesh
        mock_prepare.return_value = mock_mesh
        
        # Mock sampled points: a simple plane
        # 10x10 grid
        x, y = np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
        z = x + y  # Slope
        points = np.column_stack((x.flatten(), y.flatten(), z.flatten()))
        mock_sample.return_value = points
        
        # Create config
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_output.png")
            
            config_dict = {
                "file_path": "dummy.obj",
                "output_path": output_path,
                "sea_level": 0.5,
                "slope_map": True,
                "curvature_map": True,
                "max_resolution": 10,
                "num_samples": 100
            }
            config = ApplicationConfig.from_dict(config_dict)
            
            processor = StandardProcessor(config)
            
            # Mock bounds since we are mocking sampling
            processor.bounds = {"min_x": 0, "max_x": 1, "min_y": 0, "max_y": 1, "min_z": 0, "max_z": 2}
            
            # Run process
            processor.process()
            
            # Check outputs
            assert os.path.exists(output_path)
            assert os.path.exists(os.path.join(tmpdir, "test_output_water_mask.png"))
            assert os.path.exists(os.path.join(tmpdir, "test_output_slope_map.png"))
            assert os.path.exists(os.path.join(tmpdir, "test_output_curvature_map.png"))
