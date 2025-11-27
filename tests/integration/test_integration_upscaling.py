import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import trimesh

from heightcraft.core.config import ApplicationConfig, UpscaleConfig
from heightcraft.processors.standard_processor import StandardProcessor
from heightcraft.services.upscaling_service import UpscalingService


class TestIntegrationUpscaling(unittest.TestCase):
    """Integration tests for the upscaling pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name)
        
        # Create a simple mesh file
        self.mesh_file = self.output_dir / "test_mesh.obj"
        mesh = trimesh.creation.box(extents=(1, 1, 1))
        mesh.export(str(self.mesh_file))
        
        # Create config with upscaling enabled
        from heightcraft.core.config import ModelConfig, SamplingConfig, HeightMapConfig, OutputConfig, UpscaleConfig
        
        self.config = ApplicationConfig(
            model_config=ModelConfig(file_path=str(self.mesh_file)),
            sampling_config=SamplingConfig(num_samples=1000),
            height_map_config=HeightMapConfig(max_resolution=256),
            output_config=OutputConfig(output_path=str(self.output_dir / "output.png")),
            upscale_config=UpscaleConfig(enabled=True, upscale_factor=2)
        )
        
    def tearDown(self):
        """Tear down test fixtures."""
        self.temp_dir.cleanup()
        
    def test_standard_processor_upscaling_pipeline(self):
        """Test that StandardProcessor correctly runs the upscaling step."""
        # Initialize processor
        processor = StandardProcessor(self.config)
        
        # Mock the actual upscaling to avoid heavy computation/dependencies
        # But we want to verify the flow
        with patch.object(UpscalingService, 'upscale') as mock_upscale:
            # Setup mock return value
            original_shape = (32, 32)
            upscaled_shape = (64, 64)
            
            # Create a mock upscaled height map
            mock_upscaled_map = MagicMock()
            mock_upscaled_map.data = np.zeros(upscaled_shape)
            mock_upscale.return_value = mock_upscaled_map
            
            # Run processing
            processor.process()
            
            # Verify upscale was called
            mock_upscale.assert_called_once()
            
            # Verify internal state was updated
            self.assertEqual(processor.height_map.shape, upscaled_shape)
            
    def test_upscaling_disabled(self):
        """Test that upscaling is skipped when disabled."""
        # Create a new config with upscaling disabled
        from heightcraft.core.config import ApplicationConfig, ModelConfig, SamplingConfig, HeightMapConfig, OutputConfig, UpscaleConfig
        
        disabled_config = ApplicationConfig(
            model_config=ModelConfig(file_path=str(self.mesh_file)),
            sampling_config=SamplingConfig(num_samples=1000),
            height_map_config=HeightMapConfig(max_resolution=256),
            output_config=OutputConfig(output_path=str(self.output_dir / "output.png")),
            upscale_config=UpscaleConfig(enabled=False)
        )
        
        processor = StandardProcessor(disabled_config)
        
        with patch.object(UpscalingService, 'upscale') as mock_upscale:
            processor.process()
            mock_upscale.assert_not_called()
