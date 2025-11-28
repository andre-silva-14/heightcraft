"""
E2E test for Image-to-Image Upscaling.
"""

import os
import tempfile
import numpy as np
from PIL import Image
import pytest
from heightcraft.core.config import ApplicationConfig
from heightcraft.processors.image_processor import ImageProcessor
from heightcraft.domain.height_map import HeightMap

def test_image_upscaling_e2e():
    """Test full image upscaling pipeline."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input.png")
        output_path = os.path.join(tmpdir, "output.png")
        
        # 1. Create low-res input image (10x10, 8-bit)
        data = np.random.randint(0, 255, (10, 10), dtype=np.uint8)
        img = Image.fromarray(data)
        img.save(input_path)
        
        # 2. Configure processor
        # We simulate what main.py does by creating config manually
        # but using the actual ImageProcessor
        
        # Mock args
        args = {
            'file_path': input_path,
            'output_path': output_path,
            'upscale': True,
            'upscale_factor': 2,
            'bit_depth': 16, # Request bit depth increase
            'format': 'png'
        }
        
        config = ApplicationConfig.from_dict(args)
        
        # Verify mode detection
        from heightcraft.core.config import ProcessingMode
        assert config.model_config.mode == ProcessingMode.IMAGE
        
        # 3. Run Processor
        processor = ImageProcessor(config)
        result_path = processor.process()
        
        # 4. Verify Output
        assert os.path.exists(result_path)
        assert result_path == output_path
        
        # Check resolution (should be 20x20)
        output_hm = HeightMap.from_file(output_path, bit_depth=16)
        assert output_hm.width == 20
        assert output_hm.height == 20
        
        # Check bit depth (should be 16-bit)
        assert output_hm.bit_depth == 16
        
        # Verify data range (should be scaled to 16-bit)
        # We check the file directly using PIL to ensure it was saved as 16-bit
        with Image.open(output_path) as img:
            # Mode 'I;16' is standard for 16-bit PNG in PIL
            # Or check if it's 16-bit by checking array dtype
            img_arr = np.array(img)
            assert img_arr.dtype == np.uint16
            assert img_arr.max() > 255

def test_image_upscaling_disabled_warning():
    """Test that image processing is skipped if upscaling is disabled."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input.png")
        output_path = os.path.join(tmpdir, "output.png")
        
        # Create dummy input
        Image.new('L', (10, 10)).save(input_path)
        
        # Configure without upscale
        args = {
            'file_path': input_path,
            'output_path': output_path,
            'upscale': False
        }
        
        config = ApplicationConfig.from_dict(args)
        
        # Run Processor
        processor = ImageProcessor(config)
        result = processor.process()
        
        # Should return empty string and NOT create output
        assert result == ""
        assert not os.path.exists(output_path)
