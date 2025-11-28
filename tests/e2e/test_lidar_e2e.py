"""
E2E test for LiDAR processing.
"""

import os
import tempfile
import numpy as np
import pytest
import laspy

from heightcraft.main import main

def create_dummy_las(file_path: str, num_points: int = 1000):
    """Create a dummy LAS file."""
    # Create header
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.add_extra_dim(laspy.ExtraBytesParams(name="random", type=np.int32))
    
    # Create LAS file
    las = laspy.LasData(header)
    
    # Add random points
    x = np.random.uniform(0, 100, num_points)
    y = np.random.uniform(0, 100, num_points)
    z = np.random.uniform(0, 50, num_points)
    
    # Add corner points to ensure exact bounds (0-100)
    x = np.append(x, [0, 100, 0, 100])
    y = np.append(y, [0, 0, 100, 100])
    z = np.append(z, [0, 50, 0, 50])
    
    las.x = x
    las.y = y
    las.z = z
    
    las.write(file_path)

def test_lidar_e2e_pipeline():
    """Test the full LiDAR pipeline."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "test.las")
        output_path = os.path.join(tmpdir, "output.tiff")
        
        # Create dummy LAS file
        create_dummy_las(input_path)
        
        # Run CLI
        args = [
            input_path,
            "--output_path", output_path,
            "--bit_depth", "32",
            "--max_resolution", "128",
            "--chunk_size", "100" # Small chunk size to force multiple chunks
        ]
        
        exit_code = main(args)
        
        assert exit_code == 0
        assert os.path.exists(output_path)
        
        # Verify output
        import tifffile
        data = tifffile.imread(output_path)
        assert data.shape == (128, 128)
        assert data.dtype == np.float32

def test_lidar_upscaling_e2e():
    """Test LiDAR pipeline with upscaling."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "test_upscale.las")
        output_path = os.path.join(tmpdir, "output_upscaled.png")
        
        # Create dummy LAS file
        create_dummy_las(input_path)
        
        # Run CLI with upscaling
        # Note: We use 16-bit PNG output for upscaling test
        args = [
            input_path,
            "--output_path", output_path,
            "--bit_depth", "16",
            "--max_resolution", "64",
            "--upscale",
            "--upscale_factor", "2"
        ]
        
        exit_code = main(args)
        
        assert exit_code == 0
        assert os.path.exists(output_path)
        
        # Verify output resolution (64 * 2 = 128)
        from PIL import Image
        img = Image.open(output_path)
        assert img.size == (128, 128)
