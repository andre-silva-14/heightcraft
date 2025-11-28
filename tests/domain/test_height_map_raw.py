"""
Test RAW file support.
"""

import os
import tempfile
import numpy as np
import pytest
from heightcraft.domain.height_map import HeightMap
from heightcraft.core.config import OutputFormat

def test_save_raw_8bit():
    """Test saving 8-bit RAW."""
    data = np.random.randint(0, 256, (10, 10), dtype=np.uint8)
    # Ensure full range to avoid contrast stretching during normalization
    data[0, 0] = 0
    data[0, 1] = 255
    
    hm = HeightMap(data, bit_depth=8)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "test.raw")
        hm.save(output_path, format=OutputFormat.RAW)
        
        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) == 100  # 10x10 bytes
        
        # Verify content
        loaded_data = np.fromfile(output_path, dtype=np.uint8).reshape(10, 10)
        # Allow for small rounding errors due to float conversion
        np.testing.assert_array_equal(loaded_data, data)

def test_save_raw_16bit():
    """Test saving 16-bit RAW."""
    data = np.random.randint(0, 65536, (10, 10), dtype=np.uint16)
    # Ensure full range to avoid contrast stretching
    data[0, 0] = 0
    data[0, 1] = 65535
    
    hm = HeightMap(data, bit_depth=16)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "test.raw")
        hm.save(output_path, format=OutputFormat.RAW)
        
        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) == 200  # 10x10 * 2 bytes
        
        # Verify content
        loaded_data = np.fromfile(output_path, dtype=np.uint16).reshape(10, 10)
        # Allow for small rounding errors due to float conversion
        # We might need assert_allclose with tolerance 1
        np.testing.assert_allclose(loaded_data, data, atol=1)

def test_save_raw_32bit_downgrade():
    """Test saving 32-bit RAW downgrades to 16-bit."""
    data = np.random.rand(10, 10).astype(np.float32)
    hm = HeightMap(data, bit_depth=32)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "test.raw")
        
        with pytest.warns(UserWarning, match="downgrading to 16-bit"):
            hm.save(output_path, format=OutputFormat.RAW)
        
        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) == 200  # 10x10 * 2 bytes (16-bit)
        
        # Verify content is uint16
        loaded_data = np.fromfile(output_path, dtype=np.uint16).reshape(10, 10)
        assert loaded_data.dtype == np.uint16
