"""
Test 32-bit height map support.
"""

import os
import tempfile
import numpy as np
import pytest
from heightcraft.domain.height_map import HeightMap
from heightcraft.core.config import OutputFormat

def test_32bit_height_map_creation():
    """Test creating a 32-bit height map."""
    data = np.random.rand(100, 100).astype(np.float32)
    hm = HeightMap(data, bit_depth=32)
    
    assert hm.bit_depth == 32
    assert hm.data.dtype == np.float32
    assert hm.max_value == 1.0

def test_32bit_tiff_saving():
    """Test saving a 32-bit height map as TIFF."""
    try:
        import tifffile
    except ImportError:
        pytest.skip("tifffile not installed")
        
    data = np.random.rand(100, 100).astype(np.float32)
    hm = HeightMap(data, bit_depth=32)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "test_32.tiff")
        hm.save(output_path, format=OutputFormat.TIFF)
        
        assert os.path.exists(output_path)
        
        # Verify content
        loaded_data = tifffile.imread(output_path)
        assert loaded_data.dtype == np.float32
        np.testing.assert_array_almost_equal(loaded_data, data)
