
import os
import numpy as np
from PIL import Image
import pytest
from heightcraft.domain.height_map import HeightMap
from heightcraft.core.exceptions import HeightMapValidationError

def test_reproduce_3d_array_error(tmp_path):
    """
    Reproduction test for "Expected 2D array, got 3D" error when loading RGB images.
    """
    # Create a dummy RGB image
    width, height = 100, 100
    rgb_data = np.zeros((height, width, 3), dtype=np.uint8)
    # Add some data
    rgb_data[10:20, 10:20, :] = 255
    
    img_path = tmp_path / "test_rgb.png"
    Image.fromarray(rgb_data).save(img_path)
    
    # Try to load it as a HeightMap
    # This should SUCCEED now (after fix)
    try:
        hm = HeightMap.from_file(str(img_path), bit_depth=8)
        assert hm.data.ndim == 2
        assert hm.width == width
        assert hm.height == height
    except HeightMapValidationError as e:
        pytest.fail(f"Failed to load RGB image: {e}")

if __name__ == "__main__":
    # Manually run the test function if executed directly
    import shutil
    import tempfile
    from pathlib import Path
    
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        test_reproduce_3d_array_error(tmp_dir)
        print("Test passed (reproduction successful)!")
    finally:
        shutil.rmtree(tmp_dir)
