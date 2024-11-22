import pytest
import numpy as np
from lib.upscaler import HeightMapUpscaler

@pytest.fixture
def sample_height_map():
    return np.random.rand(100, 100).astype(np.float32)

def test_upscaler_initialization():
    upscaler = HeightMapUpscaler()
    assert upscaler.model is not None

def test_upscale(sample_height_map):
    upscaler = HeightMapUpscaler()
    upscaled = upscaler.upscale(sample_height_map, scale_factor=2)
    assert upscaled.shape == (200, 200)
    assert upscaled.dtype == sample_height_map.dtype

def test_upscale_factor(sample_height_map):
    upscaler = HeightMapUpscaler()
    for factor in [2, 3, 4]:
        upscaled = upscaler.upscale(sample_height_map, scale_factor=factor)
        assert upscaled.shape == (sample_height_map.shape[0] * factor, sample_height_map.shape[1] * factor)

def test_upscale_preserves_range(sample_height_map):
    upscaler = HeightMapUpscaler()
    upscaled = upscaler.upscale(sample_height_map)
    assert np.min(upscaled) >= np.min(sample_height_map) - 1e-6  # Allow for small floating-point errors
    assert np.max(upscaled) <= np.max(sample_height_map) + 1e-6  # Allow for small floating-point errors


def test_upscale_without_pretrained_model(sample_height_map):
    upscaler = HeightMapUpscaler()
    upscaled = upscaler.upscale(sample_height_map, scale_factor=2)
    assert upscaled.shape == (200, 200)
    assert upscaled.dtype == sample_height_map.dtype
    assert np.min(upscaled) >= np.min(sample_height_map)
    assert np.max(upscaled) <= np.max(sample_height_map)

# Note: This test requires a pretrained model file
@pytest.mark.skip(reason="Requires pretrained model file")
def test_load_pretrained_model():
    from lib.upscaler import load_pretrained_model
    upscaler = load_pretrained_model("path/to/pretrained_model.h5")
    assert upscaler.model is not None
