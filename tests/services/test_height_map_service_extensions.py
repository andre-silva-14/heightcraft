import numpy as np
import pytest
from heightcraft.domain.height_map import HeightMap
from heightcraft.services.height_map_service import HeightMapService

class TestHeightMapServiceExtensions:
    
    def setup_method(self):
        self.service = HeightMapService()
        
    def test_apply_sea_level(self):
        # Create a simple height map: 0, 1, 2, 3
        # This will be normalized to 0, 0.333, 0.666, 1.0
        data = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
        height_map = HeightMap(data, 16)
        
        # Apply sea level at 0.5 (normalized)
        # Values < 0.5 become 0.5
        # Intermediate: 0.5, 0.5, 0.666, 1.0
        # Re-normalized? No, HeightMap only normalizes if outside [0, 1].
        # Since 0.5-1.0 is within [0, 1], it keeps the values.
        # 0.5 -> 0.5
        # 0.666 -> 0.666
        # 1.0 -> 1.0
        modified_map, mask = self.service.apply_sea_level(height_map, 0.5)
        
        # Check modified map
        expected_data = np.array([[0.5, 0.5], [0.6666667, 1.0]], dtype=np.float32)
        np.testing.assert_allclose(modified_map.data, expected_data, rtol=1e-5)
        
        # Check mask (1 where water/below sea level, 0 where land)
        # Original 0.0 and 0.333 are < 0.5, so they are water (1.0)
        expected_mask = np.array([[1.0, 1.0], [0.0, 0.0]], dtype=np.float32)
        np.testing.assert_array_equal(mask.data, expected_mask)
        
    def test_generate_slope_map(self):
        # Plane with constant slope
        # z = x + y
        x, y = np.meshgrid(np.arange(5), np.arange(5))
        data = (x + y).astype(np.float32)
        height_map = HeightMap(data, 16)
        
        slope_map = self.service.generate_slope_map(height_map)
        
        assert slope_map.data.shape == data.shape
        # Slope should be constant (except borders due to gradient calculation)
        # np.gradient uses central difference in interior, one-sided at boundaries
        # Just check it runs and returns valid values
        assert np.all(slope_map.data >= 0.0)
        assert np.all(slope_map.data <= 1.0)
        
    def test_generate_curvature_map(self):
        # Bowl shape: z = x^2 + y^2
        x, y = np.meshgrid(np.arange(-5, 6), np.arange(-5, 6))
        data = (x**2 + y**2).astype(np.float32)
        height_map = HeightMap(data, 16)
        
        curvature_map = self.service.generate_curvature_map(height_map)
        
        assert curvature_map.data.shape == data.shape
        assert np.all(curvature_map.data >= 0.0)
        assert np.all(curvature_map.data <= 1.0)
