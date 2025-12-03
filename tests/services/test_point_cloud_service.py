import pytest
import numpy as np
from unittest.mock import MagicMock, patch, PropertyMock

from heightcraft.services.point_cloud_service import PointCloudService
from heightcraft.domain.point_cloud import PointCloud
from heightcraft.core.exceptions import PointCloudServiceError

class TestPointCloudService:
    @pytest.fixture
    def service(self):
        return PointCloudService()

    @pytest.fixture
    def points(self):
        return np.array([
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0]
        ])

    @pytest.fixture
    def point_cloud(self, points):
        return PointCloud(points)

    def test_create_point_cloud(self, service, points):
        pc = service.create_point_cloud(points)
        assert isinstance(pc, PointCloud)
        np.testing.assert_array_equal(pc.points, points)

    def test_create_point_cloud_error(self, service):
        with pytest.raises(PointCloudServiceError, match="Failed to create point cloud"):
            service.create_point_cloud("invalid")

    def test_filter_points_min_height(self, service, point_cloud):
        filtered = service.filter_points(point_cloud, min_height=1.5)
        assert len(filtered.points) == 2
        assert np.all(filtered.points[:, 2] >= 1.5)

    def test_filter_points_max_height(self, service, point_cloud):
        filtered = service.filter_points(point_cloud, max_height=1.5)
        assert len(filtered.points) == 2
        assert np.all(filtered.points[:, 2] <= 1.5)

    def test_filter_points_range(self, service, point_cloud):
        filtered = service.filter_points(point_cloud, min_height=0.5, max_height=2.5)
        assert len(filtered.points) == 2
        assert np.all((filtered.points[:, 2] >= 0.5) & (filtered.points[:, 2] <= 2.5))

    def test_filter_points_error(self, service):
        mock_pc = MagicMock()
        mock_pc.points = "invalid" # This will cause error when indexing
        with pytest.raises(PointCloudServiceError, match="Failed to filter points"):
            service.filter_points(mock_pc, min_height=0)

    def test_normalize_heights(self, service, point_cloud):
        normalized = service.normalize_heights(point_cloud)
        z_coords = normalized.points[:, 2]
        assert np.min(z_coords) == 0.0
        assert np.max(z_coords) == 1.0
        # Check x, y are preserved
        np.testing.assert_array_equal(normalized.points[:, :2], point_cloud.points[:, :2])

    def test_normalize_heights_error(self, service):
        mock_pc = MagicMock()
        mock_pc.normalize_z.side_effect = Exception("Normalize error")
        with pytest.raises(PointCloudServiceError, match="Failed to normalize heights"):
            service.normalize_heights(mock_pc)

    def test_compute_bounds(self, service, point_cloud):
        bounds = service.compute_bounds(point_cloud)
        assert bounds['min_x'] == 0.0
        assert bounds['max_x'] == 3.0
        assert bounds['min_y'] == 0.0
        assert bounds['max_y'] == 3.0
        assert bounds['min_z'] == 0.0
        assert bounds['max_z'] == 3.0

    def test_compute_bounds_error(self, service):
        mock_pc = MagicMock()
        type(mock_pc).bounds = PropertyMock(side_effect=Exception("Bounds error"))
        with pytest.raises(PointCloudServiceError, match="Failed to compute bounds"):
            service.compute_bounds(mock_pc)
