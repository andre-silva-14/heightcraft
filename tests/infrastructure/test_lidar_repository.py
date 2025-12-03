import pytest
from unittest.mock import MagicMock, patch, mock_open
import numpy as np
import sys
from heightcraft.infrastructure.lidar_repository import LidarRepository
from heightcraft.core.exceptions import RepositoryError
from heightcraft.domain.point_cloud import PointCloud

class TestLidarRepository:
    @pytest.fixture
    def mock_laspy(self):
        with patch('heightcraft.infrastructure.lidar_repository.laspy') as mock:
            yield mock

    @pytest.fixture
    def repo(self, mock_laspy):
        return LidarRepository()

    def test_init_no_laspy(self):
        with patch('heightcraft.infrastructure.lidar_repository.laspy', None):
            repo = LidarRepository()
            # Should log warning but not crash
            assert repo

    def test_get_chunk_iterator_success(self, repo, mock_laspy):
        file_path = "test.las"
        chunk_size = 100
        
        # Mock file handler and chunk iterator
        mock_fh = MagicMock()
        mock_laspy.open.return_value.__enter__.return_value = mock_fh
        
        # Create mock chunks
        chunk1 = MagicMock()
        chunk1.x = np.array([1, 2])
        chunk1.y = np.array([3, 4])
        chunk1.z = np.array([5, 6])
        
        chunk2 = MagicMock()
        chunk2.x = np.array([7, 8])
        chunk2.y = np.array([9, 10])
        chunk2.z = np.array([11, 12])
        
        mock_fh.chunk_iterator.return_value = [chunk1, chunk2]
        
        with patch('os.path.exists', return_value=True):
            iterator = repo.get_chunk_iterator(file_path, chunk_size)
            chunks = list(iterator)
            
            assert len(chunks) == 2
            assert isinstance(chunks[0], PointCloud)
            assert len(chunks[0].points) == 2
            assert np.array_equal(chunks[0].points, np.array([[1, 3, 5], [2, 4, 6]]))

    def test_get_chunk_iterator_file_not_found(self, repo):
        with patch('os.path.exists', return_value=False):
            with pytest.raises(RepositoryError, match="File not found"):
                list(repo.get_chunk_iterator("nonexistent.las", 100))

    def test_get_chunk_iterator_laspy_error(self, repo, mock_laspy):
        mock_laspy.open.side_effect = Exception("Laspy error")
        with patch('os.path.exists', return_value=True):
            with pytest.raises(RepositoryError, match="Failed to stream LiDAR file"):
                list(repo.get_chunk_iterator("test.las", 100))

    def test_get_bounds_success(self, repo, mock_laspy):
        file_path = "test.las"
        mock_fh = MagicMock()
        mock_laspy.open.return_value.__enter__.return_value = mock_fh
        
        mock_fh.header.x_min = 0
        mock_fh.header.x_max = 10
        mock_fh.header.y_min = 0
        mock_fh.header.y_max = 20
        mock_fh.header.z_min = 0
        mock_fh.header.z_max = 5
        
        with patch('os.path.exists', return_value=True):
            bounds = repo.get_bounds(file_path)
            
            assert bounds['min_x'] == 0
            assert bounds['max_x'] == 10
            assert bounds['max_y'] == 20

    def test_load_success(self, repo, mock_laspy):
        file_path = "test.las"
        mock_fh = MagicMock()
        mock_laspy.open.return_value.__enter__.return_value = mock_fh
        
        mock_las = MagicMock()
        mock_fh.read.return_value = mock_las
        
        mock_las.x = np.array([1, 2])
        mock_las.y = np.array([3, 4])
        mock_las.z = np.array([5, 6])
        
        with patch('os.path.exists', return_value=True):
            pc = repo.load(file_path)
            
            assert isinstance(pc, PointCloud)
            assert len(pc.points) == 2
            assert np.array_equal(pc.points, np.array([[1, 3, 5], [2, 4, 6]]))

    def test_load_file_not_found(self, repo):
        with patch('os.path.exists', return_value=False):
            with pytest.raises(RepositoryError, match="File not found"):
                repo.load("nonexistent.las")
