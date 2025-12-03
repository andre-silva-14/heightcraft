import os
import time
import pytest
import pickle
import numpy as np
from unittest.mock import MagicMock, patch, mock_open

from heightcraft.infrastructure.cache_manager import CacheManager
from heightcraft.core.exceptions import CacheError

class TestCacheManager:
    @pytest.fixture
    def cache_dir(self, tmp_path):
        return str(tmp_path / ".cache")

    @pytest.fixture
    def cache_manager(self, cache_dir):
        return CacheManager(cache_dir=cache_dir)

    def test_init(self, cache_dir):
        manager = CacheManager(cache_dir=cache_dir)
        assert manager.cache_dir == cache_dir
        assert os.path.exists(cache_dir)

    def test_get_miss(self, cache_manager):
        assert cache_manager.get("non_existent") is None
        assert cache_manager.get("non_existent", default="default") == "default"

    def test_set_get(self, cache_manager):
        key = "test_key"
        value = {"data": 123}
        
        cache_manager.set(key, value)
        result = cache_manager.get(key)
        
        assert result == value

    def test_get_expired(self, cache_manager):
        key = "test_key"
        value = "data"
        
        # Set with very short max_age
        cache_manager.max_age = 0.1
        cache_manager.set(key, value)
        
        # Wait for expiration
        time.sleep(0.2)
        
        assert cache_manager.get(key) is None

    def test_set_error(self, cache_manager):
        with patch('builtins.open', side_effect=Exception("Write error")):
            with pytest.raises(CacheError, match="Failed to set cache entry"):
                cache_manager.set("key", "value")

    def test_delete(self, cache_manager):
        key = "test_key"
        cache_manager.set(key, "value")
        assert cache_manager.get(key) == "value"
        
        cache_manager.delete(key)
        assert cache_manager.get(key) is None

    def test_delete_error(self, cache_manager):
        # Should not raise, just log warning
        with patch.object(cache_manager.file_storage, 'delete_file', side_effect=Exception("Delete error")):
            cache_manager.delete("key")

    def test_clear(self, cache_manager):
        cache_manager.set("key1", "value1")
        cache_manager.set("key2", "value2")
        
        cache_manager.clear()
        
        assert cache_manager.get("key1") is None
        assert cache_manager.get("key2") is None

    def test_clear_error(self, cache_manager):
        with patch.object(cache_manager.file_storage, 'clear_directory', side_effect=Exception("Clear error")):
            with pytest.raises(CacheError, match="Failed to clear cache"):
                cache_manager.clear()

    def test_numpy_array(self, cache_manager):
        key = "numpy_key"
        array = np.array([1, 2, 3])
        
        cache_manager.set_numpy_array(key, array)
        result = cache_manager.get_numpy_array(key)
        
        np.testing.assert_array_equal(result, array)

    def test_numpy_array_miss(self, cache_manager):
        assert cache_manager.get_numpy_array("non_existent") is None

    def test_numpy_array_expired(self, cache_manager):
        key = "numpy_key"
        array = np.array([1, 2, 3])
        
        cache_manager.max_age = 0.1
        cache_manager.set_numpy_array(key, array)
        
        time.sleep(0.2)
        
        assert cache_manager.get_numpy_array(key) is None

    def test_set_numpy_error(self, cache_manager):
        with patch('numpy.save', side_effect=Exception("Save error")):
            with pytest.raises(CacheError, match="Failed to set cache entry"):
                cache_manager.set_numpy_array("key", np.array([1]))

    def test_cache_computation(self, cache_manager):
        key = "comp_key"
        
        # First run - should execute and set cache
        with cache_manager.cache_computation(key) as set_result:
            set_result("result")
        
        assert cache_manager.get(key) == "result"
        
        # Second run - should use cache and ignore new result
        with cache_manager.cache_computation(key) as set_result:
            set_result("new_result")
            
        assert cache_manager.get(key) == "result"

    def test_cache_computation_force(self, cache_manager):
        key = "comp_key"
        cache_manager.set(key, "old_result")
        
        executed = False
        with cache_manager.cache_computation(key, force_recompute=True) as set_result:
            executed = True
            set_result("new_result")
            
        assert executed
        assert cache_manager.get(key) == "new_result"

    def test_ensure_cache_directory_error(self, cache_manager):
        with patch.object(cache_manager.file_storage, 'ensure_directory', side_effect=Exception("Dir error")):
            with pytest.raises(CacheError, match="Failed to ensure cache directory"):
                cache_manager._ensure_cache_directory()
