import time
import pytest
from unittest.mock import MagicMock
from heightcraft.utils.threading import ThreadPool, execute_with_retry
from heightcraft.core.exceptions import ThreadingError

class TestThreadPool:
    
    def test_submit_and_execute(self):
        with ThreadPool(max_workers=2) as pool:
            future = pool.submit(lambda x: x * 2, 10)
            assert future.result() == 20

    def test_map(self):
        with ThreadPool(max_workers=2) as pool:
            items = [1, 2, 3, 4]
            results = pool.map(lambda x: x * 2, items)
            assert sorted(results) == [2, 4, 6, 8]

    def test_chunked_map(self):
        with ThreadPool(max_workers=2) as pool:
            items = [1, 2, 3, 4, 5]
            # Chunk size 2: [1, 2], [3, 4], [5]
            # Function receives a list and should return a list
            def process_chunk(chunk):
                return [x * 2 for x in chunk]
            
            results = pool.chunked_map(process_chunk, items, chunk_size=2)
            assert sorted(results) == [2, 4, 6, 8, 10]

    def test_shutdown(self):
        pool = ThreadPool(max_workers=1)
        pool.start()
        pool.shutdown()
        
        with pytest.raises(ThreadingError):
            pool.submit(lambda: None)

    def test_context_manager(self):
        with ThreadPool(max_workers=1) as pool:
            assert pool._running
        assert not pool._running

    def test_map_exception(self):
        with ThreadPool(max_workers=2) as pool:
            def fail_on_two(x):
                if x == 2:
                    raise ValueError("Failed")
                return x
            
            with pytest.raises(ThreadingError):
                pool.map(fail_on_two, [1, 2, 3])

class TestExecuteWithRetry:
    
    def test_success_first_try(self):
        mock_fn = MagicMock(return_value="success")
        result = execute_with_retry(mock_fn, retries=3)
        assert result == "success"
        assert mock_fn.call_count == 1

    def test_retry_success(self):
        # Fail twice, then succeed
        mock_fn = MagicMock(side_effect=[ValueError("Fail 1"), ValueError("Fail 2"), "success"])
        result = execute_with_retry(mock_fn, retries=3, retry_delay=0.01)
        assert result == "success"
        assert mock_fn.call_count == 3

    def test_retry_fail_all(self):
        mock_fn = MagicMock(side_effect=ValueError("Always fail"))
        with pytest.raises(ThreadingError):
            execute_with_retry(mock_fn, retries=2, retry_delay=0.01)
        assert mock_fn.call_count == 3  # Initial + 2 retries
