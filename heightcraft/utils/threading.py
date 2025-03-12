"""
Threading utilities for Heightcraft.

This module provides utilities for managing thread pools and parallel tasks.
"""

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

from heightcraft.core.exceptions import ThreadingError
from heightcraft.infrastructure.profiler import profiler


# Type variables for generic functions
T = TypeVar('T')
R = TypeVar('R')


class ThreadPool:
    """
    Thread pool for managing parallel tasks.
    
    This class provides a thread pool for executing tasks in parallel,
    with support for cancellation, timeout, and progress tracking.
    """
    
    def __init__(self, max_workers: int = None, name: str = "ThreadPool"):
        """
        Initialize the thread pool.
        
        Args:
            max_workers: Maximum number of worker threads (defaults to number of CPUs + 4)
            name: Name of the thread pool (for logging)
        """
        self.max_workers = max_workers
        self.name = name
        self.logger = logging.getLogger(f"{self.__class__.__name__}_{name}")
        self._executor = None
        self._lock = threading.RLock()
        self._running = False
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()
    
    def start(self) -> None:
        """
        Start the thread pool.
        
        Raises:
            ThreadingError: If the thread pool is already running
        """
        with self._lock:
            if self._running:
                raise ThreadingError("Thread pool is already running")
            
            self._executor = ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix=self.name)
            self._running = True
            self.logger.debug(f"Started thread pool with {self.max_workers} workers")
    
    def shutdown(self, wait: bool = True) -> None:
        """
        Shut down the thread pool.
        
        Args:
            wait: Whether to wait for pending tasks to complete
        """
        with self._lock:
            if not self._running:
                return
            
            self._running = False
            if self._executor:
                self._executor.shutdown(wait=wait)
                self._executor = None
            
            self.logger.debug("Shut down thread pool")
    
    def submit(self, fn: Callable[..., R], *args, **kwargs) -> Any:
        """
        Submit a task to the thread pool.
        
        Args:
            fn: Function to execute
            *args: Arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            Future representing the task
            
        Raises:
            ThreadingError: If the thread pool is not running
        """
        if not self._running:
            raise ThreadingError("Thread pool is not running")
        
        return self._executor.submit(fn, *args, **kwargs)
    
    @profiler.profile()
    def map(
        self, 
        fn: Callable[[T], R], 
        items: List[T],
        timeout: Optional[float] = None,
        show_progress: bool = False
    ) -> List[R]:
        """
        Apply a function to each item in a list in parallel.
        
        Args:
            fn: Function to apply to each item
            items: List of items to process
            timeout: Timeout in seconds (None for no timeout)
            show_progress: Whether to log progress
            
        Returns:
            List of results
            
        Raises:
            ThreadingError: If the thread pool is not running or a task fails
        """
        if not items:
            return []
        
        if not self._running:
            raise ThreadingError("Thread pool is not running")
        
        # Submit tasks
        futures = [self._executor.submit(fn, item) for item in items]
        results = []
        completed = 0
        
        # Wait for tasks to complete
        try:
            for future in as_completed(futures, timeout=timeout):
                result = future.result()
                results.append(result)
                
                # Update progress
                completed += 1
                if show_progress and len(items) > 1:
                    if completed % max(1, len(items) // 10) == 0 or completed == len(items):
                        self.logger.info(f"Progress: {completed}/{len(items)} ({completed / len(items) * 100:.1f}%)")
            
            return results
            
        except Exception as e:
            # Cancel remaining tasks
            for future in futures:
                if not future.done():
                    future.cancel()
            
            raise ThreadingError(f"Failed to execute tasks: {e}")
    
    @profiler.profile()
    def chunked_map(
        self, 
        fn: Callable[[List[T]], List[R]], 
        items: List[T],
        chunk_size: int,
        timeout: Optional[float] = None,
        show_progress: bool = False
    ) -> List[R]:
        """
        Apply a function to chunks of items in parallel.
        
        Args:
            fn: Function to apply to each chunk of items
            items: List of items to process
            chunk_size: Size of each chunk
            timeout: Timeout in seconds (None for no timeout)
            show_progress: Whether to log progress
            
        Returns:
            List of results
            
        Raises:
            ThreadingError: If the thread pool is not running or a task fails
        """
        if not items:
            return []
        
        if not self._running:
            raise ThreadingError("Thread pool is not running")
        
        # Create chunks
        chunks = [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]
        
        # Submit tasks
        futures = [self._executor.submit(fn, chunk) for chunk in chunks]
        results = []
        completed = 0
        
        # Wait for tasks to complete
        try:
            for future in as_completed(futures, timeout=timeout):
                chunk_results = future.result()
                results.extend(chunk_results)
                
                # Update progress
                completed += 1
                if show_progress and len(chunks) > 1:
                    if completed % max(1, len(chunks) // 10) == 0 or completed == len(chunks):
                        self.logger.info(f"Progress: {completed}/{len(chunks)} chunks ({completed / len(chunks) * 100:.1f}%)")
            
            return results
            
        except Exception as e:
            # Cancel remaining tasks
            for future in futures:
                if not future.done():
                    future.cancel()
            
            raise ThreadingError(f"Failed to execute chunked tasks: {e}")


def execute_with_retry(
    fn: Callable[..., R],
    *args,
    retries: int = 3,
    retry_delay: float = 1.0,
    **kwargs
) -> R:
    """
    Execute a function with retry logic.
    
    Args:
        fn: Function to execute
        *args: Arguments to pass to the function
        retries: Number of retries
        retry_delay: Delay between retries in seconds
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        Result of the function
        
    Raises:
        ThreadingError: If all retries fail
    """
    last_error = None
    
    for attempt in range(retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_error = e
            
            if attempt < retries:
                logging.warning(f"Attempt {attempt + 1}/{retries + 1} failed: {e}, retrying in {retry_delay}s")
                time.sleep(retry_delay)
                # Increase delay for next retry
                retry_delay *= 1.5
            else:
                logging.error(f"All {retries + 1} attempts failed")
    
    raise ThreadingError(f"Function failed after {retries + 1} attempts: {last_error}") 