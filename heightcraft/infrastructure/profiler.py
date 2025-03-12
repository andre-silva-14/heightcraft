"""
Performance profiler for Heightcraft.

This module provides a Profiler class for measuring and tracking
the performance of operations.
"""

import contextlib
import logging
import time
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Union


class Profiler:
    """
    Performance profiler for measuring and tracking operation durations.
    
    This class provides methods for measuring the execution time of
    operations and tracking performance metrics.
    """
    
    def __init__(self, enabled: bool = True):
        """
        Initialize the profiler.
        
        Args:
            enabled: Whether the profiler is enabled
        """
        self.enabled = enabled
        self.metrics: Dict[str, List[float]] = {}
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @contextmanager
    def measure(self, operation_name: str):
        """
        Measure the execution time of an operation.
        
        This context manager measures the time taken to execute the code
        within its scope.
        
        Args:
            operation_name: Name of the operation being measured
            
        Yields:
            None
            
        Example:
            ```python
            with profiler.measure("load_model"):
                model = load_model()
            ```
        """
        if not self.enabled:
            yield
            return
        
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            if operation_name not in self.metrics:
                self.metrics[operation_name] = []
            self.metrics[operation_name].append(duration)
            self.logger.debug(f"{operation_name}: {duration:.4f}s")
    
    def profile(self, operation_name: Optional[str] = None) -> Callable:
        """
        Decorator for profiling a function.
        
        Args:
            operation_name: Name of the operation being measured (defaults to function name)
            
        Returns:
            Decorated function
            
        Example:
            ```python
            @profiler.profile()
            def process_data(data):
                # Process data
                return result
            ```
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                name = operation_name or func.__name__
                with self.measure(name):
                    return func(*args, **kwargs)
            return wrapper
        return decorator
    
    def get_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Get performance metrics.
        
        Returns:
            Dictionary of metrics with min, max, avg, total, and count for each operation
        """
        result = {}
        for name, durations in self.metrics.items():
            if not durations:
                continue
            
            total = sum(durations)
            count = len(durations)
            
            result[name] = {
                "min": min(durations),
                "max": max(durations),
                "avg": total / count,
                "total": total,
                "count": count,
            }
        
        return result
    
    def get_summary(self) -> str:
        """
        Get a summary of performance metrics as a string.
        
        Returns:
            Summary string
        """
        metrics = self.get_metrics()
        if not metrics:
            return "No performance metrics collected."
        
        lines = ["Performance metrics:"]
        for name, stats in metrics.items():
            lines.append(f"  {name}:")
            lines.append(f"    Count: {stats['count']}")
            lines.append(f"    Total: {stats['total']:.4f}s")
            lines.append(f"    Avg: {stats['avg']:.4f}s")
            lines.append(f"    Min: {stats['min']:.4f}s")
            lines.append(f"    Max: {stats['max']:.4f}s")
        
        return "\n".join(lines)
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.metrics.clear()
    
    def enable(self) -> None:
        """Enable the profiler."""
        self.enabled = True
    
    def disable(self) -> None:
        """Disable the profiler."""
        self.enabled = False


# Global instance
profiler = Profiler() 