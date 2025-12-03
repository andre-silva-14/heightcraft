import pytest
import time
from unittest.mock import MagicMock, patch
from heightcraft.infrastructure.profiler import Profiler

class TestProfiler:
    def test_init(self):
        profiler = Profiler()
        assert profiler.enabled
        assert profiler.metrics == {}

    def test_measure_context_manager(self):
        profiler = Profiler()
        
        with profiler.measure("test_op"):
            time.sleep(0.01)
            
        assert "test_op" in profiler.metrics
        assert len(profiler.metrics["test_op"]) == 1
        assert profiler.metrics["test_op"][0] >= 0.01

    def test_measure_disabled(self):
        profiler = Profiler(enabled=False)
        
        with profiler.measure("test_op"):
            pass
            
        assert "test_op" not in profiler.metrics

    def test_profile_decorator(self):
        profiler = Profiler()
        
        @profiler.profile("decorated_func")
        def my_func():
            time.sleep(0.01)
            return "result"
            
        result = my_func()
        
        assert result == "result"
        assert "decorated_func" in profiler.metrics
        assert len(profiler.metrics["decorated_func"]) == 1

    def test_profile_decorator_default_name(self):
        profiler = Profiler()
        
        @profiler.profile()
        def my_func_name():
            pass
            
        my_func_name()
        
        assert "my_func_name" in profiler.metrics

    def test_get_metrics(self):
        profiler = Profiler()
        profiler.metrics = {
            "op1": [1.0, 2.0, 3.0],
            "op2": []
        }
        
        metrics = profiler.get_metrics()
        
        assert "op1" in metrics
        assert metrics["op1"]["min"] == 1.0
        assert metrics["op1"]["max"] == 3.0
        assert metrics["op1"]["avg"] == 2.0
        assert metrics["op1"]["total"] == 6.0
        assert metrics["op1"]["count"] == 3
        
        assert "op2" not in metrics

    def test_get_summary(self):
        profiler = Profiler()
        profiler.metrics = {"op1": [1.0]}
        
        summary = profiler.get_summary()
        assert "Performance metrics:" in summary
        assert "op1:" in summary
        assert "Avg: 1.0000s" in summary

    def test_get_summary_empty(self):
        profiler = Profiler()
        assert profiler.get_summary() == "No performance metrics collected."

    def test_reset(self):
        profiler = Profiler()
        profiler.metrics = {"op1": [1.0]}
        profiler.reset()
        assert profiler.metrics == {}

    def test_enable_disable(self):
        profiler = Profiler()
        profiler.disable()
        assert not profiler.enabled
        profiler.enable()
        assert profiler.enabled
