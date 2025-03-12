"""
Base test case for Heightcraft tests.

This module provides a base test case with common setup and teardown methods.
"""

import os
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional

from heightcraft.core.constants import (
    APP_NAME,
    DEFAULT_BIT_DEPTH,
    DEFAULT_MAX_RESOLUTION,
    DEFAULT_NUM_SAMPLES,
    DEFAULT_NUM_THREADS,
)
from heightcraft.core.logging import LoggingConfig


class BaseTestCase(unittest.TestCase):
    """
    Base test case for Heightcraft tests.
    
    This class provides common setup and teardown methods for all tests.
    """
    
    def setUp(self) -> None:
        """Set up test environment."""
        # Configure logging
        LoggingConfig.configure(level="warning")
        
        # Create temporary directory
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Set environment variables
        os.environ["HEIGHTCRAFT_TEST_MODE"] = "1"
    
    def tearDown(self) -> None:
        """Clean up test environment."""
        # Remove temporary directory
        self.temp_dir.cleanup()
        
        # Clear environment variables
        os.environ.pop("HEIGHTCRAFT_TEST_MODE", None)
    
    def get_temp_path(self, filename: str) -> str:
        """
        Get a temporary file path.
        
        Args:
            filename: Filename
            
        Returns:
            Temporary file path
        """
        return os.path.join(self.temp_dir.name, filename)
    
    def create_temp_directory(self) -> str:
        """
        Create a temporary directory.
        
        Returns:
            Path to the temporary directory
        """
        return tempfile.mkdtemp()
    
    def get_test_data_path(self, filename: str) -> str:
        """
        Get a path to a test data file.
        
        Args:
            filename: Filename
            
        Returns:
            Path to the test data file
        """
        # Get the tests directory
        tests_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Get the test data directory
        test_data_dir = os.path.join(tests_dir, "data")
        
        # Return the path to the test data file
        return os.path.join(test_data_dir, filename) 