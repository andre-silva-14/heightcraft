"""
Tests for the logging module.
"""

import os
import unittest
import logging
from unittest.mock import Mock, patch, MagicMock

from heightcraft.core.logging import LoggingConfig, setup_logging
from tests.base_test_case import BaseTestCase


class TestLoggingConfig(BaseTestCase):
    """Tests for the LoggingConfig class."""
    
    def setUp(self) -> None:
        """Set up test environment."""
        super().setUp()
        
        # Store the original root logger handlers to restore them later
        self.original_handlers = logging.root.handlers.copy()
        
        # Create a test log file path
        self.log_file = self.get_temp_path("test_log.log")
    
    def tearDown(self) -> None:
        """Clean up after each test."""
        super().tearDown()
        
        # Reset the root logger
        logging.root.handlers = self.original_handlers
        
        # Remove the test log file if it exists
        if os.path.exists(self.log_file):
            os.remove(self.log_file)
    
    def test_configure_default(self) -> None:
        """Test configuring logging with default settings."""
        # Clear existing handlers
        logging.root.handlers = []
        
        # Configure logging
        LoggingConfig.configure()
        
        # Check that the root logger has a handler
        self.assertGreater(len(logging.root.handlers), 0)
        
        # Check that the logger level is set to INFO
        self.assertEqual(logging.root.level, logging.INFO)
    
    def test_configure_with_level(self) -> None:
        """Test configuring logging with a specific level."""
        # Clear existing handlers
        logging.root.handlers = []
        
        # Configure logging with DEBUG level
        LoggingConfig.configure(level="debug")
        
        # Check that the root logger level is set to DEBUG
        self.assertEqual(logging.root.level, logging.DEBUG)
        
        # Configure logging with WARNING level
        LoggingConfig.configure(level="warning")
        
        # Check that the root logger level is set to WARNING
        self.assertEqual(logging.root.level, logging.WARNING)
    
    def test_configure_with_format(self) -> None:
        """Test configuring logging with a specific format."""
        # Clear existing handlers
        logging.root.handlers = []
        
        # Configure logging with a custom format
        custom_format = "%(levelname)s - %(message)s"
        LoggingConfig.configure(format_string=custom_format)
        
        # Check that the handler has the custom format
        handler = logging.root.handlers[0]
        self.assertEqual(handler.formatter._fmt, custom_format)
    
    def test_configure_with_file(self) -> None:
        """Test configuring logging with a file."""
        # Clear existing handlers
        logging.root.handlers = []
        
        # Configure logging with a file
        LoggingConfig.configure(log_file=self.log_file, log_to_file=True)
        
        # Check that a FileHandler was added
        file_handlers = [h for h in logging.root.handlers if isinstance(h, logging.FileHandler)]
        self.assertEqual(len(file_handlers), 1)
        
        # Check that the file handler is writing to the correct file
        self.assertEqual(file_handlers[0].baseFilename, self.log_file)
    
    def test_configure_console_only(self) -> None:
        """Test configuring logging with console output only."""
        # Clear existing handlers
        logging.root.handlers = []
        
        # Configure logging with console output only
        LoggingConfig.configure(log_to_console=True, log_to_file=False)
        
        # Check that only a StreamHandler was added
        self.assertEqual(len(logging.root.handlers), 1)
        self.assertIsInstance(logging.root.handlers[0], logging.StreamHandler)
    
    def test_configure_file_only(self) -> None:
        """Test configuring logging with file output only."""
        # Clear existing handlers
        logging.root.handlers = []
        
        # Configure logging with file output only
        LoggingConfig.configure(log_file=self.log_file, log_to_console=False, log_to_file=True)
        
        # Check that only a FileHandler was added
        self.assertEqual(len(logging.root.handlers), 1)
        self.assertIsInstance(logging.root.handlers[0], logging.FileHandler)
    
    def test_get_logger(self) -> None:
        """Test getting a logger."""
        # Clear existing handlers
        logging.root.handlers = []
        
        # Configure logging
        LoggingConfig.configure()
        
        # Get a logger
        logger_name = "test_logger"
        logger = LoggingConfig.get_logger(logger_name)
        
        # Check that we got a logger with the correct name
        self.assertIsInstance(logger, logging.Logger)
        self.assertEqual(logger.name, logger_name)
    
    def test_setup_logging(self) -> None:
        """Test the setup_logging function."""
        # Clear existing handlers
        logging.root.handlers = []
        
        # Set up logging with verbose level 0
        setup_logging(verbose=0)
        
        # Check that the root logger level is set to WARNING
        self.assertEqual(logging.root.level, logging.WARNING)
        
        # Set up logging with verbose level 1
        setup_logging(verbose=1)
        
        # Check that the root logger level is set to INFO
        self.assertEqual(logging.root.level, logging.INFO)
        
        # Set up logging with verbose level 2
        setup_logging(verbose=2)
        
        # Check that the root logger level is still set to DEBUG
        self.assertEqual(logging.root.level, logging.DEBUG) 