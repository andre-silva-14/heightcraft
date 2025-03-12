"""
Logging configuration for Heightcraft.

This module provides centralized logging configuration for the application.
"""

import logging
import os
import sys
from typing import Dict, List, Optional


class LoggingConfig:
    """
    Logging configuration for Heightcraft.
    
    This class provides methods for configuring logging for the application,
    with support for different log levels, formatters, and output destinations.
    """
    
    # Default format string
    DEFAULT_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    
    # Default date format
    DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    
    # Log level mapping
    LEVEL_MAP = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }
    
    @staticmethod
    def configure(
        level: str = "info",
        format_string: Optional[str] = None,
        date_format: Optional[str] = None,
        log_file: Optional[str] = None,
        log_to_console: bool = True,
        log_to_file: bool = False,
    ) -> None:
        """
        Configure logging for the application.
        
        Args:
            level: Log level (debug, info, warning, error, critical)
            format_string: Log format string
            date_format: Date format string
            log_file: Path to log file
            log_to_console: Whether to log to console
            log_to_file: Whether to log to file
            
        Raises:
            ValueError: If the log level is invalid
        """
        # Get log level
        log_level = LoggingConfig.LEVEL_MAP.get(level.lower())
        if log_level is None:
            raise ValueError(f"Invalid log level: {level}")
        
        # Get format strings
        format_string = format_string or LoggingConfig.DEFAULT_FORMAT
        date_format = date_format or LoggingConfig.DEFAULT_DATE_FORMAT
        
        # Create formatter
        formatter = logging.Formatter(format_string, date_format)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        
        # Remove existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Add console handler
        if log_to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)
        
        # Add file handler
        if log_to_file and log_file:
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        
        # Log configuration information
        logging.info(f"Logging configured with level: {level}")
    
    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        """
        Get a logger with the specified name.
        
        Args:
            name: Logger name
            
        Returns:
            Logger
        """
        return logging.getLogger(name)


# Configure logging with default settings
def setup_logging(verbose: int = 0) -> None:
    """
    Set up logging for the application.
    
    Args:
        verbose: Verbosity level (0=WARNING, 1=INFO, 2+=DEBUG)
    """
    # Determine log level
    if verbose >= 2:
        level = "debug"
    elif verbose == 1:
        level = "info"
    else:
        level = "warning"
    
    # Configure logging
    LoggingConfig.configure(level=level)
    
    # Log startup information
    logging.info(f"Heightcraft starting with log level: {level.upper()}") 