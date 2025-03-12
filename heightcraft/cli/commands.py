"""
Commands for Heightcraft CLI.

This module provides command classes that implement the Command pattern for the CLI.
Each command encapsulates a specific action that can be executed by the CLI.
"""

import abc
import logging
import sys
from typing import Dict, List, Optional

import pytest

from heightcraft.cli.argument_parser import parse_arguments, validate_arguments
from heightcraft.core.config import ApplicationConfig
from heightcraft.core.exceptions import HeightcraftError
from heightcraft.processors import create_processor
from heightcraft.core.logging import setup_logging


class Command(abc.ABC):
    """
    Abstract base class for all commands.
    
    This class defines the interface that all commands must implement.
    It follows the Command pattern to encapsulate actions.
    """
    
    @abc.abstractmethod
    def execute(self) -> int:
        """
        Execute the command.
        
        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        pass


class GenerateHeightMapCommand(Command):
    """Command to generate a height map from a 3D model."""
    
    def __init__(self, args: Dict):
        """
        Initialize the command.
        
        Args:
            args: Command-line arguments
        """
        self.args = args
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def execute(self) -> int:
        """
        Execute the command.
        
        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        try:
            # Validate arguments
            validate_arguments(self.args)
            
            # Create configuration
            config = ApplicationConfig.from_dict(self.args)
            
            # Create processor
            processor = create_processor(config)
            
            # Process model
            with processor:
                output_path = processor.process()
            
            self.logger.info(f"Height map generated successfully: {output_path}")
            return 0
            
        except HeightcraftError as e:
            self.logger.error(f"Error: {e}")
            return 1
        except Exception as e:
            self.logger.exception(f"Unexpected error: {e}")
            return 2


class RunTestsCommand(Command):
    """Command to run tests."""
    
    def __init__(self, test_args: Optional[List[str]] = None):
        """
        Initialize the command.
        
        Args:
            test_args: Arguments to pass to pytest
        """
        self.test_args = test_args or ["-v", "tests"]
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def execute(self) -> int:
        """
        Execute the command.
        
        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        try:
            self.logger.info(f"Running tests with arguments: {self.test_args}")
            return pytest.main(self.test_args)
        except Exception as e:
            self.logger.exception(f"Error running tests: {e}")
            return 2


def create_command(args: Optional[List[str]] = None) -> Command:
    """
    Create a command based on command-line arguments.
    
    Args:
        args: Command-line arguments (defaults to sys.argv[1:])
        
    Returns:
        Command to execute
    """
    # Parse arguments
    parsed_args = parse_arguments(args)
    setup_logging(parsed_args.get('verbose', 0))
    
    logging.debug(f"CLI main called with arguments: {parsed_args}")
    
    # Create command
    if parsed_args.get("test", False):
        return RunTestsCommand()
    else:
        return GenerateHeightMapCommand(parsed_args)


def main(args=None):
    """
    Main CLI entry point.
    
    Args:
        args: Command line arguments
        
    Returns:
        Exit code
    """    
    try:
        # Create and execute command
        command = create_command(args)
        logging.debug(f"Command: {command}")
        
        return command.execute()
    except Exception as e:
        logging.error(f"Unhandled exception in main: {e}", exc_info=True)
        return 3


if __name__ == "__main__":
    sys.exit(main()) 