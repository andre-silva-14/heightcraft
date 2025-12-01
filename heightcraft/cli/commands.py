"""
Commands for Heightcraft CLI.

This module provides command classes that implement the Command pattern for the CLI.
Each command encapsulates a specific action that can be executed by the CLI.
"""

import abc
import logging
import sys
from typing import Dict, List, Optional



from heightcraft.cli.argument_parser import parse_arguments, validate_arguments
from heightcraft.core.config import ApplicationConfig
from heightcraft.core.exceptions import HeightcraftError
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
            # Lazy import to avoid heavy dependencies at startup
            from heightcraft.processors import create_processor
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




class TrainUpscalerCommand(Command):
    """Command to train an upscaling model."""
    
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
            # Lazy import to avoid TensorFlow overhead/crashes when not training
            from heightcraft.services.training_service import TrainingService
            
            # Validate arguments
            if not self.args.get("dataset_path"):
                self.logger.error("Dataset path is required for training")
                return 1
            
            # Create configuration
            config = ApplicationConfig.from_dict(self.args)
            
            # Create service
            service = TrainingService(config.upscale_config)
            
            # Run training
            output_path = service.train_model(
                dataset_path=config.training_config.dataset_path,
                output_model_path=config.training_config.output_model_path,
                epochs=config.training_config.epochs,
                batch_size=config.training_config.batch_size,
                learning_rate=config.training_config.learning_rate
            )
            
            self.logger.info(f"Model trained successfully: {output_path}")
            return 0
            
        except HeightcraftError as e:
            self.logger.error(f"Error: {e}")
            return 1
        except Exception as e:
            self.logger.exception(f"Unexpected error: {e}")
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

    if parsed_args.get("train", False):
        return TrainUpscalerCommand(parsed_args)
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