#!/usr/bin/env python3
"""
Heightcraft - A powerful and flexible height map generator.

This is the main entry point for the Heightcraft application.
It sets up logging and delegates to the appropriate command.
"""

import logging
import sys
from typing import List, Optional

from heightcraft.cli.commands import main as cli_main
from heightcraft.core.logging import setup_logging


def main(args: Optional[List[str]] = sys.argv[1:]) -> int:
    """
    Main entry point for the application.
    
    Args:
        args: Command-line arguments (defaults to sys.argv[1:])
        
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    # Set up logging (will be reconfigured by CLI with proper verbosity)
    setup_logging()
    
    try:
        # Delegate to CLI main
        return cli_main(args)
    except KeyboardInterrupt:
        logging.warning("Interrupted by user")
        return 130  # Standard exit code for SIGINT
    except Exception as e:
        logging.exception(f"Unhandled exception: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 