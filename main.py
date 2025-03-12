#!/usr/bin/env python3
"""
Heightcraft - A powerful and flexible height map generator.

This is the entry point script for the Heightcraft application.
This script simply delegates to the main function in the heightcraft package.
"""

import sys
from heightcraft.main import main

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--legacy":
        from legacy.main import main as legacy_main
        sys.exit(legacy_main(sys.argv[2:]))
    else:
        sys.exit(main(sys.argv[1:]))
