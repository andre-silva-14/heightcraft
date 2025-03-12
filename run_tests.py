#!/usr/bin/env python3
"""
Run all the tests in the tests directory.

This script discovers and runs all tests in the tests directory.
"""

import os
import sys
import unittest


def run_tests():
    """
    Discover and run all tests in the tests directory.
    
    Returns:
        True if all tests pass, False otherwise.
    """
    # Discover tests
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover("tests", pattern="test_*.py")
    
    # Run tests
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    
    # Return True if all tests pass
    return result.wasSuccessful()


if __name__ == "__main__":
    # Add the project root to the Python path
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, project_root)
    
    # Run tests
    success = run_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1) 