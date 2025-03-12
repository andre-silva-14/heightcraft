"""
Tests for the ResourceManager.
"""

import os
import unittest
from unittest.mock import Mock, patch, MagicMock

from heightcraft.core.exceptions import ResourceError
from heightcraft.infrastructure.resource_manager import ResourceManager
from tests.base_test_case import BaseTestCase


class TestResourceManager(BaseTestCase):
    """Tests for the ResourceManager."""
    
    def setUp(self) -> None:
        """Set up test environment."""
        super().setUp()
        
        # Create an instance of ResourceManager
        self.resource_manager = ResourceManager()
        
        # Test directory paths
        self.test_directory = self.get_temp_path("resources")
        os.makedirs(self.test_directory, exist_ok=True)
        
        # Create some test files
        self.test_files = [
            os.path.join(self.test_directory, "file1.txt"),
            os.path.join(self.test_directory, "file2.txt"),
            os.path.join(self.test_directory, "data.json")
        ]
        for file_path in self.test_files:
            with open(file_path, 'w') as f:
                f.write(f"Content of {os.path.basename(file_path)}")
        
        # Create a test subdirectory
        self.test_subdirectory = os.path.join(self.test_directory, "subdir")
        os.makedirs(self.test_subdirectory, exist_ok=True)
        
        # Create a test file in the subdirectory
        self.test_subdir_file = os.path.join(self.test_subdirectory, "subfile.txt")
        with open(self.test_subdir_file, 'w') as f:
            f.write("Content of subfile.txt")
    
    def tearDown(self) -> None:
        """Clean up after each test."""
        super().tearDown()
        
        # Remove test files and directories
        for file_path in self.test_files:
            if os.path.exists(file_path):
                os.remove(file_path)
        
        if os.path.exists(self.test_subdir_file):
            os.remove(self.test_subdir_file)
        
        if os.path.exists(self.test_subdirectory):
            os.rmdir(self.test_subdirectory)
        
        if os.path.exists(self.test_directory):
            os.rmdir(self.test_directory)
    
    def test_get_resource_path(self) -> None:
        """Test getting the path to a resource."""
        # Set up the resource path
        self.resource_manager.base_directory = self.test_directory
        
        # Call the method
        resource_path = self.resource_manager.get_resource_path("file1.txt")
        
        # Check the returned path
        self.assertEqual(resource_path, os.path.join(self.test_directory, "file1.txt"))
        
        # Test with a subdirectory
        resource_path = self.resource_manager.get_resource_path("subdir/subfile.txt")
        self.assertEqual(resource_path, os.path.join(self.test_directory, "subdir", "subfile.txt"))
    
    def test_resource_exists(self) -> None:
        """Test checking if a resource exists."""
        # Set up the resource path
        self.resource_manager.base_directory = self.test_directory
        
        # Call the method for an existing resource
        result = self.resource_manager.resource_exists("file1.txt")
        self.assertTrue(result)
        
        # Call the method for a non-existent resource
        result = self.resource_manager.resource_exists("non_existent.txt")
        self.assertFalse(result)
        
        # Call the method for an existing resource in a subdirectory
        result = self.resource_manager.resource_exists("subdir/subfile.txt")
        self.assertTrue(result)
    
    def test_load_resource(self) -> None:
        """Test loading a resource."""
        # Set up the resource path
        self.resource_manager.base_directory = self.test_directory
        
        # Call the method
        content = self.resource_manager.load_resource("file1.txt")
        
        # Check the returned content
        self.assertEqual(content, "Content of file1.txt")
        
        # Test with a non-existent resource
        with self.assertRaises(ResourceError):
            self.resource_manager.load_resource("non_existent.txt")
    
    def test_load_json_resource(self) -> None:
        """Test loading a JSON resource."""
        # Create a test JSON file
        json_file = os.path.join(self.test_directory, "config.json")
        with open(json_file, 'w') as f:
            f.write('{"key1": "value1", "key2": 42}')
        
        # Set up the resource path
        self.resource_manager.base_directory = self.test_directory
        
        # Call the method
        json_data = self.resource_manager.load_json_resource("config.json")
        
        # Check the returned JSON data
        self.assertIsInstance(json_data, dict)
        self.assertEqual(json_data["key1"], "value1")
        self.assertEqual(json_data["key2"], 42)
        
        # Test with a non-existent resource
        with self.assertRaises(ResourceError):
            self.resource_manager.load_json_resource("non_existent.json")
        
        # Test with an invalid JSON file
        invalid_json_file = os.path.join(self.test_directory, "invalid.json")
        with open(invalid_json_file, 'w') as f:
            f.write('{"key1": "value1", invalid json')
        
        with self.assertRaises(ResourceError):
            self.resource_manager.load_json_resource("invalid.json")
        
        # Clean up test files
        if os.path.exists(json_file):
            os.remove(json_file)
        if os.path.exists(invalid_json_file):
            os.remove(invalid_json_file)
    
    def test_save_resource(self) -> None:
        """Test saving a resource."""
        # Set up the resource path
        self.resource_manager.base_directory = self.test_directory
        
        # Call the method
        content = "New content"
        result = self.resource_manager.save_resource("new_file.txt", content)
        
        # Check the result
        self.assertTrue(result)
        
        # Check that the file was created with the correct content
        file_path = os.path.join(self.test_directory, "new_file.txt")
        self.assertTrue(os.path.exists(file_path))
        with open(file_path, 'r') as f:
            saved_content = f.read()
        self.assertEqual(saved_content, content)
        
        # Clean up test file
        if os.path.exists(file_path):
            os.remove(file_path)
    
    def test_save_resource_with_invalid_path(self) -> None:
        """Test saving a resource with an invalid path."""
        # Set up the resource path
        self.resource_manager.base_directory = self.test_directory
        
        # Modify the mock to simulate a file write error
        with patch('builtins.open', side_effect=IOError("Test write error")):
            # Call the method with a path that causes the mocked error
            with self.assertRaises(ResourceError):
                self.resource_manager.save_resource("any_path.txt", "content")
    
    def test_list_resources(self) -> None:
        """Test listing resources."""
        # Set up the resource path
        self.resource_manager.base_directory = self.test_directory
        
        # Create some test files and directories
        file1 = os.path.join(self.test_directory, "file1.txt")
        with open(file1, 'w') as f:
            f.write("File 1 content")
        
        file2 = os.path.join(self.test_directory, "file2.txt")
        with open(file2, 'w') as f:
            f.write("File 2 content")
        
        file3 = os.path.join(self.test_directory, "file3.png")
        with open(file3, 'w') as f:
            f.write("File 3 content")
        
        os.makedirs(os.path.join(self.test_directory, "subdir"), exist_ok=True)
        subfile = os.path.join(self.test_directory, "subdir", "subfile.txt")
        with open(subfile, 'w') as f:
            f.write("Subfile content")
        
        # Test with no pattern or directory
        resources = self.resource_manager.list_resources()
        self.assertEqual(len(resources), 5)  # 3 files + 1 directory + subdir/subfile.txt
        
        # Resource paths should be relative to the base directory
        self.assertIn("file1.txt", resources)
        self.assertIn("file2.txt", resources)
        self.assertIn("file3.png", resources)
        self.assertIn("subdir", resources)
        
        # Test with a file pattern
        resources = self.resource_manager.list_resources("*.txt")
        self.assertEqual(len(resources), 2)  # 2 text files
        self.assertIn("file1.txt", resources)
        self.assertIn("file2.txt", resources)
        
        # Test with a subdirectory
        resources = self.resource_manager.list_resources(directory="subdir")
        self.assertEqual(len(resources), 1)  # 1 file in subdirectory
        self.assertIn("subdir/subfile.txt", resources)  # Path is relative to base directory
    
    def test_delete_resource(self) -> None:
        """Test deleting a resource."""
        # Set up the resource path
        self.resource_manager.base_directory = self.test_directory
        
        # Create a file to delete
        delete_file = os.path.join(self.test_directory, "to_delete.txt")
        with open(delete_file, 'w') as f:
            f.write("Content to delete")
        
        # Call the method
        result = self.resource_manager.delete_resource("to_delete.txt")
        
        # Check the result
        self.assertTrue(result)
        
        # Check that the file was deleted
        self.assertFalse(os.path.exists(delete_file))
        
        # Test with a non-existent resource - should still return True
        # since the behavior has changed to return True if the resource is already gone
        result = self.resource_manager.delete_resource("non_existent.txt")
        self.assertTrue(result)
    
    def test_create_directory(self) -> None:
        """Test creating a directory."""
        # Set up the resource path
        self.resource_manager.base_directory = self.test_directory
        
        # Call the method
        new_dir = "new_directory"
        result = self.resource_manager.create_directory(new_dir)
        
        # Check the result
        self.assertTrue(result)
        
        # Check that the directory was created
        dir_path = os.path.join(self.test_directory, new_dir)
        self.assertTrue(os.path.exists(dir_path))
        self.assertTrue(os.path.isdir(dir_path))
        
        # Clean up test directory
        if os.path.exists(dir_path):
            os.rmdir(dir_path)
    
    def test_get_directory_size(self) -> None:
        """Test getting the size of a directory."""
        # Set up the resource path
        self.resource_manager.base_directory = self.test_directory
        
        # Call the method
        size = self.resource_manager.get_directory_size()
        
        # Check the returned size
        self.assertIsInstance(size, int)
        self.assertGreater(size, 0)  # Should have some size due to test files
        
        # Test with a subdirectory
        size = self.resource_manager.get_directory_size("subdir")
        self.assertGreater(size, 0)  # Should have some size due to test file in subdirectory
        
        # Test with a non-existent directory
        with self.assertRaises(ResourceError):
            self.resource_manager.get_directory_size("non_existent_dir") 