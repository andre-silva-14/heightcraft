import os
import pytest
import shutil
from pathlib import Path
from unittest.mock import patch

from heightcraft.infrastructure.file_storage import FileStorage
from heightcraft.core.exceptions import FileError

class TestFileStorage:
    @pytest.fixture
    def base_dir(self, tmp_path):
        return str(tmp_path)

    @pytest.fixture
    def file_storage(self, base_dir):
        return FileStorage(base_dir=base_dir)

    def test_init(self, base_dir):
        storage = FileStorage(base_dir=base_dir)
        assert storage.base_dir == base_dir

    def test_ensure_directory(self, file_storage, base_dir):
        dir_name = "test_dir"
        path = file_storage.ensure_directory(dir_name)
        
        assert os.path.isdir(path)
        assert path == os.path.join(base_dir, dir_name)

    def test_ensure_directory_error(self, file_storage):
        with patch('os.makedirs', side_effect=Exception("Mkdir error")):
            with pytest.raises(FileError, match="Failed to ensure directory exists"):
                file_storage.ensure_directory("test")

    def test_file_exists(self, file_storage, base_dir):
        file_path = os.path.join(base_dir, "test.txt")
        with open(file_path, "w") as f:
            f.write("content")
            
        assert file_storage.file_exists("test.txt")
        assert not file_storage.file_exists("non_existent.txt")

    def test_directory_exists(self, file_storage, base_dir):
        dir_path = os.path.join(base_dir, "test_dir")
        os.makedirs(dir_path)
        
        assert file_storage.directory_exists("test_dir")
        assert not file_storage.directory_exists("non_existent_dir")

    def test_list_files(self, file_storage, base_dir):
        # Create some files
        os.makedirs(os.path.join(base_dir, "subdir"))
        with open(os.path.join(base_dir, "subdir", "f1.txt"), "w") as f: f.write("")
        with open(os.path.join(base_dir, "subdir", "f2.log"), "w") as f: f.write("")
        
        files = file_storage.list_files("subdir")
        assert len(files) == 2
        assert any("f1.txt" in f for f in files)
        assert any("f2.log" in f for f in files)

    def test_list_files_pattern(self, file_storage, base_dir):
        os.makedirs(os.path.join(base_dir, "subdir"))
        with open(os.path.join(base_dir, "subdir", "f1.txt"), "w") as f: f.write("")
        with open(os.path.join(base_dir, "subdir", "f2.log"), "w") as f: f.write("")
        
        files = file_storage.list_files("subdir", pattern="*.txt")
        assert len(files) == 1
        assert "f1.txt" in files[0]

    def test_list_files_error(self, file_storage):
        with pytest.raises(FileError, match="Directory does not exist"):
            file_storage.list_files("non_existent")

    def test_delete_file(self, file_storage, base_dir):
        file_path = os.path.join(base_dir, "test.txt")
        with open(file_path, "w") as f: f.write("")
        
        file_storage.delete_file("test.txt")
        assert not os.path.exists(file_path)

    def test_delete_file_not_exist(self, file_storage):
        # Should not raise
        file_storage.delete_file("non_existent.txt")

    def test_delete_file_error(self, file_storage, base_dir):
        file_path = os.path.join(base_dir, "test.txt")
        with open(file_path, "w") as f: f.write("")
        
        with patch('os.remove', side_effect=Exception("Remove error")):
            with pytest.raises(FileError, match="Failed to delete file"):
                file_storage.delete_file("test.txt")

    def test_clear_directory(self, file_storage, base_dir):
        dir_path = os.path.join(base_dir, "subdir")
        os.makedirs(dir_path)
        with open(os.path.join(dir_path, "f1.txt"), "w") as f: f.write("")
        os.makedirs(os.path.join(dir_path, "subsubdir"))
        
        file_storage.clear_directory("subdir")
        
        assert os.path.exists(dir_path)
        assert not os.path.exists(os.path.join(dir_path, "f1.txt"))
        assert not os.path.exists(os.path.join(dir_path, "subsubdir"))

    def test_clear_directory_not_exist(self, file_storage):
        # Should not raise
        file_storage.clear_directory("non_existent")

    def test_clear_directory_error(self, file_storage, base_dir):
        dir_path = os.path.join(base_dir, "subdir")
        os.makedirs(dir_path)
        
        with patch('os.listdir', side_effect=Exception("Listdir error")):
            with pytest.raises(FileError, match="Failed to clear directory"):
                file_storage.clear_directory("subdir")

    def test_get_file_size(self, file_storage, base_dir):
        file_path = os.path.join(base_dir, "test.txt")
        content = "hello"
        with open(file_path, "w") as f: f.write(content)
        
        assert file_storage.get_file_size("test.txt") == len(content)

    def test_get_file_size_error(self, file_storage):
        with pytest.raises(FileError, match="File does not exist"):
            file_storage.get_file_size("non_existent.txt")

    def test_get_file_extension(self, file_storage):
        assert file_storage.get_file_extension("test.txt") == ".txt"
        assert file_storage.get_file_extension("path/to/file.tar.gz") == ".gz"

    def test_join_paths(self, file_storage):
        assert file_storage.join_paths("a", "b") == "a/b"
