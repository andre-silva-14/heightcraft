import os
import tempfile
from unittest.mock import MagicMock, patch
import pytest
from heightcraft.core.exceptions import ValidationError, MeshValidationError
from heightcraft.utils import validators

class TestValidators:
    
    def test_validate_positive_integer(self):
        assert validators.validate_positive_integer(5) == 5
        assert validators.validate_positive_integer("10") == 10
        
        with pytest.raises(ValidationError):
            validators.validate_positive_integer(0)
        with pytest.raises(ValidationError):
            validators.validate_positive_integer(-5)
        with pytest.raises(ValidationError):
            validators.validate_positive_integer("abc")

    def test_validate_non_negative_integer(self):
        assert validators.validate_non_negative_integer(0) == 0
        assert validators.validate_non_negative_integer(5) == 5
        
        with pytest.raises(ValidationError):
            validators.validate_non_negative_integer(-1)

    def test_validate_range(self):
        assert validators.validate_range(0.5, 0.0, 1.0) == 0.5
        assert validators.validate_range(0.0, 0.0, 1.0) == 0.0
        assert validators.validate_range(1.0, 0.0, 1.0) == 1.0
        
        with pytest.raises(ValidationError):
            validators.validate_range(-0.1, 0.0, 1.0)
        with pytest.raises(ValidationError):
            validators.validate_range(1.1, 0.0, 1.0)

    def test_validate_choice(self):
        assert validators.validate_choice("a", ["a", "b"]) == "a"
        
        with pytest.raises(ValidationError):
            validators.validate_choice("c", ["a", "b"])

    def test_validate_file_exists(self):
        with tempfile.NamedTemporaryFile() as tmp:
            assert validators.validate_file_exists(tmp.name) == tmp.name
            
        with pytest.raises(ValidationError):
            validators.validate_file_exists("non_existent_file.txt")

    def test_validate_directory_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            assert validators.validate_directory_exists(tmpdir) == tmpdir
            
        with pytest.raises(ValidationError):
            validators.validate_directory_exists("non_existent_dir")

    def test_validate_file_extension(self):
        assert validators.validate_file_extension("test.png", [".png", ".jpg"]) == "test.png"
        assert validators.validate_file_extension("test.jpg", ["png", "jpg"]) == "test.jpg"
        
        with pytest.raises(ValidationError):
            validators.validate_file_extension("test.txt", [".png"])

    def test_validate_split_value(self):
        assert validators.validate_split_value(4) == 4
        assert validators.validate_split_value(9) == 9
        assert validators.validate_split_value(12) == 12  # 3x4
        
        with pytest.raises(ValidationError):
            validators.validate_split_value(5)  # Prime
        with pytest.raises(ValidationError):
            validators.validate_split_value(-1)

    def test_validate_bit_depth(self):
        assert validators.validate_bit_depth(8) == 8
        assert validators.validate_bit_depth(16) == 16
        
        with pytest.raises(ValidationError):
            validators.validate_bit_depth(32)  # Only 8 and 16 supported by this validator?
            # Let's check the code. It says [8, 16].
            # Wait, HeightMap supports 32, but maybe this validator is strict?
            # Yes, code: if int_value not in [8, 16]: raise

    def test_validate_upscale_factor(self):
        assert validators.validate_upscale_factor(2) == 2
        assert validators.validate_upscale_factor(4) == 4
        
        with pytest.raises(ValidationError):
            validators.validate_upscale_factor(5)

    def test_validate_max_memory(self):
        assert validators.validate_max_memory(0.5) == 0.5
        
        with pytest.raises(ValidationError):
            validators.validate_max_memory(1.5)

    def test_validate_mesh(self):
        mock_mesh = MagicMock()
        mock_mesh.vertices = [1, 2, 3]
        mock_mesh.faces = [1, 2, 3]
        
        validators.validate_mesh(mock_mesh)
        
        with pytest.raises(MeshValidationError):
            validators.validate_mesh(None)
            
        mock_mesh.vertices = []
        with pytest.raises(MeshValidationError):
            validators.validate_mesh(mock_mesh)
