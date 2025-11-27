"""End-to-end tests for error handling."""
import unittest
import tempfile
from pathlib import Path
from heightcraft.main import main as heightcraft_main

class TestE2EErrors(unittest.TestCase):
    """End-to-end tests for error conditions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)
        self.output_file = self.test_dir / "output.png"
        
    def tearDown(self):
        """Tear down test fixtures."""
        self.temp_dir.cleanup()
        
    def test_missing_file(self):
        """Test that missing input file returns error."""
        args = ["non_existent_file.obj"]
        # Should return non-zero exit code
        exit_code = heightcraft_main(args)
        self.assertNotEqual(exit_code, 0)
        
    def test_invalid_resolution(self):
        """Test that invalid resolution returns error."""
        # Create a dummy file so file check passes
        dummy_file = self.test_dir / "dummy.obj"
        dummy_file.touch()
        
        args = [str(dummy_file), "--max_resolution", "-100"]
        # Should return error code 2 (handled exception)
        exit_code = heightcraft_main(args)
        self.assertEqual(exit_code, 2)
            
    def test_invalid_samples(self):
        """Test that invalid sample count returns error."""
        dummy_file = self.test_dir / "dummy.obj"
        dummy_file.touch()
        
        args = [str(dummy_file), "--num_samples", "0"]
        # Should return error code 2 (handled exception)
        exit_code = heightcraft_main(args)
        self.assertEqual(exit_code, 2)
            
    def test_invalid_split(self):
        """Test that invalid split value returns error."""
        dummy_file = self.test_dir / "dummy.obj"
        dummy_file.touch()
        
        # 3 is not a perfect square
        args = [str(dummy_file), "--split", "3"]
        with self.assertRaises(SystemExit):
            heightcraft_main(args)
            
    def test_invalid_upscale_factor(self):
        """Test that invalid upscale factor returns error."""
        dummy_file = self.test_dir / "dummy.obj"
        dummy_file.touch()
        
        args = [str(dummy_file), "--upscale", "--upscale_factor", "5"]
        with self.assertRaises(SystemExit):
            heightcraft_main(args)
