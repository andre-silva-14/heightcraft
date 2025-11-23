"""
Tests for TrainingService.
"""

import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

# Setup mocks before importing
mock_tf = MagicMock()
mock_tf.keras.Model = MagicMock()
mock_tf.data.Dataset = MagicMock()

# Configure torch mock
mock_torch = MagicMock()
mock_props = MagicMock()
mock_props.total_memory = 8 * 1024 * 1024 * 1024.0  # 8 GB as float
mock_torch.cuda.get_device_properties.return_value = mock_props
mock_torch.cuda.device_count.return_value = 1

# Apply mocks to sys.modules
modules_patch = patch.dict('sys.modules', {
    'tensorflow': mock_tf, 
    'tensorflow.keras': mock_tf.keras,
    'torch': mock_torch
})
modules_patch.start()

# Import service after mocking
import heightcraft.services.training_service as ts_module
from heightcraft.services.training_service import TrainingService, TrainingError

class TestTrainingService(unittest.TestCase):
    """Test cases for TrainingService."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.dataset_path = os.path.join(self.test_dir, "dataset")
        os.makedirs(self.dataset_path)
        
        # Create dummy images
        for i in range(5):
            with open(os.path.join(self.dataset_path, f"image_{i}.png"), "w") as f:
                f.write("dummy data")
                
        self.service = TrainingService()
        
    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.test_dir)

    def test_get_image_paths(self):
        """Test finding images in directory."""
        paths = self.service._get_image_paths(self.dataset_path)
        self.assertEqual(len(paths), 5)
        
    def test_train_model_success(self):
        """Test successful training flow."""
        # Reset mocks
        mock_tf.reset_mock()
        
        # Mock dataset creation
        mock_dataset = MagicMock()
        mock_tf.data.Dataset.from_tensor_slices.return_value = mock_dataset
        
        # Mock model
        mock_model = MagicMock()
        # When tf.keras.Model(...) is called, return mock_model
        mock_tf.keras.Model.return_value = mock_model
        
        # Run training
        output_path = os.path.join(self.test_dir, "model.h5")
        
        # Ensure TF_AVAILABLE is True
        with patch.object(ts_module, 'TF_AVAILABLE', True):
            result = self.service.train_model(
                dataset_path=self.dataset_path,
                output_model_path=output_path,
                epochs=1
            )
        
        # Verify
        self.assertEqual(result, output_path)
        
        # Verify model compilation and training
        # Note: We check if the mock_model returned by tf.keras.Model() was used
        mock_model.compile.assert_called()
        mock_model.fit.assert_called()
        mock_model.save.assert_called_with(output_path)

    def test_train_model_no_tf(self):
        """Test error when TensorFlow is missing."""
        # Force TF_AVAILABLE to False
        with patch.object(ts_module, 'TF_AVAILABLE', False):
            with self.assertRaises(TrainingError):
                self.service.train_model(self.dataset_path, "model.h5")

if __name__ == '__main__':
    unittest.main()
