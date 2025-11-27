"""
Tests for the TrainingService using PyTorch.
"""

import os
import unittest
import shutil
from unittest.mock import Mock, patch
import torch
import numpy as np
from PIL import Image

from heightcraft.services.training_service import TrainingService
from heightcraft.core.config import UpscaleConfig
from tests.base_test_case import BaseTestCase

class TestTrainingServiceTorch(BaseTestCase):
    """Tests for the TrainingService."""
    
    def setUp(self) -> None:
        """Set up test environment."""
        super().setUp()
        
        self.config = UpscaleConfig()
        self.training_service = TrainingService(config=self.config)
        
        # Create dummy dataset
        self.dataset_dir = self.get_temp_path("dataset")
        os.makedirs(self.dataset_dir, exist_ok=True)
        
        # Create some dummy images
        for i in range(5):
            img = Image.fromarray(np.random.randint(0, 255, (256, 256), dtype=np.uint8))
            img.save(os.path.join(self.dataset_dir, f"img_{i}.png"))
            
        self.output_model = self.get_temp_path("model.pt")

    def test_train_model(self):
        """Test training the model."""
        # Train for 1 epoch with small batch size
        model_path = self.training_service.train_model(
            dataset_path=self.dataset_dir,
            output_model_path=self.output_model,
            epochs=1,
            batch_size=2,
            validation_split=0.2
        )
        
        self.assertTrue(os.path.exists(model_path))
        
        # Verify we can load it back
        model = torch.load(model_path)
        self.assertIsInstance(model, dict) # It saves state_dict
        
    def tearDown(self):
        super().tearDown()
        if os.path.exists(self.dataset_dir):
            shutil.rmtree(self.dataset_dir)
