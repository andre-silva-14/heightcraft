"""
Tests for UpscalingService coverage improvement.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import torch

from heightcraft.services.upscaling_service import UpscalingService
from heightcraft.core.config import UpscaleConfig
from heightcraft.core.exceptions import UpscalingError
from heightcraft.domain.height_map import HeightMap
from tests.base_test_case import BaseTestCase

class TestUpscalingServiceCoverage(BaseTestCase):
    """Tests for UpscalingService coverage."""

    def setUp(self) -> None:
        super().setUp()
        self.config = UpscaleConfig(pretrained_model="model.pt", enabled=True)
        self.upscaling_service = UpscalingService(config=self.config)
        self.mock_height_map = Mock(spec=HeightMap)
        self.mock_height_map.data = np.zeros((10, 10))
        self.mock_height_map.bit_depth = 16
        self.mock_height_map.width = 10
        self.mock_height_map.height = 10

    @patch('heightcraft.services.upscaling_service.UpscalerModel')
    @patch('heightcraft.services.upscaling_service.torch')
    def test_upscale_with_model_success(self, mock_torch, MockModel):
        # Setup mocks
        mock_model_instance = MockModel.return_value
        mock_model_instance.parameters.return_value = iter([Mock(device='cpu')])
        
        # Mock torch.load
        mock_torch.load.return_value = {}
        mock_torch.device.return_value = 'cpu'
        mock_torch.cuda.is_available.return_value = False
        
        # Mock tensor operations
        mock_tensor = Mock()
        mock_torch.from_numpy.return_value.float.return_value.unsqueeze.return_value.unsqueeze.return_value = mock_tensor
        mock_tensor.to.return_value = mock_tensor
        
        # Mock model output
        mock_output = Mock()
        mock_model_instance.return_value = mock_output
        mock_output.cpu.return_value.numpy.return_value = np.zeros((1, 1, 20, 20))
        
        # Mock file existence
        self.upscaling_service.file_storage = Mock()
        self.upscaling_service.file_storage.file_exists.return_value = True
        
        # Call method
        result = self.upscaling_service.upscale(self.mock_height_map, scale_factor=2)
        
        # Verify
        self.assertIsInstance(result, HeightMap)
        self.assertEqual(result.width, 20)
        self.assertEqual(result.height, 20)
        MockModel.assert_called_once()

    @patch('heightcraft.services.upscaling_service.UpscalerModel')
    @patch('heightcraft.services.upscaling_service.torch')
    def test_load_model_error(self, mock_torch, MockModel):
        # Mock file existence failure
        self.upscaling_service.file_storage = Mock()
        self.upscaling_service.file_storage.file_exists.return_value = False
        
        with self.assertRaises(UpscalingError):
            self.upscaling_service._load_model(use_gpu=False)

    @patch('heightcraft.services.upscaling_service.UpscalerModel')
    @patch('heightcraft.services.upscaling_service.torch')
    def test_upscale_with_model_error(self, mock_torch, MockModel):
        # Mock file existence
        self.upscaling_service.file_storage = Mock()
        self.upscaling_service.file_storage.file_exists.return_value = True
        
        # Make torch.load fail
        mock_torch.load.side_effect = Exception("Load error")
        
        with self.assertRaises(UpscalingError):
            self.upscaling_service.upscale(self.mock_height_map, scale_factor=2)

    @patch('heightcraft.services.upscaling_service.UpscalerModel')
    @patch('heightcraft.services.upscaling_service.torch')
    def test_create_default_model_success(self, mock_torch, MockModel):
        # Setup mocks
        mock_model_instance = MockModel.return_value
        mock_model_instance.state_dict.return_value = {}
        
        # Call method
        result = UpscalingService.create_default_model("test_model.pt")
        
        # Verify
        self.assertEqual(result, "test_model.pt")
        mock_torch.save.assert_called_once()

    @patch('heightcraft.services.upscaling_service.UpscalerModel')
    def test_create_default_model_error(self, MockModel):
        MockModel.side_effect = Exception("Create error")
        with self.assertRaises(UpscalingError):
            UpscalingService.create_default_model("test_model.pt")

    def test_upscale_disabled(self):
        import dataclasses
        new_config = dataclasses.replace(self.config, enabled=False)
        self.upscaling_service.config = new_config
        result = self.upscaling_service.upscale(self.mock_height_map)
        self.assertEqual(result, self.mock_height_map)

    def test_upscale_generic_error(self):
        # Force an error by passing invalid data type that causes exception before anything else
        # Or mock internal method to raise
        import dataclasses
        new_config = dataclasses.replace(self.config, pretrained_model=None)
        self.upscaling_service.config = new_config
        with patch.object(self.upscaling_service, '_upscale_with_interpolation', side_effect=Exception("Generic error")):
            with self.assertRaises(UpscalingError):
                self.upscaling_service.upscale(self.mock_height_map, scale_factor=2)
