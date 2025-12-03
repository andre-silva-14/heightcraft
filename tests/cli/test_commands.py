import pytest
from unittest.mock import MagicMock, patch
import sys
from heightcraft.cli.commands import GenerateHeightMapCommand, TrainUpscalerCommand, create_command, main
from heightcraft.core.exceptions import HeightcraftError

class TestGenerateHeightMapCommand:
    def test_execute_success(self):
        args = {
            "input": "test.glb",
            "output": "output.png",
            "resolution": 1024
        }
        command = GenerateHeightMapCommand(args)
        
        mock_processor = MagicMock()
        mock_processor.process.return_value = "output.png"
        
        with patch('heightcraft.cli.commands.validate_arguments'), \
             patch('heightcraft.core.config.ApplicationConfig.from_dict'), \
             patch('heightcraft.processors.create_processor', return_value=mock_processor):
            
            exit_code = command.execute()
            
            assert exit_code == 0
            mock_processor.process.assert_called_once()

    def test_execute_validation_error(self):
        args = {}
        command = GenerateHeightMapCommand(args)
        
        with patch('heightcraft.cli.commands.validate_arguments', side_effect=HeightcraftError("Invalid args")):
            exit_code = command.execute()
            assert exit_code == 1

    def test_execute_unexpected_error(self):
        args = {}
        command = GenerateHeightMapCommand(args)
        
        with patch('heightcraft.cli.commands.validate_arguments', side_effect=Exception("Unexpected")):
            exit_code = command.execute()
            assert exit_code == 2

class TestTrainUpscalerCommand:
    def test_execute_success(self):
        args = {
            "train": True,
            "dataset_path": "data/",
            "output": "model.pt"
        }
        command = TrainUpscalerCommand(args)
        
        mock_service = MagicMock()
        mock_service.train_model.return_value = "model.pt"
        
        with patch('heightcraft.services.training_service.TrainingService', return_value=mock_service), \
             patch('heightcraft.core.config.ApplicationConfig.from_dict'):
            
            exit_code = command.execute()
            
            assert exit_code == 0
            mock_service.train_model.assert_called_once()

    def test_execute_missing_dataset(self):
        args = {"train": True}
        command = TrainUpscalerCommand(args)
        exit_code = command.execute()
        assert exit_code == 1

class TestCreateCommand:
    def test_create_generate_command(self):
        with patch('heightcraft.cli.commands.parse_arguments', return_value={"train": False}):
            command = create_command([])
            assert isinstance(command, GenerateHeightMapCommand)

    def test_create_train_command(self):
        with patch('heightcraft.cli.commands.parse_arguments', return_value={"train": True}):
            command = create_command([])
            assert isinstance(command, TrainUpscalerCommand)

class TestMain:
    def test_main_success(self):
        mock_command = MagicMock()
        mock_command.execute.return_value = 0
        
        with patch('heightcraft.cli.commands.create_command', return_value=mock_command):
            exit_code = main([])
            assert exit_code == 0

    def test_main_exception(self):
        with patch('heightcraft.cli.commands.create_command', side_effect=Exception("Error")):
            exit_code = main([])
            assert exit_code == 3
