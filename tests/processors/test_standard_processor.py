import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from heightcraft.processors.standard_processor import StandardProcessor
from heightcraft.core.config import ApplicationConfig, ModelConfig, HeightMapConfig, SamplingConfig, OutputConfig
from heightcraft.domain.mesh import Mesh
from heightcraft.domain.point_cloud import PointCloud
from heightcraft.domain.height_map import HeightMap

class TestStandardProcessor(unittest.TestCase):
    def setUp(self):
        self.config = ApplicationConfig(
            model_config=ModelConfig(file_path="test.obj"),
            height_map_config=HeightMapConfig(max_resolution=100, bit_depth=16),
            sampling_config=SamplingConfig(num_samples=1000, use_gpu=False, num_threads=1),
            output_config=OutputConfig(output_path="output.png")
        )
        self.processor = StandardProcessor(self.config)
        
        # Mock services
        self.processor.mesh_service = MagicMock()
        self.processor.height_map_service = MagicMock()
        self.processor.sampling_service = MagicMock()

    def test_load_model(self):
        mock_mesh = MagicMock(spec=Mesh)
        mock_mesh.vertices = []
        mock_mesh.faces = []
        self.processor.mesh_service.load_mesh.return_value = mock_mesh
        self.processor.mesh_service.prepare_mesh.return_value = mock_mesh
        
        self.processor.load_model()
        
        self.processor.mesh_service.load_mesh.assert_called_with("test.obj")
        self.processor.mesh_service.prepare_mesh.assert_called_with(mock_mesh, center=True, align=True)
        self.assertEqual(self.processor.mesh, mock_mesh)

    def test_sample_points(self):
        self.processor.mesh = MagicMock(spec=Mesh)
        mock_points = np.zeros((100, 3))
        self.processor.sampling_service.sample_from_mesh.return_value = mock_points
        
        points = self.processor.sample_points()
        
        self.processor.sampling_service.sample_from_mesh.assert_called_with(self.processor.mesh)
        np.testing.assert_array_equal(points, mock_points)
        np.testing.assert_array_equal(self.processor.points, mock_points)

    def test_generate_height_map(self):
        self.processor.mesh = MagicMock(spec=Mesh)
        self.processor.points = np.zeros((100, 3))
        
        # Mock resolution calculation
        with patch('heightcraft.utils.resolution_calculator.ResolutionCalculator') as MockCalculator:
            mock_calculator = MockCalculator.return_value
            mock_calculator.calculate_resolution_from_bounds.return_value = (100, 100)
            
            # Mock height map generation
            mock_height_map_obj = MagicMock(spec=HeightMap)
            mock_height_map_obj.data = np.zeros((100, 100))
            self.processor.height_map_service.generate_from_point_cloud.return_value = mock_height_map_obj
            
            height_map_data = self.processor.generate_height_map()
            
            mock_calculator.calculate_resolution_from_bounds.assert_called()
        
        # Verify generate_from_point_cloud call
        # We can't easily check the PointCloud argument equality, but we can check other args
        args, kwargs = self.processor.height_map_service.generate_from_point_cloud.call_args
        self.assertIsInstance(args[0], PointCloud)
        self.assertEqual(args[1], (100, 100))
        self.assertEqual(kwargs['bit_depth'], 16)
        self.assertEqual(kwargs['num_threads'], 1)
        
        np.testing.assert_array_equal(height_map_data, mock_height_map_obj.data)

if __name__ == '__main__':
    unittest.main()
