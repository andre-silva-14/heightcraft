
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import trimesh
from heightcraft.core.config import ApplicationConfig, ModelConfig, ProcessingMode
from heightcraft.processors.large_model_processor import LargeModelProcessor

class TestChunkSizeReproduction(unittest.TestCase):
    def setUp(self):
        # Create a dummy mesh with 100 vertices
        self.vertices = np.random.rand(100, 3)
        self.faces = np.random.randint(0, 100, (50, 3))
        self.mesh = trimesh.Trimesh(vertices=self.vertices, faces=self.faces)

    @patch('trimesh.load')
    def test_chunk_size_with_scene(self, mock_load):
        # Create a real scene with 10 meshes
        scene = trimesh.Scene()
        for i in range(10):
            mesh = trimesh.Trimesh(
                vertices=np.random.rand(100, 3),
                faces=np.random.randint(0, 100, (50, 3))
            )
            scene.add_geometry(mesh, node_name=f"node_{i}", geom_name=f"mesh_{i}")
            
        mock_load.return_value = scene
        
        # Test case 1: chunk_size = 1000 (larger than any single mesh)
        # Expected: 1 chunk per mesh = 10 chunks
        config1 = ApplicationConfig.from_dict({
            "file_path": "dummy.obj",
            "large_model": True,
            "chunk_size": 1000,
            "num_samples": 100,
            "max_resolution": 100,
            "output_path": "out.png"
        })
        
        processor1 = LargeModelProcessor(config1)
        processor1.load_model()
        chunks1 = len(processor1.chunks)
        print(f"Chunk size 1000 -> {chunks1} chunks")
        
        # Test case 2: chunk_size = 500 (still larger than any single mesh)
        # Expected: 1 chunk per mesh = 10 chunks
        config2 = ApplicationConfig.from_dict({
            "file_path": "dummy.obj",
            "large_model": True,
            "chunk_size": 500,
            "num_samples": 100,
            "max_resolution": 100,
            "output_path": "out.png"
        })
        
        processor2 = LargeModelProcessor(config2)
        processor2.load_model()
        chunks2 = len(processor2.chunks)
        print(f"Chunk size 500 -> {chunks2} chunks")
        
        # Test case 3: chunk_size = 10 (smaller than mesh)
        # Expected: 100 // 10 = 10 chunks per mesh * 10 meshes = 100 chunks
        config3 = ApplicationConfig.from_dict({
            "file_path": "dummy.obj",
            "large_model": True,
            "chunk_size": 10,
            "num_samples": 100,
            "max_resolution": 100,
            "output_path": "out.png"
        })
        
        processor3 = LargeModelProcessor(config3)
        processor3.load_model()
        chunks3 = len(processor3.chunks)
        print(f"Chunk size 10 -> {chunks3} chunks")

        self.assertEqual(chunks1, 10, "Expected 10 chunks for chunk_size 1000")
        self.assertEqual(chunks2, 10, "Expected 10 chunks for chunk_size 500")
        self.assertEqual(chunks1, chunks2, "Chunk count should be same when chunk_size > mesh size")
        self.assertGreater(chunks3, chunks1, "Chunk count should increase when chunk_size < mesh size")

if __name__ == '__main__':
    unittest.main()
