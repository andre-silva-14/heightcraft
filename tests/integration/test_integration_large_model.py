"""
Integration tests for LargeModelProcessor.
"""

import os
import unittest
import numpy as np
import trimesh
from unittest.mock import MagicMock, patch

from heightcraft.core.config import ApplicationConfig, ModelConfig, SamplingConfig, HeightMapConfig, OutputConfig, ProcessingMode
from heightcraft.processors.large_model_processor import LargeModelProcessor
from heightcraft.services.mesh_service import MeshService
from heightcraft.services.model_service import ModelService
from heightcraft.services.height_map_service import HeightMapService
from heightcraft.services.sampling_service import SamplingService
from heightcraft.domain.mesh import Mesh

class TestIntegrationLargeModel(unittest.TestCase):
    """Integration tests for LargeModelProcessor."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = "tests/temp_integration"
        os.makedirs(self.test_dir, exist_ok=True)
        
        # Create a synthetic "large" model (many vertices)
        # We'll use a simple grid but with enough vertices to trigger chunking if we set chunk_size low
        self.model_path = os.path.join(self.test_dir, "large_model.obj")
        self._create_synthetic_model(self.model_path, grid_size=20)
        
        # Configure application
        model_config = ModelConfig(
            file_path=self.model_path,
            chunk_size=100, # Small chunk size to force chunking
            cache_dir=os.path.join(self.test_dir, "cache")
        )
        sampling_config = SamplingConfig(
            num_samples=1000,
            use_gpu=False,
            num_threads=2
        )
        height_map_config = HeightMapConfig(
            max_resolution=128,
            bit_depth=16
        )
        output_config = OutputConfig(
            output_path=os.path.join(self.test_dir, "output.png")
        )
        
        self.config = ApplicationConfig(
            model_config=model_config,
            sampling_config=sampling_config,
            height_map_config=height_map_config,
            output_config=output_config
        )
        
        # Initialize services
        # We don't need to inject them anymore as LargeModelProcessor creates them
        # But we keep them here if we need to patch them globally
        self.mesh_service = MeshService()
        self.model_service = ModelService()
        self.height_map_service = HeightMapService()
        self.sampling_service = SamplingService(sampling_config)
        
        self.processor = LargeModelProcessor(self.config)

    def tearDown(self):
        """Clean up."""
        import shutil
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def _create_synthetic_model(self, path, grid_size):
        """Create a synthetic model."""
        # Create a grid of vertices
        x = np.linspace(0, 1, grid_size)
        y = np.linspace(0, 1, grid_size)
        xv, yv = np.meshgrid(x, y)
        
        # Z is a simple function
        z = np.sin(xv * 10) * np.cos(yv * 10) * 0.1
        
        vertices = np.column_stack((xv.flatten(), yv.flatten(), z.flatten()))
        
        # Create faces
        faces = []
        for i in range(grid_size - 1):
            for j in range(grid_size - 1):
                # Two triangles per grid cell
                v0 = i * grid_size + j
                v1 = i * grid_size + (j + 1)
                v2 = (i + 1) * grid_size + j
                v3 = (i + 1) * grid_size + (j + 1)
                
                faces.append([v0, v1, v2])
                faces.append([v1, v3, v2])
        
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh.export(path)

    def test_chunked_sampling(self):
        """Test that sampling works with chunks and produces valid points."""
        # Load model
        self.processor.load_model()
        
        # Verify we have chunks
        self.assertTrue(len(self.processor.chunks) > 1, "Model should be chunked")
        
        # Sample points
        # We patch Mesh to ensure we are NOT creating a huge mesh from all vertices
        # The processor should create small meshes for chunks
        
        original_mesh_init = Mesh.__init__
        
        mesh_sizes = []
        
        def side_effect_mesh_init(self, trimesh_obj):
            mesh_sizes.append(len(trimesh_obj.vertices))
            original_mesh_init(self, trimesh_obj)
            
        with patch.object(Mesh, '__init__', side_effect=side_effect_mesh_init, autospec=True):
            points = self.processor.sample_points()
            
        # Verify points
        self.assertEqual(len(points), 1000)
        
        # Verify that we didn't create a mesh with ALL vertices (except maybe during initial load/validation if any)
        # Initial load creates a sample mesh for validation (first 10 chunks or so)
        # But during sampling, we should see multiple small mesh creations
        
        # Total vertices in our model: 20*20 = 400
        # Chunk size is 100. So we expect chunks of size ~100.
        # If we saw a mesh of size 400 created during sampling, that would be bad (if we had millions).
        # But here 400 is small.
        # Let's just verify that we called Mesh constructor multiple times with small sizes
        
        # We expect:
        # 1. Initial load might create a small mesh for validation.
        # 2. Sampling should create meshes for each chunk.
        
        # We expect multiple meshes to be created (one per chunk)
        # Since we have multiple chunks, we should see multiple mesh initializations
        self.assertTrue(len(mesh_sizes) > 1, f"Should create multiple meshes for chunks, got {len(mesh_sizes)}")
        
        # Also verify we didn't create a huge mesh (check sizes if possible, but just count is enough for now)
        self.assertEqual(len(mesh_sizes), len([c for c in self.processor.chunks if c['vertex_count'] > 0]))
        
        # Check that points are within bounds
        # Our model is in [0,1]x[0,1] roughly (after centering/aligning it might change)
        # Just check they are valid numbers
        self.assertFalse(np.any(np.isnan(points)))
        self.assertFalse(np.any(np.isinf(points)))

    def test_bounds_coverage(self):
        """Test that sampled points cover the full extent of the model."""
        # Create a model with known bounds: a box from (0,0,0) to (10,10,1)
        # We use a simple box composed of triangles
        vertices = np.array([
            [0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0], # Bottom
            [0, 0, 1], [10, 0, 1], [10, 10, 1], [0, 10, 1]  # Top
        ])
        # Just some faces to make it valid
        faces = np.array([
            [0, 1, 2], [0, 2, 3], # Bottom
            [4, 5, 6], [4, 6, 7], # Top
            [0, 1, 5], [0, 5, 4], # Front
            [1, 2, 6], [1, 6, 5], # Right
            [2, 3, 7], [2, 7, 6], # Back
            [3, 0, 4], [3, 4, 7]  # Left
        ])
        
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh.export(self.model_path)
        
        # Configure processor with small chunk size to force splitting
        # 8 vertices, so chunk_size=2 should give 4 chunks
        self.processor.model_config = ModelConfig(
            file_path=self.model_path,
            chunk_size=2,
            mode=ProcessingMode.LARGE
        )
        # Re-initialize to pick up config
        self.processor = LargeModelProcessor(self.config)
        # We need to update the internal chunk size manually or re-init properly
        self.processor.chunk_size = 2 
        
        self.processor.load_model()
        
        # Verify chunks
        self.assertTrue(len(self.processor.chunks) > 1, "Should be chunked")
        
        # Sample points
        points = self.processor.sample_points()
        
        # Check bounds of sampled points
        # The model is centered and aligned.
        # Original size: 10x10x1.
        # After centering, it should be roughly [-5, 5] x [-5, 5] x [-0.5, 0.5]
        # After alignment (if any), it might rotate.
        # But the extent (max - min) should be preserved for the dimensions.
        
        min_vals = np.min(points, axis=0)
        max_vals = np.max(points, axis=0)
        extents = max_vals - min_vals
        
        # We expect dimensions to be close to 10, 10, 1 (in some order)
        sorted_extents = np.sort(extents)
        
        # Allow some margin for sampling randomness
        self.assertTrue(np.isclose(sorted_extents[2], 10, atol=1.0), f"Max extent should be ~10, got {sorted_extents[2]}")
        self.assertTrue(np.isclose(sorted_extents[1], 10, atol=1.0), f"Mid extent should be ~10, got {sorted_extents[1]}")
        self.assertTrue(np.isclose(sorted_extents[0], 1, atol=0.5), f"Min extent should be ~1, got {sorted_extents[0]}")

if __name__ == '__main__':
    unittest.main()
