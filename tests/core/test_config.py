import unittest
from heightcraft.core.config import ApplicationConfig

class TestConfig(unittest.TestCase):
    def test_config_creation_from_args(self):
        """Test that ApplicationConfig handles None values correctly during creation."""
        # Arguments simulating the training command where file_path is None
        args = {
            'file_path': None,
            'output_path': None,
            'max_resolution': 256,
            'bit_depth': 16,
            'split': 1,
            'use_gpu': False,
            'num_samples': 500000,
            'num_threads': 4,
            'large_model': False,
            'chunk_size': 1000000,
            'max_memory': 0.8,
            'cache_dir': None,
            'upscale': False,
            'upscale_factor': 2,
            'pretrained_model': None,
            'train': True,
            'dataset_path': 'trainer_models',
            'epochs': 10,
            'batch_size': 16,
            'learning_rate': 0.0001,
            'test': False,
            'verbose': 2
        }
        
        try:
            config = ApplicationConfig.from_dict(args)
        except TypeError as e:
            self.fail(f"ApplicationConfig.from_dict raised TypeError unexpectedly: {e}")

    def test_config_use_gpu_flag(self):
        """Test that ApplicationConfig correctly reads use_gpu flag."""
        # Case 1: use_gpu=True
        args_gpu = {
            'file_path': 'dummy.obj',
            'use_gpu': True,
            'no_gpu': False # This key shouldn't matter with the fix, but simulating CLI args
        }
        config_gpu = ApplicationConfig.from_dict(args_gpu)
        self.assertTrue(config_gpu.sampling_config.use_gpu, "use_gpu should be True when passed as True")

        # Case 2: use_gpu=False
        args_cpu = {
            'file_path': 'dummy.obj',
            'use_gpu': False
        }
        config_cpu = ApplicationConfig.from_dict(args_cpu)
        self.assertFalse(config_cpu.sampling_config.use_gpu, "use_gpu should be False when passed as False")
        
        # Case 3: Default (missing key) -> False (as per get('use_gpu', False))
        args_default = {
            'file_path': 'dummy.obj'
        }
        config_default = ApplicationConfig.from_dict(args_default)
        self.assertFalse(config_default.sampling_config.use_gpu, "use_gpu should default to False")

if __name__ == '__main__':
    unittest.main()
