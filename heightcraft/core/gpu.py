class GPUManager:
    """
    Manager for GPU resources and operations.
    
    This class manages GPU resources and provides utilities for GPU-accelerated operations.
    It should only be initialized when explicitly requested by the user.
    """
    
    _instance = None
    
    @classmethod
    def get_instance(cls, force=False):
        """
        Get or create the singleton instance of the GPU manager.
        
        Args:
            force: Force initialization even if GPU usage wasn't requested
            
        Returns:
            The singleton instance or None if GPU not available
        """
        if cls._instance is None:
            # Determine if we should initialize based on args
            from heightcraft.cli.argument_parser import parse_arguments
            import sys
            import logging
            
            # Parse arguments to check for use_gpu flag
            try:
                args = parse_arguments(sys.argv[1:])
                use_gpu = args.get('use_gpu', False)
            except:
                use_gpu = False
            
            if use_gpu or force:
                logging.info("Initializing GPU manager as requested")
                cls._instance = cls()
            else:
                logging.info("GPU not requested, skipping initialization")
                return None
                
        return cls._instance
    
    def __init__(self):
        """Initialize the GPU manager."""
        import logging
        
        # Initialize GPU-related resources
        try:
            # Import necessary libraries
            import tensorflow as tf
            import torch
            
            # Check GPU availability
            self.tf_available = tf.test.is_gpu_available()
            self.torch_available = torch.cuda.is_available()
            
            if self.tf_available or self.torch_available:
                logging.info("GPU support enabled. Checking devices...")
                
                # Get device information
                if self.torch_available:
                    device_count = torch.cuda.device_count()
                    devices = []
                    
                    for i in range(device_count):
                        name = torch.cuda.get_device_name(i)
                        memory = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
                        devices.append((name, memory))
                    
                    logging.info(f"GPU support enabled. {device_count} device(s) found.")
                    for i, (name, memory) in enumerate(devices):
                        logging.info(f"  Device {i}: {name} ({memory:.2f} GB)")
            else:
                logging.warning("No GPU support detected.")
                
        except ImportError:
            logging.warning("GPU libraries not available. Using CPU only.")
            self.tf_available = False
            self.torch_available = False 