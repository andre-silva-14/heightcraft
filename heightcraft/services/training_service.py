"""
Training service for Heightcraft.

This module provides the TrainingService class for training AI upscaling models.
"""

import logging
import os
import glob
from typing import Optional, Tuple, List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from heightcraft.core.config import UpscaleConfig
from heightcraft.core.exceptions import HeightcraftError
from heightcraft.infrastructure.file_storage import FileStorage
from heightcraft.infrastructure.gpu_manager import gpu_manager
from heightcraft.services.upscaling_service import UpscalingService
from heightcraft.infrastructure.profiler import profiler
from heightcraft.models.upscaler import UpscalerModel


class TrainingError(HeightcraftError):
    """Exception raised for training errors."""
    pass


class HeightMapDataset(Dataset):
    """Dataset for height map training."""
    
    def __init__(self, image_paths: List[str]):
        self.image_paths = image_paths
        
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        path = self.image_paths[idx]
        
        try:
            # Load image
            img = Image.open(path).convert('L') # Convert to grayscale
            
            # Resize to fixed size (128x128)
            img = img.resize((128, 128), Image.BICUBIC)
            
            # Convert to numpy and normalize
            img_np = np.array(img).astype(np.float32) / 255.0
            
            # Create low-res input (downsample by 2)
            # We simulate low-res by resizing down then up? No, usually we train on LR -> HR
            # So we create LR by downsampling HR
            h, w = img_np.shape
            lr_img = Image.fromarray((img_np * 255).astype(np.uint8)).resize((h // 2, w // 2), Image.BICUBIC)
            lr_np = np.array(lr_img).astype(np.float32) / 255.0
            
            # Add channel dimension: (1, H, W)
            img_tensor = torch.from_numpy(img_np).unsqueeze(0)
            lr_tensor = torch.from_numpy(lr_np).unsqueeze(0)
            
            return lr_tensor, img_tensor
            
        except Exception as e:
            # Return zeros in case of error to avoid crashing
            return torch.zeros(1, 64, 64), torch.zeros(1, 128, 128)


class TrainingService:
    """
    Service for training AI upscaling models.
    
    This class handles dataset preparation and model training.
    """
    
    def __init__(
        self, 
        config: Optional[UpscaleConfig] = None,
        file_storage: Optional[FileStorage] = None
    ):
        """
        Initialize the training service.
        
        Args:
            config: Upscaling configuration
            file_storage: File storage for loading/saving files
        """
        self.config = config or UpscaleConfig()
        self.file_storage = file_storage or FileStorage()
        self.logger = logging.getLogger(self.__class__.__name__)
        
    @profiler.profile()
    def train_model(
        self,
        dataset_path: str,
        output_model_path: str,
        epochs: int = 10,
        batch_size: int = 16,
        learning_rate: float = 1e-4,
        validation_split: float = 0.2
    ) -> str:
        """
        Train an upscaling model.
        
        Args:
            dataset_path: Path to directory containing high-res height maps
            output_model_path: Path to save the trained model
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            validation_split: Fraction of data to use for validation
            
        Returns:
            Path to the saved model
            
        Raises:
            TrainingError: If training fails
        """
        try:
            self.logger.info(f"Starting training with dataset: {dataset_path}")
            
            # 1. Load dataset
            image_paths = self._get_image_paths(dataset_path)
            if not image_paths:
                raise TrainingError(f"No images found in {dataset_path}")
                
            self.logger.info(f"Found {len(image_paths)} images")
            
            # 2. Split dataset
            np.random.shuffle(image_paths)
            split_idx = int(len(image_paths) * (1 - validation_split))
            train_paths = image_paths[:split_idx]
            val_paths = image_paths[split_idx:]
            
            train_dataset = HeightMapDataset(train_paths)
            val_dataset = HeightMapDataset(val_paths)
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
            
            # 3. Create or load model
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.logger.info(f"Training on {device}")
            
            model = UpscalerModel(scale_factor=2)
            if os.path.exists(output_model_path):
                self.logger.info(f"Resuming training from {output_model_path}")
                try:
                    model.load_state_dict(torch.load(output_model_path, map_location=device))
                except:
                    self.logger.warning("Could not load existing model weights, starting from scratch")
            
            model.to(device)
            
            # 4. Setup optimizer and loss
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            criterion = nn.MSELoss()
            
            # 5. Train model
            self.logger.info(f"Training for {epochs} epochs...")
            
            best_val_loss = float('inf')
            
            for epoch in range(epochs):
                # Training phase
                model.train()
                train_loss = 0.0
                
                for lr_imgs, hr_imgs in train_loader:
                    lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
                    
                    optimizer.zero_grad()
                    outputs = model(lr_imgs)
                    loss = criterion(outputs, hr_imgs)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item() * lr_imgs.size(0)
                
                train_loss /= len(train_dataset)
                
                # Validation phase
                model.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for lr_imgs, hr_imgs in val_loader:
                        lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
                        outputs = model(lr_imgs)
                        loss = criterion(outputs, hr_imgs)
                        val_loss += loss.item() * lr_imgs.size(0)
                
                val_loss /= len(val_dataset)
                
                self.logger.info(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), output_model_path)
                    self.logger.info(f"Saved best model with val_loss: {val_loss:.6f}")
            
            self.logger.info("Training complete")
            return output_model_path
            
        except Exception as e:
            raise TrainingError(f"Training failed: {e}")

    def _get_image_paths(self, dataset_path: str) -> List[str]:
        """Get list of image paths from directory."""
        extensions = ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff']
        paths = []
        for ext in extensions:
            paths.extend(glob.glob(os.path.join(dataset_path, ext)))
            paths.extend(glob.glob(os.path.join(dataset_path, "**", ext), recursive=True))
        return sorted(list(set(paths)))
