
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += residual
        return out

class UpscalerModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: (N, 1, H, W)
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        
        # 4 Residual Blocks
        self.res_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(4)])
        
        # Upscaling
        self.conv2 = nn.Conv2d(64, 256, 3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(2) # Output channels: 256 / 4 = 64
        
        # Output
        self.conv3 = nn.Conv2d(64, 1, 3, padding=1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.res_blocks(x)
        x = self.conv2(x)
        x = self.pixel_shuffle(x)
        x = self.conv3(x)
        return x

def verify_model():
    print("Verifying PyTorch model architecture...")
    model = UpscalerModel()
    print(model)
    
    # Test with random input
    # Batch size 1, 1 channel, 64x64 input
    input_tensor = torch.randn(1, 1, 64, 64)
    output = model(input_tensor)
    
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
    
    # Expected output: (1, 1, 128, 128) because scale factor is 2
    assert output.shape == (1, 1, 128, 128), f"Expected output shape (1, 1, 128, 128), got {output.shape}"
    print("Verification successful!")

if __name__ == "__main__":
    verify_model()
