import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure local site-packages on V: drive is available
site_pkg = Path(r"V:\site-packages")
if site_pkg.exists() and str(site_pkg) not in sys.path:
    sys.path.insert(0, str(site_pkg))


class ResidualBlock(nn.Module):
    """
    Residual Block with two 3x3 Conv2d layers, BatchNorm, and PReLU activation.
    Preserves spatial dimensions and uses a skip shortcut: Output = Input + F(Input).
    """
    def __init__(self, channels: int):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv_block(x)


class DeblurNetLite(nn.Module):
    """
    Enhanced DeblurNet-Lite with Global Residual Learning (Residual-in-Residual).
    
    Key Breakthrough for True Sharpness:
    - Instead of reconstructing the entire image from scratch through an autoencoder bottleneck
      (which causes loss of fine textures and overall softness), the network uses GLOBAL RESIDUAL LEARNING:
          Output = Clamp(Input + Residual_Delta, 0.0, 1.0)
    - The network ONLY predicts the high-frequency sharpening delta (Delta = I_sharp - I_blurry).
    - Preserves 100% of the native input resolution, base illumination, and color dynamics.
    - Fully convolutional: supports arbitrary image dimensions (H, W) without downscaling!
    
    Total parameters: ~293,000 (CPU-friendly, < 400 MB RAM).
    """
    def __init__(self):
        super(DeblurNetLite, self).__init__()
        
        # 1. Initial Feature Extraction Layer
        # Input: (B, 3, H, W) -> Output: (B, 32, H, W)
        self.encoder_conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.PReLU()
        )
        
        # 2. Downsampling Layer (Extracts contextual blur characteristics)
        # Input: (B, 32, H, W) -> Output: (B, 64, H/2, W/2)
        self.encoder_down = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU()
        )
        
        # 3. Residual Bottleneck (3 Residual Blocks)
        # Deep non-linear transformation for high-frequency edge restoration
        self.bottleneck = nn.Sequential(
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64)
        )
        
        # 4. Upsampling Layer (Transposed Convolution)
        # Input: (B, 64, H/2, W/2) -> Output: (B, 32, H, W)
        self.decoder_up = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.PReLU()
        )
        
        # 5. Skip Fusion Layer
        # Combines decoder features (32 ch) + early encoder features (32 ch) = 64 ch
        self.fusion = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.PReLU()
        )
        
        # 6. High-Frequency Sharpening Delta Projection
        # Predicts the residual difference Delta in [-1.0, 1.0]
        self.output_conv = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)
        
        # Initialize weights randomly (He Normal for internal convs, zero-init on output_conv for smooth initial identity pass)
        self._initialize_weights()

    def _initialize_weights(self):
        """Random weight initialization using Kaiming Normal."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # Initialize output conv with small weights to stabilize initial residual addition
        nn.init.normal_(self.output_conv.weight, mean=0.0, std=0.01)
        if self.output_conv.bias is not None:
            nn.init.constant_(self.output_conv.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dynamically pad to ensure H and W are even for stride-2 downsampling
        _, _, h, w = x.shape
        pad_h = (2 - (h % 2)) % 2
        pad_w = (2 - (w % 2)) % 2
        if pad_h > 0 or pad_w > 0:
            x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        else:
            x_pad = x
            
        # Step 1: Extract low-level spatial features
        enc1 = self.encoder_conv1(x_pad)
        
        # Step 2: Downsample to contextual scale
        enc2 = self.encoder_down(enc1)
        
        # Step 3: Transform through Residual Bottleneck
        bottleneck = self.bottleneck(enc2)
        
        # Step 4: Upsample back to full spatial scale
        dec = self.decoder_up(bottleneck)
        
        # Step 5: Merge with high-resolution skip connection
        fused = torch.cat([dec, enc1], dim=1)
        fused = self.fusion(fused)
        
        # Step 6: Predict sharpening residual delta
        delta = self.output_conv(fused)
        
        # Step 7: Global Residual Connection - directly sharpen the input image!
        restored = torch.clamp(x_pad + delta, 0.0, 1.0)
        
        # Unpad if padded
        if pad_h > 0 or pad_w > 0:
            restored = restored[:, :, :h, :w]
            
        return restored


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = DeblurNetLite()
    print(f"DeblurNetLite with Global Residual Learning initialized!")
    print(f"Trainable Parameters: {count_parameters(model):,}")
    
    # Test with arbitrary resolution (e.g. 135x177 odd dimensions)
    x = torch.rand(1, 3, 135, 177)
    out = model(x)
    print(f"Input: {tuple(x.shape)} -> Output: {tuple(out.shape)}")
    assert x.shape == out.shape, "Dimension mismatch!"
    print("SUCCESS: Global Residual DeblurNetLite verified!")
