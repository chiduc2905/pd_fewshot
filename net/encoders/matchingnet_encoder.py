"""Matching Networks encoder - paper-compliant with 1024 features."""
import torch.nn as nn


class MatchingNetEncoder(nn.Module):
    """
    4-layer CNN encoder for Matching Networks.
    
    Architecture from: Vinyals et al. "Matching Networks for One Shot Learning" (NIPS 2016)
    - 4 conv blocks: Conv(3->64) + BatchNorm + ReLU + MaxPool(2x2)
    - AdaptiveAvgPool2d(4) to ensure 4x4 spatial output
    - Flattens to 1024 features (64 * 4 * 4) - matches paper params
    - Input: (B, 3, H, W) -> Output: (B, 1024) features (any input size)
    
    Used by: MatchingNet (with --backbone conv64f)
    """
    
    def __init__(self):
        super(MatchingNetEncoder, self).__init__()
        
        def conv_block(in_channels, out_channels):
            """Standard conv block: Conv -> BatchNorm -> ReLU -> MaxPool"""
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
        
        self.features = nn.Sequential(
            conv_block(3, 64),      # H/2
            conv_block(64, 64),     # H/4
            conv_block(64, 64),     # H/8
            conv_block(64, 64),     # H/16
        )
        
        # Adaptive pooling for consistent 4x4 output (1024 features like paper)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) input images (supports any size)
        Returns:
            (B, 1024) flattened features (64 * 4 * 4)
        """
        feat = self.features(x)  # (B, 64, H/16, W/16)
        feat = self.adaptive_pool(feat)  # (B, 64, 4, 4)
        return feat.view(feat.size(0), -1)  # (B, 1024)




