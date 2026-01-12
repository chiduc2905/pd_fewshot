"""Matching Networks encoder - same as original implementation."""
import torch.nn as nn


class MatchingNetEncoder(nn.Module):
    """
    4-layer CNN encoder for Matching Networks.
    
    Architecture from: Vinyals et al. "Matching Networks for One Shot Learning" (NIPS 2016)
    Follows gitabcworld implementation exactly:
    - 4 conv blocks: Conv(3->64) + BatchNorm + ReLU + MaxPool(2x2)
    - Flattens output directly (NO GAP/AdaptivePool)
    - Output size depends on input size: outSize = (image_size/16)^2 * 64
      - 28x28 (Omniglot) -> 64 features
      - 84x84 (miniImageNet) -> 1600 features
      - 128x128 -> 4096 features
    
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
        
    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) input images
        Returns:
            (B, feat_dim) flattened features where feat_dim = 64 * (H/16) * (W/16)
        """
        feat = self.features(x)  # (B, 64, H/16, W/16)
        return feat.view(feat.size(0), -1)  # Flatten directly




