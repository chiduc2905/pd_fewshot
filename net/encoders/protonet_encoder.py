"""ProtoNet paper-standard encoder."""
import torch.nn as nn


class Conv64F_Paper_Encoder(nn.Module):
    """
    Standard 4-layer CNN encoder matching official ProtoNet paper.
    
    Architecture from: Snell et al. "Prototypical Networks for Few-shot Learning" (NeurIPS 2017)
    - 4 conv blocks: Conv(3->64) + BatchNorm + ReLU + MaxPool(2x2)
    - Global Average Pooling at the end for consistent output dimension
    - Input: (B, 3, H, W) -> Output: (B, 64) features (supports any input size)
    
    This matches the official implementation with GAP for flexible input sizes:
    https://github.com/jakesnell/prototypical-networks
    
    Used by: ProtoNet (with --encoder_type='paper')
    """
    
    def __init__(self):
        super(Conv64F_Paper_Encoder, self).__init__()
        
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
        
        # Global Average Pooling for consistent output size
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Feature dimension after GAP
        self.out_dim = 64
        
    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) input images (supports 64x64, 84x84, 128x128, etc.)
        Returns:
            (B, 64) features after global average pooling
        """
        feat = self.features(x)  # (B, 64, H/16, W/16)
        feat = self.gap(feat)    # (B, 64, 1, 1)
        return feat.view(feat.size(0), -1)  # (B, 64)

