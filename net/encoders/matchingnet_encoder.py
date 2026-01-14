"""Matching Networks encoder - paper-compliant with 64 features (GAP)."""
import torch.nn as nn


class MatchingNetEncoder(nn.Module):
    """
    4-layer CNN encoder for Matching Networks.
    
    Architecture from: Vinyals et al. "Matching Networks for One Shot Learning" (NIPS 2016)
    - 4 conv blocks: Conv(3->64) + BatchNorm + ReLU + MaxPool(2x2)
    - Global Average Pooling (GAP) for 64-dim output (paper-compliant)
    - Input: (B, 3, H, W) -> Output: (B, 64) features
    
    Paper uses 64-dim embeddings for LSTM, not 1024!
    This reduces LSTM params from ~17M to ~67K.
    
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
        
        # Global Average Pooling for 64-dim output (paper-compliant)
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Output dimension
        self.out_dim = 64
        
    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) input images (supports any size)
        Returns:
            (B, 64) features after global average pooling
        """
        feat = self.features(x)  # (B, 64, H/16, W/16)
        feat = self.gap(feat)    # (B, 64, 1, 1)
        return feat.view(feat.size(0), -1)  # (B, 64)





