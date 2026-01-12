"""RelationNet encoder for few-shot learning."""
import torch.nn as nn


class RelationNetEncoder(nn.Module):
    """
    4-layer CNN encoder for Relation Networks.
    
    Architecture from: Sung et al. "Learning to Compare: Relation Network for Few-Shot Learning" (CVPR 2018)
    - 4 conv blocks: Conv(3->64) + BatchNorm + ReLU + MaxPool(2x2)
    - AdaptiveAvgPool2d(4) at the end for consistent 4x4 spatial output
    - Output: (B, 64, 4, 4) feature maps (NOT flattened, for concatenation)
    
    Used by: RelationNet
    """
    
    def __init__(self):
        super(RelationNetEncoder, self).__init__()
        
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
        
        # Adaptive pooling for consistent 4x4 spatial output
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) input images (supports 64x64, 84x84, 128x128, etc.)
        Returns:
            (B, 64, 4, 4) feature maps (NOT flattened for RelationNet)
        """
        feat = self.features(x)  # (B, 64, H/16, W/16)
        feat = self.adaptive_pool(feat)  # (B, 64, 4, 4)
        return feat

