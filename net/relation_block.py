"""Relation block for computing relation scores between query and support."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RelationBlock(nn.Module):
    """
    Relation module that learns to compare feature pairs.
    
    From: Sung et al. "Learning to Compare: Relation Network for Few-Shot Learning" (CVPR 2018)
    
    Takes concatenated feature pairs and outputs relation scores in [0,1].
    """
    
    def __init__(self, input_channels=64, hidden_size=8, spatial_size=14):
        """
        Args:
            input_channels: Input feature dimension from encoder
            hidden_size: Hidden dimension for relation module
        """
        super(RelationBlock, self).__init__()
        padding = 1 if spatial_size < 10 else 0
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_channels * 2, input_channels, kernel_size=3, padding=padding),
            nn.BatchNorm2d(input_channels, momentum=1, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=padding),
            nn.BatchNorm2d(input_channels, momentum=1, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        shrink_s = lambda size: int((int((size - 2 + 2 * padding) / 2) - 2 + 2 * padding) / 2)
        flattened_dim = input_channels * shrink_s(spatial_size) * shrink_s(spatial_size)
        self.fc1 = nn.Linear(flattened_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        """
        Args:
            x: (B, 2*input_channels, H, W) concatenated feature pairs
        Returns:
            scores: (B, 1) relation scores in [0,1]
        """
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = torch.sigmoid(self.fc2(out))
        return out
