"""ResNet-18 encoder for few-shot learning.

Uses torchvision pre-trained ResNet18 as backbone.
Standard architecture for few-shot learning benchmarks.
"""
import torch
import torch.nn as nn
import torchvision.models as models


class ResNet18Encoder(nn.Module):
    """
    ResNet-18 encoder for few-shot learning.
    
    Architecture:
    - Uses torchvision ResNet18 (optionally pre-trained on ImageNet)
    - Removes final FC layer
    - Global Average Pooling -> 512-dim output
    
    Input: (B, 3, H, W) - works with any spatial size >= 32
    Output: (B, 512) features
    
    Used by: MatchingNet (with --backbone resnet18)
    """
    
    def __init__(self, pretrained=False):
        """
        Args:
            pretrained: If True, use ImageNet pre-trained weights.
                       For few-shot learning on scalograms, typically False.
        """
        super(ResNet18Encoder, self).__init__()
        
        # Load ResNet18 from torchvision
        if pretrained:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
        else:
            weights = None
        
        resnet = models.resnet18(weights=weights)
        
        # Remove the final FC layer (resnet.fc)
        # Keep: conv1, bn1, relu, maxpool, layer1-4, avgpool
        self.features = nn.Sequential(
            resnet.conv1,      # 3 -> 64, stride=2, 7x7
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,    # stride=2, 3x3
            resnet.layer1,     # 64 -> 64
            resnet.layer2,     # 64 -> 128, stride=2
            resnet.layer3,     # 128 -> 256, stride=2
            resnet.layer4,     # 256 -> 512, stride=2
        )
        
        # Global Average Pooling (size-agnostic)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.out_dim = 512
        
        # Re-initialize weights if not pretrained (for fair comparison)
        # if not pretrained:
        #     self._init_weights()  # Temporarily disabled
    
    def _init_weights(self):
        """Initialize weights with Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) input images
        Returns:
            (B, 512) features after global average pooling
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # (B, 512)
        return x
