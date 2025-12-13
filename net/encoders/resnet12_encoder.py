"""ResNet-12 encoder for few-shot learning.

Reference: TADAM, Meta-Baseline, and other few-shot learning papers.
Standard architecture for miniImageNet/tieredImageNet benchmarks.
"""
import torch.nn as nn


def conv3x3(in_planes, out_planes):
    """3x3 convolution with padding."""
    return nn.Conv2d(in_planes, out_planes, 3, padding=1, bias=False)


def conv1x1(in_planes, out_planes):
    """1x1 convolution for downsampling."""
    return nn.Conv2d(in_planes, out_planes, 1, bias=False)


class ResBlock(nn.Module):
    """
    Residual block with 3 convolutional layers.
    
    Structure: Conv -> BN -> LeakyReLU -> Conv -> BN -> LeakyReLU -> Conv -> BN
               + Skip connection (1x1 conv if dimensions differ)
               -> LeakyReLU -> MaxPool
    """
    
    def __init__(self, inplanes, planes):
        super(ResBlock, self).__init__()
        
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        
        # 3 convolutional layers
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        
        # Skip connection with 1x1 conv for dimension matching
        self.downsample = nn.Sequential(
            conv1x1(inplanes, planes),
            nn.BatchNorm2d(planes),
        )
        
        # Spatial downsampling
        self.maxpool = nn.MaxPool2d(2)
    
    def forward(self, x):
        # Main path
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        # Skip connection
        identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        
        # Spatial downsampling
        out = self.maxpool(out)
        
        return out


class ResNet12Encoder(nn.Module):
    """
    ResNet-12 encoder for few-shot learning.
    
    Architecture:
    - 4 residual blocks (3 conv layers each = 12 conv layers total)
    - Channels: [64, 128, 256, 512]
    - LeakyReLU(0.1) activation
    - MaxPool2d(2) after each block
    - Global Average Pooling at the end
    
    Input: (B, 3, 64, 64)
    Output: (B, 512) features
    
    Used by: MatchingNet (with --backbone resnet12)
    """
    
    def __init__(self, channels=None):
        """
        Args:
            channels: List of 4 channel sizes. Default: [64, 128, 256, 512]
        """
        super(ResNet12Encoder, self).__init__()
        
        if channels is None:
            channels = [64, 128, 256, 512]
        
        self.inplanes = 3
        
        # 4 residual blocks
        self.layer1 = self._make_layer(channels[0])
        self.layer2 = self._make_layer(channels[1])
        self.layer3 = self._make_layer(channels[2])
        self.layer4 = self._make_layer(channels[3])
        
        self.out_dim = channels[3]
        
        # Weight initialization (temporarily disabled)
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)
    
    def _make_layer(self, planes):
        block = ResBlock(self.inplanes, planes)
        self.inplanes = planes
        return block
    
    def forward(self, x):
        """
        Args:
            x: (B, 3, 64, 64) input images
        Returns:
            (B, 512) features after global average pooling
        """
        x = self.layer1(x)  # 64x64 -> 32x32
        x = self.layer2(x)  # 32x32 -> 16x16
        x = self.layer3(x)  # 16x16 -> 8x8
        x = self.layer4(x)  # 8x8 -> 4x4
        
        # Global Average Pooling
        x = x.view(x.size(0), x.size(1), -1).mean(dim=2)  # (B, 512)
        
        return x
