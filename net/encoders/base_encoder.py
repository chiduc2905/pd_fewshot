"""Base encoder for few-shot learning models."""
import torch.nn as nn
import functools


def get_norm_layer(norm_type='group'):
    """Get normalization layer by name."""
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=True)
    elif norm_type == 'group':
        # Use 8 groups for 64 channels (8 channels per group)
        norm_layer = functools.partial(nn.GroupNorm, 8)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


class Conv64F_Encoder(nn.Module):
    """
    Base 4-layer CNN encoder with BatchNorm + LeakyReLU.
    Input: 3xHxW -> Output: 64x16x16 (with adaptive pooling for consistent output)
    
    Used by: CovaMNet (default), CosineNet
    """
    
    def __init__(self, norm_layer=functools.partial(nn.BatchNorm2d, affine=True)):
        super(Conv64F_Encoder, self).__init__()
        # GroupNorm and BatchNorm use bias=False in conv, others use bias=True
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.features = nn.Sequential(                       
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(64),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(kernel_size=2, stride=2),           
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(64),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(kernel_size=2, stride=2),           
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(64),
            nn.LeakyReLU(0.2, True),                         
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(64),
            nn.LeakyReLU(0.2, True),
        )
        
        # Adaptive pooling for consistent 16x16 output regardless of input size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((16, 16))
        
    def forward(self, x):
        feat = self.features(x)
        feat = self.adaptive_pool(feat)  # Always output 64x16x16
        return feat

