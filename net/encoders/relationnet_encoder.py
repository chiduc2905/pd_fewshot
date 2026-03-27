"""RelationNet encoder for few-shot learning."""
import torch.nn as nn


class RelationNetEncoder(nn.Module):
    """4-layer CNN encoder for Relation Networks."""
    
    def __init__(self):
        super(RelationNetEncoder, self).__init__()
        
        def conv_block(in_channels, out_channels, pool, padding):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2) if pool else nn.Identity(),
            )
        
        self.features = nn.Sequential(
            conv_block(3, 64, pool=True, padding=0),
            conv_block(64, 64, pool=True, padding=0),
            conv_block(64, 64, pool=False, padding=1),
            conv_block(64, 64, pool=False, padding=1),
        )
        
    def forward(self, x):
        return self.features(x)
