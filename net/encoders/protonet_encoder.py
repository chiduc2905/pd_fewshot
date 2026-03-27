"""ProtoNet paper-standard Conv4 encoder."""
import torch.nn as nn


class Conv64F_Paper_Encoder(nn.Module):
    """Standard Conv4 encoder used in ProtoNet-style benchmarks."""

    def __init__(self, image_size=64):
        super(Conv64F_Paper_Encoder, self).__init__()

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )

        self.features = nn.Sequential(
            conv_block(3, 64),
            conv_block(64, 64),
            conv_block(64, 64),
            conv_block(64, 64),
        )

        spatial_dim = max(1, image_size // 16)
        self.out_dim = 64 * spatial_dim * spatial_dim

    def forward(self, x):
        feat = self.features(x)
        return feat.view(feat.size(0), -1)
