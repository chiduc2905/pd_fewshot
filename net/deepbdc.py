"""DeepBDC: Brownian distance covariance pooling for few-shot classification."""

import torch
import torch.nn as nn

from net.encoders.smnet_conv64f_encoder import build_resnet12_family_encoder


class BDC(nn.Module):
    """Brownian distance covariance layer from the official DeepBDC repo."""

    def __init__(self, is_vec=True, input_dim=(640, 10, 10), dimension_reduction=None, activate="relu"):
        super().__init__()
        self.is_vec = bool(is_vec)
        self.dr = dimension_reduction
        self.activate = activate
        self.input_dim = int(input_dim[0])

        if self.dr is not None and self.dr != self.input_dim:
            act = nn.ReLU(inplace=True) if activate == "relu" else nn.LeakyReLU(0.1, inplace=True)
            self.conv_dr_block = nn.Sequential(
                nn.Conv2d(self.input_dim, self.dr, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(self.dr),
                act,
            )
        else:
            self.conv_dr_block = None

        output_dim = self.dr if self.dr else self.input_dim
        if self.is_vec:
            self.output_dim = int(output_dim * (output_dim + 1) / 2)
        else:
            self.output_dim = int(output_dim * output_dim)

        spatial_h, spatial_w = int(input_dim[1]), int(input_dim[2])
        init_temp = 1.0 / (2 * spatial_h * spatial_w)
        self.temperature = nn.Parameter(torch.log(init_temp * torch.ones(1, 1)), requires_grad=True)
        self._init_weight()

    def _init_weight(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, a=0, mode="fan_out", nonlinearity="leaky_relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        if self.conv_dr_block is not None:
            x = self.conv_dr_block(x)
        x = self._bdcov_pool(x, self.temperature)
        if self.is_vec:
            return self._triu_vec(x)
        return x.reshape(x.shape[0], -1)

    @staticmethod
    def _bdcov_pool(x, temperature):
        batch_size, dim, height, width = x.shape
        node_num = height * width
        x = x.reshape(batch_size, dim, node_num)

        eye = torch.eye(dim, dim, device=x.device, dtype=x.dtype).view(1, dim, dim).repeat(batch_size, 1, 1)
        ones = torch.ones(batch_size, dim, dim, device=x.device, dtype=x.dtype)
        gram = x.bmm(x.transpose(1, 2))
        dcov = ones.bmm(gram * eye) + (gram * eye).bmm(ones) - 2 * gram
        dcov = torch.clamp(dcov, min=0.0)
        dcov = torch.exp(temperature) * dcov
        dcov = torch.sqrt(dcov + 1e-5)
        return dcov - dcov.bmm(ones) / dim - ones.bmm(dcov) / dim + ones.bmm(dcov).bmm(ones) / (dim * dim)

    @staticmethod
    def _triu_vec(x):
        batch_size, dim, _ = x.shape
        indices = torch.triu_indices(dim, dim, device=x.device)
        return x[:, indices[0], indices[1]].reshape(batch_size, -1)


class DeepBDC(nn.Module):
    """Meta DeepBDC head with official-style ResNet-12 backbone settings."""

    def __init__(self, image_size=64, reduce_dim=None, fewshot_backbone="resnet12", device="cuda"):
        super().__init__()
        self.encoder = build_resnet12_family_encoder(
            image_size=image_size,
            backbone_name=fewshot_backbone,
            pool_output=False,
            variant="deepbdc",
            drop_rate=0.0,
        )
        if reduce_dim is None:
            reduce_dim = self.encoder.out_channels
        self.bdc = BDC(
            is_vec=True,
            input_dim=(self.encoder.out_channels, self.encoder.out_spatial, self.encoder.out_spatial),
            dimension_reduction=reduce_dim,
        )
        self.to(device)

    def _encode_vector(self, x):
        feature_map = self.encoder.forward_features(x)
        return self.bdc(feature_map)

    def forward(self, query, support):
        batch_size, num_query, channels, height, width = query.size()
        _, way_num, shot_num, _, _, _ = support.size()

        query_vec = self._encode_vector(query.view(-1, channels, height, width)).view(batch_size, num_query, -1)
        support_vec = self._encode_vector(support.view(-1, channels, height, width)).view(batch_size, way_num, shot_num, -1)
        prototypes = support_vec.mean(dim=2)

        if shot_num > 1:
            logits = -(query_vec.unsqueeze(2) - prototypes.unsqueeze(1)).pow(2).sum(dim=3)
        else:
            logits = torch.einsum("bqd,bwd->bqw", query_vec, prototypes)
        return logits.view(-1, way_num)
