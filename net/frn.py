"""FRN: Feature Map Reconstruction Networks for few-shot classification."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from net.encoders.smnet_conv64f_encoder import build_resnet12_family_encoder


def _auxrank_loss(support: torch.Tensor, weight: float = 0.03) -> torch.Tensor:
    """Paper/repo auxiliary rank regularizer over class support descriptors."""
    way = support.size(0)
    if way < 2 or support.numel() == 0:
        return support.new_zeros(())

    support = F.normalize(support, p=2, dim=-1)
    lhs, rhs = torch.tril_indices(way, way, offset=-1, device=support.device)
    s1 = support.index_select(0, lhs)
    s2 = support.index_select(0, rhs)
    gram = torch.matmul(s1, s2.transpose(1, 2))
    return gram.pow(2).sum(dim=(1, 2)).sum() * weight


class FRN(nn.Module):
    """Meta-test style FRN head with a ResNet-12 backbone."""

    def __init__(self, image_size=64, fewshot_backbone="resnet12", device="cuda"):
        super().__init__()
        self.encoder = build_resnet12_family_encoder(
            image_size=image_size,
            backbone_name=fewshot_backbone,
            pool_output=False,
            variant="fewshot",
            drop_rate=0.0,
        )
        self.feat_dim = self.encoder.out_channels
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.reconstruction_params = nn.Parameter(torch.zeros(2))
        self.to(device)

    def _get_feature_map(self, x):
        feature_map = self.encoder.forward_features(x) / math.sqrt(self.feat_dim)
        return feature_map.flatten(2).transpose(1, 2).contiguous()

    def _reconstruction_distance(self, query, support, alpha, beta):
        regularizer = support.size(1) / support.size(2)
        lam = regularizer * alpha.exp() + 1e-6
        rho = beta.exp()

        support_t = support.transpose(1, 2)
        gram = torch.bmm(support_t, support)
        identity = torch.eye(gram.size(-1), device=gram.device, dtype=gram.dtype).unsqueeze(0)
        hat = torch.bmm(torch.linalg.inv(gram + lam * identity), gram)

        projected_query = torch.einsum("nd,wde->wne", query, hat) * rho
        return (projected_query - query.unsqueeze(0)).pow(2).sum(dim=2).transpose(0, 1)

    def forward(self, query, support):
        batch_size, num_query, channels, height, width = query.size()
        _, way_num, shot_num, _, _, _ = support.size()

        query_map = self._get_feature_map(query.view(-1, channels, height, width))
        support_map = self._get_feature_map(support.view(-1, channels, height, width))

        resolution = query_map.size(1)
        query_map = query_map.view(batch_size, num_query, resolution, self.feat_dim)
        support_map = support_map.view(batch_size, way_num, shot_num, resolution, self.feat_dim)

        alpha = self.reconstruction_params[0]
        beta = self.reconstruction_params[1]

        outputs = []
        aux_loss = query.new_zeros(())
        for batch_idx in range(batch_size):
            support_cls = support_map[batch_idx].reshape(way_num, shot_num * resolution, self.feat_dim)
            query_desc = query_map[batch_idx].reshape(num_query * resolution, self.feat_dim)
            recon_dist = self._reconstruction_distance(query_desc, support_cls, alpha, beta)
            logits = -recon_dist.view(num_query, resolution, way_num).mean(dim=1) * self.scale
            outputs.append(logits)
            if self.training:
                aux_loss = aux_loss + _auxrank_loss(support_cls)

        logits = torch.cat(outputs, dim=0)
        if not self.training:
            return logits
        return {"logits": logits, "aux_loss": aux_loss / max(1, batch_size)}
