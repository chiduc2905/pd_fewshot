"""CAN: Cross-Attention Network for few-shot classification."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from net.encoders.smnet_conv64f_encoder import build_resnet12_family_encoder


class CrossAttentionModule(nn.Module):
    """Cross Attention Module (CAM) defined in the CAN paper."""

    def __init__(self, spatial_dim, temperature=0.025, reduction_ratio=6):
        super().__init__()
        self.temperature = float(temperature)
        self.spatial_dim = int(spatial_dim)
        self.token_dim = self.spatial_dim * self.spatial_dim
        hidden_dim = max(1, self.token_dim // int(reduction_ratio))
        self.class_mlp = nn.Sequential(
            nn.Linear(self.token_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, self.token_dim),
        )
        self.query_mlp = nn.Sequential(
            nn.Linear(self.token_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, self.token_dim),
        )

    @staticmethod
    def _pairwise_correlation(class_feat, query_feat):
        class_tokens = class_feat.flatten(2).transpose(1, 2).contiguous()
        query_tokens = query_feat.flatten(2).transpose(1, 2).contiguous()
        class_tokens = F.normalize(class_tokens, p=2, dim=-1)
        query_tokens = F.normalize(query_tokens, p=2, dim=-1)
        return torch.bmm(class_tokens, query_tokens.transpose(1, 2))

    def _apply_attention(self, feat, corr_map, meta_learner):
        kernel = meta_learner(corr_map.mean(dim=1))
        attn = torch.bmm(corr_map, kernel.unsqueeze(-1)).squeeze(-1) / self.temperature
        attn = F.softmax(attn, dim=1)
        attn = attn.view(feat.size(0), 1, self.spatial_dim, self.spatial_dim)
        return feat * (1.0 + attn)

    def forward(self, class_feat, query_feat):
        correlation = self._pairwise_correlation(class_feat, query_feat)
        class_weighted = self._apply_attention(class_feat, correlation, self.class_mlp)
        query_weighted = self._apply_attention(query_feat, correlation.transpose(1, 2), self.query_mlp)
        return class_weighted, query_weighted


class CrossAttentionNet(nn.Module):
    """Paper-style CAN with CAM and inductive inference logits."""

    def __init__(
        self,
        image_size=64,
        way_num=4,
        cam_temperature=0.025,
        loss_weight=0.5,
        reduction_ratio=6,
        fewshot_backbone="resnet12",
        device="cuda",
    ):
        super().__init__()
        self.encoder = build_resnet12_family_encoder(
            image_size=image_size,
            backbone_name=fewshot_backbone,
            pool_output=False,
            variant="fewshot",
            drop_rate=0.0,
        )
        self.feat_dim = self.encoder.out_channels
        self.spatial_dim = self.encoder.out_spatial
        self.cam = CrossAttentionModule(
            spatial_dim=self.spatial_dim,
            temperature=cam_temperature,
            reduction_ratio=reduction_ratio,
        )
        self.loss_weight = float(loss_weight)
        self.global_classifier = nn.Linear(self.feat_dim, int(way_num))
        self.to(device)

    def _encode_feature_maps(self, x):
        return self.encoder.forward_features(x)

    def _pairwise_cam_features(self, query_feat, class_feat):
        num_query = query_feat.size(0)
        way_num = class_feat.size(0)

        class_expand = class_feat.unsqueeze(0).expand(num_query, way_num, -1, -1, -1)
        query_expand = query_feat.unsqueeze(1).expand(num_query, way_num, -1, -1, -1)
        flat_class = class_expand.reshape(-1, self.feat_dim, self.spatial_dim, self.spatial_dim)
        flat_query = query_expand.reshape(-1, self.feat_dim, self.spatial_dim, self.spatial_dim)
        class_weighted, query_weighted = self.cam(flat_class, flat_query)
        class_weighted = class_weighted.view(num_query, way_num, self.feat_dim, self.spatial_dim, self.spatial_dim)
        query_weighted = query_weighted.view(num_query, way_num, self.feat_dim, self.spatial_dim, self.spatial_dim)
        return class_weighted, query_weighted

    def _inductive_logits(self, class_weighted, query_weighted):
        class_vec = F.adaptive_avg_pool2d(class_weighted, 1).squeeze(-1).squeeze(-1)
        query_vec = F.adaptive_avg_pool2d(query_weighted, 1).squeeze(-1).squeeze(-1)
        return F.cosine_similarity(
            F.normalize(query_vec, p=2, dim=-1),
            F.normalize(class_vec, p=2, dim=-1),
            dim=-1,
        )

    def _training_loss(self, class_weighted, query_weighted, query_targets):
        num_query, way_num = class_weighted.size(0), class_weighted.size(1)
        token_num = self.spatial_dim * self.spatial_dim

        class_centers = F.adaptive_avg_pool2d(
            class_weighted.view(num_query * way_num, self.feat_dim, self.spatial_dim, self.spatial_dim),
            1,
        ).view(num_query, way_num, self.feat_dim)
        query_tokens = query_weighted.view(num_query, way_num, self.feat_dim, token_num).permute(0, 1, 3, 2)

        local_logits = []
        for class_idx in range(way_num):
            token_feat = query_tokens[:, class_idx]
            logits_k = F.cosine_similarity(
                F.normalize(token_feat.unsqueeze(2), p=2, dim=-1),
                F.normalize(class_centers.unsqueeze(1), p=2, dim=-1),
                dim=-1,
            )
            local_logits.append(logits_k)

        local_logits = torch.stack(local_logits, dim=1)
        local_logits = local_logits[torch.arange(num_query, device=query_targets.device), query_targets]
        local_logits = local_logits.view(num_query * token_num, way_num)
        local_targets = query_targets.unsqueeze(1).expand(num_query, token_num).reshape(-1)
        local_loss = F.cross_entropy(local_logits, local_targets)

        true_query_maps = query_weighted[torch.arange(num_query, device=query_targets.device), query_targets]
        true_query_tokens = true_query_maps.view(num_query, self.feat_dim, token_num).permute(0, 2, 1).reshape(-1, self.feat_dim)
        global_logits = self.global_classifier(true_query_tokens)
        global_targets = query_targets.unsqueeze(1).expand(num_query, token_num).reshape(-1)
        global_loss = F.cross_entropy(global_logits, global_targets)

        return self.loss_weight * local_loss + global_loss

    def forward(self, query, support, query_targets=None, support_targets=None):
        del support_targets
        batch_size, num_query, channels, height, width = query.size()
        _, way_num, shot_num, _, _, _ = support.size()

        query_feat = self._encode_feature_maps(query.view(-1, channels, height, width))
        support_feat = self._encode_feature_maps(support.view(-1, channels, height, width))
        query_feat = query_feat.view(batch_size, num_query, self.feat_dim, self.spatial_dim, self.spatial_dim)
        support_feat = support_feat.view(batch_size, way_num, shot_num, self.feat_dim, self.spatial_dim, self.spatial_dim)
        class_feat = support_feat.mean(dim=2)

        all_logits = []
        losses = []
        for batch_idx in range(batch_size):
            class_weighted, query_weighted = self._pairwise_cam_features(query_feat[batch_idx], class_feat[batch_idx])
            logits = self._inductive_logits(class_weighted, query_weighted)
            all_logits.append(logits)

            if self.training and query_targets is not None:
                batch_targets = query_targets.view(batch_size, num_query)[batch_idx]
                losses.append(self._training_loss(class_weighted, query_weighted, batch_targets))

        logits = torch.cat(all_logits, dim=0)
        if losses:
            return {"logits": logits, "loss": torch.stack(losses).mean()}
        return logits
