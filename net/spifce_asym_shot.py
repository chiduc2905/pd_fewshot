"""Asymmetric shot-preserving SPIFCE redesign.

This module keeps the SPIF global intuition that class-stable evidence should
be gated before prototype pooling, but it stops forcing the local matcher to
consume the same token policy. The local branch gets its own lightweight token
head, preserves the support-shot dimension until after per-shot matching, can
score correspondences with one-way or bidirectional top-r matching, and fuses
with the global branch using fixed or margin-adaptive weights.
"""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from net.fewshot_common import BaseConv64FewShotModel, feature_map_to_tokens
from net.spif import SPIFEncoder, _make_projection_head


class SPIFCEAsymShot(BaseConv64FewShotModel):
    """SPIFCE with asymmetric global/local token policies and shot-preserving local matching."""

    _VALID_SHOT_AGGREGATIONS = frozenset({"pooled", "mean", "softmax"})
    _VALID_FUSION_MODES = frozenset({"fixed", "margin_adaptive"})
    _VALID_LOCAL_SCORE_MODES = frozenset({"query_to_support", "bidirectional"})

    def __init__(
        self,
        in_channels: int = 3,
        hidden_dim: int = 64,
        stable_dim: int = 64,
        variant_dim: int = 64,
        local_dim: int | None = None,
        gate_hidden: int = 16,
        top_r: int = 4,
        alpha_init: float = 0.7,
        learnable_alpha: bool = False,
        gate_on: bool = True,
        factorization_on: bool = True,
        global_only: bool = False,
        local_only: bool = False,
        token_l2norm: bool = True,
        spif_asym_shot_agg: str = "softmax",
        spif_asym_shot_softmax_beta: float = 10.0,
        spif_asym_fusion_mode: str = "margin_adaptive",
        spif_asym_fusion_kappa: float = 8.0,
        spif_asym_global_scale: float = 1.0,
        spif_asym_local_scale: float = 1.0,
        spif_asym_local_score_mode: str = "bidirectional",
        spif_asym_share_local_head: bool = False,
        backbone_name: str = "resnet12",
        image_size: int = 64,
        resnet12_drop_rate: float = 0.0,
        resnet12_dropblock_size: int = 5,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            backbone_name=backbone_name,
            image_size=image_size,
            resnet12_drop_rate=resnet12_drop_rate,
            resnet12_dropblock_size=resnet12_dropblock_size,
        )
        if global_only and local_only:
            raise ValueError("global_only and local_only cannot both be true")
        if top_r <= 0:
            raise ValueError("top_r must be positive")
        if spif_asym_shot_agg not in self._VALID_SHOT_AGGREGATIONS:
            raise ValueError(
                f"spif_asym_shot_agg must be one of {sorted(self._VALID_SHOT_AGGREGATIONS)}, "
                f"got {spif_asym_shot_agg!r}"
            )
        if spif_asym_fusion_mode not in self._VALID_FUSION_MODES:
            raise ValueError(
                f"spif_asym_fusion_mode must be one of {sorted(self._VALID_FUSION_MODES)}, "
                f"got {spif_asym_fusion_mode!r}"
            )
        if spif_asym_local_score_mode not in self._VALID_LOCAL_SCORE_MODES:
            raise ValueError(
                f"spif_asym_local_score_mode must be one of {sorted(self._VALID_LOCAL_SCORE_MODES)}, "
                f"got {spif_asym_local_score_mode!r}"
            )
        if float(spif_asym_shot_softmax_beta) <= 0.0:
            raise ValueError("spif_asym_shot_softmax_beta must be positive")
        if float(spif_asym_fusion_kappa) < 0.0:
            raise ValueError("spif_asym_fusion_kappa must be non-negative")
        if float(spif_asym_global_scale) <= 0.0:
            raise ValueError("spif_asym_global_scale must be positive")
        if float(spif_asym_local_scale) <= 0.0:
            raise ValueError("spif_asym_local_scale must be positive")

        self.global_only = bool(global_only)
        self.local_only = bool(local_only)
        self.gate_on = bool(gate_on)
        self.factorization_on = bool(factorization_on)
        self.token_l2norm = bool(token_l2norm)
        self.top_r = int(top_r)
        self.learnable_alpha = bool(learnable_alpha)
        self.stable_dim = int(stable_dim)
        self.variant_dim = int(variant_dim)
        self.local_dim = int(self.stable_dim if local_dim is None else local_dim)
        self.spif_asym_shot_agg = str(spif_asym_shot_agg)
        self.spif_asym_shot_softmax_beta = float(spif_asym_shot_softmax_beta)
        self.spif_asym_fusion_mode = str(spif_asym_fusion_mode)
        self.spif_asym_fusion_kappa = float(spif_asym_fusion_kappa)
        self.spif_asym_global_scale = float(spif_asym_global_scale)
        self.spif_asym_local_scale = float(spif_asym_local_scale)
        self.spif_asym_local_score_mode = str(spif_asym_local_score_mode)
        self.spif_asym_share_local_head = bool(spif_asym_share_local_head)

        alpha_init = min(max(float(alpha_init), 1e-4), 1.0 - 1e-4)
        alpha_logit_value = math.log(alpha_init / (1.0 - alpha_init))
        if self.learnable_alpha:
            self.alpha_logit = nn.Parameter(torch.tensor(alpha_logit_value, dtype=torch.float32))
        else:
            self.register_buffer("alpha_logit_buffer", torch.tensor(alpha_logit_value, dtype=torch.float32))

        self.global_encoder = SPIFEncoder(
            input_dim=hidden_dim,
            stable_dim=self.stable_dim,
            variant_dim=self.variant_dim,
            gate_hidden=int(gate_hidden),
            token_l2norm=self.token_l2norm,
        )

        self.local_token_norm = nn.LayerNorm(self.local_dim)
        if self.spif_asym_share_local_head:
            self.local_head = None
            self.local_align = (
                nn.Identity()
                if self.local_dim == self.stable_dim
                else nn.Linear(self.stable_dim, self.local_dim, bias=False)
            )
        else:
            self.local_head = _make_projection_head(hidden_dim, self.local_dim)
            self.local_align = nn.Identity()

    def _base_alpha_logit(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.learnable_alpha:
            return self.alpha_logit.to(device=device, dtype=dtype)
        return self.alpha_logit_buffer.to(device=device, dtype=dtype)

    def _normalize_local_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = self.local_token_norm(tokens)
        if self.token_l2norm:
            tokens = F.normalize(tokens, p=2, dim=-1)
        return tokens

    def _encode_local_tokens(
        self,
        raw_tokens: torch.Tensor,
        global_stable_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if self.spif_asym_share_local_head:
            local_tokens = self.local_align(global_stable_tokens)
        else:
            local_tokens = self.local_head(raw_tokens)
        return self._normalize_local_tokens(local_tokens)

    def _encode_episode(self, query: torch.Tensor, support: torch.Tensor) -> dict[str, torch.Tensor]:
        way_num, shot_num = support.shape[:2]
        flat_support = support.reshape(way_num * shot_num, *support.shape[-3:])

        query_raw_tokens = feature_map_to_tokens(self.encode(query))
        support_raw_tokens = feature_map_to_tokens(self.encode(flat_support))

        query_global_outputs = self.global_encoder(
            query_raw_tokens,
            factorization_on=self.factorization_on,
            gate_on=self.gate_on,
        )
        support_global_outputs = self.global_encoder(
            support_raw_tokens,
            factorization_on=self.factorization_on,
            gate_on=self.gate_on,
        )

        query_local_tokens = self._encode_local_tokens(
            raw_tokens=query_raw_tokens,
            global_stable_tokens=query_global_outputs.stable_tokens,
        )
        support_local_tokens = self._encode_local_tokens(
            raw_tokens=support_raw_tokens,
            global_stable_tokens=support_global_outputs.stable_tokens,
        )

        return {
            "query_global": query_global_outputs.stable_global,
            "query_local_tokens": query_local_tokens,
            "query_variant_global": query_global_outputs.variant_global,
            "query_gate": query_global_outputs.gate,
            "support_global": support_global_outputs.stable_global.reshape(way_num, shot_num, -1),
            "support_local_tokens": support_local_tokens.reshape(way_num, shot_num, -1, query_local_tokens.shape[-1]),
            "support_variant_global": support_global_outputs.variant_global.reshape(way_num, shot_num, -1),
            "support_gate": support_global_outputs.gate.reshape(way_num, shot_num, -1, 1),
        }

    @staticmethod
    def build_support_prototypes(support_global: torch.Tensor) -> torch.Tensor:
        prototypes = support_global.mean(dim=1)
        return F.normalize(prototypes, p=2, dim=-1)

    def compute_global_scores(self, query_global: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        cosine_scores = torch.matmul(
            F.normalize(query_global, p=2, dim=-1),
            F.normalize(prototypes, p=2, dim=-1).transpose(0, 1),
        )
        return self.spif_asym_global_scale * cosine_scores

    def compute_partial_matching_scores(
        self,
        query_tokens: torch.Tensor,
        support_tokens: torch.Tensor,
    ) -> torch.Tensor:
        query_tokens = F.normalize(query_tokens, p=2, dim=-1)
        support_tokens = F.normalize(support_tokens, p=2, dim=-1)
        similarity = torch.einsum("qld,wtd->qwlt", query_tokens, support_tokens)
        support_top_r = min(self.top_r, similarity.shape[-1])
        q_to_s = torch.topk(similarity, k=support_top_r, dim=-1).values.mean(dim=-1).mean(dim=-1)
        if self.spif_asym_local_score_mode == "query_to_support":
            return q_to_s

        query_top_r = min(self.top_r, similarity.shape[-2])
        s_to_q = torch.topk(similarity, k=query_top_r, dim=-2).values.mean(dim=-2).mean(dim=-1)
        return 0.5 * (q_to_s + s_to_q)

    def compute_pooled_local_scores(
        self,
        query_tokens: torch.Tensor,
        support_tokens: torch.Tensor,
    ) -> torch.Tensor:
        way_num, shot_num, token_num, dim = support_tokens.shape
        support_token_pool = support_tokens.reshape(way_num, shot_num * token_num, dim)
        local_scores = self.compute_partial_matching_scores(query_tokens, support_token_pool)
        return self.spif_asym_local_scale * local_scores

    def compute_per_shot_local_scores(
        self,
        query_tokens: torch.Tensor,
        support_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if support_tokens.dim() != 4:
            raise ValueError(
                "support_tokens must have shape (Way, Shot, Tokens, Dim), "
                f"got {tuple(support_tokens.shape)}"
            )
        way_num, shot_num, token_num, dim = support_tokens.shape
        flat_support = support_tokens.reshape(way_num * shot_num, token_num, dim)
        per_shot_scores = self.compute_partial_matching_scores(query_tokens, flat_support)
        per_shot_scores = per_shot_scores.reshape(query_tokens.shape[0], way_num, shot_num)
        return self.spif_asym_local_scale * per_shot_scores

    def aggregate_shot_scores(
        self,
        per_shot_local_scores: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.spif_asym_shot_agg == "mean":
            shot_num = per_shot_local_scores.shape[-1]
            shot_weights = torch.full_like(per_shot_local_scores, 1.0 / float(max(shot_num, 1)))
            return per_shot_local_scores.mean(dim=-1), shot_weights

        shot_logits = self.spif_asym_shot_softmax_beta * per_shot_local_scores
        shot_weights = torch.softmax(shot_logits, dim=-1)
        class_scores = torch.sum(shot_weights * per_shot_local_scores, dim=-1)
        return class_scores, shot_weights

    @staticmethod
    def compute_branch_margins(scores: torch.Tensor) -> torch.Tensor:
        if scores.shape[-1] < 2:
            return scores.new_zeros(scores.shape[0])
        top2 = torch.topk(scores, k=2, dim=-1).values
        return top2[:, 0] - top2[:, 1]

    def fuse_scores(
        self,
        global_scores: torch.Tensor,
        local_scores: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.global_only:
            alpha_q = torch.ones(global_scores.shape[0], 1, device=global_scores.device, dtype=global_scores.dtype)
            global_margin = self.compute_branch_margins(global_scores)
            local_margin = self.compute_branch_margins(local_scores)
            return global_scores, alpha_q, global_margin, local_margin
        if self.local_only:
            alpha_q = torch.zeros(global_scores.shape[0], 1, device=global_scores.device, dtype=global_scores.dtype)
            global_margin = self.compute_branch_margins(global_scores)
            local_margin = self.compute_branch_margins(local_scores)
            return local_scores, alpha_q, global_margin, local_margin

        global_margin = self.compute_branch_margins(global_scores)
        local_margin = self.compute_branch_margins(local_scores)
        if self.spif_asym_fusion_mode == "fixed":
            alpha = torch.sigmoid(self._base_alpha_logit(global_scores.device, global_scores.dtype))
            alpha_q = alpha.expand(global_scores.shape[0], 1)
        else:
            alpha_bias = self._base_alpha_logit(global_scores.device, global_scores.dtype)
            margin_delta = (global_margin - local_margin).unsqueeze(-1)
            alpha_q = torch.sigmoid(alpha_bias + self.spif_asym_fusion_kappa * margin_delta)
        logits = alpha_q * global_scores + (1.0 - alpha_q) * local_scores
        return logits, alpha_q, global_margin, local_margin

    def _forward_episode(self, query: torch.Tensor, support: torch.Tensor) -> dict[str, torch.Tensor]:
        episode = self._encode_episode(query, support)
        prototypes = self.build_support_prototypes(episode["support_global"])
        global_scores = self.compute_global_scores(episode["query_global"], prototypes)

        if self.spif_asym_shot_agg == "pooled":
            local_scores = self.compute_pooled_local_scores(
                query_tokens=episode["query_local_tokens"],
                support_tokens=episode["support_local_tokens"],
            )
            per_shot_local_scores = None
            shot_weights = None
        else:
            per_shot_local_scores = self.compute_per_shot_local_scores(
                query_tokens=episode["query_local_tokens"],
                support_tokens=episode["support_local_tokens"],
            )
            local_scores, shot_weights = self.aggregate_shot_scores(per_shot_local_scores)

        logits, alpha_q, global_margin, local_margin = self.fuse_scores(global_scores, local_scores)
        mean_gate = torch.cat(
            [
                episode["query_gate"].reshape(-1, 1),
                episode["support_gate"].reshape(-1, 1),
            ],
            dim=0,
        ).mean()
        output = {
            "logits": logits,
            "aux_loss": logits.new_zeros(()),
            "global_scores": global_scores.detach(),
            "local_scores": local_scores.detach(),
            "alpha": alpha_q.mean().detach(),
            "alpha_q": alpha_q.detach(),
            "global_margin": global_margin.detach(),
            "local_margin": local_margin.detach(),
            "mean_gate": mean_gate.detach(),
            "stable_global_embeddings": torch.cat(
                [
                    episode["query_global"],
                    episode["support_global"].reshape(-1, episode["support_global"].shape[-1]),
                ],
                dim=0,
            ).detach(),
            "variant_global_embeddings": torch.cat(
                [
                    episode["query_variant_global"],
                    episode["support_variant_global"].reshape(-1, episode["support_variant_global"].shape[-1]),
                ],
                dim=0,
            ).detach(),
        }
        if per_shot_local_scores is not None:
            output["per_shot_local_scores"] = per_shot_local_scores.detach()
        if shot_weights is not None:
            output["shot_aggregation_weights"] = shot_weights.detach()
        return output

    def forward(
        self,
        query: torch.Tensor,
        support: torch.Tensor,
        return_aux: bool = False,
    ) -> torch.Tensor | Dict[str, torch.Tensor]:
        bsz, _, _, _, _, _ = self.validate_episode_inputs(query, support)
        batch_outputs = []
        diagnostics = []
        for batch_idx in range(bsz):
            episode = self._forward_episode(query[batch_idx], support[batch_idx])
            batch_outputs.append(episode["logits"])
            diagnostics.append(episode)

        logits = torch.cat(batch_outputs, dim=0)
        if not return_aux:
            return logits

        payload = {
            "logits": logits,
            "global_scores": torch.cat([item["global_scores"] for item in diagnostics], dim=0),
            "local_scores": torch.cat([item["local_scores"] for item in diagnostics], dim=0),
            "alpha": torch.stack([item["alpha"] for item in diagnostics]).mean(),
            "alpha_q": torch.cat([item["alpha_q"] for item in diagnostics], dim=0),
            "global_margin": torch.cat([item["global_margin"] for item in diagnostics], dim=0),
            "local_margin": torch.cat([item["local_margin"] for item in diagnostics], dim=0),
            "mean_gate": torch.stack([item["mean_gate"] for item in diagnostics]).mean(),
            "stable_global_embeddings": torch.cat(
                [item["stable_global_embeddings"] for item in diagnostics],
                dim=0,
            ),
            "variant_global_embeddings": torch.cat(
                [item["variant_global_embeddings"] for item in diagnostics],
                dim=0,
            ),
        }
        for key in ("per_shot_local_scores", "shot_aggregation_weights"):
            if key in diagnostics[0]:
                payload[key] = torch.cat([item[key] for item in diagnostics], dim=0)
        return payload
