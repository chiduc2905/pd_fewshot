"""SPIFAEB-v2: class-level adaptive evidence retention with a principled local head.

This redesign keeps the strong SPIF/SPIFAEB global branch and the class-pooled
few-shot protocol, but rewrites the local branch around three principles:

1. local tokens are built as a stable-anchor residual refinement instead of a
   free-standing branch that can drift away from the SPIF stable trunk;
2. local matching is anchored to the strong SPIF partial matcher and the
   adaptive controller only refines that anchor instead of replacing it;
2. the evidence controller is predicted from a competition-aware local
   similarity profile through a monotonic class prior plus a row-wise monotonic
   refinement, not from a shortcut through global embeddings;
3. the evidence budget is applied through a monotonic rank-gated operator on
   sorted local similarities, not a threshold-plus-fallback heuristic;
4. fused inference is supervised with branch-specific losses and an explicit
   controller ranking loss so the local branch must remain semantically aligned
   instead of being carried by the fused logits alone.
"""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from net.fewshot_common import BaseConv64FewShotModel, feature_map_to_tokens
from net.spif import SPIFEncoder, _make_projection_head


class SPIFAEBV2(BaseConv64FewShotModel):
    """SPIFAEB-v2 with a separate local head and monotonic rank-budget matching."""

    _VALID_FUSION_MODES = frozenset({"fixed", "margin_adaptive", "residual"})
    _VALID_LOCAL_SCORE_MODES = frozenset({"query_to_support", "bidirectional"})
    _VALID_LOCAL_MIX_MODES = frozenset({"fixed", "prior_adaptive"})

    requires_query_targets = True

    def __init__(
        self,
        in_channels: int = 3,
        hidden_dim: int = 64,
        stable_dim: int = 64,
        variant_dim: int = 64,
        local_dim: int | None = None,
        gate_hidden: int = 16,
        alpha_init: float = 0.7,
        learnable_alpha: bool = False,
        gate_on: bool = True,
        factorization_on: bool = True,
        global_only: bool = False,
        local_only: bool = False,
        token_l2norm: bool = True,
        aeb_v2_hidden: int | None = None,
        aeb_v2_min_budget: float = 0.2,
        aeb_v2_max_budget: float = 0.85,
        aeb_v2_rank_temperature: float = 0.35,
        aeb_v2_global_scale: float = 1.0,
        aeb_v2_local_scale: float = 1.0,
        aeb_v2_cond_global_weight: float = 0.35,
        aeb_v2_cond_global_temperature: float = 8.0,
        aeb_v2_cond_global_use_gate_prior: bool = True,
        aeb_v2_query_class_attention_on: bool = True,
        aeb_v2_query_class_attention_temperature: float = 8.0,
        aeb_v2_query_class_attention_use_gate_prior: bool = True,
        aeb_v2_anchor_top_r: int = 4,
        aeb_v2_anchor_mix: float = 0.35,
        aeb_v2_local_mix_mode: str = "prior_adaptive",
        aeb_v2_adaptive_mix_min: float = 0.2,
        aeb_v2_adaptive_mix_max: float = 0.8,
        aeb_v2_adaptive_mix_scale: float = 6.0,
        aeb_v2_fusion_mode: str = "fixed",
        aeb_v2_fusion_kappa: float = 8.0,
        aeb_v2_local_score_mode: str = "query_to_support",
        aeb_v2_share_local_head: bool = False,
        aeb_v2_query_gate_weighting: bool = False,
        aeb_v2_detach_budget_context: bool = True,
        aeb_v2_detach_local_backbone: bool = True,
        aeb_v2_local_residual_scale: float = 0.25,
        aeb_v2_budget_prior_scale: float = 6.0,
        aeb_v2_budget_residual_scale: float = 0.15,
        aeb_v2_budget_rank_weight: float = 0.1,
        aeb_v2_budget_rank_margin: float = 0.05,
        aeb_v2_budget_residual_reg_weight: float = 0.05,
        aeb_v2_support_coverage_temperature: float = 8.0,
        aeb_v2_coverage_bonus_weight: float = 0.0,
        aeb_v2_competition_weight: float = 1.0,
        aeb_v2_competition_temperature: float = 8.0,
        aeb_v2_local_margin_weight: float = 0.1,
        aeb_v2_local_margin_target: float = 0.5,
        aeb_v2_anchor_consistency_weight: float = 0.02,
        aeb_v2_global_ce_weight: float = 0.0,
        aeb_v2_local_ce_weight: float = 0.05,
        aeb_v2_eps: float = 1e-6,
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
        if aeb_v2_hidden is not None and int(aeb_v2_hidden) <= 0:
            raise ValueError("aeb_v2_hidden must be positive")
        if not (0.0 < float(aeb_v2_min_budget) <= float(aeb_v2_max_budget) <= 1.0):
            raise ValueError("aeb_v2 budgets must satisfy 0 < min_budget <= max_budget <= 1")
        if float(aeb_v2_rank_temperature) <= 0.0:
            raise ValueError("aeb_v2_rank_temperature must be positive")
        if float(aeb_v2_global_scale) <= 0.0:
            raise ValueError("aeb_v2_global_scale must be positive")
        if float(aeb_v2_local_scale) <= 0.0:
            raise ValueError("aeb_v2_local_scale must be positive")
        if not (0.0 <= float(aeb_v2_cond_global_weight) <= 1.0):
            raise ValueError("aeb_v2_cond_global_weight must be in [0, 1]")
        if float(aeb_v2_cond_global_temperature) <= 0.0:
            raise ValueError("aeb_v2_cond_global_temperature must be positive")
        if float(aeb_v2_query_class_attention_temperature) <= 0.0:
            raise ValueError("aeb_v2_query_class_attention_temperature must be positive")
        if int(aeb_v2_anchor_top_r) <= 0:
            raise ValueError("aeb_v2_anchor_top_r must be positive")
        if not (0.0 <= float(aeb_v2_anchor_mix) <= 1.0):
            raise ValueError("aeb_v2_anchor_mix must be in [0, 1]")
        if aeb_v2_local_mix_mode not in self._VALID_LOCAL_MIX_MODES:
            raise ValueError(
                f"aeb_v2_local_mix_mode must be one of {sorted(self._VALID_LOCAL_MIX_MODES)}, "
                f"got {aeb_v2_local_mix_mode!r}"
            )
        if not (0.0 <= float(aeb_v2_adaptive_mix_min) <= float(aeb_v2_adaptive_mix_max) <= 1.0):
            raise ValueError(
                "aeb_v2_adaptive_mix_min/max must satisfy 0 <= min <= max <= 1"
            )
        if float(aeb_v2_adaptive_mix_scale) < 0.0:
            raise ValueError("aeb_v2_adaptive_mix_scale must be non-negative")
        if float(aeb_v2_fusion_kappa) < 0.0:
            raise ValueError("aeb_v2_fusion_kappa must be non-negative")
        if float(aeb_v2_budget_prior_scale) < 0.0:
            raise ValueError("aeb_v2_budget_prior_scale must be non-negative")
        if float(aeb_v2_budget_residual_scale) < 0.0:
            raise ValueError("aeb_v2_budget_residual_scale must be non-negative")
        if float(aeb_v2_budget_rank_weight) < 0.0:
            raise ValueError("aeb_v2_budget_rank_weight must be non-negative")
        if float(aeb_v2_budget_rank_margin) < 0.0:
            raise ValueError("aeb_v2_budget_rank_margin must be non-negative")
        if float(aeb_v2_budget_residual_reg_weight) < 0.0:
            raise ValueError("aeb_v2_budget_residual_reg_weight must be non-negative")
        if float(aeb_v2_support_coverage_temperature) <= 0.0:
            raise ValueError("aeb_v2_support_coverage_temperature must be positive")
        if float(aeb_v2_coverage_bonus_weight) < 0.0:
            raise ValueError("aeb_v2_coverage_bonus_weight must be non-negative")
        if float(aeb_v2_competition_weight) < 0.0:
            raise ValueError("aeb_v2_competition_weight must be non-negative")
        if float(aeb_v2_competition_temperature) <= 0.0:
            raise ValueError("aeb_v2_competition_temperature must be positive")
        if float(aeb_v2_local_margin_weight) < 0.0:
            raise ValueError("aeb_v2_local_margin_weight must be non-negative")
        if float(aeb_v2_local_margin_target) < 0.0:
            raise ValueError("aeb_v2_local_margin_target must be non-negative")
        if float(aeb_v2_anchor_consistency_weight) < 0.0:
            raise ValueError("aeb_v2_anchor_consistency_weight must be non-negative")
        if float(aeb_v2_global_ce_weight) < 0.0 or float(aeb_v2_local_ce_weight) < 0.0:
            raise ValueError("aeb_v2 branch CE weights must be non-negative")
        if float(aeb_v2_local_residual_scale) < 0.0:
            raise ValueError("aeb_v2_local_residual_scale must be non-negative")
        if float(aeb_v2_eps) <= 0.0:
            raise ValueError("aeb_v2_eps must be positive")
        if aeb_v2_fusion_mode not in self._VALID_FUSION_MODES:
            raise ValueError(
                f"aeb_v2_fusion_mode must be one of {sorted(self._VALID_FUSION_MODES)}, "
                f"got {aeb_v2_fusion_mode!r}"
            )
        if aeb_v2_local_score_mode not in self._VALID_LOCAL_SCORE_MODES:
            raise ValueError(
                f"aeb_v2_local_score_mode must be one of {sorted(self._VALID_LOCAL_SCORE_MODES)}, "
                f"got {aeb_v2_local_score_mode!r}"
            )

        self.global_only = bool(global_only)
        self.local_only = bool(local_only)
        self.gate_on = bool(gate_on)
        self.factorization_on = bool(factorization_on)
        self.token_l2norm = bool(token_l2norm)
        self.learnable_alpha = bool(learnable_alpha)
        self.stable_dim = int(stable_dim)
        self.variant_dim = int(variant_dim)
        self.local_dim = int(self.stable_dim if local_dim is None else local_dim)
        self.aeb_v2_hidden = int(aeb_v2_hidden or max(16, self.stable_dim // 2))
        self.aeb_v2_min_budget = float(aeb_v2_min_budget)
        self.aeb_v2_max_budget = float(aeb_v2_max_budget)
        self.aeb_v2_rank_temperature = float(aeb_v2_rank_temperature)
        self.aeb_v2_global_scale = float(aeb_v2_global_scale)
        self.aeb_v2_local_scale = float(aeb_v2_local_scale)
        self.aeb_v2_cond_global_weight = float(aeb_v2_cond_global_weight)
        self.aeb_v2_cond_global_temperature = float(aeb_v2_cond_global_temperature)
        self.aeb_v2_cond_global_use_gate_prior = bool(aeb_v2_cond_global_use_gate_prior)
        self.aeb_v2_query_class_attention_on = bool(aeb_v2_query_class_attention_on)
        self.aeb_v2_query_class_attention_temperature = float(aeb_v2_query_class_attention_temperature)
        self.aeb_v2_query_class_attention_use_gate_prior = bool(aeb_v2_query_class_attention_use_gate_prior)
        self.aeb_v2_anchor_top_r = int(aeb_v2_anchor_top_r)
        self.aeb_v2_anchor_mix = float(aeb_v2_anchor_mix)
        self.aeb_v2_local_mix_mode = str(aeb_v2_local_mix_mode)
        self.aeb_v2_adaptive_mix_min = float(aeb_v2_adaptive_mix_min)
        self.aeb_v2_adaptive_mix_max = float(aeb_v2_adaptive_mix_max)
        self.aeb_v2_adaptive_mix_scale = float(aeb_v2_adaptive_mix_scale)
        self.aeb_v2_fusion_mode = str(aeb_v2_fusion_mode)
        self.aeb_v2_fusion_kappa = float(aeb_v2_fusion_kappa)
        self.aeb_v2_local_score_mode = str(aeb_v2_local_score_mode)
        self.aeb_v2_share_local_head = bool(aeb_v2_share_local_head)
        self.aeb_v2_query_gate_weighting = bool(aeb_v2_query_gate_weighting)
        self.aeb_v2_detach_budget_context = bool(aeb_v2_detach_budget_context)
        self.aeb_v2_detach_local_backbone = bool(aeb_v2_detach_local_backbone)
        self.aeb_v2_local_residual_scale = float(aeb_v2_local_residual_scale)
        self.aeb_v2_budget_prior_scale = float(aeb_v2_budget_prior_scale)
        self.aeb_v2_budget_residual_scale = float(aeb_v2_budget_residual_scale)
        self.aeb_v2_budget_rank_weight = float(aeb_v2_budget_rank_weight)
        self.aeb_v2_budget_rank_margin = float(aeb_v2_budget_rank_margin)
        self.aeb_v2_budget_residual_reg_weight = float(aeb_v2_budget_residual_reg_weight)
        self.aeb_v2_support_coverage_temperature = float(aeb_v2_support_coverage_temperature)
        self.aeb_v2_coverage_bonus_weight = float(aeb_v2_coverage_bonus_weight)
        self.aeb_v2_competition_weight = float(aeb_v2_competition_weight)
        self.aeb_v2_competition_temperature = float(aeb_v2_competition_temperature)
        self.aeb_v2_local_margin_weight = float(aeb_v2_local_margin_weight)
        self.aeb_v2_local_margin_target = float(aeb_v2_local_margin_target)
        self.aeb_v2_anchor_consistency_weight = float(aeb_v2_anchor_consistency_weight)
        self.aeb_v2_global_ce_weight = float(aeb_v2_global_ce_weight)
        self.aeb_v2_local_ce_weight = float(aeb_v2_local_ce_weight)
        self.aeb_v2_eps = float(aeb_v2_eps)

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
        if self.aeb_v2_share_local_head:
            self.local_head = None
            self.local_anchor_align = (
                nn.Identity()
                if self.local_dim == self.stable_dim
                else nn.Linear(self.stable_dim, self.local_dim, bias=False)
            )
        else:
            self.local_head = _make_projection_head(hidden_dim, self.local_dim)
            self.local_anchor_align = (
                nn.Identity()
                if self.local_dim == self.stable_dim
                else nn.Linear(self.stable_dim, self.local_dim, bias=False)
            )

        # Kept for CLI/state compatibility. The controller below is now
        # analytic and monotonic, so no learned budget MLP is used in the main
        # path.
        self.local_budget_stat_dim = 6

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
        anchor_tokens = self.local_anchor_align(global_stable_tokens)
        if self.aeb_v2_share_local_head:
            local_tokens = anchor_tokens
        else:
            residual_tokens = self.local_head(raw_tokens)
            local_tokens = anchor_tokens + self.aeb_v2_local_residual_scale * residual_tokens
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

        query_local_raw_tokens = query_raw_tokens.detach() if self.aeb_v2_detach_local_backbone else query_raw_tokens
        support_local_raw_tokens = support_raw_tokens.detach() if self.aeb_v2_detach_local_backbone else support_raw_tokens
        query_local_stable_tokens = (
            query_global_outputs.stable_tokens.detach() if self.aeb_v2_detach_local_backbone else query_global_outputs.stable_tokens
        )
        support_local_stable_tokens = (
            support_global_outputs.stable_tokens.detach()
            if self.aeb_v2_detach_local_backbone
            else support_global_outputs.stable_tokens
        )

        query_local_tokens = self._encode_local_tokens(
            raw_tokens=query_local_raw_tokens,
            global_stable_tokens=query_local_stable_tokens,
        )
        support_local_tokens = self._encode_local_tokens(
            raw_tokens=support_local_raw_tokens,
            global_stable_tokens=support_local_stable_tokens,
        )

        return {
            "query_global": query_global_outputs.stable_global,
            "query_stable_tokens": query_global_outputs.stable_tokens,
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

    @staticmethod
    def build_support_token_pool(support_tokens: torch.Tensor) -> torch.Tensor:
        way_num, shot_num, token_num, dim = support_tokens.shape
        pooled = support_tokens.reshape(way_num, shot_num * token_num, dim)
        return F.normalize(pooled, p=2, dim=-1)

    @staticmethod
    def build_support_row_weights(
        support_gate: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        # support_gate: [Way, Shot, Tokens, 1] -> [Way, Shot*Tokens]
        row_weights = support_gate.squeeze(-1).reshape(support_gate.shape[0], -1)
        row_weights = row_weights.clamp_min(0.0)
        normalizer = row_weights.sum(dim=-1, keepdim=True).clamp_min(eps)
        return row_weights / normalizer

    @staticmethod
    def compute_global_scores(query_global: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        return torch.matmul(
            F.normalize(query_global, p=2, dim=-1),
            F.normalize(prototypes, p=2, dim=-1).transpose(0, 1),
        )

    def compute_conditioned_global_scores(
        self,
        query_stable_tokens: torch.Tensor,
        query_gate: torch.Tensor,
        prototypes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prototype_bank = F.normalize(prototypes, p=2, dim=-1)
        token_similarity = torch.einsum("qld,wd->qwl", query_stable_tokens, prototype_bank)
        attention_logits = self.aeb_v2_cond_global_temperature * token_similarity
        if self.aeb_v2_cond_global_use_gate_prior:
            gate_prior = query_gate.squeeze(-1).clamp_min(self.aeb_v2_eps).log().unsqueeze(1)
            attention_logits = attention_logits + gate_prior
        attention = torch.softmax(attention_logits, dim=-1)
        conditioned_query = torch.einsum("qwl,qld->qwd", attention, query_stable_tokens)
        conditioned_query = F.normalize(conditioned_query, p=2, dim=-1)
        conditioned_scores = torch.sum(conditioned_query * prototype_bank.unsqueeze(0), dim=-1)
        attention_entropy = -(attention * attention.clamp_min(self.aeb_v2_eps).log()).sum(dim=-1)
        return conditioned_scores, attention_entropy

    def _normalize_row_weights(
        self,
        row_weights: torch.Tensor | None,
        values: torch.Tensor,
    ) -> torch.Tensor:
        if row_weights is None:
            num_rows = values.shape[-1]
            return values.new_full(values.shape, 1.0 / float(max(1, num_rows)))
        weights = row_weights.clamp_min(0.0)
        return weights / weights.sum(dim=-1, keepdim=True).clamp_min(self.aeb_v2_eps)

    def _weighted_mean_and_std(
        self,
        values: torch.Tensor,
        row_weights: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weights = self._normalize_row_weights(row_weights, values)
        mean = torch.sum(weights * values, dim=-1)
        centered = values - mean.unsqueeze(-1)
        std = torch.sqrt(torch.sum(weights * centered.square(), dim=-1).clamp_min(self.aeb_v2_eps))
        return mean, std

    def _normalized_entropy(
        self,
        probs: torch.Tensor,
        dim: int = -1,
    ) -> torch.Tensor:
        probs = probs.clamp_min(self.aeb_v2_eps)
        entropy = -(probs * probs.log()).sum(dim=dim)
        size = max(2, probs.shape[dim])
        return entropy / math.log(float(size))

    def _hard_negative_per_class(self, values: torch.Tensor) -> torch.Tensor:
        if values.shape[1] == 1:
            return values
        top2_values, top2_indices = torch.topk(values, k=2, dim=1)
        hardest_negative = top2_values[:, :1].expand_as(values).clone()
        hardest_negative.scatter_(1, top2_indices[:, :1], top2_values[:, 1:2])
        return hardest_negative

    def predict_budget(
        self,
        local_evidence_profile: dict[str, torch.Tensor],
        row_weights: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        row_quality = local_evidence_profile["row_advantage"]
        row_attention = local_evidence_profile["coverage_attention"]
        row_sharpness = local_evidence_profile["row_sharpness"]
        normalized_row_weights = self._normalize_row_weights(row_weights, row_quality)
        support_coverage = torch.einsum("qwl,qwlt->qwt", normalized_row_weights, row_attention)
        support_coverage = support_coverage / support_coverage.sum(dim=-1, keepdim=True).clamp_min(self.aeb_v2_eps)
        coverage_entropy = self._normalized_entropy(support_coverage, dim=-1)
        coverage_concentration = support_coverage.square().sum(dim=-1)

        row_coverage_entropy = self._normalized_entropy(row_attention, dim=-1)
        row_coverage_concentration = row_attention.square().sum(dim=-1)
        class_quality, _ = self._weighted_mean_and_std(row_quality, row_weights)
        class_sharpness, class_dispersion = self._weighted_mean_and_std(row_sharpness, row_weights)

        budget_range = self.aeb_v2_max_budget - self.aeb_v2_min_budget
        prior_signal = class_dispersion - class_sharpness - class_quality
        prior_unit = torch.sigmoid(self.aeb_v2_budget_prior_scale * prior_signal)
        rho_prior = self.aeb_v2_min_budget + budget_range * prior_unit

        row_delta = (class_sharpness.unsqueeze(-1) - row_sharpness) + 0.5 * (
            class_quality.unsqueeze(-1) - row_quality
        )
        row_refinement = torch.tanh(self.aeb_v2_budget_prior_scale * row_delta)
        row_budget = rho_prior.unsqueeze(-1) + budget_range * self.aeb_v2_budget_residual_scale * row_refinement
        row_budget = row_budget.clamp(min=self.aeb_v2_min_budget, max=self.aeb_v2_max_budget)

        rho = torch.sum(normalized_row_weights * row_budget, dim=-1)
        rho_residual = (rho - rho_prior) / max(budget_range, self.aeb_v2_eps)
        row_budget_std = torch.sqrt(
            torch.sum(
                normalized_row_weights * (row_budget - rho.unsqueeze(-1)).square(),
                dim=-1,
            ).clamp_min(self.aeb_v2_eps)
        )
        return {
            "rho": rho,
            "rho_prior": rho_prior,
            "rho_residual": rho_residual,
            "row_budget": row_budget,
            "row_budget_refinement": row_refinement,
            "row_budget_std": row_budget_std,
            "evidence_sharpness": class_sharpness,
            "evidence_dispersion": class_dispersion,
            "evidence_quality": class_quality,
            "coverage_entropy": coverage_entropy,
            "coverage_concentration": coverage_concentration,
            "row_coverage_entropy": row_coverage_entropy,
            "row_coverage_concentration": row_coverage_concentration,
            "evidence_advantage": class_quality,
        }

    def build_local_evidence_profile(
        self,
        query_tokens: torch.Tensor,
        support_token_pool: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        query_tokens = F.normalize(query_tokens, p=2, dim=-1)
        support_token_pool = F.normalize(support_token_pool, p=2, dim=-1)
        similarity = torch.einsum("qld,wtd->qwlt", query_tokens, support_token_pool)
        row_topk = torch.topk(similarity, k=min(2, similarity.shape[-1]), dim=-1).values
        row_top1 = row_topk[..., 0]
        row_top2 = row_topk[..., 1] if row_topk.shape[-1] > 1 else row_top1
        row_mean = similarity.mean(dim=-1)
        row_std = similarity.var(dim=-1, unbiased=False).add(self.aeb_v2_eps).sqrt()
        row_peak = row_top1 - row_top2
        row_top1_competitor = self._hard_negative_per_class(row_top1)
        row_mean_competitor = self._hard_negative_per_class(row_mean)
        row_advantage = row_top1 - row_top1_competitor
        row_contrastive = row_mean - row_mean_competitor
        row_contrast = row_top1 - row_mean
        row_sharpness = row_peak + row_contrast + self.aeb_v2_competition_weight * row_advantage + 0.5 * row_contrastive
        coverage_attention = torch.softmax(self.aeb_v2_support_coverage_temperature * similarity, dim=-1)

        top1 = row_advantage.mean(dim=-1)
        topk = min(4, similarity.shape[-1])
        topk_mean = torch.topk(similarity, k=topk, dim=-1).values.mean(dim=-1).mean(dim=-1)
        sim_mean = row_mean.mean(dim=-1)
        sim_std = row_std.mean(dim=-1)
        top1_std = row_advantage.var(dim=-1, unbiased=False).add(self.aeb_v2_eps).sqrt()
        stats = torch.stack([top1, top1_std, topk_mean, sim_mean, sim_std, row_sharpness.mean(dim=-1)], dim=-1)
        return {
            "similarity": similarity,
            "stats": stats,
            "row_top1": row_top1,
            "row_top2": row_top2,
            "row_mean": row_mean,
            "row_std": row_std,
            "row_sharpness": row_sharpness,
            "row_advantage": row_advantage,
            "row_top1_competitor": row_top1_competitor,
            "coverage_attention": coverage_attention,
        }

    def compute_query_class_row_weights(
        self,
        local_evidence_profile: dict[str, torch.Tensor],
        query_gate: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        row_saliency = local_evidence_profile["row_sharpness"] + 0.5 * local_evidence_profile["row_advantage"]
        attention_logits = self.aeb_v2_query_class_attention_temperature * row_saliency
        if self.aeb_v2_query_class_attention_use_gate_prior:
            gate_prior = query_gate.detach().squeeze(-1).clamp_min(self.aeb_v2_eps).log().unsqueeze(1)
            attention_logits = attention_logits + gate_prior
        weights = torch.softmax(attention_logits, dim=-1)
        entropy = -(weights * weights.clamp_min(self.aeb_v2_eps).log()).sum(dim=-1)
        return weights, entropy

    @staticmethod
    def compute_query_row_weights(query_gate: torch.Tensor, eps: float) -> torch.Tensor:
        row_weights = query_gate.squeeze(-1).clamp_min(0.0)
        normalizer = row_weights.sum(dim=-1, keepdim=True).clamp_min(eps)
        return row_weights / normalizer

    def _expand_row_weights(
        self,
        row_weights: torch.Tensor,
        row_scores: torch.Tensor,
    ) -> torch.Tensor:
        if row_weights.dim() == 2 and row_weights.shape == (row_scores.shape[0], row_scores.shape[2]):
            return row_weights.unsqueeze(1).expand(-1, row_scores.shape[1], -1)
        if row_weights.dim() == 2 and row_weights.shape == (row_scores.shape[1], row_scores.shape[2]):
            return row_weights.unsqueeze(0).expand(row_scores.shape[0], -1, -1)
        if row_weights.dim() == 3 and row_weights.shape == row_scores.shape:
            return row_weights
        raise ValueError(
            "row_weights must match [NumQuery, NumRows], [Way, NumRows], or [NumQuery, Way, NumRows]: "
            f"row_weights={tuple(row_weights.shape)} row_scores={tuple(row_scores.shape)}"
        )

    def _aggregate_row_scores(
        self,
        row_scores: torch.Tensor,
        row_weights: torch.Tensor | None,
    ) -> torch.Tensor:
        if row_weights is None:
            return row_scores.mean(dim=-1)
        weights = self._expand_row_weights(row_weights, row_scores)
        return torch.sum(weights * row_scores, dim=-1)

    def _reduce_budgeted_rows(
        self,
        sorted_similarity: torch.Tensor,
        budget_ratio: torch.Tensor,
        row_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # sorted_similarity: [NumQuery, Way, NumRows, NumCols]
        num_cols = int(sorted_similarity.shape[-1])
        rank_positions = torch.arange(
            1,
            num_cols + 1,
            device=sorted_similarity.device,
            dtype=sorted_similarity.dtype,
        ).view(1, 1, 1, num_cols)
        if budget_ratio.dim() == 2:
            effective_k = budget_ratio.unsqueeze(-1).unsqueeze(-1) * float(num_cols)
        elif budget_ratio.dim() == 3:
            effective_k = budget_ratio.unsqueeze(-1) * float(num_cols)
        else:
            raise ValueError(
                f"budget_ratio must have shape [NumQuery, Way] or [NumQuery, Way, NumRows], got {tuple(budget_ratio.shape)}"
            )
        rank_gate = torch.sigmoid((effective_k + 0.5 - rank_positions) / self.aeb_v2_rank_temperature)
        rank_weights = rank_gate / rank_gate.sum(dim=-1, keepdim=True).clamp_min(self.aeb_v2_eps)
        row_scores = torch.sum(rank_weights * sorted_similarity, dim=-1)
        row_budget_counts = rank_gate.sum(dim=-1).expand(-1, -1, row_scores.shape[-1])

        scores = self._aggregate_row_scores(row_scores=row_scores, row_weights=row_weights)
        counts = self._aggregate_row_scores(row_scores=row_budget_counts, row_weights=row_weights)
        return scores, counts, row_scores

    def compute_anchor_local_scores(
        self,
        similarity: torch.Tensor,
        query_row_weights: torch.Tensor | None,
        support_row_weights: torch.Tensor | None,
    ) -> torch.Tensor:
        top_r = min(self.aeb_v2_anchor_top_r, similarity.shape[-1])
        q2s_row_scores = torch.topk(similarity, k=top_r, dim=-1).values.mean(dim=-1)
        q2s_scores = self._aggregate_row_scores(row_scores=q2s_row_scores, row_weights=query_row_weights)

        anchor_scores = q2s_scores
        if self.aeb_v2_local_score_mode == "bidirectional":
            s2q_row_scores = torch.topk(similarity.transpose(-1, -2), k=min(self.aeb_v2_anchor_top_r, similarity.shape[-2]), dim=-1).values.mean(dim=-1)
            s2q_scores = self._aggregate_row_scores(row_scores=s2q_row_scores, row_weights=support_row_weights)
            anchor_scores = 0.5 * (q2s_scores + s2q_scores)
        return self.aeb_v2_local_scale * anchor_scores

    def compute_local_budget_scores(
        self,
        similarity: torch.Tensor,
        rho: torch.Tensor,
        row_budget: torch.Tensor | None,
        query_row_weights: torch.Tensor | None,
        support_row_weights: torch.Tensor | None,
        coverage_bonus: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        sorted_q2s = torch.sort(similarity, dim=-1, descending=True).values
        q2s_raw_scores, q2s_counts, q2s_row_scores = self._reduce_budgeted_rows(
            sorted_similarity=sorted_q2s,
            budget_ratio=row_budget if row_budget is not None else rho,
            row_weights=query_row_weights,
        )
        q2s_competitor = self._hard_negative_per_class(q2s_row_scores)
        q2s_advantage = q2s_row_scores - q2s_competitor
        q2s_competitive_rows = F.softplus(
            self.aeb_v2_competition_temperature * q2s_advantage
        ) / self.aeb_v2_competition_temperature
        q2s_scores = self._aggregate_row_scores(row_scores=q2s_competitive_rows, row_weights=query_row_weights)

        local_scores = q2s_scores
        if self.aeb_v2_local_score_mode == "bidirectional":
            sorted_s2q = torch.sort(similarity.transpose(-1, -2), dim=-1, descending=True).values
            s2q_raw_scores, _, s2q_row_scores = self._reduce_budgeted_rows(
                sorted_similarity=sorted_s2q,
                budget_ratio=rho,
                row_weights=support_row_weights,
            )
            s2q_competitor = self._hard_negative_per_class(s2q_row_scores)
            s2q_advantage = s2q_row_scores - s2q_competitor
            s2q_competitive_rows = F.softplus(
                self.aeb_v2_competition_temperature * s2q_advantage
            ) / self.aeb_v2_competition_temperature
            s2q_scores = self._aggregate_row_scores(row_scores=s2q_competitive_rows, row_weights=support_row_weights)
            local_scores = 0.5 * (q2s_scores + s2q_scores)
        else:
            s2q_raw_scores = None
            s2q_scores = None

        if coverage_bonus is not None and self.aeb_v2_coverage_bonus_weight > 0.0:
            local_scores = local_scores + self.aeb_v2_coverage_bonus_weight * coverage_bonus

        support_token_count = float(max(1, similarity.shape[-1]))
        retained_fraction = q2s_counts / support_token_count
        return {
            "adaptive_local_scores": self.aeb_v2_local_scale * local_scores,
            "rho": rho,
            "active_match_counts": q2s_counts,
            "retained_fraction": retained_fraction,
            "raw_local_scores": self.aeb_v2_local_scale * q2s_raw_scores,
            "competitive_local_scores": self.aeb_v2_local_scale * q2s_scores,
        }

    def compute_local_mix(
        self,
        rho_prior: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.aeb_v2_local_mix_mode == "fixed":
            anchor_mix = rho_prior.new_full(rho_prior.shape, self.aeb_v2_anchor_mix)
            adaptive_mix = 1.0 - anchor_mix
            return anchor_mix, adaptive_mix

        budget_range = max(self.aeb_v2_max_budget - self.aeb_v2_min_budget, self.aeb_v2_eps)
        prior_unit = (rho_prior - self.aeb_v2_min_budget) / budget_range
        centered_confidence = 0.5 - prior_unit
        adaptive_unit = torch.sigmoid(self.aeb_v2_adaptive_mix_scale * centered_confidence)
        adaptive_mix = self.aeb_v2_adaptive_mix_min + (
            self.aeb_v2_adaptive_mix_max - self.aeb_v2_adaptive_mix_min
        ) * adaptive_unit
        anchor_mix = 1.0 - adaptive_mix
        return anchor_mix, adaptive_mix

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
        global_margin = self.compute_branch_margins(global_scores)
        local_margin = self.compute_branch_margins(local_scores)

        if self.global_only:
            alpha_q = torch.ones(global_scores.shape[0], 1, device=global_scores.device, dtype=global_scores.dtype)
            return global_scores, alpha_q, global_margin, local_margin
        if self.local_only:
            alpha_q = torch.zeros(global_scores.shape[0], 1, device=global_scores.device, dtype=global_scores.dtype)
            return local_scores, alpha_q, global_margin, local_margin

        if self.aeb_v2_fusion_mode == "residual":
            alpha = torch.sigmoid(self._base_alpha_logit(global_scores.device, global_scores.dtype))
            local_weight = 1.0 - alpha
            local_residual = local_scores - local_scores.mean(dim=-1, keepdim=True)
            alpha_q = alpha.expand(global_scores.shape[0], 1)
            fused = global_scores + local_weight * local_residual
            return fused, alpha_q, global_margin, local_margin

        if self.aeb_v2_fusion_mode == "fixed":
            alpha = torch.sigmoid(self._base_alpha_logit(global_scores.device, global_scores.dtype))
            alpha_q = alpha.expand(global_scores.shape[0], 1)
        else:
            alpha_base = torch.sigmoid(self._base_alpha_logit(global_scores.device, global_scores.dtype))
            margin_delta = (global_margin - local_margin).unsqueeze(-1)
            boost = (2.0 * torch.sigmoid(self.aeb_v2_fusion_kappa * margin_delta) - 1.0).clamp_min(0.0)
            alpha_q = alpha_base + (1.0 - alpha_base) * boost
        fused = alpha_q * global_scores + (1.0 - alpha_q) * local_scores
        return fused, alpha_q, global_margin, local_margin

    def compute_budget_ranking_loss(
        self,
        rho: torch.Tensor,
        local_scores: torch.Tensor,
        query_targets: torch.Tensor | None,
    ) -> torch.Tensor:
        zero = rho.new_zeros(())
        if query_targets is None or rho.shape[1] < 2 or self.aeb_v2_budget_rank_weight <= 0.0:
            return zero

        true_rho = rho.gather(dim=1, index=query_targets.unsqueeze(1)).squeeze(1)
        mask = F.one_hot(query_targets, num_classes=rho.shape[1]).bool()
        hard_negative_index = local_scores.masked_fill(mask, float("-inf")).argmax(dim=1, keepdim=True)
        hard_negative_rho = rho.gather(dim=1, index=hard_negative_index).squeeze(1)
        budget_gap = hard_negative_rho - true_rho
        return F.softplus(self.aeb_v2_budget_rank_margin - budget_gap).mean()

    def compute_local_margin_loss(
        self,
        local_scores: torch.Tensor,
        query_targets: torch.Tensor | None,
    ) -> torch.Tensor:
        zero = local_scores.new_zeros(())
        if query_targets is None or local_scores.shape[1] < 2 or self.aeb_v2_local_margin_weight <= 0.0:
            return zero

        true_scores = local_scores.gather(dim=1, index=query_targets.unsqueeze(1)).squeeze(1)
        mask = F.one_hot(query_targets, num_classes=local_scores.shape[1]).bool()
        hard_negative_scores = local_scores.masked_fill(mask, float("-inf")).max(dim=1).values
        margin = true_scores - hard_negative_scores
        return F.softplus(self.aeb_v2_local_margin_target - margin).mean()

    def compute_branch_aux_loss(
        self,
        global_scores: torch.Tensor,
        local_scores: torch.Tensor,
        anchor_local_scores: torch.Tensor,
        controller_scores: torch.Tensor,
        rho: torch.Tensor,
        rho_residual: torch.Tensor,
        query_targets: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        zero = global_scores.new_zeros(())
        if query_targets is None:
            return zero, zero, zero, zero, zero, zero, zero

        global_branch_ce = F.cross_entropy(global_scores, query_targets)
        local_branch_ce = F.cross_entropy(local_scores, query_targets)
        budget_rank_loss = self.compute_budget_ranking_loss(
            rho=rho,
            local_scores=controller_scores,
            query_targets=query_targets,
        )
        local_margin_loss = self.compute_local_margin_loss(local_scores=local_scores, query_targets=query_targets)
        budget_residual_reg_loss = rho_residual.pow(2).mean()
        anchor_consistency_loss = F.smooth_l1_loss(controller_scores, anchor_local_scores.detach())
        if not self.training:
            return (
                zero,
                global_branch_ce.detach(),
                local_branch_ce.detach(),
                budget_rank_loss.detach(),
                local_margin_loss.detach(),
                budget_residual_reg_loss.detach(),
                anchor_consistency_loss.detach(),
            )

        aux_loss = zero
        if not self.local_only and self.aeb_v2_global_ce_weight > 0.0:
            aux_loss = aux_loss + self.aeb_v2_global_ce_weight * global_branch_ce
        if not self.global_only and self.aeb_v2_local_ce_weight > 0.0:
            aux_loss = aux_loss + self.aeb_v2_local_ce_weight * local_branch_ce
        if not self.global_only and self.aeb_v2_budget_rank_weight > 0.0:
            aux_loss = aux_loss + self.aeb_v2_budget_rank_weight * budget_rank_loss
        if not self.global_only and self.aeb_v2_local_margin_weight > 0.0:
            aux_loss = aux_loss + self.aeb_v2_local_margin_weight * local_margin_loss
        if not self.global_only and self.aeb_v2_budget_residual_reg_weight > 0.0:
            aux_loss = aux_loss + self.aeb_v2_budget_residual_reg_weight * budget_residual_reg_loss
        if not self.global_only and self.aeb_v2_anchor_consistency_weight > 0.0:
            aux_loss = aux_loss + self.aeb_v2_anchor_consistency_weight * anchor_consistency_loss
        return (
            aux_loss,
            global_branch_ce.detach(),
            local_branch_ce.detach(),
            budget_rank_loss.detach(),
            local_margin_loss.detach(),
            budget_residual_reg_loss.detach(),
            anchor_consistency_loss.detach(),
        )

    def _forward_episode(
        self,
        query: torch.Tensor,
        support: torch.Tensor,
        query_targets: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        episode = self._encode_episode(query, support)
        prototypes = self.build_support_prototypes(episode["support_global"])
        support_token_pool = self.build_support_token_pool(episode["support_local_tokens"])
        support_row_weights = (
            self.build_support_row_weights(episode["support_gate"], eps=self.aeb_v2_eps)
            if self.aeb_v2_query_gate_weighting
            else None
        )
        base_global_scores = self.aeb_v2_global_scale * self.compute_global_scores(episode["query_global"], prototypes)
        conditioned_global_scores, conditioned_global_entropy = self.compute_conditioned_global_scores(
            query_stable_tokens=episode["query_stable_tokens"],
            query_gate=episode["query_gate"],
            prototypes=prototypes,
        )
        conditioned_global_scores = self.aeb_v2_global_scale * conditioned_global_scores
        global_scores = (
            (1.0 - self.aeb_v2_cond_global_weight) * base_global_scores
            + self.aeb_v2_cond_global_weight * conditioned_global_scores
        )
        local_evidence_profile = self.build_local_evidence_profile(
            query_tokens=episode["query_local_tokens"],
            support_token_pool=support_token_pool,
        )
        query_row_weights = None
        query_class_attention_entropy = None
        if self.aeb_v2_query_class_attention_on:
            query_row_weights, query_class_attention_entropy = self.compute_query_class_row_weights(
                local_evidence_profile=local_evidence_profile,
                query_gate=episode["query_gate"],
            )
        elif self.aeb_v2_query_gate_weighting:
            query_row_weights = self.compute_query_row_weights(episode["query_gate"], eps=self.aeb_v2_eps)
        budget_outputs = self.predict_budget(
            local_evidence_profile=local_evidence_profile,
            row_weights=query_row_weights,
        )
        coverage_bonus = budget_outputs["coverage_entropy"] - budget_outputs["coverage_concentration"]
        local_outputs = self.compute_local_budget_scores(
            similarity=local_evidence_profile["similarity"],
            rho=budget_outputs["rho"],
            row_budget=budget_outputs["row_budget"],
            query_row_weights=query_row_weights,
            support_row_weights=support_row_weights if self.aeb_v2_local_score_mode == "bidirectional" else None,
            coverage_bonus=coverage_bonus,
        )
        anchor_local_scores = self.compute_anchor_local_scores(
            similarity=local_evidence_profile["similarity"],
            query_row_weights=query_row_weights,
            support_row_weights=support_row_weights if self.aeb_v2_local_score_mode == "bidirectional" else None,
        )
        local_anchor_mix, local_adaptive_mix = self.compute_local_mix(budget_outputs["rho_prior"])
        local_scores = local_anchor_mix * anchor_local_scores + local_adaptive_mix * local_outputs["adaptive_local_scores"]
        logits, alpha_q, global_margin, local_margin = self.fuse_scores(global_scores, local_scores)
        (
            aux_loss,
            global_branch_ce,
            local_branch_ce,
            budget_rank_loss,
            local_margin_loss,
            budget_residual_reg_loss,
            anchor_consistency_loss,
        ) = self.compute_branch_aux_loss(
            global_scores=global_scores,
            local_scores=local_scores,
            anchor_local_scores=anchor_local_scores,
            controller_scores=local_outputs["adaptive_local_scores"],
            rho=budget_outputs["rho"],
            rho_residual=budget_outputs["rho_residual"],
            query_targets=query_targets,
        )

        mean_gate = torch.cat(
            [
                episode["query_gate"].reshape(-1, 1),
                episode["support_gate"].reshape(-1, 1),
            ],
            dim=0,
        ).mean()
        return {
            "logits": logits,
            "aux_loss": aux_loss,
            "global_scores": global_scores.detach(),
            "base_global_scores": base_global_scores.detach(),
            "conditioned_global_scores": conditioned_global_scores.detach(),
            "local_scores": local_scores.detach(),
            "anchor_local_scores": anchor_local_scores.detach(),
            "adaptive_local_scores": local_outputs["adaptive_local_scores"].detach(),
            "raw_local_scores": local_outputs["raw_local_scores"].detach(),
            "competitive_local_scores": local_outputs["competitive_local_scores"].detach(),
            "rho": local_outputs["rho"].detach(),
            "rho_prior": budget_outputs["rho_prior"].detach(),
            "rho_residual": budget_outputs["rho_residual"].detach(),
            "row_budget_std": budget_outputs["row_budget_std"].detach(),
            "evidence_sharpness": budget_outputs["evidence_sharpness"].detach(),
            "evidence_advantage": budget_outputs["evidence_advantage"].detach(),
            "evidence_dispersion": budget_outputs["evidence_dispersion"].detach(),
            "evidence_quality": budget_outputs["evidence_quality"].detach(),
            "coverage_entropy": budget_outputs["coverage_entropy"].detach(),
            "coverage_concentration": budget_outputs["coverage_concentration"].detach(),
            "coverage_bonus": coverage_bonus.detach(),
            "active_match_counts": local_outputs["active_match_counts"].detach(),
            "retained_fraction": local_outputs["retained_fraction"].detach(),
            "alpha": alpha_q.mean().detach(),
            "alpha_q": alpha_q.detach(),
            "global_margin": global_margin.detach(),
            "local_margin": local_margin.detach(),
            "mean_budget": local_outputs["rho"].mean().detach(),
            "mean_conditioned_global_entropy": conditioned_global_entropy.mean().detach(),
            "mean_query_class_attention_entropy": (
                query_class_attention_entropy.mean().detach()
                if query_class_attention_entropy is not None
                else logits.new_zeros(())
            ),
            "mean_gate": mean_gate.detach(),
            "global_branch_ce": global_branch_ce,
            "local_branch_ce": local_branch_ce,
            "budget_rank_loss": budget_rank_loss,
            "local_margin_loss": local_margin_loss,
            "budget_residual_reg_loss": budget_residual_reg_loss,
            "anchor_consistency_loss": anchor_consistency_loss,
            "local_anchor_mix": local_anchor_mix.detach(),
            "local_adaptive_mix": local_adaptive_mix.detach(),
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

    def forward(
        self,
        query: torch.Tensor,
        support: torch.Tensor,
        query_targets: torch.Tensor | None = None,
        support_targets: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> torch.Tensor | Dict[str, torch.Tensor]:
        del support_targets  # reserved for future extensions without changing the trainer API.
        bsz, _, _, _, _, _ = self.validate_episode_inputs(query, support)
        batch_outputs = []
        aux_losses = []
        diagnostics = []
        query_targets = None if query_targets is None else query_targets.view(-1)

        query_offset = 0
        for batch_idx in range(bsz):
            batch_query = query[batch_idx]
            batch_targets = None
            if query_targets is not None:
                num_query = batch_query.shape[0]
                batch_targets = query_targets[query_offset : query_offset + num_query]
                query_offset += num_query
            episode = self._forward_episode(batch_query, support[batch_idx], query_targets=batch_targets)
            batch_outputs.append(episode["logits"])
            aux_losses.append(episode["aux_loss"])
            diagnostics.append(episode)

        logits = torch.cat(batch_outputs, dim=0)
        aux_loss = torch.stack(aux_losses).mean() if aux_losses else logits.new_zeros(())
        if not return_aux:
            if self.training:
                return {
                    "logits": logits,
                    "aux_loss": aux_loss,
                    "budget_rank_loss": torch.stack([item["budget_rank_loss"] for item in diagnostics]).mean(),
                    "local_margin_loss": torch.stack([item["local_margin_loss"] for item in diagnostics]).mean(),
                    "budget_residual_reg_loss": torch.stack(
                        [item["budget_residual_reg_loss"] for item in diagnostics]
                    ).mean(),
                    "anchor_consistency_loss": torch.stack(
                        [item["anchor_consistency_loss"] for item in diagnostics]
                    ).mean(),
                    "global_branch_ce": torch.stack([item["global_branch_ce"] for item in diagnostics]).mean(),
                    "local_branch_ce": torch.stack([item["local_branch_ce"] for item in diagnostics]).mean(),
                }
            return logits

        payload = {
            "logits": logits,
            "aux_loss": aux_loss,
            "global_scores": torch.cat([item["global_scores"] for item in diagnostics], dim=0),
            "base_global_scores": torch.cat([item["base_global_scores"] for item in diagnostics], dim=0),
            "conditioned_global_scores": torch.cat([item["conditioned_global_scores"] for item in diagnostics], dim=0),
            "local_scores": torch.cat([item["local_scores"] for item in diagnostics], dim=0),
            "anchor_local_scores": torch.cat([item["anchor_local_scores"] for item in diagnostics], dim=0),
            "adaptive_local_scores": torch.cat([item["adaptive_local_scores"] for item in diagnostics], dim=0),
            "raw_local_scores": torch.cat([item["raw_local_scores"] for item in diagnostics], dim=0),
            "competitive_local_scores": torch.cat([item["competitive_local_scores"] for item in diagnostics], dim=0),
            "rho": torch.cat([item["rho"] for item in diagnostics], dim=0),
            "rho_prior": torch.cat([item["rho_prior"] for item in diagnostics], dim=0),
            "rho_residual": torch.cat([item["rho_residual"] for item in diagnostics], dim=0),
            "row_budget_std": torch.cat([item["row_budget_std"] for item in diagnostics], dim=0),
            "evidence_sharpness": torch.cat([item["evidence_sharpness"] for item in diagnostics], dim=0),
            "evidence_advantage": torch.cat([item["evidence_advantage"] for item in diagnostics], dim=0),
            "evidence_dispersion": torch.cat([item["evidence_dispersion"] for item in diagnostics], dim=0),
            "evidence_quality": torch.cat([item["evidence_quality"] for item in diagnostics], dim=0),
            "coverage_entropy": torch.cat([item["coverage_entropy"] for item in diagnostics], dim=0),
            "coverage_concentration": torch.cat([item["coverage_concentration"] for item in diagnostics], dim=0),
            "coverage_bonus": torch.cat([item["coverage_bonus"] for item in diagnostics], dim=0),
            "active_match_counts": torch.cat([item["active_match_counts"] for item in diagnostics], dim=0),
            "retained_fraction": torch.cat([item["retained_fraction"] for item in diagnostics], dim=0),
            "alpha": torch.stack([item["alpha"] for item in diagnostics]).mean(),
            "alpha_q": torch.cat([item["alpha_q"] for item in diagnostics], dim=0),
            "global_margin": torch.cat([item["global_margin"] for item in diagnostics], dim=0),
            "local_margin": torch.cat([item["local_margin"] for item in diagnostics], dim=0),
            "mean_budget": torch.stack([item["mean_budget"] for item in diagnostics]).mean(),
            "mean_conditioned_global_entropy": torch.stack(
                [item["mean_conditioned_global_entropy"] for item in diagnostics]
            ).mean(),
            "mean_query_class_attention_entropy": torch.stack(
                [item["mean_query_class_attention_entropy"] for item in diagnostics]
            ).mean(),
            "mean_gate": torch.stack([item["mean_gate"] for item in diagnostics]).mean(),
            "global_branch_ce": torch.stack([item["global_branch_ce"] for item in diagnostics]).mean(),
            "local_branch_ce": torch.stack([item["local_branch_ce"] for item in diagnostics]).mean(),
            "budget_rank_loss": torch.stack([item["budget_rank_loss"] for item in diagnostics]).mean(),
            "local_margin_loss": torch.stack([item["local_margin_loss"] for item in diagnostics]).mean(),
            "budget_residual_reg_loss": torch.stack(
                [item["budget_residual_reg_loss"] for item in diagnostics]
            ).mean(),
            "anchor_consistency_loss": torch.stack(
                [item["anchor_consistency_loss"] for item in diagnostics]
            ).mean(),
            "local_anchor_mix": torch.cat([item["local_anchor_mix"] for item in diagnostics], dim=0).mean(),
            "local_adaptive_mix": torch.cat([item["local_adaptive_mix"] for item in diagnostics], dim=0).mean(),
            "stable_global_embeddings": torch.cat(
                [item["stable_global_embeddings"] for item in diagnostics],
                dim=0,
            ),
            "variant_global_embeddings": torch.cat(
                [item["variant_global_embeddings"] for item in diagnostics],
                dim=0,
            ),
        }
        return payload
