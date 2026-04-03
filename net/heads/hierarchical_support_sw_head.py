"""Hierarchical Support Sliced-Wasserstein head."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class HSSWSupportState:
    """Support-only cached quantities for one episodic support set."""

    shot_tokens: torch.Tensor
    shot_transport_weights: torch.Tensor
    redundancy_penalty: torch.Tensor
    support_pairwise_sw: torch.Tensor


class HierarchicalSupportSlicedWassersteinHead(nn.Module):
    """HS-SW few-shot classifier head with explicit support-state caching."""

    def __init__(
        self,
        sw_distance: nn.Module,
        tau: float = 0.2,
        lambda_cons: float = 0.1,
        lambda_red: float = 0.0,
        weighted_sw: bool = True,
        learn_logit_scale: bool = False,
        logit_scale_init: float = 10.0,
        logit_scale_max: float | None = 100.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if tau <= 0.0:
            raise ValueError("tau must be positive")
        if logit_scale_init <= 0.0:
            raise ValueError("logit_scale_init must be positive")
        if logit_scale_max is not None and logit_scale_max <= 0.0:
            raise ValueError("logit_scale_max must be positive when provided")
        if eps <= 0.0:
            raise ValueError("eps must be positive")

        self.sw_distance = sw_distance
        self.tau = float(tau)
        self.lambda_cons = float(lambda_cons)
        self.lambda_red = float(lambda_red)
        self.weighted_sw = bool(weighted_sw)
        self.learn_logit_scale = bool(learn_logit_scale)
        self.logit_scale_max = None if logit_scale_max is None else float(logit_scale_max)
        self.eps = float(eps)
        self._offdiag_index_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        if self.learn_logit_scale:
            self.logit_scale_raw = nn.Parameter(torch.tensor(self._softplus_inverse(float(logit_scale_init))))
        else:
            self.register_buffer("_fixed_logit_scale", torch.ones(()), persistent=False)

    @staticmethod
    def _softplus_inverse(value: float) -> float:
        return float(math.log(math.expm1(value)))

    def get_logit_scale(self) -> torch.Tensor:
        if not self.learn_logit_scale:
            return self._fixed_logit_scale
        scale = F.softplus(self.logit_scale_raw) + self.eps
        if self.logit_scale_max is not None:
            scale = scale.clamp_max(self.logit_scale_max)
        return scale

    def _normalize_transport_weights(self, weights: torch.Tensor) -> torch.Tensor:
        weights = weights.clamp_min(0.0)
        denom = weights.sum(dim=-1, keepdim=True)
        zero_mass = denom <= self.eps
        if zero_mass.any():
            weights = torch.where(zero_mass, torch.ones_like(weights), weights)
            denom = weights.sum(dim=-1, keepdim=True)
        return weights / denom.clamp_min(self.eps)

    def _resolve_transport_weights(self, weights: torch.Tensor) -> torch.Tensor:
        if self.weighted_sw:
            return self._normalize_transport_weights(weights)
        return torch.full_like(weights, 1.0 / float(weights.shape[-1]))

    def _validate_shot_measure_inputs(
        self,
        shot_tokens: torch.Tensor,
        shot_weights: torch.Tensor,
    ) -> tuple[int, int, int, int]:
        if shot_tokens.dim() != 4:
            raise ValueError(
                "shot_tokens must have shape (Way, Shot, Tokens, Dim), "
                f"got {tuple(shot_tokens.shape)}"
            )
        if shot_weights.shape != shot_tokens.shape[:-1]:
            raise ValueError(
                "shot_weights must have shape (Way, Shot, Tokens), "
                f"got {tuple(shot_weights.shape)} for tokens {tuple(shot_tokens.shape)}"
            )
        way_num, shot_num, token_num, dim = shot_tokens.shape
        return int(way_num), int(shot_num), int(token_num), int(dim)

    def _validate_query_inputs(
        self,
        query_tokens: torch.Tensor,
        query_weights: torch.Tensor,
        support_state: HSSWSupportState,
    ) -> tuple[int, int, int]:
        if query_tokens.dim() != 3:
            raise ValueError(
                "query_tokens must have shape (NumQuery, Tokens, Dim), "
                f"got {tuple(query_tokens.shape)}"
            )
        if query_weights.shape != query_tokens.shape[:-1]:
            raise ValueError(
                "query_weights must have shape (NumQuery, Tokens), "
                f"got {tuple(query_weights.shape)} for tokens {tuple(query_tokens.shape)}"
            )
        if query_tokens.shape[-1] != support_state.shot_tokens.shape[-1]:
            raise ValueError(
                "Query/support token feature dimensions must match: "
                f"query={query_tokens.shape[-1]} support={support_state.shot_tokens.shape[-1]}"
            )
        return int(query_tokens.shape[0]), int(query_tokens.shape[1]), int(query_tokens.shape[2])

    def _get_offdiag_indices(self, shot_num: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        cached = self._offdiag_index_cache.get(int(shot_num))
        if cached is None:
            left = []
            right = []
            for left_idx in range(shot_num):
                for right_idx in range(shot_num):
                    if left_idx == right_idx:
                        continue
                    left.append(left_idx)
                    right.append(right_idx)
            cached = (
                torch.tensor(left, dtype=torch.long),
                torch.tensor(right, dtype=torch.long),
            )
            self._offdiag_index_cache[int(shot_num)] = cached
        return cached[0].to(device=device), cached[1].to(device=device)

    def compute_shot_responsibilities(self, shot_distances: torch.Tensor) -> torch.Tensor:
        if shot_distances.dim() != 3:
            raise ValueError(
                "shot_distances must have shape (NumQuery, Way, Shot), "
                f"got {tuple(shot_distances.shape)}"
            )
        return torch.softmax(-shot_distances / self.tau, dim=-1)

    def compute_base_distance(
        self,
        shot_distances: torch.Tensor,
        shot_responsibilities: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if shot_distances.dim() != 3:
            raise ValueError(
                "shot_distances must have shape (NumQuery, Way, Shot), "
                f"got {tuple(shot_distances.shape)}"
            )
        if shot_responsibilities is None:
            shot_responsibilities = self.compute_shot_responsibilities(shot_distances)
        if shot_responsibilities.shape != shot_distances.shape:
            raise ValueError(
                "shot_responsibilities must match shot_distances shape: "
                f"dist={tuple(shot_distances.shape)} resp={tuple(shot_responsibilities.shape)}"
            )
        return torch.sum(shot_responsibilities * shot_distances, dim=-1)

    def compute_consistency_penalty(self, shot_responsibilities: torch.Tensor) -> torch.Tensor:
        if shot_responsibilities.dim() != 3:
            raise ValueError(
                "shot_responsibilities must have shape (NumQuery, Way, Shot), "
                f"got {tuple(shot_responsibilities.shape)}"
            )
        return -torch.log(shot_responsibilities + self.eps).sum(dim=-1)

    def compute_gamma_entropy(self, shot_responsibilities: torch.Tensor) -> torch.Tensor:
        if shot_responsibilities.dim() != 3:
            raise ValueError(
                "shot_responsibilities must have shape (NumQuery, Way, Shot), "
                f"got {tuple(shot_responsibilities.shape)}"
            )
        return -(shot_responsibilities * torch.log(shot_responsibilities + self.eps)).sum(dim=-1)

    def build_support_state(
        self,
        shot_tokens: torch.Tensor,
        shot_weights: torch.Tensor,
    ) -> HSSWSupportState:
        way_num, shot_num, token_num, dim = self._validate_shot_measure_inputs(shot_tokens, shot_weights)
        transport_weights = self._resolve_transport_weights(shot_weights)

        if self.lambda_red == 0.0 or shot_num <= 1:
            return HSSWSupportState(
                shot_tokens=shot_tokens,
                shot_transport_weights=transport_weights,
                redundancy_penalty=shot_tokens.new_zeros(way_num),
                support_pairwise_sw=shot_tokens.new_zeros(way_num, shot_num, shot_num),
            )

        left_index, right_index = self._get_offdiag_indices(shot_num, device=shot_tokens.device)
        support_left = shot_tokens.index_select(dim=1, index=left_index).reshape(-1, token_num, dim)
        support_right = shot_tokens.index_select(dim=1, index=right_index).reshape(-1, token_num, dim)
        weight_left = transport_weights.index_select(dim=1, index=left_index).reshape(-1, token_num)
        weight_right = transport_weights.index_select(dim=1, index=right_index).reshape(-1, token_num)

        offdiag_sw = self.sw_distance(
            support_left,
            support_right,
            x_weights=weight_left,
            y_weights=weight_right,
            reduction="none",
        ).reshape(way_num, -1)
        support_pairwise_sw = shot_tokens.new_zeros(way_num, shot_num, shot_num)
        support_pairwise_sw[:, left_index, right_index] = offdiag_sw
        redundancy_penalty = torch.exp(-offdiag_sw).mean(dim=-1)

        return HSSWSupportState(
            shot_tokens=shot_tokens,
            shot_transport_weights=transport_weights,
            redundancy_penalty=redundancy_penalty,
            support_pairwise_sw=support_pairwise_sw,
        )

    def compute_query_to_shot_distances(
        self,
        query_tokens: torch.Tensor,
        query_weights: torch.Tensor,
        support_state: HSSWSupportState,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_query, _, _ = self._validate_query_inputs(query_tokens, query_weights, support_state)
        way_num, shot_num, token_num, dim = support_state.shot_tokens.shape
        query_transport_weights = self._resolve_transport_weights(query_weights)
        support_tokens_flat = support_state.shot_tokens.reshape(way_num * shot_num, token_num, dim)
        support_weights_flat = support_state.shot_transport_weights.reshape(way_num * shot_num, token_num)
        shot_distances = self.sw_distance.pairwise_distance(
            query_tokens=query_tokens,
            support_tokens=support_tokens_flat,
            query_weights=query_transport_weights,
            support_weights=support_weights_flat,
            reduction="none",
        ).reshape(num_query, way_num, shot_num)
        return shot_distances, query_transport_weights

    def forward(
        self,
        query_tokens: torch.Tensor,
        query_weights: torch.Tensor,
        support_state: HSSWSupportState,
        return_aux: bool = False,
    ) -> torch.Tensor | dict[str, Any]:
        shot_distances, query_transport_weights = self.compute_query_to_shot_distances(
            query_tokens=query_tokens,
            query_weights=query_weights,
            support_state=support_state,
        )
        shot_responsibilities = self.compute_shot_responsibilities(shot_distances)
        base_distance = self.compute_base_distance(
            shot_distances=shot_distances,
            shot_responsibilities=shot_responsibilities,
        )
        consistency_penalty = self.compute_consistency_penalty(shot_responsibilities)
        total_distance = (
            base_distance
            + self.lambda_cons * consistency_penalty
            + self.lambda_red * support_state.redundancy_penalty.unsqueeze(0)
        )
        logit_scale = self.get_logit_scale()
        logits = -logit_scale * total_distance

        if not return_aux:
            return logits

        gamma_entropy = self.compute_gamma_entropy(shot_responsibilities)
        return {
            "logits": logits,
            "shot_distances": shot_distances,
            "shot_responsibilities": shot_responsibilities,
            "base_distance": base_distance,
            "consistency_penalty": consistency_penalty,
            "redundancy_penalty": support_state.redundancy_penalty,
            "support_pairwise_sw": support_state.support_pairwise_sw,
            "total_distance": total_distance,
            "query_transport_weights": query_transport_weights,
            "support_transport_weights": support_state.shot_transport_weights,
            "gamma_entropy": gamma_entropy,
            "logit_scale": logit_scale,
            "mean_shot_distance": shot_distances.mean(),
            "mean_gamma_entropy": gamma_entropy.mean(),
            "mean_consistency_penalty": consistency_penalty.mean(),
            "mean_redundancy_penalty": support_state.redundancy_penalty.mean(),
        }
