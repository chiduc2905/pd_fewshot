"""Rival-conditional ground cost for episodic few-shot transport."""

from __future__ import annotations

import math

import torch
from torch import nn


class RivalConditionalGroundCost(nn.Module):
    """Penalize locally cheap matches that are not specific to one class.

    The input remains the original query-to-support ground cost. A soft class
    posterior is computed independently for every query token by comparing its
    soft-min cost to all classes. The negative log posterior is then added to
    every edge of that token/class pair before the unchanged UOT solver.
    """

    def __init__(
        self,
        *,
        penalty_weight: float = 0.50,
        temperature: float = 0.25,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if penalty_weight < 0.0:
            raise ValueError("penalty_weight must be non-negative")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if eps <= 0.0:
            raise ValueError("eps must be positive")
        self.penalty_weight = float(penalty_weight)
        self.temperature = float(temperature)
        self.eps = float(eps)

    def forward(
        self,
        flat_cost: torch.Tensor,
        *,
        way_num: int,
        shot_num: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if flat_cost.dim() != 4:
            raise ValueError(
                "flat_cost must have shape (NumQuery, Way*Shot, QueryTokens, SupportTokens), "
                f"got {tuple(flat_cost.shape)}"
            )
        num_query, num_pairs, query_len, support_len = flat_cost.shape
        way_num = int(way_num)
        shot_num = int(shot_num)
        if way_num <= 1:
            raise ValueError("rival-conditional ground cost requires at least two classes")
        if shot_num <= 0 or num_pairs != way_num * shot_num:
            raise ValueError(
                f"flat_cost pair dimension {num_pairs} does not match way*shot={way_num * shot_num}"
            )

        if self.penalty_weight == 0.0:
            uniform_probability = flat_cost.new_full(
                (num_query, way_num, query_len),
                1.0 / float(way_num),
            )
            zero_penalty = flat_cost.new_zeros(
                num_query,
                way_num,
                query_len,
            )
            return flat_cost, self._payload(
                flat_cost,
                uniform_probability,
                zero_penalty,
                flat_cost.new_zeros(num_query, way_num, query_len),
            )

        cost = flat_cost.reshape(
            num_query,
            way_num,
            shot_num,
            query_len,
            support_len,
        )
        episode_scale = flat_cost.std(
            dim=(1, 2, 3),
            unbiased=False,
            keepdim=True,
        ).clamp_min(self.eps)
        class_temperature = (
            episode_scale.reshape(num_query, 1, 1) * self.temperature
        ).clamp_min(self.eps)

        flattened_support = cost.permute(0, 1, 3, 2, 4).reshape(
            num_query,
            way_num,
            query_len,
            shot_num * support_len,
        )
        support_count = max(shot_num * support_len, 1)
        class_energy = -class_temperature * (
            torch.logsumexp(
                -flattened_support / class_temperature.unsqueeze(-1),
                dim=-1,
            )
            - math.log(float(support_count))
        )
        log_class_probability = torch.log_softmax(
            -class_energy / class_temperature,
            dim=1,
        )
        class_probability = log_class_probability.exp()
        penalty = (
            self.penalty_weight
            * episode_scale.reshape(num_query, 1, 1)
            * (-log_class_probability)
        )
        guided_cost = cost + penalty[:, :, None, :, None]
        guided_cost = guided_cost.reshape_as(flat_cost)
        return guided_cost, self._payload(
            flat_cost,
            class_probability,
            penalty,
            class_energy,
        )

    def _payload(
        self,
        flat_cost: torch.Tensor,
        class_probability: torch.Tensor,
        penalty: torch.Tensor,
        class_energy: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        way_num = int(class_probability.shape[1])
        entropy = -(
            class_probability
            * class_probability.clamp_min(self.eps).log()
        ).sum(dim=1)
        normalized_entropy = entropy / math.log(float(way_num))
        peak = class_probability.amax(dim=1)
        return {
            "rc_cost_class_probability": class_probability.detach(),
            "rc_cost_penalty": penalty.detach(),
            "rc_cost_class_energy": class_energy.detach(),
            "rc_cost/enabled": flat_cost.new_tensor(1.0),
            "rc_cost/penalty_weight": flat_cost.new_tensor(self.penalty_weight),
            "rc_cost/temperature": flat_cost.new_tensor(self.temperature),
            "rc_cost/class_entropy": normalized_entropy.mean().detach(),
            "rc_cost/class_probability_peak": peak.mean().detach(),
            "rc_cost/common_query_share": (
                normalized_entropy > 0.90
            ).to(flat_cost.dtype).mean().detach(),
            "rc_cost/penalty_mean": penalty.mean().detach(),
            "rc_cost/penalty_to_cost_ratio": (
                penalty.mean()
                / flat_cost.detach().abs().mean().clamp_min(self.eps)
            ).detach(),
        }


__all__ = ["RivalConditionalGroundCost"]
