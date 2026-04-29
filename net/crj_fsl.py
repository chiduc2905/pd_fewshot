"""Consensus-robust J-FSL.

CRJ-FSL keeps the J-FSL transport path unchanged and replaces optimistic
log-mean-exp shot pooling with a lower-confidence support-shot aggregation.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from net.hrot_fsl import HROTFSL, HROTFSLResult


class CRJFSL(HROTFSL):
    """J-FSL with consensus-robust shot aggregation.

    The base J-FSL path computes one threshold-calibrated UOT score per
    query-class-shot pair. CRJ-FSL keeps those shot scores intact and aggregates
    them using a variance-penalized lower confidence score. This suppresses
    class decisions that are supported by only one unusually high shot score.
    """

    _EPISODE_CAT_KEYS = (
        "crj_vanilla_logits",
        "crj_robust_center",
        "crj_shot_score_std",
        "crj_robust_lcb",
        "crj_logit_delta",
        "crj_consensus_confidence",
        "crj_effective_shots",
        "crj_vanilla_pool_weights",
        "crj_robust_pool_weights",
    )
    _EPISODE_SCALAR_KEYS = (
        "crj_pool_gamma",
        "crj_config_pool_gamma",
        "crj_effective_gamma",
        "crj_active",
        "crj_shot_num",
        "crj_variance_penalty",
        "crj_min_shots",
        "crj_trim_fraction",
        "crj_clip_logit",
    )

    def __init__(
        self,
        *args: Any,
        crj_pool_gamma: float = 0.65,
        crj_variance_penalty: float = 0.25,
        crj_min_shots: int = 2,
        crj_trim_fraction: float = 0.0,
        crj_clip_logit: float = 0.0,
        **kwargs: Any,
    ) -> None:
        kwargs["variant"] = "J"
        super().__init__(*args, **kwargs)
        if not 0.0 <= crj_pool_gamma <= 1.0:
            raise ValueError("crj_pool_gamma must be in [0, 1]")
        if crj_variance_penalty < 0.0:
            raise ValueError("crj_variance_penalty must be non-negative")
        if crj_min_shots <= 0:
            raise ValueError("crj_min_shots must be positive")
        if not 0.0 <= crj_trim_fraction < 0.5:
            raise ValueError("crj_trim_fraction must be in [0, 0.5)")
        if crj_clip_logit < 0.0:
            raise ValueError("crj_clip_logit must be non-negative")
        self.crj_pool_gamma = float(crj_pool_gamma)
        self.crj_variance_penalty = float(crj_variance_penalty)
        self.crj_min_shots = int(crj_min_shots)
        self.crj_trim_fraction = float(crj_trim_fraction)
        self.crj_clip_logit = float(crj_clip_logit)
        self._last_crj_pool_diagnostics: dict[str, torch.Tensor] | None = None

    def _effective_gamma(self, shot_num: int, reference: torch.Tensor) -> torch.Tensor:
        if shot_num < self.crj_min_shots:
            value = 0.0
        else:
            value = self.crj_pool_gamma * float(shot_num - 1) / float(max(shot_num, 1))
        return reference.new_tensor(value)

    def _prepare_robust_shot_values(
        self,
        shot_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        values = shot_logits
        if self.crj_clip_logit > 0.0:
            clip = float(self.crj_clip_logit)
            values = values.clamp(min=-clip, max=clip)

        shot_num = values.shape[-1]
        trim_each = int(math.floor(float(shot_num) * self.crj_trim_fraction))
        max_trim = max(0, (shot_num - 1) // 2)
        trim_each = min(trim_each, max_trim)

        if trim_each <= 0:
            weights = torch.full_like(values, 1.0 / float(shot_num))
            return values, weights

        sorted_values, sorted_indices = torch.sort(values, dim=-1)
        kept_values = sorted_values[..., trim_each : shot_num - trim_each]
        keep_count = kept_values.shape[-1]
        sorted_weights = torch.zeros_like(values)
        sorted_weights[..., trim_each : shot_num - trim_each] = 1.0 / float(keep_count)
        weights = torch.zeros_like(values).scatter(dim=-1, index=sorted_indices, src=sorted_weights)
        return kept_values, weights

    def _pool_j_shot_scores(
        self,
        shot_logits: torch.Tensor,
        shot_transport_cost: torch.Tensor,
        shot_transport_mass: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        vanilla_logits, _, _, vanilla_weights = HROTFSL._pool_j_shot_scores(
            self,
            shot_logits,
            shot_transport_cost,
            shot_transport_mass,
        )
        shot_num = shot_logits.shape[-1]
        gamma = self._effective_gamma(shot_num, shot_logits)

        robust_values, robust_weights = self._prepare_robust_shot_values(shot_logits)
        robust_center = robust_values.mean(dim=-1)
        robust_std = robust_values.std(dim=-1, unbiased=False)
        robust_lcb = robust_center - float(self.crj_variance_penalty) * robust_std
        logits = (1.0 - gamma) * vanilla_logits + gamma * robust_lcb

        mixed_weights = (1.0 - gamma) * vanilla_weights + gamma * robust_weights
        mixed_weights = mixed_weights / mixed_weights.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        transport_cost = (mixed_weights * shot_transport_cost).sum(dim=-1)
        transport_mass = (mixed_weights * shot_transport_mass).sum(dim=-1)

        effective_shots = 1.0 / mixed_weights.pow(2).sum(dim=-1).clamp_min(self.eps)
        diagnostics = {
            "crj_vanilla_logits": vanilla_logits,
            "crj_robust_center": robust_center,
            "crj_shot_score_std": robust_std,
            "crj_robust_lcb": robust_lcb,
            "crj_logit_delta": logits - vanilla_logits,
            "crj_consensus_confidence": 1.0 / (1.0 + robust_std),
            "crj_effective_shots": effective_shots,
            "crj_vanilla_pool_weights": vanilla_weights,
            "crj_robust_pool_weights": robust_weights,
            "crj_pool_gamma": gamma.detach(),
            "crj_config_pool_gamma": shot_logits.new_tensor(self.crj_pool_gamma),
            "crj_effective_gamma": gamma.detach(),
            "crj_active": (gamma > 0.0).to(dtype=shot_logits.dtype),
            "crj_shot_num": shot_logits.new_tensor(float(shot_num)),
            "crj_variance_penalty": shot_logits.new_tensor(self.crj_variance_penalty),
            "crj_min_shots": shot_logits.new_tensor(float(self.crj_min_shots)),
            "crj_trim_fraction": shot_logits.new_tensor(self.crj_trim_fraction),
            "crj_clip_logit": shot_logits.new_tensor(self.crj_clip_logit),
        }
        self._last_crj_pool_diagnostics = diagnostics
        return logits, transport_cost, transport_mass, mixed_weights

    def _forward_episode(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        self._last_crj_pool_diagnostics = None
        outputs = super()._forward_episode(*args, **kwargs)
        if isinstance(outputs, dict) and self._last_crj_pool_diagnostics is not None:
            outputs.update(self._last_crj_pool_diagnostics)
            outputs["raw_logits"] = self._last_crj_pool_diagnostics["crj_vanilla_logits"]
            outputs["local_scores"] = outputs["logits"]
            outputs["shot_aggregation_weights"] = outputs["shot_pool_weights"]
            outputs["mean_crj_logit_delta"] = self._last_crj_pool_diagnostics["crj_logit_delta"].mean()
            outputs["mean_crj_shot_std"] = self._last_crj_pool_diagnostics["crj_shot_score_std"].mean()
            outputs["mean_crj_effective_shots"] = self._last_crj_pool_diagnostics["crj_effective_shots"].mean()
            weights = outputs["shot_pool_weights"].clamp_min(self.eps)
            outputs["mean_crj_pool_entropy"] = (-(weights * weights.log()).sum(dim=-1)).mean()
        return outputs

    @staticmethod
    def _stack_outputs(batch_outputs: list[dict[str, torch.Tensor]]) -> HROTFSLResult:
        stacked = HROTFSL._stack_outputs(batch_outputs)
        for key in CRJFSL._EPISODE_CAT_KEYS:
            if key in batch_outputs[0]:
                stacked[key] = torch.cat([item[key] for item in batch_outputs], dim=0)
        for key in CRJFSL._EPISODE_SCALAR_KEYS:
            if key in batch_outputs[0]:
                stacked[key] = torch.stack([item[key] for item in batch_outputs]).mean()
        for key in (
            "raw_logits",
            "local_scores",
            "shot_aggregation_weights",
        ):
            if key in batch_outputs[0]:
                stacked[key] = torch.cat([item[key] for item in batch_outputs], dim=0)
        for key in (
            "mean_crj_logit_delta",
            "mean_crj_shot_std",
            "mean_crj_effective_shots",
            "mean_crj_pool_entropy",
        ):
            if key in batch_outputs[0]:
                stacked[key] = torch.stack([item[key] for item in batch_outputs]).mean()
        return HROTFSLResult(stacked)
