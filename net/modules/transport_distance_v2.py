"""Strong weighted transport distances for SC-LFI v2."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from net.metrics.sliced_wasserstein_weighted import WeightedPaperSlicedWassersteinDistance


def normalize_measure_masses(
    masses: torch.Tensor | None,
    *,
    target_shape: torch.Size,
    device: torch.device,
    dtype: torch.dtype,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Normalize possibly-broadcastable masses into valid probability weights."""

    if eps <= 0.0:
        raise ValueError("eps must be positive")

    if masses is None:
        normalized = torch.ones(target_shape, device=device, dtype=dtype)
    else:
        normalized = torch.as_tensor(masses, device=device, dtype=dtype)
        try:
            normalized = torch.broadcast_to(normalized, target_shape)
        except RuntimeError as exc:
            raise ValueError(
                f"masses must be broadcastable to {tuple(target_shape)}, got {tuple(normalized.shape)}"
            ) from exc

    normalized = normalized.clamp_min(0.0)
    row_sum = normalized.sum(dim=-1, keepdim=True)
    zero_rows = row_sum <= eps
    if zero_rows.any():
        uniform = torch.full_like(normalized, 1.0 / float(target_shape[-1]))
        normalized = torch.where(zero_rows, uniform, normalized)
        row_sum = normalized.sum(dim=-1, keepdim=True)
    return normalized / row_sum.clamp_min(eps)


class WeightedEntropicOTDistanceV2(nn.Module):
    """Log-domain weighted Sinkhorn transport cost.

    Formula:
    - cost matrix: `C_ij = ||x_i - y_j||_2^p`
    - transport objective:
      `min_P <P, C> + eps * KL(P || a \\otimes b)`

    Paper grounding:
    - entropic OT as a smooth conditional distribution discrepancy is aligned
      with the GENTLE perspective.

    Engineering approximation:
    - this is a fixed-iteration log-domain Sinkhorn solver, not an exact OT
      solver.
    """

    def __init__(
        self,
        sinkhorn_epsilon: float = 0.05,
        max_iterations: int = 80,
        cost_power: float = 2.0,
        normalize_inputs: bool = True,
        reduction: str = "mean",
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if sinkhorn_epsilon <= 0.0:
            raise ValueError("sinkhorn_epsilon must be positive")
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if cost_power <= 0.0:
            raise ValueError("cost_power must be positive")
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(f"Unsupported reduction: {reduction}")
        if eps <= 0.0:
            raise ValueError("eps must be positive")
        self.sinkhorn_epsilon = float(sinkhorn_epsilon)
        self.max_iterations = int(max_iterations)
        self.cost_power = float(cost_power)
        self.normalize_inputs = bool(normalize_inputs)
        self.reduction = reduction
        self.eps = float(eps)

    def _prepare_inputs(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        source_masses: torch.Tensor | None,
        target_masses: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if source.dim() < 3 or target.dim() < 3:
            raise ValueError(
                "source and target must have at least 3 dims: "
                f"source={tuple(source.shape)} target={tuple(target.shape)}"
            )
        if source.shape[:-2] != target.shape[:-2]:
            raise ValueError(
                "source and target leading dimensions must match: "
                f"source={tuple(source.shape)} target={tuple(target.shape)}"
            )
        if source.shape[-1] != target.shape[-1]:
            raise ValueError(
                "source and target latent dims must match: "
                f"source={source.shape[-1]} target={target.shape[-1]}"
            )
        if self.normalize_inputs:
            source = F.normalize(source, p=2, dim=-1)
            target = F.normalize(target, p=2, dim=-1)
        source_masses = normalize_measure_masses(
            source_masses,
            target_shape=source.shape[:-1],
            device=source.device,
            dtype=source.dtype,
            eps=self.eps,
        )
        target_masses = normalize_measure_masses(
            target_masses,
            target_shape=target.shape[:-1],
            device=target.device,
            dtype=target.dtype,
            eps=self.eps,
        )
        return source, target, source_masses, target_masses

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        *,
        source_masses: torch.Tensor | None = None,
        target_masses: torch.Tensor | None = None,
        reduction: str | None = None,
    ) -> torch.Tensor:
        reduction = reduction or self.reduction
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(f"Unsupported reduction: {reduction}")

        source, target, source_masses, target_masses = self._prepare_inputs(
            source,
            target,
            source_masses,
            target_masses,
        )
        cost_matrix = torch.cdist(source, target, p=2).pow(self.cost_power)
        log_a = torch.log(source_masses.clamp_min(self.eps))
        log_b = torch.log(target_masses.clamp_min(self.eps))
        log_kernel = -cost_matrix / self.sinkhorn_epsilon

        log_u = torch.zeros_like(log_a)
        log_v = torch.zeros_like(log_b)
        for _ in range(self.max_iterations):
            log_u = log_a - torch.logsumexp(log_kernel + log_v.unsqueeze(-2), dim=-1)
            log_v = log_b - torch.logsumexp(log_kernel + log_u.unsqueeze(-1), dim=-2)

        log_plan = log_kernel + log_u.unsqueeze(-1) + log_v.unsqueeze(-2)
        plan = torch.exp(log_plan)
        distances = (plan * cost_matrix).sum(dim=(-2, -1))

        if reduction == "none":
            return distances
        if reduction == "sum":
            return distances.sum()
        return distances.mean()

    def pairwise_distance(
        self,
        query_tokens: torch.Tensor,
        support_tokens: torch.Tensor,
        *,
        query_masses: torch.Tensor | None = None,
        support_masses: torch.Tensor | None = None,
        reduction: str | None = None,
    ) -> torch.Tensor:
        if query_tokens.dim() != 3 or support_tokens.dim() != 3:
            raise ValueError(
                "pairwise_distance expects query_tokens=(NumQuery, Tokens, Dim) and "
                "support_tokens=(Way, Tokens, Dim)"
            )
        expanded_query = query_tokens.unsqueeze(1)
        expanded_support = support_tokens.unsqueeze(0)
        expanded_query_masses = None if query_masses is None else query_masses.unsqueeze(1)
        expanded_support_masses = None if support_masses is None else support_masses.unsqueeze(0)
        return self(
            expanded_query,
            expanded_support,
            source_masses=expanded_query_masses,
            target_masses=expanded_support_masses,
            reduction=reduction or "none",
        )


class WeightedTransportScoringDistanceV2(nn.Module):
    """Weighted exact paper-style sliced Wasserstein for query-class scoring."""

    def __init__(
        self,
        train_num_projections: int = 64,
        eval_num_projections: int = 128,
        p: float = 2.0,
        normalize_inputs: bool = True,
        train_projection_mode: str = "resample",
        eval_projection_mode: str = "fixed",
        eval_num_repeats: int = 1,
        projection_seed: int = 7,
    ) -> None:
        super().__init__()
        self.metric = WeightedPaperSlicedWassersteinDistance(
            train_num_projections=int(train_num_projections),
            eval_num_projections=int(eval_num_projections),
            p=float(p),
            reduction="mean",
            normalize_inputs=bool(normalize_inputs),
            train_projection_mode=str(train_projection_mode),
            eval_projection_mode=str(eval_projection_mode),
            eval_num_repeats=int(eval_num_repeats),
            projection_seed=int(projection_seed),
        )

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        *,
        source_masses: torch.Tensor | None = None,
        target_masses: torch.Tensor | None = None,
        reduction: str = "none",
    ) -> torch.Tensor:
        return self.metric(
            source,
            target,
            query_weights=source_masses,
            support_weights=target_masses,
            reduction=reduction,
        )

    def pairwise_distance(
        self,
        query_tokens: torch.Tensor,
        support_tokens: torch.Tensor,
        *,
        query_masses: torch.Tensor | None = None,
        support_masses: torch.Tensor | None = None,
        reduction: str = "none",
    ) -> torch.Tensor:
        return self.metric.pairwise_distance(
            query_tokens,
            support_tokens,
            query_weights=query_masses,
            support_weights=support_masses,
            reduction=reduction,
        )


class AlignmentTransportDistanceV2(nn.Module):
    """Alignment distance for anchoring generated class particles to support evidence."""

    def __init__(
        self,
        distance_type: str = "weighted_entropic_ot",
        *,
        sw_train_num_projections: int = 64,
        sw_eval_num_projections: int = 128,
        sw_p: float = 2.0,
        sw_normalize_inputs: bool = True,
        sw_train_projection_mode: str = "resample",
        sw_eval_projection_mode: str = "fixed",
        sw_eval_num_repeats: int = 1,
        sw_projection_seed: int = 7,
        sinkhorn_epsilon: float = 0.05,
        sinkhorn_iterations: int = 80,
        sinkhorn_cost_power: float = 2.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        distance_type = str(distance_type).lower()
        self.distance_type = distance_type
        if distance_type == "weighted_sw":
            self.distance = WeightedTransportScoringDistanceV2(
                train_num_projections=sw_train_num_projections,
                eval_num_projections=sw_eval_num_projections,
                p=sw_p,
                normalize_inputs=sw_normalize_inputs,
                train_projection_mode=sw_train_projection_mode,
                eval_projection_mode=sw_eval_projection_mode,
                eval_num_repeats=sw_eval_num_repeats,
                projection_seed=sw_projection_seed,
            )
        elif distance_type == "weighted_entropic_ot":
            self.distance = WeightedEntropicOTDistanceV2(
                sinkhorn_epsilon=sinkhorn_epsilon,
                max_iterations=sinkhorn_iterations,
                cost_power=sinkhorn_cost_power,
                normalize_inputs=sw_normalize_inputs,
                reduction="mean",
                eps=eps,
            )
        else:
            raise ValueError(f"Unsupported alignment distance type: {distance_type}")

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        *,
        source_masses: torch.Tensor | None = None,
        target_masses: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> torch.Tensor:
        if isinstance(self.distance, WeightedTransportScoringDistanceV2):
            return self.distance(
                source,
                target,
                source_masses=source_masses,
                target_masses=target_masses,
                reduction=reduction,
            )
        return self.distance(
            source,
            target,
            source_masses=source_masses,
            target_masses=target_masses,
            reduction=reduction,
        )
