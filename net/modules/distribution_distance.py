"""Distribution distance wrappers for SC-LFI."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from net.metrics.sliced_wasserstein import SlicedWassersteinDistance


def _prepare_distribution(tokens: torch.Tensor, normalize_inputs: bool) -> torch.Tensor:
    if tokens.dim() < 3:
        raise ValueError(f"distribution tensors must have at least 3 dimensions, got {tuple(tokens.shape)}")
    if normalize_inputs:
        tokens = F.normalize(tokens, p=2, dim=-1)
    return tokens


class UniformEntropicOTDistance(nn.Module):
    """Pure-PyTorch Sinkhorn distance with uniform marginals.

    This wrapper is included as an entropic OT ablation hook. The default SC-LFI
    path still uses sliced Wasserstein for simplicity and stability.
    """

    def __init__(
        self,
        sinkhorn_epsilon: float = 0.1,
        max_iterations: int = 50,
        cost_power: float = 2.0,
        reduction: str = "mean",
        normalize_inputs: bool = True,
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
        self.sinkhorn_epsilon = float(sinkhorn_epsilon)
        self.max_iterations = int(max_iterations)
        self.cost_power = float(cost_power)
        self.reduction = reduction
        self.normalize_inputs = bool(normalize_inputs)

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        reduction: str | None = None,
    ) -> torch.Tensor:
        reduction = reduction or self.reduction
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(f"Unsupported reduction: {reduction}")

        source = _prepare_distribution(source, normalize_inputs=self.normalize_inputs)
        target = _prepare_distribution(target, normalize_inputs=self.normalize_inputs)
        if source.shape[:-2] != target.shape[:-2]:
            raise ValueError(
                "source and target leading dimensions must match: "
                f"source={tuple(source.shape)} target={tuple(target.shape)}"
            )
        if source.shape[-1] != target.shape[-1]:
            raise ValueError(
                "source and target latent dimensions must match: "
                f"source={source.shape[-1]} target={target.shape[-1]}"
            )

        cost_matrix = torch.cdist(source, target, p=2).pow(self.cost_power)
        num_source = source.shape[-2]
        num_target = target.shape[-2]

        log_a = cost_matrix.new_full(cost_matrix.shape[:-2] + (num_source,), -math.log(float(num_source)))
        log_b = cost_matrix.new_full(cost_matrix.shape[:-2] + (num_target,), -math.log(float(num_target)))
        log_kernel = -cost_matrix / self.sinkhorn_epsilon

        log_u = torch.zeros_like(log_a)
        log_v = torch.zeros_like(log_b)

        for _ in range(self.max_iterations):
            log_u = log_a - torch.logsumexp(log_kernel + log_v.unsqueeze(-2), dim=-1)
            log_v = log_b - torch.logsumexp(log_kernel + log_u.unsqueeze(-1), dim=-2)

        log_plan = log_kernel + log_u.unsqueeze(-1) + log_v.unsqueeze(-2)
        transport_plan = torch.exp(log_plan)
        distances = (transport_plan * cost_matrix).sum(dim=(-2, -1))

        if reduction == "none":
            return distances
        if reduction == "sum":
            return distances.sum()
        return distances.mean()


class DistributionDistance(nn.Module):
    """Ablation-aware distribution distance wrapper for SC-LFI."""

    def __init__(
        self,
        distance_type: str = "sw",
        *,
        sw_num_projections: int = 64,
        sw_p: float = 2.0,
        normalize_inputs: bool = True,
        projection_seed: int = 7,
        sinkhorn_epsilon: float = 0.1,
        sinkhorn_iterations: int = 50,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        distance_type = str(distance_type).lower()
        self.distance_type = distance_type
        if distance_type == "sw":
            self.distance = SlicedWassersteinDistance(
                num_projections=int(sw_num_projections),
                p=float(sw_p),
                reduction=reduction,
                normalize_inputs=bool(normalize_inputs),
                projection_seed=int(projection_seed),
            )
        elif distance_type == "entropic_ot":
            self.distance = UniformEntropicOTDistance(
                sinkhorn_epsilon=float(sinkhorn_epsilon),
                max_iterations=int(sinkhorn_iterations),
                cost_power=float(sw_p),
                reduction=reduction,
                normalize_inputs=bool(normalize_inputs),
            )
        else:
            raise ValueError(f"Unsupported distance_type: {distance_type}")

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        reduction: str | None = None,
    ) -> torch.Tensor:
        return self.distance(source, target, reduction=reduction)
