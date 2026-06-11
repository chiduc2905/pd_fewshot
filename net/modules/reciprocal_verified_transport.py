"""Reciprocal structural verification for UOT transport plans.

The verifier is parameter-free and episode-local.  It keeps transported mass
only when a pair is low-cost from both query-to-support and support-to-query
views, and when neighboring token pairs support the same correspondence.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _infer_token_hw(token_count: int) -> tuple[int, int]:
    root = int(round(float(token_count) ** 0.5))
    if root * root == int(token_count):
        return root, root
    return 1, int(token_count)


def _local_average_excluding_center(
    values: torch.Tensor,
    *,
    spatial_hw: tuple[int, int],
    kernel_size: int,
) -> torch.Tensor:
    if values.dim() != 2:
        raise ValueError(f"values must have shape (Batch, Tokens), got {tuple(values.shape)}")
    height, width = int(spatial_hw[0]), int(spatial_hw[1])
    if height * width != values.shape[-1]:
        raise ValueError(f"spatial_hw={spatial_hw} does not match token count {values.shape[-1]}")
    if kernel_size <= 1:
        return torch.zeros_like(values)

    pad = int(kernel_size) // 2
    grid = values.reshape(-1, 1, height, width)
    kernel = grid.new_ones((1, 1, int(kernel_size), int(kernel_size)))
    summed = F.conv2d(grid, kernel, padding=pad) - grid
    counts = F.conv2d(torch.ones_like(grid), kernel, padding=pad) - 1.0
    return (summed / counts.clamp_min(1.0)).reshape_as(values)


class ReciprocalVerifiedTransport(nn.Module):
    """Build a verified sub-plan from an existing token transport plan."""

    def __init__(
        self,
        *,
        beta: float = 0.85,
        tau: float = 0.10,
        ratio_threshold: float = 0.25,
        kernel_size: int = 3,
        cost_quantile: float = 0.35,
        min_gate: float = 0.05,
        detach_gate: bool = True,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.beta = float(beta)
        self.tau = float(tau)
        self.ratio_threshold = float(ratio_threshold)
        self.kernel_size = int(kernel_size)
        self.cost_quantile = float(cost_quantile)
        self.min_gate = float(min_gate)
        self.detach_gate = bool(detach_gate)
        self.eps = float(eps)

        if not 0.0 <= self.beta <= 1.0:
            raise ValueError("rvuot_beta must be in [0, 1]")
        if self.tau <= 0.0:
            raise ValueError("rvuot_tau must be positive")
        if self.ratio_threshold < 0.0:
            raise ValueError("rvuot_ratio_threshold must be non-negative")
        if self.kernel_size < 1 or self.kernel_size % 2 != 1:
            raise ValueError("rvuot_kernel_size must be an odd positive integer")
        if not 0.0 < self.cost_quantile < 1.0:
            raise ValueError("rvuot_cost_quantile must be in (0, 1)")
        if not 0.0 <= self.min_gate <= 1.0:
            raise ValueError("rvuot_min_gate must be in [0, 1]")
        if self.eps <= 0.0:
            raise ValueError("rvuot eps must be positive")

    def _reciprocal_affinity(self, cost: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        reference_cost = cost.detach() if self.detach_gate else cost
        flat = reference_cost.reshape(*reference_cost.shape[:-2], -1)
        threshold = torch.quantile(flat, q=self.cost_quantile, dim=-1).reshape(
            *reference_cost.shape[:-2],
            1,
            1,
        )
        scale = reference_cost.std(dim=(-1, -2), keepdim=True, unbiased=False).clamp_min(self.eps)
        low_cost_gate = torch.sigmoid((threshold - reference_cost) / (self.tau * scale).clamp_min(self.eps))

        temperature = (self.tau * scale.detach()).clamp_min(self.eps)
        row_affinity = torch.softmax(-reference_cost / temperature, dim=-1)
        col_affinity = torch.softmax(-reference_cost / temperature, dim=-2)
        reciprocal = torch.sqrt((row_affinity * col_affinity).clamp_min(0.0))
        reciprocal = reciprocal / reciprocal.amax(dim=(-1, -2), keepdim=True).clamp_min(self.eps)
        return low_cost_gate * reciprocal, low_cost_gate

    def _coherence_gate(
        self,
        affinity: torch.Tensor,
        *,
        query_hw: tuple[int, int],
        support_hw: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_query, way_num, shot_num, query_len, support_len = affinity.shape
        q_neighbors = _local_average_excluding_center(
            affinity.permute(0, 1, 2, 4, 3).reshape(-1, query_len),
            spatial_hw=query_hw,
            kernel_size=self.kernel_size,
        ).reshape(num_query, way_num, shot_num, support_len, query_len).permute(0, 1, 2, 4, 3)
        s_neighbors = _local_average_excluding_center(
            affinity.reshape(-1, support_len),
            spatial_hw=support_hw,
            kernel_size=self.kernel_size,
        ).reshape(num_query, way_num, shot_num, query_len, support_len)

        neighborhood = torch.sqrt((q_neighbors * s_neighbors).clamp_min(0.0))
        support_ratio = neighborhood / affinity.clamp_min(self.eps)
        absolute_gate = torch.sigmoid((neighborhood - self.ratio_threshold) / self.tau)
        ratio_gate = torch.sigmoid((support_ratio - self.ratio_threshold) / self.tau)
        gate = absolute_gate.pow(2) * ratio_gate
        return gate, support_ratio

    def forward(
        self,
        *,
        cost: torch.Tensor,
        plan: torch.Tensor,
        spatial_hw: tuple[int, int] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if cost.dim() != 5:
            raise ValueError(f"cost must have shape (Nq, Way, Shot, Lq, Ls), got {tuple(cost.shape)}")
        if tuple(plan.shape) != tuple(cost.shape):
            raise ValueError(f"plan shape {tuple(plan.shape)} must match cost shape {tuple(cost.shape)}")

        query_len = int(cost.shape[-2])
        support_len = int(cost.shape[-1])
        query_hw = (
            spatial_hw
            if spatial_hw is not None and int(spatial_hw[0]) * int(spatial_hw[1]) == query_len
            else _infer_token_hw(query_len)
        )
        support_hw = (
            spatial_hw
            if spatial_hw is not None and int(spatial_hw[0]) * int(spatial_hw[1]) == support_len
            else _infer_token_hw(support_len)
        )

        affinity, low_cost_gate = self._reciprocal_affinity(cost)
        coherence_gate, support_ratio = self._coherence_gate(
            affinity,
            query_hw=query_hw,
            support_hw=support_hw,
        )
        soft_gate = affinity * coherence_gate
        soft_gate = soft_gate / soft_gate.amax(dim=(-1, -2), keepdim=True).clamp_min(self.eps)
        soft_gate = self.min_gate + (1.0 - self.min_gate) * soft_gate
        verifier = (1.0 - self.beta) + self.beta * soft_gate
        verified_plan = plan * verifier.to(device=plan.device, dtype=plan.dtype)

        original_mass = plan.sum(dim=(-1, -2))
        verified_mass = verified_plan.sum(dim=(-1, -2))
        retained_ratio = verified_mass / original_mass.clamp_min(self.eps)
        removed_mass = (plan - verified_plan).clamp_min(0.0).sum(dim=(-1, -2))
        diagnostics = {
            "rvuot/enabled": plan.new_tensor(1.0),
            "rvuot/beta": plan.new_tensor(self.beta),
            "rvuot/tau": plan.new_tensor(self.tau),
            "rvuot/ratio_threshold": plan.new_tensor(self.ratio_threshold),
            "rvuot/kernel_size": plan.new_tensor(float(self.kernel_size)),
            "rvuot/cost_quantile": plan.new_tensor(self.cost_quantile),
            "rvuot/min_gate": plan.new_tensor(self.min_gate),
            "rvuot/gate_mean": verifier.detach().mean(),
            "rvuot/gate_min": verifier.detach().amin(),
            "rvuot/gate_max": verifier.detach().amax(),
            "rvuot/low_cost_gate_mean": low_cost_gate.detach().mean(),
            "rvuot/coherence_gate_mean": coherence_gate.detach().mean(),
            "rvuot/support_ratio_mean": support_ratio.detach().mean(),
            "rvuot/retained_mass_ratio": retained_ratio.detach().mean(),
            "rvuot/removed_mass_mean": removed_mass.detach().mean(),
            "rvuot/original_mass_mean": original_mass.detach().mean(),
            "rvuot/verified_mass_mean": verified_mass.detach().mean(),
            "rvuot_verifier_gate": verifier.detach(),
            "rvuot_reciprocal_affinity": affinity.detach(),
        }
        return verified_plan, diagnostics
