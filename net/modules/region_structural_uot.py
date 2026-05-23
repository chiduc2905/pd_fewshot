"""Region-structural UOT guidance for few-shot scalogram matching.

This module builds a coarse feature-space region plan, then uses that plan as a
soft structural prior on the fine-token UOT cost.  It does not use image masks or
brightness thresholds; all signals come from projected backbone tokens and the
token-grid geometry.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from net.modules.fgw_uot_solver import fgw_uot_solve


def parse_region_grid_size(value: str | int | Sequence[int]) -> tuple[int, int]:
    if isinstance(value, int):
        height = width = int(value)
    elif isinstance(value, str):
        text = value.strip().lower().replace(" ", "")
        if "x" in text:
            parts = text.split("x")
            if len(parts) != 2:
                raise ValueError(f"Invalid region_uot_grid_size: {value!r}")
            height, width = int(parts[0]), int(parts[1])
        else:
            height = width = int(text)
    else:
        parts = list(value)
        if len(parts) != 2:
            raise ValueError(f"Invalid region_uot_grid_size: {value!r}")
        height, width = int(parts[0]), int(parts[1])
    if height <= 1 or width <= 1:
        raise ValueError("region_uot_grid_size must be at least 2x2")
    return height, width


def _token_grid_assignments(
    spatial_hw: tuple[int, int],
    region_hw: tuple[int, int],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    height, width = int(spatial_hw[0]), int(spatial_hw[1])
    region_h, region_w = int(region_hw[0]), int(region_hw[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid spatial_hw={spatial_hw}")
    y = torch.arange(height, device=device)
    x = torch.arange(width, device=device)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    region_y = torch.div(grid_y * region_h, height, rounding_mode="floor").clamp(max=region_h - 1)
    region_x = torch.div(grid_x * region_w, width, rounding_mode="floor").clamp(max=region_w - 1)
    region_index = (region_y * region_w + region_x).reshape(-1)
    return F.one_hot(region_index, num_classes=region_h * region_w).to(dtype=dtype)


def _normalized_region_coordinates(
    region_hw: tuple[int, int],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    region_h, region_w = int(region_hw[0]), int(region_hw[1])
    y = (torch.arange(region_h, device=device, dtype=dtype) + 0.5) / float(region_h)
    x = (torch.arange(region_w, device=device, dtype=dtype) + 0.5) / float(region_w)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    return torch.stack([grid_y.reshape(-1), grid_x.reshape(-1)], dim=-1)


class RegionStructuralUOTGuidance(nn.Module):
    """Coarse region UOT prior for fine-token transport costs."""

    def __init__(
        self,
        *,
        grid_size: str | int | Sequence[int] = "3x3",
        strength: float = 0.20,
        fgw_alpha: float = 0.35,
        tau: float = 0.5,
        sinkhorn_epsilon: float = 0.05,
        fgw_iters: int = 4,
        sinkhorn_iters: int = 40,
        sinkhorn_tol: float = 1e-5,
        ground_cost: str = "euclidean",
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.grid_size = parse_region_grid_size(grid_size)
        self.strength = float(strength)
        self.fgw_alpha = float(fgw_alpha)
        self.tau = float(tau)
        self.sinkhorn_epsilon = float(sinkhorn_epsilon)
        self.fgw_iters = int(fgw_iters)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.sinkhorn_tol = float(sinkhorn_tol)
        self.ground_cost = str(ground_cost).strip().lower().replace("-", "_")
        self.eps = float(eps)
        if not 0.0 <= self.strength < 1.0:
            raise ValueError("region_uot_strength must be in [0, 1)")
        if not 0.0 <= self.fgw_alpha <= 1.0:
            raise ValueError("region_uot_fgw_alpha must be in [0, 1]")
        if self.tau <= 0.0:
            raise ValueError("region_uot_tau must be positive")
        if self.sinkhorn_epsilon <= 0.0:
            raise ValueError("region_uot_sinkhorn_epsilon must be positive")
        if self.fgw_iters <= 0 or self.sinkhorn_iters <= 0:
            raise ValueError("region_uot solver iteration counts must be positive")
        if self.ground_cost not in {"auto", "euclidean", "cosine"}:
            raise ValueError("region_uot ground_cost must be auto/euclidean/cosine")

    def _pool_regions(self, tokens: torch.Tensor, spatial_hw: tuple[int, int]) -> torch.Tensor:
        if tokens.dim() != 3:
            raise ValueError(f"tokens must have shape (Batch, Tokens, Dim), got {tuple(tokens.shape)}")
        batch, token_count, dim = tokens.shape
        height, width = int(spatial_hw[0]), int(spatial_hw[1])
        if token_count != height * width:
            raise ValueError(f"spatial_hw={spatial_hw} does not match token count {token_count}")
        token_map = tokens.reshape(batch, height, width, dim).permute(0, 3, 1, 2).contiguous()
        pooled = F.adaptive_avg_pool2d(token_map, output_size=self.grid_size)
        regions = pooled.permute(0, 2, 3, 1).reshape(batch, self.grid_size[0] * self.grid_size[1], dim)
        return F.normalize(regions, p=2, dim=-1, eps=self.eps)

    def _pairwise_feature_cost(self, query_regions: torch.Tensor, support_regions: torch.Tensor) -> torch.Tensor:
        mode = "euclidean" if self.ground_cost == "auto" else self.ground_cost
        if mode == "cosine":
            query_norm = F.normalize(query_regions, p=2, dim=-1, eps=self.eps)
            support_norm = F.normalize(support_regions, p=2, dim=-1, eps=self.eps)
            sim = torch.einsum("qrd,psd->qprs", query_norm, support_norm)
            return (1.0 - sim).clamp_min(0.0)
        query_sq = query_regions.pow(2).sum(dim=-1)
        support_sq = support_regions.pow(2).sum(dim=-1)
        dot = torch.einsum("qrd,psd->qprs", query_regions, support_regions)
        return (query_sq[:, None, :, None] + support_sq[None, :, None, :] - 2.0 * dot).clamp_min(0.0)

    def forward(
        self,
        *,
        flat_cost: torch.Tensor,
        query_tokens: torch.Tensor,
        support_tokens: torch.Tensor,
        way_num: int,
        shot_num: int,
        spatial_hw: tuple[int, int],
        rho: torch.Tensor | float,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if flat_cost.dim() != 4:
            raise ValueError(f"flat_cost must have shape (Nq, Way*Shot, Lq, Ls), got {tuple(flat_cost.shape)}")
        num_query, num_pairs, query_len, support_len = flat_cost.shape
        if num_pairs != int(way_num) * int(shot_num):
            raise ValueError(f"flat_cost pair dimension {num_pairs} does not match way*shot={way_num * shot_num}")
        if tuple(query_tokens.shape[:2]) != (num_query, query_len):
            raise ValueError(f"query_tokens shape {tuple(query_tokens.shape)} does not match flat_cost")
        if tuple(support_tokens.shape[:3]) != (int(way_num), int(shot_num), support_len):
            raise ValueError(f"support_tokens shape {tuple(support_tokens.shape)} does not match flat_cost")

        query_regions = self._pool_regions(
            query_tokens.to(device=flat_cost.device, dtype=flat_cost.dtype),
            spatial_hw,
        )
        support_flat = support_tokens.reshape(num_pairs, support_len, support_tokens.shape[-1])
        support_regions = self._pool_regions(
            support_flat.to(device=flat_cost.device, dtype=flat_cost.dtype),
            spatial_hw,
        )
        region_count = query_regions.shape[1]
        feature_cost = self._pairwise_feature_cost(query_regions, support_regions)
        base_mean = flat_cost.detach().mean(dim=(-1, -2), keepdim=True).clamp_min(self.eps)
        feature_mean = feature_cost.detach().mean(dim=(-1, -2), keepdim=True).clamp_min(self.eps)
        feature_cost = feature_cost * (base_mean / feature_mean)

        coords = _normalized_region_coordinates(self.grid_size, device=flat_cost.device, dtype=flat_cost.dtype)
        structure = torch.cdist(coords, coords, p=2).pow(2)
        structure = structure / structure.mean().clamp_min(self.eps)
        batch = num_query * num_pairs
        pair_cost = feature_cost.reshape(batch, region_count, region_count)
        structure_q = structure.unsqueeze(0).expand(batch, -1, -1)
        structure_s = structure.unsqueeze(0).expand(batch, -1, -1)

        rho_tensor = torch.as_tensor(rho, device=flat_cost.device, dtype=flat_cost.dtype)
        if rho_tensor.numel() == 1:
            rho_flat = rho_tensor.reshape(1).expand(batch)
        else:
            rho_flat = rho_tensor.to(device=flat_cost.device, dtype=flat_cost.dtype).reshape(-1)
            if rho_flat.numel() == num_query:
                rho_flat = rho_flat[:, None].expand(num_query, num_pairs).reshape(batch)
            elif rho_flat.numel() == num_pairs:
                rho_flat = rho_flat[None, :].expand(num_query, num_pairs).reshape(batch)
            elif rho_flat.numel() != batch:
                raise ValueError(
                    "rho must be scalar or broadcastable to NumQuery*Way*Shot, "
                    f"got {rho_tensor.shape}"
                )
        rho_flat = rho_flat.clamp_min(self.eps)
        log_a = torch.log(rho_flat[:, None] / float(region_count)).expand(batch, region_count)
        log_b = torch.log(rho_flat[:, None] / float(region_count)).expand(batch, region_count)

        coarse_plan, coarse_cost = fgw_uot_solve(
            pair_cost,
            structure_q,
            structure_s,
            log_a,
            log_b,
            alpha=flat_cost.new_tensor(self.fgw_alpha),
            tau=self.tau,
            eps=self.sinkhorn_epsilon,
            fgw_iters=self.fgw_iters,
            sinkhorn_iters=self.sinkhorn_iters,
            tol=self.sinkhorn_tol,
        )
        coarse_plan = torch.nan_to_num(coarse_plan, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        coarse_cost = torch.nan_to_num(coarse_cost, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        plan_norm = coarse_plan / coarse_plan.amax(dim=(-1, -2), keepdim=True).clamp_min(self.eps)

        assignment = _token_grid_assignments(
            spatial_hw,
            self.grid_size,
            device=flat_cost.device,
            dtype=flat_cost.dtype,
        )
        if assignment.shape[0] != query_len or assignment.shape[0] != support_len:
            raise ValueError(
                "Region guidance currently expects query/support token grids with the same spatial_hw; "
                f"got assignment={assignment.shape[0]}, query_len={query_len}, support_len={support_len}"
            )
        fine_affinity = torch.matmul(torch.matmul(assignment, plan_norm), assignment.transpose(0, 1))
        fine_affinity = fine_affinity.reshape(num_query, num_pairs, query_len, support_len)
        fine_affinity = fine_affinity / fine_affinity.amax(dim=(-1, -2), keepdim=True).clamp_min(self.eps)

        discount = (self.strength * fine_affinity).clamp(min=0.0, max=self.strength)
        guided_cost = flat_cost * (1.0 - discount)

        coarse_plan_view = coarse_plan.reshape(
            num_query,
            int(way_num),
            int(shot_num),
            region_count,
            region_count,
        )
        coarse_cost_view = coarse_cost.reshape(
            num_query,
            int(way_num),
            int(shot_num),
            region_count,
            region_count,
        )
        coarse_transport_cost = (coarse_plan * coarse_cost).sum(dim=(-1, -2)).reshape(
            num_query,
            int(way_num),
            int(shot_num),
        )
        coarse_transport_mass = coarse_plan.sum(dim=(-1, -2)).reshape(num_query, int(way_num), int(shot_num))
        payload = {
            "region_uot_coarse_plan": coarse_plan_view.detach(),
            "region_uot_coarse_cost_matrix": coarse_cost_view.detach(),
            "region_uot_guided_cost_matrix": guided_cost.reshape(
                num_query,
                int(way_num),
                int(shot_num),
                query_len,
                support_len,
            ),
            "region_uot_coarse_transport_cost": coarse_transport_cost.detach(),
            "region_uot_coarse_transported_mass": coarse_transport_mass.detach(),
            "region_uot/strength": flat_cost.new_tensor(self.strength),
            "region_uot/fgw_alpha": flat_cost.new_tensor(self.fgw_alpha),
            "region_uot/coarse_mass_mean": coarse_transport_mass.mean().detach(),
            "region_uot/coarse_cost_mean": coarse_transport_cost.mean().detach(),
            "region_uot/affinity_peak": fine_affinity.amax(dim=(-1, -2)).mean().detach(),
            "region_uot/cost_delta_ratio": (
                (guided_cost.detach() - flat_cost.detach()).abs().mean()
                / flat_cost.detach().abs().mean().clamp_min(self.eps)
            ),
        }
        return guided_cost, payload

