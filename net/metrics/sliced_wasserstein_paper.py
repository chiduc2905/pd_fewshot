"""Paper-style sliced Wasserstein distance utilities.

This module keeps the original reusable SW implementation untouched and provides
an alternative estimator aligned with common paper / library practice:

    SW_p(mu, nu) = ((1 / M) * sum_m W_p(theta_m#mu, theta_m#nu)^p)^(1 / p)

Key differences from the legacy local-SW helper:
- uses the POT-style estimator for p-sliced Wasserstein;
- supports train/eval projection regimes independently;
- supports projection resampling during training;
- computes exact 1D OT for unequal token counts under uniform masses instead of
  heuristic resampling.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from net.metrics.sliced_wasserstein import prepare_token_distribution


def _int_lcm(a: int, b: int) -> int:
    return abs(int(a * b)) // math.gcd(int(a), int(b))


class PaperSlicedWassersteinDistance(nn.Module):
    """Monte-Carlo p-sliced Wasserstein with exact uniform 1D OT."""

    def __init__(
        self,
        train_num_projections: int = 128,
        eval_num_projections: int = 512,
        p: float = 2.0,
        reduction: str = "mean",
        normalize_inputs: bool = False,
        train_projection_mode: str = "resample",
        eval_projection_mode: str = "fixed",
        eval_num_repeats: int = 1,
        projection_seed: int = 7,
    ) -> None:
        super().__init__()
        if train_num_projections <= 0:
            raise ValueError("train_num_projections must be positive")
        if eval_num_projections <= 0:
            raise ValueError("eval_num_projections must be positive")
        if p <= 0.0:
            raise ValueError("p must be positive")
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(f"Unsupported reduction: {reduction}")
        if train_projection_mode not in {"resample", "fixed"}:
            raise ValueError(f"Unsupported train_projection_mode: {train_projection_mode}")
        if eval_projection_mode not in {"resample", "fixed"}:
            raise ValueError(f"Unsupported eval_projection_mode: {eval_projection_mode}")
        if eval_num_repeats <= 0:
            raise ValueError("eval_num_repeats must be positive")

        self.train_num_projections = int(train_num_projections)
        self.eval_num_projections = int(eval_num_projections)
        self.p = float(p)
        self.reduction = reduction
        self.normalize_inputs = bool(normalize_inputs)
        self.train_projection_mode = str(train_projection_mode)
        self.eval_projection_mode = str(eval_projection_mode)
        self.eval_num_repeats = int(eval_num_repeats)
        self.projection_seed = int(projection_seed)

        self.register_buffer("_fixed_projection_bank", torch.empty(0), persistent=False)
        self.register_buffer("_resample_counter", torch.zeros((), dtype=torch.long), persistent=False)
        self._fixed_projection_key: Optional[tuple[int, int, int]] = None
        self._quantile_plan_cache: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    @staticmethod
    def _sample_random_projections(
        feature_dim: int,
        num_projections: int,
        device: torch.device,
        dtype: torch.dtype,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        projections = torch.randn(
            feature_dim,
            num_projections,
            generator=generator,
            device=device,
            dtype=dtype,
        )
        return F.normalize(projections, p=2, dim=0)

    def _get_fixed_projections(
        self,
        feature_dim: int,
        num_projections: int,
        device: torch.device,
        dtype: torch.dtype,
        repeat_idx: int = 0,
    ) -> torch.Tensor:
        cache_key = (feature_dim, num_projections, repeat_idx)
        if self._fixed_projection_key != cache_key or self._fixed_projection_bank.numel() == 0:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.projection_seed + repeat_idx)
            bank = self._sample_random_projections(
                feature_dim=feature_dim,
                num_projections=num_projections,
                device=torch.device("cpu"),
                dtype=torch.float32,
                generator=generator,
            )
            self._fixed_projection_bank = bank
            self._fixed_projection_key = cache_key
        return self._fixed_projection_bank.to(device=device, dtype=dtype)

    def _get_resampled_projections(
        self,
        feature_dim: int,
        num_projections: int,
        device: torch.device,
        dtype: torch.dtype,
        repeat_idx: int = 0,
    ) -> torch.Tensor:
        # Keep projection resampling stochastic across calls while isolating it
        # from the global RNG stream so projection_seed is always honored.
        call_idx = int(self._resample_counter.item())
        self._resample_counter.add_(1)

        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.projection_seed + 1_000_003 * call_idx + 1_009 * repeat_idx)
        bank = self._sample_random_projections(
            feature_dim=feature_dim,
            num_projections=num_projections,
            device=torch.device("cpu"),
            dtype=torch.float32,
            generator=generator,
        )
        return bank.to(device=device, dtype=dtype)

    def _get_projections(
        self,
        feature_dim: int,
        device: torch.device,
        dtype: torch.dtype,
        repeat_idx: int = 0,
    ) -> torch.Tensor:
        if self.training:
            num_projections = self.train_num_projections
            mode = self.train_projection_mode
        else:
            num_projections = self.eval_num_projections
            mode = self.eval_projection_mode

        if mode == "resample":
            return self._get_resampled_projections(
                feature_dim=feature_dim,
                num_projections=num_projections,
                device=device,
                dtype=dtype,
                repeat_idx=repeat_idx,
            )
        return self._get_fixed_projections(
            feature_dim=feature_dim,
            num_projections=num_projections,
            device=device,
            dtype=dtype,
            repeat_idx=repeat_idx,
        )

    def _uniform_wasserstein_1d_cost(
        self,
        x_proj: torch.Tensor,
        y_proj: torch.Tensor,
    ) -> torch.Tensor:
        """Exact 1D OT cost W_p^p for empirical uniform measures.

        Args:
            x_proj: `(K, NX)` sorted or unsorted projected samples.
            y_proj: `(K, NY)` sorted or unsorted projected samples.
        Returns:
            `(K,)` tensor of exact projected OT costs before the final `1/p`.
        """
        if x_proj.dim() != 2 or y_proj.dim() != 2:
            raise ValueError(f"Expected 2D tensors, got x={tuple(x_proj.shape)} y={tuple(y_proj.shape)}")
        if x_proj.shape[0] != y_proj.shape[0]:
            raise ValueError(f"Batch mismatch: x={tuple(x_proj.shape)} y={tuple(y_proj.shape)}")

        x_sorted = torch.sort(x_proj, dim=-1).values
        y_sorted = torch.sort(y_proj, dim=-1).values
        return self._uniform_wasserstein_1d_cost_from_sorted(x_sorted, y_sorted)

    def _get_uniform_quantile_plan(
        self,
        nx: int,
        ny: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cache_key = (int(nx), int(ny))
        cached = self._quantile_plan_cache.get(cache_key)
        if cached is None:
            lcm = _int_lcm(int(nx), int(ny))
            x_steps = torch.arange(nx + 1, dtype=torch.long) * (lcm // nx)
            y_steps = torch.arange(ny + 1, dtype=torch.long) * (lcm // ny)
            boundaries = torch.unique(torch.cat([x_steps, y_steps], dim=0), sorted=True)
            left = boundaries[:-1]
            right = boundaries[1:]
            idx_x = torch.div(left * nx, lcm, rounding_mode="floor").clamp_max(nx - 1)
            idx_y = torch.div(left * ny, lcm, rounding_mode="floor").clamp_max(ny - 1)
            widths = (right - left).to(torch.float32) / float(lcm)
            cached = (idx_x, idx_y, widths)
            self._quantile_plan_cache[cache_key] = cached
        idx_x, idx_y, widths = cached
        return (
            idx_x.to(device=device),
            idx_y.to(device=device),
            widths.to(device=device, dtype=dtype),
        )

    def _uniform_wasserstein_1d_cost_from_sorted(
        self,
        x_sorted: torch.Tensor,
        y_sorted: torch.Tensor,
    ) -> torch.Tensor:
        if x_sorted.dim() != 2 or y_sorted.dim() != 2:
            raise ValueError(f"Expected 2D tensors, got x={tuple(x_sorted.shape)} y={tuple(y_sorted.shape)}")
        if x_sorted.shape[0] != y_sorted.shape[0]:
            raise ValueError(f"Batch mismatch: x={tuple(x_sorted.shape)} y={tuple(y_sorted.shape)}")

        nx = x_sorted.shape[1]
        ny = y_sorted.shape[1]
        if nx == ny:
            return torch.abs(x_sorted - y_sorted).pow(self.p).mean(dim=-1)

        idx_x, idx_y, widths = self._get_uniform_quantile_plan(
            nx=nx,
            ny=ny,
            device=x_sorted.device,
            dtype=x_sorted.dtype,
        )
        x_quantiles = x_sorted.index_select(dim=-1, index=idx_x)
        y_quantiles = y_sorted.index_select(dim=-1, index=idx_y)
        return (torch.abs(x_quantiles - y_quantiles).pow(self.p) * widths.unsqueeze(0)).sum(dim=-1)

    @staticmethod
    def _project_and_sort(tokens: torch.Tensor, projections: torch.Tensor) -> torch.Tensor:
        projected = torch.matmul(tokens, projections).transpose(-2, -1).contiguous()
        return torch.sort(projected, dim=-1).values

    def _pairwise_uniform_wasserstein_1d_cost_from_sorted(
        self,
        query_sorted: torch.Tensor,
        support_sorted: torch.Tensor,
    ) -> torch.Tensor:
        if query_sorted.dim() != 3:
            raise ValueError(
                "query_sorted must have shape (NumQuery, Projections, Tokens), "
                f"got {tuple(query_sorted.shape)}"
            )
        if support_sorted.dim() != 3:
            raise ValueError(
                "support_sorted must have shape (Way, Projections, Tokens), "
                f"got {tuple(support_sorted.shape)}"
            )
        if query_sorted.shape[1] != support_sorted.shape[1]:
            raise ValueError(
                "Projection counts must match: "
                f"query={tuple(query_sorted.shape)} support={tuple(support_sorted.shape)}"
            )

        nx = query_sorted.shape[-1]
        ny = support_sorted.shape[-1]
        if nx == ny:
            pairwise_diff = torch.abs(query_sorted.unsqueeze(1) - support_sorted.unsqueeze(0)).pow(self.p)
            return pairwise_diff.mean(dim=-1)

        idx_x, idx_y, widths = self._get_uniform_quantile_plan(
            nx=nx,
            ny=ny,
            device=query_sorted.device,
            dtype=query_sorted.dtype,
        )
        query_quantiles = query_sorted.index_select(dim=-1, index=idx_x)
        support_quantiles = support_sorted.index_select(dim=-1, index=idx_y)
        pairwise_diff = torch.abs(query_quantiles.unsqueeze(1) - support_quantiles.unsqueeze(0)).pow(self.p)
        return (pairwise_diff * widths.view(1, 1, 1, -1)).sum(dim=-1)

    def _compute_once(self, query_tokens: torch.Tensor, support_tokens: torch.Tensor, repeat_idx: int = 0) -> torch.Tensor:
        query_tokens = prepare_token_distribution(query_tokens, normalize=self.normalize_inputs)
        support_tokens = prepare_token_distribution(support_tokens, normalize=self.normalize_inputs)

        if query_tokens.shape[:-2] != support_tokens.shape[:-2]:
            raise ValueError(
                "Leading dimensions must match for SW computation: "
                f"query={tuple(query_tokens.shape)} support={tuple(support_tokens.shape)}"
            )
        if query_tokens.shape[-1] != support_tokens.shape[-1]:
            raise ValueError(
                "Token feature dimensions must match: "
                f"query={query_tokens.shape[-1]} support={support_tokens.shape[-1]}"
            )

        leading_shape = query_tokens.shape[:-2]
        feature_dim = query_tokens.shape[-1]
        projections = self._get_projections(
            feature_dim=feature_dim,
            device=query_tokens.device,
            dtype=query_tokens.dtype,
            repeat_idx=repeat_idx,
        )
        query_sorted = self._project_and_sort(query_tokens, projections)
        support_sorted = self._project_and_sort(support_tokens, projections)
        num_projections = query_sorted.shape[-2]

        flat_query = query_sorted.reshape(-1, query_sorted.shape[-1])
        flat_support = support_sorted.reshape(-1, support_sorted.shape[-1])
        projected_costs = self._uniform_wasserstein_1d_cost_from_sorted(flat_query, flat_support)
        projected_costs = projected_costs.reshape(*leading_shape, num_projections)
        return projected_costs.mean(dim=-1).clamp_min(0.0).pow(1.0 / self.p)

    def _pairwise_compute_once(
        self,
        query_tokens: torch.Tensor,
        support_tokens: torch.Tensor,
        repeat_idx: int = 0,
    ) -> torch.Tensor:
        query_tokens = prepare_token_distribution(query_tokens, normalize=self.normalize_inputs)
        support_tokens = prepare_token_distribution(support_tokens, normalize=self.normalize_inputs)

        if query_tokens.dim() != 3:
            raise ValueError(
                "query_tokens must have shape (NumQuery, Tokens, Dim), "
                f"got {tuple(query_tokens.shape)}"
            )
        if support_tokens.dim() != 3:
            raise ValueError(
                "support_tokens must have shape (Way, Tokens, Dim), "
                f"got {tuple(support_tokens.shape)}"
            )
        if query_tokens.shape[-1] != support_tokens.shape[-1]:
            raise ValueError(
                "Token feature dimensions must match: "
                f"query={query_tokens.shape[-1]} support={support_tokens.shape[-1]}"
            )

        projections = self._get_projections(
            feature_dim=query_tokens.shape[-1],
            device=query_tokens.device,
            dtype=query_tokens.dtype,
            repeat_idx=repeat_idx,
        )
        query_sorted = self._project_and_sort(query_tokens, projections)
        support_sorted = self._project_and_sort(support_tokens, projections)
        projected_costs = self._pairwise_uniform_wasserstein_1d_cost_from_sorted(query_sorted, support_sorted)
        return projected_costs.mean(dim=-1).clamp_min(0.0).pow(1.0 / self.p)

    def pairwise_distance(
        self,
        query_tokens: torch.Tensor,
        support_tokens: torch.Tensor,
        reduction: Optional[str] = None,
    ) -> torch.Tensor:
        reduction = reduction or self.reduction
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(f"Unsupported reduction: {reduction}")

        repeats = 1 if self.training else self.eval_num_repeats
        distances = [
            self._pairwise_compute_once(query_tokens, support_tokens, repeat_idx=repeat_idx)
            for repeat_idx in range(repeats)
        ]
        stacked = torch.stack(distances, dim=0).mean(dim=0)

        if reduction == "none":
            return stacked
        if reduction == "sum":
            return stacked.sum()
        return stacked.mean()

    def forward(
        self,
        query_tokens: torch.Tensor,
        support_tokens: torch.Tensor,
        reduction: Optional[str] = None,
    ) -> torch.Tensor:
        reduction = reduction or self.reduction
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(f"Unsupported reduction: {reduction}")

        repeats = 1 if self.training else self.eval_num_repeats
        distances = [self._compute_once(query_tokens, support_tokens, repeat_idx=repeat_idx) for repeat_idx in range(repeats)]
        stacked = torch.stack(distances, dim=0).mean(dim=0)

        if reduction == "none":
            return stacked
        if reduction == "sum":
            return stacked.sum()
        return stacked.mean()
