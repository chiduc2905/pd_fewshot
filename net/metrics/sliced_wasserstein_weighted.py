"""Weighted paper-style sliced Wasserstein distance utilities.

This module extends the existing paper-style SW estimator to weighted empirical
token distributions. The legacy SW implementation and the current unweighted
paper-style baseline remain unchanged.

Why this variant exists:
- paper-style SW averages projected 1D OT costs ``W_p^p`` over random
  projections and applies the outer ``1 / p`` root only once at the end;
- weighted 1D OT is more faithful when token gates define token importance;
- unequal token counts are handled exactly in 1D via monotone transport, not
  via interpolation heuristics.
"""

from __future__ import annotations

from typing import Optional

import torch

from net.metrics.sliced_wasserstein import prepare_token_distribution
from net.metrics.sliced_wasserstein_paper import PaperSlicedWassersteinDistance


class WeightedPaperSlicedWassersteinDistance(PaperSlicedWassersteinDistance):
    """Paper-style Monte-Carlo p-sliced Wasserstein for weighted token sets."""

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
        weight_eps: float = 1e-8,
    ) -> None:
        super().__init__(
            train_num_projections=train_num_projections,
            eval_num_projections=eval_num_projections,
            p=p,
            reduction=reduction,
            normalize_inputs=normalize_inputs,
            train_projection_mode=train_projection_mode,
            eval_projection_mode=eval_projection_mode,
            eval_num_repeats=eval_num_repeats,
            projection_seed=projection_seed,
        )
        if weight_eps <= 0.0:
            raise ValueError("weight_eps must be positive")
        self.weight_eps = float(weight_eps)

    def _prepare_weights_tensor(
        self,
        weights: torch.Tensor | None,
        target_shape: torch.Size,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if weights is None:
            weights_tensor = torch.ones(target_shape, device=device, dtype=dtype)
        else:
            weights_tensor = torch.as_tensor(weights, device=device, dtype=dtype)
            try:
                weights_tensor = torch.broadcast_to(weights_tensor, target_shape)
            except RuntimeError as exc:
                raise ValueError(
                    "weights must be broadcastable to the token shape "
                    f"{tuple(target_shape)}, got {tuple(weights_tensor.shape)}"
                ) from exc

        weights_tensor = weights_tensor.clamp_min(0.0)
        row_sum = weights_tensor.sum(dim=-1, keepdim=True)
        zero_rows = row_sum <= self.weight_eps

        # If a full row has zero usable mass after clamping, fall back to a
        # uniform distribution on that row so the OT computation stays finite.
        if zero_rows.any():
            uniform = torch.ones_like(weights_tensor)
            weights_tensor = torch.where(zero_rows, uniform, weights_tensor)
            row_sum = weights_tensor.sum(dim=-1, keepdim=True)

        return weights_tensor / row_sum.clamp_min(self.weight_eps)

    @staticmethod
    def _set_last_cdf_mass_to_one(cdf: torch.Tensor) -> torch.Tensor:
        cdf = cdf.clamp_min(0.0)
        cdf[..., -1] = 1.0
        return cdf

    def _project_sort_values_and_weights(
        self,
        tokens: torch.Tensor,
        weights: torch.Tensor,
        projections: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        projected = torch.matmul(tokens, projections).transpose(-2, -1).contiguous()
        sorted_values, sorted_indices = torch.sort(projected, dim=-1)
        expanded_weights = weights.unsqueeze(-2).expand(*weights.shape[:-1], projected.shape[-2], weights.shape[-1])
        sorted_weights = torch.gather(expanded_weights, dim=-1, index=sorted_indices)
        return sorted_values, sorted_weights

    def projected_ot_cost(
        self,
        x_proj: torch.Tensor,
        y_proj: torch.Tensor,
        x_weights: torch.Tensor | None = None,
        y_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return exact projected 1D OT costs ``W_p^p`` for weighted measures."""
        if x_proj.dim() != 2 or y_proj.dim() != 2:
            raise ValueError(f"Expected 2D tensors, got x={tuple(x_proj.shape)} y={tuple(y_proj.shape)}")
        if x_proj.shape[0] != y_proj.shape[0]:
            raise ValueError(f"Batch mismatch: x={tuple(x_proj.shape)} y={tuple(y_proj.shape)}")

        x_weights = self._prepare_weights_tensor(
            x_weights,
            target_shape=x_proj.shape,
            device=x_proj.device,
            dtype=x_proj.dtype,
        )
        y_weights = self._prepare_weights_tensor(
            y_weights,
            target_shape=y_proj.shape,
            device=y_proj.device,
            dtype=y_proj.dtype,
        )

        x_sorted, x_indices = torch.sort(x_proj, dim=-1)
        y_sorted, y_indices = torch.sort(y_proj, dim=-1)
        x_sorted_weights = torch.gather(x_weights, dim=-1, index=x_indices)
        y_sorted_weights = torch.gather(y_weights, dim=-1, index=y_indices)
        return self._weighted_wasserstein_1d_cost_from_sorted(
            x_sorted,
            y_sorted,
            x_sorted_weights,
            y_sorted_weights,
        )

    def _weighted_wasserstein_1d_cost_from_sorted(
        self,
        x_sorted: torch.Tensor,
        y_sorted: torch.Tensor,
        x_sorted_weights: torch.Tensor,
        y_sorted_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Return exact 1D weighted OT costs ``W_p^p`` from sorted supports."""
        if x_sorted.dim() != 2 or y_sorted.dim() != 2:
            raise ValueError(f"Expected 2D tensors, got x={tuple(x_sorted.shape)} y={tuple(y_sorted.shape)}")
        if x_sorted.shape[0] != y_sorted.shape[0]:
            raise ValueError(f"Batch mismatch: x={tuple(x_sorted.shape)} y={tuple(y_sorted.shape)}")
        if x_sorted_weights.shape != x_sorted.shape:
            raise ValueError(
                "x_sorted_weights must match x_sorted: "
                f"x={tuple(x_sorted.shape)} weights={tuple(x_sorted_weights.shape)}"
            )
        if y_sorted_weights.shape != y_sorted.shape:
            raise ValueError(
                "y_sorted_weights must match y_sorted: "
                f"y={tuple(y_sorted.shape)} weights={tuple(y_sorted_weights.shape)}"
            )

        x_sorted_weights = self._prepare_weights_tensor(
            x_sorted_weights,
            target_shape=x_sorted.shape,
            device=x_sorted.device,
            dtype=x_sorted.dtype,
        )
        y_sorted_weights = self._prepare_weights_tensor(
            y_sorted_weights,
            target_shape=y_sorted.shape,
            device=y_sorted.device,
            dtype=y_sorted.dtype,
        )

        x_cdf = self._set_last_cdf_mass_to_one(x_sorted_weights.cumsum(dim=-1))
        y_cdf = self._set_last_cdf_mass_to_one(y_sorted_weights.cumsum(dim=-1))

        zero = torch.zeros((x_sorted.shape[0], 1), device=x_sorted.device, dtype=x_sorted.dtype)
        boundaries = torch.cat([zero, x_cdf, y_cdf], dim=-1)
        boundaries = torch.sort(boundaries, dim=-1).values

        left_mass = boundaries[..., :-1].contiguous()
        interval_mass = (boundaries[..., 1:] - boundaries[..., :-1]).clamp_min(0.0)

        idx_x = torch.searchsorted(x_cdf.contiguous(), left_mass, right=True).clamp_max(x_sorted.shape[-1] - 1)
        idx_y = torch.searchsorted(y_cdf.contiguous(), left_mass, right=True).clamp_max(y_sorted.shape[-1] - 1)

        x_values = torch.gather(x_sorted, dim=-1, index=idx_x)
        y_values = torch.gather(y_sorted, dim=-1, index=idx_y)
        return (interval_mass * torch.abs(x_values - y_values).pow(self.p)).sum(dim=-1)

    def _compute_once(
        self,
        query_tokens: torch.Tensor,
        support_tokens: torch.Tensor,
        query_weights: torch.Tensor | None = None,
        support_weights: torch.Tensor | None = None,
        repeat_idx: int = 0,
    ) -> torch.Tensor:
        query_tokens = prepare_token_distribution(query_tokens, normalize=self.normalize_inputs)
        support_tokens = prepare_token_distribution(support_tokens, normalize=self.normalize_inputs)

        if query_tokens.shape[:-2] != support_tokens.shape[:-2]:
            raise ValueError(
                "Leading dimensions must match for weighted SW computation: "
                f"query={tuple(query_tokens.shape)} support={tuple(support_tokens.shape)}"
            )
        if query_tokens.shape[-1] != support_tokens.shape[-1]:
            raise ValueError(
                "Token feature dimensions must match: "
                f"query={query_tokens.shape[-1]} support={support_tokens.shape[-1]}"
            )

        query_weights = self._prepare_weights_tensor(
            query_weights,
            target_shape=query_tokens.shape[:-1],
            device=query_tokens.device,
            dtype=query_tokens.dtype,
        )
        support_weights = self._prepare_weights_tensor(
            support_weights,
            target_shape=support_tokens.shape[:-1],
            device=support_tokens.device,
            dtype=support_tokens.dtype,
        )

        projections = self._get_projections(
            feature_dim=query_tokens.shape[-1],
            device=query_tokens.device,
            dtype=query_tokens.dtype,
            repeat_idx=repeat_idx,
        )
        query_sorted, query_sorted_weights = self._project_sort_values_and_weights(
            query_tokens,
            query_weights,
            projections,
        )
        support_sorted, support_sorted_weights = self._project_sort_values_and_weights(
            support_tokens,
            support_weights,
            projections,
        )

        leading_shape = query_sorted.shape[:-2]
        num_projections = query_sorted.shape[-2]
        projected_costs = self._weighted_wasserstein_1d_cost_from_sorted(
            query_sorted.reshape(-1, query_sorted.shape[-1]),
            support_sorted.reshape(-1, support_sorted.shape[-1]),
            query_sorted_weights.reshape(-1, query_sorted_weights.shape[-1]),
            support_sorted_weights.reshape(-1, support_sorted_weights.shape[-1]),
        ).reshape(*leading_shape, num_projections)

        # Paper-style SW averages projected OT costs W_p^p first, then applies
        # the outer 1/p root only once at the end.
        return projected_costs.mean(dim=-1).clamp_min(0.0).pow(1.0 / self.p)

    def _pairwise_compute_once(
        self,
        query_tokens: torch.Tensor,
        support_tokens: torch.Tensor,
        query_weights: torch.Tensor | None = None,
        support_weights: torch.Tensor | None = None,
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

        query_weights = self._prepare_weights_tensor(
            query_weights,
            target_shape=query_tokens.shape[:-1],
            device=query_tokens.device,
            dtype=query_tokens.dtype,
        )
        support_weights = self._prepare_weights_tensor(
            support_weights,
            target_shape=support_tokens.shape[:-1],
            device=support_tokens.device,
            dtype=support_tokens.dtype,
        )

        projections = self._get_projections(
            feature_dim=query_tokens.shape[-1],
            device=query_tokens.device,
            dtype=query_tokens.dtype,
            repeat_idx=repeat_idx,
        )
        query_sorted, query_sorted_weights = self._project_sort_values_and_weights(
            query_tokens,
            query_weights,
            projections,
        )
        support_sorted, support_sorted_weights = self._project_sort_values_and_weights(
            support_tokens,
            support_weights,
            projections,
        )

        num_query = query_sorted.shape[0]
        way_num = support_sorted.shape[0]
        num_projections = query_sorted.shape[1]

        projected_costs = self._weighted_wasserstein_1d_cost_from_sorted(
            query_sorted.unsqueeze(1).expand(-1, way_num, -1, -1).reshape(-1, query_sorted.shape[-1]),
            support_sorted.unsqueeze(0).expand(num_query, -1, -1, -1).reshape(-1, support_sorted.shape[-1]),
            query_sorted_weights.unsqueeze(1).expand(-1, way_num, -1, -1).reshape(-1, query_sorted_weights.shape[-1]),
            support_sorted_weights.unsqueeze(0).expand(num_query, -1, -1, -1).reshape(
                -1,
                support_sorted_weights.shape[-1],
            ),
        ).reshape(num_query, way_num, num_projections)

        return projected_costs.mean(dim=-1).clamp_min(0.0).pow(1.0 / self.p)

    @staticmethod
    def _reduce_distances(distances: torch.Tensor, reduction: str) -> torch.Tensor:
        if reduction == "none":
            return distances
        if reduction == "sum":
            return distances.sum()
        return distances.mean()

    def pairwise_distance(
        self,
        query_tokens: torch.Tensor,
        support_tokens: torch.Tensor,
        query_weights: torch.Tensor | None = None,
        support_weights: torch.Tensor | None = None,
        reduction: Optional[str] = None,
    ) -> torch.Tensor:
        reduction = reduction or self.reduction
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(f"Unsupported reduction: {reduction}")

        repeats = 1 if self.training else self.eval_num_repeats
        distances = [
            self._pairwise_compute_once(
                query_tokens,
                support_tokens,
                query_weights=query_weights,
                support_weights=support_weights,
                repeat_idx=repeat_idx,
            )
            for repeat_idx in range(repeats)
        ]
        stacked = torch.stack(distances, dim=0).mean(dim=0)
        return self._reduce_distances(stacked, reduction)

    def forward(
        self,
        query_tokens: torch.Tensor,
        support_tokens: torch.Tensor,
        query_weights: torch.Tensor | None = None,
        support_weights: torch.Tensor | None = None,
        reduction: Optional[str] = None,
    ) -> torch.Tensor:
        reduction = reduction or self.reduction
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(f"Unsupported reduction: {reduction}")

        repeats = 1 if self.training else self.eval_num_repeats
        distances = [
            self._compute_once(
                query_tokens,
                support_tokens,
                query_weights=query_weights,
                support_weights=support_weights,
                repeat_idx=repeat_idx,
            )
            for repeat_idx in range(repeats)
        ]
        stacked = torch.stack(distances, dim=0).mean(dim=0)
        return self._reduce_distances(stacked, reduction)
