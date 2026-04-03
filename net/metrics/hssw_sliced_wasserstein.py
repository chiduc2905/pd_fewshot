"""Research-grade Sliced Wasserstein transport for HS-SW.

This implementation is standalone and intentionally separate from the older SW
helpers in the repository. It supports exact 1D Wasserstein transport between
weighted empirical measures after random 1D projections.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class HSSWSlicedWassersteinDistance(nn.Module):
    """Monte-Carlo sliced Wasserstein over token measures.

    Expected token shapes:
    - rowwise forward: `(Batch, Tokens, Dim)` vs `(Batch, Tokens, Dim)`
    - pairwise path: `(NumQuery, Tokens, Dim)` vs `(NumSupport, Tokens, Dim)`
    """

    def __init__(
        self,
        train_num_projections: int = 32,
        eval_num_projections: int = 64,
        p: float = 2.0,
        weighted: bool = True,
        normalize_tokens: bool = True,
        train_projection_mode: str = "resample",
        eval_projection_mode: str = "fixed",
        eval_num_repeats: int = 1,
        projection_seed: int = 7,
        pairwise_chunk_size: int | None = None,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if train_num_projections <= 0:
            raise ValueError("train_num_projections must be positive")
        if eval_num_projections <= 0:
            raise ValueError("eval_num_projections must be positive")
        if p <= 0.0:
            raise ValueError("p must be positive")
        if train_projection_mode not in {"resample", "fixed"}:
            raise ValueError(f"Unsupported train_projection_mode: {train_projection_mode}")
        if eval_projection_mode not in {"resample", "fixed"}:
            raise ValueError(f"Unsupported eval_projection_mode: {eval_projection_mode}")
        if eval_num_repeats <= 0:
            raise ValueError("eval_num_repeats must be positive")
        if pairwise_chunk_size is not None and pairwise_chunk_size <= 0:
            raise ValueError("pairwise_chunk_size must be positive when provided")
        if eps <= 0.0:
            raise ValueError("eps must be positive")

        self.train_num_projections = int(train_num_projections)
        self.eval_num_projections = int(eval_num_projections)
        self.p = float(p)
        self.weighted = bool(weighted)
        self.normalize_tokens = bool(normalize_tokens)
        self.train_projection_mode = str(train_projection_mode)
        self.eval_projection_mode = str(eval_projection_mode)
        self.eval_num_repeats = int(eval_num_repeats)
        self.projection_seed = int(projection_seed)
        self.pairwise_chunk_size = None if pairwise_chunk_size is None else int(pairwise_chunk_size)
        self.eps = float(eps)

        self.register_buffer("_fixed_projection_bank", torch.empty(0), persistent=False)
        self.register_buffer("_resample_counter", torch.zeros((), dtype=torch.long), persistent=False)
        self._fixed_projection_key: Optional[tuple[int, int, int]] = None

    @staticmethod
    def _working_dtype(dtype: torch.dtype) -> torch.dtype:
        if dtype in {torch.float16, torch.bfloat16}:
            return torch.float32
        return dtype

    def _normalize_measure_weights(
        self,
        weights: torch.Tensor | None,
        batch_size: int,
        num_tokens: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if weights is None or not self.weighted:
            return torch.full((batch_size, num_tokens), 1.0 / float(num_tokens), device=device, dtype=dtype)

        if weights.dim() != 2:
            raise ValueError(f"weights must have shape (Batch, Tokens), got {tuple(weights.shape)}")
        if weights.shape != (batch_size, num_tokens):
            raise ValueError(
                "weights shape must match token shape: "
                f"expected {(batch_size, num_tokens)}, got {tuple(weights.shape)}"
            )

        weights = weights.to(device=device, dtype=dtype).clamp_min(0.0)
        denom = weights.sum(dim=-1, keepdim=True)
        zero_mass = denom <= self.eps
        if zero_mass.any():
            weights = torch.where(zero_mass, torch.ones_like(weights), weights)
            denom = weights.sum(dim=-1, keepdim=True)
        return weights / denom.clamp_min(self.eps)

    def _prepare_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.dim() != 3:
            raise ValueError(f"tokens must have shape (Batch, Tokens, Dim), got {tuple(tokens.shape)}")
        tokens = tokens.to(dtype=self._working_dtype(tokens.dtype))
        if self.normalize_tokens:
            tokens = F.normalize(tokens, p=2, dim=-1, eps=self.eps)
        return tokens

    @staticmethod
    def _sample_projection_bank(
        feature_dim: int,
        num_projections: int,
        device: torch.device,
        dtype: torch.dtype,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        projections = torch.randn(
            feature_dim,
            num_projections,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        return F.normalize(projections, p=2, dim=0)

    def _get_fixed_projection_bank(
        self,
        feature_dim: int,
        num_projections: int,
        device: torch.device,
        dtype: torch.dtype,
        repeat_idx: int,
    ) -> torch.Tensor:
        cache_key = (feature_dim, num_projections, repeat_idx)
        if self._fixed_projection_key != cache_key or self._fixed_projection_bank.numel() == 0:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.projection_seed + repeat_idx)
            self._fixed_projection_bank = self._sample_projection_bank(
                feature_dim=feature_dim,
                num_projections=num_projections,
                device=torch.device("cpu"),
                dtype=torch.float32,
                generator=generator,
            )
            self._fixed_projection_key = cache_key
        return self._fixed_projection_bank.to(device=device, dtype=dtype)

    def _get_resampled_projection_bank(
        self,
        feature_dim: int,
        num_projections: int,
        device: torch.device,
        dtype: torch.dtype,
        repeat_idx: int,
    ) -> torch.Tensor:
        call_idx = int(self._resample_counter.item())
        self._resample_counter.add_(1)

        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.projection_seed + 1_000_003 * call_idx + 1_009 * repeat_idx)
        bank = self._sample_projection_bank(
            feature_dim=feature_dim,
            num_projections=num_projections,
            device=torch.device("cpu"),
            dtype=torch.float32,
            generator=generator,
        )
        return bank.to(device=device, dtype=dtype)

    def _get_projection_bank(
        self,
        feature_dim: int,
        device: torch.device,
        dtype: torch.dtype,
        repeat_idx: int,
    ) -> torch.Tensor:
        if self.training:
            num_projections = self.train_num_projections
            mode = self.train_projection_mode
        else:
            num_projections = self.eval_num_projections
            mode = self.eval_projection_mode

        if mode == "fixed":
            return self._get_fixed_projection_bank(
                feature_dim=feature_dim,
                num_projections=num_projections,
                device=device,
                dtype=dtype,
                repeat_idx=repeat_idx,
            )
        return self._get_resampled_projection_bank(
            feature_dim=feature_dim,
            num_projections=num_projections,
            device=device,
            dtype=dtype,
            repeat_idx=repeat_idx,
        )

    @staticmethod
    def _project_and_sort(
        tokens: torch.Tensor,
        weights: torch.Tensor,
        projections: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # projected values: [Batch, Proj, Tokens]
        projected = torch.einsum("btd,dl->blt", tokens, projections)
        order = projected.argsort(dim=-1)
        sorted_values = torch.gather(projected, dim=-1, index=order)
        sorted_weights = torch.gather(
            weights.unsqueeze(1).expand(-1, projected.shape[1], -1),
            dim=-1,
            index=order,
        )
        return sorted_values.contiguous(), sorted_weights.contiguous()

    def _rowwise_projected_transport_cost(
        self,
        x_sorted: torch.Tensor,
        x_weights: torch.Tensor,
        y_sorted: torch.Tensor,
        y_weights: torch.Tensor,
    ) -> torch.Tensor:
        if x_sorted.shape[:2] != y_sorted.shape[:2]:
            raise ValueError(
                "Projected rowwise inputs must align in batch and projection dims: "
                f"x={tuple(x_sorted.shape)} y={tuple(y_sorted.shape)}"
            )

        batch_size, num_projections, num_x = x_sorted.shape
        num_y = y_sorted.shape[-1]
        x_cdf = x_weights.cumsum(dim=-1)
        y_cdf = y_weights.cumsum(dim=-1)
        x_cdf[..., -1] = 1.0
        y_cdf[..., -1] = 1.0

        zero = x_sorted.new_zeros(batch_size, num_projections, 1)
        boundaries = torch.sort(torch.cat([zero, x_cdf, y_cdf], dim=-1), dim=-1).values
        left = boundaries[..., :-1].contiguous()
        right = boundaries[..., 1:].contiguous()
        interval_mass = (right - left).clamp_min(0.0)

        left_flat = left.reshape(-1, left.shape[-1])
        x_cdf_flat = x_cdf.reshape(-1, num_x).contiguous()
        y_cdf_flat = y_cdf.reshape(-1, num_y).contiguous()
        x_sorted_flat = x_sorted.reshape(-1, num_x)
        y_sorted_flat = y_sorted.reshape(-1, num_y)

        x_index = torch.searchsorted(x_cdf_flat, left_flat, right=True).clamp_max(num_x - 1)
        y_index = torch.searchsorted(y_cdf_flat, left_flat, right=True).clamp_max(num_y - 1)
        x_quantiles = torch.gather(x_sorted_flat, dim=-1, index=x_index)
        y_quantiles = torch.gather(y_sorted_flat, dim=-1, index=y_index)

        cost = (interval_mass.reshape(-1, interval_mass.shape[-1]) * (x_quantiles - y_quantiles).abs().pow(self.p)).sum(
            dim=-1
        )
        return cost.reshape(batch_size, num_projections)

    def _pairwise_projected_transport_cost(
        self,
        x_sorted: torch.Tensor,
        x_weights: torch.Tensor,
        y_sorted: torch.Tensor,
        y_weights: torch.Tensor,
    ) -> torch.Tensor:
        if x_sorted.shape[1] != y_sorted.shape[1]:
            raise ValueError(
                "Projected pairwise inputs must align in projection dimension: "
                f"x={tuple(x_sorted.shape)} y={tuple(y_sorted.shape)}"
            )

        num_query, num_projections, num_x = x_sorted.shape
        num_support, _, num_y = y_sorted.shape

        x_cdf = x_weights.cumsum(dim=-1)
        y_cdf = y_weights.cumsum(dim=-1)
        x_cdf[..., -1] = 1.0
        y_cdf[..., -1] = 1.0

        x_cdf_expanded = x_cdf.unsqueeze(1).expand(-1, num_support, -1, -1)
        y_cdf_expanded = y_cdf.unsqueeze(0).expand(num_query, -1, -1, -1)
        zero = x_sorted.new_zeros(num_query, num_support, num_projections, 1)
        boundaries = torch.sort(torch.cat([zero, x_cdf_expanded, y_cdf_expanded], dim=-1), dim=-1).values
        left = boundaries[..., :-1].contiguous()
        right = boundaries[..., 1:].contiguous()
        interval_mass = (right - left).clamp_min(0.0)

        left_flat = left.reshape(-1, left.shape[-1])
        x_cdf_flat = x_cdf_expanded.reshape(-1, num_x).contiguous()
        y_cdf_flat = y_cdf_expanded.reshape(-1, num_y).contiguous()
        x_sorted_flat = x_sorted.unsqueeze(1).expand(-1, num_support, -1, -1).reshape(-1, num_x)
        y_sorted_flat = y_sorted.unsqueeze(0).expand(num_query, -1, -1, -1).reshape(-1, num_y)

        x_index = torch.searchsorted(x_cdf_flat, left_flat, right=True).clamp_max(num_x - 1)
        y_index = torch.searchsorted(y_cdf_flat, left_flat, right=True).clamp_max(num_y - 1)
        x_quantiles = torch.gather(x_sorted_flat, dim=-1, index=x_index)
        y_quantiles = torch.gather(y_sorted_flat, dim=-1, index=y_index)

        cost = (interval_mass.reshape(-1, interval_mass.shape[-1]) * (x_quantiles - y_quantiles).abs().pow(self.p)).sum(
            dim=-1
        )
        return cost.reshape(num_query, num_support, num_projections)

    @staticmethod
    def _reduce(distances: torch.Tensor, reduction: str) -> torch.Tensor:
        if reduction == "none":
            return distances
        if reduction == "sum":
            return distances.sum()
        if reduction == "mean":
            return distances.mean()
        raise ValueError(f"Unsupported reduction: {reduction}")

    def _distance_once(
        self,
        x_tokens: torch.Tensor,
        y_tokens: torch.Tensor,
        x_weights: torch.Tensor | None,
        y_weights: torch.Tensor | None,
        repeat_idx: int,
    ) -> torch.Tensor:
        x_tokens = self._prepare_tokens(x_tokens)
        y_tokens = self._prepare_tokens(y_tokens)
        if x_tokens.shape[0] != y_tokens.shape[0]:
            raise ValueError(f"Rowwise batch sizes must match: x={tuple(x_tokens.shape)} y={tuple(y_tokens.shape)}")
        if x_tokens.shape[-1] != y_tokens.shape[-1]:
            raise ValueError(
                "Token feature dimensions must match: "
                f"x={x_tokens.shape[-1]} y={y_tokens.shape[-1]}"
            )

        x_weights = self._normalize_measure_weights(
            weights=x_weights,
            batch_size=x_tokens.shape[0],
            num_tokens=x_tokens.shape[1],
            device=x_tokens.device,
            dtype=x_tokens.dtype,
        )
        y_weights = self._normalize_measure_weights(
            weights=y_weights,
            batch_size=y_tokens.shape[0],
            num_tokens=y_tokens.shape[1],
            device=y_tokens.device,
            dtype=y_tokens.dtype,
        )

        projections = self._get_projection_bank(
            feature_dim=x_tokens.shape[-1],
            device=x_tokens.device,
            dtype=x_tokens.dtype,
            repeat_idx=repeat_idx,
        )
        x_sorted, x_sorted_weights = self._project_and_sort(x_tokens, x_weights, projections)
        y_sorted, y_sorted_weights = self._project_and_sort(y_tokens, y_weights, projections)
        projected_cost = self._rowwise_projected_transport_cost(
            x_sorted=x_sorted,
            x_weights=x_sorted_weights,
            y_sorted=y_sorted,
            y_weights=y_sorted_weights,
        )
        return projected_cost.mean(dim=-1).clamp_min(0.0).pow(1.0 / self.p)

    def forward(
        self,
        x_tokens: torch.Tensor,
        y_tokens: torch.Tensor,
        x_weights: torch.Tensor | None = None,
        y_weights: torch.Tensor | None = None,
        reduction: str = "none",
    ) -> torch.Tensor:
        repeats = 1 if self.training else self.eval_num_repeats
        distances = [
            self._distance_once(
                x_tokens=x_tokens,
                y_tokens=y_tokens,
                x_weights=x_weights,
                y_weights=y_weights,
                repeat_idx=repeat_idx,
            )
            for repeat_idx in range(repeats)
        ]
        return self._reduce(torch.stack(distances, dim=0).mean(dim=0), reduction)

    def _pairwise_distance_once(
        self,
        query_tokens: torch.Tensor,
        support_tokens: torch.Tensor,
        query_weights: torch.Tensor | None,
        support_weights: torch.Tensor | None,
        repeat_idx: int,
    ) -> torch.Tensor:
        query_tokens = self._prepare_tokens(query_tokens)
        support_tokens = self._prepare_tokens(support_tokens)
        if query_tokens.shape[-1] != support_tokens.shape[-1]:
            raise ValueError(
                "Token feature dimensions must match: "
                f"query={query_tokens.shape[-1]} support={support_tokens.shape[-1]}"
            )

        query_weights = self._normalize_measure_weights(
            weights=query_weights,
            batch_size=query_tokens.shape[0],
            num_tokens=query_tokens.shape[1],
            device=query_tokens.device,
            dtype=query_tokens.dtype,
        )
        support_weights = self._normalize_measure_weights(
            weights=support_weights,
            batch_size=support_tokens.shape[0],
            num_tokens=support_tokens.shape[1],
            device=support_tokens.device,
            dtype=support_tokens.dtype,
        )

        projections = self._get_projection_bank(
            feature_dim=query_tokens.shape[-1],
            device=query_tokens.device,
            dtype=query_tokens.dtype,
            repeat_idx=repeat_idx,
        )
        query_sorted, query_sorted_weights = self._project_and_sort(query_tokens, query_weights, projections)
        support_sorted, support_sorted_weights = self._project_and_sort(support_tokens, support_weights, projections)

        def _compute_chunk(start_idx: int, end_idx: int) -> torch.Tensor:
            cost = self._pairwise_projected_transport_cost(
                x_sorted=query_sorted,
                x_weights=query_sorted_weights,
                y_sorted=support_sorted[start_idx:end_idx],
                y_weights=support_sorted_weights[start_idx:end_idx],
            )
            return cost.mean(dim=-1).clamp_min(0.0).pow(1.0 / self.p)

        if self.pairwise_chunk_size is None:
            return _compute_chunk(0, support_tokens.shape[0])

        chunk_outputs = []
        for start_idx in range(0, support_tokens.shape[0], self.pairwise_chunk_size):
            end_idx = min(start_idx + self.pairwise_chunk_size, support_tokens.shape[0])
            chunk_outputs.append(_compute_chunk(start_idx, end_idx))
        return torch.cat(chunk_outputs, dim=1)

    def pairwise_distance(
        self,
        query_tokens: torch.Tensor,
        support_tokens: torch.Tensor,
        query_weights: torch.Tensor | None = None,
        support_weights: torch.Tensor | None = None,
        reduction: str = "none",
    ) -> torch.Tensor:
        repeats = 1 if self.training else self.eval_num_repeats
        distances = [
            self._pairwise_distance_once(
                query_tokens=query_tokens,
                support_tokens=support_tokens,
                query_weights=query_weights,
                support_weights=support_weights,
                repeat_idx=repeat_idx,
            )
            for repeat_idx in range(repeats)
        ]
        return self._reduce(torch.stack(distances, dim=0).mean(dim=0), reduction)
