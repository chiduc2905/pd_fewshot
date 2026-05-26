"""Task-Adaptive Distributional Sliced Wasserstein few-shot model."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from net.fewshot_common import BaseConv64FewShotModel, feature_map_to_tokens


class TADSWResult(dict):
    """Dict-like result exposing `.shape` through logits."""

    @property
    def shape(self):
        logits = self.get("logits")
        return None if logits is None else logits.shape


def _inverse_softplus(value: float) -> float:
    if value <= 0.0:
        raise ValueError("inverse softplus expects a positive value")
    return math.log(math.expm1(float(value)))


def _normalize_measure_weights(weights: torch.Tensor, eps: float) -> torch.Tensor:
    weights = weights.clamp_min(0.0)
    total = weights.sum(dim=-1, keepdim=True)
    uniform = torch.full_like(weights, 1.0 / float(weights.shape[-1]))
    normalized = weights / total.clamp_min(float(eps))
    return torch.where(total > float(eps), normalized, uniform)


def sliced_wasserstein_distance(
    X: torch.Tensor,
    Y: torch.Tensor,
    thetas: torch.Tensor,
    p: float = 2.0,
    num_quantiles: int = 256,
) -> torch.Tensor:
    """Compute SW_p between two empirical point clouds using shared slices."""
    if X.dim() != 2 or Y.dim() != 2:
        raise ValueError(f"X and Y must be 2D point clouds, got X={tuple(X.shape)} Y={tuple(Y.shape)}")
    if thetas.dim() != 2:
        raise ValueError(f"thetas must have shape (NumSlices, Dim), got {tuple(thetas.shape)}")
    if X.shape[-1] != Y.shape[-1] or X.shape[-1] != thetas.shape[-1]:
        raise ValueError(
            "Feature dimensions must match: "
            f"X={X.shape[-1]} Y={Y.shape[-1]} thetas={thetas.shape[-1]}"
        )
    if p <= 0.0:
        raise ValueError("p must be positive")
    if num_quantiles <= 0:
        raise ValueError("num_quantiles must be positive")

    thetas = F.normalize(thetas, p=2, dim=1)
    proj_X = X @ thetas.transpose(0, 1)
    proj_Y = Y @ thetas.transpose(0, 1)

    if X.shape[0] == Y.shape[0]:
        proj_X_sorted = proj_X.sort(dim=0).values
        proj_Y_sorted = proj_Y.sort(dim=0).values
        wasserstein_per_slice = (proj_X_sorted - proj_Y_sorted).abs().pow(p).mean(dim=0)
    else:
        quantiles = torch.linspace(
            0.0,
            1.0,
            int(num_quantiles) + 2,
            device=X.device,
            dtype=X.dtype,
        )[1:-1]
        q_x = torch.quantile(proj_X, quantiles, dim=0)
        q_y = torch.quantile(proj_Y, quantiles, dim=0)
        wasserstein_per_slice = (q_x - q_y).abs().pow(p).mean(dim=0)

    return wasserstein_per_slice.mean().clamp_min(0.0).pow(1.0 / float(p))


def _project_sort_values_and_weights(
    tokens: torch.Tensor,
    weights: torch.Tensor,
    thetas: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    projected = torch.matmul(tokens, thetas.transpose(0, 1)).transpose(-2, -1).contiguous()
    order = projected.argsort(dim=-1)
    sorted_values = torch.gather(projected, dim=-1, index=order)
    sorted_weights = torch.gather(
        weights.unsqueeze(1).expand(-1, projected.shape[1], -1),
        dim=-1,
        index=order,
    )
    return sorted_values.contiguous(), sorted_weights.contiguous()


def _pairwise_weighted_wasserstein_1d_from_sorted(
    query_sorted: torch.Tensor,
    query_weights: torch.Tensor,
    support_sorted: torch.Tensor,
    support_weights: torch.Tensor,
    p: float,
    eps: float,
) -> torch.Tensor:
    if query_sorted.dim() != 3 or support_sorted.dim() != 3:
        raise ValueError(
            "sorted projections must have shape (Batch, Slices, Tokens), "
            f"got query={tuple(query_sorted.shape)} support={tuple(support_sorted.shape)}"
        )
    if query_sorted.shape[:2] != query_weights.shape[:2] or support_sorted.shape[:2] != support_weights.shape[:2]:
        raise ValueError("projection values and weights must align in batch and slice dimensions")
    if query_sorted.shape[1] != support_sorted.shape[1]:
        raise ValueError(
            "Slice counts must match: "
            f"query={tuple(query_sorted.shape)} support={tuple(support_sorted.shape)}"
        )

    num_query, num_slices, num_query_tokens = query_sorted.shape
    num_support, _, num_support_tokens = support_sorted.shape

    query_weights = _normalize_measure_weights(query_weights, eps)
    support_weights = _normalize_measure_weights(support_weights, eps)
    query_cdf = query_weights.cumsum(dim=-1)
    support_cdf = support_weights.cumsum(dim=-1)
    query_cdf[..., -1] = 1.0
    support_cdf[..., -1] = 1.0

    query_cdf_expanded = query_cdf.unsqueeze(1).expand(-1, num_support, -1, -1)
    support_cdf_expanded = support_cdf.unsqueeze(0).expand(num_query, -1, -1, -1)
    zero = query_sorted.new_zeros(num_query, num_support, num_slices, 1)
    boundaries = torch.sort(torch.cat([zero, query_cdf_expanded, support_cdf_expanded], dim=-1), dim=-1).values
    left = boundaries[..., :-1].contiguous()
    interval_mass = (boundaries[..., 1:] - boundaries[..., :-1]).clamp_min(0.0).contiguous()

    flat_left = left.reshape(-1, left.shape[-1])
    query_cdf_flat = query_cdf_expanded.reshape(-1, num_query_tokens).contiguous()
    support_cdf_flat = support_cdf_expanded.reshape(-1, num_support_tokens).contiguous()
    query_values_flat = query_sorted.unsqueeze(1).expand(-1, num_support, -1, -1).reshape(-1, num_query_tokens)
    support_values_flat = support_sorted.unsqueeze(0).expand(num_query, -1, -1, -1).reshape(-1, num_support_tokens)

    query_index = torch.searchsorted(query_cdf_flat, flat_left, right=True).clamp_max(num_query_tokens - 1)
    support_index = torch.searchsorted(support_cdf_flat, flat_left, right=True).clamp_max(num_support_tokens - 1)
    query_quantiles = torch.gather(query_values_flat, dim=-1, index=query_index)
    support_quantiles = torch.gather(support_values_flat, dim=-1, index=support_index)

    cost = (
        interval_mass.reshape(-1, interval_mass.shape[-1])
        * (query_quantiles - support_quantiles).abs().pow(float(p))
    ).sum(dim=-1)
    return cost.reshape(num_query, num_support, num_slices)


def pairwise_weighted_sliced_wasserstein_distance(
    query_tokens: torch.Tensor,
    class_tokens: torch.Tensor,
    thetas: torch.Tensor,
    query_weights: torch.Tensor | None = None,
    class_weights: torch.Tensor | None = None,
    p: float = 2.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Exact weighted pairwise sliced Wasserstein distances for token measures."""
    if query_tokens.dim() != 3:
        raise ValueError(f"query_tokens must have shape (NumQuery, Tokens, Dim), got {tuple(query_tokens.shape)}")
    if class_tokens.dim() != 3:
        raise ValueError(f"class_tokens must have shape (Way, Tokens, Dim), got {tuple(class_tokens.shape)}")
    if thetas.dim() != 2:
        raise ValueError(f"thetas must have shape (NumSlices, Dim), got {tuple(thetas.shape)}")
    if query_tokens.shape[-1] != class_tokens.shape[-1] or query_tokens.shape[-1] != thetas.shape[-1]:
        raise ValueError(
            "Feature dimensions must match: "
            f"query={query_tokens.shape[-1]} class={class_tokens.shape[-1]} thetas={thetas.shape[-1]}"
        )
    if p <= 0.0:
        raise ValueError("p must be positive")

    if query_weights is None:
        query_weights = query_tokens.new_full(query_tokens.shape[:-1], 1.0 / float(query_tokens.shape[-2]))
    if class_weights is None:
        class_weights = class_tokens.new_full(class_tokens.shape[:-1], 1.0 / float(class_tokens.shape[-2]))
    query_weights = _normalize_measure_weights(query_weights.to(device=query_tokens.device, dtype=query_tokens.dtype), eps)
    class_weights = _normalize_measure_weights(class_weights.to(device=class_tokens.device, dtype=class_tokens.dtype), eps)

    thetas = F.normalize(thetas, p=2, dim=1, eps=float(eps))
    query_sorted, query_sorted_weights = _project_sort_values_and_weights(query_tokens, query_weights, thetas)
    class_sorted, class_sorted_weights = _project_sort_values_and_weights(class_tokens, class_weights, thetas)
    projected_costs = _pairwise_weighted_wasserstein_1d_from_sorted(
        query_sorted=query_sorted,
        query_weights=query_sorted_weights,
        support_sorted=class_sorted,
        support_weights=class_sorted_weights,
        p=float(p),
        eps=float(eps),
    )
    return projected_costs.mean(dim=-1).clamp_min(0.0).pow(1.0 / float(p))


def pairwise_sliced_wasserstein_distance(
    query_tokens: torch.Tensor,
    class_tokens: torch.Tensor,
    thetas: torch.Tensor,
    p: float = 2.0,
    num_quantiles: int = 256,
) -> torch.Tensor:
    """Vectorized query-to-class SW distances.

    Args:
        query_tokens: ``(NumQuery, QueryTokens, Dim)``.
        class_tokens: ``(Way, ClassTokens, Dim)``.
        thetas: ``(NumSlices, Dim)``.

    Returns:
        Distance matrix ``(NumQuery, Way)``.
    """
    if query_tokens.dim() != 3:
        raise ValueError(f"query_tokens must have shape (NumQuery, Tokens, Dim), got {tuple(query_tokens.shape)}")
    if class_tokens.dim() != 3:
        raise ValueError(f"class_tokens must have shape (Way, Tokens, Dim), got {tuple(class_tokens.shape)}")
    if thetas.dim() != 2:
        raise ValueError(f"thetas must have shape (NumSlices, Dim), got {tuple(thetas.shape)}")
    if query_tokens.shape[-1] != class_tokens.shape[-1] or query_tokens.shape[-1] != thetas.shape[-1]:
        raise ValueError(
            "Feature dimensions must match: "
            f"query={query_tokens.shape[-1]} class={class_tokens.shape[-1]} thetas={thetas.shape[-1]}"
        )
    if p <= 0.0:
        raise ValueError("p must be positive")
    if num_quantiles <= 0:
        raise ValueError("num_quantiles must be positive")

    thetas = F.normalize(thetas, p=2, dim=1)
    query_proj = torch.matmul(query_tokens, thetas.transpose(0, 1))
    class_proj = torch.matmul(class_tokens, thetas.transpose(0, 1))

    if query_tokens.shape[1] == class_tokens.shape[1]:
        query_sorted = query_proj.sort(dim=1).values
        class_sorted = class_proj.sort(dim=1).values
        projected_costs = (query_sorted.unsqueeze(1) - class_sorted.unsqueeze(0)).abs().pow(p).mean(dim=2)
    else:
        quantiles = torch.linspace(
            0.0,
            1.0,
            int(num_quantiles) + 2,
            device=query_tokens.device,
            dtype=query_tokens.dtype,
        )[1:-1]
        query_quantiles = torch.quantile(query_proj, quantiles, dim=1).permute(1, 0, 2).contiguous()
        class_quantiles = torch.quantile(class_proj, quantiles, dim=1).permute(1, 0, 2).contiguous()
        projected_costs = (
            query_quantiles.unsqueeze(1) - class_quantiles.unsqueeze(0)
        ).abs().pow(p).mean(dim=2)

    return projected_costs.mean(dim=-1).clamp_min(0.0).pow(1.0 / float(p))


def _sample_w(
    kappa: torch.Tensor,
    dim: int,
    num_samples: int,
    device: torch.device,
    dtype: torch.dtype,
    max_iter: int = 1000,
) -> torch.Tensor:
    """Wood (1994) rejection sampler for the vMF axial component."""
    if dim < 2:
        raise ValueError("vMF sampling requires dim >= 2")
    kappa = torch.as_tensor(kappa, device=device, dtype=dtype).clamp_min(0.0)
    dim_minus_one = torch.tensor(float(dim - 1), device=device, dtype=dtype)
    sqrt_term = torch.sqrt(4.0 * kappa.pow(2) + dim_minus_one.pow(2))
    b = (-2.0 * kappa + sqrt_term) / dim_minus_one
    a = (dim_minus_one + 2.0 * kappa + sqrt_term) / 4.0
    d_const = 4.0 * a * b / (1.0 + b) - dim_minus_one * math.log(float(dim - 1))

    w = torch.zeros(int(num_samples), device=device, dtype=dtype)
    done = torch.zeros(int(num_samples), dtype=torch.bool, device=device)
    beta_param = torch.full((), float(dim - 1) / 2.0, device=device, dtype=dtype)
    beta = torch.distributions.Beta(beta_param, beta_param)

    for _ in range(int(max_iter)):
        if bool(done.all().item()):
            break
        remaining = int((~done).sum().item())
        eps = beta.sample((remaining,)).to(device=device, dtype=dtype)
        w_proposal = (1.0 - (1.0 + b) * eps) / (1.0 - (1.0 - b) * eps)
        t = 2.0 * a * b / (1.0 - (1.0 - b) * eps)
        u = torch.rand(remaining, device=device, dtype=dtype)
        accept = dim_minus_one * t.log() - t + d_const >= u.log()

        idx = (~done).nonzero(as_tuple=True)[0]
        if bool(accept.any().item()):
            accepted_idx = idx[accept]
            w[accepted_idx] = w_proposal[accept]
            done[accepted_idx] = True

    if not bool(done.all().item()):
        w[~done] = 1.0 - 1e-6
    return w


def sample_vmf(
    mean_dir: torch.Tensor,
    concentration: torch.Tensor,
    num_samples: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Sample unit vectors from vMF(mean_dir, concentration)."""
    if mean_dir.dim() != 1:
        raise ValueError(f"mean_dir must have shape (Dim,), got {tuple(mean_dir.shape)}")
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")

    d = int(mean_dir.shape[0])
    device = mean_dir.device
    dtype = mean_dir.dtype
    mean_dir = F.normalize(mean_dir, p=2, dim=0, eps=eps)
    concentration = torch.as_tensor(concentration, device=device, dtype=dtype).clamp(min=0.0, max=500.0)

    if bool((concentration < 1e-4).detach().item()):
        z = torch.randn(int(num_samples), d, device=device, dtype=dtype)
        return F.normalize(z, p=2, dim=1, eps=eps)

    w = _sample_w(concentration, d, int(num_samples), device, dtype)
    v = torch.randn(int(num_samples), d, device=device, dtype=dtype)
    v[:, 0] = 0.0
    v = F.normalize(v, p=2, dim=1, eps=eps)

    sqrt_component = (1.0 - w.pow(2)).clamp_min(eps).sqrt()
    samples = torch.zeros(int(num_samples), d, device=device, dtype=dtype)
    samples[:, 0] = w
    samples[:, 1:] = sqrt_component.unsqueeze(1) * v[:, 1:]

    e1 = torch.zeros(d, device=device, dtype=dtype)
    e1[0] = 1.0
    u = F.normalize(e1 - mean_dir, p=2, dim=0, eps=eps)
    samples = samples - 2.0 * (samples @ u).unsqueeze(1) * u.unsqueeze(0)
    return F.normalize(samples, p=2, dim=1, eps=eps)


class TaskAdaptiveSlicing(nn.Module):
    """Infer vMF slicing parameters from all support tokens in an episode."""

    def __init__(self, feat_dim: int, hidden_dim: int = 256, eps: float = 1e-8) -> None:
        super().__init__()
        if feat_dim <= 0:
            raise ValueError("feat_dim must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if eps <= 0.0:
            raise ValueError("eps must be positive")
        self.task_encoder = nn.Sequential(
            nn.Linear(2 * int(feat_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ReLU(),
        )
        self.direction_head = nn.Linear(int(hidden_dim), int(feat_dim))
        self.concentration_head = nn.Linear(int(hidden_dim), 1)
        self.eps = float(eps)

    def forward(self, all_support_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if all_support_features.dim() != 2:
            raise ValueError(
                "all_support_features must have shape (NumSupportTokens, Dim), "
                f"got {tuple(all_support_features.shape)}"
            )
        support_mean = all_support_features.mean(dim=0)
        support_std = all_support_features.std(dim=0, unbiased=False)
        task_descriptor = torch.cat([support_mean, support_std], dim=0)
        hidden = self.task_encoder(task_descriptor)
        mean_dir = F.normalize(self.direction_head(hidden), p=2, dim=0, eps=self.eps)
        concentration = F.softplus(self.concentration_head(hidden)).squeeze()
        return mean_dir, concentration.clamp(min=0.0, max=500.0)


class TaskAdaptiveDistributionalSlicedWassersteinNet(BaseConv64FewShotModel):
    """Few-shot classifier using task-adaptive vMF-sampled SW projections."""

    def __init__(
        self,
        in_channels: int = 3,
        hidden_dim: int = 640,
        token_dim: int = 128,
        backbone_name: str = "resnet12",
        image_size: int = 64,
        resnet12_drop_rate: float = 0.0,
        resnet12_dropblock_size: int = 5,
        num_slices: int = 64,
        sw_p: float = 2.0,
        temperature: float = 0.1,
        task_hidden_dim: int = 256,
        token_weight_hidden_dim: int = 64,
        token_weight_uniform_mix: float = 0.2,
        num_quantiles: int = 256,
        normalize_tokens: bool = True,
        train_projection_mode: str = "deterministic",
        eval_projection_mode: str = "fixed",
        projection_seed: int = 7,
        shot_aggregation: str = "softmin",
        shot_softmin_beta: float = 5.0,
        proto_scale_init: float = 10.0,
        sw_scale_init: float | None = None,
        learnable_scales: bool = True,
        eps: float = 1e-8,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            backbone_name=backbone_name,
            image_size=image_size,
            resnet12_drop_rate=resnet12_drop_rate,
            resnet12_dropblock_size=resnet12_dropblock_size,
        )
        if num_slices <= 0:
            raise ValueError("num_slices must be positive")
        if token_dim <= 0:
            raise ValueError("token_dim must be positive")
        if sw_p <= 0.0:
            raise ValueError("sw_p must be positive")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if token_weight_hidden_dim <= 0:
            raise ValueError("token_weight_hidden_dim must be positive")
        if not 0.0 <= token_weight_uniform_mix <= 1.0:
            raise ValueError("token_weight_uniform_mix must be in [0, 1]")
        if num_quantiles <= 0:
            raise ValueError("num_quantiles must be positive")
        if train_projection_mode not in {"deterministic", "fixed", "resample", "vmf"}:
            raise ValueError(f"Unsupported train_projection_mode: {train_projection_mode}")
        if eval_projection_mode not in {"fixed", "deterministic", "resample"}:
            raise ValueError(f"Unsupported eval_projection_mode: {eval_projection_mode}")
        if shot_aggregation not in {"concat", "mean", "softmin"}:
            raise ValueError(f"Unsupported shot_aggregation: {shot_aggregation}")
        if shot_softmin_beta <= 0.0:
            raise ValueError("shot_softmin_beta must be positive")
        if proto_scale_init <= 0.0:
            raise ValueError("proto_scale_init must be positive")
        if eps <= 0.0:
            raise ValueError("eps must be positive")

        sw_scale_init = (1.0 / float(temperature)) if sw_scale_init is None else float(sw_scale_init)
        if sw_scale_init <= 0.0:
            raise ValueError("sw_scale_init must be positive")

        self.token_dim = int(token_dim)
        self.num_slices = int(num_slices)
        self.sw_p = float(sw_p)
        self.temperature = float(temperature)
        self.token_weight_uniform_mix = float(token_weight_uniform_mix)
        self.num_quantiles = int(num_quantiles)
        self.normalize_tokens = bool(normalize_tokens)
        self.train_projection_mode = str(train_projection_mode)
        self.eval_projection_mode = str(eval_projection_mode)
        self.projection_seed = int(projection_seed)
        self.shot_aggregation = str(shot_aggregation)
        self.shot_softmin_beta = float(shot_softmin_beta)
        self.eps = float(eps)
        self.token_projector = nn.Sequential(
            nn.LayerNorm(int(hidden_dim)),
            nn.Linear(int(hidden_dim), self.token_dim, bias=False),
        )
        self.token_weighter = nn.Sequential(
            nn.LayerNorm(self.token_dim),
            nn.Linear(self.token_dim, int(token_weight_hidden_dim)),
            nn.GELU(),
            nn.Linear(int(token_weight_hidden_dim), 1),
        )
        self.task_adaptive = TaskAdaptiveSlicing(
            feat_dim=self.token_dim,
            hidden_dim=int(task_hidden_dim),
            eps=float(eps),
        )
        proto_raw = torch.tensor(_inverse_softplus(float(proto_scale_init)), dtype=torch.float32)
        sw_raw = torch.tensor(_inverse_softplus(float(sw_scale_init)), dtype=torch.float32)
        if learnable_scales:
            self.raw_proto_scale = nn.Parameter(proto_raw)
            self.raw_sw_scale = nn.Parameter(sw_raw)
        else:
            self.register_buffer("raw_proto_scale", proto_raw)
            self.register_buffer("raw_sw_scale", sw_raw)
        self.register_buffer("_fixed_projection_bank", torch.empty(0), persistent=False)
        self.register_buffer("_resample_counter", torch.zeros((), dtype=torch.long), persistent=False)
        self._fixed_projection_key: tuple[int, int] | None = None

    @property
    def proto_scale(self) -> torch.Tensor:
        return F.softplus(self.raw_proto_scale).clamp_min(self.eps)

    @property
    def sw_scale(self) -> torch.Tensor:
        return F.softplus(self.raw_sw_scale).clamp_min(self.eps)

    def _encode_token_set(self, images: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        feature_map = self.encode(images)
        spatial_hw = (feature_map.shape[-2], feature_map.shape[-1])
        tokens = self.token_projector(feature_map_to_tokens(feature_map))
        if self.normalize_tokens:
            tokens = F.normalize(tokens, p=2, dim=-1, eps=self.eps)
        return tokens, spatial_hw

    def _compute_token_weights(self, tokens: torch.Tensor) -> torch.Tensor:
        logits = self.token_weighter(tokens).squeeze(-1)
        learned = torch.softmax(logits, dim=-1)
        if self.token_weight_uniform_mix == 0.0:
            return learned
        uniform = torch.full_like(learned, 1.0 / float(learned.shape[-1]))
        return (
            self.token_weight_uniform_mix * uniform
            + (1.0 - self.token_weight_uniform_mix) * learned
        )

    @staticmethod
    def _weighted_pool(tokens: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.sum(tokens * weights.unsqueeze(-1), dim=-2)

    def _get_fixed_projection_bank(
        self,
        feature_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        key = (int(feature_dim), self.num_slices)
        if self._fixed_projection_key != key or self._fixed_projection_bank.numel() == 0:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.projection_seed)
            bank = torch.randn(int(feature_dim), self.num_slices, generator=generator)
            bank = F.normalize(bank, p=2, dim=0).transpose(0, 1).contiguous()
            self._fixed_projection_bank = bank
            self._fixed_projection_key = key
        return self._fixed_projection_bank.to(device=device, dtype=dtype)

    def _sample_resampled_projection_bank(
        self,
        feature_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        call_idx = int(self._resample_counter.item())
        self._resample_counter.add_(1)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.projection_seed + 1_000_003 * call_idx)
        bank = torch.randn(int(feature_dim), self.num_slices, generator=generator)
        bank = F.normalize(bank, p=2, dim=0).transpose(0, 1).contiguous()
        return bank.to(device=device, dtype=dtype)

    def _deterministic_task_thetas(self, mean_dir: torch.Tensor, concentration: torch.Tensor) -> torch.Tensor:
        base = self._get_fixed_projection_bank(mean_dir.shape[0], mean_dir.device, mean_dir.dtype)
        dim = mean_dir.new_tensor(float(mean_dir.shape[0]))
        alignment = (concentration / (concentration + dim)).clamp(min=0.0, max=0.95)
        mixed = (1.0 - alignment) * base + alignment * mean_dir.unsqueeze(0)
        return F.normalize(mixed, p=2, dim=1, eps=self.eps)

    def _projection_directions(self, mean_dir: torch.Tensor, concentration: torch.Tensor) -> torch.Tensor:
        mode = self.train_projection_mode if self.training else self.eval_projection_mode
        if mode == "vmf":
            return sample_vmf(mean_dir, concentration, self.num_slices, eps=self.eps)
        if mode == "resample":
            base = self._sample_resampled_projection_bank(mean_dir.shape[0], mean_dir.device, mean_dir.dtype)
            dim = mean_dir.new_tensor(float(mean_dir.shape[0]))
            alignment = (concentration / (concentration + dim)).clamp(min=0.0, max=0.95)
            return F.normalize((1.0 - alignment) * base + alignment * mean_dir.unsqueeze(0), p=2, dim=1, eps=self.eps)
        return self._deterministic_task_thetas(mean_dir, concentration)

    def _prototype_logits(
        self,
        query_tokens: torch.Tensor,
        query_weights: torch.Tensor,
        class_tokens: torch.Tensor,
        class_weights: torch.Tensor,
    ) -> torch.Tensor:
        query_proto = F.normalize(self._weighted_pool(query_tokens, query_weights), p=2, dim=-1, eps=self.eps)
        class_proto = F.normalize(self._weighted_pool(class_tokens, class_weights), p=2, dim=-1, eps=self.eps)
        return torch.matmul(query_proto, class_proto.transpose(0, 1))

    def _shot_aware_distances(
        self,
        query_tokens: torch.Tensor,
        query_weights: torch.Tensor,
        support_tokens_by_shot: torch.Tensor,
        support_weights_by_shot: torch.Tensor,
        thetas: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        way_num, shot_num = support_tokens_by_shot.shape[:2]
        flat_support_tokens = support_tokens_by_shot.reshape(
            way_num * shot_num,
            support_tokens_by_shot.shape[-2],
            support_tokens_by_shot.shape[-1],
        )
        flat_support_weights = support_weights_by_shot.reshape(way_num * shot_num, support_weights_by_shot.shape[-1])
        shot_distances = pairwise_weighted_sliced_wasserstein_distance(
            query_tokens=query_tokens,
            class_tokens=flat_support_tokens,
            query_weights=query_weights,
            class_weights=flat_support_weights,
            thetas=thetas,
            p=self.sw_p,
            eps=self.eps,
        ).reshape(query_tokens.shape[0], way_num, shot_num)
        if shot_num == 1:
            return shot_distances.squeeze(-1), shot_distances
        if self.shot_aggregation == "mean":
            return shot_distances.mean(dim=-1), shot_distances
        if self.shot_aggregation == "softmin":
            weights = torch.softmax(-float(self.shot_softmin_beta) * shot_distances, dim=-1)
            return (weights * shot_distances).sum(dim=-1), shot_distances
        return shot_distances.mean(dim=-1), shot_distances

    @staticmethod
    def _stack_auxiliary(batch_outputs: list[dict[str, Any]]) -> TADSWResult:
        stacked: dict[str, torch.Tensor] = {
            "logits": torch.cat([item["logits"] for item in batch_outputs], dim=0),
            "aux_loss": torch.stack([item["aux_loss"] for item in batch_outputs]).mean(),
            "class_scores": torch.cat([item["class_scores"] for item in batch_outputs], dim=0),
            "total_distance": torch.cat([item["total_distance"] for item in batch_outputs], dim=0),
            "ta_dsw_distance": torch.cat([item["ta_dsw_distance"] for item in batch_outputs], dim=0),
            "ta_dsw_concentration": torch.stack(
                [item["ta_dsw_concentration"] for item in batch_outputs],
                dim=0,
            ),
            "ta_dsw_mean_dir": torch.stack([item["ta_dsw_mean_dir"] for item in batch_outputs], dim=0),
            "ta_dsw_thetas": torch.stack([item["ta_dsw_thetas"] for item in batch_outputs], dim=0),
            "ta_dsw_temperature": torch.stack(
                [item["ta_dsw_temperature"] for item in batch_outputs],
                dim=0,
            ).mean(),
            "ta_dsw_proto_scale": torch.stack([item["ta_dsw_proto_scale"] for item in batch_outputs]).mean(),
            "ta_dsw_sw_scale": torch.stack([item["ta_dsw_sw_scale"] for item in batch_outputs]).mean(),
            "ta_dsw_proto_logits": torch.cat([item["ta_dsw_proto_logits"] for item in batch_outputs], dim=0),
            "ta_dsw_sw_logits": torch.cat([item["ta_dsw_sw_logits"] for item in batch_outputs], dim=0),
        }
        if "ta_dsw_shot_distance" in batch_outputs[0]:
            stacked["ta_dsw_shot_distance"] = torch.cat(
                [item["ta_dsw_shot_distance"] for item in batch_outputs],
                dim=0,
            )
        if "query_tokens" in batch_outputs[0]:
            stacked["query_tokens"] = torch.cat([item["query_tokens"] for item in batch_outputs], dim=0)
            stacked["support_tokens"] = torch.stack([item["support_tokens"] for item in batch_outputs], dim=0)
            stacked["query_token_weights"] = torch.cat([item["query_token_weights"] for item in batch_outputs], dim=0)
            stacked["support_token_weights"] = torch.stack(
                [item["support_token_weights"] for item in batch_outputs],
                dim=0,
            )
        return TADSWResult(stacked)

    def _forward_episode(
        self,
        query: torch.Tensor,
        support: torch.Tensor,
        return_aux: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        way_num, shot_num = support.shape[:2]
        query_tokens, query_hw = self._encode_token_set(query)
        support_tokens, support_hw = self._encode_token_set(
            support.reshape(way_num * shot_num, *support.shape[-3:])
        )
        if query_hw != support_hw:
            raise ValueError(f"Query/support token grids must match, got {query_hw} vs {support_hw}")

        token_num = support_tokens.shape[-2]
        query_weights = self._compute_token_weights(query_tokens)
        support_weights = self._compute_token_weights(support_tokens)
        support_tokens_by_shot = support_tokens.reshape(way_num, shot_num, token_num, support_tokens.shape[-1])
        support_weights_by_shot = support_weights.reshape(way_num, shot_num, token_num)
        class_tokens = support_tokens.reshape(way_num, shot_num * token_num, support_tokens.shape[-1])
        class_weights = support_weights.reshape(way_num, shot_num * token_num)
        class_weights = _normalize_measure_weights(class_weights, self.eps)
        mean_dir, concentration = self.task_adaptive(class_tokens.reshape(-1, class_tokens.shape[-1]))
        thetas = self._projection_directions(mean_dir, concentration)
        if self.shot_aggregation == "concat" or shot_num == 1:
            distances = pairwise_weighted_sliced_wasserstein_distance(
                query_tokens=query_tokens,
                class_tokens=class_tokens,
                query_weights=query_weights,
                class_weights=class_weights,
                thetas=thetas,
                p=self.sw_p,
                eps=self.eps,
            )
            shot_distances = distances.unsqueeze(-1)
        else:
            distances, shot_distances = self._shot_aware_distances(
                query_tokens=query_tokens,
                query_weights=query_weights,
                support_tokens_by_shot=support_tokens_by_shot,
                support_weights_by_shot=support_weights_by_shot,
                thetas=thetas,
            )
        proto_logits = self._prototype_logits(query_tokens, query_weights, class_tokens, class_weights)
        sw_logits = -self.sw_scale.to(device=distances.device, dtype=distances.dtype) * distances
        logits = self.proto_scale.to(device=proto_logits.device, dtype=proto_logits.dtype) * proto_logits + sw_logits

        if not return_aux:
            return logits

        zero = logits.new_zeros(())
        return {
            "logits": logits,
            "aux_loss": zero,
            "class_scores": logits,
            "total_distance": distances,
            "ta_dsw_distance": distances,
            "ta_dsw_shot_distance": shot_distances,
            "ta_dsw_concentration": concentration,
            "ta_dsw_mean_dir": mean_dir,
            "ta_dsw_thetas": thetas,
            "ta_dsw_temperature": logits.new_tensor(float(self.temperature)),
            "ta_dsw_proto_scale": self.proto_scale.to(device=logits.device, dtype=logits.dtype),
            "ta_dsw_sw_scale": self.sw_scale.to(device=logits.device, dtype=logits.dtype),
            "ta_dsw_proto_logits": proto_logits,
            "ta_dsw_sw_logits": sw_logits,
            "query_tokens": query_tokens,
            "support_tokens": class_tokens,
            "query_token_weights": query_weights,
            "support_token_weights": class_weights,
        }

    def forward(
        self,
        query: torch.Tensor,
        support: torch.Tensor,
        *,
        query_targets: torch.Tensor | None = None,
        support_targets: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        del query_targets, support_targets
        batch_size, _, _, _, _, _ = self.validate_episode_inputs(query, support)
        needs_payload = bool(return_aux or self.training)
        batch_outputs = []
        batch_logits = []

        for batch_idx in range(batch_size):
            outputs = self._forward_episode(
                query=query[batch_idx],
                support=support[batch_idx],
                return_aux=needs_payload,
            )
            if needs_payload:
                batch_outputs.append(outputs)
                batch_logits.append(outputs["logits"])
            else:
                batch_logits.append(outputs)

        logits = torch.cat(batch_logits, dim=0)
        if not needs_payload:
            return logits

        stacked = self._stack_auxiliary(batch_outputs)
        stacked["logits"] = logits
        if return_aux:
            return stacked
        return TADSWResult({"logits": logits, "aux_loss": stacked["aux_loss"]})


TADSW = TaskAdaptiveDistributionalSlicedWassersteinNet


__all__ = [
    "TADSW",
    "TADSWResult",
    "TaskAdaptiveDistributionalSlicedWassersteinNet",
    "TaskAdaptiveSlicing",
    "pairwise_sliced_wasserstein_distance",
    "pairwise_weighted_sliced_wasserstein_distance",
    "sample_vmf",
    "sliced_wasserstein_distance",
]
