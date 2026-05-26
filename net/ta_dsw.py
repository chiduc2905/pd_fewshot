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
            nn.Linear(int(feat_dim), int(hidden_dim)),
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
        task_descriptor = all_support_features.mean(dim=0)
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
        backbone_name: str = "resnet12",
        image_size: int = 64,
        resnet12_drop_rate: float = 0.0,
        resnet12_dropblock_size: int = 5,
        num_slices: int = 64,
        sw_p: float = 2.0,
        temperature: float = 0.1,
        task_hidden_dim: int = 256,
        num_quantiles: int = 256,
        normalize_tokens: bool = True,
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
        if sw_p <= 0.0:
            raise ValueError("sw_p must be positive")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if num_quantiles <= 0:
            raise ValueError("num_quantiles must be positive")
        if eps <= 0.0:
            raise ValueError("eps must be positive")

        self.num_slices = int(num_slices)
        self.sw_p = float(sw_p)
        self.temperature = float(temperature)
        self.num_quantiles = int(num_quantiles)
        self.normalize_tokens = bool(normalize_tokens)
        self.eps = float(eps)
        self.task_adaptive = TaskAdaptiveSlicing(
            feat_dim=int(hidden_dim),
            hidden_dim=int(task_hidden_dim),
            eps=float(eps),
        )

    def _encode_token_set(self, images: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        feature_map = self.encode(images)
        spatial_hw = (feature_map.shape[-2], feature_map.shape[-1])
        tokens = feature_map_to_tokens(feature_map)
        if self.normalize_tokens:
            tokens = F.normalize(tokens, p=2, dim=-1, eps=self.eps)
        return tokens, spatial_hw

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
        }
        if "query_tokens" in batch_outputs[0]:
            stacked["query_tokens"] = torch.cat([item["query_tokens"] for item in batch_outputs], dim=0)
            stacked["support_tokens"] = torch.stack([item["support_tokens"] for item in batch_outputs], dim=0)
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
        class_tokens = support_tokens.reshape(way_num, shot_num * token_num, support_tokens.shape[-1])
        mean_dir, concentration = self.task_adaptive(class_tokens.reshape(-1, class_tokens.shape[-1]))
        thetas = sample_vmf(mean_dir, concentration, self.num_slices, eps=self.eps)
        distances = pairwise_sliced_wasserstein_distance(
            query_tokens=query_tokens,
            class_tokens=class_tokens,
            thetas=thetas,
            p=self.sw_p,
            num_quantiles=self.num_quantiles,
        )
        logits = -distances / float(self.temperature)

        if not return_aux:
            return logits

        zero = logits.new_zeros(())
        return {
            "logits": logits,
            "aux_loss": zero,
            "class_scores": logits,
            "total_distance": distances,
            "ta_dsw_distance": distances,
            "ta_dsw_concentration": concentration,
            "ta_dsw_mean_dir": mean_dir,
            "ta_dsw_thetas": thetas,
            "ta_dsw_temperature": logits.new_tensor(float(self.temperature)),
            "query_tokens": query_tokens,
            "support_tokens": class_tokens,
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
    "sample_vmf",
    "sliced_wasserstein_distance",
]
