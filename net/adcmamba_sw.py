"""ADC-Mamba-SW: old ADC-HOT with official Mamba tokens and standard SW distance.

Design goal:
1. Keep the old ADC-HOT inductive bias: EMA class priors + calibrated diagonal Gaussian scoring.
2. Replace plain pooled image embeddings with an official Mamba token encoder.
3. Use sliced Wasserstein as a direct distribution distance, not as a control variate.

Let Z(x) in R^(T x d) be Mamba-refined spatial tokens for image x and h(x) = mean_t Z_t(x).
For each class c in the support set:

    mu_s[c] = mean_i h(x_i)
    var_s[c] = Var_i h(x_i)
    P_s[c] = empirical token distribution from the class token prototype

Running class priors keep the old ADC-HOT flavor:

    mu_b[c] <- (1 - eta) mu_b[c] + eta mu_s[c]
    var_b[c] <- (1 - eta) var_b[c] + eta (var_s[c] + alpha)
    P_b[c] <- (1 - eta) P_b[c] + eta P_s[c]

We estimate prior trust from the sliced Wasserstein distance:

    d_sw[c] = SW(P_s[c], P_b[c])
    rho[c]  = (1 - lambda) * exp(-tau * d_sw[c])

Then the calibrated class statistics are:

    mu_hat[c]  = (1 - rho[c]) mu_s[c] + rho[c] mu_b[c]
    var_hat[c] = var_s[c] + rho[c] var_b[c] + alpha

The query score stays close to the old ADC-HOT implementation, with one additional
distribution penalty on query tokens:

    score(q, c) = -E_gauss(h(q); mu_hat[c], var_hat[c]) - beta_sw * SW(P_q, P_mix[c])

where P_mix[c] = (1 - rho[c]) P_s[c] + rho[c] P_b[c].
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from net.encoders.smnet_conv64f_encoder import build_resnet12_family_encoder

try:
    from mamba_ssm import Mamba
except ImportError:  # pragma: no cover - dependency is optional in the current env
    Mamba = None

try:
    import ot
except ImportError:  # pragma: no cover - dependency is optional in the current env
    ot = None


def _sanitize_tensor(
    x: torch.Tensor,
    *,
    nan: float = 0.0,
    posinf: float = 1e4,
    neginf: float = -1e4,
) -> torch.Tensor:
    """Keep tensors finite so one unstable episode does not poison the whole run."""

    return torch.nan_to_num(x, nan=nan, posinf=posinf, neginf=neginf)


def feature_map_to_tokens(feature_map: torch.Tensor) -> torch.Tensor:
    if feature_map.dim() != 4:
        raise ValueError(f"Expected feature_map with shape (N, C, H, W), got {tuple(feature_map.shape)}")
    return feature_map.flatten(2).transpose(1, 2).contiguous()


class ResidualMambaBlock(nn.Module):
    """Pre-norm residual block around the official Mamba layer."""

    def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int) -> None:
        super().__init__()
        if Mamba is None:
            raise ImportError(
                "mamba_ssm is required for ADCMambaSWNet. Install it in your conda env, "
                "for example: pip install mamba-ssm"
            )
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = _sanitize_tensor(x)
        mamba_out = self.mamba(self.norm(residual))
        return _sanitize_tensor(residual + _sanitize_tensor(mamba_out))


class MambaTokenEncoder(nn.Module):
    """Encode spatial tokens with the official Mamba implementation."""

    def __init__(
        self,
        in_dim: int,
        token_dim: int,
        depth: int,
        d_state: int,
        d_conv: int,
        expand: int,
    ) -> None:
        super().__init__()
        self.in_proj = nn.Linear(in_dim, token_dim)
        self.in_norm = nn.LayerNorm(token_dim)
        self.layers = nn.ModuleList(
            [
                ResidualMambaBlock(
                    d_model=token_dim,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                )
                for _ in range(depth)
            ]
        )
        self.out_norm = nn.LayerNorm(token_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        hidden = _sanitize_tensor(self.in_norm(self.in_proj(tokens)))
        for layer in self.layers:
            hidden = layer(hidden)
        return _sanitize_tensor(self.out_norm(hidden))


class POTSlicedWassersteinDistance(nn.Module):
    """Standard sliced Wasserstein distance via POT."""

    def __init__(
        self,
        num_projections: int = 64,
        p: float = 2.0,
        normalize_inputs: bool = True,
        seed: int = 7,
    ) -> None:
        super().__init__()
        if num_projections <= 0:
            raise ValueError("num_projections must be positive")
        self.num_projections = int(num_projections)
        self.p = float(p)
        self.normalize_inputs = bool(normalize_inputs)
        self.seed = int(seed)
        if ot is None:
            raise ImportError(
                "POT is required for ADCMambaSWNet. Install it in your conda env, "
                "for example: pip install POT"
            )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.dim() < 3 or y.dim() < 3:
            raise ValueError(f"Expected token tensors (..., T, D), got x={tuple(x.shape)} y={tuple(y.shape)}")
        if x.shape[:-2] != y.shape[:-2]:
            raise ValueError(
                "Leading dimensions must match for SW distance: "
                f"x={tuple(x.shape)} y={tuple(y.shape)}"
            )
        if x.shape[-1] != y.shape[-1]:
            raise ValueError(
                "Feature dimensions must match for SW distance: "
                f"x={x.shape[-1]} y={y.shape[-1]}"
            )

        if self.normalize_inputs:
            x = F.normalize(_sanitize_tensor(x), p=2, dim=-1, eps=1e-6)
            y = F.normalize(_sanitize_tensor(y), p=2, dim=-1, eps=1e-6)
        else:
            x = _sanitize_tensor(x)
            y = _sanitize_tensor(y)

        leading_shape = x.shape[:-2]
        x_flat = x.reshape(-1, x.shape[-2], x.shape[-1])
        y_flat = y.reshape(-1, y.shape[-2], y.shape[-1])

        distances = []
        for x_item, y_item in zip(x_flat, y_flat):
            dist = ot.sliced_wasserstein_distance(
                x_item,
                y_item,
                n_projections=self.num_projections,
                p=self.p,
                seed=self.seed,
            )
            if not torch.is_tensor(dist):
                dist = x_item.new_tensor(float(dist))
            dist = dist.to(device=x_item.device, dtype=x_item.dtype)
            distances.append(_sanitize_tensor(dist, nan=1e4, posinf=1e4, neginf=0.0))

        return torch.stack(distances, dim=0).reshape(*leading_shape)


class ADCMambaSWNet(nn.Module):
    """Old ADC-HOT prior calibration upgraded with official Mamba tokens and SW distance."""

    def __init__(
        self,
        image_size: int = 64,
        way_num: int = 4,
        token_dim: int = 128,
        state_dim: int = 16,
        mamba_depth: int = 2,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        alpha: float = 0.21,
        lambd: float = 0.3,
        ema: float = 0.05,
        eps: float = 1e-6,
        sw_num_projections: int = 64,
        sw_weight: float = 1.0,
        transport_temperature: float = 6.0,
        fewshot_backbone: str = "resnet12",
        device: str = "cuda",
    ) -> None:
        super().__init__()
        self.encoder = build_resnet12_family_encoder(
            image_size=image_size,
            backbone_name=fewshot_backbone,
            pool_output=False,
            variant="fewshot",
            drop_rate=0.0,
        )
        self.way_num = int(way_num)
        self.feat_dim = int(self.encoder.out_channels)
        self.token_dim = int(token_dim)
        self.alpha = float(alpha)
        self.lambd = float(lambd)
        self.ema = float(ema)
        self.eps = float(eps)
        self.sw_weight = float(sw_weight)
        self.transport_temperature = float(transport_temperature)
        self.token_count = int(self.encoder.out_spatial * self.encoder.out_spatial)

        self.token_encoder = MambaTokenEncoder(
            in_dim=self.feat_dim,
            token_dim=self.token_dim,
            depth=int(mamba_depth),
            d_state=int(state_dim),
            d_conv=int(mamba_d_conv),
            expand=int(mamba_expand),
        )
        self.global_norm = nn.LayerNorm(self.token_dim)
        self.output_scale = nn.Parameter(torch.tensor(1.0))

        self.sw_distance = POTSlicedWassersteinDistance(
            num_projections=sw_num_projections,
            p=2.0,
            normalize_inputs=True,
            seed=7,
        )

        self.register_buffer("base_means", torch.zeros(self.way_num, self.token_dim))
        self.register_buffer("base_vars", torch.ones(self.way_num, self.token_dim))
        self.register_buffer("base_tokens", torch.zeros(self.way_num, self.token_count, self.token_dim))
        self.register_buffer("base_initialized", torch.zeros(self.way_num, dtype=torch.bool))
        self.to(device)

    def _encode_image(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feature_map = self.encoder.forward_features(x)
        tokens = feature_map_to_tokens(feature_map)
        tokens = self.token_encoder(tokens)
        tokens = _sanitize_tensor(tokens)
        pooled = _sanitize_tensor(self.global_norm(tokens.mean(dim=1)))
        return pooled, tokens

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        pooled, _ = self._encode_image(x)
        return pooled

    @torch.no_grad()
    def _update_base_state(
        self,
        support_means: torch.Tensor,
        support_vars: torch.Tensor,
        support_token_proto: torch.Tensor,
    ) -> None:
        for class_idx in range(self.way_num):
            if (
                not torch.isfinite(support_means[class_idx]).all()
                or not torch.isfinite(support_vars[class_idx]).all()
                or not torch.isfinite(support_token_proto[class_idx]).all()
            ):
                continue
            if not self.base_initialized[class_idx]:
                self.base_means[class_idx] = support_means[class_idx]
                self.base_vars[class_idx] = support_vars[class_idx] + self.alpha
                self.base_tokens[class_idx] = support_token_proto[class_idx]
                self.base_initialized[class_idx] = True
                continue
            self.base_means[class_idx].mul_(1.0 - self.ema).add_(self.ema * support_means[class_idx])
            self.base_vars[class_idx].mul_(1.0 - self.ema).add_(self.ema * (support_vars[class_idx] + self.alpha))
            self.base_tokens[class_idx].mul_(1.0 - self.ema).add_(self.ema * support_token_proto[class_idx])

    def _base_reference(
        self,
        class_idx: int,
        support_mean: torch.Tensor,
        support_var: torch.Tensor,
        support_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if bool(self.base_initialized[class_idx].item()):
            return (
                _sanitize_tensor(self.base_means[class_idx]),
                _sanitize_tensor(self.base_vars[class_idx], nan=self.alpha, posinf=1e4, neginf=self.alpha),
                _sanitize_tensor(self.base_tokens[class_idx]),
            )
        return (
            _sanitize_tensor(support_mean.detach()),
            _sanitize_tensor(support_var.detach() + self.alpha, nan=self.alpha, posinf=1e4, neginf=self.alpha),
            _sanitize_tensor(support_tokens.detach()),
        )

    def _prior_weight(self, support_tokens: torch.Tensor, base_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        support_prior_sw = self.sw_distance(support_tokens.unsqueeze(0), base_tokens.unsqueeze(0)).squeeze(0)
        prior_weight = (1.0 - self.lambd) * torch.exp(-self.transport_temperature * support_prior_sw)
        prior_weight = _sanitize_tensor(prior_weight, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        support_prior_sw = _sanitize_tensor(support_prior_sw, nan=1e4, posinf=1e4, neginf=0.0)
        return prior_weight, support_prior_sw

    def forward(self, query: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        batch_size, num_query, channels, height, width = query.size()
        _, way_num, shot_num, _, _, _ = support.size()
        if way_num != self.way_num:
            raise ValueError(f"ADCMambaSWNet initialized for way_num={self.way_num}, got {way_num}")

        query_embed, query_tokens = self._encode_image(query.view(-1, channels, height, width))
        support_embed, support_tokens = self._encode_image(support.view(-1, channels, height, width))

        query_embed = _sanitize_tensor(query_embed.view(batch_size, num_query, self.token_dim))
        query_tokens = _sanitize_tensor(query_tokens.view(batch_size, num_query, self.token_count, self.token_dim))

        support_embed = _sanitize_tensor(support_embed.view(batch_size, way_num, shot_num, self.token_dim))
        support_tokens = _sanitize_tensor(
            support_tokens.view(batch_size, way_num, shot_num, self.token_count, self.token_dim)
        )

        all_scores = []
        for batch_idx in range(batch_size):
            support_means = _sanitize_tensor(support_embed[batch_idx].mean(dim=1))
            if shot_num > 1:
                support_vars = _sanitize_tensor(support_embed[batch_idx].var(dim=1, unbiased=False))
            else:
                support_vars = torch.zeros_like(support_means)
            support_token_proto = _sanitize_tensor(support_tokens[batch_idx].mean(dim=1))

            if self.training:
                self._update_base_state(
                    support_means.detach(),
                    support_vars.detach(),
                    support_token_proto.detach(),
                )

            class_scores = []
            for class_idx in range(way_num):
                base_mean, base_var, base_tokens = self._base_reference(
                    class_idx,
                    support_means[class_idx],
                    support_vars[class_idx],
                    support_token_proto[class_idx],
                )
                prior_weight, _ = self._prior_weight(support_token_proto[class_idx], base_tokens)
                support_weight = 1.0 - prior_weight

                calibrated_mean = _sanitize_tensor(support_weight * support_means[class_idx] + prior_weight * base_mean)
                calibrated_var = support_vars[class_idx] + prior_weight * base_var + self.alpha
                calibrated_var = _sanitize_tensor(
                    calibrated_var,
                    nan=self.alpha,
                    posinf=1e4,
                    neginf=self.alpha,
                ).clamp_min(self.eps)
                mixed_tokens = _sanitize_tensor(support_weight * support_token_proto[class_idx] + prior_weight * base_tokens)

                diff = _sanitize_tensor(query_embed[batch_idx] - calibrated_mean.unsqueeze(0))
                mahal = _sanitize_tensor((diff.square() / calibrated_var.unsqueeze(0)).sum(dim=-1), nan=1e4, posinf=1e4)
                log_det = torch.log(calibrated_var).sum().expand_as(mahal)
                gaussian_score = _sanitize_tensor(-(mahal + log_det), nan=-1e4, posinf=1e4, neginf=-1e4)

                mixed_dist = mixed_tokens.unsqueeze(0).expand(num_query, -1, -1)
                query_sw = self.sw_distance(query_tokens[batch_idx], mixed_dist)
                scale = torch.clamp(self.output_scale, min=1e-4, max=100.0)
                score = scale * (gaussian_score - self.sw_weight * query_sw)
                score = _sanitize_tensor(score, nan=-1e4, posinf=1e4, neginf=-1e4)
                class_scores.append(score)

            all_scores.append(torch.stack(class_scores, dim=1))

        return torch.cat(all_scores, dim=0)
