"""Slim-Mamba few-shot encoder with dual local/global outputs."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _hw_to_tokens(x: torch.Tensor) -> torch.Tensor:
    return x.flatten(2).transpose(1, 2).contiguous()


def _resolve_group_count(num_channels: int, max_groups: int = 32) -> int:
    groups = min(max_groups, int(num_channels))
    while groups > 1 and num_channels % groups != 0:
        groups //= 2
    return max(groups, 1)


class EpisodicGN(nn.GroupNorm):
    """GroupNorm with stable grouping for low-batch episodic training."""

    def __init__(self, num_channels: int, eps: float = 1e-4) -> None:
        super().__init__(_resolve_group_count(num_channels), num_channels, eps=eps)


class EpisodicLayerNorm(nn.Module):
    """LayerNorm over the last dimension for token tensors."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, (x.shape[-1],), self.weight, self.bias, self.eps)


class DropPath(nn.Module):
    """Per-sample stochastic depth."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob <= 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.bernoulli(torch.full(shape, keep_prob, device=x.device, dtype=x.dtype))
        return x * (mask / keep_prob)


class LightS6(nn.Module):
    """Compact selective state-space block operating on token sequences."""

    def __init__(self, dim: int, d_state: int = 8) -> None:
        super().__init__()
        if int(dim) <= 0:
            raise ValueError("dim must be positive")
        if int(d_state) <= 0:
            raise ValueError("d_state must be positive")

        self.dim = int(dim)
        self.d_state = int(d_state)
        dt_rank = max(self.dim // 8, 4)

        self.x_proj = nn.Linear(self.dim, dt_rank + 2 * self.d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, self.dim, bias=True)

        a = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.dim, 1)
        self.A_log = nn.Parameter(torch.log(a))
        self.D = nn.Parameter(torch.ones(self.dim))

        nn.init.constant_(self.dt_proj.bias, math.log(math.expm1(0.1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = x.shape
        if dim != self.dim:
            raise ValueError(f"Expected dim={self.dim}, got {dim}")

        dt_rank = self.dt_proj.in_features
        projected = self.x_proj(x)
        dt_raw, b_state, c_state = projected.split([dt_rank, self.d_state, self.d_state], dim=-1)

        dt = F.silu(self.dt_proj(dt_raw))
        a = -torch.exp(self.A_log.float())
        dA = torch.exp(dt.unsqueeze(-1) * a)
        dB = dt.unsqueeze(-1) * b_state.unsqueeze(2)

        state = torch.zeros(batch_size, dim, self.d_state, device=x.device, dtype=x.dtype)
        outputs = []
        for token_idx in range(seq_len):
            state = dA[:, token_idx] * state + dB[:, token_idx] * x[:, token_idx].unsqueeze(-1)
            y = (state * c_state[:, token_idx].unsqueeze(1)).sum(dim=-1)
            outputs.append(y)

        y = torch.stack(outputs, dim=1)
        return y + x * self.D


def window_partition(x: torch.Tensor, window_size: int = 2) -> torch.Tensor:
    """Convert [B, H, W, C] to [B * num_windows, window_size^2, C]."""

    batch_size, height, width, channels = x.shape
    x = x.reshape(
        batch_size,
        height // window_size,
        window_size,
        width // window_size,
        window_size,
        channels,
    )
    return x.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size * window_size, channels)


def window_unpartition(
    x: torch.Tensor,
    window_size: int,
    height: int,
    width: int,
    batch_size: int,
) -> torch.Tensor:
    """Convert [B * num_windows, window_size^2, C] back to [B, H, W, C]."""

    channels = x.shape[-1]
    num_h = height // window_size
    num_w = width // window_size
    x = x.reshape(batch_size, num_h, num_w, window_size, window_size, channels)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(batch_size, height, width, channels)


class HFFVSSBlock(nn.Module):
    """High-frequency fusion state-space block in channel-last layout."""

    def __init__(
        self,
        dim: int,
        d_state: int = 8,
        expand: float = 1.5,
        window_size: int = 2,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        if int(dim) <= 0:
            raise ValueError("dim must be positive")
        if float(expand) <= 0.0:
            raise ValueError("expand must be positive")
        if int(window_size) <= 0:
            raise ValueError("window_size must be positive")

        self.window_size = int(window_size)
        inner_dim = max(int(dim * expand), int(dim))

        self.norm = nn.LayerNorm(dim)
        self.in_proj = nn.Linear(dim, inner_dim * 2, bias=False)

        self.ssm_fwd = LightS6(inner_dim, d_state=d_state)
        self.ssm_bwd = LightS6(inner_dim, d_state=d_state)

        self.dw_conv = nn.Conv2d(inner_dim, inner_dim, kernel_size=3, padding=1, groups=inner_dim, bias=False)
        self.pw_conv = nn.Conv2d(inner_dim, inner_dim, kernel_size=1, bias=False)

        self.gate = nn.Parameter(torch.zeros(inner_dim))

        self.out_proj = nn.Linear(inner_dim, dim, bias=False)
        self.norm2 = nn.LayerNorm(dim)
        self.fuse = nn.Sequential(
            nn.Linear(dim, dim * 2, bias=True),
            nn.GELU(),
            nn.Linear(dim * 2, dim, bias=True),
        )
        self.drop = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, height, width, _ = x.shape
        shortcut = x
        x_norm = self.norm(x)
        x_proj, z = self.in_proj(x_norm).chunk(2, dim=-1)

        window_size = min(self.window_size, height, width)
        if window_size > 0 and height % window_size == 0 and width % window_size == 0:
            windows = window_partition(x_proj, window_size=window_size)
            y_fwd = self.ssm_fwd(windows)
            y_bwd = self.ssm_bwd(windows.flip(1)).flip(1)
            y_ssm = window_unpartition(
                0.5 * (y_fwd + y_bwd),
                window_size=window_size,
                height=height,
                width=width,
                batch_size=batch_size,
            )
        else:
            sequence = x_proj.reshape(batch_size, height * width, -1)
            y_ssm = 0.5 * (self.ssm_fwd(sequence) + self.ssm_bwd(sequence.flip(1)).flip(1))
            y_ssm = y_ssm.reshape(batch_size, height, width, -1)

        conv_input = x_proj.permute(0, 3, 1, 2).contiguous()
        y_conv = F.gelu(self.pw_conv(self.dw_conv(conv_input)))
        y_conv = y_conv.permute(0, 2, 3, 1).contiguous()

        alpha = torch.sigmoid(self.gate).view(1, 1, 1, -1)
        y = alpha * y_ssm + (1.0 - alpha) * y_conv
        y = self.out_proj(y * F.silu(z))

        x = shortcut + self.drop(y)
        x = x + self.drop(self.fuse(self.norm2(x)))
        return x


class ConvBlock(nn.Module):
    """ResNet-style residual conv block with GroupNorm and max-pooling."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.gn1 = EpisodicGN(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.gn2 = EpisodicGN(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.gn3 = EpisodicGN(out_channels)

        self.skip = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                EpisodicGN(out_channels),
            )
            if in_channels != out_channels
            else nn.Identity()
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = self.skip(x)
        x = F.leaky_relu(self.gn1(self.conv1(x)), negative_slope=0.1, inplace=False)
        x = F.leaky_relu(self.gn2(self.conv2(x)), negative_slope=0.1, inplace=False)
        x = F.leaky_relu(self.gn3(self.conv3(x)), negative_slope=0.1, inplace=False)
        x = F.leaky_relu(x + shortcut, negative_slope=0.1, inplace=False)
        return self.pool(x)


class FeaturePerturbation(nn.Module):
    """Training-time Gaussian perturbation for feature-map regularization."""

    def __init__(self, sigma: float = 0.05) -> None:
        super().__init__()
        self.sigma = float(sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.sigma <= 0.0:
            return x
        return x + torch.randn_like(x) * self.sigma


def _stage4_spatial(image_size: int) -> int:
    spatial = max(1, int(image_size))
    spatial = max(1, spatial // 2)
    spatial = max(1, spatial // 2)
    spatial = max(1, (spatial + 1) // 2)
    spatial = max(1, (spatial + 1) // 2)
    return spatial


class SlimMambaEncoder(nn.Module):
    """SLIM-Mamba encoder with feature-map and dual local/global APIs."""

    def __init__(
        self,
        image_size: int = 84,
        pool_output: bool = False,
        in_channels: int = 3,
        base_dim: int = 64,
        output_dim: int = 640,
        d_state: int = 8,
        drop_path: float = 0.05,
        perturb_sigma: float = 0.05,
        window_size: int = 2,
    ) -> None:
        super().__init__()

        self.pool_output = bool(pool_output)
        self.out_channels = int(output_dim)
        self.out_dim = int(output_dim)
        self.out_spatial = _stage4_spatial(image_size)
        self.feat_dim = [self.out_channels, self.out_spatial, self.out_spatial]

        dim1 = int(base_dim)
        dim2 = int(base_dim) * 2 + 32
        dim3 = int(base_dim) * 5
        dim4 = int(output_dim)

        self.stage1 = ConvBlock(in_channels, dim1)
        self.stage2 = ConvBlock(dim1, dim2)

        self.stem3 = nn.Sequential(
            nn.Conv2d(dim2, dim3, kernel_size=3, stride=2, padding=1, bias=False),
            EpisodicGN(dim3),
            nn.GELU(),
        )
        self.vss3 = HFFVSSBlock(
            dim3,
            d_state=d_state,
            window_size=window_size,
            drop_path=float(drop_path) * 0.5,
        )

        self.stem4 = nn.Sequential(
            nn.Conv2d(dim3, dim4, kernel_size=3, stride=2, padding=1, bias=False),
            EpisodicGN(dim4),
            nn.GELU(),
        )
        self.vss4 = HFFVSSBlock(
            dim4,
            d_state=d_state,
            window_size=window_size,
            drop_path=float(drop_path),
        )

        self.token_norm = EpisodicLayerNorm(dim4)
        self.perturb = FeaturePerturbation(perturb_sigma)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="leaky_relu")
            elif isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage1(x)
        x = self.stage2(x)

        x = self.stem3(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.vss3(x)
        x = x.permute(0, 3, 1, 2).contiguous()

        x = self.stem4(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.vss4(x)
        x = x.permute(0, 3, 1, 2).contiguous()

        return self.perturb(x)

    def forward_dual(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feature_map = self.forward_feature_map(x)
        local_feat = self.token_norm(_hw_to_tokens(feature_map))
        global_feat = local_feat.mean(dim=1)
        return feature_map, local_feat, global_feat

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_feature_map(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature_map, _, global_feat = self.forward_dual(x)
        if not self.pool_output:
            return feature_map
        return global_feat
