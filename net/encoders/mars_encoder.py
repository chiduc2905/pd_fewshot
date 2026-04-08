"""MARS encoder adapted for few-shot backbones.

This module exposes a feature-map compatible encoder for the rest of the
few-shot codebase while still making MARS' dual token/global outputs available
to models that want to consume them directly.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _tokens_to_hw(tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
    return tokens.reshape(tokens.shape[0], height, width, tokens.shape[-1])


def _hw_to_tokens(x: torch.Tensor) -> torch.Tensor:
    return x.flatten(2).transpose(1, 2).contiguous()


class EpisodicLayerNorm(nn.Module):
    """LayerNorm over the last dimension for token-like tensors."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, (x.shape[-1],), self.weight, self.bias, self.eps)


class SelectiveSSM(nn.Module):
    """Simplified selective scan block over token sequences."""

    def __init__(self, d_inner: int, d_state: int = 16, dt_rank: int | None = None) -> None:
        super().__init__()
        self.d_inner = int(d_inner)
        self.d_state = int(d_state)
        dt_rank = int(dt_rank or math.ceil(float(d_inner) / 16.0))

        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
        self.x_proj = nn.Linear(d_inner, dt_rank + 2 * d_state, bias=False)

        a = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(a))
        self.D = nn.Parameter(torch.ones(d_inner))

        nn.init.uniform_(self.dt_proj.weight, -0.1, 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, inner_dim = x.shape
        xz = self.x_proj(x)
        dt_rank = self.dt_proj.in_features
        dt_raw, b_ssm, c_ssm = xz.split([dt_rank, self.d_state, self.d_state], dim=-1)

        dt = F.softplus(self.dt_proj(dt_raw))
        a = -torch.exp(self.A_log.float()).to(device=x.device, dtype=x.dtype)

        d_a = torch.exp(dt.unsqueeze(-1) * a.unsqueeze(0).unsqueeze(0))
        d_b = dt.unsqueeze(-1) * b_ssm.unsqueeze(2)

        hidden = torch.zeros(
            batch_size,
            inner_dim,
            self.d_state,
            device=x.device,
            dtype=x.dtype,
        )
        outputs = []
        for token_idx in range(seq_len):
            hidden = d_a[:, token_idx] * hidden + d_b[:, token_idx] * x[:, token_idx].unsqueeze(-1)
            out = (hidden * c_ssm[:, token_idx].unsqueeze(1)).sum(dim=-1)
            outputs.append(out)
        y = torch.stack(outputs, dim=1)
        return y + x * self.D


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep_prob) / keep_prob
        return x * random_tensor


class VSSBlock(nn.Module):
    """Visual state-space block with 4-direction aggregation."""

    def __init__(self, dim: int, d_state: int = 16, expand: float = 2.0, drop_path: float = 0.0) -> None:
        super().__init__()
        d_inner = int(dim * expand)
        self.norm = EpisodicLayerNorm(dim)
        self.in_proj = nn.Linear(dim, d_inner * 2, bias=False)
        self.out_proj = nn.Linear(d_inner, dim, bias=False)
        self.act = nn.SiLU()

        self.ssm_h = SelectiveSSM(d_inner, d_state)
        self.ssm_hr = SelectiveSSM(d_inner, d_state)
        self.ssm_v = SelectiveSSM(d_inner, d_state)
        self.ssm_vr = SelectiveSSM(d_inner, d_state)

        self.dir_weights = nn.Parameter(torch.full((4,), 0.25))

        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = EpisodicLayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def _scan_4dir(self, tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
        batch_size, _, inner_dim = tokens.shape

        y_h = self.ssm_h(tokens)
        y_hr = self.ssm_hr(tokens.flip(1)).flip(1)

        vertical_tokens = tokens.reshape(batch_size, height, width, inner_dim).transpose(1, 2).contiguous()
        vertical_tokens = vertical_tokens.reshape(batch_size, width * height, inner_dim)
        y_v = self.ssm_v(vertical_tokens)
        y_v = y_v.reshape(batch_size, width, height, inner_dim).transpose(1, 2).contiguous()
        y_v = y_v.reshape(batch_size, height * width, inner_dim)

        vertical_tokens_rev = vertical_tokens.flip(1)
        y_vr = self.ssm_vr(vertical_tokens_rev).flip(1)
        y_vr = y_vr.reshape(batch_size, width, height, inner_dim).transpose(1, 2).contiguous()
        y_vr = y_vr.reshape(batch_size, height * width, inner_dim)

        weights = F.softmax(self.dir_weights, dim=0)
        return weights[0] * y_h + weights[1] * y_hr + weights[2] * y_v + weights[3] * y_vr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, height, width, channels = x.shape
        shortcut = x
        x_norm = self.norm(x)
        xz = self.in_proj(x_norm)
        x_in, z = xz.chunk(2, dim=-1)
        x_in = x_in.reshape(batch_size, height * width, -1)

        y = self._scan_4dir(x_in, height, width)
        z = z.reshape(batch_size, height * width, -1)
        y = y * self.act(z)
        y = self.out_proj(y).reshape(batch_size, height, width, channels)

        x = shortcut + self.drop_path(y)
        x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x


class RFAM(nn.Module):
    """Receptive field adaptive module."""

    def __init__(self, dim: int, reduction: int = 4) -> None:
        super().__init__()
        self.dw3 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.dw5 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim, bias=False)
        self.dw7 = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=False)
        self.pw = nn.Conv2d(dim * 3, dim, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(min(32, dim), dim)

        inner_dim = max(dim // reduction, 4)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dim, inner_dim),
            nn.ReLU(inplace=False),
            nn.Linear(inner_dim, 3),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s3 = self.dw3(x)
        s5 = self.dw5(x)
        s7 = self.dw7(x)

        weights = self.se(x)
        fused = (
            weights[:, 0:1, None, None] * s3
            + weights[:, 1:2, None, None] * s5
            + weights[:, 2:3, None, None] * s7
        )

        out = self.pw(torch.cat([s3, s5, s7], dim=1))
        return x + F.gelu(self.norm(out + fused))


class CSSF(nn.Module):
    """Cross-scale state fusion."""

    def __init__(self, scale_dims: list[int], out_dim: int, n_heads: int = 4) -> None:
        super().__init__()
        self.projs = nn.ModuleList([nn.Linear(dim, out_dim) for dim in scale_dims])
        num_heads = n_heads if out_dim % n_heads == 0 else 1
        self.attn = nn.MultiheadAttention(out_dim, num_heads, batch_first=True)
        self.norm = EpisodicLayerNorm(out_dim)
        self.ffn = nn.Sequential(
            nn.Linear(out_dim, out_dim * 2),
            nn.GELU(),
            nn.Linear(out_dim * 2, out_dim),
        )
        self.norm2 = EpisodicLayerNorm(out_dim)

    def forward(self, feats: list[torch.Tensor], target_hw: tuple[int, int]) -> torch.Tensor:
        target_h, target_w = target_hw
        tokens_all = []
        for feat, proj in zip(feats, self.projs):
            feat_map = feat.permute(0, 3, 1, 2).contiguous()
            feat_map = F.interpolate(feat_map, size=target_hw, mode="bilinear", align_corners=False)
            feat_tokens = _hw_to_tokens(feat_map)
            tokens_all.append(proj(feat_tokens))

        kv = torch.cat(tokens_all, dim=1)
        q = tokens_all[-1]
        out, _ = self.attn(self.norm(q), self.norm(kv), self.norm(kv))
        q = q + out
        q = q + self.ffn(self.norm2(q))
        return _tokens_to_hw(q, target_h, target_w)


class MARSStage(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        depth: int,
        downsample: bool = True,
        use_rfam: bool = True,
        drop_path_rates: list[float] | None = None,
    ) -> None:
        super().__init__()
        drop_path_rates = drop_path_rates or [0.0] * depth

        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2, bias=False),
                nn.GroupNorm(min(32, out_dim), out_dim),
            )
        elif in_dim == out_dim:
            self.downsample = nn.Identity()
        else:
            self.downsample = nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False)

        self.blocks = nn.ModuleList(
            [VSSBlock(out_dim, drop_path=drop_path_rates[idx]) for idx in range(depth)]
        )
        self.rfam = RFAM(out_dim) if use_rfam else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        if self.rfam is not None:
            x = self.rfam(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        for block in self.blocks:
            x = block(x)
        return x.permute(0, 3, 1, 2).contiguous()


class FeaturePerturbation(nn.Module):
    def __init__(self, sigma: float = 0.05) -> None:
        super().__init__()
        self.sigma = float(sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.sigma <= 0.0:
            return x
        return x + torch.randn_like(x) * self.sigma


class MARSEncoder(nn.Module):
    """MARS few-shot encoder with feature-map and dual-output APIs."""

    def __init__(
        self,
        image_size: int = 64,
        pool_output: bool = False,
        in_channels: int = 3,
        base_dim: int = 64,
        output_dim: int = 640,
        drop_path: float = 0.1,
        perturb_sigma: float = 0.05,
    ) -> None:
        super().__init__()
        self.pool_output = bool(pool_output)
        self.out_channels = int(output_dim)
        self.out_dim = int(output_dim)

        spatial = int(image_size)
        for _ in range(3):
            spatial = max(1, spatial // 2)
        self.out_spatial = spatial
        self.feat_dim = [self.out_channels, self.out_spatial, self.out_spatial]

        dims = [int(base_dim), int(base_dim) * 2, int(base_dim) * 4, int(output_dim)]
        depths = [2, 2, 4, 2]
        total_blocks = sum(depths)
        drop_path_rates = torch.linspace(0.0, float(drop_path), total_blocks).tolist()

        block_idx = 0
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(16, dims[0]), dims[0]),
            nn.GELU(),
        )
        self.stage1 = MARSStage(
            dims[0],
            dims[0],
            depth=depths[0],
            downsample=False,
            use_rfam=True,
            drop_path_rates=drop_path_rates[block_idx:block_idx + depths[0]],
        )
        block_idx += depths[0]
        self.stage2 = MARSStage(
            dims[0],
            dims[1],
            depth=depths[1],
            downsample=True,
            use_rfam=True,
            drop_path_rates=drop_path_rates[block_idx:block_idx + depths[1]],
        )
        block_idx += depths[1]
        self.stage3 = MARSStage(
            dims[1],
            dims[2],
            depth=depths[2],
            downsample=True,
            use_rfam=False,
            drop_path_rates=drop_path_rates[block_idx:block_idx + depths[2]],
        )
        block_idx += depths[2]
        self.stage4 = MARSStage(
            dims[2],
            dims[3],
            depth=depths[3],
            downsample=True,
            use_rfam=True,
            drop_path_rates=drop_path_rates[block_idx:block_idx + depths[3]],
        )
        self.cssf = CSSF([dims[0], dims[1], dims[2]], out_dim=dims[2], n_heads=4)
        self.perturb = FeaturePerturbation(perturb_sigma)
        self.token_norm = EpisodicLayerNorm(output_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")

    def forward_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        stage1 = self.stage1(x)
        stage2 = self.stage2(stage1)
        stage3 = self.stage3(stage2)

        target_h, target_w = stage3.shape[-2:]
        stage1_hw = stage1.permute(0, 2, 3, 1).contiguous()
        stage2_hw = stage2.permute(0, 2, 3, 1).contiguous()
        stage3_hw = stage3.permute(0, 2, 3, 1).contiguous()
        stage3_fused = self.cssf([stage1_hw, stage2_hw, stage3_hw], (target_h, target_w))
        stage3 = stage3_fused.permute(0, 3, 1, 2).contiguous()

        stage4 = self.stage4(stage3)
        return self.perturb(stage4)

    def forward_dual(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feature_map = self.forward_feature_map(x)
        local_feat = _hw_to_tokens(feature_map)
        local_feat = self.token_norm(local_feat)
        global_feat = local_feat.mean(dim=1)
        return feature_map, local_feat, global_feat

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_feature_map(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature_map, _, global_feat = self.forward_dual(x)
        if not self.pool_output:
            return feature_map
        return global_feat
