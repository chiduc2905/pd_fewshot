"""MARS encoder backed by the official VMamba VSS block."""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


VMAMBA_REPO_ENV = "VMAMBA_REPO_ROOT"


def _tokens_to_hw(tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
    return tokens.reshape(tokens.shape[0], height, width, tokens.shape[-1])


def _hw_to_tokens(x: torch.Tensor) -> torch.Tensor:
    return x.flatten(2).transpose(1, 2).contiguous()


def _iter_vmamba_module_candidates(repo_root: str | None):
    yield ("module", "vmamba")
    yield ("module", "classification.models.vmamba")

    roots = []
    if repo_root:
        roots.append(Path(repo_root).expanduser())
    env_root = os.environ.get(VMAMBA_REPO_ENV)
    if env_root:
        roots.append(Path(env_root).expanduser())

    seen = set()
    for root in roots:
        key = str(root)
        if key in seen:
            continue
        seen.add(key)
        yield ("file", root / "vmamba.py")
        yield ("file", root / "classification" / "models" / "vmamba.py")


def _load_vmamba_module(repo_root: str | None = None):
    errors: list[str] = []
    for kind, candidate in _iter_vmamba_module_candidates(repo_root):
        if kind == "module":
            try:
                module = importlib.import_module(candidate)
            except Exception as exc:
                errors.append(f"{candidate}: {exc}")
                continue
        else:
            module_path = candidate
            if not module_path.is_file():
                errors.append(f"{module_path}: not found")
                continue
            module_name = f"_pulse_vmamba_{abs(hash(str(module_path.resolve())))}"
            module = sys.modules.get(module_name)
            if module is None:
                spec = importlib.util.spec_from_file_location(module_name, module_path)
                if spec is None or spec.loader is None:
                    errors.append(f"{module_path}: could not build import spec")
                    continue
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                try:
                    spec.loader.exec_module(module)
                except Exception as exc:
                    sys.modules.pop(module_name, None)
                    errors.append(f"{module_path}: {exc}")
                    continue
        if hasattr(module, "VSSBlock"):
            return module
        errors.append(f"{candidate}: VSSBlock not found")

    detail = "; ".join(errors[-4:])
    raise ImportError(
        "MARS backbone requires the official VMamba `VSSBlock`. "
        "Install VMamba so `import vmamba` works, or pass `--vmamba_repo_root` / "
        f"`{VMAMBA_REPO_ENV}` to the official repository root. "
        f"Recent import attempts: {detail}"
    )


class EpisodicLayerNorm(nn.Module):
    """LayerNorm over the last dimension for token-like tensors."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, (x.shape[-1],), self.weight, self.bias, self.eps)


class OfficialVMambaVSSBlock(nn.Module):
    """Thin wrapper around the upstream VMamba VSSBlock."""

    def __init__(
        self,
        dim: int,
        drop_path: float = 0.0,
        d_state: int = 16,
        ssm_ratio: float = 2.0,
        vmamba_repo_root: str | None = None,
    ) -> None:
        super().__init__()
        vmamba_module = _load_vmamba_module(vmamba_repo_root)
        vss_block_cls = getattr(vmamba_module, "VSSBlock")

        signature = inspect.signature(vss_block_cls.__init__)
        params = signature.parameters
        kwargs = {}

        if "hidden_dim" in params:
            kwargs["hidden_dim"] = int(dim)
        elif "dim" in params:
            kwargs["dim"] = int(dim)
        else:
            raise TypeError("Official VMamba VSSBlock signature is missing `hidden_dim`/`dim`.")

        if "drop_path" in params:
            kwargs["drop_path"] = float(drop_path)
        if "ssm_d_state" in params:
            kwargs["ssm_d_state"] = int(d_state)
        elif "d_state" in params:
            kwargs["d_state"] = int(d_state)
        if "ssm_ratio" in params:
            kwargs["ssm_ratio"] = float(ssm_ratio)
        if "channel_first" in params:
            kwargs["channel_first"] = False

        try:
            self.block = vss_block_cls(**kwargs)
        except TypeError as exc:
            raise TypeError(
                "Failed to instantiate the official VMamba VSSBlock with the detected "
                f"adapter kwargs {kwargs}. The installed VMamba version is likely "
                "incompatible with this MARS adapter."
            ) from exc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.block(x)
        if y.shape != x.shape:
            raise RuntimeError(
                "Official VMamba VSSBlock returned an unexpected shape: "
                f"input={tuple(x.shape)} output={tuple(y.shape)}"
            )
        return y


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
        vmamba_repo_root: str | None = None,
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
            [
                OfficialVMambaVSSBlock(
                    out_dim,
                    drop_path=drop_path_rates[idx],
                    vmamba_repo_root=vmamba_repo_root,
                )
                for idx in range(depth)
            ]
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
        vmamba_repo_root: str | None = None,
    ) -> None:
        super().__init__()
        self.pool_output = bool(pool_output)
        self.out_channels = int(output_dim)
        self.out_dim = int(output_dim)
        self.vmamba_repo_root = vmamba_repo_root

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
            vmamba_repo_root=vmamba_repo_root,
        )
        block_idx += depths[0]
        self.stage2 = MARSStage(
            dims[0],
            dims[1],
            depth=depths[1],
            downsample=True,
            use_rfam=True,
            drop_path_rates=drop_path_rates[block_idx:block_idx + depths[1]],
            vmamba_repo_root=vmamba_repo_root,
        )
        block_idx += depths[1]
        self.stage3 = MARSStage(
            dims[1],
            dims[2],
            depth=depths[2],
            downsample=True,
            use_rfam=False,
            drop_path_rates=drop_path_rates[block_idx:block_idx + depths[2]],
            vmamba_repo_root=vmamba_repo_root,
        )
        block_idx += depths[2]
        self.stage4 = MARSStage(
            dims[2],
            dims[3],
            depth=depths[3],
            downsample=True,
            use_rfam=True,
            drop_path_rates=drop_path_rates[block_idx:block_idx + depths[3]],
            vmamba_repo_root=vmamba_repo_root,
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
