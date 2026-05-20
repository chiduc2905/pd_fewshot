"""Spatial Context Enrichment via multi-scale depthwise convolution.

Enriches each spatial token with neighbor context before OT matching.
When the residual gate is zero (initialization), output === input.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


CONTEXT_FUSION_MODES = frozenset({"weighted_sum", "bottleneck"})


def parse_context_kernel_sizes(raw: str | Sequence[int]) -> tuple[int, ...]:
    if isinstance(raw, str):
        items = [s.strip() for s in raw.split(",") if s.strip()]
    else:
        items = list(raw)
    sizes = []
    for item in items:
        k = int(item)
        if k < 3 or k % 2 == 0:
            raise ValueError(f"Context kernel size must be odd >= 3, got {k}")
        sizes.append(k)
    if not sizes:
        raise ValueError("context_kernel_sizes must contain at least one kernel size")
    return tuple(sorted(set(sizes)))


class SpatialContextEnrichment(nn.Module):
    """Multi-scale depthwise convolution with residual gating.

    Each branch applies a depthwise conv at a different kernel size,
    capturing progressively wider spatial neighborhoods.  A learnable
    residual gate (initialized near zero) controls how much enrichment
    is mixed into the original feature map.
    """

    def __init__(
        self,
        channels: int,
        kernel_sizes: tuple[int, ...] = (3, 5),
        fusion: str = "weighted_sum",
        gate_max: float = 1.0,
    ) -> None:
        super().__init__()
        fusion = str(fusion).strip().lower().replace("-", "_")
        if fusion not in CONTEXT_FUSION_MODES:
            raise ValueError(
                f"Unsupported context fusion mode: {fusion}. "
                f"Expected one of {sorted(CONTEXT_FUSION_MODES)}"
            )
        self.channels = int(channels)
        self.kernel_sizes = tuple(int(k) for k in kernel_sizes)
        self.fusion_mode = fusion
        self.gate_max = float(gate_max)

        self.branches = nn.ModuleList()
        for k in self.kernel_sizes:
            self.branches.append(
                nn.Conv2d(
                    channels,
                    channels,
                    kernel_size=k,
                    padding=k // 2,
                    groups=channels,
                    bias=False,
                )
            )

        num_branches = 1 + len(self.kernel_sizes)

        if fusion == "weighted_sum":
            self.branch_weights = nn.Parameter(torch.zeros(num_branches))
            self.raw_gate = nn.Parameter(torch.tensor(0.0))
        else:
            self.fusion_conv = nn.Sequential(
                nn.Conv2d(channels * num_branches, channels // 4, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels // 4, channels, 1),
            )
            nn.init.zeros_(self.fusion_conv[-1].weight)
            nn.init.zeros_(self.fusion_conv[-1].bias)

    @property
    def gate_value(self) -> torch.Tensor:
        if self.fusion_mode == "weighted_sum":
            g = torch.sigmoid(self.raw_gate - 4.0)
            if self.gate_max < 1.0:
                g = g.clamp(max=self.gate_max)
            return g
        return self.raw_gate.new_tensor(1.0)

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        if feature_map.dim() != 4:
            raise ValueError(
                f"SpatialContextEnrichment expects 4D input, got shape={tuple(feature_map.shape)}"
            )
        branch_outputs = [feature_map]
        for branch in self.branches:
            branch_outputs.append(branch(feature_map))

        if self.fusion_mode == "weighted_sum":
            weights = torch.softmax(
                self.branch_weights.to(device=feature_map.device, dtype=feature_map.dtype),
                dim=0,
            )
            fused = torch.zeros_like(feature_map)
            for w, b in zip(weights, branch_outputs):
                fused = fused + w * b
            gate = torch.sigmoid(
                self.raw_gate.to(device=feature_map.device, dtype=feature_map.dtype) - 4.0
            )
            if self.gate_max < 1.0:
                gate = gate.clamp(max=self.gate_max)
            return feature_map + gate * (fused - feature_map)
        else:
            concat = torch.cat(branch_outputs, dim=1)
            delta = self.fusion_conv(concat)
            return feature_map + delta

    def diagnostics(self, feature_map: torch.Tensor, enriched: torch.Tensor) -> dict[str, torch.Tensor]:
        ref = feature_map.detach()
        diag: dict[str, torch.Tensor] = {}
        if self.fusion_mode == "weighted_sum":
            weights = torch.softmax(self.branch_weights.detach().float(), dim=0)
            diag["context/gate_value"] = self.gate_value.detach().float()
            diag["context/branch_weight_original"] = weights[0]
            for idx, k in enumerate(self.kernel_sizes):
                diag[f"context/branch_weight_conv{k}"] = weights[1 + idx]
        change = (enriched.detach().float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-8)
        diag["context/token_change_ratio"] = change
        return diag
