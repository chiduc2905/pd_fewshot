"""Shared few-shot utilities for pulse_fewshot models with selectable backbones."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from net.encoders.smnet_conv64f_encoder import build_resnet12_family_encoder


def feature_map_to_tokens(feature_map: torch.Tensor) -> torch.Tensor:
    """Convert `(N, C, H, W)` feature maps to `(N, H*W, C)` token sets."""
    if feature_map.dim() != 4:
        raise ValueError(f"Expected a 4D feature map, got shape={tuple(feature_map.shape)}")
    return feature_map.flatten(2).transpose(1, 2).contiguous()


def merge_support_tokens(support_tokens: torch.Tensor, merge_mode: str = "concat") -> torch.Tensor:
    """Merge support token sets class-wise.

    Args:
        support_tokens: `(Way, Shot, Tokens, Dim)`.
        merge_mode: `concat` preserves every support token; `mean` averages
            aligned tokens across shots and returns `(Way, Tokens, Dim)`.
    """
    if support_tokens.dim() != 4:
        raise ValueError(
            "support_tokens must have shape (Way, Shot, Tokens, Dim), "
            f"got {tuple(support_tokens.shape)}"
        )
    if merge_mode == "concat":
        way_num, shot_num, token_num, dim = support_tokens.shape
        return support_tokens.reshape(way_num, shot_num * token_num, dim)
    if merge_mode == "mean":
        return support_tokens.mean(dim=1)
    raise ValueError(f"Unsupported merge_mode: {merge_mode}")


def pooled_episode_features(feature_map: torch.Tensor) -> torch.Tensor:
    """Global-average-pool a `(N, C, H, W)` tensor to `(N, C)`."""
    if feature_map.dim() != 4:
        raise ValueError(f"Expected a 4D feature map, got shape={tuple(feature_map.shape)}")
    return feature_map.mean(dim=(2, 3))


class BaseConv64FewShotModel(nn.Module):
    """Base class for benchmark models with a configurable feature backbone."""

    def __init__(
        self,
        in_channels: int = 3,
        hidden_dim: int = 64,
        backbone_name: str = "resnet12",
        image_size: int = 64,
        resnet12_drop_rate: float = 0.0,
        resnet12_dropblock_size: int = 5,
    ) -> None:
        super().__init__()
        if in_channels != 3:
            raise ValueError("pulse_fewshot backbones currently expect 3-channel inputs")

        self.backbone_name = str(backbone_name).lower()
        self.hidden_dim = int(hidden_dim)
        self.backbone = build_resnet12_family_encoder(
            image_size=image_size,
            backbone_name=self.backbone_name,
            pool_output=False,
            variant="fewshot",
            drop_rate=resnet12_drop_rate,
            dropblock_size=resnet12_dropblock_size,
        )
        backbone_out_dim = int(self.backbone.out_channels)

        if backbone_out_dim != self.hidden_dim:
            self.backbone_adapter = nn.Sequential(
                nn.Conv2d(backbone_out_dim, self.hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(self.hidden_dim),
                nn.ReLU(inplace=False),
            )
        else:
            self.backbone_adapter = nn.Identity()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode images into backbone-aligned feature maps."""
        return self.backbone_adapter(self.backbone(x))

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract global pooled features for visualization utilities."""
        return pooled_episode_features(self.encode(x))

    def get_features(self, images: torch.Tensor) -> torch.Tensor:
        """Return spatial feature maps for external analysis utilities."""
        return self.encode(images)

    @staticmethod
    def validate_episode_inputs(query: torch.Tensor, support: torch.Tensor) -> Tuple[int, int, int, int, int, int]:
        """Validate episode tensor shapes and return `(B, NQ, C, H, W, Way)`."""
        if query.dim() != 5:
            raise ValueError(f"query must have shape (B, NQ, C, H, W), got {tuple(query.shape)}")
        if support.dim() != 6:
            raise ValueError(
                f"support must have shape (B, Way, Shot, C, H, W), got {tuple(support.shape)}"
            )
        bsz, nq, channels, height, width = query.shape
        if support.shape[0] != bsz or support.shape[3:] != (channels, height, width):
            raise ValueError(
                "Support/query episode shapes are inconsistent: "
                f"query={tuple(query.shape)} support={tuple(support.shape)}"
            )
        way_num = support.shape[1]
        return bsz, nq, channels, height, width, way_num
