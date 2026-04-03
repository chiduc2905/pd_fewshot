"""SPIF local-SW variants.

These variants keep the original SPIF architecture intact:
- stable / variant token factorization;
- stable evidence gate;
- global stable prototype branch with cosine similarity;
- alpha fusion and the original auxiliary losses.

Only the local branch is replaced. Instead of cosine top-r partial matching,
the local branch now routes through the weighted paper-style SW estimator while
keeping the legacy SW utilities available elsewhere in the repo. Stable gate
values define token masses for the local transport distance.
"""

from __future__ import annotations

import torch

from net.metrics.sliced_wasserstein_weighted import WeightedPaperSlicedWassersteinDistance
from net.spif import SPIFCE, SPIFMAX
from net.spif_weighted_local_sw import _SPIFWeightedLocalSWMixin


class _SPIFLocalSWMixin(_SPIFWeightedLocalSWMixin):
    """Swap only SPIF's local matcher from cosine to weighted paper-style SW."""

    def _init_local_sw(
        self,
        *,
        local_sw_num_projections: int = 64,
        local_sw_p: float = 2.0,
        local_sw_normalize: bool = True,
        local_sw_score_scale: float = 1.0,
    ) -> None:
        if local_sw_num_projections <= 0:
            raise ValueError("local_sw_num_projections must be positive")
        if local_sw_p <= 0.0:
            raise ValueError("local_sw_p must be positive")
        if local_sw_score_scale <= 0.0:
            raise ValueError("local_sw_score_scale must be positive")

        self.local_transport_score_scale = float(local_sw_score_scale)
        self.local_transport_distance = WeightedPaperSlicedWassersteinDistance(
            train_num_projections=int(local_sw_num_projections),
            eval_num_projections=int(local_sw_num_projections),
            p=float(local_sw_p),
            reduction="none",
            normalize_inputs=bool(local_sw_normalize),
            train_projection_mode="resample",
            eval_projection_mode="fixed",
            eval_num_repeats=1,
        )


class SPIFCELocalSW(_SPIFLocalSWMixin, SPIFCE):
    """SPIFCE with cosine global branch and local token-set SW branch."""

    def __init__(
        self,
        *args,
        local_sw_num_projections: int = 64,
        local_sw_p: float = 2.0,
        local_sw_normalize: bool = True,
        local_sw_score_scale: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._init_local_sw(
            local_sw_num_projections=local_sw_num_projections,
            local_sw_p=local_sw_p,
            local_sw_normalize=local_sw_normalize,
            local_sw_score_scale=local_sw_score_scale,
        )


class SPIFMAXLocalSW(_SPIFLocalSWMixin, SPIFMAX):
    """SPIFMAX with cosine global branch and local token-set SW branch."""

    def __init__(
        self,
        *args,
        local_sw_num_projections: int = 64,
        local_sw_p: float = 2.0,
        local_sw_normalize: bool = True,
        local_sw_score_scale: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._init_local_sw(
            local_sw_num_projections=local_sw_num_projections,
            local_sw_p=local_sw_p,
            local_sw_normalize=local_sw_normalize,
            local_sw_score_scale=local_sw_score_scale,
        )
