"""SPIF local paper-SW variants.

This file is separate from ``net/spif_local_sw.py`` on purpose. The older SW
metric implementations remain reusable, while this module routes SPIF local
transport through a weighted paper-style SW estimator with stable gate masses.

It now supports three local aggregation regimes:

- ``pooled``: old shot-collapsed baseline, ``SW(query, concat_all_support_tokens)``;
- ``mean``: shot-preserving per-shot SW followed by uniform shot averaging;
- ``softmin``: shot-preserving per-shot SW followed by query-adaptive softmin
  aggregation over support shots.

- POT-style p-sliced Wasserstein estimator;
- exact 1D OT for unequal token counts under weighted empirical masses;
- train/eval projection regimes can differ;
- projection resampling can be enabled during training.

The rest of SPIF is intentionally preserved:
- stable / variant factorization;
- stable evidence gate;
- cosine global prototype branch;
- original alpha fusion and auxiliary losses.
"""

from __future__ import annotations

from typing import Dict

import torch

from net.metrics.sliced_wasserstein_weighted import WeightedPaperSlicedWassersteinDistance
from net.spif import SPIFCE, SPIFMAX
from net.spif_shot_preserving_local_sw import _SPIFShotPreservingLocalSWMixin


class _SPIFLocalPaperSWMixin(_SPIFShotPreservingLocalSWMixin):
    """Swap only the local SPIF matcher with shot-preserving weighted paper-style SW."""

    def _init_local_papersw(
        self,
        *,
        local_papersw_train_num_projections: int = 128,
        local_papersw_eval_num_projections: int = 512,
        local_papersw_p: float = 2.0,
        local_papersw_normalize_inputs: bool = False,
        local_papersw_train_projection_mode: str = "resample",
        local_papersw_eval_projection_mode: str = "fixed",
        local_papersw_eval_num_repeats: int = 1,
        local_papersw_score_scale: float = 8.0,
        local_papersw_projection_seed: int = 7,
        local_papersw_shot_agg: str = "pooled",
        local_papersw_shot_softmin_beta: float = 10.0,
        local_papersw_debug: bool = False,
    ) -> None:
        if local_papersw_score_scale <= 0.0:
            raise ValueError("local_papersw_score_scale must be positive")
        if local_papersw_shot_agg not in self._VALID_SHOT_AGGREGATIONS:
            raise ValueError(
                "local_papersw_shot_agg must be one of "
                f"{sorted(self._VALID_SHOT_AGGREGATIONS)}, got {local_papersw_shot_agg!r}"
            )
        if local_papersw_shot_softmin_beta <= 0.0:
            raise ValueError("local_papersw_shot_softmin_beta must be positive")

        self.local_transport_score_scale = float(local_papersw_score_scale)
        self.local_shot_agg = str(local_papersw_shot_agg)
        self.local_shot_softmin_beta = float(local_papersw_shot_softmin_beta)
        self.local_debug = bool(local_papersw_debug)
        self.local_transport_distance = WeightedPaperSlicedWassersteinDistance(
            train_num_projections=int(local_papersw_train_num_projections),
            eval_num_projections=int(local_papersw_eval_num_projections),
            p=float(local_papersw_p),
            reduction="none",
            normalize_inputs=bool(local_papersw_normalize_inputs),
            train_projection_mode=str(local_papersw_train_projection_mode),
            eval_projection_mode=str(local_papersw_eval_projection_mode),
            eval_num_repeats=int(local_papersw_eval_num_repeats),
            projection_seed=int(local_papersw_projection_seed),
        )

    def _build_return_payload(
        self,
        diagnostics: list[dict[str, torch.Tensor]],
        logits: torch.Tensor,
        aux_loss: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        first = diagnostics[0]
        payload: Dict[str, torch.Tensor] = {
            "logits": logits,
            "global_scores": torch.cat([item["global_scores"] for item in diagnostics], dim=0),
            "local_scores": torch.cat([item["local_scores"] for item in diagnostics], dim=0),
            "alpha": first["alpha"],
            "mean_gate": torch.stack([item["mean_gate"] for item in diagnostics]).mean(),
            "stable_global_embeddings": torch.cat(
                [item["stable_global_embeddings"] for item in diagnostics],
                dim=0,
            ),
            "variant_global_embeddings": torch.cat(
                [item["variant_global_embeddings"] for item in diagnostics],
                dim=0,
            ),
            "local_class_distances": torch.cat([item["local_class_distances"] for item in diagnostics], dim=0),
        }
        if aux_loss is not None:
            payload["aux_loss"] = aux_loss

        per_shot_distances = self._cat_optional(diagnostics, "per_shot_local_distances")
        if per_shot_distances is not None:
            payload["per_shot_local_distances"] = per_shot_distances

        shot_weights = self._cat_optional(diagnostics, "shot_aggregation_weights")
        if shot_weights is not None:
            payload["shot_aggregation_weights"] = shot_weights

        for key in (
            "positive_local_logit_mean",
            "negative_local_logit_mean",
            "hardest_negative_local_logit_mean",
            "local_margin_mean",
        ):
            value = self._mean_optional(diagnostics, key)
            if value is not None:
                payload[key] = value
        return payload


class SPIFCELocalPaperSW(_SPIFLocalPaperSWMixin, SPIFCE):
    """SPIFCE with global cosine branch and paper-style local SW branch."""

    def __init__(
        self,
        *args,
        local_papersw_train_num_projections: int = 128,
        local_papersw_eval_num_projections: int = 512,
        local_papersw_p: float = 2.0,
        local_papersw_normalize_inputs: bool = False,
        local_papersw_train_projection_mode: str = "resample",
        local_papersw_eval_projection_mode: str = "fixed",
        local_papersw_eval_num_repeats: int = 1,
        local_papersw_score_scale: float = 8.0,
        local_papersw_projection_seed: int = 7,
        local_papersw_shot_agg: str = "pooled",
        local_papersw_shot_softmin_beta: float = 10.0,
        local_papersw_debug: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._init_local_papersw(
            local_papersw_train_num_projections=local_papersw_train_num_projections,
            local_papersw_eval_num_projections=local_papersw_eval_num_projections,
            local_papersw_p=local_papersw_p,
            local_papersw_normalize_inputs=local_papersw_normalize_inputs,
            local_papersw_train_projection_mode=local_papersw_train_projection_mode,
            local_papersw_eval_projection_mode=local_papersw_eval_projection_mode,
            local_papersw_eval_num_repeats=local_papersw_eval_num_repeats,
            local_papersw_score_scale=local_papersw_score_scale,
            local_papersw_projection_seed=local_papersw_projection_seed,
            local_papersw_shot_agg=local_papersw_shot_agg,
            local_papersw_shot_softmin_beta=local_papersw_shot_softmin_beta,
            local_papersw_debug=local_papersw_debug,
        )

    def forward(
        self,
        query: torch.Tensor,
        support: torch.Tensor,
        query_targets: torch.Tensor | None = None,
        support_targets: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> torch.Tensor | Dict[str, torch.Tensor]:
        del support_targets
        bsz, _, _, _, _, _ = self.validate_episode_inputs(query, support)
        batch_outputs = []
        diagnostics = []
        query_targets_2d = None if query_targets is None else query_targets.view(bsz, -1)
        for batch_idx in range(bsz):
            episode = self._forward_episode(
                query[batch_idx],
                support[batch_idx],
                query_targets=None if query_targets_2d is None else query_targets_2d[batch_idx],
            )
            batch_outputs.append(episode["logits"])
            diagnostics.append(episode)

        logits = torch.cat(batch_outputs, dim=0)
        if not return_aux:
            return logits
        return self._build_return_payload(diagnostics, logits)


class SPIFMAXLocalPaperSW(_SPIFLocalPaperSWMixin, SPIFMAX):
    """SPIFMAX with global cosine branch and paper-style local SW branch."""

    def __init__(
        self,
        *args,
        local_papersw_train_num_projections: int = 128,
        local_papersw_eval_num_projections: int = 512,
        local_papersw_p: float = 2.0,
        local_papersw_normalize_inputs: bool = False,
        local_papersw_train_projection_mode: str = "resample",
        local_papersw_eval_projection_mode: str = "fixed",
        local_papersw_eval_num_repeats: int = 1,
        local_papersw_score_scale: float = 8.0,
        local_papersw_projection_seed: int = 7,
        local_papersw_shot_agg: str = "pooled",
        local_papersw_shot_softmin_beta: float = 10.0,
        local_papersw_debug: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._init_local_papersw(
            local_papersw_train_num_projections=local_papersw_train_num_projections,
            local_papersw_eval_num_projections=local_papersw_eval_num_projections,
            local_papersw_p=local_papersw_p,
            local_papersw_normalize_inputs=local_papersw_normalize_inputs,
            local_papersw_train_projection_mode=local_papersw_train_projection_mode,
            local_papersw_eval_projection_mode=local_papersw_eval_projection_mode,
            local_papersw_eval_num_repeats=local_papersw_eval_num_repeats,
            local_papersw_score_scale=local_papersw_score_scale,
            local_papersw_projection_seed=local_papersw_projection_seed,
            local_papersw_shot_agg=local_papersw_shot_agg,
            local_papersw_shot_softmin_beta=local_papersw_shot_softmin_beta,
            local_papersw_debug=local_papersw_debug,
        )

    def forward(
        self,
        query: torch.Tensor,
        support: torch.Tensor,
        query_targets: torch.Tensor | None = None,
        support_targets: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> torch.Tensor | Dict[str, torch.Tensor]:
        del support_targets
        bsz, _, _, _, _, _ = self.validate_episode_inputs(query, support)
        batch_outputs = []
        aux_losses = []
        diagnostics = []
        query_targets_2d = None if query_targets is None else query_targets.view(bsz, -1)
        for batch_idx in range(bsz):
            episode = self._forward_episode(
                query[batch_idx],
                support[batch_idx],
                query_targets=None if query_targets_2d is None else query_targets_2d[batch_idx],
            )
            batch_outputs.append(episode["logits"])
            aux_losses.append(episode["aux_loss"])
            diagnostics.append(episode)

        logits = torch.cat(batch_outputs, dim=0)
        aux_loss = torch.stack(aux_losses).mean() if aux_losses else logits.new_zeros(())

        if not self.training and not return_aux:
            return logits
        return self._build_return_payload(diagnostics, logits, aux_loss=aux_loss)
