"""Hierarchical Support Sliced-Wasserstein few-shot model."""

from __future__ import annotations

import math
from typing import Sequence
from typing import Any

import torch
import torch.nn as nn

from net.fewshot_common import BaseConv64FewShotModel, multiscale_feature_map_to_tokens
from net.heads.hierarchical_support_sw_head import (
    HSSWSupportState,
    HierarchicalSupportSlicedWassersteinHead,
)
from net.metrics.hssw_sliced_wasserstein import HSSWSlicedWassersteinDistance


class HSSWTokenWeighting(nn.Module):
    """Lightweight scorer producing token masses within each image."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        temperature: float = 1.0,
        mix_alpha: float = 1.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if not 0.0 <= mix_alpha <= 1.0:
            raise ValueError("mix_alpha must be in [0, 1]")
        if eps <= 0.0:
            raise ValueError("eps must be positive")
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.temperature = float(temperature)
        self.mix_alpha = float(mix_alpha)
        self.eps = float(eps)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [Batch, Tokens, Dim]
        if tokens.dim() != 3:
            raise ValueError(f"tokens must have shape (Batch, Tokens, Dim), got {tuple(tokens.shape)}")
        logits = self.network(tokens).squeeze(-1) / self.temperature
        learned = torch.softmax(logits, dim=-1)
        if self.mix_alpha == 1.0:
            return learned
        uniform = torch.full_like(learned, 1.0 / float(learned.shape[-1]))
        return (1.0 - self.mix_alpha) * uniform + self.mix_alpha * learned


class HierarchicalSupportSlicedWassersteinNet(BaseConv64FewShotModel):
    """Episodic few-shot model with a standalone HS-SW classifier head."""

    @staticmethod
    def _normalize_scale_mass_budgets(scale_mass_budgets: Sequence[float]) -> tuple[float, ...]:
        if not scale_mass_budgets:
            raise ValueError("scale_mass_budgets must be non-empty")
        budgets = [float(value) for value in scale_mass_budgets]
        if any(value <= 0.0 for value in budgets):
            raise ValueError(f"scale_mass_budgets must be positive, got {scale_mass_budgets}")
        total = sum(budgets)
        return tuple(value / total for value in budgets)

    def __init__(
        self,
        in_channels: int = 3,
        hidden_dim: int = 64,
        token_weight_hidden: int = 128,
        token_weight_temperature: float = 1.0,
        token_weight_mix_alpha: float = 1.0,
        token_mass_reg_weight: float = 0.0,
        multiscale_grids: Sequence[int] = (4, 1),
        multiscale_mass_budgets: Sequence[float] = (0.6, 0.3, 0.1),
        weighted_sw: bool = True,
        normalize_tokens_before_sw: bool = True,
        backbone_name: str = "resnet12",
        image_size: int = 64,
        resnet12_drop_rate: float = 0.0,
        resnet12_dropblock_size: int = 5,
        train_num_projections: int = 32,
        eval_num_projections: int = 64,
        sw_p: float = 2.0,
        train_projection_mode: str = "resample",
        eval_projection_mode: str = "fixed",
        eval_num_repeats: int = 1,
        projection_seed: int = 7,
        pairwise_chunk_size: int | None = None,
        tau: float = 0.2,
        lambda_cons: float = 0.1,
        lambda_red: float = 0.02,
        learn_logit_scale: bool = False,
        logit_scale_init: float = 10.0,
        logit_scale_max: float | None = 100.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            backbone_name=backbone_name,
            image_size=image_size,
            resnet12_drop_rate=resnet12_drop_rate,
            resnet12_dropblock_size=resnet12_dropblock_size,
        )
        if token_mass_reg_weight < 0.0:
            raise ValueError("token_mass_reg_weight must be non-negative")
        if len(multiscale_mass_budgets) != len(multiscale_grids) + 1:
            raise ValueError(
                "multiscale_mass_budgets must provide one budget for local tokens plus one for each pooled grid: "
                f"got budgets={multiscale_mass_budgets} grids={multiscale_grids}"
            )
        self.weighted_sw = bool(weighted_sw)
        self.token_mass_reg_weight = float(token_mass_reg_weight)
        self.eps = float(eps)
        self.multiscale_grids = tuple(int(grid_size) for grid_size in multiscale_grids)
        self.multiscale_mass_budgets = self._normalize_scale_mass_budgets(multiscale_mass_budgets)
        self.token_weighter = HSSWTokenWeighting(
            input_dim=hidden_dim,
            hidden_dim=int(token_weight_hidden),
            temperature=float(token_weight_temperature),
            mix_alpha=float(token_weight_mix_alpha),
            eps=float(eps),
        )
        self.sw_distance = HSSWSlicedWassersteinDistance(
            train_num_projections=int(train_num_projections),
            eval_num_projections=int(eval_num_projections),
            p=float(sw_p),
            weighted=self.weighted_sw,
            normalize_tokens=bool(normalize_tokens_before_sw),
            train_projection_mode=str(train_projection_mode),
            eval_projection_mode=str(eval_projection_mode),
            eval_num_repeats=int(eval_num_repeats),
            projection_seed=int(projection_seed),
            pairwise_chunk_size=pairwise_chunk_size,
            eps=float(eps),
        )
        self.head = HierarchicalSupportSlicedWassersteinHead(
            sw_distance=self.sw_distance,
            tau=float(tau),
            lambda_cons=float(lambda_cons),
            lambda_red=float(lambda_red),
            weighted_sw=self.weighted_sw,
            learn_logit_scale=bool(learn_logit_scale),
            logit_scale_init=float(logit_scale_init),
            logit_scale_max=logit_scale_max,
            eps=float(eps),
        )

    def _mass_kl_to_uniform(self, weights: torch.Tensor) -> torch.Tensor:
        num_tokens = int(weights.shape[-1])
        uniform_log_prob = -math.log(float(num_tokens))
        return (weights * (torch.log(weights + self.eps) - uniform_log_prob)).sum(dim=-1).mean()

    def _compute_mass_regularization(
        self,
        query_weights: torch.Tensor,
        support_weights: torch.Tensor,
    ) -> torch.Tensor:
        if not self.weighted_sw or self.token_mass_reg_weight == 0.0:
            return query_weights.new_zeros(())
        query_reg = self._mass_kl_to_uniform(query_weights)
        support_reg = self._mass_kl_to_uniform(support_weights)
        return 0.5 * (query_reg + support_reg)

    def _mean_mass_entropy(self, weights: torch.Tensor) -> torch.Tensor:
        return -(weights * torch.log(weights + self.eps)).sum(dim=-1).mean()

    def encode_token_measures(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # features: [Batch, Dim, Hf, Wf]
        features = self.encode(images)
        token_groups = multiscale_feature_map_to_tokens(features, pooled_grids=self.multiscale_grids)
        weights_by_scale = []
        for scale_mass, tokens in zip(self.multiscale_mass_budgets, token_groups):
            weights_by_scale.append(scale_mass * self.token_weighter(tokens))
        tokens = torch.cat(token_groups, dim=1)
        weights = torch.cat(weights_by_scale, dim=1)
        return tokens, weights

    def _encode_episode(
        self,
        query: torch.Tensor,
        support: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        way_num, shot_num = support.shape[:2]
        query_tokens, query_weights = self.encode_token_measures(query)
        support_tokens, support_weights = self.encode_token_measures(
            support.reshape(way_num * shot_num, *support.shape[-3:])
        )
        support_tokens = support_tokens.reshape(way_num, shot_num, support_tokens.shape[-2], support_tokens.shape[-1])
        support_weights = support_weights.reshape(way_num, shot_num, support_weights.shape[-1])
        return query_tokens, query_weights, support_tokens, support_weights

    @staticmethod
    def _stack_auxiliary(batch_outputs: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        return {
            "logits": torch.cat([item["logits"] for item in batch_outputs], dim=0),
            "shot_distances": torch.cat([item["shot_distances"] for item in batch_outputs], dim=0),
            "shot_responsibilities": torch.cat([item["shot_responsibilities"] for item in batch_outputs], dim=0),
            "base_distance": torch.cat([item["base_distance"] for item in batch_outputs], dim=0),
            "consistency_penalty": torch.cat([item["consistency_penalty"] for item in batch_outputs], dim=0),
            "redundancy_penalty": torch.stack([item["redundancy_penalty"] for item in batch_outputs], dim=0),
            "support_pairwise_sw": torch.stack([item["support_pairwise_sw"] for item in batch_outputs], dim=0),
            "total_distance": torch.cat([item["total_distance"] for item in batch_outputs], dim=0),
            "query_transport_weights": torch.cat([item["query_transport_weights"] for item in batch_outputs], dim=0),
            "support_transport_weights": torch.stack([item["support_transport_weights"] for item in batch_outputs], dim=0),
            "gamma_entropy": torch.cat([item["gamma_entropy"] for item in batch_outputs], dim=0),
            "logit_scale": torch.stack([item["logit_scale"] for item in batch_outputs]).mean(),
            "mass_regularization": torch.stack([item["mass_regularization"] for item in batch_outputs]).mean(),
            "mean_shot_distance": torch.stack([item["mean_shot_distance"] for item in batch_outputs]).mean(),
            "mean_gamma_entropy": torch.stack([item["mean_gamma_entropy"] for item in batch_outputs]).mean(),
            "mean_consistency_penalty": torch.stack([item["mean_consistency_penalty"] for item in batch_outputs]).mean(),
            "mean_redundancy_penalty": torch.stack([item["mean_redundancy_penalty"] for item in batch_outputs]).mean(),
            "query_token_weights": torch.cat([item["query_token_weights"] for item in batch_outputs], dim=0),
            "support_token_weights": torch.stack([item["support_token_weights"] for item in batch_outputs], dim=0),
            "mean_query_mass_entropy": torch.stack([item["mean_query_mass_entropy"] for item in batch_outputs]).mean(),
            "mean_support_mass_entropy": torch.stack([item["mean_support_mass_entropy"] for item in batch_outputs]).mean(),
        }

    def _forward_episode(
        self,
        query: torch.Tensor,
        support: torch.Tensor,
        return_aux: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        query_tokens, query_weights, support_tokens, support_weights = self._encode_episode(query, support)
        support_state: HSSWSupportState = self.head.build_support_state(
            shot_tokens=support_tokens,
            shot_weights=support_weights,
        )
        outputs = self.head(
            query_tokens=query_tokens,
            query_weights=query_weights,
            support_state=support_state,
            return_aux=return_aux,
        )
        if not return_aux:
            return outputs
        outputs["mass_regularization"] = self._compute_mass_regularization(
            query_weights=query_weights,
            support_weights=support_weights,
        )
        outputs["mean_query_mass_entropy"] = self._mean_mass_entropy(query_weights)
        outputs["mean_support_mass_entropy"] = self._mean_mass_entropy(support_weights)
        outputs["query_token_weights"] = query_weights
        outputs["support_token_weights"] = support_weights
        return outputs

    def forward(
        self,
        query: torch.Tensor,
        support: torch.Tensor,
        return_aux: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        bsz, _, _, _, _, _ = self.validate_episode_inputs(query, support)
        needs_payload = bool(return_aux or (self.training and self.token_mass_reg_weight > 0.0))
        batch_outputs = []
        batch_logits = []

        for batch_idx in range(bsz):
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
        if self.training and self.token_mass_reg_weight > 0.0:
            stacked["aux_loss"] = self.token_mass_reg_weight * stacked["mass_regularization"]
        if return_aux:
            return stacked
        return {
            "logits": logits,
            "aux_loss": stacked["aux_loss"],
        }
