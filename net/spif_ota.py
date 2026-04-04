"""SPIF-OTA: Single-branch physics-aware optimal transport inference."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from net.fewshot_common import BaseConv64FewShotModel, feature_map_to_tokens


class SPIFOTAOutput(dict):
    """Dict-like output that still exposes `.shape` via logits."""

    @property
    def shape(self):
        logits = self.get("logits")
        return None if logits is None else logits.shape


class TransportTokenProjector(nn.Module):
    """Project backbone tokens into a transport geometry space."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int | None = None,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim and output_dim must be positive")
        hidden_dim = int(hidden_dim or output_dim)
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.eps = float(eps)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        projected = self.network(tokens)
        return F.normalize(projected, p=2, dim=-1, eps=self.eps)


class PhysicsAwareMassGenerator(nn.Module):
    """Content-position token masses with learnable row and column biases."""

    def __init__(
        self,
        token_dim: int,
        spatial_height: int,
        spatial_width: int,
        hidden_dim: int | None = None,
        temperature: float = 1.0,
        use_column_bias: bool = True,
    ) -> None:
        super().__init__()
        if token_dim <= 0:
            raise ValueError("token_dim must be positive")
        if spatial_height <= 0 or spatial_width <= 0:
            raise ValueError("spatial_height and spatial_width must be positive")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")

        hidden_dim = int(hidden_dim or token_dim)
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")

        self.max_height = int(spatial_height)
        self.max_width = int(spatial_width)
        self.temperature = float(temperature)
        self.use_column_bias = bool(use_column_bias)

        self.content_proj = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, hidden_dim),
            nn.GELU(),
        )
        self.row_embedding = nn.Embedding(self.max_height, hidden_dim)
        self.col_embedding = nn.Embedding(self.max_width, hidden_dim)
        self.score_head = nn.Linear(hidden_dim, 1)
        self.row_bias = nn.Parameter(torch.zeros(self.max_height))
        self.col_bias = nn.Parameter(torch.zeros(self.max_width))

    @staticmethod
    def build_position_ids(height: int, width: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        row_ids = torch.arange(height, device=device).repeat_interleave(width)
        col_ids = torch.arange(width, device=device).repeat(height)
        return row_ids, col_ids

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        height: int,
        width: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if tokens.dim() != 3:
            raise ValueError(f"tokens must have shape (Batch, Tokens, Dim), got {tuple(tokens.shape)}")
        if height <= 0 or width <= 0 or height * width != tokens.shape[-2]:
            raise ValueError(
                f"Inconsistent spatial metadata: height={height}, width={width}, tokens={tokens.shape[-2]}"
            )
        if height > self.max_height or width > self.max_width:
            raise ValueError(
                f"Spatial grid {(height, width)} exceeds configured maximum {(self.max_height, self.max_width)}"
            )

        row_ids, col_ids = self.build_position_ids(height, width, tokens.device)
        hidden = self.content_proj(tokens)
        hidden = hidden + self.row_embedding(row_ids).unsqueeze(0)
        if self.use_column_bias:
            hidden = hidden + self.col_embedding(col_ids).unsqueeze(0)
        logits = self.score_head(hidden).squeeze(-1)
        logits = logits + self.row_bias[row_ids].unsqueeze(0)
        if self.use_column_bias:
            logits = logits + self.col_bias[col_ids].unsqueeze(0)
        logits = logits / self.temperature
        masses = torch.softmax(logits, dim=-1)
        return masses, logits


class BatchedSinkhornOT(nn.Module):
    """Batched log-domain Sinkhorn transport between token measures."""

    def __init__(
        self,
        sinkhorn_epsilon: float = 0.05,
        max_iterations: int = 60,
        position_cost_weight: float = 0.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if sinkhorn_epsilon <= 0.0:
            raise ValueError("sinkhorn_epsilon must be positive")
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if position_cost_weight < 0.0:
            raise ValueError("position_cost_weight must be non-negative")
        if eps <= 0.0:
            raise ValueError("eps must be positive")
        self.sinkhorn_epsilon = float(sinkhorn_epsilon)
        self.max_iterations = int(max_iterations)
        self.position_cost_weight = float(position_cost_weight)
        self.eps = float(eps)
        self.register_buffer("_cached_position_cost", torch.empty(0), persistent=False)
        self._cached_hw: tuple[int, int] | None = None

    def _position_cost(self, height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.position_cost_weight == 0.0:
            return torch.empty(0, device=device, dtype=dtype)
        if self._cached_hw != (height, width) or self._cached_position_cost.numel() == 0:
            row_coords = torch.linspace(0.0, 1.0, steps=height, dtype=torch.float32)
            col_coords = torch.linspace(0.0, 1.0, steps=width, dtype=torch.float32)
            mesh_y, mesh_x = torch.meshgrid(row_coords, col_coords, indexing="ij")
            coords = torch.stack([mesh_y, mesh_x], dim=-1).reshape(height * width, 2)
            position_cost = torch.cdist(coords, coords, p=1) / 2.0
            self._cached_position_cost = position_cost
            self._cached_hw = (height, width)
        return self._cached_position_cost.to(device=device, dtype=dtype)

    def _pairwise_cost(
        self,
        query_tokens: torch.Tensor,
        support_tokens: torch.Tensor,
        *,
        height: int,
        width: int,
    ) -> torch.Tensor:
        similarity = torch.einsum("qmd,wknd->qwkmn", query_tokens, support_tokens)
        cost = 1.0 - similarity
        if self.position_cost_weight > 0.0:
            pos_cost = self._position_cost(height, width, cost.device, cost.dtype)
            cost = cost + self.position_cost_weight * pos_cost.view(1, 1, 1, height * width, height * width)
        return cost

    def forward(
        self,
        query_tokens: torch.Tensor,
        support_tokens: torch.Tensor,
        query_masses: torch.Tensor,
        support_masses: torch.Tensor,
        *,
        height: int,
        width: int,
        return_plan: bool = False,
    ) -> dict[str, torch.Tensor]:
        if query_tokens.dim() != 3:
            raise ValueError(f"query_tokens must have shape (NumQuery, Tokens, Dim), got {tuple(query_tokens.shape)}")
        if support_tokens.dim() != 4:
            raise ValueError(
                f"support_tokens must have shape (Way, Shot, Tokens, Dim), got {tuple(support_tokens.shape)}"
            )
        if query_masses.shape != query_tokens.shape[:-1]:
            raise ValueError(
                f"query_masses must have shape {tuple(query_tokens.shape[:-1])}, got {tuple(query_masses.shape)}"
            )
        if support_masses.shape != support_tokens.shape[:-1]:
            raise ValueError(
                f"support_masses must have shape {tuple(support_tokens.shape[:-1])}, got {tuple(support_masses.shape)}"
            )

        cost = self._pairwise_cost(query_tokens, support_tokens, height=height, width=width)
        log_a = torch.log(query_masses.clamp_min(self.eps)).unsqueeze(1).unsqueeze(1)
        log_b = torch.log(support_masses.clamp_min(self.eps)).unsqueeze(0)
        log_kernel = -cost / self.sinkhorn_epsilon

        log_u = torch.zeros_like(log_a)
        log_v = torch.zeros_like(log_b)
        for _ in range(self.max_iterations):
            log_u = log_a - torch.logsumexp(log_kernel + log_v.unsqueeze(-2), dim=-1)
            log_v = log_b - torch.logsumexp(log_kernel + log_u.unsqueeze(-1), dim=-2)

        log_plan = log_kernel + log_u.unsqueeze(-1) + log_v.unsqueeze(-2)
        plan = torch.exp(log_plan)
        transport_cost = (plan * cost).sum(dim=(-1, -2))
        entropy = -(plan * log_plan).sum(dim=(-1, -2))

        outputs = {
            "transport_cost": transport_cost,
            "plan_entropy": entropy,
        }
        if return_plan:
            outputs["transport_plan"] = plan
            outputs["cost_matrix"] = cost
        return outputs


class ShotEvidenceAggregator(nn.Module):
    """Aggregate per-shot OT evidence into one class score."""

    def __init__(
        self,
        hidden_dim: int = 32,
        mode: str = "learned",
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if mode not in {"learned", "mean"}:
            raise ValueError(f"Unsupported shot aggregation mode: {mode}")
        self.mode = str(mode)
        self.network = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        shot_scores: torch.Tensor,
        plan_entropy: torch.Tensor,
        query_mass_entropy: torch.Tensor,
        support_mass_entropy: torch.Tensor,
    ) -> torch.Tensor:
        if shot_scores.dim() != 3:
            raise ValueError(f"shot_scores must have shape (NumQuery, Way, Shot), got {tuple(shot_scores.shape)}")
        shot_num = int(shot_scores.shape[-1])
        if shot_num == 1:
            return torch.ones_like(shot_scores)
        if self.mode == "mean":
            return torch.full_like(shot_scores, 1.0 / float(shot_num))

        query_entropy_feature = query_mass_entropy.unsqueeze(1).unsqueeze(2).expand_as(shot_scores)
        support_entropy_feature = support_mass_entropy.unsqueeze(0).expand_as(shot_scores)
        features = torch.stack(
            [
                shot_scores,
                -plan_entropy,
                -query_entropy_feature,
                -support_entropy_feature,
            ],
            dim=-1,
        )
        logits = self.network(features).squeeze(-1) + shot_scores
        return torch.softmax(logits, dim=-1)


class SPIFOTA(BaseConv64FewShotModel):
    """Single-branch physics-aware OT few-shot classifier."""

    def __init__(
        self,
        in_channels: int = 3,
        hidden_dim: int = 64,
        transport_dim: int = 128,
        projector_hidden_dim: int | None = None,
        mass_hidden_dim: int | None = None,
        mass_temperature: float = 1.0,
        sinkhorn_epsilon: float = 0.05,
        sinkhorn_iterations: int = 60,
        shot_hidden_dim: int = 32,
        shot_aggregation: str = "learned",
        position_cost_weight: float = 0.0,
        use_column_bias: bool = True,
        lambda_mass: float = 0.0,
        lambda_consistency: float = 0.0,
        backbone_name: str = "resnet12",
        image_size: int = 64,
        resnet12_drop_rate: float = 0.0,
        resnet12_dropblock_size: int = 5,
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
        if transport_dim <= 0:
            raise ValueError("transport_dim must be positive")
        if lambda_mass < 0.0 or lambda_consistency < 0.0:
            raise ValueError("lambda_mass and lambda_consistency must be non-negative")

        out_spatial = int(getattr(self.backbone, "out_spatial", 0))
        if out_spatial <= 0:
            raise ValueError("Backbone must expose a positive out_spatial for SPIF-OTA")

        self.transport_dim = int(transport_dim)
        self.lambda_mass = float(lambda_mass)
        self.lambda_consistency = float(lambda_consistency)
        self.eps = float(eps)

        self.token_projector = TransportTokenProjector(
            input_dim=hidden_dim,
            output_dim=transport_dim,
            hidden_dim=projector_hidden_dim,
            eps=eps,
        )
        self.mass_generator = PhysicsAwareMassGenerator(
            token_dim=transport_dim,
            spatial_height=out_spatial,
            spatial_width=out_spatial,
            hidden_dim=mass_hidden_dim,
            temperature=mass_temperature,
            use_column_bias=use_column_bias,
        )
        self.ot_matcher = BatchedSinkhornOT(
            sinkhorn_epsilon=sinkhorn_epsilon,
            max_iterations=sinkhorn_iterations,
            position_cost_weight=position_cost_weight,
            eps=eps,
        )
        self.shot_aggregator = ShotEvidenceAggregator(
            hidden_dim=shot_hidden_dim,
            mode=shot_aggregation,
        )

    @staticmethod
    def _reshape_query_targets(
        query_targets: torch.Tensor | None,
        *,
        batch_size: int,
        num_query: int,
    ) -> torch.Tensor | None:
        if query_targets is None:
            return None
        if query_targets.dim() == 1:
            if query_targets.numel() != batch_size * num_query:
                raise ValueError(
                    f"query_targets must have {batch_size * num_query} elements, got {query_targets.numel()}"
                )
            return query_targets.view(batch_size, num_query)
        if query_targets.dim() == 2 and tuple(query_targets.shape) == (batch_size, num_query):
            return query_targets
        raise ValueError(
            "query_targets must have shape (Batch * NumQuery,) or (Batch, NumQuery), "
            f"got {tuple(query_targets.shape)}"
        )

    def _encode_images(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, int]]:
        feature_map = self.encode(images)
        height, width = feature_map.shape[-2:]
        tokens = feature_map_to_tokens(feature_map)
        transport_tokens = self.token_projector(tokens)
        masses, mass_logits = self.mass_generator(transport_tokens, height=height, width=width)
        return transport_tokens, masses, mass_logits, (height, width)

    def _entropy(self, masses: torch.Tensor) -> torch.Tensor:
        return -(masses.clamp_min(self.eps) * masses.clamp_min(self.eps).log()).sum(dim=-1)

    def _mass_regularization(self, query_masses: torch.Tensor, support_masses: torch.Tensor) -> torch.Tensor:
        num_query_tokens = float(query_masses.shape[-1])
        num_support_tokens = float(support_masses.shape[-1])
        query_kl = (query_masses * (query_masses.clamp_min(self.eps).log() + math.log(num_query_tokens))).sum(dim=-1)
        support_kl = (
            support_masses * (support_masses.clamp_min(self.eps).log() + math.log(num_support_tokens))
        ).sum(dim=-1)
        return 0.5 * (query_kl.mean() + support_kl.mean())

    def _shot_consistency_loss(
        self,
        shot_scores: torch.Tensor,
        query_targets: torch.Tensor | None,
    ) -> torch.Tensor:
        if query_targets is None or shot_scores.shape[-1] <= 1:
            return shot_scores.new_zeros(())
        true_scores = shot_scores[torch.arange(shot_scores.shape[0], device=shot_scores.device), query_targets]
        return true_scores.var(dim=-1, unbiased=False).mean()

    def _forward_episode(
        self,
        query: torch.Tensor,
        support: torch.Tensor,
        *,
        query_targets: torch.Tensor | None = None,
        needs_payload: bool = False,
        return_aux: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        way_num, shot_num = support.shape[:2]
        query_tokens, query_masses, query_mass_logits, (height, width) = self._encode_images(query)
        support_tokens, support_masses, support_mass_logits, support_hw = self._encode_images(
            support.reshape(way_num * shot_num, *support.shape[-3:])
        )
        if support_hw != (height, width):
            raise ValueError(f"Query/support token grids must match, got {height, width} vs {support_hw}")

        support_tokens = support_tokens.reshape(way_num, shot_num, support_tokens.shape[-2], support_tokens.shape[-1])
        support_masses = support_masses.reshape(way_num, shot_num, support_masses.shape[-1])
        support_mass_logits = support_mass_logits.reshape(way_num, shot_num, support_mass_logits.shape[-1])

        match_outputs = self.ot_matcher(
            query_tokens=query_tokens,
            support_tokens=support_tokens,
            query_masses=query_masses,
            support_masses=support_masses,
            height=height,
            width=width,
            return_plan=return_aux,
        )

        transport_cost = match_outputs["transport_cost"]
        plan_entropy = match_outputs["plan_entropy"]
        shot_scores = -transport_cost

        query_mass_entropy = self._entropy(query_masses)
        support_mass_entropy = self._entropy(support_masses)
        entropy_normalizer = math.log(float(max(2, query_tokens.shape[-2] * support_tokens.shape[-2])))
        shot_weights = self.shot_aggregator(
            shot_scores=shot_scores,
            plan_entropy=plan_entropy / entropy_normalizer,
            query_mass_entropy=query_mass_entropy,
            support_mass_entropy=support_mass_entropy,
        )
        class_scores = (shot_weights * shot_scores).sum(dim=-1)
        logits = class_scores
        total_distance = (shot_weights * transport_cost).sum(dim=-1)

        mass_regularization = self._mass_regularization(query_masses, support_masses)
        shot_consistency_loss = self._shot_consistency_loss(shot_scores, query_targets)
        aux_loss = logits.new_zeros(())
        if self.lambda_mass > 0.0:
            aux_loss = aux_loss + self.lambda_mass * mass_regularization
        if self.lambda_consistency > 0.0:
            aux_loss = aux_loss + self.lambda_consistency * shot_consistency_loss

        if not needs_payload:
            return logits

        outputs = {
            "logits": logits,
            "aux_loss": aux_loss,
            "class_scores": class_scores,
            "total_distance": total_distance,
            "shot_scores": shot_scores,
            "shot_aggregation_weights": shot_weights,
            "transport_cost": transport_cost,
            "plan_entropy": plan_entropy,
            "query_tokens": query_tokens,
            "support_tokens": support_tokens,
            "query_masses": query_masses,
            "support_masses": support_masses,
            "query_mass_logits": query_mass_logits,
            "support_mass_logits": support_mass_logits,
            "mean_query_mass_entropy": query_mass_entropy.mean(),
            "mean_support_mass_entropy": support_mass_entropy.mean(),
            "mass_regularization": mass_regularization,
            "shot_consistency_loss": shot_consistency_loss,
            "mean_gamma_entropy": plan_entropy.mean(),
            "mean_shot_distance": transport_cost.mean(),
        }
        outputs.update(match_outputs)
        return outputs

    @staticmethod
    def _stack_outputs(batch_outputs: list[dict[str, torch.Tensor]]) -> SPIFOTAOutput:
        stacked: dict[str, Any] = {
            "logits": torch.cat([item["logits"] for item in batch_outputs], dim=0),
            "aux_loss": torch.stack([item["aux_loss"] for item in batch_outputs]).mean(),
            "class_scores": torch.cat([item["class_scores"] for item in batch_outputs], dim=0),
            "total_distance": torch.cat([item["total_distance"] for item in batch_outputs], dim=0),
            "shot_scores": torch.cat([item["shot_scores"] for item in batch_outputs], dim=0),
            "shot_aggregation_weights": torch.cat([item["shot_aggregation_weights"] for item in batch_outputs], dim=0),
            "transport_cost": torch.cat([item["transport_cost"] for item in batch_outputs], dim=0),
            "plan_entropy": torch.cat([item["plan_entropy"] for item in batch_outputs], dim=0),
            "query_tokens": torch.cat([item["query_tokens"] for item in batch_outputs], dim=0),
            "support_tokens": torch.stack([item["support_tokens"] for item in batch_outputs], dim=0),
            "query_masses": torch.cat([item["query_masses"] for item in batch_outputs], dim=0),
            "support_masses": torch.stack([item["support_masses"] for item in batch_outputs], dim=0),
            "query_mass_logits": torch.cat([item["query_mass_logits"] for item in batch_outputs], dim=0),
            "support_mass_logits": torch.stack([item["support_mass_logits"] for item in batch_outputs], dim=0),
            "mean_query_mass_entropy": torch.stack([item["mean_query_mass_entropy"] for item in batch_outputs]).mean(),
            "mean_support_mass_entropy": torch.stack([item["mean_support_mass_entropy"] for item in batch_outputs]).mean(),
            "mass_regularization": torch.stack([item["mass_regularization"] for item in batch_outputs]).mean(),
            "shot_consistency_loss": torch.stack([item["shot_consistency_loss"] for item in batch_outputs]).mean(),
            "mean_gamma_entropy": torch.stack([item["mean_gamma_entropy"] for item in batch_outputs]).mean(),
            "mean_shot_distance": torch.stack([item["mean_shot_distance"] for item in batch_outputs]).mean(),
        }
        if "transport_plan" in batch_outputs[0]:
            stacked["transport_plan"] = torch.cat([item["transport_plan"] for item in batch_outputs], dim=0)
        if "cost_matrix" in batch_outputs[0]:
            stacked["cost_matrix"] = torch.cat([item["cost_matrix"] for item in batch_outputs], dim=0)
        return SPIFOTAOutput(stacked)

    def forward(
        self,
        query: torch.Tensor,
        support: torch.Tensor,
        *,
        query_targets: torch.Tensor | None = None,
        support_targets: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        del support_targets
        batch_size, num_query, _, _, _, _ = self.validate_episode_inputs(query, support)
        query_targets = self._reshape_query_targets(query_targets, batch_size=batch_size, num_query=num_query)

        needs_payload = bool(return_aux or self.training)
        batch_outputs = []
        batch_logits = []

        for batch_idx in range(batch_size):
            outputs = self._forward_episode(
                query=query[batch_idx],
                support=support[batch_idx],
                query_targets=None if query_targets is None else query_targets[batch_idx],
                needs_payload=needs_payload,
                return_aux=return_aux,
            )
            if needs_payload:
                batch_outputs.append(outputs)
                batch_logits.append(outputs["logits"])
            else:
                batch_logits.append(outputs)

        logits = torch.cat(batch_logits, dim=0)
        if not needs_payload:
            return logits

        stacked = self._stack_outputs(batch_outputs)
        stacked["logits"] = logits
        if return_aux:
            return stacked
        return SPIFOTAOutput({"logits": logits, "aux_loss": stacked["aux_loss"]})
