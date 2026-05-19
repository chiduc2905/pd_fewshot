"""Multi-mass Partial OT evidence matcher for few-shot scalograms."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from net.fewshot_common import BaseConv64FewShotModel, feature_map_to_tokens, merge_support_tokens
from net.modules.partial_ot import (
    compute_partial_transport_cost,
    compute_partial_transported_mass,
    partial_transport_plan_residuals,
    solve_partial_transport,
)


def parse_mass_bank(value: str | list[float] | tuple[float, ...]) -> tuple[float, ...]:
    if isinstance(value, str):
        masses = [float(item.strip()) for item in value.split(",") if item.strip()]
    else:
        masses = [float(item) for item in value]
    if not masses:
        raise ValueError("spot_mass_bank must contain at least one mass")
    unique: list[float] = []
    for mass in masses:
        if not 0.0 < mass <= 1.0:
            raise ValueError("all spot_mass_bank values must be in (0, 1]")
        if not any(abs(mass - kept) <= 1e-8 for kept in unique):
            unique.append(mass)
    return tuple(unique)


class MMSPOTFSLResult(dict):
    """Dict-like output that exposes `.shape` through logits."""

    @property
    def shape(self):
        logits = self.get("logits")
        return None if logits is None else logits.shape


class MMSPOTFSL(BaseConv64FewShotModel):
    """Multi-mass Partial OT evidence classifier.

    The default path has no auxiliary loss and no learned controller. It solves
    entropic partial OT for a small fixed mass bank, scores each mass by
    normalized transport cost, then aggregates the scores.
    """

    VALID_AGGREGATIONS = {"softmax", "mean", "logmeanexp", "max"}

    def __init__(
        self,
        in_channels: int = 3,
        hidden_dim: int = 640,
        token_dim: int = 128,
        backbone_name: str = "resnet12",
        image_size: int = 84,
        resnet12_drop_rate: float = 0.0,
        resnet12_dropblock_size: int = 5,
        spot_mass_bank: str | list[float] | tuple[float, ...] = "0.3,0.4,0.5",
        sinkhorn_epsilon: float = 0.05,
        sinkhorn_iterations: int = 120,
        sinkhorn_tolerance: float = 1e-6,
        score_scale: float = 16.0,
        mass_aggregation: str = "softmax",
        mass_temperature: float = 1.0,
        partial_backend: str = "native",
        use_controller: bool = False,
        controller_hidden_dim: int = 32,
        proto_weight: float = 0.0,
        support_merge_mode: str = "concat",
        normalize_tokens: bool = True,
        return_transport_plan: bool = False,
        eps: float = 1e-8,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            backbone_name=backbone_name,
            image_size=image_size,
            resnet12_drop_rate=resnet12_drop_rate,
            resnet12_dropblock_size=resnet12_dropblock_size,
        )
        if int(token_dim) <= 0:
            raise ValueError("token_dim must be positive")
        if float(sinkhorn_epsilon) <= 0.0:
            raise ValueError("sinkhorn_epsilon must be positive")
        if int(sinkhorn_iterations) <= 0:
            raise ValueError("sinkhorn_iterations must be positive")
        if float(sinkhorn_tolerance) < 0.0:
            raise ValueError("sinkhorn_tolerance must be non-negative")
        if float(score_scale) <= 0.0:
            raise ValueError("score_scale must be positive")
        if float(mass_temperature) <= 0.0:
            raise ValueError("mass_temperature must be positive")
        mass_aggregation = str(mass_aggregation).lower()
        if mass_aggregation not in self.VALID_AGGREGATIONS:
            raise ValueError(f"mass_aggregation must be one of {sorted(self.VALID_AGGREGATIONS)}")
        partial_backend = str(partial_backend).lower().replace("-", "_")
        partial_backend_aliases = {
            "native": "native",
            "pot": "pot",
            "pot_torch": "pot",
            "torch_pot": "pot",
        }
        if partial_backend not in partial_backend_aliases:
            raise ValueError("partial_backend must be one of {'native', 'pot', 'pot_torch'}")
        partial_backend = partial_backend_aliases[partial_backend]
        if support_merge_mode not in {"concat", "mean"}:
            raise ValueError("support_merge_mode must be 'concat' or 'mean'")
        if int(controller_hidden_dim) <= 0:
            raise ValueError("controller_hidden_dim must be positive")
        if float(eps) <= 0.0:
            raise ValueError("eps must be positive")

        self.token_dim = int(token_dim)
        self.spot_mass_bank = parse_mass_bank(spot_mass_bank)
        self.sinkhorn_epsilon = float(sinkhorn_epsilon)
        self.sinkhorn_iterations = int(sinkhorn_iterations)
        self.sinkhorn_tolerance = float(sinkhorn_tolerance)
        self.score_scale = float(score_scale)
        self.mass_aggregation = mass_aggregation
        self.mass_temperature = float(mass_temperature)
        self.partial_backend = partial_backend
        self.use_controller = bool(use_controller)
        self.proto_weight = float(proto_weight)
        self.support_merge_mode = str(support_merge_mode)
        self.normalize_tokens = bool(normalize_tokens)
        self.return_transport_plan = bool(return_transport_plan)
        self.eps = float(eps)

        self.register_buffer(
            "mass_bank_tensor",
            torch.tensor(self.spot_mass_bank, dtype=torch.float32),
            persistent=True,
        )
        self.token_projector = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, self.token_dim, bias=False),
        )
        self.controller = (
            nn.Sequential(
                nn.LayerNorm(6),
                nn.Linear(6, int(controller_hidden_dim)),
                nn.GELU(),
                nn.Linear(int(controller_hidden_dim), 1),
            )
            if self.use_controller
            else None
        )

    @property
    def num_masses(self) -> int:
        return int(self.mass_bank_tensor.numel())

    def _encode_images(self, images: torch.Tensor) -> torch.Tensor:
        tokens = feature_map_to_tokens(self.encode(images))
        tokens = self.token_projector(tokens)
        if self.normalize_tokens:
            tokens = F.normalize(tokens, p=2, dim=-1, eps=self.eps)
        return tokens

    def _encode_episode(self, query: torch.Tensor, support: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if query.dim() != 4:
            raise ValueError(f"query episode must have shape (NumQuery, C, H, W), got {tuple(query.shape)}")
        if support.dim() != 5:
            raise ValueError(
                "support episode must have shape (Way, Shot, C, H, W), "
                f"got {tuple(support.shape)}"
            )
        way_num, shot_num = support.shape[:2]
        images = torch.cat([query, support.reshape(way_num * shot_num, *support.shape[2:])], dim=0)
        tokens = self._encode_images(images)
        query_tokens = tokens[: query.shape[0]]
        support_tokens = tokens[query.shape[0] :].reshape(way_num, shot_num, tokens.shape[-2], tokens.shape[-1])
        return query_tokens, support_tokens

    @staticmethod
    def _pairwise_squared_distance(query_tokens: torch.Tensor, class_tokens: torch.Tensor) -> torch.Tensor:
        if query_tokens.dim() != 3:
            raise ValueError(
                "query_tokens must have shape (NumQuery, QueryTokens, Dim), "
                f"got {tuple(query_tokens.shape)}"
            )
        if class_tokens.dim() != 3:
            raise ValueError(
                "class_tokens must have shape (Way, SupportTokens, Dim), "
                f"got {tuple(class_tokens.shape)}"
            )
        if query_tokens.shape[-1] != class_tokens.shape[-1]:
            raise ValueError(
                "query/support token dimensions differ: "
                f"{query_tokens.shape[-1]} vs {class_tokens.shape[-1]}"
            )
        diff = query_tokens[:, None, :, None, :] - class_tokens[None, :, None, :, :]
        return diff.square().sum(dim=-1).clamp_min(0.0)

    def _uniform_marginals(self, cost: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        leading_shape = cost.shape[:-2]
        query_tokens = cost.shape[-2]
        support_tokens = cost.shape[-1]
        a = cost.new_full((*leading_shape, query_tokens), 1.0 / float(query_tokens))
        b = cost.new_full((*leading_shape, support_tokens), 1.0 / float(support_tokens))
        return a, b

    def _aggregate_mass_scores(
        self,
        mass_scores: torch.Tensor,
        controller_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.controller is not None:
            logits = self.controller(controller_features).squeeze(-1)
            mass_weights = torch.softmax(logits, dim=-1)
            return (mass_weights * mass_scores).sum(dim=-1), mass_weights

        if self.mass_aggregation == "mean":
            mass_weights = torch.full_like(mass_scores, 1.0 / float(mass_scores.shape[-1]))
            return mass_scores.mean(dim=-1), mass_weights
        if self.mass_aggregation == "max":
            indices = mass_scores.argmax(dim=-1, keepdim=True)
            mass_weights = torch.zeros_like(mass_scores).scatter_(-1, indices, 1.0)
            return mass_scores.gather(-1, indices).squeeze(-1), mass_weights
        if self.mass_aggregation == "logmeanexp":
            temperature = float(self.mass_temperature)
            logits = temperature * (
                torch.logsumexp(mass_scores / temperature, dim=-1) - math.log(float(mass_scores.shape[-1]))
            )
            mass_weights = torch.softmax(mass_scores / temperature, dim=-1)
            return logits, mass_weights

        mass_weights = torch.softmax(mass_scores / float(self.mass_temperature), dim=-1)
        return (mass_weights * mass_scores).sum(dim=-1), mass_weights

    def compute_partial_match(
        self,
        query_tokens: torch.Tensor,
        class_tokens: torch.Tensor,
    ) -> MMSPOTFSLResult:
        """Compute multi-mass SPOT scores for one episode from projected tokens."""
        cost = self._pairwise_squared_distance(query_tokens, class_tokens)
        num_query, way_num, query_len, support_len = cost.shape
        num_pairs = num_query * way_num
        num_masses = self.num_masses
        a, b = self._uniform_marginals(cost)

        pair_cost = cost.reshape(num_pairs, query_len, support_len)
        pair_a = a.reshape(num_pairs, query_len)
        pair_b = b.reshape(num_pairs, support_len)
        flat_cost = (
            pair_cost[:, None, :, :]
            .expand(num_pairs, num_masses, query_len, support_len)
            .reshape(num_pairs * num_masses, query_len, support_len)
            .float()
        )
        flat_a = (
            pair_a[:, None, :]
            .expand(num_pairs, num_masses, query_len)
            .reshape(num_pairs * num_masses, query_len)
            .float()
        )
        flat_b = (
            pair_b[:, None, :]
            .expand(num_pairs, num_masses, support_len)
            .reshape(num_pairs * num_masses, support_len)
            .float()
        )
        mass_bank = self.mass_bank_tensor.to(device=flat_cost.device, dtype=flat_cost.dtype)
        flat_mass = mass_bank.reshape(1, num_masses).expand(num_pairs, num_masses).reshape(-1)

        flat_plan = solve_partial_transport(
            flat_cost,
            flat_a,
            flat_b,
            transport_mass=flat_mass,
            backend=self.partial_backend,
            reg=self.sinkhorn_epsilon,
            max_iter=self.sinkhorn_iterations,
            tol=self.sinkhorn_tolerance,
            eps=self.eps,
        )
        flat_transport_cost = compute_partial_transport_cost(flat_plan, flat_cost)
        flat_transport_mass = compute_partial_transported_mass(flat_plan)
        residuals = partial_transport_plan_residuals(flat_plan, flat_a, flat_b, flat_mass)

        plan_bank = flat_plan.reshape(num_query, way_num, num_masses, query_len, support_len).to(dtype=cost.dtype)
        transport_cost_bank = flat_transport_cost.reshape(num_query, way_num, num_masses).to(dtype=cost.dtype)
        transported_mass_bank = flat_transport_mass.reshape(num_query, way_num, num_masses).to(dtype=cost.dtype)
        mass_bank_view = self.mass_bank_tensor.to(device=cost.device, dtype=cost.dtype).view(1, 1, num_masses)
        normalized_cost_bank = transport_cost_bank / mass_bank_view.clamp_min(self.eps)
        mass_scores = -self.score_scale * normalized_cost_bank

        query_evidence_bank = plan_bank.sum(dim=-1)
        support_evidence_bank = plan_bank.sum(dim=-2)
        query_weight = query_evidence_bank / query_evidence_bank.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        support_weight = support_evidence_bank / support_evidence_bank.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        query_evidence_repr = torch.einsum("qwkl,qld->qwkd", query_weight, query_tokens)
        support_evidence_repr = torch.einsum("qwks,wsd->qwkd", support_weight, class_tokens)
        proto_similarity = F.cosine_similarity(
            query_evidence_repr,
            support_evidence_repr,
            dim=-1,
            eps=self.eps,
        )
        if self.proto_weight != 0.0:
            mass_scores = mass_scores + float(self.proto_weight) * proto_similarity

        plan_prob = plan_bank / mass_bank_view[..., None, None].clamp_min(self.eps)
        plan_entropy = -(plan_prob.clamp_min(self.eps) * plan_prob.clamp_min(self.eps).log()).sum(dim=(-1, -2))
        plan_entropy = plan_entropy / math.log(float(max(query_len * support_len, 2)))
        query_coverage = 1.0 / (
            query_len * query_weight.square().sum(dim=-1).clamp_min(self.eps)
        )
        support_coverage = 1.0 / (
            support_len * support_weight.square().sum(dim=-1).clamp_min(self.eps)
        )
        controller_features = torch.stack(
            [
                normalized_cost_bank,
                plan_entropy,
                query_coverage,
                support_coverage,
                proto_similarity,
                mass_bank_view.expand_as(normalized_cost_bank),
            ],
            dim=-1,
        )
        logits, mass_weights = self._aggregate_mass_scores(mass_scores, controller_features)
        transport_cost = (mass_weights * transport_cost_bank).sum(dim=-1)
        transported_mass = (mass_weights * transported_mass_bank).sum(dim=-1)
        total_distance = (mass_weights * normalized_cost_bank).sum(dim=-1)

        plan_residuals_bank = torch.stack(
            [
                residuals["row_violation"],
                residuals["column_violation"],
                residuals["mass_residual"],
            ],
            dim=-1,
        ).reshape(num_query, way_num, num_masses, 3)
        plan_residuals_bank = plan_residuals_bank.to(device=cost.device, dtype=cost.dtype)

        output: dict[str, Any] = {
            "logits": logits,
            "class_scores": logits,
            "aux_loss": logits.new_zeros(()),
            "mass_scores": mass_scores,
            "mass_weights": mass_weights,
            "mass_bank": self.mass_bank_tensor.to(device=cost.device, dtype=cost.dtype),
            "total_distance": total_distance,
            "transport_cost": transport_cost,
            "transported_mass": transported_mass,
            "transport_cost_bank": transport_cost_bank.detach(),
            "transported_mass_bank": transported_mass_bank.detach(),
            "normalized_cost_bank": normalized_cost_bank.detach(),
            "query_evidence_bank": query_evidence_bank.detach(),
            "support_evidence_bank": support_evidence_bank.detach(),
            "query_evidence_repr": query_evidence_repr.detach(),
            "support_evidence_repr": support_evidence_repr.detach(),
            "proto_similarity": proto_similarity.detach(),
            "plan_entropy": plan_entropy.detach(),
            "query_coverage": query_coverage.detach(),
            "support_coverage": support_coverage.detach(),
            "plan_residuals_bank": plan_residuals_bank.detach(),
            "plan_row_violation": plan_residuals_bank[..., 0].detach(),
            "plan_column_violation": plan_residuals_bank[..., 1].detach(),
            "plan_mass_error": plan_residuals_bank[..., 2].detach(),
        }
        if self.return_transport_plan:
            output["transport_plan_bank"] = plan_bank.detach()
            output["cost_matrix"] = cost.detach()
        return MMSPOTFSLResult(output)

    def _forward_episode(self, query: torch.Tensor, support: torch.Tensor) -> MMSPOTFSLResult:
        query_tokens, support_tokens = self._encode_episode(query, support)
        class_tokens = merge_support_tokens(support_tokens, merge_mode=self.support_merge_mode)
        outputs = self.compute_partial_match(query_tokens, class_tokens)
        outputs["query_tokens"] = query_tokens.detach()
        outputs["support_tokens"] = class_tokens.detach()
        return outputs

    @staticmethod
    def _stack_outputs(batch_outputs: list[MMSPOTFSLResult]) -> MMSPOTFSLResult:
        stacked: dict[str, Any] = {
            "logits": torch.cat([item["logits"] for item in batch_outputs], dim=0),
            "aux_loss": torch.stack([item["aux_loss"] for item in batch_outputs]).mean(),
            "class_scores": torch.cat([item["class_scores"] for item in batch_outputs], dim=0),
            "mass_scores": torch.cat([item["mass_scores"] for item in batch_outputs], dim=0),
            "mass_weights": torch.cat([item["mass_weights"] for item in batch_outputs], dim=0),
            "mass_bank": batch_outputs[0]["mass_bank"],
            "total_distance": torch.cat([item["total_distance"] for item in batch_outputs], dim=0),
            "transport_cost": torch.cat([item["transport_cost"] for item in batch_outputs], dim=0),
            "transported_mass": torch.cat([item["transported_mass"] for item in batch_outputs], dim=0),
            "transport_cost_bank": torch.cat([item["transport_cost_bank"] for item in batch_outputs], dim=0),
            "transported_mass_bank": torch.cat([item["transported_mass_bank"] for item in batch_outputs], dim=0),
            "normalized_cost_bank": torch.cat([item["normalized_cost_bank"] for item in batch_outputs], dim=0),
            "query_evidence_bank": torch.cat([item["query_evidence_bank"] for item in batch_outputs], dim=0),
            "support_evidence_bank": torch.cat([item["support_evidence_bank"] for item in batch_outputs], dim=0),
            "query_evidence_repr": torch.cat([item["query_evidence_repr"] for item in batch_outputs], dim=0),
            "support_evidence_repr": torch.cat([item["support_evidence_repr"] for item in batch_outputs], dim=0),
            "proto_similarity": torch.cat([item["proto_similarity"] for item in batch_outputs], dim=0),
            "plan_entropy": torch.cat([item["plan_entropy"] for item in batch_outputs], dim=0),
            "query_coverage": torch.cat([item["query_coverage"] for item in batch_outputs], dim=0),
            "support_coverage": torch.cat([item["support_coverage"] for item in batch_outputs], dim=0),
            "plan_residuals_bank": torch.cat([item["plan_residuals_bank"] for item in batch_outputs], dim=0),
            "plan_row_violation": torch.cat([item["plan_row_violation"] for item in batch_outputs], dim=0),
            "plan_column_violation": torch.cat([item["plan_column_violation"] for item in batch_outputs], dim=0),
            "plan_mass_error": torch.cat([item["plan_mass_error"] for item in batch_outputs], dim=0),
        }
        for key in ("transport_plan_bank", "cost_matrix", "query_tokens", "support_tokens"):
            if key in batch_outputs[0]:
                stacked[key] = torch.cat([item[key] for item in batch_outputs], dim=0)
        return MMSPOTFSLResult(stacked)

    def forward(
        self,
        query: torch.Tensor,
        support: torch.Tensor,
        return_aux: bool = False,
    ) -> MMSPOTFSLResult:
        del return_aux
        bsz, _, _, _, _, _ = self.validate_episode_inputs(query, support)
        batch_outputs = [self._forward_episode(query[idx], support[idx]) for idx in range(bsz)]
        return self._stack_outputs(batch_outputs)
