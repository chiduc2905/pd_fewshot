"""Support-Conditioned Latent Flow Inference v3.

Core claim preserved:
- each class is represented as a support-conditioned latent evidence distribution
- query-class scoring is a distribution-fit score

Theory upgrade:
- `v3` treats the class object as a posterior predictive latent evidence
  measure, not as an anchor-plus-generated particle cloud
- the flow is a posterior transport operator over a support-conditioned base
  measure
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from net.fewshot_common import BaseConv64FewShotModel, feature_map_to_tokens, merge_support_tokens
from net.modules.latent_projector_v2 import LatentEvidenceProjectorV2
from net.modules.posterior_context_v3 import PosteriorContextBuilderV3
from net.modules.posterior_losses_v3 import (
    compute_distribution_margin_loss_v3,
    compute_posterior_alignment_loss_v3,
    compute_posterior_flow_matching_loss_v3,
    compute_posterior_regularization_loss_v3,
)
from net.modules.posterior_transport_flow_v3 import PosteriorTransportFlowModelV3
from net.modules.query_conditioned_transport_v3 import QueryConditionedTransportScorerV3
from net.modules.transport_distance_v2 import AlignmentTransportDistanceV2


class SupportConditionedLatentFlowInferenceNetV3(BaseConv64FewShotModel):
    """Posterior evidence transport few-shot classifier."""

    requires_query_targets = True

    def __init__(
        self,
        in_channels: int = 3,
        hidden_dim: int = 64,
        latent_dim: int = 64,
        context_dim: int = 128,
        context_hidden_dim: int | None = None,
        latent_hidden_dim: int | None = None,
        mass_hidden_dim: int | None = None,
        mass_temperature: float = 1.0,
        memory_size: int = 4,
        memory_num_heads: int = 4,
        memory_ffn_multiplier: int = 2,
        prior_num_atoms: int = 4,
        prior_scale: float = 0.5,
        episode_num_heads: int = 4,
        alpha_hidden_dim: int | None = None,
        alpha_shot_scale: float = 1.0,
        alpha_uncertainty_scale: float = 1.0,
        flow_hidden_dim: int = 128,
        flow_time_embedding_dim: int = 32,
        flow_memory_num_heads: int = 4,
        backbone_name: str = "resnet12",
        image_size: int = 64,
        resnet12_drop_rate: float = 0.0,
        resnet12_dropblock_size: int = 5,
        flow_conditioning_type: str = "film",
        use_transport_flow: bool = True,
        use_prior_measure: bool = True,
        use_episode_adapter: bool = True,
        use_query_reweighting: bool = True,
        use_support_barycenter_only: bool = False,
        use_global_proto_branch: bool = False,
        use_align_loss: bool = True,
        use_margin_loss: bool = True,
        train_num_integration_steps: int = 8,
        eval_num_integration_steps: int = 16,
        solver_type: str = "heun",
        fm_time_schedule: str = "uniform",
        score_temperature: float = 8.0,
        query_reweight_temperature: float = 1.0,
        proto_branch_weight: float = 0.1,
        lambda_fm: float = 0.05,
        lambda_align: float = 0.1,
        lambda_margin: float = 0.1,
        lambda_reg: float = 0.05,
        margin_value: float = 0.1,
        support_entropy_target_ratio: float = 0.5,
        query_entropy_target_ratio: float = 0.4,
        relevance_entropy_target_ratio: float = 0.35,
        score_train_num_projections: int = 64,
        score_eval_num_projections: int = 128,
        score_sw_p: float = 2.0,
        score_normalize_inputs: bool = True,
        score_train_projection_mode: str = "resample",
        score_eval_projection_mode: str = "fixed",
        score_eval_num_repeats: int = 1,
        score_projection_seed: int = 7,
        align_distance_type: str = "weighted_entropic_ot",
        align_train_num_projections: int = 64,
        align_eval_num_projections: int = 128,
        align_sw_p: float = 2.0,
        align_normalize_inputs: bool = True,
        align_train_projection_mode: str = "resample",
        align_eval_projection_mode: str = "fixed",
        align_eval_num_repeats: int = 1,
        align_projection_seed: int = 11,
        sinkhorn_epsilon: float = 0.05,
        sinkhorn_iterations: int = 80,
        sinkhorn_cost_power: float = 2.0,
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
        if latent_dim <= 0 or context_dim <= 0:
            raise ValueError("latent_dim and context_dim must be positive")
        if train_num_integration_steps <= 0 or eval_num_integration_steps <= 0:
            raise ValueError("integration steps must be positive")
        if score_temperature <= 0.0 or query_reweight_temperature <= 0.0:
            raise ValueError("temperatures must be positive")
        if not 0.0 <= proto_branch_weight <= 1.0:
            raise ValueError("proto_branch_weight must be in [0, 1]")
        if any(weight < 0.0 for weight in (lambda_fm, lambda_align, lambda_margin, lambda_reg)):
            raise ValueError("loss weights must be non-negative")
        if margin_value < 0.0:
            raise ValueError("margin_value must be non-negative")
        if eps <= 0.0:
            raise ValueError("eps must be positive")

        self.latent_dim = int(latent_dim)
        self.context_dim = int(context_dim)
        self.use_transport_flow = bool(use_transport_flow)
        self.use_prior_measure = bool(use_prior_measure)
        self.use_query_reweighting = bool(use_query_reweighting)
        self.use_global_proto_branch = bool(use_global_proto_branch)
        self.use_align_loss = bool(use_align_loss)
        self.use_margin_loss = bool(use_margin_loss)
        self.train_num_integration_steps = int(train_num_integration_steps)
        self.eval_num_integration_steps = int(eval_num_integration_steps)
        self.solver_type = str(solver_type)
        self.fm_time_schedule = str(fm_time_schedule)
        self.score_temperature = float(score_temperature)
        self.query_reweight_temperature = float(query_reweight_temperature)
        self.proto_branch_weight = float(proto_branch_weight)
        self.lambda_fm = float(lambda_fm)
        self.lambda_align = float(lambda_align)
        self.lambda_margin = float(lambda_margin)
        self.lambda_reg = float(lambda_reg)
        self.margin_value = float(margin_value)
        self.support_entropy_target_ratio = float(support_entropy_target_ratio)
        self.query_entropy_target_ratio = float(query_entropy_target_ratio)
        self.relevance_entropy_target_ratio = float(relevance_entropy_target_ratio)
        self.sinkhorn_epsilon = float(sinkhorn_epsilon)
        self.sinkhorn_iterations = int(sinkhorn_iterations)
        self.sinkhorn_cost_power = float(sinkhorn_cost_power)
        self.align_normalize_inputs = bool(align_normalize_inputs)
        self.eps = float(eps)

        self.latent_projector = LatentEvidenceProjectorV2(
            input_dim=hidden_dim,
            latent_dim=latent_dim,
            hidden_dim=latent_hidden_dim,
            mass_hidden_dim=mass_hidden_dim,
            mass_temperature=mass_temperature,
            eps=eps,
        )
        self.posterior_context = PosteriorContextBuilderV3(
            latent_dim=latent_dim,
            context_dim=context_dim,
            memory_size=memory_size,
            memory_num_heads=memory_num_heads,
            memory_ffn_multiplier=memory_ffn_multiplier,
            summary_hidden_dim=context_hidden_dim,
            mass_feature_dim=context_hidden_dim,
            prior_num_atoms=prior_num_atoms if self.use_prior_measure else 0,
            episode_num_heads=episode_num_heads,
            alpha_hidden_dim=alpha_hidden_dim,
            prior_scale=prior_scale,
            alpha_shot_scale=alpha_shot_scale,
            alpha_uncertainty_scale=alpha_uncertainty_scale,
            use_episode_adapter=use_episode_adapter,
            use_support_barycenter_only=use_support_barycenter_only,
            eps=eps,
        )
        self.flow_model = PosteriorTransportFlowModelV3(
            latent_dim=latent_dim,
            context_dim=context_dim,
            hidden_dim=flow_hidden_dim,
            time_embedding_dim=flow_time_embedding_dim,
            memory_num_heads=flow_memory_num_heads,
            conditioning_type=flow_conditioning_type,
        )
        self.query_transport_scorer = QueryConditionedTransportScorerV3(
            latent_dim=latent_dim,
            context_dim=context_dim,
            hidden_dim=flow_hidden_dim,
            relevance_temperature=query_reweight_temperature,
            use_query_reweighting=use_query_reweighting,
            eps=eps,
            score_train_num_projections=score_train_num_projections,
            score_eval_num_projections=score_eval_num_projections,
            score_sw_p=score_sw_p,
            score_normalize_inputs=score_normalize_inputs,
            score_train_projection_mode=score_train_projection_mode,
            score_eval_projection_mode=score_eval_projection_mode,
            score_eval_num_repeats=score_eval_num_repeats,
            score_projection_seed=score_projection_seed,
        )
        self.alignment_distance = AlignmentTransportDistanceV2(
            distance_type=align_distance_type,
            sw_train_num_projections=align_train_num_projections,
            sw_eval_num_projections=align_eval_num_projections,
            sw_p=align_sw_p,
            sw_normalize_inputs=align_normalize_inputs,
            sw_train_projection_mode=align_train_projection_mode,
            sw_eval_projection_mode=align_eval_projection_mode,
            sw_eval_num_repeats=align_eval_num_repeats,
            sw_projection_seed=align_projection_seed,
            sinkhorn_epsilon=sinkhorn_epsilon,
            sinkhorn_iterations=sinkhorn_iterations,
            sinkhorn_cost_power=sinkhorn_cost_power,
            eps=eps,
        )

    def _encode_episode_tokens(
        self,
        query: torch.Tensor,
        support: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        way_num, shot_num = support.shape[:2]
        query_tokens = feature_map_to_tokens(self.encode(query))
        support_tokens = feature_map_to_tokens(
            self.encode(support.reshape(way_num * shot_num, *support.shape[-3:]))
        ).reshape(way_num, shot_num, -1, self.hidden_dim)
        return query_tokens, support_tokens

    def _build_episode_representations(
        self,
        query: torch.Tensor,
        support: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        way_num, shot_num = support.shape[:2]
        query_tokens, support_tokens = self._encode_episode_tokens(query, support)
        merged_support_tokens = merge_support_tokens(support_tokens, merge_mode="concat")

        query_outputs = self.latent_projector(query_tokens)
        support_outputs = self.latent_projector(merged_support_tokens)

        posterior = self.posterior_context(
            support_outputs["latent_tokens"],
            support_outputs["masses"],
            shot_num=shot_num,
        )
        return {
            "query_tokens": query_tokens,
            "support_tokens": support_tokens,
            "query_latents": query_outputs["latent_tokens"],
            "query_masses": query_outputs["masses"],
            "query_mass_logits": query_outputs["mass_logits"],
            "support_latents": support_outputs["latent_tokens"],
            "support_token_masses": support_outputs["masses"],
            "support_mass_logits": support_outputs["mass_logits"],
            **posterior,
        }

    def _integration_steps(self) -> int:
        return self.train_num_integration_steps if self.training else self.eval_num_integration_steps

    def _shot_profile(self, shot_num: int) -> dict[str, float]:
        """Return an internal few-shot profile derived from the episode shot count.

        Theory motivation:
        - in 1-shot, the posterior base measure is highly uncertain, so scoring should
          stay softer and mass learning should be kept less collapsed;
        - as shots increase, the posterior becomes better estimated, so transport-fit
          losses can be trusted more and entropy floors can relax.

        This is an architectural/objective-level adaptation, not a training recipe trick.
        """
        if shot_num <= 0:
            raise ValueError("shot_num must be positive")

        if shot_num <= 1:
            blend = 0.0
        elif shot_num >= 5:
            blend = 1.0
        else:
            blend = float(shot_num - 1) / 4.0

        one_shot = {
            "score_temperature": 0.85 * self.score_temperature,
            "relevance_temperature": 1.15 * self.query_reweight_temperature,
            "reweight_strength": 0.60,
            "lambda_fm": 0.80 * self.lambda_fm,
            "lambda_align": 0.90 * self.lambda_align,
            "lambda_margin": 1.35 * self.lambda_margin,
            "lambda_reg": 1.35 * self.lambda_reg,
            "margin_value": 1.20 * self.margin_value,
            "support_entropy_target_ratio": max(self.support_entropy_target_ratio, 0.65),
            "query_entropy_target_ratio": max(self.query_entropy_target_ratio, 0.50),
            "relevance_entropy_target_ratio": max(self.relevance_entropy_target_ratio, 0.45),
        }
        multi_shot = {
            "score_temperature": self.score_temperature,
            "relevance_temperature": self.query_reweight_temperature,
            "reweight_strength": 0.30,
            "lambda_fm": 1.10 * self.lambda_fm,
            "lambda_align": 1.10 * self.lambda_align,
            "lambda_margin": self.lambda_margin,
            "lambda_reg": 0.85 * self.lambda_reg,
            "margin_value": self.margin_value,
            "support_entropy_target_ratio": self.support_entropy_target_ratio,
            "query_entropy_target_ratio": self.query_entropy_target_ratio,
            "relevance_entropy_target_ratio": self.relevance_entropy_target_ratio,
        }
        return {
            key: one_shot[key] + blend * (multi_shot[key] - one_shot[key])
            for key in one_shot
        }

    def _compute_global_proto_scores(
        self,
        query_latents: torch.Tensor,
        query_masses: torch.Tensor,
        class_barycenters: torch.Tensor,
        *,
        score_temperature: float,
    ) -> torch.Tensor:
        query_global = F.normalize((query_masses.unsqueeze(-1) * query_latents).sum(dim=1), p=2, dim=-1)
        class_global = F.normalize(class_barycenters, p=2, dim=-1)
        return float(score_temperature) * torch.einsum("qd,wd->qw", query_global, class_global)

    def _transport_posterior_measure(
        self,
        base_atoms: torch.Tensor,
        base_masses: torch.Tensor,
        class_summary: torch.Tensor,
        support_memory: torch.Tensor,
        episode_context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.use_transport_flow:
            return base_atoms, base_masses
        transported_atoms = self.flow_model.transport(
            base_atoms,
            class_summary,
            support_memory,
            episode_context,
            num_steps=self._integration_steps(),
            solver_type=self.solver_type,
        )
        return transported_atoms, base_masses

    def _compute_auxiliary_losses(
        self,
        *,
        support_atoms: torch.Tensor,
        support_masses: torch.Tensor,
        support_memory: torch.Tensor,
        class_summary: torch.Tensor,
        episode_context: torch.Tensor,
        base_atoms: torch.Tensor,
        base_masses: torch.Tensor,
        posterior_atoms: torch.Tensor,
        posterior_masses: torch.Tensor,
        query_masses: torch.Tensor,
        query_conditioned_class_masses: torch.Tensor,
        pairwise_distances: torch.Tensor,
        shot_profile: dict[str, float],
        query_targets: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        zero = support_atoms.new_zeros(())
        fm_loss = (
            compute_posterior_flow_matching_loss_v3(
                self.flow_model,
                base_atoms,
                base_masses,
                support_atoms,
                support_masses,
                class_summary,
                support_memory,
                episode_context,
                sinkhorn_epsilon=self.sinkhorn_epsilon,
                sinkhorn_iterations=self.sinkhorn_iterations,
                sinkhorn_cost_power=self.sinkhorn_cost_power,
                normalize_inputs=self.align_normalize_inputs,
                time_schedule=self.fm_time_schedule,
                eps=self.eps,
            )
            if self.use_transport_flow
            else zero
        )
        align_loss = (
            compute_posterior_alignment_loss_v3(
                posterior_atoms,
                posterior_masses,
                support_atoms,
                support_masses,
                alignment_distance=self.alignment_distance,
            )
            if self.use_align_loss
            else zero
        )
        margin_loss = (
            compute_distribution_margin_loss_v3(
                pairwise_distances,
                query_targets,
                margin=shot_profile["margin_value"],
            )
            if self.use_margin_loss and query_targets is not None
            else zero
        )
        reg_loss = compute_posterior_regularization_loss_v3(
            support_masses,
            query_masses,
            query_conditioned_class_masses,
            support_entropy_target_ratio=shot_profile["support_entropy_target_ratio"],
            query_entropy_target_ratio=shot_profile["query_entropy_target_ratio"],
            relevance_entropy_target_ratio=shot_profile["relevance_entropy_target_ratio"],
            eps=self.eps,
        )
        aux_loss = (
            shot_profile["lambda_fm"] * fm_loss
            + shot_profile["lambda_align"] * align_loss
            + shot_profile["lambda_margin"] * margin_loss
            + shot_profile["lambda_reg"] * reg_loss
        )
        return {
            "fm_loss": fm_loss,
            "align_loss": align_loss,
            "margin_loss": margin_loss,
            "reg_loss": reg_loss,
            "aux_loss": aux_loss,
        }

    def _forward_episode(
        self,
        query: torch.Tensor,
        support: torch.Tensor,
        *,
        query_targets: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        episode = self._build_episode_representations(query, support)
        shot_profile = self._shot_profile(int(support.shape[1]))
        posterior_atoms, posterior_masses = self._transport_posterior_measure(
            episode["base_atoms"],
            episode["base_masses"],
            episode["class_summary"],
            episode["support_memory"],
            episode["episode_context"],
        )
        score_outputs = self.query_transport_scorer.score(
            episode["query_latents"],
            episode["query_masses"],
            posterior_atoms,
            posterior_masses,
            episode["class_summary"],
            episode["episode_context"],
            score_temperature=shot_profile["score_temperature"],
            relevance_temperature=shot_profile["relevance_temperature"],
            reweight_strength=shot_profile["reweight_strength"],
        )
        logits = score_outputs["logits"]
        distribution_scores = logits

        if self.use_global_proto_branch:
            global_proto_scores = self._compute_global_proto_scores(
                episode["query_latents"],
                episode["query_masses"],
                episode["weighted_mean"],
                score_temperature=shot_profile["score_temperature"],
            )
            logits = (1.0 - self.proto_branch_weight) * distribution_scores + self.proto_branch_weight * global_proto_scores
        else:
            global_proto_scores = distribution_scores.new_zeros(distribution_scores.shape)

        needs_payload = bool(return_aux or self.training)
        if not needs_payload:
            return logits

        loss_payload = self._compute_auxiliary_losses(
            support_atoms=episode["support_atoms"],
            support_masses=episode["support_masses"],
            support_memory=episode["support_memory"],
            class_summary=episode["class_summary"],
            episode_context=episode["episode_context"],
            base_atoms=episode["base_atoms"],
            base_masses=episode["base_masses"],
            posterior_atoms=posterior_atoms,
            posterior_masses=posterior_masses,
            query_masses=episode["query_masses"],
            query_conditioned_class_masses=score_outputs["query_conditioned_class_masses"],
            pairwise_distances=score_outputs["pairwise_distances"],
            shot_profile=shot_profile,
            query_targets=query_targets,
        )
        return {
            "logits": logits,
            "pairwise_distances": score_outputs["pairwise_distances"],
            "distribution_scores": distribution_scores,
            "global_proto_scores": global_proto_scores,
            "query_latents": episode["query_latents"],
            "query_masses": episode["query_masses"],
            "support_latents": episode["support_latents"],
            "support_masses": episode["support_masses"],
            "support_token_masses": episode["support_token_masses"],
            "support_atoms": episode["support_atoms"],
            "class_summary": episode["class_summary"],
            "episode_context": episode["episode_context"],
            "support_memory": episode["support_memory"],
            "weighted_mean": episode["weighted_mean"],
            "uncertainty_stats": episode["uncertainty_stats"],
            "prior_atoms": episode["prior_atoms"],
            "prior_masses": episode["prior_masses"],
            "alpha": episode["alpha"],
            "base_atoms": episode["base_atoms"],
            "base_masses": episode["base_masses"],
            "posterior_atoms": posterior_atoms,
            "posterior_masses": posterior_masses,
            "query_conditioned_class_masses": score_outputs["query_conditioned_class_masses"],
            "relevance_entropy": score_outputs["relevance_entropy"],
            "shot_profile_score_temperature": logits.new_tensor(shot_profile["score_temperature"]),
            "shot_profile_relevance_temperature": logits.new_tensor(shot_profile["relevance_temperature"]),
            "shot_profile_lambda_fm": logits.new_tensor(shot_profile["lambda_fm"]),
            "shot_profile_lambda_align": logits.new_tensor(shot_profile["lambda_align"]),
            "shot_profile_lambda_margin": logits.new_tensor(shot_profile["lambda_margin"]),
            "shot_profile_lambda_reg": logits.new_tensor(shot_profile["lambda_reg"]),
            "shot_profile_margin_value": logits.new_tensor(shot_profile["margin_value"]),
            "shot_profile_reweight_strength": logits.new_tensor(shot_profile["reweight_strength"]),
            **loss_payload,
        }

    @staticmethod
    def _stack_batch_outputs(batch_outputs: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        stacked = {
            "logits": torch.cat([item["logits"] for item in batch_outputs], dim=0),
            "pairwise_distances": torch.cat([item["pairwise_distances"] for item in batch_outputs], dim=0),
            "distribution_scores": torch.cat([item["distribution_scores"] for item in batch_outputs], dim=0),
            "global_proto_scores": torch.cat([item["global_proto_scores"] for item in batch_outputs], dim=0),
            "query_latents": torch.stack([item["query_latents"] for item in batch_outputs], dim=0),
            "query_masses": torch.stack([item["query_masses"] for item in batch_outputs], dim=0),
            "support_latents": torch.stack([item["support_latents"] for item in batch_outputs], dim=0),
            "support_masses": torch.stack([item["support_masses"] for item in batch_outputs], dim=0),
            "support_token_masses": torch.stack([item["support_token_masses"] for item in batch_outputs], dim=0),
            "support_atoms": torch.stack([item["support_atoms"] for item in batch_outputs], dim=0),
            "class_summary": torch.stack([item["class_summary"] for item in batch_outputs], dim=0),
            "episode_context": torch.stack([item["episode_context"] for item in batch_outputs], dim=0),
            "support_memory": torch.stack([item["support_memory"] for item in batch_outputs], dim=0),
            "weighted_mean": torch.stack([item["weighted_mean"] for item in batch_outputs], dim=0),
            "uncertainty_stats": torch.stack([item["uncertainty_stats"] for item in batch_outputs], dim=0),
            "prior_atoms": torch.stack([item["prior_atoms"] for item in batch_outputs], dim=0),
            "prior_masses": torch.stack([item["prior_masses"] for item in batch_outputs], dim=0),
            "alpha": torch.stack([item["alpha"] for item in batch_outputs], dim=0),
            "base_atoms": torch.stack([item["base_atoms"] for item in batch_outputs], dim=0),
            "base_masses": torch.stack([item["base_masses"] for item in batch_outputs], dim=0),
            "posterior_atoms": torch.stack([item["posterior_atoms"] for item in batch_outputs], dim=0),
            "posterior_masses": torch.stack([item["posterior_masses"] for item in batch_outputs], dim=0),
            "query_conditioned_class_masses": torch.stack(
                [item["query_conditioned_class_masses"] for item in batch_outputs],
                dim=0,
            ),
            "relevance_entropy": torch.stack([item["relevance_entropy"] for item in batch_outputs], dim=0),
        }
        for scalar_name in (
            "shot_profile_score_temperature",
            "shot_profile_relevance_temperature",
            "shot_profile_lambda_fm",
            "shot_profile_lambda_align",
            "shot_profile_lambda_margin",
            "shot_profile_lambda_reg",
            "shot_profile_margin_value",
            "shot_profile_reweight_strength",
        ):
            stacked[scalar_name] = torch.stack([item[scalar_name] for item in batch_outputs], dim=0)
        for scalar_name in ("fm_loss", "align_loss", "margin_loss", "reg_loss", "aux_loss"):
            stacked[scalar_name] = torch.stack([item[scalar_name] for item in batch_outputs], dim=0).mean()
        return stacked

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
                raise ValueError("Flat query_targets must have length batch_size * num_query")
            return query_targets.reshape(batch_size, num_query)
        if query_targets.dim() == 2 and tuple(query_targets.shape) == (batch_size, num_query):
            return query_targets
        raise ValueError("query_targets must have shape (Batch * NumQuery,) or (Batch, NumQuery)")

    def forward(
        self,
        query: torch.Tensor,
        support: torch.Tensor,
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

        stacked = self._stack_batch_outputs(batch_outputs)
        stacked["logits"] = logits
        if return_aux:
            return stacked
        return {
            "logits": logits,
            "aux_loss": stacked["aux_loss"],
            "fm_loss": stacked["fm_loss"],
            "align_loss": stacked["align_loss"],
            "margin_loss": stacked["margin_loss"],
            "reg_loss": stacked["reg_loss"],
        }
