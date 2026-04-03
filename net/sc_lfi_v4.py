"""Support-Conditioned Latent Flow Inference v4.

Core claim preserved:
- each class is represented as a support-conditioned latent evidence distribution
- query-class scoring is a distribution-fit problem

Theory upgrade in `v4`:
- support is preserved hierarchically at the shot level before class aggregation;
- class posterior base measures shrink empirical support-shot evidence toward a
  learned global meta-prior dictionary;
- query-class interaction modulates the transport metric, not the class
  posterior masses themselves;
- leave-one-shot-out predictive supervision is added to make the model more
  genuinely few-shot-specific.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from net.fewshot_common import BaseConv64FewShotModel, feature_map_to_tokens
from net.modules.latent_projector_v2 import LatentEvidenceProjectorV2
from net.modules.posterior_context_v4 import HierarchicalPosteriorContextBuilderV4
from net.modules.posterior_losses_v4 import (
    compute_distribution_margin_loss_v4,
    compute_hierarchical_alignment_loss_v4,
    compute_hierarchical_posterior_flow_matching_loss_v4,
    compute_hierarchical_regularization_loss_v4,
)
from net.modules.posterior_transport_flow_v4 import PosteriorTransportFlowModelV4
from net.modules.query_conditioned_transport_v4 import QueryConditionedTransportScorerV4
from net.modules.transport_distance_v2 import AlignmentTransportDistanceV2


class SupportConditionedLatentFlowInferenceNetV4(BaseConv64FewShotModel):
    """Hierarchical predictive posterior transport few-shot classifier."""

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
        shot_memory_size: int = 2,
        class_memory_size: int = 4,
        memory_num_heads: int = 4,
        memory_ffn_multiplier: int = 2,
        prior_num_atoms: int = 4,
        global_prior_size: int = 16,
        prior_scale: float = 0.25,
        episode_num_heads: int = 4,
        alpha_hidden_dim: int | None = None,
        shrinkage_kappa: float = 2.0,
        alpha_uncertainty_scale: float = 1.0,
        use_shot_barycenter_only: bool = False,
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
        use_metric_conditioning: bool = True,
        use_align_loss: bool = True,
        use_margin_loss: bool = True,
        use_loo_loss: bool = True,
        train_num_integration_steps: int = 4,
        eval_num_integration_steps: int = 8,
        solver_type: str = "heun",
        fm_time_schedule: str = "uniform",
        score_temperature: float = 8.0,
        lambda_fm: float = 0.03,
        lambda_align: float = 0.05,
        lambda_margin: float = 0.1,
        lambda_loo: float = 0.1,
        lambda_reg: float = 0.05,
        margin_value: float = 0.1,
        support_token_entropy_target_ratio: float = 0.55,
        query_entropy_target_ratio: float = 0.45,
        shot_entropy_target_ratio: float = 0.55,
        prior_entropy_target_ratio: float = 0.45,
        score_distance_type: str = "weighted_entropic_ot",
        score_train_num_projections: int = 64,
        score_eval_num_projections: int = 128,
        score_sw_p: float = 2.0,
        score_normalize_inputs: bool = True,
        score_train_projection_mode: str = "resample",
        score_eval_projection_mode: str = "fixed",
        score_eval_num_repeats: int = 1,
        score_projection_seed: int = 7,
        score_sinkhorn_epsilon: float = 0.05,
        score_sinkhorn_iterations: int = 60,
        score_sinkhorn_cost_power: float = 2.0,
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
        if score_temperature <= 0.0:
            raise ValueError("score_temperature must be positive")
        if any(weight < 0.0 for weight in (lambda_fm, lambda_align, lambda_margin, lambda_loo, lambda_reg)):
            raise ValueError("loss weights must be non-negative")
        if margin_value < 0.0:
            raise ValueError("margin_value must be non-negative")
        if eps <= 0.0:
            raise ValueError("eps must be positive")

        self.latent_dim = int(latent_dim)
        self.context_dim = int(context_dim)
        self.use_transport_flow = bool(use_transport_flow)
        self.use_prior_measure = bool(use_prior_measure)
        self.use_metric_conditioning = bool(use_metric_conditioning)
        self.use_align_loss = bool(use_align_loss)
        self.use_margin_loss = bool(use_margin_loss)
        self.use_loo_loss = bool(use_loo_loss)
        self.train_num_integration_steps = int(train_num_integration_steps)
        self.eval_num_integration_steps = int(eval_num_integration_steps)
        self.solver_type = str(solver_type)
        self.fm_time_schedule = str(fm_time_schedule)
        self.score_temperature = float(score_temperature)
        self.lambda_fm = float(lambda_fm)
        self.lambda_align = float(lambda_align)
        self.lambda_margin = float(lambda_margin)
        self.lambda_loo = float(lambda_loo)
        self.lambda_reg = float(lambda_reg)
        self.margin_value = float(margin_value)
        self.support_token_entropy_target_ratio = float(support_token_entropy_target_ratio)
        self.query_entropy_target_ratio = float(query_entropy_target_ratio)
        self.shot_entropy_target_ratio = float(shot_entropy_target_ratio)
        self.prior_entropy_target_ratio = float(prior_entropy_target_ratio)
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
        self.posterior_context = HierarchicalPosteriorContextBuilderV4(
            latent_dim=latent_dim,
            context_dim=context_dim,
            shot_memory_size=shot_memory_size,
            class_memory_size=class_memory_size,
            memory_num_heads=memory_num_heads,
            memory_ffn_multiplier=memory_ffn_multiplier,
            summary_hidden_dim=context_hidden_dim,
            prior_num_atoms=prior_num_atoms if self.use_prior_measure else 0,
            global_prior_size=global_prior_size,
            episode_num_heads=episode_num_heads,
            alpha_hidden_dim=alpha_hidden_dim,
            prior_scale=prior_scale,
            shrinkage_kappa=shrinkage_kappa,
            alpha_uncertainty_scale=alpha_uncertainty_scale,
            use_episode_adapter=use_episode_adapter,
            use_shot_barycenter_only=use_shot_barycenter_only,
            eps=eps,
        )
        self.flow_model = PosteriorTransportFlowModelV4(
            latent_dim=latent_dim,
            context_dim=context_dim,
            hidden_dim=flow_hidden_dim,
            time_embedding_dim=flow_time_embedding_dim,
            memory_num_heads=flow_memory_num_heads,
            conditioning_type=flow_conditioning_type,
        )
        self.query_transport_scorer = QueryConditionedTransportScorerV4(
            latent_dim=latent_dim,
            context_dim=context_dim,
            hidden_dim=flow_hidden_dim,
            use_metric_conditioning=use_metric_conditioning,
            distance_type=score_distance_type,
            train_num_projections=score_train_num_projections,
            eval_num_projections=score_eval_num_projections,
            sw_p=score_sw_p,
            normalize_inputs=score_normalize_inputs,
            train_projection_mode=score_train_projection_mode,
            eval_projection_mode=score_eval_projection_mode,
            eval_num_repeats=score_eval_num_repeats,
            projection_seed=score_projection_seed,
            sinkhorn_epsilon=score_sinkhorn_epsilon,
            sinkhorn_iterations=score_sinkhorn_iterations,
            sinkhorn_cost_power=score_sinkhorn_cost_power,
            eps=eps,
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

        query_outputs = self.latent_projector(query_tokens)
        support_outputs = self.latent_projector(support_tokens)
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

    def _transport_posterior_measure(
        self,
        base_atoms: torch.Tensor,
        base_masses: torch.Tensor,
        class_summary: torch.Tensor,
        class_memory: torch.Tensor,
        episode_context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.use_transport_flow:
            return base_atoms, base_masses
        transported_atoms = self.flow_model.transport(
            base_atoms,
            class_summary,
            class_memory,
            episode_context,
            num_steps=self._integration_steps(),
            solver_type=self.solver_type,
        )
        return transported_atoms, base_masses

    def _compute_leave_one_out_loss(
        self,
        episode: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        support_latents = episode["support_latents"]
        support_token_masses = episode["support_token_masses"]
        shot_atoms = episode["shot_atoms"]
        shot_basis_masses = episode["shot_basis_masses"]
        shot_num = support_latents.shape[1]
        if not self.use_loo_loss or shot_num < 2:
            return support_latents.new_zeros(())

        way_num = support_latents.shape[0]
        targets = torch.arange(way_num, device=support_latents.device, dtype=torch.long)
        ce_losses = []
        margin_losses = []
        for held_out_idx in range(shot_num):
            keep_mask = torch.ones(shot_num, device=support_latents.device, dtype=torch.bool)
            keep_mask[held_out_idx] = False
            subset_context = self.posterior_context(
                support_latents[:, keep_mask],
                support_token_masses[:, keep_mask],
                shot_num=shot_num - 1,
            )
            subset_atoms, subset_masses = self._transport_posterior_measure(
                subset_context["base_atoms"],
                subset_context["base_masses"],
                subset_context["class_summary"],
                subset_context["class_memory"],
                subset_context["episode_context"],
            )
            pseudo_scores = self.query_transport_scorer.score(
                shot_atoms[:, held_out_idx],
                shot_basis_masses[:, held_out_idx],
                subset_atoms,
                subset_masses,
                subset_context["class_summary"],
                subset_context["episode_context"],
                score_temperature=self.score_temperature,
            )
            ce_losses.append(F.cross_entropy(pseudo_scores["logits"], targets))
            margin_losses.append(
                compute_distribution_margin_loss_v4(
                    pseudo_scores["pairwise_distances"],
                    targets,
                    margin=self.margin_value,
                )
            )
        return torch.stack(ce_losses).mean() + 0.5 * torch.stack(margin_losses).mean()

    def _compute_auxiliary_losses(
        self,
        episode: dict[str, torch.Tensor],
        posterior_atoms: torch.Tensor,
        posterior_masses: torch.Tensor,
        pairwise_distances: torch.Tensor,
        query_targets: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        zero = posterior_atoms.new_zeros(())
        fm_loss = (
            compute_hierarchical_posterior_flow_matching_loss_v4(
                self.flow_model,
                episode["base_atoms"],
                episode["base_masses"],
                episode["support_atoms"],
                episode["support_masses"],
                episode["class_summary"],
                episode["class_memory"],
                episode["episode_context"],
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
            compute_hierarchical_alignment_loss_v4(
                posterior_atoms,
                posterior_masses,
                episode["support_atoms"],
                episode["support_masses"],
                alignment_distance=self.alignment_distance,
            )
            if self.use_align_loss
            else zero
        )
        margin_loss = (
            compute_distribution_margin_loss_v4(
                pairwise_distances,
                query_targets,
                margin=self.margin_value,
            )
            if self.use_margin_loss and query_targets is not None
            else zero
        )
        loo_loss = self._compute_leave_one_out_loss(episode)
        reg_loss = compute_hierarchical_regularization_loss_v4(
            episode["support_token_masses"],
            episode["query_masses"],
            episode["shot_masses"],
            episode["prior_masses"],
            support_token_entropy_target_ratio=self.support_token_entropy_target_ratio,
            query_entropy_target_ratio=self.query_entropy_target_ratio,
            shot_entropy_target_ratio=self.shot_entropy_target_ratio,
            prior_entropy_target_ratio=self.prior_entropy_target_ratio,
            eps=self.eps,
        )
        aux_loss = (
            self.lambda_fm * fm_loss
            + self.lambda_align * align_loss
            + self.lambda_margin * margin_loss
            + self.lambda_loo * loo_loss
            + self.lambda_reg * reg_loss
        )
        return {
            "fm_loss": fm_loss,
            "align_loss": align_loss,
            "margin_loss": margin_loss,
            "loo_loss": loo_loss,
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
        posterior_atoms, posterior_masses = self._transport_posterior_measure(
            episode["base_atoms"],
            episode["base_masses"],
            episode["class_summary"],
            episode["class_memory"],
            episode["episode_context"],
        )
        score_outputs = self.query_transport_scorer.score(
            episode["query_latents"],
            episode["query_masses"],
            posterior_atoms,
            posterior_masses,
            episode["class_summary"],
            episode["episode_context"],
            score_temperature=self.score_temperature,
        )
        logits = score_outputs["logits"]

        needs_payload = bool(return_aux or self.training)
        if not needs_payload:
            return logits

        loss_payload = self._compute_auxiliary_losses(
            episode=episode,
            posterior_atoms=posterior_atoms,
            posterior_masses=posterior_masses,
            pairwise_distances=score_outputs["pairwise_distances"],
            query_targets=query_targets,
        )
        return {
            "logits": logits,
            "pairwise_distances": score_outputs["pairwise_distances"],
            "distribution_scores": logits,
            "query_latents": episode["query_latents"],
            "query_masses": episode["query_masses"],
            "support_latents": episode["support_latents"],
            "support_token_masses": episode["support_token_masses"],
            "shot_atoms": episode["shot_atoms"],
            "shot_basis_masses": episode["shot_basis_masses"],
            "shot_masses": episode["shot_masses"],
            "support_atoms": episode["support_atoms"],
            "support_masses": episode["support_masses"],
            "class_summary": episode["class_summary"],
            "episode_context": episode["episode_context"],
            "class_memory": episode["class_memory"],
            "shot_weighted_mean": episode["shot_weighted_mean"],
            "shot_weighted_std": episode["shot_weighted_std"],
            "shot_uncertainty": episode["shot_uncertainty"],
            "class_uncertainty": episode["class_uncertainty"],
            "prior_atoms": episode["prior_atoms"],
            "prior_masses": episode["prior_masses"],
            "alpha": episode["alpha"],
            "base_atoms": episode["base_atoms"],
            "base_masses": episode["base_masses"],
            "posterior_atoms": posterior_atoms,
            "posterior_masses": posterior_masses,
            "metric_scales": score_outputs["metric_scales"],
            **loss_payload,
        }

    @staticmethod
    def _stack_batch_outputs(batch_outputs: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        stacked = {
            "logits": torch.cat([item["logits"] for item in batch_outputs], dim=0),
            "pairwise_distances": torch.cat([item["pairwise_distances"] for item in batch_outputs], dim=0),
            "distribution_scores": torch.cat([item["distribution_scores"] for item in batch_outputs], dim=0),
            "query_latents": torch.stack([item["query_latents"] for item in batch_outputs], dim=0),
            "query_masses": torch.stack([item["query_masses"] for item in batch_outputs], dim=0),
            "support_latents": torch.stack([item["support_latents"] for item in batch_outputs], dim=0),
            "support_token_masses": torch.stack([item["support_token_masses"] for item in batch_outputs], dim=0),
            "shot_atoms": torch.stack([item["shot_atoms"] for item in batch_outputs], dim=0),
            "shot_basis_masses": torch.stack([item["shot_basis_masses"] for item in batch_outputs], dim=0),
            "shot_masses": torch.stack([item["shot_masses"] for item in batch_outputs], dim=0),
            "support_atoms": torch.stack([item["support_atoms"] for item in batch_outputs], dim=0),
            "support_masses": torch.stack([item["support_masses"] for item in batch_outputs], dim=0),
            "class_summary": torch.stack([item["class_summary"] for item in batch_outputs], dim=0),
            "episode_context": torch.stack([item["episode_context"] for item in batch_outputs], dim=0),
            "class_memory": torch.stack([item["class_memory"] for item in batch_outputs], dim=0),
            "shot_weighted_mean": torch.stack([item["shot_weighted_mean"] for item in batch_outputs], dim=0),
            "shot_weighted_std": torch.stack([item["shot_weighted_std"] for item in batch_outputs], dim=0),
            "shot_uncertainty": torch.stack([item["shot_uncertainty"] for item in batch_outputs], dim=0),
            "class_uncertainty": torch.stack([item["class_uncertainty"] for item in batch_outputs], dim=0),
            "prior_atoms": torch.stack([item["prior_atoms"] for item in batch_outputs], dim=0),
            "prior_masses": torch.stack([item["prior_masses"] for item in batch_outputs], dim=0),
            "alpha": torch.stack([item["alpha"] for item in batch_outputs], dim=0),
            "base_atoms": torch.stack([item["base_atoms"] for item in batch_outputs], dim=0),
            "base_masses": torch.stack([item["base_masses"] for item in batch_outputs], dim=0),
            "posterior_atoms": torch.stack([item["posterior_atoms"] for item in batch_outputs], dim=0),
            "posterior_masses": torch.stack([item["posterior_masses"] for item in batch_outputs], dim=0),
            "metric_scales": torch.stack([item["metric_scales"] for item in batch_outputs], dim=0),
        }
        for scalar_name in ("fm_loss", "align_loss", "margin_loss", "loo_loss", "reg_loss", "aux_loss"):
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
            else:
                batch_logits.append(outputs)

        if not needs_payload:
            return torch.cat(batch_logits, dim=0)
        return self._stack_batch_outputs(batch_outputs)
