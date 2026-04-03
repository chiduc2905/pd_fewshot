"""Support-Conditioned Latent Flow Inference v2."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from net.fewshot_common import BaseConv64FewShotModel, feature_map_to_tokens, merge_support_tokens
from net.modules.conditional_flow_v2 import ConditionalLatentFlowModelV2
from net.modules.flow_losses_v2 import (
    compute_context_smoothness_loss_v2,
    compute_distribution_margin_loss_v2,
    compute_flow_matching_loss_v2,
    compute_support_distribution_fit_loss_v2,
    compute_support_anchoring_loss_v2,
)
from net.modules.latent_projector_v2 import LatentEvidenceProjectorV2
from net.modules.set_context_v2 import SupportConditionerV2
from net.modules.transport_distance_v2 import (
    AlignmentTransportDistanceV2,
    WeightedTransportScoringDistanceV2,
    normalize_measure_masses,
)


class SupportConditionedLatentFlowInferenceNetV2(BaseConv64FewShotModel):
    """Stronger theory-first SC-LFI implementation.

    Core claim preserved:
    - each class is a support-conditioned latent evidence distribution;
    - query-class scoring is a distribution-fit score.

    Main formulas:
    - support/query latent evidence:
      `e = Psi(z)`
    - weighted evidence masses:
      `a = softmax(W_mass(e))`
    - support conditioning:
      `h_c, M_c, mu_c^anchor = Phi(E_c, a_c^sup)`
    - velocity field:
      `v_theta(y, t; h_c, M_c)`
    - mixed class measure:
      `muhat_c = rho_c * mu_c^anchor + (1 - rho_c) * mu_c^flow`
    - class score:
      `s_c(q) = -D_score(nu_q, muhat_c)`

    Engineering approximations:
    - class distributions are represented by finite particles;
    - sampling uses fixed-step Euler or Heun integration.
    """

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
        flow_hidden_dim: int = 128,
        flow_time_embedding_dim: int = 32,
        flow_memory_num_heads: int = 4,
        backbone_name: str = "resnet12",
        image_size: int = 64,
        resnet12_drop_rate: float = 0.0,
        resnet12_dropblock_size: int = 5,
        flow_conditioning_type: str = "film",
        use_global_proto_branch: bool = False,
        use_flow_branch: bool = True,
        use_align_loss: bool = True,
        use_margin_loss: bool = True,
        use_smooth_loss: bool = False,
        train_num_flow_particles: int = 8,
        eval_num_flow_particles: int = 16,
        train_num_integration_steps: int = 8,
        eval_num_integration_steps: int = 16,
        solver_type: str = "heun",
        fm_time_schedule: str = "uniform",
        score_temperature: float = 8.0,
        proto_branch_weight: float = 0.15,
        lambda_fm: float = 0.05,
        lambda_align: float = 0.1,
        lambda_margin: float = 0.1,
        lambda_support_fit: float = 0.1,
        lambda_smooth: float = 0.0,
        margin_value: float = 0.1,
        support_mix_min: float = 0.45,
        support_mix_max: float = 0.8,
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
        eval_particle_seed: int = 7,
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
        if train_num_flow_particles <= 0 or eval_num_flow_particles <= 0:
            raise ValueError("train_num_flow_particles and eval_num_flow_particles must be positive")
        if train_num_integration_steps <= 0 or eval_num_integration_steps <= 0:
            raise ValueError("train_num_integration_steps and eval_num_integration_steps must be positive")
        if score_temperature <= 0.0:
            raise ValueError("score_temperature must be positive")
        if not 0.0 <= proto_branch_weight <= 1.0:
            raise ValueError("proto_branch_weight must be in [0, 1]")
        if (
            lambda_fm < 0.0
            or lambda_align < 0.0
            or lambda_margin < 0.0
            or lambda_support_fit < 0.0
            or lambda_smooth < 0.0
        ):
            raise ValueError("loss weights must be non-negative")
        if margin_value < 0.0:
            raise ValueError("margin_value must be non-negative")
        if not 0.0 <= support_mix_min <= support_mix_max <= 1.0:
            raise ValueError("support_mix_min and support_mix_max must satisfy 0 <= min <= max <= 1")
        if eps <= 0.0:
            raise ValueError("eps must be positive")

        self.latent_dim = int(latent_dim)
        self.context_dim = int(context_dim)
        self.use_global_proto_branch = bool(use_global_proto_branch)
        self.use_flow_branch = bool(use_flow_branch)
        self.use_align_loss = bool(use_align_loss)
        self.use_margin_loss = bool(use_margin_loss)
        self.use_smooth_loss = bool(use_smooth_loss)
        self.train_num_flow_particles = int(train_num_flow_particles)
        self.eval_num_flow_particles = int(eval_num_flow_particles)
        self.train_num_integration_steps = int(train_num_integration_steps)
        self.eval_num_integration_steps = int(eval_num_integration_steps)
        self.solver_type = str(solver_type)
        self.fm_time_schedule = str(fm_time_schedule)
        self.score_temperature = float(score_temperature)
        self.proto_branch_weight = float(proto_branch_weight)
        self.lambda_fm = float(lambda_fm)
        self.lambda_align = float(lambda_align)
        self.lambda_margin = float(lambda_margin)
        self.lambda_support_fit = float(lambda_support_fit)
        self.lambda_smooth = float(lambda_smooth)
        self.margin_value = float(margin_value)
        self.support_mix_min = float(support_mix_min)
        self.support_mix_max = float(support_mix_max)
        self.eval_particle_seed = int(eval_particle_seed)
        self.eps = float(eps)

        self.latent_projector = LatentEvidenceProjectorV2(
            input_dim=hidden_dim,
            latent_dim=latent_dim,
            hidden_dim=latent_hidden_dim,
            mass_hidden_dim=mass_hidden_dim,
            mass_temperature=mass_temperature,
            eps=eps,
        )
        self.support_conditioner = SupportConditionerV2(
            latent_dim=latent_dim,
            context_dim=context_dim,
            memory_size=memory_size,
            memory_num_heads=memory_num_heads,
            memory_ffn_multiplier=memory_ffn_multiplier,
            summary_hidden_dim=context_hidden_dim,
            mass_feature_dim=context_hidden_dim,
            eps=eps,
        )
        self.flow_model = ConditionalLatentFlowModelV2(
            latent_dim=latent_dim,
            context_dim=context_dim,
            hidden_dim=flow_hidden_dim,
            time_embedding_dim=flow_time_embedding_dim,
            memory_num_heads=flow_memory_num_heads,
            conditioning_type=flow_conditioning_type,
            eps=eps,
        )
        self.score_distance = WeightedTransportScoringDistanceV2(
            train_num_projections=score_train_num_projections,
            eval_num_projections=score_eval_num_projections,
            p=score_sw_p,
            normalize_inputs=score_normalize_inputs,
            train_projection_mode=score_train_projection_mode,
            eval_projection_mode=score_eval_projection_mode,
            eval_num_repeats=score_eval_num_repeats,
            projection_seed=score_projection_seed,
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
        self.support_flow_mixer = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, max(1, context_dim // 2)),
            nn.GELU(),
            nn.Linear(max(1, context_dim // 2), 1),
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
        query_tokens, support_tokens = self._encode_episode_tokens(query, support)
        merged_support_tokens = merge_support_tokens(support_tokens, merge_mode="concat")

        query_outputs = self.latent_projector(query_tokens)
        support_outputs = self.latent_projector(merged_support_tokens)

        query_latents = query_outputs["latent_tokens"]
        query_masses = query_outputs["masses"]
        support_latents = support_outputs["latent_tokens"]
        support_masses = support_outputs["masses"]

        conditioner_outputs = self.support_conditioner(support_latents, support_masses)
        class_summary = conditioner_outputs["class_summary"]
        support_memory = conditioner_outputs["memory_tokens"]
        class_barycenters = conditioner_outputs["weighted_mean"]
        anchor_particles = conditioner_outputs["anchor_particles"]
        anchor_masses = conditioner_outputs["anchor_masses"]

        return {
            "query_tokens": query_tokens,
            "support_tokens": support_tokens,
            "query_latents": query_latents,
            "query_masses": query_masses,
            "support_latents": support_latents,
            "support_masses": support_masses,
            "class_summary": class_summary,
            "support_memory": support_memory,
            "class_barycenters": class_barycenters,
            "anchor_particles": anchor_particles,
            "anchor_masses": anchor_masses,
        }

    def _sampling_budget(self) -> tuple[int, int]:
        if self.training:
            return self.train_num_flow_particles, self.train_num_integration_steps
        return self.eval_num_flow_particles, self.eval_num_integration_steps

    def _support_mix_weights(self, class_summary: torch.Tensor) -> torch.Tensor:
        raw_mix = self.support_flow_mixer(class_summary).squeeze(-1)
        support_mix = torch.sigmoid(raw_mix)
        return self.support_mix_min + (self.support_mix_max - self.support_mix_min) * support_mix

    def _build_class_particles(
        self,
        support_latents: torch.Tensor,
        support_masses: torch.Tensor,
        class_summary: torch.Tensor,
        support_memory: torch.Tensor,
        anchor_particles: torch.Tensor,
        anchor_masses: torch.Tensor,
        class_barycenters: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        num_particles, num_steps = self._sampling_budget()
        way_num = support_latents.shape[0]

        if self.use_flow_branch:
            base_noise = None
            if not self.training:
                generator = torch.Generator(device="cpu")
                generator.manual_seed(self.eval_particle_seed)
                base_noise = torch.randn(
                    way_num,
                    num_particles,
                    self.latent_dim,
                    generator=generator,
                    dtype=torch.float32,
                ).to(device=support_latents.device, dtype=support_latents.dtype)
            particles = self.flow_model.sample_particles(
                class_summary,
                support_memory,
                num_particles=num_particles,
                num_steps=num_steps,
                solver_type=self.solver_type,
                base_noise=base_noise,
            )
            particle_masses = self.flow_model.estimate_particle_masses(
                particles,
                class_summary,
                support_memory,
            )
            support_mix = self._support_mix_weights(class_summary)
            class_measure_particles = torch.cat([anchor_particles, particles], dim=1)
            class_measure_masses = torch.cat(
                [
                    support_mix.unsqueeze(-1) * anchor_masses,
                    (1.0 - support_mix).unsqueeze(-1) * particle_masses,
                ],
                dim=1,
            )
            class_measure_masses = normalize_measure_masses(
                class_measure_masses,
                target_shape=class_measure_masses.shape,
                device=class_measure_particles.device,
                dtype=class_measure_particles.dtype,
                eps=self.eps,
            )
            return {
                "generated_particles": particles,
                "generated_particle_masses": particle_masses,
                "class_measure_particles": class_measure_particles,
                "class_measure_masses": class_measure_masses,
                "support_mix_weights": support_mix,
            }
        else:
            del support_masses
            particles = class_barycenters.unsqueeze(1).expand(-1, num_particles, -1).contiguous()
            particle_masses = normalize_measure_masses(
                None,
                target_shape=torch.Size((way_num, num_particles)),
                device=support_latents.device,
                dtype=support_latents.dtype,
                eps=self.eps,
            )
            return {
                "generated_particles": particles,
                "generated_particle_masses": particle_masses,
                "class_measure_particles": particles,
                "class_measure_masses": particle_masses,
                "support_mix_weights": support_latents.new_ones((way_num,)),
            }

    def _compute_global_proto_scores(
        self,
        query_latents: torch.Tensor,
        query_masses: torch.Tensor,
        class_barycenters: torch.Tensor,
    ) -> torch.Tensor:
        query_global = F.normalize((query_masses.unsqueeze(-1) * query_latents).sum(dim=1), p=2, dim=-1)
        class_global = F.normalize(class_barycenters, p=2, dim=-1)
        return self.score_temperature * torch.einsum("qd,wd->qw", query_global, class_global)

    def _score_query_against_classes(
        self,
        query_latents: torch.Tensor,
        query_masses: torch.Tensor,
        class_measure_particles: torch.Tensor,
        class_measure_masses: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pairwise_distances = self.score_distance.pairwise_distance(
            query_latents,
            class_measure_particles,
            query_masses=query_masses,
            support_masses=class_measure_masses,
            reduction="none",
        )
        logits = -self.score_temperature * pairwise_distances
        return logits, pairwise_distances

    def _compute_auxiliary_losses(
        self,
        *,
        support_latents: torch.Tensor,
        support_masses: torch.Tensor,
        class_summary: torch.Tensor,
        support_memory: torch.Tensor,
        anchor_particles: torch.Tensor,
        anchor_masses: torch.Tensor,
        class_particles: torch.Tensor,
        class_particle_masses: torch.Tensor,
        query_latents: torch.Tensor,
        query_masses: torch.Tensor,
        pairwise_distances: torch.Tensor,
        query_targets: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        zero = support_latents.new_zeros(())
        fm_loss = (
            compute_flow_matching_loss_v2(
                self.flow_model,
                support_latents,
                class_summary,
                support_memory,
                support_masses=support_masses,
                time_schedule=self.fm_time_schedule,
            )
            if self.use_flow_branch
            else zero
        )
        align_loss = (
            compute_support_anchoring_loss_v2(
                class_particles,
                support_latents,
                generated_masses=class_particle_masses,
                support_masses=support_masses,
                alignment_distance=self.alignment_distance,
            )
            if self.use_align_loss
            else zero
        )
        margin_loss = (
            compute_distribution_margin_loss_v2(
                pairwise_distances,
                query_targets,
                margin=self.margin_value,
            )
            if self.use_margin_loss and query_targets is not None
            else zero
        )
        support_fit_loss = (
            compute_support_distribution_fit_loss_v2(
                query_latents,
                query_masses,
                anchor_particles,
                anchor_masses,
                query_targets,
                scoring_distance=self.score_distance,
                score_temperature=self.score_temperature,
            )
            if query_targets is not None
            else zero
        )
        smooth_loss = (
            compute_context_smoothness_loss_v2(
                class_particles,
                class_summary,
                generated_masses=class_particle_masses,
                alignment_distance=self.alignment_distance,
                eps=self.eps,
            )
            if self.use_smooth_loss
            else zero
        )
        aux_loss = (
            self.lambda_fm * fm_loss
            + self.lambda_align * align_loss
            + self.lambda_margin * margin_loss
            + self.lambda_support_fit * support_fit_loss
            + self.lambda_smooth * smooth_loss
        )
        return {
            "fm_loss": fm_loss,
            "align_loss": align_loss,
            "margin_loss": margin_loss,
            "support_fit_loss": support_fit_loss,
            "smooth_loss": smooth_loss,
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
        class_measure = self._build_class_particles(
            episode["support_latents"],
            episode["support_masses"],
            episode["class_summary"],
            episode["support_memory"],
            episode["anchor_particles"],
            episode["anchor_masses"],
            episode["class_barycenters"],
        )
        logits, pairwise_distances = self._score_query_against_classes(
            episode["query_latents"],
            episode["query_masses"],
            class_measure["class_measure_particles"],
            class_measure["class_measure_masses"],
        )
        distribution_scores = logits

        if self.use_global_proto_branch:
            global_proto_scores = self._compute_global_proto_scores(
                episode["query_latents"],
                episode["query_masses"],
                episode["class_barycenters"],
            )
            logits = (1.0 - self.proto_branch_weight) * distribution_scores + self.proto_branch_weight * global_proto_scores
        else:
            global_proto_scores = distribution_scores.new_zeros(distribution_scores.shape)

        needs_payload = bool(return_aux or self.training)
        if not needs_payload:
            return logits

        loss_payload = self._compute_auxiliary_losses(
            support_latents=episode["support_latents"],
            support_masses=episode["support_masses"],
            class_summary=episode["class_summary"],
            support_memory=episode["support_memory"],
            anchor_particles=episode["anchor_particles"],
            anchor_masses=episode["anchor_masses"],
            class_particles=class_measure["generated_particles"],
            class_particle_masses=class_measure["generated_particle_masses"],
            query_latents=episode["query_latents"],
            query_masses=episode["query_masses"],
            pairwise_distances=pairwise_distances,
            query_targets=query_targets,
        )
        return {
            "logits": logits,
            "pairwise_distances": pairwise_distances,
            "distribution_scores": distribution_scores,
            "global_proto_scores": global_proto_scores,
            "query_latents": episode["query_latents"],
            "query_masses": episode["query_masses"],
            "support_latents": episode["support_latents"],
            "support_masses": episode["support_masses"],
            "class_summary": episode["class_summary"],
            "support_memory": episode["support_memory"],
            "class_barycenters": episode["class_barycenters"],
            "anchor_particles": episode["anchor_particles"],
            "anchor_masses": episode["anchor_masses"],
            "generated_particles": class_measure["generated_particles"],
            "generated_particle_masses": class_measure["generated_particle_masses"],
            "class_measure_particles": class_measure["class_measure_particles"],
            "class_measure_masses": class_measure["class_measure_masses"],
            "support_mix_weights": class_measure["support_mix_weights"],
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
            "class_summary": torch.stack([item["class_summary"] for item in batch_outputs], dim=0),
            "support_memory": torch.stack([item["support_memory"] for item in batch_outputs], dim=0),
            "class_barycenters": torch.stack([item["class_barycenters"] for item in batch_outputs], dim=0),
            "anchor_particles": torch.stack([item["anchor_particles"] for item in batch_outputs], dim=0),
            "anchor_masses": torch.stack([item["anchor_masses"] for item in batch_outputs], dim=0),
            "generated_particles": torch.stack([item["generated_particles"] for item in batch_outputs], dim=0),
            "generated_particle_masses": torch.stack(
                [item["generated_particle_masses"] for item in batch_outputs],
                dim=0,
            ),
            "class_measure_particles": torch.stack(
                [item["class_measure_particles"] for item in batch_outputs],
                dim=0,
            ),
            "class_measure_masses": torch.stack(
                [item["class_measure_masses"] for item in batch_outputs],
                dim=0,
            ),
            "support_mix_weights": torch.stack([item["support_mix_weights"] for item in batch_outputs], dim=0),
        }
        for scalar_name in (
            "fm_loss",
            "align_loss",
            "margin_loss",
            "support_fit_loss",
            "smooth_loss",
            "aux_loss",
        ):
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
                raise ValueError(
                    "Flat query_targets must have length batch_size * num_query: "
                    f"targets={tuple(query_targets.shape)} batch_size={batch_size} num_query={num_query}"
                )
            return query_targets.reshape(batch_size, num_query)
        if query_targets.dim() == 2 and tuple(query_targets.shape) == (batch_size, num_query):
            return query_targets
        raise ValueError(
            "query_targets must have shape (Batch * NumQuery,) or (Batch, NumQuery), "
            f"got {tuple(query_targets.shape)}"
        )

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
            "support_fit_loss": stacked["support_fit_loss"],
            "smooth_loss": stacked["smooth_loss"],
        }
