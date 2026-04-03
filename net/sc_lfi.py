"""Support-Conditioned Latent Flow Inference (SC-LFI) few-shot model."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from net.fewshot_common import BaseConv64FewShotModel, feature_map_to_tokens, merge_support_tokens
from net.modules.conditional_flow import ConditionalLatentFlowModel
from net.modules.distribution_distance import DistributionDistance
from net.modules.flow_losses import (
    compute_context_smoothness_loss,
    compute_flow_matching_loss,
    compute_support_anchoring_loss,
)
from net.modules.latent_projector import LatentEvidenceProjector
from net.modules.set_context import SupportSetContextEncoder


class SupportConditionedLatentFlowInferenceNet(BaseConv64FewShotModel):
    """Few-shot classifier with support-conditioned latent evidence distributions.

    This model treats prototype matching as a degenerate special case. The main
    inference object is the distance between the query latent evidence
    distribution and a sampled class-conditional latent distribution.
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_dim: int = 64,
        latent_dim: int = 64,
        context_dim: int = 128,
        context_hidden_dim: int | None = None,
        latent_hidden_dim: int | None = None,
        flow_hidden_dim: int = 128,
        flow_time_embedding_dim: int = 32,
        backbone_name: str = "resnet12",
        image_size: int = 64,
        resnet12_drop_rate: float = 0.0,
        resnet12_dropblock_size: int = 5,
        class_context_type: str = "deepsets",
        flow_conditioning_type: str = "concat",
        distance_type: str = "sw",
        use_global_proto_branch: bool = False,
        use_flow_branch: bool = True,
        use_align_loss: bool = True,
        use_smooth_loss: bool = False,
        num_flow_particles: int = 16,
        num_flow_integration_steps: int = 8,
        fm_time_schedule: str = "uniform",
        score_temperature: float = 8.0,
        proto_branch_weight: float = 0.2,
        lambda_fm: float = 0.05,
        lambda_align: float = 0.1,
        lambda_smooth: float = 0.0,
        distance_normalize_inputs: bool = True,
        distance_sw_num_projections: int = 64,
        distance_sw_p: float = 2.0,
        distance_projection_seed: int = 7,
        sinkhorn_epsilon: float = 0.1,
        sinkhorn_iterations: int = 50,
        context_num_memory_tokens: int = 4,
        context_num_heads: int = 4,
        context_ffn_multiplier: int = 2,
        eval_particle_seed: int = 7,
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
        if latent_dim <= 0 or context_dim <= 0:
            raise ValueError("latent_dim and context_dim must be positive")
        if num_flow_particles <= 0 or num_flow_integration_steps <= 0:
            raise ValueError("num_flow_particles and num_flow_integration_steps must be positive")
        if score_temperature <= 0.0:
            raise ValueError("score_temperature must be positive")
        if not 0.0 <= proto_branch_weight <= 1.0:
            raise ValueError("proto_branch_weight must be in [0, 1]")
        if lambda_fm < 0.0 or lambda_align < 0.0 or lambda_smooth < 0.0:
            raise ValueError("loss weights must be non-negative")
        if eps <= 0.0:
            raise ValueError("eps must be positive")

        self.latent_dim = int(latent_dim)
        self.context_dim = int(context_dim)
        self.use_global_proto_branch = bool(use_global_proto_branch)
        self.use_flow_branch = bool(use_flow_branch)
        self.use_align_loss = bool(use_align_loss)
        self.use_smooth_loss = bool(use_smooth_loss)
        self.num_flow_particles = int(num_flow_particles)
        self.num_flow_integration_steps = int(num_flow_integration_steps)
        self.fm_time_schedule = str(fm_time_schedule)
        self.score_temperature = float(score_temperature)
        self.proto_branch_weight = float(proto_branch_weight)
        self.lambda_fm = float(lambda_fm)
        self.lambda_align = float(lambda_align)
        self.lambda_smooth = float(lambda_smooth)
        self.eval_particle_seed = int(eval_particle_seed)
        self.eps = float(eps)

        self.context_encoder = SupportSetContextEncoder(
            input_dim=hidden_dim,
            context_dim=context_dim,
            context_type=class_context_type,
            hidden_dim=context_hidden_dim,
            num_memory_tokens=context_num_memory_tokens,
            num_heads=context_num_heads,
            ffn_multiplier=context_ffn_multiplier,
        )
        self.latent_projector = LatentEvidenceProjector(
            input_dim=hidden_dim,
            latent_dim=latent_dim,
            hidden_dim=latent_hidden_dim,
        )
        self.flow_model = ConditionalLatentFlowModel(
            latent_dim=latent_dim,
            context_dim=context_dim,
            hidden_dim=flow_hidden_dim,
            time_embedding_dim=flow_time_embedding_dim,
            conditioning_type=flow_conditioning_type,
        )
        self.distribution_distance = DistributionDistance(
            distance_type=distance_type,
            sw_num_projections=distance_sw_num_projections,
            sw_p=distance_sw_p,
            normalize_inputs=distance_normalize_inputs,
            projection_seed=distance_projection_seed,
            sinkhorn_epsilon=sinkhorn_epsilon,
            sinkhorn_iterations=sinkhorn_iterations,
            reduction="mean",
        )

    def _encode_episode_tokens(
        self,
        query: torch.Tensor,
        support: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode one episode to query/support token tensors.

        Returns:
        - query_tokens: `[NumQuery, TokensPerImage, HiddenDim]`
        - support_tokens: `[Way, Shot, TokensPerImage, HiddenDim]`
        """

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
        # merged_support_tokens: [Way, Shot * TokensPerImage, HiddenDim]
        merged_support_tokens = merge_support_tokens(support_tokens, merge_mode="concat")
        class_contexts = self.context_encoder(merged_support_tokens)
        query_latents = self.latent_projector(query_tokens)
        support_latents = self.latent_projector(merged_support_tokens)
        class_prototypes = support_latents.mean(dim=1)
        return {
            "query_tokens": query_tokens,
            "support_tokens": support_tokens,
            "merged_support_tokens": merged_support_tokens,
            "class_contexts": class_contexts,
            "query_latents": query_latents,
            "support_latents": support_latents,
            "class_prototypes": class_prototypes,
        }

    def _build_class_particles(
        self,
        support_latents: torch.Tensor,
        class_contexts: torch.Tensor,
    ) -> torch.Tensor:
        """Return sampled class particles or a degenerate class-mean distribution."""

        if self.use_flow_branch:
            base_noise = None
            if not self.training:
                generator = torch.Generator(device="cpu")
                generator.manual_seed(self.eval_particle_seed)
                base_noise = torch.randn(
                    support_latents.shape[0],
                    self.num_flow_particles,
                    self.latent_dim,
                    generator=generator,
                    dtype=torch.float32,
                ).to(device=support_latents.device, dtype=support_latents.dtype)
            return self.flow_model.sample_particles(
                class_contexts,
                num_particles=self.num_flow_particles,
                num_steps=self.num_flow_integration_steps,
                base_noise=base_noise,
            )

        class_means = support_latents.mean(dim=1, keepdim=True)
        return class_means.expand(-1, self.num_flow_particles, -1).contiguous()

    def _score_query_against_classes(
        self,
        query_latents: torch.Tensor,
        class_particles: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the distribution-fit logits for all query/class pairs."""

        num_query, num_query_tokens, latent_dim = query_latents.shape
        way_num, num_particles, particle_dim = class_particles.shape
        if latent_dim != particle_dim:
            raise ValueError(
                "query_latents and class_particles must share the latent dimension: "
                f"query={latent_dim} particles={particle_dim}"
            )

        # query_distributions: [NumQuery, Way, QueryTokens, LatentDim]
        query_distributions = query_latents.unsqueeze(1).expand(num_query, way_num, num_query_tokens, latent_dim)
        # class_distributions: [NumQuery, Way, FlowParticles, LatentDim]
        class_distributions = class_particles.unsqueeze(0).expand(num_query, way_num, num_particles, latent_dim)
        distances = self.distribution_distance(query_distributions, class_distributions, reduction="none")
        return -self.score_temperature * distances

    def _compute_global_proto_scores(
        self,
        query_latents: torch.Tensor,
        class_prototypes: torch.Tensor,
    ) -> torch.Tensor:
        query_global = F.normalize(query_latents.mean(dim=1), p=2, dim=-1)
        class_global = F.normalize(class_prototypes, p=2, dim=-1)
        return self.score_temperature * torch.einsum("qd,wd->qw", query_global, class_global)

    def _compute_auxiliary_losses(
        self,
        support_latents: torch.Tensor,
        class_contexts: torch.Tensor,
        class_particles: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        zero = support_latents.new_zeros(())
        fm_loss = (
            compute_flow_matching_loss(
                self.flow_model,
                support_latents,
                class_contexts,
                time_schedule=self.fm_time_schedule,
            )
            if self.use_flow_branch
            else zero
        )
        align_loss = (
            compute_support_anchoring_loss(
                class_particles,
                support_latents,
                self.distribution_distance,
            )
            if self.use_align_loss
            else zero
        )
        smooth_loss = (
            compute_context_smoothness_loss(
                class_particles,
                class_contexts,
                self.distribution_distance,
                eps=self.eps,
            )
            if self.use_smooth_loss
            else zero
        )
        total_aux = self.lambda_fm * fm_loss + self.lambda_align * align_loss + self.lambda_smooth * smooth_loss
        return {
            "fm_loss": fm_loss,
            "align_loss": align_loss,
            "smooth_loss": smooth_loss,
            "aux_loss": total_aux,
        }

    def _forward_episode(
        self,
        query: torch.Tensor,
        support: torch.Tensor,
        *,
        return_aux: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        episode = self._build_episode_representations(query, support)
        class_particles = self._build_class_particles(
            episode["support_latents"],
            episode["class_contexts"],
        )
        distribution_scores = self._score_query_against_classes(
            episode["query_latents"],
            class_particles,
        )

        if self.use_global_proto_branch:
            global_proto_scores = self._compute_global_proto_scores(
                episode["query_latents"],
                episode["class_prototypes"],
            )
            logits = (1.0 - self.proto_branch_weight) * distribution_scores + self.proto_branch_weight * global_proto_scores
        else:
            global_proto_scores = distribution_scores.new_zeros(distribution_scores.shape)
            logits = distribution_scores

        needs_payload = bool(return_aux or self.training)
        if not needs_payload:
            return logits

        loss_payload = self._compute_auxiliary_losses(
            episode["support_latents"],
            episode["class_contexts"],
            class_particles,
        )
        return {
            "logits": logits,
            "distribution_scores": distribution_scores,
            "global_proto_scores": global_proto_scores,
            "class_contexts": episode["class_contexts"],
            "query_latents": episode["query_latents"],
            "support_latents": episode["support_latents"],
            "class_prototypes": episode["class_prototypes"],
            "generated_particles": class_particles,
            **loss_payload,
        }

    @staticmethod
    def _stack_batch_outputs(batch_outputs: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        stacked = {
            "logits": torch.cat([item["logits"] for item in batch_outputs], dim=0),
            "distribution_scores": torch.cat([item["distribution_scores"] for item in batch_outputs], dim=0),
            "global_proto_scores": torch.cat([item["global_proto_scores"] for item in batch_outputs], dim=0),
            "class_contexts": torch.stack([item["class_contexts"] for item in batch_outputs], dim=0),
            "query_latents": torch.stack([item["query_latents"] for item in batch_outputs], dim=0),
            "support_latents": torch.stack([item["support_latents"] for item in batch_outputs], dim=0),
            "class_prototypes": torch.stack([item["class_prototypes"] for item in batch_outputs], dim=0),
            "generated_particles": torch.stack([item["generated_particles"] for item in batch_outputs], dim=0),
        }
        for scalar_name in ("fm_loss", "align_loss", "smooth_loss", "aux_loss"):
            stacked[scalar_name] = torch.stack([item[scalar_name] for item in batch_outputs], dim=0).mean()
        return stacked

    def forward(
        self,
        query: torch.Tensor,
        support: torch.Tensor,
        return_aux: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        bsz, _, _, _, _, _ = self.validate_episode_inputs(query, support)
        needs_payload = bool(return_aux or self.training)
        batch_logits = []
        batch_outputs = []

        for batch_idx in range(bsz):
            outputs = self._forward_episode(
                query[batch_idx],
                support[batch_idx],
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
            "smooth_loss": stacked["smooth_loss"],
        }
