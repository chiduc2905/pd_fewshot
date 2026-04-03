"""SPIFAEB-v3: structured-evidence global head + unchanged v2 local branch.

This file adds a separate v3 model instead of mutating SPIFAEB-v2 in place.
The only architectural change is the global head. The local branch remains the
same v2 branch by design so global class-level evidence and local
query-conditioned refinement stay complementary.
"""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from net.spif_aeb_v2 import SPIFAEBV2


def _inverse_softplus(value: float, floor: float = 1e-6) -> float:
    """Return a raw parameter whose softplus matches `value`."""
    value = max(float(value), float(floor))
    return math.log(math.expm1(value))


class StructuredEvidenceGlobalHead(nn.Module):
    """Few-shot-safe structured evidence scorer on stable global embeddings."""

    def __init__(
        self,
        stable_dim: int,
        use_structured_global: bool = True,
        sigma_min: float = 0.05,
        use_reliability: bool = True,
        reliability_detach: bool = True,
        beta_align: float = 1.0,
        beta_dev: float = 0.5,
        beta_rel: float = 0.2,
        learnable_betas: bool = False,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if int(stable_dim) <= 0:
            raise ValueError("stable_dim must be positive")
        if float(sigma_min) <= 0.0:
            raise ValueError("sigma_min must be positive")
        if float(beta_dev) < 0.0:
            raise ValueError("beta_dev must be non-negative")
        if float(eps) <= 0.0:
            raise ValueError("eps must be positive")

        self.stable_dim = int(stable_dim)
        self.use_structured_global = bool(use_structured_global)
        self.sigma_min = float(sigma_min)
        self.use_reliability = bool(use_reliability)
        self.reliability_detach = bool(reliability_detach)
        self.learnable_betas = bool(learnable_betas)
        self.eps = float(eps)

        beta_align_raw = torch.tensor(_inverse_softplus(beta_align), dtype=torch.float32)
        beta_dev_raw = torch.tensor(_inverse_softplus(beta_dev), dtype=torch.float32)
        beta_rel_raw = torch.tensor(_inverse_softplus(beta_rel), dtype=torch.float32)

        if self.learnable_betas:
            self.beta_align_raw = nn.Parameter(beta_align_raw)
            self.beta_dev_raw = nn.Parameter(beta_dev_raw)
            self.beta_rel_raw = nn.Parameter(beta_rel_raw)
        else:
            self.register_buffer("beta_align_raw", beta_align_raw)
            self.register_buffer("beta_dev_raw", beta_dev_raw)
            self.register_buffer("beta_rel_raw", beta_rel_raw)

    def _betas(self, dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        beta_align = F.softplus(self.beta_align_raw).to(device=device, dtype=dtype)
        # ===== Stabilizer: beta_dev is constrained non-negative with softplus =====
        beta_dev = F.softplus(self.beta_dev_raw).to(device=device, dtype=dtype)
        beta_rel = F.softplus(self.beta_rel_raw).to(device=device, dtype=dtype)
        return beta_align, beta_dev, beta_rel

    def _compute_class_centers(self, support_global: torch.Tensor) -> torch.Tensor:
        # support_global: [Way, Shot, D]
        class_centers = support_global.mean(dim=1)
        # ===== Stabilizer: re-normalize class centers before cosine / deviation scoring =====
        return F.normalize(class_centers, p=2, dim=-1)

    def _compute_dispersion(
        self,
        support_global: torch.Tensor,
        class_centers: torch.Tensor,
    ) -> torch.Tensor:
        shot_num = int(support_global.shape[1])
        if shot_num == 1:
            # ===== Stabilizer: 1-shot-safe dispersion fallback =====
            sigma_sq = support_global.new_full((support_global.shape[0],), self.sigma_min * self.sigma_min)
        else:
            centered_support = support_global - class_centers.unsqueeze(1)
            sigma_sq = centered_support.square().sum(dim=-1).mean(dim=1)
        # ===== Stabilizer: numerical clamp to avoid degenerate variance =====
        return sigma_sq.clamp_min(1e-4)

    def _compute_gate_reliability(self, support_gate: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gate_values = support_gate
        if gate_values.dim() != 4 or gate_values.shape[-1] != 1:
            raise ValueError(
                "support_gate must have shape [Way, Shot, Tokens, 1], "
                f"got {tuple(gate_values.shape)}"
            )
        if self.reliability_detach:
            gate_values = gate_values.detach()
        gate_values = gate_values.squeeze(-1)
        gate_mean = gate_values.mean(dim=-1).clamp_min(self.eps)
        gate_std = gate_values.std(dim=-1, unbiased=False)
        # ===== Architectural contribution: support-only class reliability term =====
        reliability_raw = (gate_std / gate_mean).mean(dim=1)
        reliability = torch.sigmoid(reliability_raw)
        return reliability, reliability_raw

    def _compute_fallback_reliability(
        self,
        support_global: torch.Tensor,
        class_centers: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        support_consistency = torch.sum(support_global * class_centers.unsqueeze(1), dim=-1).mean(dim=1)
        reliability = 0.5 * (support_consistency.clamp(-1.0, 1.0) + 1.0)
        return reliability, support_consistency

    def forward(
        self,
        query_global: torch.Tensor,
        support_global: torch.Tensor,
        support_gate: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if query_global.dim() != 2:
            raise ValueError(f"query_global must have shape [NumQuery, D], got {tuple(query_global.shape)}")
        if support_global.dim() != 3:
            raise ValueError(f"support_global must have shape [Way, Shot, D], got {tuple(support_global.shape)}")
        if query_global.shape[-1] != self.stable_dim or support_global.shape[-1] != self.stable_dim:
            raise ValueError(
                f"StructuredEvidenceGlobalHead expected stable_dim={self.stable_dim}, "
                f"got query={query_global.shape[-1]} support={support_global.shape[-1]}"
            )

        # ===== Stabilizer: head-level normalization guard for all stable global vectors =====
        query_global = F.normalize(query_global, p=2, dim=-1)
        support_global = F.normalize(support_global, p=2, dim=-1)

        class_centers = self._compute_class_centers(support_global)
        dispersion = self._compute_dispersion(support_global, class_centers)

        if self.use_reliability:
            if support_gate is not None:
                reliability, reliability_raw = self._compute_gate_reliability(support_gate)
                reliability_source = query_global.new_ones(reliability.shape)
            else:
                reliability, reliability_raw = self._compute_fallback_reliability(support_global, class_centers)
                reliability_source = query_global.new_zeros(reliability.shape)
        else:
            reliability = query_global.new_zeros(class_centers.shape[0])
            reliability_raw = reliability
            reliability_source = query_global.new_full(reliability.shape, -1.0)

        # ===== Architectural contribution: Structured Evidence Global Head =====
        alignment = torch.matmul(query_global, class_centers.transpose(0, 1))
        squared_offset = (query_global.unsqueeze(1) - class_centers.unsqueeze(0)).square().sum(dim=-1)
        deviation = squared_offset / (dispersion.unsqueeze(0) + self.eps)

        beta_align, beta_dev, beta_rel = self._betas(dtype=query_global.dtype, device=query_global.device)
        structured_scores = beta_align * alignment - beta_dev * deviation + beta_rel * reliability.unsqueeze(0)
        scores = structured_scores if self.use_structured_global else alignment

        return {
            "scores": scores,
            "structured_scores": structured_scores,
            "alignment": alignment,
            "deviation": deviation,
            "reliability": reliability,
            "reliability_raw": reliability_raw,
            "reliability_source": reliability_source,
            "dispersion": dispersion,
            "class_centers": class_centers,
            "beta_align": beta_align.detach(),
            "beta_dev": beta_dev.detach(),
            "beta_rel": beta_rel.detach(),
        }


class SPIFAEBv3(SPIFAEBV2):
    """SPIFAEB-v3 with a new structured-evidence global head and the v2 local branch."""

    def __init__(
        self,
        spifaeb_v3_use_structured_global: bool = True,
        spifaeb_v3_sigma_min: float = 0.05,
        spifaeb_v3_use_reliability: bool = True,
        spifaeb_v3_reliability_detach: bool = True,
        spifaeb_v3_beta_align: float = 1.0,
        spifaeb_v3_beta_dev: float = 0.5,
        spifaeb_v3_beta_rel: float = 0.2,
        spifaeb_v3_learnable_betas: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.structured_global_head = StructuredEvidenceGlobalHead(
            stable_dim=self.stable_dim,
            use_structured_global=spifaeb_v3_use_structured_global,
            sigma_min=spifaeb_v3_sigma_min,
            use_reliability=spifaeb_v3_use_reliability,
            reliability_detach=spifaeb_v3_reliability_detach,
            beta_align=spifaeb_v3_beta_align,
            beta_dev=spifaeb_v3_beta_dev,
            beta_rel=spifaeb_v3_beta_rel,
            learnable_betas=spifaeb_v3_learnable_betas,
            eps=self.aeb_v2_eps,
        )

    def _forward_episode(
        self,
        query: torch.Tensor,
        support: torch.Tensor,
        query_targets: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        episode = self._encode_episode(query, support)

        global_outputs = self.structured_global_head(
            query_global=episode["query_global"],
            support_global=episode["support_global"],
            support_gate=episode["support_gate"],
        )
        global_scores = self.aeb_v2_global_scale * global_outputs["scores"]

        support_token_pool = self.build_support_token_pool(episode["support_local_tokens"])
        support_row_weights = (
            self.build_support_row_weights(episode["support_gate"], eps=self.aeb_v2_eps)
            if self.aeb_v2_query_gate_weighting
            else None
        )

        # ===== Architectural reuse: keep the full v2 local branch unchanged =====
        local_evidence_profile = self.build_local_evidence_profile(
            query_tokens=episode["query_local_tokens"],
            support_token_pool=support_token_pool,
        )
        query_row_weights = None
        query_class_attention_entropy = None
        if self.aeb_v2_query_class_attention_on:
            query_row_weights, query_class_attention_entropy = self.compute_query_class_row_weights(
                local_evidence_profile=local_evidence_profile,
                query_gate=episode["query_gate"],
            )
        elif self.aeb_v2_query_gate_weighting:
            query_row_weights = self.compute_query_row_weights(episode["query_gate"], eps=self.aeb_v2_eps)

        budget_outputs = self.predict_budget(
            local_evidence_profile=local_evidence_profile,
            row_weights=query_row_weights,
        )
        coverage_bonus = budget_outputs["coverage_entropy"] - budget_outputs["coverage_concentration"]
        local_outputs = self.compute_local_budget_scores(
            similarity=local_evidence_profile["similarity"],
            rho=budget_outputs["rho"],
            row_budget=budget_outputs["row_budget"],
            query_row_weights=query_row_weights,
            support_row_weights=support_row_weights if self.aeb_v2_local_score_mode == "bidirectional" else None,
            coverage_bonus=coverage_bonus,
        )
        anchor_local_scores = self.compute_anchor_local_scores(
            similarity=local_evidence_profile["similarity"],
            query_row_weights=query_row_weights,
            support_row_weights=support_row_weights if self.aeb_v2_local_score_mode == "bidirectional" else None,
        )
        local_anchor_mix, local_adaptive_mix = self.compute_local_mix(budget_outputs["rho_prior"])
        local_scores = local_anchor_mix * anchor_local_scores + local_adaptive_mix * local_outputs["adaptive_local_scores"]
        logits, alpha_q, global_margin, local_margin = self.fuse_scores(global_scores, local_scores)
        (
            aux_loss,
            global_branch_ce,
            local_branch_ce,
            budget_rank_loss,
            local_margin_loss,
            budget_residual_reg_loss,
            anchor_consistency_loss,
        ) = self.compute_branch_aux_loss(
            global_scores=global_scores,
            local_scores=local_scores,
            anchor_local_scores=anchor_local_scores,
            controller_scores=local_outputs["adaptive_local_scores"],
            rho=budget_outputs["rho"],
            rho_residual=budget_outputs["rho_residual"],
            query_targets=query_targets,
        )

        num_query = episode["query_global"].shape[0]
        structured_dispersion = global_outputs["dispersion"].unsqueeze(0).expand(num_query, -1)
        structured_reliability = global_outputs["reliability"].unsqueeze(0).expand(num_query, -1)
        structured_reliability_raw = global_outputs["reliability_raw"].unsqueeze(0).expand(num_query, -1)
        structured_reliability_source = global_outputs["reliability_source"].unsqueeze(0).expand(num_query, -1)

        mean_gate = torch.cat(
            [
                episode["query_gate"].reshape(-1, 1),
                episode["support_gate"].reshape(-1, 1),
            ],
            dim=0,
        ).mean()
        zero = logits.new_zeros(())
        return {
            "logits": logits,
            "aux_loss": aux_loss,
            "global_scores": global_scores.detach(),
            # v2 compatibility slots: keep these populated for existing diagnostics.
            "base_global_scores": global_outputs["alignment"].detach(),
            "conditioned_global_scores": global_outputs["structured_scores"].detach(),
            "structured_alignment": global_outputs["alignment"].detach(),
            "structured_deviation": global_outputs["deviation"].detach(),
            "structured_reliability": structured_reliability.detach(),
            "structured_reliability_raw": structured_reliability_raw.detach(),
            "structured_reliability_source": structured_reliability_source.detach(),
            "structured_dispersion": structured_dispersion.detach(),
            "structured_class_centers": global_outputs["class_centers"].detach(),
            "structured_beta_align": global_outputs["beta_align"],
            "structured_beta_dev": global_outputs["beta_dev"],
            "structured_beta_rel": global_outputs["beta_rel"],
            "local_scores": local_scores.detach(),
            "anchor_local_scores": anchor_local_scores.detach(),
            "adaptive_local_scores": local_outputs["adaptive_local_scores"].detach(),
            "raw_local_scores": local_outputs["raw_local_scores"].detach(),
            "competitive_local_scores": local_outputs["competitive_local_scores"].detach(),
            "rho": local_outputs["rho"].detach(),
            "rho_prior": budget_outputs["rho_prior"].detach(),
            "rho_residual": budget_outputs["rho_residual"].detach(),
            "row_budget_std": budget_outputs["row_budget_std"].detach(),
            "evidence_sharpness": budget_outputs["evidence_sharpness"].detach(),
            "evidence_advantage": budget_outputs["evidence_advantage"].detach(),
            "evidence_dispersion": budget_outputs["evidence_dispersion"].detach(),
            "evidence_quality": budget_outputs["evidence_quality"].detach(),
            "coverage_entropy": budget_outputs["coverage_entropy"].detach(),
            "coverage_concentration": budget_outputs["coverage_concentration"].detach(),
            "coverage_bonus": coverage_bonus.detach(),
            "active_match_counts": local_outputs["active_match_counts"].detach(),
            "retained_fraction": local_outputs["retained_fraction"].detach(),
            "alpha": alpha_q.mean().detach(),
            "alpha_q": alpha_q.detach(),
            "global_margin": global_margin.detach(),
            "local_margin": local_margin.detach(),
            "mean_budget": local_outputs["rho"].mean().detach(),
            "mean_conditioned_global_entropy": zero.detach(),
            "mean_query_class_attention_entropy": (
                query_class_attention_entropy.mean().detach()
                if query_class_attention_entropy is not None
                else zero.detach()
            ),
            "mean_gate": mean_gate.detach(),
            "global_branch_ce": global_branch_ce,
            "local_branch_ce": local_branch_ce,
            "budget_rank_loss": budget_rank_loss,
            "local_margin_loss": local_margin_loss,
            "budget_residual_reg_loss": budget_residual_reg_loss,
            "anchor_consistency_loss": anchor_consistency_loss,
            "local_anchor_mix": local_anchor_mix.detach(),
            "local_adaptive_mix": local_adaptive_mix.detach(),
            "stable_global_embeddings": torch.cat(
                [
                    episode["query_global"],
                    episode["support_global"].reshape(-1, episode["support_global"].shape[-1]),
                ],
                dim=0,
            ).detach(),
            "variant_global_embeddings": torch.cat(
                [
                    episode["query_variant_global"],
                    episode["support_variant_global"].reshape(-1, episode["support_variant_global"].shape[-1]),
                ],
                dim=0,
            ).detach(),
        }

    def forward(
        self,
        query: torch.Tensor,
        support: torch.Tensor,
        query_targets: torch.Tensor | None = None,
        support_targets: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> torch.Tensor | Dict[str, torch.Tensor]:
        del support_targets  # reserved for trainer API compatibility.
        bsz, _, _, _, _, _ = self.validate_episode_inputs(query, support)
        batch_outputs = []
        aux_losses = []
        diagnostics = []
        query_targets = None if query_targets is None else query_targets.view(-1)

        query_offset = 0
        for batch_idx in range(bsz):
            batch_query = query[batch_idx]
            batch_targets = None
            if query_targets is not None:
                num_query = batch_query.shape[0]
                batch_targets = query_targets[query_offset : query_offset + num_query]
                query_offset += num_query
            episode = self._forward_episode(batch_query, support[batch_idx], query_targets=batch_targets)
            batch_outputs.append(episode["logits"])
            aux_losses.append(episode["aux_loss"])
            diagnostics.append(episode)

        logits = torch.cat(batch_outputs, dim=0)
        aux_loss = torch.stack(aux_losses).mean() if aux_losses else logits.new_zeros(())
        if not return_aux:
            if self.training:
                return {
                    "logits": logits,
                    "aux_loss": aux_loss,
                    "budget_rank_loss": torch.stack([item["budget_rank_loss"] for item in diagnostics]).mean(),
                    "local_margin_loss": torch.stack([item["local_margin_loss"] for item in diagnostics]).mean(),
                    "budget_residual_reg_loss": torch.stack(
                        [item["budget_residual_reg_loss"] for item in diagnostics]
                    ).mean(),
                    "anchor_consistency_loss": torch.stack(
                        [item["anchor_consistency_loss"] for item in diagnostics]
                    ).mean(),
                    "global_branch_ce": torch.stack([item["global_branch_ce"] for item in diagnostics]).mean(),
                    "local_branch_ce": torch.stack([item["local_branch_ce"] for item in diagnostics]).mean(),
                }
            return logits

        return {
            "logits": logits,
            "aux_loss": aux_loss,
            "global_scores": torch.cat([item["global_scores"] for item in diagnostics], dim=0),
            "base_global_scores": torch.cat([item["base_global_scores"] for item in diagnostics], dim=0),
            "conditioned_global_scores": torch.cat(
                [item["conditioned_global_scores"] for item in diagnostics],
                dim=0,
            ),
            "structured_alignment": torch.cat([item["structured_alignment"] for item in diagnostics], dim=0),
            "structured_deviation": torch.cat([item["structured_deviation"] for item in diagnostics], dim=0),
            "structured_reliability": torch.cat([item["structured_reliability"] for item in diagnostics], dim=0),
            "structured_reliability_raw": torch.cat(
                [item["structured_reliability_raw"] for item in diagnostics],
                dim=0,
            ),
            "structured_reliability_source": torch.cat(
                [item["structured_reliability_source"] for item in diagnostics],
                dim=0,
            ),
            "structured_dispersion": torch.cat([item["structured_dispersion"] for item in diagnostics], dim=0),
            "structured_class_centers": torch.stack(
                [item["structured_class_centers"] for item in diagnostics],
                dim=0,
            ),
            "structured_beta_align": torch.stack(
                [item["structured_beta_align"] for item in diagnostics]
            ).mean(),
            "structured_beta_dev": torch.stack([item["structured_beta_dev"] for item in diagnostics]).mean(),
            "structured_beta_rel": torch.stack([item["structured_beta_rel"] for item in diagnostics]).mean(),
            "local_scores": torch.cat([item["local_scores"] for item in diagnostics], dim=0),
            "anchor_local_scores": torch.cat([item["anchor_local_scores"] for item in diagnostics], dim=0),
            "adaptive_local_scores": torch.cat([item["adaptive_local_scores"] for item in diagnostics], dim=0),
            "raw_local_scores": torch.cat([item["raw_local_scores"] for item in diagnostics], dim=0),
            "competitive_local_scores": torch.cat([item["competitive_local_scores"] for item in diagnostics], dim=0),
            "rho": torch.cat([item["rho"] for item in diagnostics], dim=0),
            "rho_prior": torch.cat([item["rho_prior"] for item in diagnostics], dim=0),
            "rho_residual": torch.cat([item["rho_residual"] for item in diagnostics], dim=0),
            "row_budget_std": torch.cat([item["row_budget_std"] for item in diagnostics], dim=0),
            "evidence_sharpness": torch.cat([item["evidence_sharpness"] for item in diagnostics], dim=0),
            "evidence_advantage": torch.cat([item["evidence_advantage"] for item in diagnostics], dim=0),
            "evidence_dispersion": torch.cat([item["evidence_dispersion"] for item in diagnostics], dim=0),
            "evidence_quality": torch.cat([item["evidence_quality"] for item in diagnostics], dim=0),
            "coverage_entropy": torch.cat([item["coverage_entropy"] for item in diagnostics], dim=0),
            "coverage_concentration": torch.cat([item["coverage_concentration"] for item in diagnostics], dim=0),
            "coverage_bonus": torch.cat([item["coverage_bonus"] for item in diagnostics], dim=0),
            "active_match_counts": torch.cat([item["active_match_counts"] for item in diagnostics], dim=0),
            "retained_fraction": torch.cat([item["retained_fraction"] for item in diagnostics], dim=0),
            "alpha": torch.stack([item["alpha"] for item in diagnostics]).mean(),
            "alpha_q": torch.cat([item["alpha_q"] for item in diagnostics], dim=0),
            "global_margin": torch.cat([item["global_margin"] for item in diagnostics], dim=0),
            "local_margin": torch.cat([item["local_margin"] for item in diagnostics], dim=0),
            "mean_budget": torch.stack([item["mean_budget"] for item in diagnostics]).mean(),
            "mean_conditioned_global_entropy": torch.stack(
                [item["mean_conditioned_global_entropy"] for item in diagnostics]
            ).mean(),
            "mean_query_class_attention_entropy": torch.stack(
                [item["mean_query_class_attention_entropy"] for item in diagnostics]
            ).mean(),
            "mean_gate": torch.stack([item["mean_gate"] for item in diagnostics]).mean(),
            "global_branch_ce": torch.stack([item["global_branch_ce"] for item in diagnostics]).mean(),
            "local_branch_ce": torch.stack([item["local_branch_ce"] for item in diagnostics]).mean(),
            "budget_rank_loss": torch.stack([item["budget_rank_loss"] for item in diagnostics]).mean(),
            "local_margin_loss": torch.stack([item["local_margin_loss"] for item in diagnostics]).mean(),
            "budget_residual_reg_loss": torch.stack(
                [item["budget_residual_reg_loss"] for item in diagnostics]
            ).mean(),
            "anchor_consistency_loss": torch.stack(
                [item["anchor_consistency_loss"] for item in diagnostics]
            ).mean(),
            "local_anchor_mix": torch.cat([item["local_anchor_mix"] for item in diagnostics], dim=0).mean(),
            "local_adaptive_mix": torch.cat([item["local_adaptive_mix"] for item in diagnostics], dim=0).mean(),
            "stable_global_embeddings": torch.cat(
                [item["stable_global_embeddings"] for item in diagnostics],
                dim=0,
            ),
            "variant_global_embeddings": torch.cat(
                [item["variant_global_embeddings"] for item in diagnostics],
                dim=0,
            ),
        }
