"""PARS-Net: Prior-Augmented Reconstruction Subspace Network.

The model key remains ``warn`` for runner compatibility, but the implementation
follows the PARS thesis: build a support basis from the shot-agnostic support
measure, retrieve class-agnostic prior subspaces with sliced Wasserstein
distance, estimate support insufficiency, and complete the class subspace
before query-conditioned reconstruction scoring.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from net.fewshot_common import BaseConv64FewShotModel, feature_map_to_tokens
from net.heads.token_sw_head import TokenSetProjector
from net.metrics.sliced_wasserstein import SlicedWassersteinDistance
from net.modules.warn_transformer import (
    IntraImageTransformerEncoder,
    QueryConditionedBasisDistiller,
    SupportSetBasisDistiller,
)


class PriorAugmentedReconstructionSubspaceNet(BaseConv64FewShotModel):
    """Support-insufficiency-aware few-shot classifier with prior completion."""

    def __init__(
        self,
        in_channels: int = 3,
        hidden_dim: int = 64,
        token_dim: int = 128,
        backbone_name: str = "resnet12",
        image_size: int = 64,
        resnet12_drop_rate: float = 0.0,
        resnet12_dropblock_size: int = 5,
        transformer_depth: int = 1,
        num_heads: int = 4,
        ffn_multiplier: int = 4,
        num_support_basis_tokens: int = 8,
        num_readout_basis_tokens: int = 8,
        num_prior_subspaces: int = 16,
        num_prior_atoms: int = 8,
        attn_dropout: float = 0.0,
        sw_num_projections: int = 64,
        sw_p: float = 2.0,
        sw_normalize: bool = True,
        prior_temperature: float = 1.0,
        score_scale: float = 16.0,
        diversity_weight: float = 0.01,
        recon_lambda_init: float = 0.1,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            backbone_name=backbone_name,
            image_size=image_size,
            resnet12_drop_rate=resnet12_drop_rate,
            resnet12_dropblock_size=resnet12_dropblock_size,
        )
        if num_prior_subspaces <= 0:
            raise ValueError("num_prior_subspaces must be positive")
        if num_prior_atoms <= 0:
            raise ValueError("num_prior_atoms must be positive")

        self.prior_temperature = float(prior_temperature)
        self.score_scale = float(score_scale)
        self.diversity_weight = float(diversity_weight)

        self.token_projector = TokenSetProjector(hidden_dim, token_dim)
        self.token_encoder = IntraImageTransformerEncoder(
            dim=token_dim,
            depth=transformer_depth,
            num_heads=num_heads,
            ffn_multiplier=ffn_multiplier,
            dropout=attn_dropout,
        )
        self.support_distiller = SupportSetBasisDistiller(
            dim=token_dim,
            num_basis_tokens=num_support_basis_tokens,
            num_heads=num_heads,
            ffn_multiplier=ffn_multiplier,
            dropout=attn_dropout,
        )
        self.readout_distiller = QueryConditionedBasisDistiller(
            dim=token_dim,
            num_basis_tokens=num_readout_basis_tokens,
            num_heads=num_heads,
            ffn_multiplier=ffn_multiplier,
            dropout=attn_dropout,
        )
        self.router = SlicedWassersteinDistance(
            num_projections=sw_num_projections,
            p=sw_p,
            reduction="none",
            normalize_inputs=sw_normalize,
        )
        self.prior_bank = nn.Parameter(torch.randn(num_prior_subspaces, num_prior_atoms, token_dim) * 0.02)
        self.query_summary = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )
        self.gate_mlp = nn.Sequential(
            nn.LayerNorm(2),
            nn.Linear(2, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, 1),
        )
        final_linear = self.gate_mlp[-1]
        if isinstance(final_linear, nn.Linear):
            nn.init.zeros_(final_linear.weight)
            nn.init.constant_(final_linear.bias, -2.0)
        self.fusion_distiller = SupportSetBasisDistiller(
            dim=token_dim,
            num_basis_tokens=num_readout_basis_tokens,
            num_heads=num_heads,
            ffn_multiplier=ffn_multiplier,
            dropout=attn_dropout,
        )
        self.log_recon_lambda = nn.Parameter(torch.log(torch.tensor(float(recon_lambda_init))))

    def _encode_episode(self, query: torch.Tensor, support: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        way_num, shot_num = support.shape[:2]
        q_features = self.encode(query)
        s_features = self.encode(support.reshape(way_num * shot_num, *support.shape[-3:]))
        spatial_hw = (q_features.shape[-2], q_features.shape[-1])

        all_tokens = self.token_projector(feature_map_to_tokens(torch.cat([q_features, s_features], dim=0)))
        all_tokens = self.token_encoder(all_tokens, spatial_hw=spatial_hw)

        q_count = query.shape[0]
        q_tokens = all_tokens[:q_count]
        s_tokens = all_tokens[q_count:].reshape(way_num, shot_num, -1, all_tokens.shape[-1])
        return q_tokens, s_tokens

    @staticmethod
    def _build_projector(basis: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
        pair_count, num_basis, _ = basis.shape
        gram = torch.bmm(basis, basis.transpose(1, 2))
        identity = torch.eye(num_basis, device=basis.device, dtype=basis.dtype).unsqueeze(0)
        coeff = torch.linalg.solve(gram + lam * identity, basis)
        return torch.bmm(basis.transpose(1, 2), coeff)

    def _support_measure(self, support_tokens: torch.Tensor) -> torch.Tensor:
        way_num, shot_num, token_num, dim = support_tokens.shape
        return support_tokens.reshape(way_num, shot_num * token_num, dim)

    def _retrieve_prior(
        self,
        support_basis: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        way_num, _, dim = support_basis.shape
        num_prior = self.prior_bank.shape[0]

        support_expand = support_basis.unsqueeze(1).expand(-1, num_prior, -1, -1)
        prior_expand = self.prior_bank.unsqueeze(0).expand(way_num, -1, -1, -1)
        distances = self.router(
            support_expand.reshape(way_num * num_prior, support_basis.shape[1], dim),
            prior_expand.reshape(way_num * num_prior, self.prior_bank.shape[1], dim),
            reduction="none",
        ).reshape(way_num, num_prior)

        weights = torch.softmax(-distances / max(self.prior_temperature, 1e-6), dim=-1)
        retrieved = torch.einsum("wp,pmd->wmd", weights, self.prior_bank)
        entropy = -(weights.clamp_min(1e-6) * weights.clamp_min(1e-6).log()).sum(dim=-1)
        if num_prior > 1:
            entropy = entropy / torch.log(torch.tensor(float(num_prior), device=entropy.device, dtype=entropy.dtype))
        return retrieved, weights, entropy

    def _support_insufficiency(
        self,
        support_measure: torch.Tensor,
        support_basis: torch.Tensor,
    ) -> torch.Tensor:
        lam = F.softplus(self.log_recon_lambda) + 1e-6
        projector = self._build_projector(support_basis, lam)
        recon = torch.bmm(support_measure, projector)
        return (support_measure - recon).pow(2).mean(dim=(1, 2))

    def _complete_basis(
        self,
        support_basis: torch.Tensor,
        prior_basis: torch.Tensor,
        insufficiency: torch.Tensor,
        prior_entropy: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gate_inputs = torch.stack([insufficiency, prior_entropy], dim=-1)
        completion_gate = torch.sigmoid(self.gate_mlp(gate_inputs)).squeeze(-1)

        fused_measure = torch.cat(
            [
                (1.0 - completion_gate).unsqueeze(-1).unsqueeze(-1) * support_basis,
                completion_gate.unsqueeze(-1).unsqueeze(-1) * prior_basis,
            ],
            dim=1,
        )
        completed_basis = self.fusion_distiller(fused_measure)
        return completed_basis, completion_gate

    def _query_conditioned_basis(
        self,
        query_tokens: torch.Tensor,
        class_basis: torch.Tensor,
    ) -> torch.Tensor:
        nq, _, dim = query_tokens.shape
        way_num = class_basis.shape[0]
        query_summary = self.query_summary(query_tokens.mean(dim=1))
        pair_summary = query_summary.unsqueeze(1).expand(-1, way_num, -1).reshape(nq * way_num, dim)
        pair_basis = class_basis.unsqueeze(0).expand(nq, -1, -1, -1).reshape(
            nq * way_num,
            class_basis.shape[1],
            dim,
        )
        readout_basis = self.readout_distiller(pair_summary, pair_basis)
        return readout_basis.reshape(nq, way_num, readout_basis.shape[1], dim)

    def _reconstruction_logits(self, query_tokens: torch.Tensor, basis_tokens: torch.Tensor) -> torch.Tensor:
        nq, token_num, dim = query_tokens.shape
        _, way_num, _, _ = basis_tokens.shape
        pair_count = nq * way_num

        basis = basis_tokens.reshape(pair_count, basis_tokens.shape[2], dim)
        query_pair = query_tokens.unsqueeze(1).expand(-1, way_num, -1, -1).reshape(pair_count, token_num, dim)
        lam = F.softplus(self.log_recon_lambda) + 1e-6
        projector = self._build_projector(basis, lam)
        recon = torch.bmm(query_pair, projector)
        residual = (query_pair - recon).pow(2).mean(dim=(1, 2)).reshape(nq, way_num)
        return -self.score_scale * residual

    def _class_basis_logits(self, query_tokens: torch.Tensor, class_basis: torch.Tensor) -> torch.Tensor:
        nq, token_num, dim = query_tokens.shape
        way_num, basis_tokens, _ = class_basis.shape
        pair_count = nq * way_num

        basis = class_basis.unsqueeze(0).expand(nq, -1, -1, -1).reshape(pair_count, basis_tokens, dim)
        query_pair = query_tokens.unsqueeze(1).expand(-1, way_num, -1, -1).reshape(pair_count, token_num, dim)
        lam = F.softplus(self.log_recon_lambda) + 1e-6
        projector = self._build_projector(basis, lam)
        recon = torch.bmm(query_pair, projector)
        residual = (query_pair - recon).pow(2).mean(dim=(1, 2)).reshape(nq, way_num)
        return -self.score_scale * residual

    @staticmethod
    def _basis_diversity_loss(basis_tokens: torch.Tensor) -> torch.Tensor:
        batch, num_basis, _ = basis_tokens.shape
        if num_basis <= 1:
            return basis_tokens.new_zeros(())
        normalized = F.normalize(basis_tokens, p=2, dim=-1)
        gram = torch.bmm(normalized, normalized.transpose(1, 2))
        identity = torch.eye(num_basis, device=basis_tokens.device, dtype=basis_tokens.dtype).unsqueeze(0)
        return ((gram - identity) ** 2).sum(dim=(1, 2)).mean() / float(batch)

    def forward(self, query: torch.Tensor, support: torch.Tensor) -> torch.Tensor | Dict[str, torch.Tensor]:
        bsz, _, _, _, _, _ = self.validate_episode_inputs(query, support)

        logits_per_batch = []
        aux_losses = []
        completion_gates = []

        for batch_idx in range(bsz):
            q_tokens, s_tokens = self._encode_episode(query[batch_idx], support[batch_idx])
            support_measure = self._support_measure(s_tokens)
            support_basis = self.support_distiller(support_measure)
            base_logits = self._class_basis_logits(q_tokens, support_basis)
            prior_basis, prior_weights, prior_entropy = self._retrieve_prior(support_basis)
            insufficiency = self._support_insufficiency(support_measure, support_basis)
            class_basis, completion_gate = self._complete_basis(
                support_basis,
                prior_basis,
                insufficiency,
                prior_entropy,
            )
            completed_logits = self._class_basis_logits(q_tokens, class_basis)
            logits = base_logits + completion_gate.unsqueeze(0) * (completed_logits - base_logits)

            logits_per_batch.append(logits)
            completion_gates.append(completion_gate.detach())

            if self.training and self.diversity_weight > 0.0:
                aux_losses.append(
                    self.diversity_weight
                    * (
                        self._basis_diversity_loss(support_basis)
                        + self._basis_diversity_loss(class_basis)
                    )
                )

        logits = torch.cat(logits_per_batch, dim=0)
        if self.training and aux_losses:
            return {
                "logits": logits,
                "aux_loss": torch.stack(aux_losses).mean(),
                "support_gate": torch.stack(completion_gates).mean(),
            }
        return logits


# Keep the previous import path stable for the registry.
WassersteinRoutedAttentiveReconstructionNet = PriorAugmentedReconstructionSubspaceNet
