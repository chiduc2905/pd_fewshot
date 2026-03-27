"""Hierarchical Consensus-Slot Mamba few-shot model."""

from __future__ import annotations

from typing import Dict, Tuple

import torch

from net.fewshot_common import BaseConv64FewShotModel, feature_map_to_tokens
from net.heads.distribution_alignment_head import HierarchicalQueryMatcher
from net.heads.token_sw_head import TokenSetProjector
from net.metrics.sliced_wasserstein import SlicedWassersteinDistance
from net.modules.hierarchical_consensus_slot_mamba import (
    AttentiveTokenPool,
    IntraImageConsensusMambaEncoder,
)
from net.modules.hierarchical_reliability_consensus import (
    DualConsensusReliabilityHead,
    ReliabilityCoupledTokenSWHead,
    ShotConditionedTokenAdapter,
)
from net.modules.hierarchical_set_v2 import ConsensusClassAggregator, SetConditionedShotRefiner
from net.ssm.set_invariant_pool import SetInvariantMemoryPool


class HierarchicalConsensusSlotMambaNet(BaseConv64FewShotModel):
    """Practical V3: intra-image Mamba + reliability-coupled set residual."""

    def __init__(
        self,
        in_channels: int = 3,
        hidden_dim: int = 64,
        token_dim: int | None = None,
        ssm_state_dim: int = 16,
        temperature: float = 16.0,
        conv64f_pool_last: bool = True,
        backbone_name: str = "conv64f",
        image_size: int = 64,
        resnet12_drop_rate: float = 0.0,
        resnet12_dropblock_size: int = 5,
        use_sw: bool = True,
        sw_weight: float = 0.25,
        sw_num_projections: int = 64,
        sw_p: float = 2.0,
        sw_normalize: bool = True,
        intra_image_depth: int = 1,
        num_write_tokens: int = 4,
        num_transport_tokens: int = 12,
        num_consensus_slots: int = 4,
        pool_heads: int = 4,
        slot_mamba_depth: int = 1,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        mamba_dropout: float = 0.0,
        sw_merged_weight: float = 1.0,
        sw_shot_weight: float = 1.0,
        use_shot_refiner: bool = True,
        use_reliability_residual: bool = True,
        use_token_adapter: bool = True,
        use_local_gate: bool = True,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            backbone_name=backbone_name,
            image_size=image_size,
            resnet12_drop_rate=resnet12_drop_rate,
            resnet12_dropblock_size=resnet12_dropblock_size,
        )
        token_dim = token_dim or hidden_dim
        self.use_sw = bool(use_sw)
        self.sw_weight = float(sw_weight)
        self.use_shot_refiner = bool(use_shot_refiner)
        self.use_reliability_residual = bool(use_reliability_residual)
        self.use_token_adapter = bool(use_token_adapter)
        self.use_local_gate = bool(use_local_gate)

        self.token_projector = TokenSetProjector(hidden_dim, token_dim)
        self.intra_image_encoder = IntraImageConsensusMambaEncoder(
            dim=token_dim,
            d_state=ssm_state_dim,
            depth=intra_image_depth,
            d_conv=mamba_d_conv,
            expand=mamba_expand,
            dropout=mamba_dropout,
        )
        self.token_pool = AttentiveTokenPool(token_dim)
        self.class_pool = SetInvariantMemoryPool(
            dim=token_dim,
            num_memory_tokens=num_consensus_slots,
            num_heads=pool_heads,
        )
        self.shot_refiner = SetConditionedShotRefiner(dim=token_dim, num_heads=pool_heads)
        self.base_aggregator = ConsensusClassAggregator(token_dim)
        self.reliability_head = DualConsensusReliabilityHead(token_dim)
        self.support_token_adapter = ShotConditionedTokenAdapter(token_dim)
        self.matcher = HierarchicalQueryMatcher(token_dim, temperature=temperature)
        self.sw_head = ReliabilityCoupledTokenSWHead(
            sw_distance=SlicedWassersteinDistance(
                num_projections=sw_num_projections,
                p=sw_p,
                reduction="none",
                normalize_inputs=sw_normalize,
            ),
            score_scale=temperature,
            merged_weight=sw_merged_weight,
            shot_weight=sw_shot_weight,
        )

    def _encode_episode(
        self,
        query: torch.Tensor,
        support: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
        way_num, shot_num = support.shape[:2]
        q_features = self.encode(query)
        s_features = self.encode(support.reshape(way_num * shot_num, *support.shape[-3:]))
        spatial_hw = (q_features.shape[-2], q_features.shape[-1])

        all_tokens = self.token_projector(
            feature_map_to_tokens(torch.cat([q_features, s_features], dim=0))
        )
        all_tokens = self.intra_image_encoder(all_tokens, spatial_hw=spatial_hw)

        q_count = query.shape[0]
        q_tokens = all_tokens[:q_count]
        s_tokens = all_tokens[q_count:].reshape(way_num, shot_num, -1, all_tokens.shape[-1])
        return q_tokens, s_tokens, spatial_hw

    def forward(
        self,
        query: torch.Tensor,
        support: torch.Tensor,
        return_aux: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        bsz, _, _, _, _, _ = self.validate_episode_inputs(query, support)

        all_scores = []
        aux_payload = {
            "matcher_logits": [],
            "sw_logits": [],
            "shot_weights": [],
            "class_uncertainty": [],
            "local_gate": [],
        }

        for batch_idx in range(bsz):
            q_tokens, s_tokens, _ = self._encode_episode(query[batch_idx], support[batch_idx])

            q_embeddings = self.token_pool(q_tokens)
            way_num, shot_num, token_num, dim = s_tokens.shape
            support_flat = s_tokens.reshape(way_num * shot_num, token_num, dim)
            s_embeddings = self.token_pool(support_flat).reshape(way_num, shot_num, dim)

            memory_tokens, class_summary = self.class_pool(s_embeddings)
            if self.use_shot_refiner:
                refined_shots = self.shot_refiner(s_embeddings, memory_tokens, class_summary)
            else:
                refined_shots = s_embeddings

            base_class_repr = []
            for class_idx in range(way_num):
                class_repr, _ = self.base_aggregator(refined_shots[class_idx], class_summary[class_idx])
                base_class_repr.append(class_repr)
            base_class_repr = torch.stack(base_class_repr, dim=0)

            if self.use_reliability_residual:
                class_reprs, shot_weights, local_gate = self.reliability_head(
                    refined_shots,
                    s_tokens,
                    base_class_repr,
                    class_summary,
                )
            else:
                class_reprs = base_class_repr
                shot_weights = torch.full(
                    (way_num, shot_num),
                    1.0 / float(max(shot_num, 1)),
                    device=s_tokens.device,
                    dtype=s_tokens.dtype,
                )
                local_gate = torch.ones(way_num, device=s_tokens.device, dtype=s_tokens.dtype)

            if not self.use_local_gate:
                local_gate = torch.ones_like(local_gate)

            if self.use_token_adapter:
                conditioned_support = self.support_token_adapter(
                    s_tokens,
                    refined_shots,
                    class_reprs,
                    shot_weights,
                )
            else:
                conditioned_support = s_tokens
            merged_tokens = torch.sum(
                shot_weights.unsqueeze(-1).unsqueeze(-1) * conditioned_support,
                dim=1,
            )

            matcher_logits = self.matcher(q_embeddings, class_reprs)

            sw_logits = torch.zeros_like(matcher_logits)
            if self.use_sw:
                sw_logits = self.sw_head(
                    query_tokens=q_tokens,
                    support_tokens=conditioned_support,
                    merged_tokens=merged_tokens,
                    shot_weights=shot_weights,
                    local_gate=local_gate,
                )
            logits = matcher_logits + self.sw_weight * sw_logits

            all_scores.append(logits)
            aux_payload["matcher_logits"].append(matcher_logits.detach())
            aux_payload["sw_logits"].append(sw_logits.detach())
            aux_payload["shot_weights"].append(shot_weights.detach())
            aux_payload["class_uncertainty"].append((local_gate - 0.75).detach())
            aux_payload["local_gate"].append(local_gate.detach())

        scores = torch.cat(all_scores, dim=0)
        if not return_aux:
            return scores
        return scores, {
            "matcher_logits": torch.cat(aux_payload["matcher_logits"], dim=0),
            "sw_logits": torch.cat(aux_payload["sw_logits"], dim=0),
            "shot_weights": torch.stack(aux_payload["shot_weights"], dim=0),
            "class_uncertainty": torch.stack(aux_payload["class_uncertainty"], dim=0),
            "local_gate": torch.stack(aux_payload["local_gate"], dim=0),
        }
