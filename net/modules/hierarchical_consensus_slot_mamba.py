"""Consensus-slot Mamba modules for hierarchical few-shot set reasoning."""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from net.ssm.intra_image_mamba import build_2d_position_grid

try:
    from mamba_ssm import Mamba
except ImportError as exc:  # pragma: no cover - handled in model construction
    Mamba = None
    MAMBA_IMPORT_ERROR = exc
else:
    MAMBA_IMPORT_ERROR = None


def _require_mamba() -> None:
    if Mamba is None:
        raise ImportError(
            "hierarchical_consensus_slot_mamba_net requires the official mamba-ssm package"
        ) from MAMBA_IMPORT_ERROR


class AttentiveTokenPool(nn.Module):
    """Pool image tokens into a single shot descriptor with learned token weights."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
        )
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.dim() != 3:
            raise ValueError(
                "tokens must have shape (Batch, Tokens, Dim), "
                f"got {tuple(tokens.shape)}"
            )
        token_logits = self.score(tokens).squeeze(-1)
        token_weights = torch.softmax(token_logits, dim=-1)
        pooled = torch.sum(token_weights.unsqueeze(-1) * tokens, dim=1)
        return self.out_norm(pooled)


class BidirectionalMambaTokenBlock(nn.Module):
    """Residual bidirectional Mamba block over an image token sequence."""

    def __init__(
        self,
        dim: int,
        d_state: int,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        _require_mamba()
        hidden_dim = max(dim, dim * 2)
        self.input_norm = nn.LayerNorm(dim)
        self.position_proj = nn.Sequential(
            nn.Linear(6, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.forward_mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.backward_mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.mix = nn.Linear(dim * 3, dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.output_norm = nn.LayerNorm(dim)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, tokens: torch.Tensor, spatial_hw: Tuple[int, int]) -> torch.Tensor:
        if tokens.dim() != 3:
            raise ValueError(
                "tokens must have shape (Batch, Tokens, Dim), "
                f"got {tuple(tokens.shape)}"
            )
        height, width = spatial_hw
        if tokens.shape[1] != height * width:
            raise ValueError(
                "Token count does not match spatial_hw: "
                f"tokens={tokens.shape[1]} spatial_hw={spatial_hw}"
            )

        position = self.position_proj(
            build_2d_position_grid(height, width, tokens.device, tokens.dtype)
        ).expand(tokens.shape[0], -1, -1)
        normalized = self.input_norm(tokens)
        scan_inputs = normalized + position

        forward_outputs = self.forward_mamba(scan_inputs)
        backward_outputs = torch.flip(
            self.backward_mamba(torch.flip(scan_inputs, dims=[1])),
            dims=[1],
        )
        mixed = self.mix(torch.cat([forward_outputs, backward_outputs, position], dim=-1))
        tokens = self.output_norm(tokens + self.dropout(mixed))
        tokens = tokens + self.dropout(self.ffn(self.ffn_norm(tokens)))
        return tokens


class IntraImageConsensusMambaEncoder(nn.Module):
    """Stack bidirectional Mamba blocks over image tokens."""

    def __init__(
        self,
        dim: int,
        d_state: int,
        depth: int = 1,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if depth < 0:
            raise ValueError("depth must be non-negative")
        self.blocks = nn.ModuleList(
            [
                BidirectionalMambaTokenBlock(
                    dim=dim,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, tokens: torch.Tensor, spatial_hw: Tuple[int, int]) -> torch.Tensor:
        for block in self.blocks:
            tokens = block(tokens, spatial_hw=spatial_hw)
        return tokens


class CanonicalConsensusSlotPool(nn.Module):
    """Induce canonical class slots from an unordered support evidence set."""

    def __init__(
        self,
        dim: int,
        num_slots: int = 4,
        num_heads: int = 4,
        ffn_multiplier: int = 2,
    ) -> None:
        super().__init__()
        if num_slots <= 0:
            raise ValueError("num_slots must be positive")
        hidden_dim = max(dim, dim * ffn_multiplier)
        attn_heads = num_heads if dim % num_heads == 0 else 1
        self.num_slots = int(num_slots)
        self.scale = dim ** -0.5

        self.slot_queries = nn.Parameter(torch.randn(1, num_slots, dim) * 0.02)
        self.input_norm = nn.LayerNorm(dim)
        self.query_norm = nn.LayerNorm(dim)
        self.query_proj = nn.Linear(dim, dim, bias=False)
        self.key_proj = nn.Linear(dim, dim, bias=False)
        self.value_proj = nn.Linear(dim, dim, bias=False)
        self.slot_norm = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, attn_heads, dropout=0.0, batch_first=True)
        self.self_norm = nn.LayerNorm(dim)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        self.summary_proj = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.summary_norm = nn.LayerNorm(dim)

    def forward(
        self,
        write_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pool `(Way, Shot, WriteTokens, Dim)` into canonical slots.

        Returns:
            slots: `(Way, Slots, Dim)`
            summary: `(Way, Dim)`
            slot_assignments: `(Way, Shot, WriteTokens, Slots)`
            shot_consistency: `(Way, Shot)`
            assignment_entropy: `(Way,)`
        """
        if write_tokens.dim() != 4:
            raise ValueError(
                "write_tokens must have shape (Way, Shot, WriteTokens, Dim), "
                f"got {tuple(write_tokens.shape)}"
            )

        way_num, shot_num, write_num, dim = write_tokens.shape
        elements = write_tokens.reshape(way_num, shot_num * write_num, dim)
        element_repr = self.input_norm(elements)
        slot_queries = self.query_norm(self.slot_queries.expand(way_num, -1, -1))

        keys = self.key_proj(element_repr)
        values = self.value_proj(element_repr)
        query_repr = self.query_proj(slot_queries)
        assignment_logits = torch.einsum("bed,bsd->bes", keys, query_repr) * self.scale
        slot_assignments = torch.softmax(assignment_logits, dim=-1)

        slot_mass = slot_assignments.sum(dim=1).clamp_min(1e-6)
        pooled = torch.einsum("bes,bed->bsd", slot_assignments, values) / slot_mass.unsqueeze(-1)
        slots = self.slot_norm(slot_queries + pooled)
        self_attended, _ = self.self_attn(slots, slots, slots)
        slots = self.self_norm(slots + self_attended)
        slots = slots + self.ffn(self.ffn_norm(slots))

        slot_assignments = slot_assignments.reshape(way_num, shot_num, write_num, self.num_slots)
        shot_consistency = slot_assignments.amax(dim=-1).mean(dim=-1)
        assignment_entropy = -(slot_assignments.clamp_min(1e-6) * slot_assignments.clamp_min(1e-6).log()).sum(dim=-1)
        if self.num_slots > 1:
            assignment_entropy = assignment_entropy / math.log(float(self.num_slots))
        assignment_entropy = assignment_entropy.mean(dim=(1, 2))

        summary_input = torch.cat([slots.mean(dim=1), element_repr.mean(dim=1)], dim=-1)
        summary = self.summary_norm(self.summary_proj(summary_input) + slots.mean(dim=1))
        return slots, summary, slot_assignments, shot_consistency, assignment_entropy


class CanonicalSlotMambaBlock(nn.Module):
    """Bidirectional Mamba refinement over canonical class slots."""

    def __init__(
        self,
        dim: int,
        d_state: int,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        _require_mamba()
        hidden_dim = max(dim, dim * 2)
        self.input_norm = nn.LayerNorm(dim)
        self.forward_mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.backward_mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.mix = nn.Linear(dim * 3, dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.output_norm = nn.LayerNorm(dim)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, slots: torch.Tensor, slot_positions: torch.Tensor) -> torch.Tensor:
        normalized = self.input_norm(slots + slot_positions)
        forward_outputs = self.forward_mamba(normalized)
        backward_outputs = torch.flip(
            self.backward_mamba(torch.flip(normalized, dims=[1])),
            dims=[1],
        )
        mixed = self.mix(torch.cat([forward_outputs, backward_outputs, slot_positions], dim=-1))
        slots = self.output_norm(slots + self.dropout(mixed))
        slots = slots + self.dropout(self.ffn(self.ffn_norm(slots)))
        return slots


class CanonicalSlotMambaEncoder(nn.Module):
    """Refine canonical slots and produce a class anchor vector."""

    def __init__(
        self,
        dim: int,
        num_slots: int,
        d_state: int,
        depth: int = 1,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.slot_positions = nn.Parameter(torch.randn(1, num_slots, dim) * 0.02)
        self.blocks = nn.ModuleList(
            [
                CanonicalSlotMambaBlock(
                    dim=dim,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.anchor_proj = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.anchor_norm = nn.LayerNorm(dim)

    def forward(self, slots: torch.Tensor, summary: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if slots.dim() != 3:
            raise ValueError(
                "slots must have shape (Way, Slots, Dim), "
                f"got {tuple(slots.shape)}"
            )
        slot_positions = self.slot_positions[:, : slots.shape[1]].expand(slots.shape[0], -1, -1)
        for block in self.blocks:
            slots = block(slots, slot_positions)
        anchor = self.anchor_norm(
            self.anchor_proj(torch.cat([slots.mean(dim=1), summary], dim=-1)) + slots.mean(dim=1)
        )
        return slots, anchor


class ConsensusPrototypeHead(nn.Module):
    """Build memory-first class prototypes with slot-derived shot reliability."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.scale = dim ** -0.5
        self.shot_norm = nn.LayerNorm(dim)
        self.slot_norm = nn.LayerNorm(dim)
        self.shot_query_proj = nn.Linear(dim, dim, bias=False)
        self.slot_key_proj = nn.Linear(dim, dim, bias=False)
        self.slot_value_proj = nn.Linear(dim, dim, bias=False)

        self.memory_seed = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.score = nn.Sequential(
            nn.LayerNorm(dim * 3 + 2),
            nn.Linear(dim * 3 + 2, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
        )
        self.residual_proj = nn.Sequential(
            nn.LayerNorm(dim * 4),
            nn.Linear(dim * 4, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.uncertainty_head = nn.Sequential(
            nn.LayerNorm(4),
            nn.Linear(4, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        shot_embeddings: torch.Tensor,
        class_slots: torch.Tensor,
        class_anchor: torch.Tensor,
        shot_consistency: torch.Tensor,
        assignment_entropy: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if shot_embeddings.dim() != 3:
            raise ValueError(
                "shot_embeddings must have shape (Way, Shot, Dim), "
                f"got {tuple(shot_embeddings.shape)}"
            )
        if class_anchor.dim() != 2:
            raise ValueError(
                "class_anchor must have shape (Way, Dim), "
                f"got {tuple(class_anchor.shape)}"
            )
        if class_slots.dim() != 3:
            raise ValueError(
                "class_slots must have shape (Way, Slots, Dim), "
                f"got {tuple(class_slots.shape)}"
            )

        shot_query = self.shot_query_proj(self.shot_norm(shot_embeddings))
        slot_keys = self.slot_key_proj(self.slot_norm(class_slots))
        slot_values = self.slot_value_proj(class_slots)
        shot_slot_logits = torch.einsum("wsd,wmd->wsm", shot_query, slot_keys) * self.scale
        shot_slot_weights = torch.softmax(shot_slot_logits, dim=-1)
        reconstructed_shots = torch.einsum("wsm,wmd->wsd", shot_slot_weights, slot_values)

        anchor_expanded = class_anchor.unsqueeze(1).expand(-1, shot_embeddings.shape[1], -1)
        shot_dist = (shot_embeddings - reconstructed_shots).square().mean(dim=-1, keepdim=True)
        score_inputs = torch.cat(
            [
                shot_embeddings,
                reconstructed_shots,
                torch.abs(shot_embeddings - reconstructed_shots),
                shot_consistency.unsqueeze(-1),
                -shot_dist,
            ],
            dim=-1,
        )
        shot_logits = self.score(score_inputs).squeeze(-1)
        shot_weights = torch.softmax(shot_logits, dim=1)

        memory_seed = self.memory_seed(torch.cat([class_anchor, class_slots.mean(dim=1)], dim=-1))
        shot_residual = self.residual_proj(
            torch.cat(
                [
                    shot_embeddings,
                    reconstructed_shots,
                    shot_embeddings - reconstructed_shots,
                    shot_embeddings * reconstructed_shots,
                ],
                dim=-1,
            )
        )
        weighted_residual = torch.sum(shot_weights.unsqueeze(-1) * shot_residual, dim=1)

        agreement_mean = torch.sum(shot_weights * shot_consistency, dim=1)
        dispersion = torch.sum(shot_weights * shot_dist.squeeze(-1), dim=1)
        entropy = -(shot_weights.clamp_min(1e-6) * shot_weights.clamp_min(1e-6).log()).sum(dim=1)
        if shot_embeddings.shape[1] > 1:
            entropy = entropy / math.log(float(shot_embeddings.shape[1]))
        uncertainty_features = torch.stack(
            [
                dispersion,
                entropy,
                1.0 - agreement_mean,
                assignment_entropy,
            ],
            dim=-1,
        )
        uncertainty = torch.sigmoid(self.uncertainty_head(uncertainty_features)).squeeze(-1)
        shrink = uncertainty.unsqueeze(-1)
        prototype = self.norm((1.0 - shrink) * (memory_seed + weighted_residual) + shrink * memory_seed)
        local_gate = 0.5 + uncertainty
        return prototype, shot_weights, uncertainty, local_gate


class ConsensusTransportMemoryPool(nn.Module):
    """Build class-local transport memory from canonical slots and support transport tokens."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.scale = dim ** -0.5
        self.slot_norm = nn.LayerNorm(dim)
        self.transport_norm = nn.LayerNorm(dim)
        self.slot_query_proj = nn.Linear(dim, dim, bias=False)
        self.transport_key_proj = nn.Linear(dim, dim, bias=False)
        self.transport_value_proj = nn.Linear(dim, dim, bias=False)
        self.output_norm = nn.LayerNorm(dim)

    def forward(
        self,
        class_slots: torch.Tensor,
        support_transport: torch.Tensor,
        shot_weights: torch.Tensor,
    ) -> torch.Tensor:
        if class_slots.dim() != 3:
            raise ValueError(
                "class_slots must have shape (Way, Slots, Dim), "
                f"got {tuple(class_slots.shape)}"
            )
        if support_transport.dim() != 4:
            raise ValueError(
                "support_transport must have shape (Way, Shot, Tokens, Dim), "
                f"got {tuple(support_transport.shape)}"
            )
        if shot_weights.dim() != 2:
            raise ValueError(
                "shot_weights must have shape (Way, Shot), "
                f"got {tuple(shot_weights.shape)}"
            )

        way_num, shot_num, token_num, dim = support_transport.shape
        flat_transport = support_transport.reshape(way_num, shot_num * token_num, dim)
        query_repr = self.slot_query_proj(self.slot_norm(class_slots))
        key_repr = self.transport_key_proj(self.transport_norm(flat_transport))
        value_repr = self.transport_value_proj(flat_transport)

        shot_bias = shot_weights.clamp_min(1e-6).log().unsqueeze(-1).expand(-1, -1, token_num).reshape(
            way_num,
            shot_num * token_num,
        )
        attn_logits = torch.einsum("wsd,wtd->wst", query_repr, key_repr) * self.scale
        attn_logits = attn_logits + shot_bias.unsqueeze(1)
        attn_weights = torch.softmax(attn_logits, dim=-1)
        local_memory = torch.einsum("wst,wtd->wsd", attn_weights, value_repr)
        return self.output_norm(class_slots + local_memory)


class SlotConditionedQueryReadout(nn.Module):
    """Read query descriptors from canonical class slots."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.scale = dim ** -0.5
        self.query_norm = nn.LayerNorm(dim)
        self.slot_norm = nn.LayerNorm(dim)
        self.query_proj = nn.Linear(dim, dim, bias=False)
        self.key_proj = nn.Linear(dim, dim, bias=False)
        self.value_proj = nn.Linear(dim, dim, bias=False)
        self.fusion = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, query_embeddings: torch.Tensor, class_slots: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if query_embeddings.dim() != 2:
            raise ValueError(
                "query_embeddings must have shape (NQ, Dim), "
                f"got {tuple(query_embeddings.shape)}"
            )
        if class_slots.dim() != 3:
            raise ValueError(
                "class_slots must have shape (Way, Slots, Dim), "
                f"got {tuple(class_slots.shape)}"
            )

        query_repr = self.query_proj(self.query_norm(query_embeddings))
        slot_repr = self.key_proj(self.slot_norm(class_slots))
        slot_values = self.value_proj(class_slots)

        attn_logits = torch.einsum("nd,wsd->nws", query_repr, slot_repr) * self.scale
        attn_weights = torch.softmax(attn_logits, dim=-1)
        readout = torch.einsum("nws,wsd->nwd", attn_weights, slot_values)
        query_expanded = query_embeddings.unsqueeze(1).expand(-1, class_slots.shape[0], -1)
        query_states = self.out_norm(
            query_expanded + self.fusion(torch.cat([query_expanded, readout], dim=-1))
        )
        return query_states, attn_weights


class ConsensusPairMatcher(nn.Module):
    """Score class-conditioned query states against class prototypes."""

    def __init__(self, dim: int, temperature: float = 16.0) -> None:
        super().__init__()
        self.temperature = float(temperature)
        self.fusion = nn.Sequential(
            nn.LayerNorm(dim * 3),
            nn.Linear(dim * 3, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, query_states: torch.Tensor, class_prototypes: torch.Tensor) -> torch.Tensor:
        if query_states.dim() != 3:
            raise ValueError(
                "query_states must have shape (NQ, Way, Dim), "
                f"got {tuple(query_states.shape)}"
            )
        if class_prototypes.dim() != 2:
            raise ValueError(
                "class_prototypes must have shape (Way, Dim), "
                f"got {tuple(class_prototypes.shape)}"
            )

        proto_expanded = class_prototypes.unsqueeze(0).expand(query_states.shape[0], -1, -1)
        fused = self.fusion(
            torch.cat(
                [
                    query_states,
                    proto_expanded,
                    torch.abs(query_states - proto_expanded),
                ],
                dim=-1,
            )
        )
        fused = self.norm(fused + query_states)
        return self.temperature * torch.sum(
            F.normalize(fused, p=2, dim=-1) * F.normalize(proto_expanded, p=2, dim=-1),
            dim=-1,
        )


class ReliabilityCoupledSlicedWassersteinHead(nn.Module):
    """Combine merged and shot-wise SW using the same support reliability weights."""

    def __init__(
        self,
        sw_distance: nn.Module,
        score_scale: float = 16.0,
        merged_weight: float = 1.0,
        shot_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.sw_distance = sw_distance
        self.score_scale = float(score_scale)
        self.merged_weight = float(merged_weight)
        self.shot_weight = float(shot_weight)

    def forward(
        self,
        query_transport: torch.Tensor,
        support_transport: torch.Tensor,
        class_local_memory: torch.Tensor,
        shot_weights: torch.Tensor,
        local_gate: torch.Tensor,
    ) -> torch.Tensor:
        if query_transport.dim() != 3:
            raise ValueError(
                "query_transport must have shape (NQ, Tokens, Dim), "
                f"got {tuple(query_transport.shape)}"
            )
        if support_transport.dim() != 4:
            raise ValueError(
                "support_transport must have shape (Way, Shot, Tokens, Dim), "
                f"got {tuple(support_transport.shape)}"
            )
        if class_local_memory.dim() != 3:
            raise ValueError(
                "class_local_memory must have shape (Way, Tokens, Dim), "
                f"got {tuple(class_local_memory.shape)}"
            )

        nq = query_transport.shape[0]
        way_num, shot_num, token_num, dim = support_transport.shape

        query_memory = query_transport.unsqueeze(1).expand(-1, way_num, -1, -1)
        support_memory = class_local_memory.unsqueeze(0).expand(nq, -1, -1, -1)
        memory_distance = self.sw_distance(query_memory, support_memory, reduction="none")

        query_shot = query_transport.unsqueeze(1).unsqueeze(2).expand(-1, way_num, shot_num, -1, -1)
        shot_distance = self.sw_distance(
            query_shot.reshape(nq * way_num * shot_num, token_num, dim),
            support_transport.unsqueeze(0).expand(nq, -1, -1, -1, -1).reshape(
                nq * way_num * shot_num,
                token_num,
                dim,
            ),
            reduction="none",
        ).reshape(nq, way_num, shot_num)
        weighted_shot_distance = torch.sum(shot_distance * shot_weights.unsqueeze(0), dim=-1)

        total_distance = self.merged_weight * memory_distance + self.shot_weight * weighted_shot_distance
        return -self.score_scale * local_gate.unsqueeze(0) * total_distance
