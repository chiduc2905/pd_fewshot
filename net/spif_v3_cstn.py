"""Canonical Set Transport Network (SPIF-v3 CSTN).

README-style summary
- What it does:
  Dense backbone tokens are projected into a shared latent metric space,
  softly assigned to a learnable canonical atom bank, pooled into canonical
  atom descriptors, aggregated across same-class support shots with a
  permutation-invariant set module, and matched with sliced Wasserstein
  distance in canonical atom space.
- Why it is different from SPIF-v2:
  SPIF-v2 factorizes tokens into stable/variant branches and scores with a
  prototype + partial local matcher. SPIF-v3 CSTN instead treats each image
  as an unordered local descriptor set, transports descriptors into a shared
  canonical atom basis, and compares support/query distributions directly in
  that canonical space.
- Why the support aggregator is permutation-aware:
  Support-shot aggregation never scans shots as a sequence. Each canonical
  atom uses a shared seed query to pool over the support-shot set, followed by
  atom-wise self-attention and symmetric reductions only. Reordering support
  shots therefore leaves the class representation unchanged up to numerical
  precision.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from net.fewshot_common import BaseConv64FewShotModel, feature_map_to_tokens


def _normalize_mass(mass: torch.Tensor, eps: float) -> torch.Tensor:
    return mass / mass.sum(dim=-1, keepdim=True).clamp_min(eps)


def _mean_normalized_entropy(mass: torch.Tensor, eps: float) -> torch.Tensor:
    distribution = _normalize_mass(mass, eps=eps)
    entropy = -(distribution.clamp_min(eps) * distribution.clamp_min(eps).log()).sum(dim=-1)
    normalizer = math.log(max(int(distribution.shape[-1]), 2))
    return (entropy / normalizer).mean()


@dataclass
class CanonicalAtomOutputs:
    """Canonical atom descriptors and assignment statistics for one token set."""

    descriptors: torch.Tensor
    assignment_weights: torch.Tensor
    atom_mass: torch.Tensor


class CanonicalTokenProjector(nn.Module):
    """Project spatial tokens into the canonical metric space."""

    def __init__(self, input_dim: int, d_model: int) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.proj = nn.Linear(input_dim, d_model)
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.output_norm(self.proj(self.input_norm(tokens)))


class CanonicalAtomizer(nn.Module):
    """Softly assign tokens to a shared canonical atom bank."""

    def __init__(
        self,
        d_model: int,
        num_atoms: int,
        assignment_temperature: float = 1.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if num_atoms <= 0:
            raise ValueError("num_atoms must be positive")
        if assignment_temperature <= 0.0:
            raise ValueError("assignment_temperature must be positive")

        self.num_atoms = int(num_atoms)
        self.assignment_temperature = float(assignment_temperature)
        self.eps = float(eps)

        self.token_norm = nn.LayerNorm(d_model)
        self.atom_bank = nn.Parameter(torch.randn(self.num_atoms, d_model) * 0.02)
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, tokens: torch.Tensor) -> CanonicalAtomOutputs:
        if tokens.dim() != 3:
            raise ValueError(
                "tokens must have shape (Batch, Tokens, Dim), "
                f"got {tuple(tokens.shape)}"
            )

        token_repr = self.token_norm(tokens)
        normalized_tokens = F.normalize(token_repr, p=2, dim=-1, eps=self.eps)
        normalized_atoms = F.normalize(self.atom_bank, p=2, dim=-1, eps=self.eps)

        similarity = torch.einsum("bnd,md->bnm", normalized_tokens, normalized_atoms)
        assignment_weights = torch.softmax(similarity / self.assignment_temperature, dim=-1)

        atom_mass = assignment_weights.sum(dim=1)
        pooled = torch.einsum("bnm,bnd->bmd", assignment_weights, token_repr)
        pooled = pooled / atom_mass.unsqueeze(-1).clamp_min(self.eps)
        descriptors = self.output_norm(pooled)

        return CanonicalAtomOutputs(
            descriptors=descriptors,
            assignment_weights=assignment_weights,
            atom_mass=atom_mass,
        )


class CanonicalSupportSetPool(nn.Module):
    """Permutation-invariant support-shot pooling with shared canonical seeds."""

    def __init__(
        self,
        d_model: int,
        num_atoms: int,
        num_heads: int = 4,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if num_atoms <= 0:
            raise ValueError("num_atoms must be positive")

        attn_heads = num_heads if d_model % num_heads == 0 else 1
        hidden_dim = max(d_model, d_model * 2)

        self.num_atoms = int(num_atoms)
        self.eps = float(eps)
        self.score_scale = d_model ** -0.5

        self.seed_queries = nn.Parameter(torch.randn(1, self.num_atoms, d_model) * 0.02)
        self.shot_norm = nn.LayerNorm(d_model)
        self.atom_self_attn = nn.MultiheadAttention(d_model, attn_heads, dropout=0.0, batch_first=True)
        self.attn_norm = nn.LayerNorm(d_model)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )
        self.output_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        shot_atoms: torch.Tensor,
        atom_mass: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if shot_atoms.dim() != 4:
            raise ValueError(
                "shot_atoms must have shape (Way, Shot, Atoms, Dim), "
                f"got {tuple(shot_atoms.shape)}"
            )
        if shot_atoms.shape[2] != self.num_atoms:
            raise ValueError(
                f"Expected {self.num_atoms} atoms, got {shot_atoms.shape[2]}"
            )
        if atom_mass is not None and atom_mass.shape[:3] != shot_atoms.shape[:3]:
            raise ValueError(
                "atom_mass must align with shot_atoms in (Way, Shot, Atoms), "
                f"got mass={tuple(atom_mass.shape)} atoms={tuple(shot_atoms.shape)}"
            )

        # Pool each canonical atom across the unordered support-shot set.
        shot_values = shot_atoms.permute(0, 2, 1, 3).contiguous()
        shot_keys = self.shot_norm(shot_values)
        seed_queries = self.seed_queries.expand(shot_values.shape[0], -1, -1)

        scores = (shot_keys * seed_queries.unsqueeze(2)).sum(dim=-1) * self.score_scale
        if atom_mass is not None:
            scores = scores + atom_mass.permute(0, 2, 1).clamp_min(self.eps).log()
        shot_pool_weights = torch.softmax(scores, dim=-1)

        pooled_atoms = torch.einsum("wms,wmsd->wmd", shot_pool_weights, shot_values)
        refined_atoms, _ = self.atom_self_attn(pooled_atoms, pooled_atoms, pooled_atoms)
        pooled_atoms = self.attn_norm(pooled_atoms + refined_atoms)
        pooled_atoms = self.output_norm(pooled_atoms + self.ffn(self.ffn_norm(pooled_atoms)))
        return pooled_atoms, shot_pool_weights


class CanonicalSlicedWassersteinMetric(nn.Module):
    """Pairwise sliced Wasserstein distance over canonical atom sets."""

    def __init__(
        self,
        d_model: int,
        num_projections: int = 64,
        normalize_inputs: bool = True,
        learnable_projections: bool = False,
        projection_seed: int = 7,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if num_projections <= 0:
            raise ValueError("num_projections must be positive")

        self.normalize_inputs = bool(normalize_inputs)
        self.learnable_projections = bool(learnable_projections)
        self.eps = float(eps)

        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(projection_seed))
        projection_bank = torch.randn(d_model, int(num_projections), generator=generator)
        projection_bank = F.normalize(projection_bank, p=2, dim=0, eps=self.eps)
        if self.learnable_projections:
            self.projection_bank = nn.Parameter(projection_bank)
        else:
            self.register_buffer("projection_bank", projection_bank, persistent=False)

    def forward(self, query_atoms: torch.Tensor, class_atoms: torch.Tensor) -> torch.Tensor:
        if query_atoms.dim() != 3:
            raise ValueError(
                "query_atoms must have shape (NumQuery, Atoms, Dim), "
                f"got {tuple(query_atoms.shape)}"
            )
        if class_atoms.dim() != 3:
            raise ValueError(
                "class_atoms must have shape (Way, Atoms, Dim), "
                f"got {tuple(class_atoms.shape)}"
            )
        if query_atoms.shape[1:] != class_atoms.shape[1:]:
            raise ValueError(
                "query/class atom shapes must match in (Atoms, Dim): "
                f"query={tuple(query_atoms.shape)} class={tuple(class_atoms.shape)}"
            )

        projections = F.normalize(self.projection_bank, p=2, dim=0, eps=self.eps)
        if self.normalize_inputs:
            query_atoms = F.normalize(query_atoms, p=2, dim=-1, eps=self.eps)
            class_atoms = F.normalize(class_atoms, p=2, dim=-1, eps=self.eps)

        query_proj = torch.matmul(query_atoms, projections)
        class_proj = torch.matmul(class_atoms, projections)

        query_sorted = torch.sort(query_proj, dim=1).values.unsqueeze(1)
        class_sorted = torch.sort(class_proj, dim=1).values.unsqueeze(0)
        squared_error = (query_sorted - class_sorted).pow(2)
        return squared_error.mean(dim=2).mean(dim=-1)


class CanonicalSetTransportNet(BaseConv64FewShotModel):
    """Few-shot classifier with canonical transport atoms and SW matching."""

    def __init__(
        self,
        in_channels: int = 3,
        hidden_dim: int = 64,
        d_model: int | None = None,
        temperature: float = 16.0,
        backbone_name: str = "resnet12",
        image_size: int = 64,
        resnet12_drop_rate: float = 0.0,
        resnet12_dropblock_size: int = 5,
        num_atoms: int = 8,
        num_sw_projections: int = 64,
        assignment_temperature: float = 1.0,
        set_pool_heads: int = 4,
        use_atom_entropy_reg: bool = False,
        atom_entropy_reg_weight: float = 0.01,
        use_support_consistency_reg: bool = False,
        support_consistency_reg_weight: float = 0.01,
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
        d_model = int(d_model or hidden_dim)
        self.d_model = d_model
        self.num_atoms = int(num_atoms)
        self.logit_scale = float(temperature)
        self.use_atom_entropy_reg = bool(use_atom_entropy_reg)
        self.atom_entropy_reg_weight = float(atom_entropy_reg_weight)
        self.use_support_consistency_reg = bool(use_support_consistency_reg)
        self.support_consistency_reg_weight = float(support_consistency_reg_weight)
        self.eps = float(eps)
        self.supports_aux_loss = self.use_atom_entropy_reg or self.use_support_consistency_reg

        self.token_projector = CanonicalTokenProjector(hidden_dim, d_model)
        self.atomizer = CanonicalAtomizer(
            d_model=d_model,
            num_atoms=self.num_atoms,
            assignment_temperature=assignment_temperature,
            eps=eps,
        )
        self.support_pool = CanonicalSupportSetPool(
            d_model=d_model,
            num_atoms=self.num_atoms,
            num_heads=set_pool_heads,
            eps=eps,
        )
        self.metric = CanonicalSlicedWassersteinMetric(
            d_model=d_model,
            num_projections=num_sw_projections,
            normalize_inputs=True,
            eps=eps,
        )

    def _encode_episode(self, query: torch.Tensor, support: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        way_num, shot_num = support.shape[:2]
        query_features = self.encode(query)
        support_features = self.encode(support.reshape(way_num * shot_num, *support.shape[-3:]))

        query_tokens = self.token_projector(feature_map_to_tokens(query_features))
        support_tokens = self.token_projector(feature_map_to_tokens(support_features)).reshape(
            way_num,
            shot_num,
            -1,
            query_tokens.shape[-1],
        )
        return query_tokens, support_tokens

    def _compute_aux_loss(
        self,
        query_atom_mass: torch.Tensor,
        support_atom_descriptors: torch.Tensor,
        support_atom_mass: torch.Tensor,
        class_atoms: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        zero = class_atoms.new_zeros(())

        query_entropy = _mean_normalized_entropy(query_atom_mass, eps=self.eps)
        support_entropy = _mean_normalized_entropy(
            support_atom_mass.reshape(-1, support_atom_mass.shape[-1]),
            eps=self.eps,
        )

        atom_entropy_loss = zero
        if self.use_atom_entropy_reg:
            atom_entropy_loss = 1.0 - 0.5 * (query_entropy + support_entropy)

        support_consistency_loss = zero
        if self.use_support_consistency_reg and support_atom_descriptors.shape[1] > 1:
            diff = (support_atom_descriptors - class_atoms.unsqueeze(1)).pow(2).mean(dim=-1)
            atom_weights = _normalize_mass(support_atom_mass, eps=self.eps)
            support_consistency_loss = (diff * atom_weights).sum(dim=-1).mean()

        aux_loss = (
            self.atom_entropy_reg_weight * atom_entropy_loss
            + self.support_consistency_reg_weight * support_consistency_loss
        )
        return {
            "aux_loss": aux_loss,
            "atom_entropy_loss": atom_entropy_loss,
            "support_consistency_loss": support_consistency_loss,
            "query_atom_entropy": query_entropy,
            "support_atom_entropy": support_entropy,
        }

    def _forward_episode(self, query: torch.Tensor, support: torch.Tensor) -> Dict[str, torch.Tensor]:
        query_tokens, support_tokens = self._encode_episode(query, support)
        way_num, shot_num = support_tokens.shape[:2]

        query_atoms = self.atomizer(query_tokens)
        support_atom_outputs = self.atomizer(support_tokens.reshape(-1, *support_tokens.shape[-2:]))

        support_atom_descriptors = support_atom_outputs.descriptors.reshape(
            way_num,
            shot_num,
            self.num_atoms,
            self.d_model,
        )
        support_atom_mass = support_atom_outputs.atom_mass.reshape(
            way_num,
            shot_num,
            self.num_atoms,
        )
        class_atoms, shot_pool_weights = self.support_pool(
            support_atom_descriptors,
            atom_mass=support_atom_mass,
        )

        canonical_sw_distances = self.metric(query_atoms.descriptors, class_atoms)
        logits = -self.logit_scale * canonical_sw_distances

        output = {
            "logits": logits,
            "canonical_sw_distances": canonical_sw_distances.detach(),
            "query_atom_mass": query_atoms.atom_mass.detach(),
            "support_atom_mass": support_atom_mass.detach(),
            "shot_pool_weights": shot_pool_weights.detach(),
        }
        output.update(
            self._compute_aux_loss(
                query_atom_mass=query_atoms.atom_mass,
                support_atom_descriptors=support_atom_descriptors,
                support_atom_mass=support_atom_mass,
                class_atoms=class_atoms,
            )
        )
        return output

    def forward(
        self,
        query: torch.Tensor,
        support: torch.Tensor,
        return_aux: bool = False,
    ) -> torch.Tensor | Dict[str, torch.Tensor]:
        bsz, _, _, _, _, _ = self.validate_episode_inputs(query, support)

        episode_outputs = []
        for batch_idx in range(bsz):
            episode_outputs.append(self._forward_episode(query[batch_idx], support[batch_idx]))

        logits = torch.cat([episode["logits"] for episode in episode_outputs], dim=0)

        if not self.training and not return_aux:
            return logits
        if not self.supports_aux_loss and not return_aux:
            return logits

        output = {
            "logits": logits,
            "canonical_sw_distances": torch.cat(
                [episode["canonical_sw_distances"] for episode in episode_outputs],
                dim=0,
            ),
            "query_atom_mass": torch.cat(
                [episode["query_atom_mass"] for episode in episode_outputs],
                dim=0,
            ),
            "support_atom_mass": torch.stack(
                [episode["support_atom_mass"] for episode in episode_outputs],
                dim=0,
            ),
            "shot_pool_weights": torch.stack(
                [episode["shot_pool_weights"] for episode in episode_outputs],
                dim=0,
            ),
            "aux_loss": torch.stack([episode["aux_loss"] for episode in episode_outputs]).mean(),
            "atom_entropy_loss": torch.stack(
                [episode["atom_entropy_loss"] for episode in episode_outputs]
            ).mean(),
            "support_consistency_loss": torch.stack(
                [episode["support_consistency_loss"] for episode in episode_outputs]
            ).mean(),
            "query_atom_entropy": torch.stack(
                [episode["query_atom_entropy"] for episode in episode_outputs]
            ).mean(),
            "support_atom_entropy": torch.stack(
                [episode["support_atom_entropy"] for episode in episode_outputs]
            ).mean(),
        }
        return output
