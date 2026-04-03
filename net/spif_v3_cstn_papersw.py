"""Paper-SW variant of SPIF-v3 CSTN.

This file keeps the original canonical token projection, atomization, and
permutation-invariant support-set pooling intact, but upgrades the final
canonical matching score to the weighted paper-style SW distance family used by
``spif*_local_papersw``.

The canonical atom masses predicted by the atomizer now define the empirical
transport masses. Query atom masses are used directly, while class atom masses
are pooled across support shots with the same shot-pool weights used for atom
descriptors. This keeps the backbone and support pooling unchanged while making
the final transport score more faithful to the learned atom distributions.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from net.metrics.sliced_wasserstein_weighted import WeightedPaperSlicedWassersteinDistance
from net.spif_v3_cstn import CanonicalSetTransportNet


class CanonicalPaperSlicedWassersteinMetric(nn.Module):
    """Canonical-atom pairwise weighted paper-SW metric."""

    def __init__(
        self,
        d_model: int,
        train_num_projections: int = 128,
        eval_num_projections: int = 512,
        p: float = 2.0,
        normalize_inputs: bool = False,
        train_projection_mode: str = "resample",
        eval_projection_mode: str = "fixed",
        eval_num_repeats: int = 1,
        projection_seed: int = 7,
    ) -> None:
        super().__init__()
        del d_model  # Kept for API symmetry with the legacy CSTN metric.
        self.distance = WeightedPaperSlicedWassersteinDistance(
            train_num_projections=train_num_projections,
            eval_num_projections=eval_num_projections,
            p=p,
            reduction="none",
            normalize_inputs=normalize_inputs,
            train_projection_mode=train_projection_mode,
            eval_projection_mode=eval_projection_mode,
            eval_num_repeats=eval_num_repeats,
            projection_seed=projection_seed,
        )

    def forward(
        self,
        query_atoms: torch.Tensor,
        class_atoms: torch.Tensor,
        query_weights: torch.Tensor | None = None,
        class_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.distance.pairwise_distance(
            query_atoms,
            class_atoms,
            query_weights=query_weights,
            support_weights=class_weights,
            reduction="none",
        )


class CanonicalSetTransportPaperSWNet(CanonicalSetTransportNet):
    """SPIF-v3 CSTN with weighted paper-style SW over canonical atoms."""

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
        assignment_temperature: float = 1.0,
        set_pool_heads: int = 4,
        use_atom_entropy_reg: bool = False,
        atom_entropy_reg_weight: float = 0.01,
        use_support_consistency_reg: bool = False,
        support_consistency_reg_weight: float = 0.01,
        eps: float = 1e-6,
        train_num_projections: int = 128,
        eval_num_projections: int = 512,
        p: float = 2.0,
        normalize_inputs: bool = False,
        train_projection_mode: str = "resample",
        eval_projection_mode: str = "fixed",
        eval_num_repeats: int = 1,
        projection_seed: int = 7,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            d_model=d_model,
            temperature=temperature,
            backbone_name=backbone_name,
            image_size=image_size,
            resnet12_drop_rate=resnet12_drop_rate,
            resnet12_dropblock_size=resnet12_dropblock_size,
            num_atoms=num_atoms,
            num_sw_projections=max(int(train_num_projections), int(eval_num_projections)),
            assignment_temperature=assignment_temperature,
            set_pool_heads=set_pool_heads,
            use_atom_entropy_reg=use_atom_entropy_reg,
            atom_entropy_reg_weight=atom_entropy_reg_weight,
            use_support_consistency_reg=use_support_consistency_reg,
            support_consistency_reg_weight=support_consistency_reg_weight,
            eps=eps,
        )
        self.metric = CanonicalPaperSlicedWassersteinMetric(
            d_model=self.d_model,
            train_num_projections=train_num_projections,
            eval_num_projections=eval_num_projections,
            p=p,
            normalize_inputs=normalize_inputs,
            train_projection_mode=train_projection_mode,
            eval_projection_mode=eval_projection_mode,
            eval_num_repeats=eval_num_repeats,
            projection_seed=projection_seed,
        )

    @staticmethod
    def _pool_class_atom_mass(
        support_atom_mass: torch.Tensor,
        shot_pool_weights: torch.Tensor,
    ) -> torch.Tensor:
        if support_atom_mass.dim() != 3:
            raise ValueError(
                "support_atom_mass must have shape (Way, Shot, Atoms), "
                f"got {tuple(support_atom_mass.shape)}"
            )
        if shot_pool_weights.dim() != 3:
            raise ValueError(
                "shot_pool_weights must have shape (Way, Atoms, Shot), "
                f"got {tuple(shot_pool_weights.shape)}"
            )
        if (
            support_atom_mass.shape[0] != shot_pool_weights.shape[0]
            or support_atom_mass.shape[1] != shot_pool_weights.shape[2]
        ):
            raise ValueError(
                "support_atom_mass and shot_pool_weights must align in (Way, Shot): "
                f"mass={tuple(support_atom_mass.shape)} weights={tuple(shot_pool_weights.shape)}"
            )
        if support_atom_mass.shape[2] != shot_pool_weights.shape[1]:
            raise ValueError(
                "support_atom_mass atoms must match shot_pool_weights atoms: "
                f"mass={tuple(support_atom_mass.shape)} weights={tuple(shot_pool_weights.shape)}"
            )

        support_atom_mass = support_atom_mass.permute(0, 2, 1).contiguous()
        return (shot_pool_weights * support_atom_mass).sum(dim=-1)

    def _forward_episode(self, query: torch.Tensor, support: torch.Tensor) -> dict[str, torch.Tensor]:
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
        class_atom_mass = self._pool_class_atom_mass(
            support_atom_mass=support_atom_mass,
            shot_pool_weights=shot_pool_weights,
        )

        canonical_sw_distances = self.metric(
            query_atoms.descriptors,
            class_atoms,
            query_weights=query_atoms.atom_mass,
            class_weights=class_atom_mass,
        )
        logits = -self.logit_scale * canonical_sw_distances

        output = {
            "logits": logits,
            "canonical_sw_distances": canonical_sw_distances.detach(),
            "query_atom_mass": query_atoms.atom_mass.detach(),
            "support_atom_mass": support_atom_mass.detach(),
            "shot_pool_weights": shot_pool_weights.detach(),
            "class_atom_mass": class_atom_mass.detach(),
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
    ) -> torch.Tensor | dict[str, torch.Tensor]:
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
            "class_atom_mass": torch.stack(
                [episode["class_atom_mass"] for episode in episode_outputs],
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
