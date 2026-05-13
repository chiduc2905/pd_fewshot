"""Standalone Ours model and contribution ablations.

The full design is intentionally treated as one coherent model:
J-ECOT-M2/SB-ECOT with UOT fixed to the paper-facing defaults (CATA off by
default; enable with ``--hrot_use_cata true``).
Contribution ablations swap only high-level design choices while leaving the
full model path untouched.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from net.hrot_fsl import HROTFSLResult
from net.jecot_m2 import JECOTM2


OURS_ABLATIONS = frozenset({"full", "balanced_ot", "uniform_evidence", "prototype"})


def normalize_ours_ablation(value: str | None) -> str:
    name = "full" if value is None else str(value).strip().lower().replace("-", "_")
    aliases = {
        "none": "full",
        "ours": "full",
        "uot": "full",
        "full_ot": "balanced_ot",
        "balanced": "balanced_ot",
        "ot": "balanced_ot",
        "uniform": "uniform_evidence",
        "no_adaptive_evidence": "uniform_evidence",
        "no_evidence": "uniform_evidence",
        "proto": "prototype",
        "global_prototype": "prototype",
    }
    name = aliases.get(name, name)
    if name not in OURS_ABLATIONS:
        raise ValueError(f"Unsupported Ours ablation: {value}. Expected one of {sorted(OURS_ABLATIONS)}")
    return name


def apply_ours_design_defaults(kwargs: dict, ablation: str) -> dict:
    """Apply the standalone Ours design contract to HROT/M2 constructor kwargs."""
    kwargs = dict(kwargs)

    # Full Ours: spatial tokens + one UOT budget, with the M2 mass score removed (CATA optional).
    kwargs.setdefault("use_cata", False)
    kwargs.setdefault("cata_num_anchors", 8)
    kwargs.setdefault("cata_num_heads", 4)
    kwargs.setdefault("cata_attn_dropout", 0.0)
    kwargs["ecot_rho_bank"] = "0.8"
    kwargs["ecot_base_rho"] = 0.8
    kwargs["ecot_transport_mode"] = "unbalanced"
    kwargs["ecot_m2_ablate_threshold_mass"] = True
    kwargs["ecot_m2_cost_per_mass_score"] = False
    kwargs["ecot_m2_cost_per_mass_detach_mass"] = False
    kwargs["ecot_m2_use_aqm"] = False
    kwargs["ecot_m2_tau_aqm"] = 1.0
    kwargs["ecot_m2_use_swts"] = False
    kwargs["ecot_m2_swts_temp"] = 1.0

    if ablation == "balanced_ot":
        kwargs["ecot_rho_bank"] = "1.0"
        kwargs["ecot_base_rho"] = 1.0
        kwargs["ecot_transport_mode"] = "balanced"
    elif ablation == "uniform_evidence":
        # Retained for compatibility with older ablation scripts; full Ours
        # already keeps AQM and SWTS off.
        kwargs["ecot_m2_use_aqm"] = False
        kwargs["ecot_m2_use_swts"] = False
    elif ablation == "prototype":
        # The prototype control bypasses the transport forward path, but keeping
        # the same constructor defaults preserves the backbone/projector surface.
        pass

    return kwargs


class OursM2(JECOTM2):
    """Paper-facing Ours entrypoint plus coarse contribution ablations."""

    def __init__(self, *args, ours_ablation: str = "full", **kwargs) -> None:
        self.ours_ablation = normalize_ours_ablation(ours_ablation)
        kwargs = apply_ours_design_defaults(kwargs, self.ours_ablation)
        super().__init__(*args, rho=float(kwargs["ecot_base_rho"]), transport_mode=kwargs["ecot_transport_mode"], **kwargs)

    @property
    def uses_ours_prototype_control(self) -> bool:
        return self.ours_ablation == "prototype"

    def _forward_prototype_episode(
        self,
        query: torch.Tensor,
        support: torch.Tensor,
        *,
        return_aux: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        way_num, shot_num = support.shape[:2]
        query_tokens, _query_hyp, query_hw = self._encode_images(query)
        support_tokens, _support_hyp, support_hw = self._encode_images(
            support.reshape(way_num * shot_num, *support.shape[-3:])
        )
        if query_hw != support_hw:
            raise ValueError(f"Query/support token grids must match, got {query_hw} vs {support_hw}")

        query_repr = F.normalize(query_tokens.mean(dim=1), p=2, dim=-1, eps=self.eps)
        support_repr = support_tokens.mean(dim=1).reshape(way_num, shot_num, -1).mean(dim=1)
        prototypes = F.normalize(support_repr, p=2, dim=-1, eps=self.eps)
        distances = (query_repr[:, None, :] - prototypes[None, :, :]).pow(2).sum(dim=-1)
        logits = -float(self.score_scale) * distances

        if not (return_aux or self.training):
            return logits

        zero = logits.new_zeros(())
        return {
            "logits": logits,
            "class_scores": logits,
            "aux_loss": zero,
            "prototype_distance": distances,
            "prototype_control": logits.new_tensor(1.0),
        }

    def forward(
        self,
        query: torch.Tensor,
        support: torch.Tensor,
        *,
        query_targets: torch.Tensor | None = None,
        support_targets: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        if not self.uses_ours_prototype_control:
            return super().forward(
                query,
                support,
                query_targets=query_targets,
                support_targets=support_targets,
                return_aux=return_aux,
            )

        del query_targets, support_targets
        batch_size, _num_query, _, _, _, _ = self.validate_episode_inputs(query, support)
        needs_payload = bool(return_aux or self.training)
        batch_logits = []
        batch_outputs = []

        for batch_idx in range(batch_size):
            episode_outputs = self._forward_prototype_episode(
                query[batch_idx],
                support[batch_idx],
                return_aux=return_aux,
            )
            if needs_payload:
                batch_outputs.append(episode_outputs)
                batch_logits.append(episode_outputs["logits"])
            else:
                batch_logits.append(episode_outputs)

        logits = torch.cat(batch_logits, dim=0)
        if not needs_payload:
            return logits

        stacked = {
            "logits": logits,
            "class_scores": logits,
            "aux_loss": torch.stack([item["aux_loss"] for item in batch_outputs]).mean(),
            "prototype_distance": torch.cat([item["prototype_distance"] for item in batch_outputs], dim=0),
            "prototype_control": logits.new_tensor(1.0),
        }
        if return_aux:
            return HROTFSLResult(stacked)
        return HROTFSLResult({"logits": logits, "aux_loss": stacked["aux_loss"]})
