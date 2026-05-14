"""Ours-CPM: Cost-Per-Mass multi-budget variant.

Addresses the monotone-score degeneracy of single-budget Ours by:
  1. Using ``score = -alpha * cost / (mass + eps)`` instead of ``score = -cost``
  2. Expanding the rho-bank from a single value to multiple budgets (default
     0.7, 0.8, 0.9) so the EpisodeBudgetController sees non-redundant signals

With score = -cost (Ours default), all budgets give monotonically-ordered
scores across classes, rendering the budget policy trivial.  cost/mass
normalisation breaks this monotonicity because the *average* transport cost per
unit of matched mass behaves differently for correct vs. wrong classes as the
budget grows:
  - correct class: additional tokens still match well -> cost/mass stays low
  - wrong class:   forced to match bad tokens       -> cost/mass rises fast

The mass denominator is detached by default to avoid 1/mass^2 gradient
explosion; the budget controller learns only through the numerator (cost)
gradients and the policy softmax.
"""

from __future__ import annotations

import torch

from net.hrot_fsl import HROTFSLResult
from net.jecot_m2 import JECOTM2


OURS_CPM_ABLATIONS = frozenset({"full", "single_budget", "no_egsm"})


def _normalize_cpm_ablation(value: str | None) -> str:
    name = "full" if value is None else str(value).strip().lower().replace("-", "_")
    aliases = {
        "none": "full",
        "default": "full",
        "single": "single_budget",
        "1budget": "single_budget",
    }
    name = aliases.get(name, name)
    if name not in OURS_CPM_ABLATIONS:
        raise ValueError(
            f"Unsupported Ours-CPM ablation: {value!r}. "
            f"Expected one of {sorted(OURS_CPM_ABLATIONS)}"
        )
    return name


def apply_ours_cpm_defaults(kwargs: dict, ablation: str) -> dict:
    """Apply Ours-CPM design contract to HROT/M2 constructor kwargs."""
    kwargs = dict(kwargs)

    kwargs.setdefault("use_cata", False)
    kwargs.setdefault("cata_num_anchors", 8)
    kwargs.setdefault("cata_num_heads", 4)
    kwargs.setdefault("cata_attn_dropout", 0.0)

    if kwargs.get("ecot_rho_bank") is None:
        kwargs["ecot_rho_bank"] = "0.7,0.8,0.9"
    if kwargs.get("ecot_base_rho") is None:
        kwargs["ecot_base_rho"] = 0.8
    if kwargs.get("ecot_transport_mode") is None:
        kwargs["ecot_transport_mode"] = "unbalanced"

    kwargs["ecot_m2_ablate_threshold_mass"] = False
    kwargs["ecot_m2_cost_per_mass_score"] = True
    kwargs["ecot_m2_cost_per_mass_alpha"] = kwargs.pop("cpm_alpha", 1.0)
    kwargs["ecot_m2_cost_per_mass_detach_mass"] = True

    kwargs["ecot_m2_use_aqm"] = False
    kwargs["ecot_m2_tau_aqm"] = 1.0
    kwargs["ecot_m2_use_swts"] = False
    kwargs["ecot_m2_swts_temp"] = 1.0
    kwargs["ecot_enable_ccdm_marginal"] = False
    kwargs["ecot_enable_mea_marginal"] = False
    kwargs["ecot_enable_crs_marginal"] = False

    if kwargs.get("ecot_enable_egsm") is None:
        kwargs["ecot_enable_egsm"] = ablation != "no_egsm"

    if ablation == "single_budget":
        kwargs["ecot_rho_bank"] = "0.8"
    elif ablation == "no_egsm":
        kwargs["ecot_enable_egsm"] = False

    return kwargs


class OursCPM(JECOTM2):
    """Cost-Per-Mass multi-budget model.

    Compared to OursM2:
      - score = -alpha * cost / (mass_detached + eps) instead of -cost
      - rho_bank = [0.7, 0.8, 0.9] instead of [0.8]
      - EpisodeBudgetController has 3 non-degenerate experts to mix
    """

    def __init__(
        self,
        *args,
        cpm_ablation: str = "full",
        cpm_alpha: float = 1.0,
        cpm_rho_bank: str | None = None,
        **kwargs,
    ) -> None:
        if float(cpm_alpha) <= 0.0:
            raise ValueError("cpm_alpha must be positive")
        self.cpm_ablation = _normalize_cpm_ablation(cpm_ablation)
        self.cpm_alpha = float(cpm_alpha)

        if cpm_rho_bank is not None:
            kwargs["ecot_rho_bank"] = str(cpm_rho_bank)

        kwargs["cpm_alpha"] = self.cpm_alpha
        kwargs = apply_ours_cpm_defaults(kwargs, self.cpm_ablation)

        super().__init__(
            *args,
            rho=float(kwargs["ecot_base_rho"]),
            transport_mode=kwargs["ecot_transport_mode"],
            **kwargs,
        )

    def _stack_outputs(
        self, batch_outputs: list[dict[str, torch.Tensor]]
    ) -> HROTFSLResult:
        stacked = super()._stack_outputs(batch_outputs)
        for key in ("ecot_pi_budget", "ecot_budget_logits", "ecot_rho_bank"):
            if key in batch_outputs[0]:
                stacked[key] = torch.stack(
                    [item[key] for item in batch_outputs]
                ).mean(dim=0)
        return stacked


__all__ = ["OursCPM", "apply_ours_cpm_defaults", "OURS_CPM_ABLATIONS"]
