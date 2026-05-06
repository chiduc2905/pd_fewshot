"""Paper-facing EC-MROT model route.

EC-MROT expands to Episode-Conditioned Mass-Response Optimal Transport.
The default implementation is an exact, separate clone of the current
``HROTFSL(variant="J_ECOT")`` behavior:

* no learned token mass,
* no hyperbolic-core claim,
* no threshold-offset module unless explicitly enabled,
* the fixed response grid [0.45, 0.60, 0.75, 0.80, 0.90],
* base budget 0.80,
* base-budget homotopy initialized near the base model.

The paper-facing names below intentionally describe the same mechanics in a
more theory-aligned way. The rho grid is not the novelty; it is a discrete
quadrature rule over a mass-response path. The controller estimates an
episode-conditioned budget measure over that path. The base anchor is a
homotopy from stable base-budget UOT to adaptive response integration. The
shot-level log-mean-exp pooling is entropic support-risk aggregation.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from net.hrot_fsl import HROTFSL, HROTFSLResult, _inverse_softplus


DEFAULT_MASS_RESPONSE_GRID = (0.45, 0.60, 0.75, 0.80, 0.90)
DEFAULT_BASE_BUDGET = 0.80
GRID_BOUNDARY_EPS = 1e-4


def _as_bool(value: bool | str | None, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def _format_grid(values: tuple[float, ...]) -> str:
    return ",".join(f"{value:.8g}" for value in values)


class ECMROT(HROTFSL):
    """Episode-Conditioned Mass-Response OT.

    This class deliberately subclasses the current J_ECOT implementation rather
    than editing it. With all EC-MROT flags at their defaults, state-dict keys,
    logits, legacy ``ecot_*`` diagnostics, and auxiliary losses match the
    existing J_ECOT path. EC-MROT adds paper-facing aliases and optional
    theory-motivated regularizers behind explicit flags.
    """

    def __init__(
        self,
        *args: Any,
        variant: str = "J_ECOT",
        mass_response_grid: str | list[float] | tuple[float, ...] | None = None,
        base_budget: float = DEFAULT_BASE_BUDGET,
        response_mode: str = "discrete_quadrature",
        learn_response_grid: bool | str = False,
        budget_prior: str = "uniform",
        budget_kl_reg: float = 0.0,
        budget_entropy_reg: float = 0.0,
        budget_tau: float | None = None,
        homotopy_schedule: str = "none",
        grid_spacing_reg: float = 0.0,
        grid_min_spacing: float = 0.02,
        **kwargs: Any,
    ) -> None:
        response_mode = str(response_mode).strip().lower().replace("-", "_")
        if response_mode not in {"discrete_quadrature"}:
            raise ValueError(f"Unsupported EC-MROT response_mode: {response_mode}")

        budget_prior = str(budget_prior).strip().lower().replace("-", "_")
        if budget_prior not in {"uniform", "base_anchored"}:
            raise ValueError(f"Unsupported EC-MROT budget_prior: {budget_prior}")

        homotopy_schedule = str(homotopy_schedule).strip().lower().replace("-", "_")
        if homotopy_schedule not in {"none", "linear", "cosine"}:
            raise ValueError(f"Unsupported EC-MROT homotopy_schedule: {homotopy_schedule}")

        if float(budget_kl_reg) < 0.0:
            raise ValueError("budget_kl_reg must be non-negative")
        if float(budget_entropy_reg) < 0.0:
            raise ValueError("budget_entropy_reg must be non-negative")
        if float(grid_spacing_reg) < 0.0:
            raise ValueError("grid_spacing_reg must be non-negative")
        if float(grid_min_spacing) < 0.0:
            raise ValueError("grid_min_spacing must be non-negative")

        requested_variant = str(variant).strip().upper().replace("-", "_")
        if requested_variant not in {"J_ECOT", "JECOT", "ECOT"}:
            raise ValueError("ECMROT is the paper-facing clone of J_ECOT; use hrot_fsl for other HROT variants")

        if mass_response_grid is not None and kwargs.get("ecot_rho_bank") is None:
            kwargs["ecot_rho_bank"] = mass_response_grid
        if kwargs.get("ecot_rho_bank") is None:
            kwargs["ecot_rho_bank"] = _format_grid(DEFAULT_MASS_RESPONSE_GRID)
        if kwargs.get("ecot_base_rho") is None:
            kwargs["ecot_base_rho"] = float(base_budget)
        if budget_tau is not None:
            kwargs["ecot_budget_tau"] = float(budget_tau)

        super().__init__(*args, variant="J_ECOT", **kwargs)

        self.ec_mrot_response_mode = response_mode
        self.ec_mrot_learn_response_grid = _as_bool(learn_response_grid, default=False)
        self.ec_mrot_budget_prior = budget_prior
        self.ec_mrot_budget_kl_reg = float(budget_kl_reg)
        self.ec_mrot_budget_entropy_reg = float(budget_entropy_reg)
        self.ec_mrot_homotopy_schedule = homotopy_schedule
        self.ec_mrot_homotopy_progress = 1.0
        self.ec_mrot_grid_spacing_reg = float(grid_spacing_reg)
        self.ec_mrot_grid_min_spacing = float(grid_min_spacing)
        self.ec_mrot_display_name = "EC-MROT"
        self.ec_mrot_paper_name = "Episode-Conditioned Mass-Response Optimal Transport"

        self.ec_mrot_response_grid_left_raw_gaps: nn.Parameter | None = None
        self.ec_mrot_response_grid_right_raw_gaps: nn.Parameter | None = None
        if self.ec_mrot_learn_response_grid:
            self._init_learnable_response_grid()

    def _init_learnable_response_grid(self) -> None:
        values = [float(value) for value in self.ecot_rho_bank]
        base_idx = int(self.ecot_base_idx)
        base_budget = float(self.ecot_base_rho)
        lower = GRID_BOUNDARY_EPS
        upper = 1.0 - GRID_BOUNDARY_EPS
        if not lower < base_budget < upper:
            raise ValueError("EC-MROT learnable response grid requires base_budget in (0, 1)")

        before = values[:base_idx]
        after = values[base_idx + 1 :]

        left_gaps: list[float] = []
        cursor = lower
        for value in before:
            left_gaps.append(max(float(value) - cursor, 1e-5))
            cursor = float(value)
        left_gaps.append(max(base_budget - cursor, 1e-5))

        right_gaps: list[float] = []
        cursor = base_budget
        for value in after:
            right_gaps.append(max(float(value) - cursor, 1e-5))
            cursor = float(value)
        right_gaps.append(max(upper - cursor, 1e-5))

        self.ec_mrot_response_grid_left_raw_gaps = nn.Parameter(
            torch.tensor([_inverse_softplus(gap) for gap in left_gaps], dtype=torch.float32)
        )
        self.ec_mrot_response_grid_right_raw_gaps = nn.Parameter(
            torch.tensor([_inverse_softplus(gap) for gap in right_gaps], dtype=torch.float32)
        )

    def set_homotopy_progress(self, progress: float) -> None:
        """Set training progress in [0, 1] for scheduled homotopy strength."""

        self.ec_mrot_homotopy_progress = float(min(max(progress, 0.0), 1.0))

    def _homotopy_schedule_scale(self, reference: torch.Tensor) -> torch.Tensor:
        progress = reference.new_tensor(self.ec_mrot_homotopy_progress)
        if self.ec_mrot_homotopy_schedule == "linear":
            return progress.clamp(0.0, 1.0)
        if self.ec_mrot_homotopy_schedule == "cosine":
            progress = progress.clamp(0.0, 1.0)
            return 0.5 - 0.5 * torch.cos(progress * math.pi)
        return reference.new_tensor(1.0)

    @property
    def ecot_lambda(self) -> torch.Tensor:
        homotopy_lambda = super().ecot_lambda
        return homotopy_lambda * self._homotopy_schedule_scale(homotopy_lambda)

    def _positive_normalized_gaps(
        self,
        raw_gaps: torch.Tensor,
        span: torch.Tensor,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        gaps = F.softplus(raw_gaps.to(device=device, dtype=dtype)) + self.eps
        return gaps / gaps.sum().clamp_min(self.eps) * span.clamp_min(self.eps)

    def _current_response_grid(self, reference: torch.Tensor) -> torch.Tensor:
        if not self.ec_mrot_learn_response_grid:
            if self.ecot_rho_bank_tensor is None:
                raise RuntimeError("EC-MROT requires an ecot_rho_bank_tensor")
            return self.ecot_rho_bank_tensor.to(device=reference.device, dtype=reference.dtype)

        if self.ec_mrot_response_grid_left_raw_gaps is None or self.ec_mrot_response_grid_right_raw_gaps is None:
            raise RuntimeError("EC-MROT learnable response grid was not initialized")

        lower = reference.new_tensor(GRID_BOUNDARY_EPS)
        upper = reference.new_tensor(1.0 - GRID_BOUNDARY_EPS)
        base = reference.new_tensor(float(self.ecot_base_rho))
        base_idx = int(self.ecot_base_idx)
        budget_count = len(self.ecot_rho_bank)

        parts: list[torch.Tensor] = []
        if base_idx > 0:
            left_gaps = self._positive_normalized_gaps(
                self.ec_mrot_response_grid_left_raw_gaps,
                base - lower,
                dtype=reference.dtype,
                device=reference.device,
            )
            parts.append(lower + torch.cumsum(left_gaps[:-1], dim=0))

        parts.append(base.reshape(1))

        after_count = budget_count - base_idx - 1
        if after_count > 0:
            right_gaps = self._positive_normalized_gaps(
                self.ec_mrot_response_grid_right_raw_gaps,
                upper - base,
                dtype=reference.dtype,
                device=reference.device,
            )
            parts.append(base + torch.cumsum(right_gaps[:-1], dim=0))

        return torch.cat(parts, dim=0)

    def _response_grid_spacing_loss(self, response_grid: torch.Tensor) -> torch.Tensor:
        if response_grid.numel() <= 1:
            return response_grid.new_zeros(())
        spacing = response_grid[1:] - response_grid[:-1]
        min_spacing = response_grid.new_tensor(self.ec_mrot_grid_min_spacing)
        return F.relu(min_spacing - spacing).pow(2).mean()

    def _budget_prior_weights(self, response_grid: torch.Tensor) -> torch.Tensor:
        if self.ec_mrot_budget_prior == "uniform":
            return torch.full_like(response_grid, 1.0 / float(response_grid.numel()))

        base = response_grid.new_tensor(float(self.ecot_base_rho))
        # A smooth base-anchored prior stabilizes scarce-label episodes without
        # hard-selecting the base budget. The controller can still move mass
        # across the response path when the KL weight is nonzero.
        logits = -((response_grid - base) / response_grid.new_tensor(0.10)).pow(2)
        return torch.softmax(logits, dim=-1)

    def _forward_ecot_budget_bank(
        self,
        flat_cost: torch.Tensor,
        *,
        way_num: int,
        shot_num: int,
    ) -> dict[str, torch.Tensor]:
        if self.episode_controller is None:
            raise RuntimeError("EC-MROT requires an episode_controller")
        if flat_cost.dim() != 4:
            raise ValueError(f"flat_cost must have shape (Nq, Way*Shot, Lq, Ls), got {tuple(flat_cost.shape)}")

        num_query, num_pairs, query_len, support_len = flat_cost.shape
        budget_count = len(self.ecot_rho_bank)
        if num_pairs != way_num * shot_num:
            raise ValueError(f"flat_cost pair dimension {num_pairs} does not match Way*Shot={way_num * shot_num}")

        # Discrete quadrature over the mass-response path:
        # S_mix(q,c,k | E) = sum_b pi_b(E) * s_{rho_b}(q,c,k),
        # approximating Integral s_rho(q,c,k) d pi_E(rho).
        response_grid = self._current_response_grid(flat_cost)
        cost_bank = flat_cost.unsqueeze(2).expand(num_query, num_pairs, budget_count, query_len, support_len)
        cost_bank = cost_bank.reshape(num_query, num_pairs * budget_count, query_len, support_len)
        response_grid_flat = response_grid.view(1, 1, budget_count).expand(num_query, num_pairs, budget_count)
        response_grid_flat = response_grid_flat.reshape(num_query, num_pairs * budget_count)

        plan_bank, cost_out, mass_out = self._transport_match(cost_bank, response_grid_flat)
        plan_bank = plan_bank.reshape(num_query, way_num, shot_num, budget_count, query_len, support_len)
        shot_cost_bank = cost_out.reshape(num_query, way_num, shot_num, budget_count)
        shot_mass_bank = mass_out.reshape(num_query, way_num, shot_num, budget_count)

        threshold = self.transport_cost_threshold.to(device=flat_cost.device, dtype=flat_cost.dtype)
        diagnostics = self._compute_ecot_diagnostics(
            self.score_scale * (threshold * shot_mass_bank - shot_cost_bank),
            shot_cost_bank,
            shot_mass_bank,
        )
        controller_outputs = self.episode_controller(diagnostics)
        budget_logits = controller_outputs["budget_logits"]
        if self.ecot_uniform_budget_policy:
            budget_logits = torch.zeros_like(budget_logits)
            budget_weights = torch.full_like(budget_logits, 1.0 / float(budget_count))
        else:
            budget_weights = torch.softmax(budget_logits / float(self.ecot_budget_tau), dim=-1)

        raw_delta_threshold = controller_outputs.get("raw_delta_threshold")
        if raw_delta_threshold is not None:
            delta = raw_delta_threshold.to(device=flat_cost.device, dtype=flat_cost.dtype)
            threshold = (threshold * torch.exp(0.25 * torch.tanh(delta))).clamp_min(self.eps)

        # Each grid node is a deterministic mass-budget expert. The
        # episode-conditioned budget measure integrates those expert scores.
        budget_scores = self.score_scale * (threshold * shot_mass_bank - shot_cost_bank)
        base_budget_score = budget_scores[..., self.ecot_base_idx]
        budget_view = budget_weights.view(1, 1, 1, budget_count)
        mass_response_score = (budget_view * budget_scores).sum(dim=-1)
        homotopy_lambda = self.ecot_lambda.to(device=flat_cost.device, dtype=flat_cost.dtype)
        shot_cost_diag = (budget_view * shot_cost_bank).sum(dim=-1)
        shot_mass_diag = (budget_view * shot_mass_bank).sum(dim=-1)

        shot_logits = base_budget_score + homotopy_lambda * (mass_response_score - base_budget_score)
        (
            logits,
            transport_cost,
            transport_mass,
            shot_pool_weights,
            support_risk_temperature,
        ) = self._pool_ecot_shot_scores(
            shot_logits,
            shot_cost_diag,
            shot_mass_diag,
            controller_outputs.get("raw_tau_shot"),
        )
        identity_loss = (shot_logits - base_budget_score).pow(2).mean()

        plan_diag = (budget_view.unsqueeze(-1).unsqueeze(-1) * plan_bank).sum(dim=3)
        shot_rho = flat_cost.new_full((num_query, way_num, shot_num), self.ecot_base_rho)
        policy_entropy = -(
            budget_weights.clamp_min(self.eps) * budget_weights.clamp_min(self.eps).log()
        ).sum()

        budget_prior = self._budget_prior_weights(response_grid)
        budget_kl = (
            budget_weights.clamp_min(self.eps)
            * (budget_weights.clamp_min(self.eps).log() - budget_prior.clamp_min(self.eps).log())
        ).sum()
        budget_kl_loss = (
            self.ec_mrot_budget_kl_reg * budget_kl
            if self.ec_mrot_budget_kl_reg > 0.0
            else budget_kl.new_zeros(())
        )
        budget_entropy_loss = (
            -self.ec_mrot_budget_entropy_reg * policy_entropy
            if self.ec_mrot_budget_entropy_reg > 0.0
            else policy_entropy.new_zeros(())
        )
        spacing_loss = self._response_grid_spacing_loss(response_grid)
        response_grid_spacing_loss = (
            self.ec_mrot_grid_spacing_reg * spacing_loss
            if self.ec_mrot_learn_response_grid and self.ec_mrot_grid_spacing_reg > 0.0
            else spacing_loss.new_zeros(())
        )

        ecot_aux_loss = (
            self.ecot_identity_reg * identity_loss
            + budget_kl_loss
            + budget_entropy_loss
            + response_grid_spacing_loss
        )

        payload: dict[str, torch.Tensor] = {
            "logits": logits,
            "transport_cost": transport_cost,
            "transported_mass": transport_mass,
            "rho": shot_rho,
            "shot_rho": shot_rho,
            "shot_transport_cost": shot_cost_diag,
            "shot_transported_mass": shot_mass_diag,
            "shot_logits": shot_logits,
            "shot_pool_weights": shot_pool_weights,
            "transport_plan": plan_diag,
            # Legacy ECOT payload retained for existing diagnostics.
            "ecot_pi_budget": budget_weights,
            "ecot_budget_logits": budget_logits,
            "ecot_lambda": homotopy_lambda,
            "ecot_base_idx": flat_cost.new_tensor(self.ecot_base_idx, dtype=torch.long),
            "ecot_rho_bank": response_grid,
            "ecot_base_score": base_budget_score,
            "ecot_score": mass_response_score,
            "ecot_budget_scores": budget_scores,
            "ecot_shot_transport_cost_bank": shot_cost_bank,
            "ecot_shot_transported_mass_bank": shot_mass_bank,
            "ecot_diagnostics": diagnostics,
            "ecot_identity_loss": identity_loss,
            "ecot_policy_entropy": policy_entropy,
            "ecot_policy_entropy_loss": -self.ecot_policy_entropy_reg * policy_entropy,
            "ecot_aux_loss": ecot_aux_loss,
            # EC-MROT paper-facing aliases.
            "ec_mrot_budget_weights": budget_weights,
            "ec_mrot_episode_budget_measure": budget_weights,
            "ec_mrot_budget_logits": budget_logits,
            "ec_mrot_homotopy_lambda": homotopy_lambda,
            "ec_mrot_base_budget_score": base_budget_score,
            "ec_mrot_mass_response_score": mass_response_score,
            "ec_mrot_response_grid": response_grid,
            "ec_mrot_mass_response_grid": response_grid,
            "ec_mrot_budget_scores": budget_scores,
            "ec_mrot_budget_prior_weights": budget_prior,
            "ec_mrot_budget_kl": budget_kl,
            "ec_mrot_budget_kl_loss": budget_kl_loss,
            "ec_mrot_budget_entropy": policy_entropy,
            "ec_mrot_budget_entropy_loss": budget_entropy_loss,
            "ec_mrot_response_grid_spacing_loss": response_grid_spacing_loss,
            "ec_mrot_response_grid_raw_spacing_loss": spacing_loss,
            "ec_mrot_aux_loss": ecot_aux_loss,
        }
        if support_risk_temperature is not None:
            payload["ecot_tau_shot"] = support_risk_temperature
            payload["ec_mrot_support_risk_temperature"] = support_risk_temperature
        if raw_delta_threshold is not None:
            payload["ecot_threshold"] = threshold
            payload["ecot_raw_delta_threshold"] = raw_delta_threshold.to(
                device=flat_cost.device,
                dtype=flat_cost.dtype,
            )
            payload["ec_mrot_threshold"] = threshold
        return payload

    @staticmethod
    def _stack_outputs(batch_outputs: list[dict[str, torch.Tensor]]) -> HROTFSLResult:
        stacked = HROTFSL._stack_outputs(batch_outputs)

        cat_keys = (
            "ec_mrot_base_budget_score",
            "ec_mrot_mass_response_score",
            "ec_mrot_budget_scores",
        )
        for key in cat_keys:
            if key in batch_outputs[0]:
                stacked[key] = torch.cat([item[key] for item in batch_outputs], dim=0)

        vector_mean_keys = (
            "ec_mrot_budget_weights",
            "ec_mrot_episode_budget_measure",
            "ec_mrot_budget_logits",
            "ec_mrot_response_grid",
            "ec_mrot_mass_response_grid",
            "ec_mrot_budget_prior_weights",
        )
        for key in vector_mean_keys:
            if key in batch_outputs[0]:
                stacked[key] = torch.stack([item[key] for item in batch_outputs]).mean(dim=0)

        scalar_mean_keys = (
            "ec_mrot_homotopy_lambda",
            "ec_mrot_support_risk_temperature",
            "ec_mrot_budget_kl",
            "ec_mrot_budget_kl_loss",
            "ec_mrot_budget_entropy",
            "ec_mrot_budget_entropy_loss",
            "ec_mrot_response_grid_spacing_loss",
            "ec_mrot_response_grid_raw_spacing_loss",
            "ec_mrot_aux_loss",
            "ec_mrot_threshold",
        )
        for key in scalar_mean_keys:
            if key in batch_outputs[0]:
                stacked[key] = torch.stack([item[key] for item in batch_outputs]).mean()

        return HROTFSLResult(stacked)

