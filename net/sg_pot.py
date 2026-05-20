"""Saliency-Guided Partial Optimal Transport for Few-Shot Classification.

Inherits the full Ours-Final pipeline (OursM2 → JECOTM2 → HROTFSL).
Replaces the ECOT budget-bank transport+scoring with:
  - Saliency-guided non-uniform marginals (cross-conditioned)
  - Adaptive per-pair mass fraction from cost statistics
  - Entropic partial OT with hard mass constraint
  - Dual evidence score: transport cost + residual concentration (Gini)
Keeps backbone encoding, token projection, and episode-level batching.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from net.hrot_fsl import HROTFSLResult
from net.modules.partial_ot import entropic_partial_wasserstein, resolve_partial_transport_mass
from net.modules.fast_partial_ot import fast_partial_wasserstein
from net.ours import OursM2


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------

class SaliencyGate(nn.Module):
    """Cross-conditioned saliency: a token's importance depends on the partner set."""

    def __init__(self, dim: int, hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = hidden_dim or max(dim // 4, 32)
        self.proj = nn.Linear(dim, hidden_dim)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def forward(
        self,
        tokens_self: torch.Tensor,
        tokens_partner: torch.Tensor,
    ) -> torch.Tensor:
        """Return [B, L_self] softmax marginals summing to 1.0."""
        h_self = self.proj(tokens_self)
        h_partner = self.proj(tokens_partner)
        partner_ctx = h_partner.mean(dim=1, keepdim=True).expand_as(h_self)
        combined = torch.cat([h_self, partner_ctx], dim=-1)
        logits = self.gate(combined).squeeze(-1)
        return torch.softmax(logits, dim=-1)


class AdaptiveMassFraction(nn.Module):
    """Predict per-pair mass fraction s from cost-matrix statistics.

    Input features are normalized via LayerNorm before the MLP to prevent
    distribution shift in the cost statistics from destabilizing training.
    """

    _NUM_FEATURES = 6

    def __init__(self, s_min: float = 0.3, s_max: float = 0.8):
        super().__init__()
        self.s_min = s_min
        self.s_max = s_max
        self.feat_norm = nn.LayerNorm(self._NUM_FEATURES)
        self.predictor = nn.Sequential(
            nn.Linear(self._NUM_FEATURES, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
        )
        nn.init.zeros_(self.predictor[-1].weight)
        nn.init.zeros_(self.predictor[-1].bias)

    def forward(self, cost_matrix: torch.Tensor) -> torch.Tensor:
        """cost_matrix: [B, Lq, Ls] → s: [B, 1] in [s_min, s_max]."""
        row_min = cost_matrix.min(dim=-1).values
        feat_mean = row_min.mean(dim=-1)
        feat_std = row_min.std(dim=-1)
        feat_min = row_min.min(dim=-1).values
        feat_max = row_min.max(dim=-1).values
        feat_gap = (feat_max - feat_min) / (feat_mean + 1e-8)
        feat_cv = feat_std / (feat_mean + 1e-8)
        features = torch.stack(
            [feat_mean, feat_std, feat_min, feat_max, feat_gap, feat_cv],
            dim=-1,
        )
        features = self.feat_norm(features)
        raw = self.predictor(features)
        return self.s_min + (self.s_max - self.s_min) * torch.sigmoid(raw)


class DualEvidenceScore(nn.Module):
    """Score = exp(log_scale) * (alpha * cost_score + beta * residual_gini).

    Uses learnable log_scale (initialized to log(10)) instead of a fixed
    multiplier so the model self-calibrates logit magnitude.

    cost_score = -C / M  (average transport cost per unit mass).
    Because the POT plan is solved under ``torch.no_grad()``, M is a
    per-sample constant — no gradient instability from dividing by it.
    """

    _INIT_LOG_SCALE = 2.3  # ~log(10)

    def __init__(self, score_scale: float = 16.0):
        super().__init__()
        self.nominal_score_scale = score_scale
        self.log_scale = nn.Parameter(torch.tensor(self._INIT_LOG_SCALE))
        self.raw_alpha = nn.Parameter(torch.tensor(0.5))
        self.raw_beta = nn.Parameter(torch.tensor(-0.5))

    @property
    def effective_scale(self) -> torch.Tensor:
        return self.log_scale.exp().clamp(max=50.0)

    def forward(
        self,
        plan: torch.Tensor,
        cost_matrix: torch.Tensor,
        a: torch.Tensor,
        s: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """
        plan:        [B, Lq, Ls]  (detached — no gradient through solver)
        cost_matrix: [B, Lq, Ls]  (live — gradient flows to encoder)
        a:           [B, Lq]       (live — gradient flows to saliency gate)
        s:           [B, 1]
        Returns E [B] and diagnostics dict.
        """
        alpha = F.softplus(self.raw_alpha)
        beta = F.softplus(self.raw_beta)
        scale = self.effective_scale

        C = (plan * cost_matrix).sum(dim=(-2, -1))
        M = plan.sum(dim=(-2, -1))
        cost_score = -C / (M + 1e-8)

        q_transported = plan.sum(dim=-1)
        q_residual = (a - q_transported).clamp(min=0)
        residual_sum = q_residual.sum(dim=-1, keepdim=True) + 1e-8
        residual_dist = q_residual / residual_sum

        Lq = plan.shape[1]
        sorted_res, _ = torch.sort(residual_dist, dim=-1)
        idx = torch.arange(1, Lq + 1, device=plan.device, dtype=plan.dtype)
        gini = (
            (2.0 * (idx * sorted_res).sum(dim=-1))
            / (Lq * sorted_res.sum(dim=-1) + 1e-8)
            - (Lq + 1.0) / Lq
        )

        E = scale * (alpha * cost_score + beta * gini)
        diag = {
            "cost_score": cost_score.detach(),
            "residual_gini": gini.detach(),
            "transported_cost": C.detach(),
            "transported_mass": M.detach(),
            "alpha": alpha.detach(),
            "beta": beta.detach(),
            "score_scale": scale.detach(),
        }
        return E, diag


# ---------------------------------------------------------------------------
# Regularization
# ---------------------------------------------------------------------------

def saliency_entropy_reg(
    marginals: torch.Tensor,
    target_ratio: float = 0.7,
    weight: float = 0.05,
) -> torch.Tensor:
    L = marginals.shape[-1]
    max_ent = torch.tensor(float(L), device=marginals.device, dtype=marginals.dtype).log()
    ent = -(marginals * (marginals + 1e-8).log()).sum(dim=-1)
    target = target_ratio * max_ent
    return weight * ((ent - target) ** 2).mean()


# ---------------------------------------------------------------------------
# Main Model
# ---------------------------------------------------------------------------

class SGPOT(OursM2):
    """Saliency-Guided Partial Optimal Transport for Few-Shot Classification.

    Inherits image encoding, token projection, L2 normalization, and the
    episode-level batching loop from OursM2/JECOTM2/HROTFSL.
    Overrides ``_forward_episode`` to replace the ECOT budget-bank path with
    the SG-POT transport-and-score pipeline.
    """

    def __init__(self, *args, **kwargs):
        pot_cfg = _extract_pot_kwargs(kwargs)
        super().__init__(*args, **kwargs)

        self.pot_epsilon = pot_cfg["pot_epsilon"]
        self.pot_iterations = pot_cfg["pot_iterations"]
        self.pot_s_min = pot_cfg["pot_s_min"]
        self.pot_s_max = pot_cfg["pot_s_max"]
        self.entropy_reg_weight = pot_cfg["pot_entropy_reg"]
        self.entropy_target_ratio = pot_cfg["pot_entropy_target"]

        self._ablate_uniform_marginals = pot_cfg["pot_ablate_uniform_marginals"]
        self._ablate_fixed_s = pot_cfg["pot_ablate_fixed_s"]
        self._ablate_cost_only_score = pot_cfg["pot_ablate_cost_only_score"]
        self._ablate_no_entropy_reg = pot_cfg["pot_ablate_no_entropy_reg"]

        dim = self.token_dim
        self.saliency_gate = SaliencyGate(dim)
        self.s_predictor = AdaptiveMassFraction(
            s_min=self.pot_s_min,
            s_max=self.pot_s_max,
        )
        self.scorer = DualEvidenceScore(score_scale=self.score_scale)

        self._sgpot_diag: dict[str, Any] = {}
        self._sgpot_aux_loss = torch.tensor(0.0)

    # ------------------------------------------------------------------
    # Step 5: saliency marginals
    # ------------------------------------------------------------------
    def _build_sgpot_marginals(
        self,
        query_tokens: torch.Tensor,
        support_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._ablate_uniform_marginals:
            B, Lq, _ = query_tokens.shape
            Ls = support_tokens.shape[1]
            a = torch.full((B, Lq), 1.0 / Lq, device=query_tokens.device, dtype=query_tokens.dtype)
            b = torch.full((B, Ls), 1.0 / Ls, device=support_tokens.device, dtype=support_tokens.dtype)
            self._sgpot_aux_loss = torch.tensor(0.0, device=query_tokens.device)
            return a, b

        a = self.saliency_gate(query_tokens, support_tokens)
        b = self.saliency_gate(support_tokens, query_tokens)

        if self._ablate_no_entropy_reg:
            self._sgpot_aux_loss = torch.tensor(0.0, device=query_tokens.device)
        else:
            self._sgpot_aux_loss = (
                saliency_entropy_reg(a, self.entropy_target_ratio, self.entropy_reg_weight)
                + saliency_entropy_reg(b, self.entropy_target_ratio, self.entropy_reg_weight)
            )

        self._sgpot_diag["saliency_q_entropy"] = (
            -(a * (a + 1e-8).log()).sum(-1).mean().item()
        )
        self._sgpot_diag["saliency_s_entropy"] = (
            -(b * (b + 1e-8).log()).sum(-1).mean().item()
        )
        self._sgpot_diag["saliency_q_max"] = a.max(dim=-1).values.mean().item()
        self._sgpot_diag["saliency_q_min"] = a.min(dim=-1).values.mean().item()
        return a, b

    # ------------------------------------------------------------------
    # Step 6: adaptive mass fraction
    # ------------------------------------------------------------------
    def _get_mass_fraction(self, cost_matrix: torch.Tensor) -> torch.Tensor:
        B = cost_matrix.shape[0]
        if self._ablate_fixed_s is not None:
            return torch.full(
                (B, 1),
                self._ablate_fixed_s,
                device=cost_matrix.device,
                dtype=cost_matrix.dtype,
            )
        s = self.s_predictor(cost_matrix)
        self._sgpot_diag["adaptive_s_mean"] = s.mean().item()
        self._sgpot_diag["adaptive_s_std"] = s.std().item()
        return s

    # ------------------------------------------------------------------
    # Step 7: solve POT
    # ------------------------------------------------------------------
    def _solve_pot(
        self,
        cost: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        s: torch.Tensor,
    ) -> torch.Tensor:
        """Fast augmented Sinkhorn partial OT solver.

        s is [B, 1] — fractional mass to transport.  Marginals a, b sum to 1.0,
        so ``transport_mass_ratio=s`` means the solver transports ``s * min(1, 1) = s``.

        The iterative solver produces a deep compute graph (100 projections)
        that is numerically fragile for backprop.  We detach the plan;
        gradients flow through the scorer which re-uses the live cost matrix,
        marginals, and s alongside the fixed plan geometry.
        """
        mass = resolve_partial_transport_mass(
            a, b, transport_mass_ratio=s.squeeze(-1),
        )
        with torch.no_grad():
            plan = fast_partial_wasserstein(
                cost=cost,
                a=a,
                b=b,
                transport_mass=mass,
                reg=self.pot_epsilon,
                max_iter=self.pot_iterations,
            )
        return plan

    # ------------------------------------------------------------------
    # Steps 5-8 combined: per-pair transport & score
    # ------------------------------------------------------------------
    def _sgpot_pair_evidence(
        self,
        query_tokens: torch.Tensor,
        support_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute SG-POT evidence for a batch of (query, support-shot) pairs.

        Args:
            query_tokens:   [B, Lq, C]
            support_tokens: [B, Ls, C]
        Returns:
            E      [B]
            cost   [B, Lq, Ls]
            plan   [B, Lq, Ls]
            s      [B, 1]
        """
        D = torch.cdist(query_tokens, support_tokens).pow(2)
        a, b = self._build_sgpot_marginals(query_tokens, support_tokens)
        s = self._get_mass_fraction(D)
        plan = self._solve_pot(D, a, b, s)
        E, score_diag = self.scorer(plan, D, a, s)

        if self._ablate_cost_only_score:
            alpha = F.softplus(self.scorer.raw_alpha)
            scale = self.scorer.effective_scale
            cost_score = score_diag["cost_score"].to(dtype=E.dtype, device=E.device)
            E = scale * alpha * cost_score

        self._sgpot_diag.update({k: v.mean().item() if torch.is_tensor(v) else v for k, v in score_diag.items()})
        self._sgpot_diag["pot_plan_sparsity"] = (plan < 1e-6).float().mean().item()
        return E, D, plan, s

    # ------------------------------------------------------------------
    # Override: full _forward_episode
    # ------------------------------------------------------------------
    def _forward_episode(
        self,
        query: torch.Tensor,
        support: torch.Tensor,
        *,
        query_targets: torch.Tensor | None = None,
        needs_payload: bool = False,
        return_aux: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        self._sgpot_diag = {}
        way_num, shot_num = support.shape[:2]

        query_euclidean, query_hyperbolic, query_hw = self._encode_images(query)
        support_euclidean, support_hyperbolic, support_hw = self._encode_images(
            support.reshape(way_num * shot_num, *support.shape[-3:])
        )
        if query_hw != support_hw:
            raise ValueError(f"Query/support token grids must match, got {query_hw} vs {support_hw}")

        support_euclidean = support_euclidean.reshape(
            way_num, shot_num,
            support_euclidean.shape[-2],
            support_euclidean.shape[-1],
        )
        flat_support_euclidean = support_euclidean.reshape(
            way_num * shot_num,
            support_euclidean.shape[-2],
            support_euclidean.shape[-1],
        )

        Nq = query.shape[0]
        Lq = query_euclidean.shape[1]
        Ls = flat_support_euclidean.shape[1]

        q_exp = query_euclidean.unsqueeze(1).expand(Nq, way_num * shot_num, Lq, -1)
        s_exp = flat_support_euclidean.unsqueeze(0).expand(Nq, way_num * shot_num, Ls, -1)
        q_flat = q_exp.reshape(Nq * way_num * shot_num, Lq, -1)
        s_flat = s_exp.reshape(Nq * way_num * shot_num, Ls, -1)

        E_flat, D_flat, plan_flat, s_flat_val = self._sgpot_pair_evidence(q_flat, s_flat)
        shot_evidence = E_flat.reshape(Nq, way_num, shot_num)

        shot_logits = shot_evidence
        if shot_num == 1:
            logits = shot_logits.squeeze(-1)
        else:
            weights = torch.softmax(shot_logits, dim=-1)
            logits = torch.logsumexp(shot_logits, dim=-1) - math.log(float(shot_num))

        transport_cost_per_shot = self._sgpot_diag.get("transported_cost", 0.0)
        transport_mass_per_shot = self._sgpot_diag.get("transported_mass", 0.0)

        zero = logits.new_zeros(())
        aux_loss = self._sgpot_aux_loss.to(device=logits.device, dtype=logits.dtype)

        if not needs_payload:
            return logits

        shot_rho = torch.full(
            (Nq, way_num, shot_num),
            self._sgpot_diag.get("adaptive_s_mean", 0.5),
            device=logits.device,
            dtype=logits.dtype,
        )

        cost_5d = D_flat.reshape(Nq, way_num, shot_num, Lq, Ls)
        plan_5d = plan_flat.reshape(Nq, way_num, shot_num, Lq, Ls)

        transport_cost_3d = (plan_5d * cost_5d).sum(dim=(-2, -1))
        transport_mass_3d = plan_5d.sum(dim=(-2, -1))

        outputs: dict[str, torch.Tensor] = {
            "logits": logits,
            "aux_loss": aux_loss,
            "class_scores": logits,
            "total_distance": transport_cost_3d.mean(dim=-1),
            "transport_cost": transport_cost_3d.mean(dim=-1),
            "transported_mass": transport_mass_3d.mean(dim=-1),
            "rho": shot_rho,
            "rho_regularization": zero,
            "rho_rank_loss": zero,
            "cost_margin_aux_loss": zero,
            "hlm_rho_regularization": zero,
            "hlm_eff_loss": zero,
            "hlm_shot_cov_loss": zero,
            "curvature_regularization": zero,
            "curvature": self.curvature.detach().to(dtype=logits.dtype),
            "mass_bonus": logits.new_tensor(0.0),
            "transport_cost_threshold": self.transport_cost_threshold.detach().to(
                device=logits.device, dtype=logits.dtype,
            ),
            "hyperbolic_backend": logits.new_tensor(
                0.0 if self.hyperbolic_backend == "native" else 1.0
            ),
            "ot_backend": logits.new_tensor(2.0),
            "shot_transport_cost": transport_cost_3d,
            "shot_transported_mass": transport_mass_3d,
            "shot_rho": shot_rho,
        }
        if return_aux:
            outputs.update({
                "transport_plan": plan_5d,
                "cost_matrix": cost_5d,
                "query_euclidean_tokens": query_euclidean,
                "support_euclidean_tokens": support_euclidean,
                "query_hyperbolic_tokens": query_hyperbolic,
                "support_hyperbolic_tokens": support_hyperbolic.reshape(
                    way_num, shot_num,
                    support_hyperbolic.shape[-2] // shot_num if support_hyperbolic.dim() == 2 else support_hyperbolic.shape[-2],
                    support_hyperbolic.shape[-1],
                ) if support_hyperbolic.dim() == 3 else support_hyperbolic,
            })

        for key, val in self._sgpot_diag.items():
            if isinstance(val, (int, float)):
                outputs[f"sgpot_{key}"] = logits.new_tensor(val)

        return outputs

    # ------------------------------------------------------------------
    # Stack outputs across batch episodes
    # ------------------------------------------------------------------
    def _stack_outputs(self, batch_outputs: list[dict[str, torch.Tensor]]) -> HROTFSLResult:
        stacked = super()._stack_outputs(batch_outputs)
        sgpot_keys = [k for k in batch_outputs[0] if k.startswith("sgpot_")]
        for key in sgpot_keys:
            vals = [item[key] for item in batch_outputs if key in item]
            if vals:
                stacked[key] = torch.stack(vals).mean()
        return stacked

    # ------------------------------------------------------------------
    # Diagnostics health warnings (call externally for logging)
    # ------------------------------------------------------------------
    def get_sgpot_health_warnings(self) -> list[str]:
        warnings = []
        diag = self._sgpot_diag
        if not diag:
            return warnings

        Lq = 25
        max_ent = math.log(Lq) if Lq > 1 else 1.0

        q_ent = diag.get("saliency_q_entropy", None)
        if q_ent is not None:
            if q_ent < 0.3 * max_ent:
                warnings.append("Saliency collapsing to peaked")
            elif q_ent > 0.95 * max_ent:
                warnings.append("Saliency collapsing to uniform")

        s_mean = diag.get("adaptive_s_mean", None)
        if s_mean is not None:
            if abs(s_mean - self.pot_s_max) < 0.02:
                warnings.append(f"s saturating high ({s_mean:.3f} ≈ s_max={self.pot_s_max})")
            elif abs(s_mean - self.pot_s_min) < 0.02:
                warnings.append(f"s saturating low ({s_mean:.3f} ≈ s_min={self.pot_s_min})")

        gini = diag.get("residual_gini", None)
        if gini is not None and gini < 0.1:
            warnings.append("No signal/noise separation detected (gini < 0.1)")

        beta = diag.get("beta", None)
        if beta is not None and beta < 0.01:
            warnings.append("Model ignoring residual signal (beta → 0)")

        return warnings


# ---------------------------------------------------------------------------
# Config extraction utility
# ---------------------------------------------------------------------------

_POT_DEFAULTS = {
    "pot_epsilon": 0.05,
    "pot_iterations": 100,
    "pot_s_min": 0.3,
    "pot_s_max": 0.8,
    "pot_entropy_reg": 0.05,
    "pot_entropy_target": 0.7,
    "pot_ablate_uniform_marginals": False,
    "pot_ablate_fixed_s": None,
    "pot_ablate_cost_only_score": False,
    "pot_ablate_no_entropy_reg": False,
}


def _extract_pot_kwargs(kwargs: dict) -> dict:
    """Pop SG-POT-specific keys from kwargs before passing to parent __init__."""
    cfg = {}
    for key, default in _POT_DEFAULTS.items():
        cfg[key] = kwargs.pop(key, default)
    return cfg


__all__ = [
    "SGPOT",
    "SaliencyGate",
    "AdaptiveMassFraction",
    "DualEvidenceScore",
    "saliency_entropy_reg",
]
