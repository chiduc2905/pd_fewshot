"""Standalone Ours model and contribution ablations.

The full design is intentionally treated as one coherent model:
J-ECOT-M2/SB-ECOT with UOT fixed to the paper-facing defaults.  Ours
``full`` enables Episode-Gated Shrinkage Marginals (EGSM) and turns off
MEA/CCDM/CRS for the ECOT token priors.
The ``ours_final`` registry entry reuses this class with factory-level defaults
for the final paper setup: EGSM off and threshold-mass scoring on.
Contribution ablations swap only high-level design choices while leaving the
full model path untouched.  The ``gap`` control uses global-average-pooled
tokens for the cost while keeping the same EGSM + UOT stack as ``full``.
"""

from __future__ import annotations

import csv
import math
import os

import torch
import torch.nn.functional as F

from net.fewshot_common import feature_map_to_tokens
from net.hrot_fsl import HROTFSLResult
from net.hyperbolic.poincare_ops import safe_project_to_ball
from net.jecot_m2 import JECOTM2


OURS_ABLATIONS = frozenset({"full", "full_ot", "no_egsm", "gap"})
TOKEN_G_KINDS = frozenset({"none", "episode_mean_dist", "token_norm_pre_l2"})
DMUOT_MARGINAL_KINDS = frozenset({"uniform", "discriminative"})


def normalize_ours_ablation(value: str | None) -> str:
    name = "full" if value is None else str(value).strip().lower().replace("-", "_")
    aliases = {
        "none": "full",
        "ours": "full",
        "uot": "full",
        "balanced_ot": "full_ot",
        "balanced": "full_ot",
        "ot": "full_ot",
        "uniform_evidence": "no_egsm",
        "uniform": "no_egsm",
        "no_adaptive_evidence": "no_egsm",
        "no_evidence": "no_egsm",
        "proto": "gap",
        "prototype": "gap",
        "global_prototype": "gap",
        "gap_proto": "gap",
    }
    name = aliases.get(name, name)
    if name not in OURS_ABLATIONS:
        raise ValueError(f"Unsupported Ours ablation: {value}. Expected one of {sorted(OURS_ABLATIONS)}")
    return name


def apply_ours_design_defaults(kwargs: dict, ablation: str) -> dict:
    """Apply the standalone Ours design contract to HROT/M2 constructor kwargs."""
    kwargs = dict(kwargs)

    # Legacy Ours: local descriptors + one UOT budget, with the M2 mass score removed.
    # The ours_final registry entry passes ecot_m2_ablate_threshold_mass=False to
    # keep the threshold-mass score on.
    kwargs.setdefault("use_cata", False)
    kwargs.setdefault("cata_num_anchors", 8)
    kwargs.setdefault("cata_num_heads", 4)
    kwargs.setdefault("cata_attn_dropout", 0.0)
    kwargs.setdefault("ecot_rho_bank", "0.8")
    kwargs.setdefault("ecot_base_rho", 0.8)
    kwargs.setdefault("ecot_transport_mode", "unbalanced")
    # Default Ours score is -cost only. If the caller already enabled cost/mass
    # (CLI / run_all_experiments --m2_cost_per_mass), do not overwrite it here —
    # otherwise cost/mass appears to have no effect vs baseline.
    if kwargs.get("ecot_m2_cost_per_mass_score"):
        kwargs["ecot_m2_ablate_threshold_mass"] = False
        kwargs.setdefault("ecot_m2_cost_per_mass_detach_mass", True)
    elif kwargs.get("ecot_m2_ablate_threshold_mass") is False:
        kwargs["ecot_m2_cost_per_mass_score"] = False
    else:
        kwargs["ecot_m2_ablate_threshold_mass"] = True
        kwargs["ecot_m2_cost_per_mass_score"] = False
        kwargs["ecot_m2_cost_per_mass_detach_mass"] = False
    kwargs["ecot_m2_use_aqm"] = False
    kwargs["ecot_m2_tau_aqm"] = 1.0
    kwargs["ecot_m2_use_swts"] = False
    kwargs["ecot_m2_swts_temp"] = 1.0
    kwargs["ecot_enable_ccdm_marginal"] = False
    kwargs["ecot_enable_mea_marginal"] = False
    kwargs["ecot_enable_crs_marginal"] = False
    if ablation in {"full", "full_ot", "gap"}:
        kwargs.setdefault("ecot_enable_egsm", True)

    if ablation == "full_ot":
        kwargs["ecot_rho_bank"] = "1.0"
        kwargs["ecot_base_rho"] = 1.0
        kwargs["ecot_transport_mode"] = "balanced"
    elif ablation == "no_egsm":
        # Full Ours without episode-gated shrinkage marginals.
        kwargs["ecot_m2_use_aqm"] = False
        kwargs["ecot_m2_use_swts"] = False
        kwargs["ecot_enable_egsm"] = False
        kwargs["ecot_enable_ccdm_marginal"] = False
        kwargs["ecot_enable_mea_marginal"] = False
        kwargs["ecot_enable_crs_marginal"] = False
    return kwargs


def _bool_config(value: bool | str) -> bool:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True
        if normalized in {"false", "0", "no", "n", "off"}:
            return False
        raise ValueError(f"Invalid boolean value: {value}")
    return bool(value)


def _resolve_dm_debug(value: bool | str, use_differential_mode: bool) -> bool:
    if isinstance(value, str) and value.strip().lower() == "auto":
        return bool(use_differential_mode)
    return _bool_config(value) and bool(use_differential_mode)


def _normalize_token_g_kind(value: str | None) -> str:
    name = "none" if value is None else str(value).strip().lower().replace("-", "_")
    aliases = {
        "off": "none",
        "false": "none",
        "mean_dist": "episode_mean_dist",
        "episode_mean_distance": "episode_mean_dist",
        "prenorm": "token_norm_pre_l2",
        "pre_l2": "token_norm_pre_l2",
        "token_norm": "token_norm_pre_l2",
    }
    name = aliases.get(name, name)
    if name not in TOKEN_G_KINDS:
        raise ValueError(f"Unsupported token_g_kind: {value}. Expected one of {sorted(TOKEN_G_KINDS)}")
    return name


def _normalize_dmuot_marginal_kind(value: str | None) -> str:
    name = "uniform" if value is None else str(value).strip().lower().replace("-", "_")
    aliases = {
        "none": "uniform",
        "off": "uniform",
        "disc": "discriminative",
        "token_g": "discriminative",
    }
    name = aliases.get(name, name)
    if name not in DMUOT_MARGINAL_KINDS:
        raise ValueError(
            f"Unsupported marginal_kind: {value}. Expected one of {sorted(DMUOT_MARGINAL_KINDS)}"
        )
    return name


def _infer_token_hw(num_tokens: int) -> tuple[int, int]:
    if num_tokens <= 0:
        raise ValueError("num_tokens must be positive")
    root = int(math.sqrt(num_tokens))
    best = (1, num_tokens)
    best_gap = num_tokens - 1
    for height in range(1, root + 1):
        if num_tokens % height != 0:
            continue
        width = num_tokens // height
        gap = abs(width - height)
        if gap < best_gap:
            best = (height, width)
            best_gap = gap
    return best


class OursM2(JECOTM2):
    """Paper-facing Ours entrypoint plus coarse contribution ablations."""

    def __init__(
        self,
        *args,
        ours_ablation: str = "full",
        use_differential_mode: bool = False,
        dm_alpha: float = 0.0,
        dm_debug: bool | str = "auto",
        dm_debug_dir: str = "results/dmt_debug",
        dm_debug_max_episodes: int = 5,
        token_g_kind: str = "none",
        lambda_cost: float = 0.0,
        marginal_kind: str = "uniform",
        tau_marg: float = 1.0,
        **kwargs,
    ) -> None:
        if not 0.0 <= float(dm_alpha) <= 1.0:
            raise ValueError("dm_alpha must be in [0, 1]")
        if int(dm_debug_max_episodes) < 0:
            raise ValueError("dm_debug_max_episodes must be non-negative")
        self.token_g_kind = _normalize_token_g_kind(token_g_kind)
        self.lambda_cost = float(lambda_cost)
        self.marginal_kind = _normalize_dmuot_marginal_kind(marginal_kind)
        self.tau_marg = float(tau_marg)
        if self.lambda_cost < 0.0:
            raise ValueError("lambda_cost must be non-negative")
        if self.lambda_cost > 0.0 and self.token_g_kind == "none":
            raise ValueError("lambda_cost > 0 requires token_g_kind != 'none'")
        if self.marginal_kind == "discriminative" and self.token_g_kind == "none":
            raise ValueError("marginal_kind='discriminative' requires token_g_kind != 'none'")
        if not (self.tau_marg > 0.0 or math.isinf(self.tau_marg)):
            raise ValueError("tau_marg must be positive or inf")
        self.ours_ablation = normalize_ours_ablation(ours_ablation)
        self.use_differential_mode = _bool_config(use_differential_mode)
        self.dm_alpha = float(dm_alpha)
        self.dm_debug = _resolve_dm_debug(dm_debug, self.use_differential_mode)
        self.dm_debug_dir = str(dm_debug_dir)
        self.dm_debug_max_episodes = int(dm_debug_max_episodes)
        self._dm_debug_count = 0
        self._last_dm_diagnostics: dict[str, torch.Tensor] | None = None
        self._token_g_prenorm_cache: list[torch.Tensor] | None = None
        kwargs = apply_ours_design_defaults(kwargs, self.ours_ablation)
        super().__init__(
            *args,
            rho=float(kwargs["ecot_base_rho"]),
            transport_mode=kwargs["ecot_transport_mode"],
            **kwargs,
        )
        self.register_buffer("dm_global_template", torch.empty(0), persistent=False)
        self.register_buffer("dm_template_count", torch.tensor(0.0), persistent=False)

    @property
    def uses_ours_gap_control(self) -> bool:
        return self.ours_ablation == "gap"

    def _encode_images(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
        if self.token_g_kind != "token_norm_pre_l2" and not self.uses_ours_gap_control:
            return super()._encode_images(images)

        feature_map = self.encode(images)
        spatial_hw = (feature_map.shape[-2], feature_map.shape[-1])
        if self.uses_ours_gap_control:
            tokens = F.adaptive_avg_pool2d(feature_map, output_size=1).flatten(1).unsqueeze(1)
            spatial_hw = (1, 1)
        else:
            tokens = feature_map_to_tokens(feature_map)
        projected = self._project_backbone_tokens(tokens)
        if not self.uses_ours_gap_control:
            projected = self._apply_cata(projected)
        self._record_token_g_prenorm(projected)
        euclidean_tokens, hyperbolic_tokens = self._euclidean_and_hyperbolic_from_projected(projected)
        return euclidean_tokens, hyperbolic_tokens, spatial_hw

    def _project_tokens_from_images(self, images: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        if self.token_g_kind != "token_norm_pre_l2":
            return super()._project_tokens_from_images(images)
        feature_map = self.encode(images)
        spatial_hw = (feature_map.shape[-2], feature_map.shape[-1])
        tokens = feature_map_to_tokens(feature_map)
        projected = self._project_backbone_tokens(tokens)
        projected = self._apply_cata(projected)
        self._record_token_g_prenorm(projected)
        return projected, spatial_hw

    def _encode_images_with_amp_norms(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, int]]:
        if self.token_g_kind != "token_norm_pre_l2":
            return super()._encode_images_with_amp_norms(images)
        feature_map = self.encode(images)
        spatial_hw = (feature_map.shape[-2], feature_map.shape[-1])
        tokens = feature_map_to_tokens(feature_map)
        projected, pre_proj_norms = self._project_backbone_tokens_with_prenorm(tokens)
        if self.use_cata:
            projected = self._apply_cata(projected)
            pre_proj_norms = projected.norm(p=2, dim=-1).clamp(min=1e-6)
        self._record_token_g_prenorm(projected)
        euclidean_tokens = (
            F.normalize(projected, p=2, dim=-1, eps=self.eps)
            if self.normalize_euclidean_tokens
            else projected
        )
        ball = self._build_ball(projected)
        hyperbolic_tokens = safe_project_to_ball(projected * self.projection_scale, ball)
        return euclidean_tokens, hyperbolic_tokens, pre_proj_norms, spatial_hw

    def _record_token_g_prenorm(self, projected: torch.Tensor) -> None:
        cache = getattr(self, "_token_g_prenorm_cache", None)
        if cache is not None:
            cache.append(projected.norm(p=2, dim=-1).clamp_min(float(self.eps)))

    @staticmethod
    def _minmax_normalize_per_image(values: torch.Tensor, eps: float) -> torch.Tensor:
        min_value = values.amin(dim=-1, keepdim=True)
        max_value = values.amax(dim=-1, keepdim=True)
        return (values - min_value) / (max_value - min_value).clamp_min(float(eps))

    def _episode_mean_dist_token_g(
        self,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        mean_token = tokens.mean(dim=-2, keepdim=True)
        distance = torch.linalg.vector_norm(tokens - mean_token, dim=-1)
        return self._minmax_normalize_per_image(distance, float(self.eps)).to(dtype=tokens.dtype)

    def _prenorm_token_g(
        self,
        query_tokens: torch.Tensor,
        support_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cache = getattr(self, "_token_g_prenorm_cache", None)
        if cache is None or len(cache) < 2:
            raise RuntimeError(
                "token_g_kind='token_norm_pre_l2' requires cached projector-output norms. "
                "Use the standard Ours-Final local token path or token_g_kind='episode_mean_dist'."
            )
        query_norm = cache[0].to(device=query_tokens.device, dtype=query_tokens.dtype)
        support_norm = cache[1].to(device=support_tokens.device, dtype=support_tokens.dtype)
        if tuple(query_norm.shape) != tuple(query_tokens.shape[:-1]):
            raise ValueError(
                "Cached query token norms do not match query token shape: "
                f"{tuple(query_norm.shape)} vs {tuple(query_tokens.shape[:-1])}"
            )
        expected_support_flat = int(support_tokens.shape[0] * support_tokens.shape[1])
        if support_norm.dim() != 2 or support_norm.shape[0] != expected_support_flat:
            raise ValueError(
                "Cached support token norms do not match shot-decomposed support shape: "
                f"{tuple(support_norm.shape)} vs flat support count {expected_support_flat}"
            )
        support_norm = support_norm.reshape(
            support_tokens.shape[0],
            support_tokens.shape[1],
            support_tokens.shape[-2],
        )
        return (
            self._minmax_normalize_per_image(query_norm, float(self.eps)),
            self._minmax_normalize_per_image(support_norm, float(self.eps)),
        )

    def _compute_token_g(
        self,
        query_tokens: torch.Tensor,
        support_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.token_g_kind == "episode_mean_dist":
            return (
                self._episode_mean_dist_token_g(query_tokens),
                self._episode_mean_dist_token_g(support_tokens),
            )
        if self.token_g_kind == "token_norm_pre_l2":
            return self._prenorm_token_g(query_tokens, support_tokens)
        raise RuntimeError("token_g_kind='none' does not compute token-g diagnostics")

    def _dmuot_marginal_probs(
        self,
        token_g_query: torch.Tensor,
        token_g_support: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if math.isinf(float(self.tau_marg)):
            query_prob = torch.full_like(token_g_query, 1.0 / float(token_g_query.shape[-1]))
            support_prob = torch.full_like(token_g_support, 1.0 / float(token_g_support.shape[-1]))
            return query_prob, support_prob
        tau = max(float(self.tau_marg), float(self.eps))
        return torch.softmax(token_g_query / tau, dim=-1), torch.softmax(token_g_support / tau, dim=-1)

    def _forward_ecot_budget_bank(
        self,
        flat_cost: torch.Tensor,
        *,
        way_num: int,
        shot_num: int,
        support_weight: torch.Tensor | None = None,
        query_weight: torch.Tensor | None = None,
        support_tokens: torch.Tensor | None = None,
        query_tokens: torch.Tensor | None = None,
        spatial_hw: tuple[int, int] | None = None,
    ) -> dict[str, torch.Tensor]:
        dmuot_payload: dict[str, torch.Tensor] = {}
        token_g_query = None
        token_g_support = None
        cost_for_transport = flat_cost

        if self.token_g_kind != "none":
            if query_tokens is None or support_tokens is None:
                raise ValueError("token_g_kind != 'none' requires query_tokens and support_tokens")
            token_g_query, token_g_support = self._compute_token_g(query_tokens, support_tokens)
            dmuot_payload["token_g_query"] = token_g_query
            dmuot_payload["token_g_support"] = token_g_support

            flat_support_g = token_g_support.reshape(way_num * shot_num, token_g_support.shape[-1])
            cost_modulator = 1.0 + float(self.lambda_cost) * (
                1.0 - token_g_query[:, None, :, None] * flat_support_g[None, :, None, :]
            )
            cost_modulator = cost_modulator.to(device=flat_cost.device, dtype=flat_cost.dtype)
            dmuot_payload["cost_modulator"] = cost_modulator.reshape(
                flat_cost.shape[0],
                way_num,
                shot_num,
                flat_cost.shape[-2],
                flat_cost.shape[-1],
            )
            if self.lambda_cost > 0.0:
                cost_for_transport = flat_cost * cost_modulator
                dmuot_payload["cost_matrix_modulated"] = cost_for_transport.reshape(
                    flat_cost.shape[0],
                    way_num,
                    shot_num,
                    flat_cost.shape[-2],
                    flat_cost.shape[-1],
                )

            base_rho = self._ecot_base_rho_tensor(flat_cost)
            if self.marginal_kind == "discriminative":
                if query_weight is not None or support_weight is not None:
                    raise ValueError(
                        "marginal_kind='discriminative' cannot be combined with explicit query/support weights"
                    )
                query_prob, support_prob = self._dmuot_marginal_probs(token_g_query, token_g_support)
                query_weight = query_prob
                support_weight = support_prob
                dmuot_payload["marginal_query"] = query_prob * base_rho
                dmuot_payload["marginal_support"] = support_prob * base_rho
            else:
                base_rho = base_rho.to(device=token_g_query.device, dtype=token_g_query.dtype)
                dmuot_payload["marginal_query"] = token_g_query.new_full(
                    token_g_query.shape,
                    1.0 / float(token_g_query.shape[-1]),
                ) * base_rho
                dmuot_payload["marginal_support"] = token_g_support.new_full(
                    token_g_support.shape,
                    1.0 / float(token_g_support.shape[-1]),
                ) * base_rho.to(device=token_g_support.device, dtype=token_g_support.dtype)

        payload = super()._forward_ecot_budget_bank(
            cost_for_transport,
            way_num=way_num,
            shot_num=shot_num,
            support_weight=support_weight,
            query_weight=query_weight,
            support_tokens=support_tokens,
            query_tokens=query_tokens,
            spatial_hw=spatial_hw,
        )

        if dmuot_payload:
            plan = payload["transport_plan"]
            dmuot_payload["transport_plan_modulated"] = plan
            if "marginal_query" in dmuot_payload and "marginal_support" in dmuot_payload:
                marginal_query = dmuot_payload["marginal_query"].to(device=plan.device, dtype=plan.dtype)
                marginal_support = dmuot_payload["marginal_support"].to(device=plan.device, dtype=plan.dtype)
                row_mass = plan.sum(dim=-1)
                col_mass = plan.sum(dim=-2)
                query_l1 = (
                    row_mass
                    - marginal_query[:, None, None, :].expand_as(row_mass)
                ).abs().sum(dim=-1)
                support_l1 = (col_mass - marginal_support[None, :, :, :].expand_as(col_mass)).abs().sum(dim=-1)
                dmuot_payload["marginal_query_l1_drift"] = query_l1.mean()
                dmuot_payload["marginal_support_l1_drift"] = support_l1.mean()
                dmuot_payload["marginal_l1_drift"] = 0.5 * (
                    dmuot_payload["marginal_query_l1_drift"]
                    + dmuot_payload["marginal_support_l1_drift"]
                )
            payload.update(dmuot_payload)
        return payload

    def _update_differential_mode_template(self, episode_template: torch.Tensor) -> None:
        if not self.training:
            return
        with torch.no_grad():
            template = episode_template.detach()
            if tuple(self.dm_global_template.shape) != tuple(template.shape):
                self.dm_global_template = template.clone()
                self.dm_template_count = template.new_tensor(1.0)
                return
            count = self.dm_template_count.to(device=template.device, dtype=template.dtype)
            next_count = count + 1.0
            updated = self.dm_global_template.to(device=template.device, dtype=template.dtype)
            updated = updated + (template - updated) / next_count
            self.dm_global_template = updated.detach()
            self.dm_template_count = next_count.detach()

    def _differential_mode_template(self, support_tokens: torch.Tensor) -> torch.Tensor:
        episode_template = support_tokens.mean(dim=0)
        self._update_differential_mode_template(episode_template)
        if (
            self.training
            or self.dm_alpha <= 0.0
            or self.dm_global_template.numel() == 0
            or tuple(self.dm_global_template.shape) != tuple(episode_template.shape)
        ):
            return episode_template
        global_template = self.dm_global_template.to(
            device=episode_template.device,
            dtype=episode_template.dtype,
        )
        alpha = float(self.dm_alpha)
        return alpha * global_template + (1.0 - alpha) * episode_template

    def _apply_differential_mode_to_cost_tokens(
        self,
        query_tokens: torch.Tensor,
        support_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.use_differential_mode:
            return query_tokens, support_tokens
        if query_tokens.dim() != 3 or support_tokens.dim() != 3:
            raise ValueError(
                "Ours differential mode expects query/support cost tokens shaped "
                "(NumQuery, Tokens, Dim) and (NumSupport, Tokens, Dim)"
            )
        if query_tokens.shape[-2:] != support_tokens.shape[-2:]:
            raise ValueError(
                "Ours differential mode requires aligned spatial token grids, "
                f"got {tuple(query_tokens.shape[-2:])} vs {tuple(support_tokens.shape[-2:])}"
            )

        template = self._differential_mode_template(support_tokens)
        query_diff = query_tokens - template.unsqueeze(0)
        support_diff = support_tokens - template.unsqueeze(0)
        torch._assert(query_diff.shape == query_tokens.shape, "DMT changed query token shape")
        torch._assert(support_diff.shape == support_tokens.shape, "DMT changed support token shape")
        self._last_dm_diagnostics = self._build_differential_mode_diagnostics(
            template,
            query_tokens,
            support_tokens,
            query_diff,
            support_diff,
        )
        self._maybe_export_differential_mode_debug(template, self._last_dm_diagnostics)
        return query_diff, support_diff

    def _build_differential_mode_diagnostics(
        self,
        template: torch.Tensor,
        query_tokens: torch.Tensor,
        support_tokens: torch.Tensor,
        query_diff: torch.Tensor,
        support_diff: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        eps = float(self.eps)
        query_residual_ratio = query_diff.norm() / query_tokens.norm().clamp_min(eps)
        support_residual_ratio = support_diff.norm() / support_tokens.norm().clamp_min(eps)
        residual_ratio = 0.5 * (query_residual_ratio + support_residual_ratio)
        token_norm = template.norm(dim=-1)
        return {
            "dm_mu_norm": token_norm.mean(),
            "dm_mu_max_norm": token_norm.max(),
            "dm_mu_support_ratio": template.norm() / support_tokens.norm().clamp_min(eps),
            "dm_query_residual_ratio": query_residual_ratio,
            "dm_support_residual_ratio": support_residual_ratio,
            "dm_residual_ratio": residual_ratio,
            "dm_alpha": template.new_tensor(float(self.dm_alpha)),
            "dm_template_count": self.dm_template_count.to(device=template.device, dtype=template.dtype),
            "dm_mu_token_norm": token_norm.detach(),
        }

    def _maybe_export_differential_mode_debug(
        self,
        template: torch.Tensor,
        diagnostics: dict[str, torch.Tensor],
    ) -> None:
        if not self.dm_debug:
            return
        if self.dm_debug_max_episodes > 0 and self._dm_debug_count >= self.dm_debug_max_episodes:
            return
        self._dm_debug_count += 1
        debug_idx = self._dm_debug_count
        try:
            os.makedirs(self.dm_debug_dir, exist_ok=True)
        except OSError as exc:  # pragma: no cover - depends on filesystem permissions
            print(f"[Ours-DMT] debug disabled: cannot create {self.dm_debug_dir}: {exc}")
            return

        token_norm = diagnostics["dm_mu_token_norm"].detach().float().cpu()
        token_hw = _infer_token_hw(int(token_norm.numel()))
        heatmap_path = os.path.join(self.dm_debug_dir, f"dmt_mu_heatmap_{debug_idx:04d}.png")

        try:
            import matplotlib

            matplotlib.use("Agg", force=True)
            import matplotlib.pyplot as plt

            heatmap = token_norm.reshape(*token_hw).numpy()
            fig, ax = plt.subplots(figsize=(5.0, 4.2))
            image = ax.imshow(heatmap, cmap="magma")
            ax.set_title(
                "Ours DMT common component | "
                f"mu_norm={float(diagnostics['dm_mu_norm'].detach().cpu()):.4f}"
            )
            ax.set_xlabel("token x")
            ax.set_ylabel("token y")
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            fig.savefig(heatmap_path, dpi=180, bbox_inches="tight")
            plt.close(fig)
        except Exception as exc:  # pragma: no cover - defensive debug path
            heatmap_path = f"failed: {exc}"

        csv_path = os.path.join(self.dm_debug_dir, "dmt_debug_metrics.csv")
        write_header = not os.path.exists(csv_path)
        row = {
            "episode": debug_idx,
            "training": int(self.training),
            "dm_alpha": float(diagnostics["dm_alpha"].detach().cpu()),
            "dm_template_count": float(diagnostics["dm_template_count"].detach().cpu()),
            "dm_mu_norm": float(diagnostics["dm_mu_norm"].detach().cpu()),
            "dm_mu_max_norm": float(diagnostics["dm_mu_max_norm"].detach().cpu()),
            "dm_mu_support_ratio": float(diagnostics["dm_mu_support_ratio"].detach().cpu()),
            "dm_query_residual_ratio": float(diagnostics["dm_query_residual_ratio"].detach().cpu()),
            "dm_support_residual_ratio": float(diagnostics["dm_support_residual_ratio"].detach().cpu()),
            "dm_residual_ratio": float(diagnostics["dm_residual_ratio"].detach().cpu()),
            "heatmap_path": heatmap_path,
        }
        try:
            with open(csv_path, "a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
        except OSError as exc:  # pragma: no cover - depends on filesystem permissions
            print(f"[Ours-DMT] debug metrics skipped: cannot write {csv_path}: {exc}")
            return

        print(
            "[Ours-DMT] "
            f"episode={debug_idx} "
            f"mu_norm={row['dm_mu_norm']:.6f} "
            f"mu_support_ratio={row['dm_mu_support_ratio']:.6f} "
            f"residual_ratio={row['dm_residual_ratio']:.6f} "
            f"q_residual={row['dm_query_residual_ratio']:.6f} "
            f"s_residual={row['dm_support_residual_ratio']:.6f} "
            f"heatmap={heatmap_path}"
        )

    def _ground_cost(self, query_tokens: torch.Tensor, class_tokens: torch.Tensor) -> torch.Tensor:
        query_tokens, class_tokens = self._apply_differential_mode_to_cost_tokens(
            query_tokens,
            class_tokens,
        )
        return super()._ground_cost(query_tokens, class_tokens)

    def _forward_episode(self, *args, **kwargs) -> torch.Tensor | dict[str, torch.Tensor]:
        self._last_dm_diagnostics = None
        use_prenorm_cache = self.token_g_kind == "token_norm_pre_l2"
        if use_prenorm_cache:
            self._token_g_prenorm_cache = []
        try:
            outputs = super()._forward_episode(*args, **kwargs)
        finally:
            if use_prenorm_cache:
                self._token_g_prenorm_cache = None
        if isinstance(outputs, dict):
            if self.lambda_cost > 0.0 and "cost_matrix_modulated" in outputs:
                if "base_cost_matrix" not in outputs and "cost_matrix" in outputs:
                    outputs["base_cost_matrix"] = outputs["cost_matrix"]
                outputs["cost_matrix"] = outputs["cost_matrix_modulated"]
            if self._last_dm_diagnostics is not None:
                outputs.update(self._last_dm_diagnostics)
        return outputs

    def _stack_outputs(self, batch_outputs: list[dict[str, torch.Tensor]]) -> HROTFSLResult:
        stacked = super()._stack_outputs(batch_outputs)
        for key in (
            "token_g_query",
            "token_g_support",
            "cost_modulator",
            "transport_plan_modulated",
            "marginal_query",
            "marginal_support",
            "cost_matrix_modulated",
        ):
            if key in batch_outputs[0]:
                stacked[key] = torch.stack([item[key] for item in batch_outputs], dim=0)
        for key in (
            "marginal_query_l1_drift",
            "marginal_support_l1_drift",
            "marginal_l1_drift",
        ):
            if key in batch_outputs[0]:
                stacked[key] = torch.stack([item[key] for item in batch_outputs]).mean()
        for key in (
            "dm_mu_norm",
            "dm_mu_max_norm",
            "dm_mu_support_ratio",
            "dm_query_residual_ratio",
            "dm_support_residual_ratio",
            "dm_residual_ratio",
            "dm_alpha",
            "dm_template_count",
        ):
            if key in batch_outputs[0]:
                stacked[key] = torch.stack([item[key] for item in batch_outputs]).mean()
        if "dm_mu_token_norm" in batch_outputs[0]:
            stacked["dm_mu_token_norm"] = torch.stack(
                [item["dm_mu_token_norm"] for item in batch_outputs],
                dim=0,
            )
        return stacked

__all__ = ["OursM2", "apply_ours_design_defaults", "OURS_ABLATIONS", "normalize_ours_ablation"]
