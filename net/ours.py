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
from net.hrot_fsl import HROTFSLResult, _compute_cross_shot_support_marginals, _inverse_softplus
from net.hyperbolic.poincare_ops import safe_project_to_ball
from net.modules.multiscale_tokenizer import MultiScaleTokenizer
from net.modules.pulse_region_guidance import PulseRegionGuidance, normalize_saliency
from net.modules.region_structural_uot import RegionStructuralUOTGuidance
from net.modules.spatial_context_enrichment import SpatialContextEnrichment, parse_context_kernel_sizes
from net.modules.structural_token_augmentation import StructuralTokenAugmentation
from net.jecot_m2 import JECOTM2
from net.modules.partial_ot import (
    compute_partial_transport_cost,
    compute_partial_transported_mass,
    solve_partial_transport,
)
from net.modules.unbalanced_ot import (
    compute_transport_cost,
    compute_transported_mass,
    sinkhorn_balanced_log,
    sinkhorn_balanced_pot,
    sinkhorn_unbalanced_log,
    sinkhorn_unbalanced_pot,
)


OURS_ABLATIONS = frozenset({"full", "full_ot", "no_egsm", "gap"})
TOKEN_G_KINDS = frozenset({"none", "episode_mean_dist", "token_norm_pre_l2", "learned_attention"})
DMUOT_MARGINAL_KINDS = frozenset({"uniform", "discriminative"})
DMUOT_SHOT_STRENGTH_KINDS = frozenset({"none", "inverse_sqrt", "inverse"})
PULSE_TRAIN_SCHEDULES = frozenset({"constant", "decay", "warmup_decay", "eval_only"})


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
        "learned": "learned_attention",
        "attention": "learned_attention",
        "learned_attn": "learned_attention",
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


def _normalize_dmuot_shot_strength(value: str | None) -> str:
    name = "none" if value is None else str(value).strip().lower().replace("-", "_")
    aliases = {
        "off": "none",
        "false": "none",
        "sqrt": "inverse_sqrt",
        "inv_sqrt": "inverse_sqrt",
        "shot_sqrt": "inverse_sqrt",
        "inv": "inverse",
        "shot_inverse": "inverse",
    }
    name = aliases.get(name, name)
    if name not in DMUOT_SHOT_STRENGTH_KINDS:
        raise ValueError(
            f"Unsupported dmuot_shot_strength: {value}. "
            f"Expected one of {sorted(DMUOT_SHOT_STRENGTH_KINDS)}"
        )
    return name


def _normalize_pulse_train_schedule(value: str | None) -> str:
    name = "constant" if value is None else str(value).strip().lower().replace("-", "_")
    aliases = {
        "off": "eval_only",
        "visualize_only": "eval_only",
        "visualization_only": "eval_only",
        "cosine_decay": "decay",
        "linear_decay": "decay",
        "warmup": "warmup_decay",
        "warm_up_decay": "warmup_decay",
    }
    name = aliases.get(name, name)
    if name not in PULSE_TRAIN_SCHEDULES:
        raise ValueError(
            f"Unsupported pulse_region_train_schedule: {value}. "
            f"Expected one of {sorted(PULSE_TRAIN_SCHEDULES)}"
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
        dmuot_shot_strength: str = "none",
        **kwargs,
    ) -> None:
        enable_multiscale_ot = _bool_config(kwargs.pop("enable_multiscale_ot", False))
        multiscale_pool_sizes = kwargs.pop("multiscale_pool_sizes", "original,2x2,1x1")
        multiscale_per_scale_T = _bool_config(kwargs.pop("multiscale_per_scale_T", False))
        enable_context_enrichment = _bool_config(kwargs.pop("enable_context_enrichment", False))
        context_kernel_sizes_raw = kwargs.pop("context_kernel_sizes", "3,5")
        context_fusion = str(kwargs.pop("context_fusion", "weighted_sum"))
        context_gate_max = float(kwargs.pop("context_gate_max", 1.0))
        context_change_max = float(kwargs.pop("context_change_max", 0.0))
        context_debug = _bool_config(kwargs.pop("context_debug", False))
        context_debug_dir = str(kwargs.pop("context_debug_dir", "results/context_debug"))
        context_debug_max_episodes = int(kwargs.pop("context_debug_max_episodes", 3))
        enable_structural_augmentation = _bool_config(kwargs.pop("enable_structural_augmentation", False))
        struct_dim = int(kwargs.pop("struct_dim", 16))
        enable_region_structural_uot = _bool_config(kwargs.pop("enable_region_structural_uot", False))
        region_uot_grid_size = kwargs.pop("region_uot_grid_size", "3x3")
        region_uot_strength = float(kwargs.pop("region_uot_strength", 0.20))
        region_uot_fgw_alpha = float(kwargs.pop("region_uot_fgw_alpha", 0.35))
        region_uot_sinkhorn_epsilon = float(kwargs.pop("region_uot_sinkhorn_epsilon", 0.08))
        region_uot_fgw_iters = int(kwargs.pop("region_uot_fgw_iters", 4))
        region_uot_sinkhorn_iters = int(kwargs.pop("region_uot_sinkhorn_iters", 40))
        enable_pulse_region_uot = _bool_config(kwargs.pop("enable_pulse_region_uot", False))
        pulse_region_kernel_size = int(kwargs.pop("pulse_region_kernel_size", 5))
        pulse_region_cost_weight = float(kwargs.pop("pulse_region_cost_weight", 0.35))
        pulse_saliency_mass_mix = float(kwargs.pop("pulse_saliency_mass_mix", 0.50))
        pulse_saliency_cost_discount = float(kwargs.pop("pulse_saliency_cost_discount", 0.10))
        pulse_saliency_image_weight = float(kwargs.pop("pulse_saliency_image_weight", 0.45))
        pulse_saliency_feature_weight = float(kwargs.pop("pulse_saliency_feature_weight", 0.35))
        pulse_saliency_contrast_weight = float(kwargs.pop("pulse_saliency_contrast_weight", 0.20))
        pulse_region_train_strength = float(kwargs.pop("pulse_region_train_strength", 1.0))
        pulse_region_eval_strength = float(kwargs.pop("pulse_region_eval_strength", 1.0))
        pulse_region_multishot_train_strength = kwargs.pop("pulse_region_multishot_train_strength", None)
        pulse_region_multishot_eval_strength = kwargs.pop("pulse_region_multishot_eval_strength", None)
        pulse_region_train_schedule = _normalize_pulse_train_schedule(
            kwargs.pop("pulse_region_train_schedule", "constant")
        )
        pulse_support_consensus_weight = float(kwargs.pop("pulse_support_consensus_weight", 0.0))
        pulse_support_consensus_beta = float(kwargs.pop("pulse_support_consensus_beta", 5.0))
        pulse_support_consensus_eta = float(kwargs.pop("pulse_support_consensus_eta", 0.05))
        enable_pot_guide = bool(kwargs.pop("enable_pot_guide", False))
        pot_guide_s = float(kwargs.pop("pot_guide_s", 0.5))
        pot_guide_adaptive_s = bool(kwargs.pop("pot_guide_adaptive_s", False))
        pot_guide_s_min = float(kwargs.pop("pot_guide_s_min", 0.2))
        pot_guide_s_max = float(kwargs.pop("pot_guide_s_max", 0.8))
        pot_guide_epsilon = float(kwargs.pop("pot_guide_epsilon", 0.05))
        pot_guide_max_iter = int(kwargs.pop("pot_guide_max_iter", 50))
        if not 0.0 <= float(dm_alpha) <= 1.0:
            raise ValueError("dm_alpha must be in [0, 1]")
        if int(dm_debug_max_episodes) < 0:
            raise ValueError("dm_debug_max_episodes must be non-negative")
        self.token_g_kind = _normalize_token_g_kind(token_g_kind)
        self.lambda_cost = float(lambda_cost)
        self.marginal_kind = _normalize_dmuot_marginal_kind(marginal_kind)
        self.tau_marg = float(tau_marg)
        self.dmuot_shot_strength = _normalize_dmuot_shot_strength(dmuot_shot_strength)
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
        if enable_multiscale_ot:
            if self.ours_ablation == "gap":
                raise ValueError("enable_multiscale_ot is not supported with ours_ablation='gap'")
            if self.token_g_kind != "none":
                raise ValueError("enable_multiscale_ot currently requires token_g_kind='none'")
            if _bool_config(kwargs.get("hrot_amp_marginals", False)):
                raise ValueError("enable_multiscale_ot is not supported with hrot_amp_marginals")
            if _bool_config(kwargs.get("ecot_episode_feature_normalize", False)):
                raise ValueError("enable_multiscale_ot is not supported with ecot_episode_feature_normalize")
            if _bool_config(kwargs.get("pre_transport_shot_pool", False)):
                raise ValueError("enable_multiscale_ot is not supported with pre_transport_shot_pool")
        if enable_region_structural_uot:
            if self.ours_ablation == "gap":
                raise ValueError("enable_region_structural_uot is not supported with ours_ablation='gap'")
            if enable_multiscale_ot:
                raise ValueError("enable_region_structural_uot is not supported with enable_multiscale_ot")
            if enable_pulse_region_uot:
                raise ValueError("enable_region_structural_uot is not supported with enable_pulse_region_uot")
            if _bool_config(kwargs.get("pre_transport_shot_pool", False)):
                raise ValueError("enable_region_structural_uot is not supported with pre_transport_shot_pool")
        if enable_pulse_region_uot:
            if self.ours_ablation == "gap":
                raise ValueError("enable_pulse_region_uot is not supported with ours_ablation='gap'")
            if enable_multiscale_ot:
                raise ValueError("enable_pulse_region_uot is not supported with enable_multiscale_ot")
            if _bool_config(kwargs.get("pre_transport_shot_pool", False)):
                raise ValueError("enable_pulse_region_uot is not supported with pre_transport_shot_pool")
        super().__init__(
            *args,
            rho=float(kwargs["ecot_base_rho"]),
            transport_mode=kwargs["ecot_transport_mode"],
            **kwargs,
        )
        self.enable_pulse_region_uot = bool(enable_pulse_region_uot)
        self.enable_region_structural_uot = bool(enable_region_structural_uot)
        self._pulse_saliency_cache: list[torch.Tensor] | None = None
        self._homotopy_progress = 0.0
        self.pulse_saliency_image_weight = float(pulse_saliency_image_weight)
        self.pulse_saliency_feature_weight = float(pulse_saliency_feature_weight)
        self.pulse_saliency_contrast_weight = float(pulse_saliency_contrast_weight)
        self.pulse_region_train_strength = float(pulse_region_train_strength)
        self.pulse_region_eval_strength = float(pulse_region_eval_strength)
        self.pulse_region_multishot_train_strength = (
            None if pulse_region_multishot_train_strength is None else float(pulse_region_multishot_train_strength)
        )
        self.pulse_region_multishot_eval_strength = (
            None if pulse_region_multishot_eval_strength is None else float(pulse_region_multishot_eval_strength)
        )
        self.pulse_region_train_schedule = str(pulse_region_train_schedule)
        for name, value in (
            ("pulse_region_train_strength", self.pulse_region_train_strength),
            ("pulse_region_eval_strength", self.pulse_region_eval_strength),
            ("pulse_region_multishot_train_strength", self.pulse_region_multishot_train_strength),
            ("pulse_region_multishot_eval_strength", self.pulse_region_multishot_eval_strength),
        ):
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        self.pulse_support_consensus_weight = float(pulse_support_consensus_weight)
        self.pulse_support_consensus_beta = float(pulse_support_consensus_beta)
        self.pulse_support_consensus_eta = float(pulse_support_consensus_eta)
        if not 0.0 <= self.pulse_support_consensus_weight <= 1.0:
            raise ValueError("pulse_support_consensus_weight must be in [0, 1]")
        if self.pulse_support_consensus_beta <= 0.0:
            raise ValueError("pulse_support_consensus_beta must be positive")
        if not 0.0 <= self.pulse_support_consensus_eta <= 1.0:
            raise ValueError("pulse_support_consensus_eta must be in [0, 1]")
        if self.enable_pulse_region_uot:
            self.pulse_region_guidance = PulseRegionGuidance(
                kernel_size=pulse_region_kernel_size,
                region_cost_weight=pulse_region_cost_weight,
                saliency_mass_mix=pulse_saliency_mass_mix,
                saliency_cost_discount=pulse_saliency_cost_discount,
                ground_cost=self.ground_cost,
                eps=self.eps,
            )
        if self.enable_region_structural_uot:
            self.region_structural_uot = RegionStructuralUOTGuidance(
                grid_size=region_uot_grid_size,
                strength=region_uot_strength,
                fgw_alpha=region_uot_fgw_alpha,
                tau=float(self.tau_q),
                sinkhorn_epsilon=region_uot_sinkhorn_epsilon,
                fgw_iters=region_uot_fgw_iters,
                sinkhorn_iters=region_uot_sinkhorn_iters,
                sinkhorn_tol=self.sinkhorn_tolerance,
                ground_cost=self.ground_cost,
                eps=self.eps,
            )
        if enable_multiscale_ot:
            self.enable_multiscale_ot = True
            self.multi_scale_tokenizer = MultiScaleTokenizer(multiscale_pool_sizes)
            self.num_multiscale_scales = len(self.multi_scale_tokenizer.scale_names)
            self.scale_weights = torch.nn.Parameter(torch.zeros(self.num_multiscale_scales, dtype=torch.float32))
            self.multiscale_per_scale_T = bool(multiscale_per_scale_T)
            if self.multiscale_per_scale_T:
                if (
                    len(self.ecot_rho_bank) != 1
                    or self.ecot_m2_ablate_threshold_mass
                    or self.ecot_m2_cost_per_mass_score
                    or self.ecot_m2_per_shot_threshold
                    or self.ecot_m2_use_swts
                ):
                    raise ValueError(
                        "multiscale_per_scale_T requires single-budget active threshold-mass scoring"
                    )
                threshold_init = float(self.transport_cost_threshold.detach().float().cpu().item())
                raw_threshold = _inverse_softplus(max(threshold_init, float(self.eps)))
                self.raw_multiscale_transport_cost_thresholds = torch.nn.Parameter(
                    torch.full((self.num_multiscale_scales,), raw_threshold, dtype=torch.float32)
                )
        self.enable_pot_guide = bool(enable_pot_guide)
        if self.enable_pot_guide:
            from net.modules.pot_guide import POTGuideMarginals

            self.pot_guide_module = POTGuideMarginals(
                fixed_s=pot_guide_s,
                adaptive_s=pot_guide_adaptive_s,
                s_min=pot_guide_s_min,
                s_max=pot_guide_s_max,
                epsilon=pot_guide_epsilon,
                max_iter=pot_guide_max_iter,
                eps=self.eps,
            )
            self._last_pot_guide_diagnostics: dict[str, torch.Tensor] | None = None
        self.token_attention_query = (
            torch.nn.Parameter(torch.randn(self.token_dim) * 0.01)
            if self.token_g_kind == "learned_attention"
            else None
        )
        self.register_buffer("dm_global_template", torch.empty(0), persistent=False)
        self.register_buffer("dm_template_count", torch.tensor(0.0), persistent=False)
        self.enable_context_enrichment = bool(enable_context_enrichment)
        self._context_debug = bool(context_debug)
        self._context_debug_dir = str(context_debug_dir)
        self._context_debug_max = int(context_debug_max_episodes)
        self._context_debug_count = 0
        if self.enable_context_enrichment:
            context_ks = parse_context_kernel_sizes(context_kernel_sizes_raw)
            self.context_enrichment = SpatialContextEnrichment(
                channels=self.hidden_dim,
                kernel_sizes=context_ks,
                fusion=context_fusion,
                gate_max=context_gate_max,
                change_max=context_change_max,
            )
            self._last_context_diagnostics: dict[str, torch.Tensor] | None = None

        self.enable_structural_augmentation = bool(enable_structural_augmentation)
        if self.enable_structural_augmentation:
            self.structural_augmentation = StructuralTokenAugmentation(
                token_dim=self.token_dim,
                struct_dim=struct_dim,
            )
            self._last_struct_diagnostics: dict[str, torch.Tensor] | None = None

    @property
    def uses_ours_gap_control(self) -> bool:
        return self.ours_ablation == "gap"

    @staticmethod
    def _spatial_minmax(values: torch.Tensor, eps: float) -> torch.Tensor:
        flat = values.flatten(1)
        v_min = flat.amin(dim=1).reshape(-1, 1, 1, 1)
        v_max = flat.amax(dim=1).reshape(-1, 1, 1, 1)
        return (values - v_min) / (v_max - v_min).clamp_min(float(eps))

    def _compute_pulse_saliency(
        self,
        images: torch.Tensor,
        feature_map: torch.Tensor,
    ) -> torch.Tensor:
        """Return token-grid saliency for bright/energetic pulse regions."""
        with torch.no_grad():
            spatial_hw = feature_map.shape[-2:]
            image_energy = images.detach().float().abs().mean(dim=1, keepdim=True)
            image_energy = F.adaptive_avg_pool2d(image_energy, output_size=spatial_hw)
            feature_energy = feature_map.detach().float().pow(2).sum(dim=1, keepdim=True).sqrt()

            image_norm = self._spatial_minmax(image_energy, self.eps)
            feature_norm = self._spatial_minmax(feature_energy, self.eps)
            base = 0.5 * (image_norm + feature_norm)
            local_mean = F.avg_pool2d(base, kernel_size=3, stride=1, padding=1, count_include_pad=False)
            contrast = self._spatial_minmax((base - local_mean).clamp_min(0.0), self.eps)

            weights = image_norm.new_tensor(
                [
                    max(self.pulse_saliency_image_weight, 0.0),
                    max(self.pulse_saliency_feature_weight, 0.0),
                    max(self.pulse_saliency_contrast_weight, 0.0),
                ]
            )
            weights = weights / weights.sum().clamp_min(self.eps)
            saliency = weights[0] * image_norm + weights[1] * feature_norm + weights[2] * contrast
            saliency = saliency.clamp_min(0.0)
            return saliency.flatten(1).to(device=feature_map.device, dtype=feature_map.dtype)

    def _record_pulse_saliency(self, images: torch.Tensor, feature_map: torch.Tensor) -> None:
        if not getattr(self, "enable_pulse_region_uot", False):
            return
        cache = getattr(self, "_pulse_saliency_cache", None)
        if cache is None:
            return
        cache.append(self._compute_pulse_saliency(images, feature_map).detach())

    def set_homotopy_progress(self, progress: float) -> None:
        parent_setter = getattr(super(), "set_homotopy_progress", None)
        if parent_setter is not None:
            parent_setter(progress)
        self._homotopy_progress = float(min(max(float(progress), 0.0), 1.0))

    def _pulse_region_effective_strength(
        self,
        reference: torch.Tensor,
        *,
        shot_num: int,
    ) -> torch.Tensor:
        if not getattr(self, "enable_pulse_region_uot", False):
            return reference.new_tensor(0.0)
        train_strength = self.pulse_region_train_strength
        eval_strength = self.pulse_region_eval_strength
        if int(shot_num) > 1:
            if self.pulse_region_multishot_train_strength is not None:
                train_strength = self.pulse_region_multishot_train_strength
            if self.pulse_region_multishot_eval_strength is not None:
                eval_strength = self.pulse_region_multishot_eval_strength
        if not self.training:
            return reference.new_tensor(eval_strength)

        progress = float(min(max(getattr(self, "_homotopy_progress", 0.0), 0.0), 1.0))
        schedule = getattr(self, "pulse_region_train_schedule", "constant")
        if schedule == "eval_only":
            factor = 0.0
        elif schedule == "decay":
            factor = 1.0 - progress
        elif schedule == "warmup_decay":
            warmup = 0.10
            if progress <= warmup:
                factor = progress / warmup
            else:
                factor = max(0.0, (1.0 - progress) / (1.0 - warmup))
        else:
            factor = 1.0
        return reference.new_tensor(train_strength * factor)

    def _apply_pulse_support_consensus(
        self,
        support_tokens: torch.Tensor,
        support_saliency: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        weight = float(getattr(self, "pulse_support_consensus_weight", 0.0))
        if support_tokens.dim() != 4 or support_tokens.shape[1] <= 1 or weight <= 0.0:
            return support_saliency, {}

        consensus_mass, consistency, sigma = _compute_cross_shot_support_marginals(
            support_tokens.unsqueeze(0),
            rho=1.0,
            beta=float(self.pulse_support_consensus_beta),
            eta=float(self.pulse_support_consensus_eta),
            detach=True,
            distance=self.ground_cost,
            eps=float(self.eps),
            min_sigma=float(self.eps),
        )
        consensus_prob = normalize_saliency(
            consensus_mass.squeeze(0).to(device=support_saliency.device, dtype=support_saliency.dtype),
            eps=float(self.eps),
        )
        saliency_prob = normalize_saliency(support_saliency, eps=float(self.eps))
        mixed = (1.0 - weight) * saliency_prob + weight * consensus_prob
        entropy = -(consensus_prob * consensus_prob.clamp_min(self.eps).log()).sum(dim=-1).mean()
        payload = {
            "pulse/support_consensus_weight": support_saliency.new_tensor(weight),
            "pulse/support_consensus_peak": consensus_prob.max(dim=-1).values.mean().detach(),
            "pulse/support_consensus_entropy": entropy.detach(),
            "pulse/support_consistency_mean": consistency.to(
                device=support_saliency.device,
                dtype=support_saliency.dtype,
            ).mean().detach(),
            "pulse/support_consistency_sigma_peak": sigma.to(
                device=support_saliency.device,
                dtype=support_saliency.dtype,
            ).max(dim=-1).values.mean().detach(),
        }
        return mixed, payload

    @staticmethod
    def _combine_token_weight(
        current: torch.Tensor | None,
        pulse_weight: torch.Tensor,
    ) -> torch.Tensor:
        if current is None:
            return pulse_weight
        return current.to(device=pulse_weight.device, dtype=pulse_weight.dtype) * pulse_weight

    def _pulse_saliency_pair(
        self,
        *,
        query_tokens: torch.Tensor,
        support_tokens: torch.Tensor,
        way_num: int,
        shot_num: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cache = getattr(self, "_pulse_saliency_cache", None)
        if cache is None or len(cache) < 2:
            query_saliency = query_tokens.detach().norm(dim=-1)
            support_saliency = support_tokens.detach().norm(dim=-1)
            return query_saliency, support_saliency
        query_saliency = cache[0].to(device=query_tokens.device, dtype=query_tokens.dtype)
        support_saliency = cache[1].to(device=support_tokens.device, dtype=support_tokens.dtype)
        if tuple(query_saliency.shape) != tuple(query_tokens.shape[:2]):
            query_saliency = query_tokens.detach().norm(dim=-1)
        if support_saliency.dim() == 2 and support_saliency.shape[0] == way_num * shot_num:
            support_saliency = support_saliency.reshape(way_num, shot_num, support_saliency.shape[-1])
        if tuple(support_saliency.shape) != tuple(support_tokens.shape[:3]):
            support_saliency = support_tokens.detach().norm(dim=-1)
        return query_saliency, support_saliency

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        feature_map = super().encode(x)
        if self._context_debug and hasattr(self, "_context_debug_images"):
            self._context_debug_images.append(x.detach().cpu())
        if self.enable_context_enrichment:
            enriched = self.context_enrichment(feature_map)
            self._last_context_diagnostics = self.context_enrichment.diagnostics(
                feature_map, enriched,
            )
            if self._context_debug and hasattr(self, "_context_debug_fmaps"):
                self._context_debug_fmaps.append(feature_map.detach().cpu())
            return enriched
        if self._context_debug and hasattr(self, "_context_debug_fmaps"):
            self._context_debug_fmaps.append(feature_map.detach().cpu())
        return feature_map

    def _apply_structural_augmentation(
        self,
        projected: torch.Tensor,
        spatial_hw: tuple[int, int],
    ) -> torch.Tensor:
        if not getattr(self, "enable_structural_augmentation", False):
            return projected
        before = projected
        projected = self.structural_augmentation(projected, spatial_hw)
        self._last_struct_diagnostics = self.structural_augmentation.diagnostics(before, projected)
        return projected

    @property
    def multiscale_transport_cost_thresholds(self) -> torch.Tensor:
        raw_thresholds = getattr(self, "raw_multiscale_transport_cost_thresholds", None)
        if raw_thresholds is None:
            return self.transport_cost_threshold.reshape(1)
        return F.softplus(raw_thresholds).clamp_min(self.eps)

    def _encode_multiscale_images(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
        feature_map = self.encode(images)
        scale_records = []
        for scale_name, spatial_hw, tokens in self.multi_scale_tokenizer(feature_map):
            projected = self._project_backbone_tokens(tokens)
            projected = self._apply_cata(projected)
            projected = self._apply_structural_augmentation(projected, spatial_hw)
            euclidean_tokens, hyperbolic_tokens = self._euclidean_and_hyperbolic_from_projected(projected)
            scale_records.append(
                {
                    "name": scale_name,
                    "spatial_hw": spatial_hw,
                    "euclidean": euclidean_tokens,
                    "hyperbolic": hyperbolic_tokens,
                }
            )
        if not scale_records:
            raise RuntimeError("Multi-scale tokenizer returned no scales")
        cache = getattr(self, "_multiscale_ot_cache", None)
        if cache is not None:
            cache.append(scale_records)
        first = scale_records[0]
        return first["euclidean"], first["hyperbolic"], first["spatial_hw"]

    def _encode_images(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
        if getattr(self, "enable_multiscale_ot", False):
            return self._encode_multiscale_images(images)

        need_local = (
            self.token_g_kind == "token_norm_pre_l2"
            or self.uses_ours_gap_control
            or getattr(self, "enable_structural_augmentation", False)
            or getattr(self, "enable_pulse_region_uot", False)
        )
        if not need_local:
            return super()._encode_images(images)

        feature_map = self.encode(images)
        self._record_pulse_saliency(images, feature_map)
        spatial_hw = (feature_map.shape[-2], feature_map.shape[-1])
        if self.uses_ours_gap_control:
            tokens = F.adaptive_avg_pool2d(feature_map, output_size=1).flatten(1).unsqueeze(1)
            spatial_hw = (1, 1)
        else:
            tokens = feature_map_to_tokens(feature_map)
        projected = self._project_backbone_tokens(tokens)
        if not self.uses_ours_gap_control:
            projected = self._apply_cata(projected)
        projected = self._apply_structural_augmentation(projected, spatial_hw)
        self._record_token_g_prenorm(projected)
        euclidean_tokens, hyperbolic_tokens = self._euclidean_and_hyperbolic_from_projected(projected)
        return euclidean_tokens, hyperbolic_tokens, spatial_hw

    def _project_tokens_from_images(self, images: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        need_local = (
            self.token_g_kind == "token_norm_pre_l2"
            or getattr(self, "enable_structural_augmentation", False)
            or getattr(self, "enable_pulse_region_uot", False)
        )
        if not need_local:
            return super()._project_tokens_from_images(images)
        feature_map = self.encode(images)
        self._record_pulse_saliency(images, feature_map)
        spatial_hw = (feature_map.shape[-2], feature_map.shape[-1])
        tokens = feature_map_to_tokens(feature_map)
        projected = self._project_backbone_tokens(tokens)
        projected = self._apply_cata(projected)
        projected = self._apply_structural_augmentation(projected, spatial_hw)
        self._record_token_g_prenorm(projected)
        return projected, spatial_hw

    def _encode_images_with_amp_norms(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, int]]:
        need_local = (
            self.token_g_kind == "token_norm_pre_l2"
            or getattr(self, "enable_structural_augmentation", False)
            or getattr(self, "enable_pulse_region_uot", False)
        )
        if not need_local:
            return super()._encode_images_with_amp_norms(images)
        feature_map = self.encode(images)
        self._record_pulse_saliency(images, feature_map)
        spatial_hw = (feature_map.shape[-2], feature_map.shape[-1])
        tokens = feature_map_to_tokens(feature_map)
        projected, pre_proj_norms = self._project_backbone_tokens_with_prenorm(tokens)
        if self.use_cata:
            projected = self._apply_cata(projected)
            pre_proj_norms = projected.norm(p=2, dim=-1).clamp(min=1e-6)
        projected = self._apply_structural_augmentation(projected, spatial_hw)
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

    def _learned_attention_token_g(
        self,
        query_tokens: torch.Tensor,
        support_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.token_attention_query is None:
            raise RuntimeError("token_attention_query not initialized for learned_attention")
        v = self.token_attention_query.to(device=query_tokens.device, dtype=query_tokens.dtype)
        d_sqrt = math.sqrt(float(v.shape[-1]))
        query_scores = (query_tokens @ v) / d_sqrt
        support_scores = (support_tokens @ v) / d_sqrt
        return (
            self._minmax_normalize_per_image(query_scores, float(self.eps)),
            self._minmax_normalize_per_image(support_scores, float(self.eps)),
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
        if self.token_g_kind == "learned_attention":
            return self._learned_attention_token_g(query_tokens, support_tokens)
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

    def _dmuot_shot_strength_value(self, shot_num: int, reference: torch.Tensor) -> torch.Tensor:
        if self.dmuot_shot_strength == "inverse_sqrt":
            value = 1.0 / math.sqrt(float(max(1, shot_num)))
        elif self.dmuot_shot_strength == "inverse":
            value = 1.0 / float(max(1, shot_num))
        else:
            value = 1.0
        return reference.new_tensor(value)

    def _blend_dmuot_prob_with_uniform(
        self,
        prob: torch.Tensor,
        strength: torch.Tensor,
    ) -> torch.Tensor:
        if self.dmuot_shot_strength == "none":
            return prob
        uniform = torch.full_like(prob, 1.0 / float(prob.shape[-1]))
        return strength.to(device=prob.device, dtype=prob.dtype) * prob + (
            1.0 - strength.to(device=prob.device, dtype=prob.dtype)
        ) * uniform

    def _prepare_multiscale_cost_tokens(
        self,
        query_tokens: torch.Tensor,
        support_tokens: torch.Tensor,
        *,
        way_num: int,
        shot_num: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cost_query_tokens = query_tokens
        cost_support_tokens = support_tokens
        if self.hrot_token_center:
            support_for_center = cost_support_tokens.reshape(
                way_num,
                shot_num,
                cost_support_tokens.shape[-2],
                cost_support_tokens.shape[-1],
            )
            support_mean = support_for_center.mean(dim=2, keepdim=True)
            centered_support = support_for_center - support_mean
            if self.training:
                centered_norms = centered_support.norm(dim=-1)
                if (centered_norms < 1e-4).any():
                    import logging

                    logging.getLogger(__name__).warning(
                        "Token centering produced near-zero vectors "
                        "(min norm %.2e)", centered_norms.min().item(),
                    )
            cost_support_tokens = centered_support.reshape(
                way_num * shot_num,
                cost_support_tokens.shape[-2],
                cost_support_tokens.shape[-1],
            )
            if self.hrot_token_center_query:
                query_mean = cost_query_tokens.mean(dim=1, keepdim=True)
                cost_query_tokens = cost_query_tokens - query_mean
        return cost_query_tokens, cost_support_tokens

    @staticmethod
    def _weighted_multiscale_value(
        payloads: list[dict[str, torch.Tensor]],
        key: str,
        weights: torch.Tensor,
    ) -> torch.Tensor | None:
        values = [payload.get(key) for payload in payloads]
        if not values or any(not torch.is_tensor(value) for value in values):
            return None
        first = values[0]
        if any(tuple(value.shape) != tuple(first.shape) for value in values):
            return None
        stacked = torch.stack(values, dim=0)
        view_shape = (weights.shape[0],) + (1,) * first.dim()
        return (weights.reshape(view_shape).to(device=stacked.device, dtype=stacked.dtype) * stacked).sum(dim=0)

    def _forward_multiscale_ecot_budget_bank(
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
        del support_tokens, query_tokens, spatial_hw
        if support_weight is not None or query_weight is not None:
            raise ValueError("enable_multiscale_ot does not support explicit query/support weights")
        cache = getattr(self, "_multiscale_ot_cache", None)
        if cache is None or len(cache) < 2:
            raise RuntimeError("Multi-scale OT cache was not populated before ECOT scoring")
        query_scales = cache[0]
        support_scales = cache[1]
        if len(query_scales) != len(support_scales):
            raise ValueError(
                f"Query/support scale count mismatch: {len(query_scales)} vs {len(support_scales)}"
            )
        if len(query_scales) != self.num_multiscale_scales:
            raise ValueError(
                f"Expected {self.num_multiscale_scales} multi-scale token sets, got {len(query_scales)}"
            )

        scale_weights = torch.softmax(self.scale_weights.to(device=flat_cost.device, dtype=flat_cost.dtype), dim=0)
        threshold_per_scale = self.multiscale_transport_cost_thresholds.to(
            device=flat_cost.device,
            dtype=flat_cost.dtype,
        )

        payloads: list[dict[str, torch.Tensor]] = []
        scale_shot_logits = []
        scale_shot_cost = []
        scale_shot_mass = []
        for scale_idx, (query_record, support_record) in enumerate(zip(query_scales, support_scales)):
            if query_record["spatial_hw"] != support_record["spatial_hw"]:
                raise ValueError(
                    "Query/support multi-scale token grids must match for "
                    f"{query_record['name']}: {query_record['spatial_hw']} vs {support_record['spatial_hw']}"
                )
            query_euclidean = query_record["euclidean"].to(device=flat_cost.device, dtype=flat_cost.dtype)
            flat_support_euclidean = support_record["euclidean"].to(
                device=flat_cost.device,
                dtype=flat_cost.dtype,
            )
            expected_support = int(way_num) * int(shot_num)
            if flat_support_euclidean.shape[0] != expected_support:
                raise ValueError(
                    "Multi-scale support token cache does not match Way*Shot: "
                    f"{flat_support_euclidean.shape[0]} vs {expected_support}"
                )
            cost_query_tokens, cost_support_tokens = self._prepare_multiscale_cost_tokens(
                query_euclidean,
                flat_support_euclidean,
                way_num=way_num,
                shot_num=shot_num,
            )
            scale_cost = self._ground_cost(cost_query_tokens, cost_support_tokens)
            scale_cost, _tsw_payload = self._apply_tsw_to_cost(
                scale_cost,
                cost_query_tokens,
                cost_support_tokens,
            )
            scale_support_tokens = flat_support_euclidean.reshape(
                way_num,
                shot_num,
                flat_support_euclidean.shape[-2],
                flat_support_euclidean.shape[-1],
            )
            payload = super()._forward_ecot_budget_bank(
                scale_cost,
                way_num=way_num,
                shot_num=shot_num,
                support_weight=None,
                query_weight=None,
                support_tokens=scale_support_tokens,
                query_tokens=query_euclidean,
                spatial_hw=support_record["spatial_hw"],
            )
            if self.multiscale_per_scale_T:
                threshold = threshold_per_scale[scale_idx]
                shot_logits = self.score_scale * (
                    threshold * payload["shot_transported_mass"] - payload["shot_transport_cost"]
                )
                payload = dict(payload)
                payload["shot_logits"] = shot_logits
                payload["ecot_base_score"] = shot_logits
                if "ecot_score" in payload:
                    payload["ecot_score"] = shot_logits
                if "ecot_budget_scores" in payload and payload["ecot_budget_scores"].shape[-1] == 1:
                    payload["ecot_budget_scores"] = shot_logits.unsqueeze(-1)
            payloads.append(payload)
            scale_shot_logits.append(payload["shot_logits"])
            scale_shot_cost.append(payload["shot_transport_cost"])
            scale_shot_mass.append(payload["shot_transported_mass"])

        def fuse_stack(values: list[torch.Tensor]) -> torch.Tensor:
            stacked = torch.stack(values, dim=0)
            view_shape = (scale_weights.shape[0],) + (1,) * values[0].dim()
            return (scale_weights.reshape(view_shape) * stacked).sum(dim=0)

        fused_shot_logits = fuse_stack(scale_shot_logits)
        fused_shot_cost = fuse_stack(scale_shot_cost)
        fused_shot_mass = fuse_stack(scale_shot_mass)

        raw_tau_shot = None
        fused_budget_scores = self._weighted_multiscale_value(payloads, "ecot_budget_scores", scale_weights)
        fused_cost_bank = self._weighted_multiscale_value(
            payloads,
            "ecot_shot_transport_cost_bank",
            scale_weights,
        )
        fused_mass_bank = self._weighted_multiscale_value(
            payloads,
            "ecot_shot_transported_mass_bank",
            scale_weights,
        )
        if (
            self.episode_controller is not None
            and fused_budget_scores is not None
            and fused_cost_bank is not None
            and fused_mass_bank is not None
        ):
            diagnostics = self._compute_ecot_diagnostics(fused_budget_scores, fused_cost_bank, fused_mass_bank)
            controller_outputs = self.episode_controller(diagnostics)
            raw_tau_shot = controller_outputs.get("raw_tau_shot")

        logits, transport_cost, transport_mass, shot_pool_weights, tau_shot = self._pool_ecot_shot_scores(
            fused_shot_logits,
            fused_shot_cost,
            fused_shot_mass,
            raw_tau_shot,
        )

        fused_payload = dict(payloads[0])
        fused_payload.update(
            {
                "logits": logits,
                "transport_cost": transport_cost,
                "transported_mass": transport_mass,
                "shot_transport_cost": fused_shot_cost,
                "shot_transported_mass": fused_shot_mass,
                "shot_logits": fused_shot_logits,
                "shot_pool_weights": shot_pool_weights,
            }
        )
        if tau_shot is not None:
            fused_payload["ecot_tau_shot"] = tau_shot
        for key in (
            "rho",
            "shot_rho",
            "ecot_base_score",
            "ecot_score",
            "ecot_budget_scores",
            "ecot_shot_transport_cost_bank",
            "ecot_shot_transported_mass_bank",
            "ecot_m2_mass_for_score_bank",
            "ecot_m2_mass_consensus_bank",
            "ecot_aux_loss",
            "ecot_identity_loss",
            "ecot_policy_entropy",
            "ecot_policy_entropy_loss",
        ):
            fused_value = self._weighted_multiscale_value(payloads, key, scale_weights)
            if fused_value is not None:
                fused_payload[key] = fused_value

        fused_payload["multiscale/scale_weights"] = scale_weights
        for scale_idx, query_record in enumerate(query_scales):
            name = str(query_record["name"])
            fused_payload[f"multiscale/scale_weight_{name}"] = scale_weights[scale_idx]
            fused_payload[f"multiscale/score_{name}_mean"] = scale_shot_logits[scale_idx].mean()
            fused_payload[f"multiscale/mass_{name}_mean"] = scale_shot_mass[scale_idx].mean()
            fused_payload[f"multiscale/cost_{name}_mean"] = scale_shot_cost[scale_idx].mean()
        if self.multiscale_per_scale_T:
            for scale_idx, query_record in enumerate(query_scales):
                name = str(query_record["name"])
                fused_payload[f"multiscale/T_{name}"] = threshold_per_scale[scale_idx]
        return fused_payload

    def _transport_match(
        self,
        cost: torch.Tensor,
        rho: torch.Tensor,
        a: torch.Tensor | None = None,
        b: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_query, num_way, query_tokens, class_tokens = cost.shape
        partial_transport_mass = None
        if self.enable_pot_guide and a is None and b is None:
            a, b, pot_guide_diagnostics = self.pot_guide_module(cost, rho)
            self._last_pot_guide_diagnostics = pot_guide_diagnostics
        else:
            if a is None:
                base_a = cost.new_full((num_query, query_tokens), 1.0 / float(query_tokens))
                a = base_a.unsqueeze(1).expand(-1, num_way, -1)
                if self.uses_partial_transport:
                    partial_transport_mass = rho.to(device=cost.device, dtype=cost.dtype)
                elif self.uses_unbalanced_transport:
                    a = a * rho.unsqueeze(-1)
            else:
                if tuple(a.shape) != (num_query, num_way, query_tokens):
                    raise ValueError(
                        "a must have shape (NumQuery, NumPairs, QueryTokens), "
                        f"got {tuple(a.shape)}"
                    )
                a = a.to(device=cost.device, dtype=cost.dtype)

            if b is None:
                base_b = cost.new_full((num_way, class_tokens), 1.0 / float(class_tokens))
                b = base_b.unsqueeze(0).expand(num_query, -1, -1)
                if self.uses_partial_transport:
                    partial_transport_mass = rho.to(device=cost.device, dtype=cost.dtype)
                elif self.uses_unbalanced_transport:
                    b = b * rho.unsqueeze(-1)
            else:
                if tuple(b.shape) != (num_query, num_way, class_tokens):
                    raise ValueError(
                        "b must have shape (NumQuery, NumPairs, SupportTokens), "
                        f"got {tuple(b.shape)}"
                    )
                b = b.to(device=cost.device, dtype=cost.dtype)

        pair_cost = cost.reshape(num_query * num_way, query_tokens, class_tokens)
        pair_a = a.reshape(num_query * num_way, query_tokens)
        pair_b = b.reshape(num_query * num_way, class_tokens)

        if self.uses_partial_transport:
            max_mass = torch.minimum(pair_a.sum(dim=-1), pair_b.sum(dim=-1))
            if partial_transport_mass is None:
                pair_transport_mass = max_mass
            else:
                pair_transport_mass = partial_transport_mass.reshape(num_query * num_way).to(
                    device=cost.device,
                    dtype=cost.dtype,
                )
                pair_transport_mass = torch.minimum(pair_transport_mass, max_mass)
            pair_plan = solve_partial_transport(
                pair_cost,
                pair_a,
                pair_b,
                transport_mass=pair_transport_mass,
                backend=self.partial_backend,
                reg=self.sinkhorn_epsilon,
                max_iter=self.sinkhorn_iterations,
                tol=self.sinkhorn_tolerance,
                exact=self.partial_exact,
                eps=self.eps,
            )
            pair_transport_cost = compute_partial_transport_cost(pair_plan, pair_cost)
            pair_transport_mass = compute_partial_transported_mass(pair_plan)
        elif self.ot_backend == "pot":
            if self.uses_unbalanced_transport:
                pair_plan = sinkhorn_unbalanced_pot(
                    pair_cost,
                    pair_a,
                    pair_b,
                    tau_q=self.tau_q,
                    tau_c=self.tau_c,
                    eps=self.sinkhorn_epsilon,
                    max_iter=self.sinkhorn_iterations,
                    tol=self.sinkhorn_tolerance,
                )
            else:
                pair_plan = sinkhorn_balanced_pot(
                    pair_cost,
                    pair_a,
                    pair_b,
                    eps=self.sinkhorn_epsilon,
                    max_iter=self.sinkhorn_iterations,
                    tol=self.sinkhorn_tolerance,
                )
            pair_transport_cost = compute_transport_cost(pair_plan, pair_cost)
            pair_transport_mass = compute_transported_mass(pair_plan)
        else:
            if self.uses_unbalanced_transport:
                pair_plan = sinkhorn_unbalanced_log(
                    pair_cost,
                    pair_a,
                    pair_b,
                    tau_q=self.tau_q,
                    tau_c=self.tau_c,
                    eps=self.sinkhorn_epsilon,
                    max_iter=self.sinkhorn_iterations,
                    tol=self.sinkhorn_tolerance,
                )
            else:
                pair_plan = sinkhorn_balanced_log(
                    pair_cost,
                    pair_a,
                    pair_b,
                    eps=self.sinkhorn_epsilon,
                    max_iter=self.sinkhorn_iterations,
                    tol=self.sinkhorn_tolerance,
                )
            pair_transport_cost = compute_transport_cost(pair_plan, pair_cost)
            pair_transport_mass = compute_transported_mass(pair_plan)
        plan = pair_plan.reshape(num_query, num_way, query_tokens, class_tokens)
        transport_cost = pair_transport_cost.reshape(num_query, num_way)
        transport_mass = pair_transport_mass.reshape(num_query, num_way)
        return plan, transport_cost, transport_mass

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
        if getattr(self, "enable_multiscale_ot", False):
            return self._forward_multiscale_ecot_budget_bank(
                flat_cost,
                way_num=way_num,
                shot_num=shot_num,
                support_weight=support_weight,
                query_weight=query_weight,
                support_tokens=support_tokens,
                query_tokens=query_tokens,
                spatial_hw=spatial_hw,
            )
        dmuot_payload: dict[str, torch.Tensor] = {}
        pulse_payload: dict[str, torch.Tensor] = {}
        region_uot_payload: dict[str, torch.Tensor] = {}
        token_g_query = None
        token_g_support = None
        cost_for_transport = flat_cost
        if self.enable_pot_guide:
            self._last_pot_guide_diagnostics = None

        if getattr(self, "enable_region_structural_uot", False):
            if query_tokens is None or support_tokens is None or spatial_hw is None:
                raise ValueError(
                    "enable_region_structural_uot requires query_tokens, support_tokens, and spatial_hw"
                )
            cost_for_transport, region_uot_payload = self.region_structural_uot(
                flat_cost=cost_for_transport,
                query_tokens=query_tokens,
                support_tokens=support_tokens,
                way_num=way_num,
                shot_num=shot_num,
                spatial_hw=spatial_hw,
                rho=self._ecot_base_rho_tensor(flat_cost),
            )

        if getattr(self, "enable_pulse_region_uot", False):
            if query_tokens is None or support_tokens is None or spatial_hw is None:
                raise ValueError(
                    "enable_pulse_region_uot requires query_tokens, support_tokens, and spatial_hw"
                )
            query_saliency, support_saliency = self._pulse_saliency_pair(
                query_tokens=query_tokens,
                support_tokens=support_tokens,
                way_num=way_num,
                shot_num=shot_num,
            )
            support_saliency, pulse_consensus_payload = self._apply_pulse_support_consensus(
                support_tokens,
                support_saliency,
            )
            pulse_base_cost = cost_for_transport
            pulse_guided_cost, pulse_query_weight, pulse_support_weight, pulse_payload = (
                self.pulse_region_guidance(
                    flat_cost=pulse_base_cost,
                    query_tokens=query_tokens,
                    support_tokens=support_tokens,
                    query_saliency=query_saliency,
                    support_saliency=support_saliency,
                    way_num=way_num,
                    shot_num=shot_num,
                    spatial_hw=spatial_hw,
                )
            )
            pulse_payload.update(pulse_consensus_payload)
            pulse_strength = self._pulse_region_effective_strength(pulse_base_cost, shot_num=shot_num)
            cost_for_transport = pulse_base_cost + pulse_strength * (pulse_guided_cost - pulse_base_cost)
            if pulse_strength.item() < 1.0:
                q_uniform = pulse_query_weight.new_full(
                    pulse_query_weight.shape,
                    1.0 / float(pulse_query_weight.shape[-1]),
                )
                s_uniform = pulse_support_weight.new_full(
                    pulse_support_weight.shape,
                    1.0 / float(pulse_support_weight.shape[-1]),
                )
                pulse_query_weight = q_uniform + pulse_strength * (pulse_query_weight - q_uniform)
                pulse_support_weight = s_uniform + pulse_strength * (pulse_support_weight - s_uniform)
            pulse_payload["pulse/effective_strength"] = pulse_strength.detach()
            if pulse_strength.item() < 1.0:
                pulse_payload["pulse_guided_cost_matrix"] = cost_for_transport.reshape(
                    flat_cost.shape[0],
                    way_num,
                    shot_num,
                    flat_cost.shape[-2],
                    flat_cost.shape[-1],
                )
            query_weight = self._combine_token_weight(query_weight, pulse_query_weight)
            support_weight = self._combine_token_weight(support_weight, pulse_support_weight)

        if self.token_g_kind != "none":
            if query_tokens is None or support_tokens is None:
                raise ValueError("token_g_kind != 'none' requires query_tokens and support_tokens")
            token_g_query, token_g_support = self._compute_token_g(query_tokens, support_tokens)
            dmuot_payload["token_g_query"] = token_g_query
            dmuot_payload["token_g_support"] = token_g_support
            dmuot_strength = self._dmuot_shot_strength_value(shot_num, flat_cost)
            lambda_cost_effective = dmuot_strength * float(self.lambda_cost)
            dmuot_payload["dmuot_shot_strength"] = dmuot_strength.detach()
            dmuot_payload["lambda_cost_effective"] = lambda_cost_effective.detach()

            flat_support_g = token_g_support.reshape(way_num * shot_num, token_g_support.shape[-1])
            cost_modulator = 1.0 + lambda_cost_effective * (
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
                cost_for_transport = cost_for_transport * cost_modulator
                dmuot_payload["cost_matrix_modulated"] = cost_for_transport.reshape(
                    flat_cost.shape[0],
                    way_num,
                    shot_num,
                    flat_cost.shape[-2],
                    flat_cost.shape[-1],
                )

            base_rho = self._ecot_base_rho_tensor(flat_cost)
            if self.marginal_kind == "discriminative":
                query_prob, support_prob = self._dmuot_marginal_probs(token_g_query, token_g_support)
                query_prob = self._blend_dmuot_prob_with_uniform(query_prob, dmuot_strength)
                support_prob = self._blend_dmuot_prob_with_uniform(support_prob, dmuot_strength)
                if query_weight is not None:
                    query_prob = query_prob * query_weight.to(device=query_prob.device, dtype=query_prob.dtype)
                    query_prob = query_prob / query_prob.sum(dim=-1, keepdim=True).clamp_min(self.eps)
                if support_weight is not None:
                    support_prob = support_prob * support_weight.to(
                        device=support_prob.device,
                        dtype=support_prob.dtype,
                    )
                    support_prob = support_prob / support_prob.sum(dim=-1, keepdim=True).clamp_min(self.eps)
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
        if pulse_payload:
            payload.update(pulse_payload)
        if region_uot_payload:
            payload.update(region_uot_payload)
        if self.enable_pot_guide and self._last_pot_guide_diagnostics is not None:
            payload.update(self._last_pot_guide_diagnostics)
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
        use_context_debug = (
            self._context_debug
            and self._context_debug_count < self._context_debug_max
        )
        if self.enable_context_enrichment:
            self._last_context_diagnostics = None
        if use_context_debug:
            self._context_debug_images = []
            self._context_debug_fmaps = []
            if self.enable_context_enrichment:
                self.context_enrichment._debug_cache = []
        use_prenorm_cache = self.token_g_kind == "token_norm_pre_l2"
        use_multiscale_cache = getattr(self, "enable_multiscale_ot", False)
        use_pulse_cache = getattr(self, "enable_pulse_region_uot", False)
        if use_prenorm_cache:
            self._token_g_prenorm_cache = []
        if use_multiscale_cache:
            self._multiscale_ot_cache = []
        if use_pulse_cache:
            self._pulse_saliency_cache = []
        try:
            outputs = super()._forward_episode(*args, **kwargs)
        finally:
            if use_prenorm_cache:
                self._token_g_prenorm_cache = None
            if use_multiscale_cache:
                self._multiscale_ot_cache = None
            if use_pulse_cache:
                self._pulse_saliency_cache = None
            if use_context_debug:
                self._export_context_debug()
                if self.enable_context_enrichment:
                    self.context_enrichment._debug_cache = None
                self._context_debug_images = []
                self._context_debug_fmaps = []
        if isinstance(outputs, dict):
            if "region_uot_guided_cost_matrix" in outputs and "cost_matrix" in outputs:
                outputs.setdefault("base_cost_matrix", outputs["cost_matrix"])
                outputs["cost_matrix"] = outputs["region_uot_guided_cost_matrix"]
            if "pulse_guided_cost_matrix" in outputs and "cost_matrix" in outputs:
                outputs.setdefault("base_cost_matrix", outputs["cost_matrix"])
                outputs["cost_matrix"] = outputs["pulse_guided_cost_matrix"]
            if self.lambda_cost > 0.0 and "cost_matrix_modulated" in outputs:
                if "base_cost_matrix" not in outputs and "cost_matrix" in outputs:
                    outputs["base_cost_matrix"] = outputs["cost_matrix"]
                outputs["cost_matrix"] = outputs["cost_matrix_modulated"]
            if self._last_dm_diagnostics is not None:
                outputs.update(self._last_dm_diagnostics)
            if self.enable_context_enrichment and self._last_context_diagnostics is not None:
                outputs.update(self._last_context_diagnostics)
            if getattr(self, "enable_structural_augmentation", False) and self._last_struct_diagnostics is not None:
                outputs.update(self._last_struct_diagnostics)
        return outputs

    def _export_context_debug(self) -> None:
        from net.modules.spatial_context_enrichment import (
            export_context_debug_figure,
            export_baseline_debug_figure,
        )

        images = getattr(self, "_context_debug_images", [])
        fmaps = getattr(self, "_context_debug_fmaps", [])

        if self.enable_context_enrichment:
            cache = getattr(self.context_enrichment, "_debug_cache", None)
            if not cache:
                return
        else:
            if not fmaps:
                return
            cache = None

        self._context_debug_count += 1
        ep_idx = self._context_debug_count

        if cache is not None:
            bw = None
            gv = None
            if self.context_enrichment.fusion_mode == "weighted_sum":
                bw = (
                    torch.softmax(self.context_enrichment.branch_weights.detach().float(), dim=0)
                    .cpu()
                    .numpy()
                )
                gv = float(self.context_enrichment.gate_value.detach().float().cpu().item())

            for call_idx, record in enumerate(cache):
                batch_size = record["original"].shape[0]
                flat_images = images[call_idx] if call_idx < len(images) else None
                for img_idx in range(min(batch_size, 8)):
                    inp = None
                    if flat_images is not None and img_idx < flat_images.shape[0]:
                        inp = flat_images[img_idx]
                    path = os.path.join(
                        self._context_debug_dir,
                        f"ep{ep_idx:03d}_call{call_idx}_img{img_idx:03d}.png",
                    )
                    try:
                        export_context_debug_figure(
                            debug_record={
                                "original": record["original"][img_idx : img_idx + 1],
                                "enriched": record["enriched"][img_idx : img_idx + 1],
                                "delta": record["delta"][img_idx : img_idx + 1],
                                "branches": [b[img_idx : img_idx + 1] for b in record["branches"]],
                            },
                            input_image=inp,
                            kernel_sizes=self.context_enrichment.kernel_sizes,
                            image_index=0,
                            save_path=path,
                            branch_weights=bw,
                            gate_value=gv,
                        )
                    except Exception as exc:
                        print(f"[context-debug] failed to export {path}: {exc}")
                        return
            print(
                f"[context-debug] episode {ep_idx}: exported {len(cache)} calls, "
                f"gate={gv:.4f}, dir={self._context_debug_dir}"
            )
        else:
            for call_idx, fmap in enumerate(fmaps):
                batch_size = fmap.shape[0]
                flat_images = images[call_idx] if call_idx < len(images) else None
                for img_idx in range(min(batch_size, 8)):
                    inp = None
                    if flat_images is not None and img_idx < flat_images.shape[0]:
                        inp = flat_images[img_idx]
                    path = os.path.join(
                        self._context_debug_dir,
                        f"ep{ep_idx:03d}_call{call_idx}_img{img_idx:03d}_baseline.png",
                    )
                    try:
                        export_baseline_debug_figure(
                            feature_map=fmap[img_idx : img_idx + 1],
                            input_image=inp,
                            save_path=path,
                        )
                    except Exception as exc:
                        print(f"[context-debug] failed to export {path}: {exc}")
                        return
            print(
                f"[context-debug] episode {ep_idx} (baseline): exported {len(fmaps)} calls, "
                f"dir={self._context_debug_dir}"
            )

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
            "pulse_query_saliency",
            "pulse_query_marginal_weight",
            "pulse_region_cost_matrix",
            "pulse_guided_cost_matrix",
            "region_uot_coarse_plan",
            "region_uot_coarse_cost_matrix",
            "region_uot_guided_cost_matrix",
            "region_uot_coarse_transport_cost",
            "region_uot_coarse_transported_mass",
        ):
            if key in batch_outputs[0]:
                stacked[key] = torch.cat([item[key] for item in batch_outputs], dim=0)
        for key in (
            "pulse_support_saliency",
            "pulse_support_marginal_weight",
        ):
            if key in batch_outputs[0]:
                stacked[key] = torch.stack([item[key] for item in batch_outputs], dim=0)
        for key in (
            "marginal_query_l1_drift",
            "marginal_support_l1_drift",
            "marginal_l1_drift",
            "dmuot_shot_strength",
            "lambda_cost_effective",
            "pot_guide/alpha",
            "pot_guide/temperature",
            "pot_guide/s_mean",
            "pot_guide/pot_sparsity",
            "pot_guide/marginal_q_max",
            "pot_guide/marginal_q_min",
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
        for key in sorted(k for k in batch_outputs[0] if str(k).startswith("multiscale/")):
            values = [item[key] for item in batch_outputs]
            if all(torch.is_tensor(value) for value in values):
                if values[0].dim() == 0:
                    stacked[key] = torch.stack(values).mean()
                else:
                    stacked[key] = torch.stack(values, dim=0).mean(dim=0)
        for prefix in ("context/", "struct/", "pulse/", "region_uot/"):
            for key in sorted(k for k in batch_outputs[0] if str(k).startswith(prefix)):
                values = [item[key] for item in batch_outputs if key in item]
                if values and all(torch.is_tensor(value) for value in values):
                    stacked[key] = torch.stack(values).mean()
        return stacked

__all__ = ["OursM2", "apply_ours_design_defaults", "OURS_ABLATIONS", "normalize_ours_ablation"]
