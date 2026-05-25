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
from net.modules.adaptive_region_uot import AdaptiveRegionUOTGuidance
from net.modules.cost_evidence_marginals import CostEvidenceMarginals, _normalize_evidence_mode
from net.modules.multiscale_tokenizer import MultiScaleTokenizer
from net.modules.mspta import MSPTATokenizer
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
MSPTA_GUIDANCE_SOURCES = frozenset({"image", "feature", "mixed"})


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


def _normalize_mspta_guidance_source(value: str | None) -> str:
    name = "mixed" if value is None else str(value).strip().lower().replace("-", "_")
    aliases = {
        "both": "mixed",
        "input": "image",
        "img": "image",
        "feat": "feature",
    }
    name = aliases.get(name, name)
    if name not in MSPTA_GUIDANCE_SOURCES:
        raise ValueError(
            f"Unsupported mspta_guidance_source: {value}. "
            f"Expected one of {sorted(MSPTA_GUIDANCE_SOURCES)}"
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
        enable_evidence_marginals: bool = False,
        evidence_tau: float = 0.1,
        evidence_tau_marginal: float = 1.0,
        evidence_mode: str = "both",
        evidence_rival_margin: float = 0.5,
        evidence_detach: bool = True,
        **kwargs,
    ) -> None:
        enable_multiscale_ot = _bool_config(kwargs.pop("enable_multiscale_ot", False))
        multiscale_pool_sizes = kwargs.pop("multiscale_pool_sizes", "original,2x2,1x1")
        multiscale_per_scale_T = _bool_config(kwargs.pop("multiscale_per_scale_T", False))
        enable_mspta = _bool_config(kwargs.pop("enable_mspta", False))
        mspta_scales = kwargs.pop("mspta_scales", "1,2,3")
        mspta_mass_mode = str(kwargs.pop("mspta_mass_mode", "compact_area"))
        mspta_normalize = _bool_config(kwargs.pop("mspta_normalize", True))
        mspta_proj_dim = int(kwargs.pop("mspta_proj_dim", 0))
        mspta_learnable_weights = _bool_config(kwargs.pop("mspta_learnable_weights", False))
        mspta_compact_floor = float(kwargs.pop("mspta_compact_floor", 0.05))
        mspta_vertical_suppression = float(kwargs.pop("mspta_vertical_suppression", 1.0))
        mspta_saliency_cost_discount = float(kwargs.pop("mspta_saliency_cost_discount", 0.10))
        mspta_guidance_source = _normalize_mspta_guidance_source(
            kwargs.pop("mspta_guidance_source", "mixed")
        )
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
        region_uot_strength = float(kwargs.pop("region_uot_strength", 0.08))
        region_uot_fgw_alpha = float(kwargs.pop("region_uot_fgw_alpha", 0.35))
        region_uot_sinkhorn_epsilon = float(kwargs.pop("region_uot_sinkhorn_epsilon", 0.08))
        region_uot_fgw_iters = int(kwargs.pop("region_uot_fgw_iters", 4))
        region_uot_sinkhorn_iters = int(kwargs.pop("region_uot_sinkhorn_iters", 40))
        region_uot_topk = int(kwargs.pop("region_uot_topk", 3))
        region_uot_fine_gate_quantile = float(kwargs.pop("region_uot_fine_gate_quantile", 0.35))
        region_uot_min_confidence = float(kwargs.pop("region_uot_min_confidence", 0.10))
        region_uot_importance_temperature = float(kwargs.pop("region_uot_importance_temperature", 0.50))
        enable_adaptive_region_uot = _bool_config(kwargs.pop("enable_adaptive_region_uot", False))
        adaptive_region_num_slots = int(kwargs.pop("adaptive_region_num_slots", 4))
        adaptive_region_context_kernels = kwargs.pop("adaptive_region_context_kernels", "1,3,5")
        adaptive_region_cost_discount = float(kwargs.pop("adaptive_region_cost_discount", 0.12))
        adaptive_region_mass_mix = float(kwargs.pop("adaptive_region_mass_mix", 0.60))
        adaptive_region_sinkhorn_epsilon = float(kwargs.pop("adaptive_region_sinkhorn_epsilon", 0.08))
        adaptive_region_sinkhorn_iters = int(kwargs.pop("adaptive_region_sinkhorn_iters", 30))
        adaptive_region_fine_gate_quantile = float(kwargs.pop("adaptive_region_fine_gate_quantile", 0.40))
        adaptive_region_temperature_min = float(kwargs.pop("adaptive_region_temperature_min", 0.35))
        adaptive_region_temperature_max = float(kwargs.pop("adaptive_region_temperature_max", 1.25))
        adaptive_region_init_gate = float(kwargs.pop("adaptive_region_init_gate", 0.05))
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
        pulse_evidence_score = _bool_config(kwargs.pop("pulse_evidence_score", False))
        pulse_evidence_mass_weight = float(kwargs.pop("pulse_evidence_mass_weight", 1.0))
        pulse_evidence_cost_weight = float(kwargs.pop("pulse_evidence_cost_weight", 1.0))
        pulse_background_penalty = float(kwargs.pop("pulse_background_penalty", 0.25))
        enable_discriminative_uot = _bool_config(kwargs.pop("enable_discriminative_uot", False))
        discriminative_uot_tau = float(kwargs.pop("discriminative_uot_tau", 0.05))
        discriminative_uot_margin = float(kwargs.pop("discriminative_uot_margin", 0.02))
        discriminative_uot_mix = float(kwargs.pop("discriminative_uot_mix", 1.0))
        discriminative_uot_background_penalty = float(kwargs.pop("discriminative_uot_background_penalty", 0.25))
        discriminative_uot_mass_weight = float(kwargs.pop("discriminative_uot_mass_weight", 1.0))
        discriminative_uot_cost_weight = float(kwargs.pop("discriminative_uot_cost_weight", 1.0))
        pulse_discriminative_evidence = _bool_config(kwargs.pop("pulse_discriminative_evidence", False))
        pulse_discriminative_tau = float(kwargs.pop("pulse_discriminative_tau", 0.05))
        pulse_discriminative_margin = float(kwargs.pop("pulse_discriminative_margin", 0.02))
        pulse_discriminative_mix = float(kwargs.pop("pulse_discriminative_mix", 1.0))
        enable_label_ot = _bool_config(
            kwargs.pop("enable_label_ot", kwargs.pop("enable_transductive_label_ot", False))
        )
        label_ot_epsilon = float(kwargs.pop("label_ot_epsilon", 0.75))
        label_ot_iterations = int(kwargs.pop("label_ot_iterations", 30))
        label_ot_mix = float(kwargs.pop("label_ot_mix", 1.0))
        label_ot_min_queries_per_class = int(kwargs.pop("label_ot_min_queries_per_class", 2))
        label_ot_min_column_imbalance = float(kwargs.pop("label_ot_min_column_imbalance", 0.0))
        label_ot_max_bias = float(kwargs.pop("label_ot_max_bias", 2.0))
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
        if enable_mspta:
            if self.ours_ablation == "gap":
                raise ValueError("enable_mspta is not supported with ours_ablation='gap'")
            if mspta_proj_dim not in {0, int(kwargs.get("token_dim", mspta_proj_dim))}:
                raise ValueError("mspta_proj_dim must be 0 or match hrot_token_dim; use hrot_token_dim for Ours-Final")
            if enable_multiscale_ot:
                raise ValueError("enable_mspta is not supported with enable_multiscale_ot")
            if self.token_g_kind != "none":
                raise ValueError("enable_mspta currently requires token_g_kind='none'")
            if _bool_config(kwargs.get("use_cata", False)):
                raise ValueError("enable_mspta is not supported with CATA token aggregation")
            if _bool_config(kwargs.get("hrot_amp_marginals", False)):
                raise ValueError("enable_mspta is not supported with hrot_amp_marginals")
            if _bool_config(kwargs.get("ecot_episode_feature_normalize", False)):
                raise ValueError("enable_mspta is not supported with ecot_episode_feature_normalize")
            if _bool_config(kwargs.get("pre_transport_shot_pool", False)):
                raise ValueError("enable_mspta is not supported with pre_transport_shot_pool")
            if enable_structural_augmentation:
                raise ValueError("enable_mspta is not supported with enable_structural_augmentation")
            if enable_region_structural_uot:
                raise ValueError("enable_mspta is not supported with enable_region_structural_uot")
            if enable_adaptive_region_uot:
                raise ValueError("enable_mspta is not supported with enable_adaptive_region_uot")
            if enable_pulse_region_uot:
                raise ValueError("enable_mspta is not supported with enable_pulse_region_uot")
            if enable_pot_guide:
                raise ValueError("enable_mspta is not supported with enable_pot_guide")
            if not 0.0 <= mspta_saliency_cost_discount < 1.0:
                raise ValueError("mspta_saliency_cost_discount must be in [0, 1)")
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
        if enable_adaptive_region_uot:
            if self.ours_ablation == "gap":
                raise ValueError("enable_adaptive_region_uot is not supported with ours_ablation='gap'")
            if enable_multiscale_ot:
                raise ValueError("enable_adaptive_region_uot is not supported with enable_multiscale_ot")
            if enable_region_structural_uot:
                raise ValueError("enable_adaptive_region_uot is not supported with enable_region_structural_uot")
            if enable_pulse_region_uot:
                raise ValueError("enable_adaptive_region_uot is not supported with enable_pulse_region_uot")
            if _bool_config(kwargs.get("pre_transport_shot_pool", False)):
                raise ValueError("enable_adaptive_region_uot is not supported with pre_transport_shot_pool")
        if enable_pulse_region_uot:
            if self.ours_ablation == "gap":
                raise ValueError("enable_pulse_region_uot is not supported with ours_ablation='gap'")
            if enable_multiscale_ot:
                raise ValueError("enable_pulse_region_uot is not supported with enable_multiscale_ot")
            if _bool_config(kwargs.get("pre_transport_shot_pool", False)):
                raise ValueError("enable_pulse_region_uot is not supported with pre_transport_shot_pool")
        if _bool_config(enable_evidence_marginals):
            if self.marginal_kind == "discriminative":
                raise ValueError(
                    "enable_evidence_marginals cannot be combined with marginal_kind='discriminative'; "
                    "evidence marginals replace the DMUOT discriminative marginal system"
                )
            if enable_multiscale_ot:
                raise ValueError("enable_evidence_marginals is not supported with enable_multiscale_ot")
        super().__init__(
            *args,
            rho=float(kwargs["ecot_base_rho"]),
            transport_mode=kwargs["ecot_transport_mode"],
            **kwargs,
        )
        self.enable_pulse_region_uot = bool(enable_pulse_region_uot)
        self.enable_region_structural_uot = bool(enable_region_structural_uot)
        self.enable_adaptive_region_uot = bool(enable_adaptive_region_uot)
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
        self.pulse_evidence_score = bool(pulse_evidence_score)
        self.pulse_evidence_mass_weight = float(pulse_evidence_mass_weight)
        self.pulse_evidence_cost_weight = float(pulse_evidence_cost_weight)
        self.pulse_background_penalty = float(pulse_background_penalty)
        self.enable_discriminative_uot = bool(enable_discriminative_uot)
        self.discriminative_uot_tau = float(discriminative_uot_tau)
        self.discriminative_uot_margin = float(discriminative_uot_margin)
        self.discriminative_uot_mix = float(discriminative_uot_mix)
        self.discriminative_uot_background_penalty = float(discriminative_uot_background_penalty)
        self.discriminative_uot_mass_weight = float(discriminative_uot_mass_weight)
        self.discriminative_uot_cost_weight = float(discriminative_uot_cost_weight)
        self.pulse_discriminative_evidence = bool(pulse_discriminative_evidence)
        self.pulse_discriminative_tau = float(pulse_discriminative_tau)
        self.pulse_discriminative_margin = float(pulse_discriminative_margin)
        self.pulse_discriminative_mix = float(pulse_discriminative_mix)
        self.enable_label_ot = bool(enable_label_ot)
        self.label_ot_epsilon = float(label_ot_epsilon)
        self.label_ot_iterations = int(label_ot_iterations)
        self.label_ot_mix = float(label_ot_mix)
        self.label_ot_min_queries_per_class = int(label_ot_min_queries_per_class)
        self.label_ot_min_column_imbalance = float(label_ot_min_column_imbalance)
        self.label_ot_max_bias = float(label_ot_max_bias)
        for name, value in (
            ("pulse_evidence_mass_weight", self.pulse_evidence_mass_weight),
            ("pulse_evidence_cost_weight", self.pulse_evidence_cost_weight),
            ("pulse_background_penalty", self.pulse_background_penalty),
            ("discriminative_uot_margin", self.discriminative_uot_margin),
            ("discriminative_uot_background_penalty", self.discriminative_uot_background_penalty),
            ("discriminative_uot_mass_weight", self.discriminative_uot_mass_weight),
            ("discriminative_uot_cost_weight", self.discriminative_uot_cost_weight),
            ("pulse_discriminative_margin", self.pulse_discriminative_margin),
        ):
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.discriminative_uot_tau <= 0.0:
            raise ValueError("discriminative_uot_tau must be positive")
        if not 0.0 <= self.discriminative_uot_mix <= 1.0:
            raise ValueError("discriminative_uot_mix must be in [0, 1]")
        if self.pulse_discriminative_tau <= 0.0:
            raise ValueError("pulse_discriminative_tau must be positive")
        if not 0.0 <= self.pulse_discriminative_mix <= 1.0:
            raise ValueError("pulse_discriminative_mix must be in [0, 1]")
        if self.label_ot_epsilon <= 0.0:
            raise ValueError("label_ot_epsilon must be positive")
        if self.label_ot_iterations < 1:
            raise ValueError("label_ot_iterations must be positive")
        if not 0.0 <= self.label_ot_mix <= 1.0:
            raise ValueError("label_ot_mix must be in [0, 1]")
        if self.label_ot_min_queries_per_class < 1:
            raise ValueError("label_ot_min_queries_per_class must be positive")
        if self.label_ot_min_column_imbalance < 0.0:
            raise ValueError("label_ot_min_column_imbalance must be non-negative")
        if self.label_ot_max_bias < 0.0:
            raise ValueError("label_ot_max_bias must be non-negative")
        if self.enable_pulse_region_uot:
            self.pulse_region_guidance = PulseRegionGuidance(
                kernel_size=pulse_region_kernel_size,
                region_cost_weight=pulse_region_cost_weight,
                saliency_mass_mix=pulse_saliency_mass_mix,
                saliency_cost_discount=pulse_saliency_cost_discount,
                ground_cost=self.ground_cost,
                eps=self.eps,
            )
        if self.enable_adaptive_region_uot:
            self.adaptive_region_uot = AdaptiveRegionUOTGuidance(
                token_dim=self.token_dim,
                num_slots=adaptive_region_num_slots,
                context_kernels=adaptive_region_context_kernels,
                cost_discount=adaptive_region_cost_discount,
                mass_mix=adaptive_region_mass_mix,
                sinkhorn_epsilon=adaptive_region_sinkhorn_epsilon,
                sinkhorn_iters=adaptive_region_sinkhorn_iters,
                sinkhorn_tol=self.sinkhorn_tolerance,
                tau=float(self.tau_q),
                fine_gate_quantile=adaptive_region_fine_gate_quantile,
                temperature_min=adaptive_region_temperature_min,
                temperature_max=adaptive_region_temperature_max,
                init_gate=adaptive_region_init_gate,
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
                topk=region_uot_topk,
                fine_gate_quantile=region_uot_fine_gate_quantile,
                min_confidence=region_uot_min_confidence,
                importance_temperature=region_uot_importance_temperature,
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
        self.enable_mspta = bool(enable_mspta)
        self._mspta_cache: list[dict[str, torch.Tensor | tuple[str, ...] | tuple[int, ...]]] | None = None
        if self.enable_mspta:
            self.mspta_tokenizer = MSPTATokenizer(
                scales=mspta_scales,
                mass_mode=mspta_mass_mode,
                compact_floor=mspta_compact_floor,
                vertical_suppression=mspta_vertical_suppression,
            )
            self.mspta_normalize = bool(mspta_normalize)
            self.mspta_proj_dim = int(mspta_proj_dim)
            self.mspta_guidance_source = str(mspta_guidance_source)
            self.mspta_saliency_cost_discount = float(mspta_saliency_cost_discount)
            self.mspta_learnable_weights = bool(mspta_learnable_weights)
            if self.mspta_learnable_weights:
                self.mspta_scale_logits = torch.nn.Parameter(
                    torch.zeros(len(self.mspta_tokenizer.scale_names), dtype=torch.float32)
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
        self.enable_evidence_marginals = _bool_config(enable_evidence_marginals)
        if self.enable_evidence_marginals:
            if self.ours_ablation == "gap":
                raise ValueError("enable_evidence_marginals is not supported with ours_ablation='gap'")
            if _bool_config(kwargs.get("pre_transport_shot_pool", False)):
                raise ValueError("enable_evidence_marginals is not supported with pre_transport_shot_pool")
            self.evidence_marginal_module = CostEvidenceMarginals(
                tau_evidence=float(evidence_tau),
                tau_marginal=float(evidence_tau_marginal),
                mode=str(evidence_mode),
                rival_margin=float(evidence_rival_margin),
                detach_cost=bool(evidence_detach),
                eps=self.eps,
            )
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

    def _pulse_region_evidence_mask(
        self,
        saliency: torch.Tensor,
        spatial_hw: tuple[int, int],
    ) -> torch.Tensor:
        """Convert token saliency into a soft connected pulse-region evidence mask."""
        if saliency.dim() < 2:
            raise ValueError(f"saliency must have a token dimension, got {tuple(saliency.shape)}")
        original_shape = saliency.shape
        token_count = int(original_shape[-1])
        height, width = int(spatial_hw[0]), int(spatial_hw[1])
        flat = torch.nan_to_num(
            saliency.detach(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clamp_min(0.0).reshape(-1, token_count)
        values_min = flat.amin(dim=-1, keepdim=True)
        values_max = flat.amax(dim=-1, keepdim=True)
        denom = (values_max - values_min).clamp_min(self.eps)
        base = (flat - values_min) / denom
        base = torch.where(
            (values_max - values_min) > self.eps,
            base,
            torch.ones_like(base),
        )
        if token_count != height * width:
            return base.reshape(original_shape).to(device=saliency.device, dtype=saliency.dtype)

        evidence_map = base.reshape(-1, 1, height, width)
        kernel = int(getattr(getattr(self, "pulse_region_guidance", None), "kernel_size", 3))
        if kernel > 1:
            padding = kernel // 2
            expanded = F.max_pool2d(evidence_map, kernel_size=kernel, stride=1, padding=padding)
            regional = F.avg_pool2d(
                expanded,
                kernel_size=kernel,
                stride=1,
                padding=padding,
                count_include_pad=False,
            )
            evidence_map = 0.5 * evidence_map + 0.5 * regional
        evidence = evidence_map.flatten(1)
        peak = evidence.amax(dim=-1, keepdim=True)
        evidence = torch.where(peak > self.eps, evidence / peak.clamp_min(self.eps), torch.ones_like(evidence))
        return evidence.clamp(0.0, 1.0).reshape(original_shape).to(device=saliency.device, dtype=saliency.dtype)

    def _pulse_discriminative_pair_evidence(
        self,
        *,
        flat_cost: torch.Tensor,
        query_evidence: torch.Tensor,
        support_evidence: torch.Tensor,
        way_num: int,
        shot_num: int,
        strength: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if flat_cost.dim() != 4:
            raise ValueError(f"flat_cost must have shape (Nq, Way*Shot, Lq, Ls), got {tuple(flat_cost.shape)}")
        num_query, num_pairs, query_len, support_len = flat_cost.shape
        if num_pairs != int(way_num) * int(shot_num):
            raise ValueError(f"flat_cost pair dimension {num_pairs} does not match way*shot={way_num * shot_num}")
        if tuple(query_evidence.shape) != (num_query, query_len):
            raise ValueError(
                "query_evidence must have shape (NumQuery, QueryTokens), "
                f"got {tuple(query_evidence.shape)}"
            )
        if tuple(support_evidence.shape) != (int(way_num), int(shot_num), support_len):
            raise ValueError(
                "support_evidence must have shape (Way, Shot, SupportTokens), "
                f"got {tuple(support_evidence.shape)}"
            )

        query_evidence = query_evidence.to(device=flat_cost.device, dtype=flat_cost.dtype).clamp(0.0, 1.0)
        support_evidence = support_evidence.to(device=flat_cost.device, dtype=flat_cost.dtype).clamp(0.0, 1.0)
        pulse_pair = torch.sqrt(
            (
                query_evidence[:, None, None, :, None]
                * support_evidence[None, :, :, None, :]
            ).clamp_min(0.0)
        )

        if int(way_num) <= 1 or float(getattr(self, "pulse_discriminative_mix", 1.0)) <= 0.0:
            rival_gate = torch.ones_like(pulse_pair)
            rival_advantage = torch.zeros_like(pulse_pair)
        else:
            cost_view = flat_cost.reshape(num_query, int(way_num), int(shot_num), query_len, support_len)
            tau = max(float(self.pulse_discriminative_tau), float(self.eps))
            margin = flat_cost.new_tensor(float(self.pulse_discriminative_margin))
            gates = []
            advantages = []
            all_classes = torch.arange(int(way_num), device=flat_cost.device)
            for class_idx in range(int(way_num)):
                rival_mask = all_classes != int(class_idx)
                rival_cost = cost_view[:, rival_mask]
                rival_cost = rival_cost.permute(0, 3, 1, 2, 4).reshape(num_query, query_len, -1)
                rival_count = max(int(rival_cost.shape[-1]), 1)
                rival_softmin = -tau * (
                    torch.logsumexp(-rival_cost / tau, dim=-1)
                    - math.log(float(rival_count))
                )
                advantage = rival_softmin[:, None, :, None] - cost_view[:, class_idx] - margin
                gate = torch.sigmoid(advantage / tau)
                advantages.append(advantage)
                gates.append(gate)
            rival_gate = torch.stack(gates, dim=1)
            rival_advantage = torch.stack(advantages, dim=1)

        strength_value = strength.to(device=flat_cost.device, dtype=flat_cost.dtype)
        discriminative_mix = flat_cost.new_tensor(float(self.pulse_discriminative_mix))
        effective_gate = 1.0 + strength_value * discriminative_mix * (rival_gate - 1.0)
        pair_evidence = (pulse_pair * effective_gate).clamp(0.0, 1.0)
        payload = {
            "pulse_discriminative_gate": effective_gate.detach(),
            "pulse_rival_advantage": rival_advantage.detach(),
            "pulse/discriminative_enabled": flat_cost.new_tensor(1.0),
            "pulse/discriminative_tau": flat_cost.new_tensor(float(self.pulse_discriminative_tau)),
            "pulse/discriminative_margin": flat_cost.new_tensor(float(self.pulse_discriminative_margin)),
            "pulse/discriminative_mix": discriminative_mix.detach(),
            "pulse/discriminative_gate_mean": effective_gate.mean().detach(),
            "pulse/discriminative_gate_low_share": (effective_gate < 0.5).to(flat_cost.dtype).mean().detach(),
            "pulse/rival_advantage_mean": rival_advantage.mean().detach(),
        }
        return pair_evidence, payload

    def _origin_discriminative_pair_evidence(
        self,
        *,
        flat_cost: torch.Tensor,
        way_num: int,
        shot_num: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if flat_cost.dim() != 4:
            raise ValueError(f"flat_cost must have shape (Nq, Way*Shot, Lq, Ls), got {tuple(flat_cost.shape)}")
        num_query, num_pairs, query_len, support_len = flat_cost.shape
        if num_pairs != int(way_num) * int(shot_num):
            raise ValueError(f"flat_cost pair dimension {num_pairs} does not match way*shot={way_num * shot_num}")

        pair_prior = torch.ones(
            num_query,
            int(way_num),
            int(shot_num),
            query_len,
            support_len,
            device=flat_cost.device,
            dtype=flat_cost.dtype,
        )
        if int(way_num) <= 1 or float(self.discriminative_uot_mix) <= 0.0:
            rival_gate = torch.ones_like(pair_prior)
            rival_advantage = torch.zeros_like(pair_prior)
        else:
            cost_view = flat_cost.reshape(num_query, int(way_num), int(shot_num), query_len, support_len)
            tau = max(float(self.discriminative_uot_tau), float(self.eps))
            margin = flat_cost.new_tensor(float(self.discriminative_uot_margin))
            gates = []
            advantages = []
            all_classes = torch.arange(int(way_num), device=flat_cost.device)
            for class_idx in range(int(way_num)):
                rival_mask = all_classes != int(class_idx)
                rival_cost = cost_view[:, rival_mask]
                rival_cost = rival_cost.permute(0, 3, 1, 2, 4).reshape(num_query, query_len, -1)
                rival_count = max(int(rival_cost.shape[-1]), 1)
                rival_softmin = -tau * (
                    torch.logsumexp(-rival_cost / tau, dim=-1)
                    - math.log(float(rival_count))
                )
                advantage = rival_softmin[:, None, :, None] - cost_view[:, class_idx] - margin
                gate = torch.sigmoid(advantage / tau)
                advantages.append(advantage)
                gates.append(gate)
            rival_gate = torch.stack(gates, dim=1)
            rival_advantage = torch.stack(advantages, dim=1)

        mix = flat_cost.new_tensor(float(self.discriminative_uot_mix))
        effective_gate = 1.0 + mix * (rival_gate - 1.0)
        pair_evidence = (pair_prior * effective_gate).clamp(0.0, 1.0)
        payload = {
            "discriminative_uot_gate": effective_gate.detach(),
            "discriminative_uot_rival_advantage": rival_advantage.detach(),
            "discriminative_uot/enabled": flat_cost.new_tensor(1.0),
            "discriminative_uot/tau": flat_cost.new_tensor(float(self.discriminative_uot_tau)),
            "discriminative_uot/margin": flat_cost.new_tensor(float(self.discriminative_uot_margin)),
            "discriminative_uot/mix": mix.detach(),
            "discriminative_uot/background_penalty": flat_cost.new_tensor(
                float(self.discriminative_uot_background_penalty)
            ),
            "discriminative_uot/gate_mean": effective_gate.mean().detach(),
            "discriminative_uot/gate_low_share": (effective_gate < 0.5).to(flat_cost.dtype).mean().detach(),
            "discriminative_uot/rival_advantage_mean": rival_advantage.mean().detach(),
        }
        return pair_evidence, payload

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

    def _mspta_scale_multipliers(self, reference: torch.Tensor) -> torch.Tensor:
        scale_names = getattr(getattr(self, "mspta_tokenizer", None), "scale_names", ())
        logits = getattr(self, "mspta_scale_logits", None)
        if logits is None:
            return reference.new_ones((len(scale_names),))
        weights = torch.softmax(logits.to(device=reference.device, dtype=reference.dtype), dim=0)
        return weights * float(len(scale_names))

    @staticmethod
    def _normalize_spatial_guidance(values: torch.Tensor, eps: float) -> torch.Tensor:
        values = values.detach().float()
        values = values - values.amin(dim=(-2, -1), keepdim=True)
        denom = values.amax(dim=(-2, -1), keepdim=True).clamp_min(float(eps))
        return values / denom

    @staticmethod
    def _pulse_guidance_kernel_size(height: int, width: int) -> int:
        base = max(5, min(int(height), int(width)) // 10)
        if base % 2 == 0:
            base += 1
        return min(base, 31)

    def _image_pulse_guidance(
        self,
        images: torch.Tensor,
        spatial_hw: tuple[int, int],
    ) -> torch.Tensor:
        image = images.detach().float()
        energy = image.abs().mean(dim=1, keepdim=True)
        color_change = image.std(dim=1, keepdim=True, unbiased=False)
        base = self._normalize_spatial_guidance(0.7 * energy + 0.3 * color_change, self.eps)
        kernel_size = self._pulse_guidance_kernel_size(int(base.shape[-2]), int(base.shape[-1]))
        local_mean = F.avg_pool2d(
            base,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            count_include_pad=False,
        )
        local_deviation = (base - local_mean).abs()
        dx = F.pad(base[..., :, 1:] - base[..., :, :-1], (0, 1, 0, 0))
        dy = F.pad(base[..., 1:, :] - base[..., :-1, :], (0, 0, 0, 1))
        gradient = torch.sqrt(dx.pow(2) + dy.pow(2) + float(self.eps))
        local_anomaly = self._normalize_spatial_guidance(local_deviation + 0.5 * gradient, self.eps)
        column_profile = local_anomaly.mean(dim=2, keepdim=True)
        compact = (local_anomaly - column_profile).clamp_min(0.0)
        compact = self._normalize_spatial_guidance(compact, self.eps)
        pooled_avg = F.adaptive_avg_pool2d(compact, output_size=spatial_hw)
        pooled_max = F.adaptive_max_pool2d(compact, output_size=spatial_hw)
        return self._normalize_spatial_guidance(0.5 * pooled_avg + 0.5 * pooled_max, self.eps)

    def _feature_pulse_guidance(self, feature_map: torch.Tensor) -> torch.Tensor:
        feature_energy = feature_map.detach().float().norm(dim=1, keepdim=True)
        feature_norm = self._normalize_spatial_guidance(feature_energy, self.eps)
        local_mean = F.avg_pool2d(
            feature_norm,
            kernel_size=3,
            stride=1,
            padding=1,
            count_include_pad=False,
        )
        feature_contrast = self._normalize_spatial_guidance((feature_norm - local_mean).abs(), self.eps)
        column_profile = feature_contrast.mean(dim=2, keepdim=True)
        return self._normalize_spatial_guidance((feature_contrast - column_profile).clamp_min(0.0), self.eps)

    def _build_mspta_guidance_map(
        self,
        images: torch.Tensor,
        feature_map: torch.Tensor,
    ) -> torch.Tensor:
        source = getattr(self, "mspta_guidance_source", "mixed")
        feature_guidance = self._feature_pulse_guidance(feature_map)
        if source == "feature":
            return feature_guidance.to(device=feature_map.device, dtype=feature_map.dtype)

        image_guidance = self._image_pulse_guidance(images, feature_map.shape[-2:])
        if source == "image":
            return image_guidance.to(device=feature_map.device, dtype=feature_map.dtype)
        guidance = 0.65 * image_guidance + 0.35 * feature_guidance
        return guidance.to(device=feature_map.device, dtype=feature_map.dtype)

    def _encode_mspta_images(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
        feature_map = self.encode(images)
        base_spatial_hw = (feature_map.shape[-2], feature_map.shape[-1])
        guidance_map = self._build_mspta_guidance_map(images, feature_map)
        scale_records = self.mspta_tokenizer(feature_map, guidance_map=guidance_map)
        if not scale_records:
            raise RuntimeError("MSPTA tokenizer returned no scales")

        scale_multipliers = self._mspta_scale_multipliers(feature_map)
        euclidean_chunks = []
        hyperbolic_chunks = []
        weight_chunks = []
        scale_lengths = []
        scale_names = []
        for scale_idx, record in enumerate(scale_records):
            projected = self._project_backbone_tokens(record.tokens)
            euclidean_tokens = (
                F.normalize(projected, p=2, dim=-1, eps=self.eps)
                if self.mspta_normalize
                else projected
            )
            ball = self._build_ball(projected)
            hyperbolic_tokens = safe_project_to_ball(projected * self.projection_scale, ball)
            multiplier = scale_multipliers[scale_idx]
            euclidean_tokens = euclidean_tokens * multiplier
            euclidean_chunks.append(euclidean_tokens)
            hyperbolic_chunks.append(hyperbolic_tokens)
            weight_chunks.append(record.weights.to(device=euclidean_tokens.device, dtype=euclidean_tokens.dtype))
            scale_lengths.append(int(record.tokens.shape[-2]))
            scale_names.append(str(record.name))

        euclidean = torch.cat(euclidean_chunks, dim=-2)
        hyperbolic = torch.cat(hyperbolic_chunks, dim=-2)
        token_weight = torch.cat(weight_chunks, dim=-1)
        token_weight = token_weight / token_weight.sum(dim=-1, keepdim=True).clamp_min(self.eps)

        cache = getattr(self, "_mspta_cache", None)
        if cache is not None:
            cache.append(
                {
                    "weights": token_weight,
                    "scale_multipliers": scale_multipliers,
                    "scale_lengths": tuple(scale_lengths),
                    "scale_names": tuple(scale_names),
                }
            )
        return euclidean, hyperbolic, base_spatial_hw

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
        if getattr(self, "enable_mspta", False):
            return self._encode_mspta_images(images)
        if getattr(self, "enable_multiscale_ot", False):
            return self._encode_multiscale_images(images)

        need_local = (
            self.token_g_kind == "token_norm_pre_l2"
            or self.uses_ours_gap_control
            or getattr(self, "enable_structural_augmentation", False)
            or getattr(self, "enable_pulse_region_uot", False)
            or getattr(self, "enable_adaptive_region_uot", False)
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
            or getattr(self, "enable_adaptive_region_uot", False)
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
            or getattr(self, "enable_adaptive_region_uot", False)
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

    def _mspta_weights_for_ecot(
        self,
        flat_cost: torch.Tensor,
        *,
        way_num: int,
        shot_num: int,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        cache = getattr(self, "_mspta_cache", None)
        if cache is None or len(cache) < 2:
            raise RuntimeError("MSPTA cache was not populated before ECOT scoring")
        query_record = cache[0]
        support_record = cache[1]
        query_weight = query_record["weights"]
        support_weight = support_record["weights"]
        if not torch.is_tensor(query_weight) or not torch.is_tensor(support_weight):
            raise RuntimeError("MSPTA cache is missing query/support token weights")

        num_query, num_pairs, query_len, support_len = flat_cost.shape
        expected_support = int(way_num) * int(shot_num)
        if tuple(query_weight.shape) != (num_query, query_len):
            raise ValueError(
                "MSPTA query weights do not match ECOT cost shape: "
                f"{tuple(query_weight.shape)} vs {(num_query, query_len)}"
            )
        if tuple(support_weight.shape) != (expected_support, support_len):
            raise ValueError(
                "MSPTA support weights do not match ECOT cost shape: "
                f"{tuple(support_weight.shape)} vs {(expected_support, support_len)}"
            )
        if num_pairs != expected_support:
            raise ValueError(f"flat_cost pair dimension {num_pairs} does not match Way*Shot={expected_support}")

        query_weight = query_weight.to(device=flat_cost.device, dtype=flat_cost.dtype)
        support_weight = support_weight.to(device=flat_cost.device, dtype=flat_cost.dtype)
        support_weight = support_weight.reshape(way_num, shot_num, support_len)

        scale_lengths = query_record.get("scale_lengths", ())
        scale_names = query_record.get("scale_names", ())
        scale_multipliers = query_record.get("scale_multipliers", None)
        if not torch.is_tensor(scale_multipliers):
            scale_multipliers = flat_cost.new_ones((len(scale_names),))
        else:
            scale_multipliers = scale_multipliers.to(device=flat_cost.device, dtype=flat_cost.dtype)

        payload: dict[str, torch.Tensor] = {
            "mspta/token_count": flat_cost.new_tensor(float(query_len)),
            "mspta/query_weight_sum": query_weight.sum(dim=-1).mean(),
            "mspta/support_weight_sum": support_weight.sum(dim=-1).mean(),
            "mspta/scale_multipliers": scale_multipliers,
        }
        for idx, name in enumerate(scale_names):
            length = int(scale_lengths[idx])
            start = sum(int(item) for item in scale_lengths[:idx])
            stop = start + length
            payload[f"mspta/token_count_{name}"] = flat_cost.new_tensor(float(length))
            payload[f"mspta/query_weight_share_{name}"] = query_weight[:, start:stop].sum(dim=-1).mean()
            payload[f"mspta/support_weight_share_{name}"] = support_weight[..., start:stop].sum(dim=-1).mean()
            payload[f"mspta/query_weight_mean_{name}"] = query_weight[:, start:stop].mean()
            payload[f"mspta/support_weight_mean_{name}"] = support_weight[..., start:stop].mean()
            payload[f"mspta/scale_multiplier_{name}"] = scale_multipliers[idx]
        return query_weight, support_weight, payload

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
        discriminative_payload: dict[str, torch.Tensor] = {}
        region_uot_payload: dict[str, torch.Tensor] = {}
        adaptive_region_payload: dict[str, torch.Tensor] = {}
        token_g_query = None
        token_g_support = None
        cost_for_transport = flat_cost
        score_query_evidence = None
        score_support_evidence = None
        score_pair_evidence = None
        score_evidence_mass_weight = float(self.pulse_evidence_mass_weight)
        score_evidence_cost_weight = float(self.pulse_evidence_cost_weight)
        score_background_penalty = float(self.pulse_background_penalty)
        mspta_payload: dict[str, torch.Tensor] = {}
        if getattr(self, "enable_mspta", False):
            if query_weight is not None or support_weight is not None:
                raise ValueError("enable_mspta cannot be combined with explicit query/support weights")
            query_weight, support_weight, mspta_payload = self._mspta_weights_for_ecot(
                flat_cost,
                way_num=way_num,
                shot_num=shot_num,
            )
            discount = float(getattr(self, "mspta_saliency_cost_discount", 0.0))
            if discount > 0.0:
                support_weight_flat = support_weight.reshape(way_num * shot_num, support_weight.shape[-1])
                q_peak = query_weight / query_weight.amax(dim=-1, keepdim=True).clamp_min(self.eps)
                s_peak = support_weight_flat / support_weight_flat.amax(dim=-1, keepdim=True).clamp_min(self.eps)
                saliency_pair = torch.sqrt(
                    (q_peak[:, None, :, None] * s_peak[None, :, None, :]).clamp_min(0.0)
                )
                cost_for_transport = cost_for_transport * (1.0 - discount * saliency_pair)
                mspta_payload["mspta_guided_cost_matrix"] = cost_for_transport.reshape(
                    flat_cost.shape[0],
                    way_num,
                    shot_num,
                    flat_cost.shape[-2],
                    flat_cost.shape[-1],
                )
                mspta_payload["mspta/saliency_cost_discount"] = flat_cost.new_tensor(discount)
                mspta_payload["mspta/cost_delta_ratio"] = (
                    (cost_for_transport.detach() - flat_cost.detach()).abs().mean()
                    / flat_cost.detach().abs().mean().clamp_min(self.eps)
                )
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

        if getattr(self, "enable_adaptive_region_uot", False):
            if query_tokens is None or support_tokens is None or spatial_hw is None:
                raise ValueError(
                    "enable_adaptive_region_uot requires query_tokens, support_tokens, and spatial_hw"
                )
            (
                adaptive_guided_cost,
                adaptive_query_weight,
                adaptive_support_weight,
                adaptive_region_payload,
            ) = self.adaptive_region_uot(
                flat_cost=cost_for_transport,
                query_tokens=query_tokens,
                support_tokens=support_tokens,
                way_num=way_num,
                shot_num=shot_num,
                spatial_hw=spatial_hw,
                rho=self._ecot_base_rho_tensor(flat_cost),
            )
            cost_for_transport = adaptive_guided_cost
            query_weight = self._combine_token_weight(query_weight, adaptive_query_weight)
            support_weight = self._combine_token_weight(support_weight, adaptive_support_weight)

        if (
            getattr(self, "enable_discriminative_uot", False)
            and not getattr(self, "enable_pulse_region_uot", False)
        ):
            score_pair_evidence, discriminative_payload = self._origin_discriminative_pair_evidence(
                flat_cost=cost_for_transport,
                way_num=way_num,
                shot_num=shot_num,
            )
            score_evidence_mass_weight = float(self.discriminative_uot_mass_weight)
            score_evidence_cost_weight = float(self.discriminative_uot_cost_weight)
            score_background_penalty = float(self.discriminative_uot_background_penalty)

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
            if getattr(self, "pulse_evidence_score", True):
                score_query_evidence = self._pulse_region_evidence_mask(query_saliency, spatial_hw)
                score_support_evidence = self._pulse_region_evidence_mask(support_saliency, spatial_hw)
                pulse_strength_value = float(pulse_strength.detach().cpu().item())
                score_evidence_mass_weight = 1.0 + pulse_strength_value * (
                    float(self.pulse_evidence_mass_weight) - 1.0
                )
                score_evidence_cost_weight = 1.0 + pulse_strength_value * (
                    float(self.pulse_evidence_cost_weight) - 1.0
                )
                score_background_penalty = pulse_strength_value * float(self.pulse_background_penalty)
                if pulse_strength.item() < 1.0:
                    q_ones = torch.ones_like(score_query_evidence)
                    s_ones = torch.ones_like(score_support_evidence)
                    score_query_evidence = q_ones + pulse_strength * (score_query_evidence - q_ones)
                    score_support_evidence = s_ones + pulse_strength * (score_support_evidence - s_ones)
                if getattr(self, "pulse_discriminative_evidence", True):
                    score_pair_evidence, discriminative_payload = self._pulse_discriminative_pair_evidence(
                        flat_cost=cost_for_transport,
                        query_evidence=score_query_evidence,
                        support_evidence=score_support_evidence,
                        way_num=way_num,
                        shot_num=shot_num,
                        strength=pulse_strength,
                    )
                    pulse_payload.update(discriminative_payload)
                pulse_payload.update(
                    {
                        "pulse_query_evidence": score_query_evidence.detach(),
                        "pulse_support_evidence": score_support_evidence.detach(),
                        "pulse/evidence_score_enabled": flat_cost.new_tensor(1.0),
                        "pulse/evidence_mass_weight": flat_cost.new_tensor(score_evidence_mass_weight),
                        "pulse/evidence_cost_weight": flat_cost.new_tensor(score_evidence_cost_weight),
                        "pulse/background_penalty": flat_cost.new_tensor(score_background_penalty),
                        "pulse/background_penalty_config": flat_cost.new_tensor(
                            float(self.pulse_background_penalty)
                        ),
                        "pulse/query_evidence_area50": (score_query_evidence > 0.5).to(flat_cost.dtype).mean().detach(),
                        "pulse/support_evidence_area50": (
                            score_support_evidence > 0.5
                        ).to(flat_cost.dtype).mean().detach(),
                    }
                )
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

        evidence_payload: dict[str, torch.Tensor] = {}
        if getattr(self, "enable_evidence_marginals", False):
            base_rho_ev = self._ecot_base_rho_tensor(flat_cost)
            ev_query_marginal, ev_support_marginal, evidence_payload = (
                self.evidence_marginal_module(
                    cost_for_transport,
                    rho=base_rho_ev,
                    way_num=way_num,
                    shot_num=shot_num,
                )
            )
            if ev_query_marginal is not None:
                query_weight = ev_query_marginal
            if ev_support_marginal is not None:
                support_weight = ev_support_marginal

        payload = super()._forward_ecot_budget_bank(
            cost_for_transport,
            way_num=way_num,
            shot_num=shot_num,
            support_weight=support_weight,
            query_weight=query_weight,
            support_tokens=support_tokens,
            query_tokens=query_tokens,
            spatial_hw=spatial_hw,
            score_query_evidence=score_query_evidence,
            score_support_evidence=score_support_evidence,
            score_pair_evidence=score_pair_evidence,
            score_evidence_mass_weight=score_evidence_mass_weight,
            score_evidence_cost_weight=score_evidence_cost_weight,
            score_background_penalty=score_background_penalty,
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
        if discriminative_payload:
            payload.update(discriminative_payload)
        if region_uot_payload:
            payload.update(region_uot_payload)
        if adaptive_region_payload:
            payload.update(adaptive_region_payload)
        if mspta_payload:
            payload.update(mspta_payload)
        if evidence_payload:
            payload.update(evidence_payload)
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

    def _label_ot_inactive_payload(
        self,
        logits: torch.Tensor,
        *,
        reason_code: float,
        imbalance_before: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        zero = logits.new_zeros(())
        return {
            "label_ot/enabled": logits.new_tensor(1.0),
            "label_ot/active": zero,
            "label_ot/reason": logits.new_tensor(float(reason_code)),
            "label_ot/epsilon": logits.new_tensor(float(self.label_ot_epsilon)),
            "label_ot/iterations": logits.new_tensor(float(self.label_ot_iterations)),
            "label_ot/mix": logits.new_tensor(float(self.label_ot_mix)),
            "label_ot/column_imbalance_before": zero if imbalance_before is None else imbalance_before.detach(),
            "label_ot/column_imbalance_after": zero if imbalance_before is None else imbalance_before.detach(),
            "label_ot/bias_abs_mean": zero,
            "label_ot/bias_abs_max": zero,
            "label_ot/logit_delta_abs_mean": zero,
            "label_ot/assignment_entropy": zero,
        }

    def _apply_label_ot_to_logits(
        self,
        logits: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if not getattr(self, "enable_label_ot", False):
            return logits, {}
        if logits.dim() != 2:
            return logits, self._label_ot_inactive_payload(logits, reason_code=4.0)
        if self.training:
            return logits, self._label_ot_inactive_payload(logits, reason_code=1.0)

        num_query, way_num = logits.shape
        min_queries = int(self.label_ot_min_queries_per_class) * int(way_num)
        if way_num <= 1 or num_query < min_queries:
            return logits, self._label_ot_inactive_payload(logits, reason_code=2.0)

        eps = max(float(self.label_ot_epsilon), float(self.eps))
        row_target = logits.new_ones(num_query)
        col_target = logits.new_full((way_num,), float(num_query) / float(way_num))
        soft_before = torch.softmax(logits / eps, dim=-1)
        col_before = soft_before.sum(dim=0)
        imbalance_before = (col_before - col_target).abs().sum() / float(max(1, num_query))
        if imbalance_before < float(self.label_ot_min_column_imbalance):
            return logits, self._label_ot_inactive_payload(
                logits,
                reason_code=3.0,
                imbalance_before=imbalance_before,
            )

        stabilized = (logits - logits.amax(dim=1, keepdim=True)) / eps
        kernel = stabilized.clamp(min=-60.0, max=60.0).exp().clamp_min(float(self.eps))
        u = logits.new_ones(num_query)
        v = logits.new_ones(way_num)
        for _ in range(int(self.label_ot_iterations)):
            u = row_target / (kernel @ v).clamp_min(float(self.eps))
            v = col_target / (kernel.transpose(0, 1) @ u).clamp_min(float(self.eps))

        bias = eps * v.clamp_min(float(self.eps)).log()
        bias = bias - bias.mean()
        if self.label_ot_max_bias > 0.0:
            bias = bias.clamp(min=-float(self.label_ot_max_bias), max=float(self.label_ot_max_bias))
        delta = float(self.label_ot_mix) * bias.unsqueeze(0)
        adjusted = logits + delta

        assignment = torch.softmax(adjusted / eps, dim=-1)
        col_after = assignment.sum(dim=0)
        imbalance_after = (col_after - col_target).abs().sum() / float(max(1, num_query))
        entropy = -(assignment * assignment.clamp_min(float(self.eps)).log()).sum(dim=-1).mean()
        payload = {
            "label_ot/enabled": logits.new_tensor(1.0),
            "label_ot/active": logits.new_tensor(1.0),
            "label_ot/reason": logits.new_tensor(0.0),
            "label_ot/epsilon": logits.new_tensor(float(self.label_ot_epsilon)),
            "label_ot/iterations": logits.new_tensor(float(self.label_ot_iterations)),
            "label_ot/mix": logits.new_tensor(float(self.label_ot_mix)),
            "label_ot/column_imbalance_before": imbalance_before.detach(),
            "label_ot/column_imbalance_after": imbalance_after.detach(),
            "label_ot/bias_abs_mean": bias.detach().abs().mean(),
            "label_ot/bias_abs_max": bias.detach().abs().amax(),
            "label_ot/logit_delta_abs_mean": delta.detach().abs().mean(),
            "label_ot/assignment_entropy": entropy.detach(),
        }
        return adjusted, payload

    def _apply_label_ot_to_outputs(
        self,
        outputs: torch.Tensor | dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor | dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs
        adjusted, payload = self._apply_label_ot_to_logits(logits)
        if not payload:
            return outputs, payload
        if isinstance(outputs, dict):
            outputs = dict(outputs)
            outputs["logits"] = adjusted
            outputs["class_scores"] = adjusted
            outputs.update(payload)
            return outputs, payload
        return adjusted, payload

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
        use_mspta_cache = getattr(self, "enable_mspta", False)
        use_pulse_cache = getattr(self, "enable_pulse_region_uot", False)
        if use_prenorm_cache:
            self._token_g_prenorm_cache = []
        if use_multiscale_cache:
            self._multiscale_ot_cache = []
        if use_mspta_cache:
            self._mspta_cache = []
        if use_pulse_cache:
            self._pulse_saliency_cache = []
        try:
            outputs = super()._forward_episode(*args, **kwargs)
        finally:
            if use_prenorm_cache:
                self._token_g_prenorm_cache = None
            if use_multiscale_cache:
                self._multiscale_ot_cache = None
            if use_mspta_cache:
                self._mspta_cache = None
            if use_pulse_cache:
                self._pulse_saliency_cache = None
            if use_context_debug:
                self._export_context_debug()
                if self.enable_context_enrichment:
                    self.context_enrichment._debug_cache = None
                self._context_debug_images = []
                self._context_debug_fmaps = []
        outputs, _label_ot_payload = self._apply_label_ot_to_outputs(outputs)
        if isinstance(outputs, dict):
            if "mspta_guided_cost_matrix" in outputs and "cost_matrix" in outputs:
                outputs.setdefault("base_cost_matrix", outputs["cost_matrix"])
                outputs["cost_matrix"] = outputs["mspta_guided_cost_matrix"]
            if "region_uot_guided_cost_matrix" in outputs and "cost_matrix" in outputs:
                outputs.setdefault("base_cost_matrix", outputs["cost_matrix"])
                outputs["cost_matrix"] = outputs["region_uot_guided_cost_matrix"]
            if "pulse_guided_cost_matrix" in outputs and "cost_matrix" in outputs:
                outputs.setdefault("base_cost_matrix", outputs["cost_matrix"])
                outputs["cost_matrix"] = outputs["pulse_guided_cost_matrix"]
            if "adaptive_region_guided_cost_matrix" in outputs and "cost_matrix" in outputs:
                outputs.setdefault("base_cost_matrix", outputs["cost_matrix"])
                outputs["cost_matrix"] = outputs["adaptive_region_guided_cost_matrix"]
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
            "mspta_guided_cost_matrix",
        ):
            if key in batch_outputs[0]:
                stacked[key] = torch.stack([item[key] for item in batch_outputs], dim=0)
        for key in (
            "pulse_query_saliency",
            "pulse_query_evidence",
            "pulse_discriminative_gate",
            "pulse_rival_advantage",
            "discriminative_uot_gate",
            "discriminative_uot_rival_advantage",
            "pulse_query_marginal_weight",
            "pulse_region_cost_matrix",
            "pulse_guided_cost_matrix",
            "region_uot_coarse_plan",
            "region_uot_sparse_coarse_plan",
            "region_uot_coarse_cost_matrix",
            "region_uot_guided_cost_matrix",
            "region_uot_coarse_transport_cost",
            "region_uot_coarse_transported_mass",
            "adaptive_region_query_masks",
            "adaptive_region_guided_cost_matrix",
            "adaptive_region_plan",
            "adaptive_region_cost_matrix",
            "adaptive_region_transport_cost",
            "adaptive_region_transported_mass",
            "adaptive_region_query_weight",
        ):
            if key in batch_outputs[0]:
                stacked[key] = torch.cat([item[key] for item in batch_outputs], dim=0)
        for key in (
            "pulse_support_saliency",
            "pulse_support_evidence",
            "pulse_support_marginal_weight",
            "adaptive_region_support_masks",
            "adaptive_region_support_weight",
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
        for prefix in (
            "context/",
            "struct/",
            "pulse/",
            "discriminative_uot/",
            "label_ot/",
            "region_uot/",
            "adaptive_region/",
            "mspta/",
        ):
            for key in sorted(k for k in batch_outputs[0] if str(k).startswith(prefix)):
                values = [item[key] for item in batch_outputs if key in item]
                if values and all(torch.is_tensor(value) for value in values):
                    stacked[key] = torch.stack(values).mean()
        return stacked

__all__ = ["OursM2", "apply_ours_design_defaults", "OURS_ABLATIONS", "normalize_ours_ablation"]
