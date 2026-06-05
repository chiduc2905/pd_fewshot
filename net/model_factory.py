"""Model registry and builders for pulse_fewshot benchmarks."""

from importlib import import_module


def _bool_flag(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).lower() == "true"


def _auto_bool_flag(value, default=False):
    if value is None:
        return default
    if isinstance(value, str) and value.strip().lower() == "auto":
        return default
    return _bool_flag(value, default=default)


def _first_attr(args, *names, default=None):
    for name in names:
        if hasattr(args, name):
            value = getattr(args, name)
            if value is not None:
                return value
    return default


def _parse_csv_ints(value, default):
    if value is None:
        return tuple(default)
    if isinstance(value, (list, tuple)):
        return tuple(int(item) for item in value)
    text = str(value).strip()
    if not text:
        return tuple(default)
    return tuple(int(item.strip()) for item in text.split(",") if item.strip())


def _parse_csv_floats(value, default):
    if value is None:
        return tuple(default)
    if isinstance(value, (list, tuple)):
        return tuple(float(item) for item in value)
    text = str(value).strip()
    if not text:
        return tuple(default)
    return tuple(float(item.strip()) for item in text.split(",") if item.strip())


def _load_symbol(module_name, symbol_name):
    # Keep model imports branch-local so unrelated experimental files cannot
    # break other models during main.py startup.
    module = import_module(module_name)
    try:
        return getattr(module, symbol_name)
    except AttributeError as exc:
        raise ImportError(f"Module '{module_name}' does not define '{symbol_name}'") from exc


def _load_symbols(module_name, *symbol_names):
    return tuple(_load_symbol(module_name, symbol_name) for symbol_name in symbol_names)


M2_MODEL_NAMES = frozenset(
    {"m2", "m2_uot", "m2_full_ot", "jecot_m2", "jecot_m2_uot", "jecot_m2_full_ot"}
)
OURS_MODEL_NAMES = frozenset({"ours"})
CANONICAL_OURS_FINAL_MODEL_NAMES = frozenset({"ours_final"})
OURS_FINAL_VERIFIED_UOT_MODEL_NAMES = frozenset({"ours_final_verified_uot"})
OURS_FINAL_PARTIAL_OT_MODEL_NAMES = frozenset({"ours_final_partial_ot"})
OURS_FINAL_MODEL_NAMES = frozenset(
    set(CANONICAL_OURS_FINAL_MODEL_NAMES)
    | set(OURS_FINAL_VERIFIED_UOT_MODEL_NAMES)
    | set(OURS_FINAL_PARTIAL_OT_MODEL_NAMES)
)
OURS_ENTRYPOINT_MODEL_NAMES = frozenset(set(OURS_MODEL_NAMES) | set(OURS_FINAL_MODEL_NAMES))
OURS_CPM_MODEL_NAMES = frozenset({"ours_cpm"})
SGPOT_MODEL_NAMES = frozenset({"sg_pot"})
TA_DSW_MODEL_NAMES = frozenset({"ta_dsw", "ta-dsw"})
M2_LIKE_MODEL_NAMES = frozenset(
    set(M2_MODEL_NAMES) | set(OURS_ENTRYPOINT_MODEL_NAMES) | set(OURS_CPM_MODEL_NAMES) | set(SGPOT_MODEL_NAMES)
)
# Registry name for the single canonical SB-ECOT entrypoint (aliases share the same class but different CLI defaults).
CANONICAL_M2_MODEL_NAME = "m2"
M2_FULL_OT_MODEL_NAMES = frozenset({"m2_full_ot", "jecot_m2_full_ot"})
HROT_FSL_MODEL_NAMES = frozenset(
    {"hrot_fsl"} | set(M2_LIKE_MODEL_NAMES) | set(OURS_CPM_MODEL_NAMES) | set(SGPOT_MODEL_NAMES)
)
OURS_FINAL_DMUOT_ABLATION_CONFIGS = {
    "off": {
        "token_g_kind": "none",
        "lambda_cost": 0.0,
        "marginal_kind": "uniform",
        "tau_marg": 1.0,
    },
    "exp0_g_episode_mean_dist": {
        "token_g_kind": "episode_mean_dist",
        "lambda_cost": 0.0,
        "marginal_kind": "uniform",
        "tau_marg": 1.0,
    },
    "exp0_g_token_norm_pre_l2": {
        "token_g_kind": "token_norm_pre_l2",
        "lambda_cost": 0.0,
        "marginal_kind": "uniform",
        "tau_marg": 1.0,
    },
}
for _lambda_cost in (0.0, 0.5, 1.0, 2.0, 5.0):
    _suffix = f"{_lambda_cost:.6g}".replace(".", "p")
    OURS_FINAL_DMUOT_ABLATION_CONFIGS[f"exp1_lambda_cost_{_suffix}"] = {
        "token_g_kind": "episode_mean_dist",
        "lambda_cost": _lambda_cost,
        "marginal_kind": "uniform",
        "tau_marg": 1.0,
    }
for _tau_marg in (0.25, 0.5, 1.0, 2.0, float("inf")):
    _suffix = "inf" if _tau_marg == float("inf") else f"{_tau_marg:.6g}".replace(".", "p")
    OURS_FINAL_DMUOT_ABLATION_CONFIGS[f"exp2_tau_marg_{_suffix}"] = {
        "token_g_kind": "episode_mean_dist",
        "lambda_cost": 0.0,
        "marginal_kind": "discriminative",
        "tau_marg": _tau_marg,
    }
for _tau_marg in (1.0, 2.0):
    _suffix = f"{_tau_marg:.6g}".replace(".", "p")
    for _strength_name, _strength_value in (
        ("shot_sqrt", "inverse_sqrt"),
        ("shot_inverse", "inverse"),
    ):
        OURS_FINAL_DMUOT_ABLATION_CONFIGS[f"exp5_tau_marg_{_suffix}_{_strength_name}"] = {
            "token_g_kind": "episode_mean_dist",
            "lambda_cost": 0.0,
            "marginal_kind": "discriminative",
            "tau_marg": _tau_marg,
            "dmuot_shot_strength": _strength_value,
        }

for _tau_marg_la in (0.5, 1.0, 2.0):
    _suffix_la = f"{_tau_marg_la:.6g}".replace(".", "p")
    OURS_FINAL_DMUOT_ABLATION_CONFIGS[f"exp6_learned_attn_tau_{_suffix_la}"] = {
        "token_g_kind": "learned_attention",
        "lambda_cost": 0.0,
        "marginal_kind": "discriminative",
        "tau_marg": _tau_marg_la,
    }
    OURS_FINAL_DMUOT_ABLATION_CONFIGS[f"exp6_learned_attn_tau_{_suffix_la}_shot_sqrt"] = {
        "token_g_kind": "learned_attention",
        "lambda_cost": 0.0,
        "marginal_kind": "discriminative",
        "tau_marg": _tau_marg_la,
        "dmuot_shot_strength": "inverse_sqrt",
    }


OURS_FINAL_EVIDENCE_ABLATION_CONFIGS = {
    "off": {},
    "query_only": {
        "enable_evidence_marginals": True,
        "evidence_mode": "query_only",
        "evidence_tau": 0.1,
        "evidence_tau_marginal": 1.0,
        "evidence_detach": True,
    },
    "support_only": {
        "enable_evidence_marginals": True,
        "evidence_mode": "support_only",
        "evidence_tau": 0.1,
        "evidence_tau_marginal": 1.0,
        "evidence_detach": True,
    },
    "both": {
        "enable_evidence_marginals": True,
        "evidence_mode": "both",
        "evidence_tau": 0.1,
        "evidence_tau_marginal": 1.0,
        "evidence_detach": True,
    },
    "rival_aware": {
        "enable_evidence_marginals": True,
        "evidence_mode": "rival_aware",
        "evidence_tau": 0.1,
        "evidence_tau_marginal": 1.0,
        "evidence_rival_margin": 0.5,
        "evidence_detach": True,
    },
}
for _ev_tau in (0.05, 0.1, 0.2, 0.5):
    _ev_suffix = f"{_ev_tau:.6g}".replace(".", "p")
    OURS_FINAL_EVIDENCE_ABLATION_CONFIGS[f"both_tau_{_ev_suffix}"] = {
        "enable_evidence_marginals": True,
        "evidence_mode": "both",
        "evidence_tau": _ev_tau,
        "evidence_tau_marginal": 1.0,
        "evidence_detach": True,
    }
for _ev_tau_m in (0.5, 1.0, 2.0):
    _ev_suffix_m = f"{_ev_tau_m:.6g}".replace(".", "p")
    OURS_FINAL_EVIDENCE_ABLATION_CONFIGS[f"both_tau_m_{_ev_suffix_m}"] = {
        "enable_evidence_marginals": True,
        "evidence_mode": "both",
        "evidence_tau": 0.1,
        "evidence_tau_marginal": _ev_tau_m,
        "evidence_detach": True,
    }
for _ev_rival_m in (0.0, 0.25, 0.5, 1.0):
    _ev_rival_suffix = f"{_ev_rival_m:.6g}".replace(".", "p")
    OURS_FINAL_EVIDENCE_ABLATION_CONFIGS[f"rival_margin_{_ev_rival_suffix}"] = {
        "enable_evidence_marginals": True,
        "evidence_mode": "rival_aware",
        "evidence_tau": 0.1,
        "evidence_tau_marginal": 1.0,
        "evidence_rival_margin": _ev_rival_m,
        "evidence_detach": True,
    }


def get_ours_final_evidence_choices() -> tuple[str, ...]:
    return tuple(OURS_FINAL_EVIDENCE_ABLATION_CONFIGS.keys())


def normalize_ours_final_evidence_ablation(value) -> str:
    name = "off" if value is None else str(value).strip().lower().replace("-", "_")
    aliases = {
        "none": "off",
        "false": "off",
        "0": "off",
        "query": "query_only",
        "support": "support_only",
        "all": "both",
        "full": "both",
        "rival": "rival_aware",
        "disc": "rival_aware",
        "discriminative": "rival_aware",
    }
    name = aliases.get(name, name)
    if name not in OURS_FINAL_EVIDENCE_ABLATION_CONFIGS:
        raise ValueError(
            f"Unsupported ours_final_evidence_ablation: {value}. "
            f"Expected one of {sorted(OURS_FINAL_EVIDENCE_ABLATION_CONFIGS)}"
        )
    return name


def resolve_ours_final_evidence_config(args) -> dict:
    ablation = normalize_ours_final_evidence_ablation(
        getattr(args, "ours_final_evidence_ablation", "off")
    )
    config = dict(OURS_FINAL_EVIDENCE_ABLATION_CONFIGS[ablation])
    for override_key in (
        "evidence_tau",
        "evidence_tau_marginal",
        "evidence_mode",
        "evidence_rival_margin",
        "evidence_detach",
    ):
        override_value = getattr(args, override_key, None)
        if override_value is not None:
            if override_key in ("evidence_tau", "evidence_tau_marginal", "evidence_rival_margin"):
                config[override_key] = float(override_value)
            elif override_key == "evidence_detach":
                config[override_key] = str(override_value).strip().lower() in ("true", "1", "yes")
            else:
                config[override_key] = str(override_value)
    return config


def resolve_hrot_ecot_enable_egsm(args) -> str:
    """Resolve ``--hrot_ecot_enable_egsm`` to ``'true'`` or ``'false'`` (no auto).

    If the flag is omitted, defaults to ``true`` for legacy ``ours`` /
    ``ours_cpm`` and ``false`` for ``ours_final`` and other J_ECOT_M2-style
    models (e.g. ``m2``).
    """
    raw = getattr(args, "hrot_ecot_enable_egsm", None)
    if raw is not None and str(raw).strip() != "":
        resolved = str(raw).strip().lower()
        if resolved in {"auto", "none"}:
            raise ValueError(
                "hrot_ecot_enable_egsm no longer accepts 'auto' or 'none'; "
                "use true or false, or omit the flag for model defaults."
            )
        if resolved not in {"true", "false"}:
            raise ValueError(
                "hrot_ecot_enable_egsm must be 'true' or 'false' "
                f"(got {raw!r}; legacy 'auto' is no longer supported)."
            )
        return resolved
    model = str(getattr(args, "model", "")).strip().lower()
    if model in OURS_MODEL_NAMES or model in OURS_CPM_MODEL_NAMES:
        return "true"
    return "false"


def get_ours_final_dmuot_choices() -> tuple[str, ...]:
    return tuple(OURS_FINAL_DMUOT_ABLATION_CONFIGS.keys())


def normalize_ours_final_dmuot_ablation(value) -> str:
    name = "off" if value is None else str(value).strip().lower().replace("-", "_")
    aliases = {
        "none": "off",
        "false": "off",
        "0": "off",
        "g_episode_mean_dist": "exp0_g_episode_mean_dist",
        "g_token_norm_pre_l2": "exp0_g_token_norm_pre_l2",
    }
    name = aliases.get(name, name)
    if name not in OURS_FINAL_DMUOT_ABLATION_CONFIGS:
        raise ValueError(
            f"Unsupported ours_final_dmuot_ablation: {value}. "
            f"Expected one of {sorted(OURS_FINAL_DMUOT_ABLATION_CONFIGS)}"
        )
    return name


def resolve_ours_final_dmuot_config(args) -> dict:
    ablation = normalize_ours_final_dmuot_ablation(
        getattr(args, "ours_final_dmuot_ablation", "off")
    )
    config = dict(OURS_FINAL_DMUOT_ABLATION_CONFIGS[ablation])
    config.setdefault("dmuot_shot_strength", "none")
    return config


def validate_dmuot_scope(args) -> None:
    ablation = normalize_ours_final_dmuot_ablation(
        getattr(args, "ours_final_dmuot_ablation", "off")
    )
    if ablation != "off" and str(getattr(args, "model", "")).strip().lower() != "ours_final":
        raise ValueError("--ours_final_dmuot_ablation is supported only with --model ours_final")
    if _bool_flag(getattr(args, "enable_pot_guide", False), default=False) and (
        str(getattr(args, "model", "")).strip().lower() != "ours_final"
    ):
        raise ValueError("--enable_pot_guide is supported only with --model ours_final")
    if _bool_flag(getattr(args, "enable_multiscale_ot", False), default=False) and (
        str(getattr(args, "model", "")).strip().lower() != "ours_final"
    ):
        raise ValueError("--enable_multiscale_ot is supported only with --model ours_final")
    if _bool_flag(getattr(args, "enable_mspta", False), default=False) and (
        str(getattr(args, "model", "")).strip().lower() != "ours_final"
    ):
        raise ValueError("--enable_mspta is supported only with --model ours_final")
    if _bool_flag(getattr(args, "enable_context_enrichment", False), default=False) and (
        str(getattr(args, "model", "")).strip().lower() != "ours_final"
    ):
        raise ValueError("--enable_context_enrichment is supported only with --model ours_final")
    if _bool_flag(getattr(args, "enable_structural_augmentation", False), default=False) and (
        str(getattr(args, "model", "")).strip().lower() != "ours_final"
    ):
        raise ValueError("--enable_structural_augmentation is supported only with --model ours_final")
    if _bool_flag(getattr(args, "enable_region_structural_uot", False), default=False) and (
        str(getattr(args, "model", "")).strip().lower() != "ours_final"
    ):
        raise ValueError("--enable_region_structural_uot is supported only with --model ours_final")
    if _bool_flag(getattr(args, "enable_adaptive_region_uot", False), default=False) and (
        str(getattr(args, "model", "")).strip().lower() != "ours_final"
    ):
        raise ValueError("--enable_adaptive_region_uot is supported only with --model ours_final")
    if _bool_flag(getattr(args, "enable_pulse_region_uot", False), default=False) and (
        str(getattr(args, "model", "")).strip().lower() != "ours_final"
    ):
        raise ValueError("--enable_pulse_region_uot is supported only with --model ours_final")
    if _bool_flag(getattr(args, "enable_verified_uot_score", False), default=False) and (
        str(getattr(args, "model", "")).strip().lower() not in OURS_FINAL_MODEL_NAMES
    ):
        raise ValueError("--enable_verified_uot_score is supported only with Ours-Final models")
    if _bool_flag(getattr(args, "enable_global_residual_score", False), default=False) and (
        str(getattr(args, "model", "")).strip().lower() != "ours_final"
    ):
        raise ValueError("--enable_global_residual_score is supported only with --model ours_final")
    if _bool_flag(getattr(args, "enable_discriminative_uot", False), default=False) and (
        str(getattr(args, "model", "")).strip().lower() != "ours_final"
    ):
        raise ValueError("--enable_discriminative_uot is supported only with --model ours_final")
    if _bool_flag(getattr(args, "enable_label_ot", False), default=False) and (
        str(getattr(args, "model", "")).strip().lower() != "ours_final"
    ):
        raise ValueError("--enable_label_ot is supported only with --model ours_final")
    evidence_ablation = normalize_ours_final_evidence_ablation(
        getattr(args, "ours_final_evidence_ablation", "off")
    )
    if evidence_ablation != "off" and str(getattr(args, "model", "")).strip().lower() != "ours_final":
        raise ValueError("--ours_final_evidence_ablation is supported only with --model ours_final")


MODEL_REGISTRY = {
    "cosine": {
        "display_name": "Cosine Classifier",
        "architecture": "Conv4 embedding + cosine prototype similarity",
        "metric": "Cosine Similarity",
    },
    "protonet": {
        "display_name": "ProtoNet",
        "architecture": "Conv4 embedding + prototype Euclidean metric",
        "metric": "Squared Euclidean Distance",
    },
    "rada_fsl": {
        "display_name": "RADA-FSL",
        "architecture": "Backbone pooled embeddings -> evidence-bounded query-conditioned reliability prototypes -> adaptive diagonal dispersion matching",
        "metric": "Evidence-Bounded Reliability Prototype + Diagonal Mahalanobis Distance",
    },
    "covamnet": {
        "display_name": "CovaMNet",
        "architecture": "Conv64F + covariance similarity",
        "metric": "Covariance Metric",
    },
    "matchingnet": {
        "display_name": "MatchingNet",
        "architecture": "Conv4 embedding + BiLSTM/FCE matching",
        "metric": "Cosine Attention Matching",
    },
    "relationnet": {
        "display_name": "RelationNet",
        "architecture": "Conv4-nopool encoder + relation head",
        "metric": "Learned Relation Score",
    },
    "dn4": {
        "display_name": "DN4",
        "architecture": "Conv64F + local descriptor k-NN",
        "metric": "Local Descriptor Cosine",
    },
    "feat": {
        "display_name": "FEAT",
        "architecture": "ResNet12 + FEAT self-attention adaptation",
        "metric": "Task-adapted Euclidean Distance",
    },
    "deepemd": {
        "display_name": "DeepEMD",
        "architecture": "ResNet12 local descriptors + transport matching",
        "metric": "Earth Mover's Distance",
    },
    "evidence_deepemd": {
        "display_name": "Evidence-DeepEMD",
        "architecture": "DeepEMD local descriptors + class-stable evidence masses + support-shot reliability aggregation",
        "metric": "Evidence-Weighted Balanced Earth Mover's Distance",
    },
    "evidence_budget_ot": {
        "display_name": "EBOT",
        "architecture": "Backbone local tokens + scalogram reliability priors + evidence-budgeted dustbin OT + k-shot evidence aggregation",
        "metric": "Reliability-Budgeted Partial Optimal Transport",
    },
    "ebot": {
        "display_name": "EBOT",
        "architecture": "Alias for Evidence-Budgeted Optimal Transport with scalogram reliability priors",
        "metric": "Reliability-Budgeted Partial Optimal Transport",
    },
    "ebot_scalogram": {
        "display_name": "EBOT-Scalogram",
        "architecture": "Alias for Evidence-Budgeted Optimal Transport with scalogram reliability priors",
        "metric": "Reliability-Budgeted Partial Optimal Transport",
    },
    "evidence_budgeted_ot": {
        "display_name": "EvidenceBudgetedOT",
        "architecture": "Alias for Evidence-Budgeted Optimal Transport with scalogram reliability priors",
        "metric": "Reliability-Budgeted Partial Optimal Transport",
    },
    "mm_spot_fsl": {
        "display_name": "MM-SPOT-FSL",
        "architecture": "Backbone local tokens + multi-mass partial OT evidence selection + mass-score aggregation",
        "metric": "Multi-Mass Partial Optimal Transport",
    },
    "pare_fsl": {
        "display_name": "PARE-FSL",
        "architecture": "Backbone tokens -> EGSM marginals (sum=1) -> learned partial mass alpha per shot -> native POT -> partial discrepancy pooling",
        "metric": "Partial Adaptive Relational Evidence Transport",
    },
    "dice_emd": {
        "display_name": "DICE-EMD",
        "architecture": "ResNet12 dense descriptors + class-competitive evidence-aware balanced transport",
        "metric": "Discriminative Class-Competitive Earth Mover's Distance",
    },
    "transport_recon_emd": {
        "display_name": "Transport-Recon EMD",
        "architecture": "ResNet12 dense descriptors + balanced transport alignment + bidirectional reconstruction + Gram structure consistency",
        "metric": "Transport-Guided Reconstruction EMD",
    },
    "tardis_emd": {
        "display_name": "TARDIS-EMD",
        "architecture": "Alias for Transport-Recon EMD: ResNet12 dense descriptors + transport reconstruction consistency",
        "metric": "Transport-Guided Reconstruction EMD",
    },
    "can": {
        "display_name": "CAN",
        "architecture": "ResNet12 + paper-style CAM + inductive cosine inference",
        "metric": "Cross-Attention Similarity",
    },
    "frn": {
        "display_name": "FRN",
        "architecture": "ResNet12 feature maps + class-wise reconstruction metric",
        "metric": "Reconstruction Distance",
    },
    "deepbdc": {
        "display_name": "DeepBDC",
        "architecture": "Meta DeepBDC with ResNet12 + BDC pooling",
        "metric": "Second-Order Prototype Distance",
    },
    "maml": {
        "display_name": "MAML",
        "architecture": "Conv4-32 encoder + inner-loop adaptation",
        "metric": "Task-adapted Linear Classifier",
    },
    "uscmamba": {
        "display_name": "USCMambaNet",
        "architecture": "PatchEmbed -> Conv stem -> dual-branch fusion -> late attention -> cosine metric head",
        "metric": "Cosine Similarity",
    },
    "conv64f_token_sw_metric_net": {
        "display_name": "Conv64FTokenSWMetricNet",
        "architecture": "Conv64F -> spatial tokens -> sliced Wasserstein metric (+ optional global pooled distance)",
        "metric": "Sliced Wasserstein Distance",
    },
    "class_memory_scan_mamba_net": {
        "display_name": "ClassMemoryScanMambaNet",
        "architecture": "Conv64F -> support write SSM -> class memory -> query read SSM + auxiliary SW",
        "metric": "Class-Memory Similarity",
    },
    "episodic_selective_scan_mamba_net": {
        "display_name": "EpisodicSelectiveScanMambaNet",
        "architecture": "Conv64F -> role-aware support scan -> class state -> role-aware query readout + SW alignment",
        "metric": "Class-Conditioned Scan Similarity",
    },
    "permutation_robust_class_memory_mamba_net": {
        "display_name": "PermutationRobustClassMemoryMambaNet",
        "architecture": "Conv64F -> multi-permutation class memory scan -> permutation-aware readout + SW",
        "metric": "Permutation-Robust Class-Memory Similarity",
    },
    "hierarchical_episodic_ssm_net": {
        "display_name": "HierarchicalEpisodicSSMNet",
        "architecture": "Conv64F -> token-level SSM -> shot-level memory SSM -> hierarchical query matcher + SW",
        "metric": "Hierarchical Memory Similarity",
    },
    "hierarchical_consensus_slot_mamba_net": {
        "display_name": "HierarchicalConsensusSlotMambaNet",
        "architecture": "Conv64F/ResNet12 -> intra-image Mamba -> set memory pooling -> reliability-coupled matcher + SW",
        "metric": "Consensus-Slot Memory Similarity",
    },
    "hierarchical_support_sw_net": {
        "display_name": "HS-SW",
        "architecture": "Backbone token pyramid -> scale-budgeted shot measures -> exact weighted sliced Wasserstein -> support-aware hierarchical inference",
        "metric": "Hierarchical Support Sliced-Wasserstein",
    },
    "ta_dsw": {
        "display_name": "TA-DSW",
        "architecture": "Backbone spatial tokens -> learned token projection/weights -> deterministic task-adaptive slices -> shot-aware weighted SW + cosine prototype hybrid scoring",
        "metric": "Task-Adaptive Weighted Sliced Wasserstein + Cosine Prototype",
    },
    "ta-dsw": {
        "display_name": "TA-DSW",
        "architecture": "Alias for TA-DSW with learned token projection/weights, deterministic task-adaptive slices, and hybrid scoring",
        "metric": "Task-Adaptive Weighted Sliced Wasserstein + Cosine Prototype",
    },
    "sc_lfi": {
        "display_name": "SC-LFI",
        "architecture": "Backbone tokens -> permutation-invariant class context -> latent evidence projector -> support-conditioned latent flow -> distribution-fit scoring",
        "metric": "Support-Conditioned Distribution Fit",
    },
    "sc_lfi_v2": {
        "display_name": "SC-LFI-v2",
        "architecture": "Backbone tokens -> latent evidence + reliability masses -> weighted class summary + support memory -> memory-conditioned latent flow -> weighted transport scoring + direct distribution margin",
        "metric": "Weighted Support-Conditioned Transport Fit",
    },
    "sc_lfi_v3": {
        "display_name": "SC-LFI-v3",
        "architecture": "Backbone tokens -> latent evidence -> support empirical measure + support-conditioned prior -> shot-aware posterior base measure -> posterior transport flow -> query-conditioned weighted transport scoring",
        "metric": "Posterior Evidence Transport Fit",
    },
    "sc_lfi_v4": {
        "display_name": "SC-LFI-v4",
        "architecture": "Backbone tokens -> per-shot latent evidence bases -> shot-weighted empirical class measure + global meta-prior -> shrinkage posterior base measure -> residual posterior transport flow -> query-conditioned transport metric scoring + leave-one-shot-out prediction",
        "metric": "Hierarchical Predictive Posterior Transport Fit",
    },
    "spifce": {
        "display_name": "SPIFCE",
        "architecture": "Stable/variant token factorization -> stable gate -> global prototype + local top-r partial matching",
        "metric": "Cosine Prototype + Partial Token Matching",
    },
    "spifce_shot": {
        "display_name": "SPIFCE-Shot",
        "architecture": "SPIFCE with shot-preserving local top-r partial matching and query-adaptive shot aggregation",
        "metric": "Cosine Prototype + Shot-Preserving Partial Token Matching",
    },
    "spifce_asym_shot": {
        "display_name": "SPIFCE-AsymShot",
        "architecture": "Asymmetric SPIFCE: gated stable global prototypes + separate local token head + shot-preserving local top-r matching + adaptive fusion",
        "metric": "Adaptive Global Prototype + Shot-Preserving Partial Token Matching",
    },
    "spifaeb": {
        "display_name": "SPIFAEB",
        "architecture": "Stable/variant token factorization -> stable gate -> global prototype + adaptive evidence-budget sparse local matching",
        "metric": "Cosine Prototype + Adaptive Sparse Token Matching",
    },
    "spifaeb_shot": {
        "display_name": "SPIFAEB-Shot",
        "architecture": "Stable/variant token factorization -> stable gate -> global prototype + shot-preserving adaptive evidence-budget local matching",
        "metric": "Cosine Prototype + Shot-Preserving Adaptive Sparse Token Matching",
    },
    "spif_rdp": {
        "display_name": "SPIF-RDP",
        "architecture": "SPIF stable/variant encoder -> reliability-weighted distributional class object (prototype, diagonal variance, reliability) -> reliability-modulated global distance",
        "metric": "Reliability-Calibrated Distributional Prototype",
    },
    "spif_ota": {
        "display_name": "SPIF-OTA",
        "architecture": "Backbone tokens -> transport projector -> physics-aware token masses -> single-branch entropic OT matching -> shot-wise evidence aggregation",
        "metric": "Physics-Aware Entropic Optimal Transport",
    },
    "spif_otccls": {
        "display_name": "SPIF-OTCCLS",
        "architecture": "ResNet12 RGB backbone -> stable/variant factorization with feature-wise gate -> energy-weighted sliced Wasserstein global scoring + confusability-conditioned local cosine scoring",
        "metric": "Energy-Sliced Wasserstein + Confusability Local Cosine",
    },
    "spifaeb_v2": {
        "display_name": "SPIFAEB-v2",
        "architecture": "Stable/variant global branch + separate local token head + class-dispersion-aware rank-budget local matching + branch-supervised adaptive fusion",
        "metric": "Cosine Prototype + Monotonic Adaptive Evidence Matching",
    },
    "spifaeb_v3": {
        "display_name": "SPIFAEB-v3",
        "architecture": "Stable/variant SPIF encoder -> structured evidence global head with class center, scalar compactness, and support reliability + unchanged SPIFAEB-v2 local evidence branch",
        "metric": "Structured Global Evidence + Monotonic Adaptive Evidence Matching",
    },
    "spifmax": {
        "display_name": "SPIFMAX",
        "architecture": "SPIF core + consistency/decorr/sparse regularization",
        "metric": "Cosine Prototype + Partial Token Matching",
    },
    "spifce_local_sw": {
        "display_name": "SPIFCE-LocalSW",
        "architecture": "SPIF stable/variant factorization -> stable gate -> global cosine prototypes + local token-set sliced Wasserstein matching",
        "metric": "Cosine Prototype + Local Token-Set Sliced Wasserstein",
    },
    "spifmax_local_sw": {
        "display_name": "SPIFMAX-LocalSW",
        "architecture": "SPIF core + consistency/decorr/sparse regularization, with local token-set sliced Wasserstein matching",
        "metric": "Cosine Prototype + Local Token-Set Sliced Wasserstein",
    },
    "spifce_local_papersw": {
        "display_name": "SPIFCE-LocalPaperSW",
        "architecture": "SPIF stable/variant factorization -> stable gate -> global cosine prototypes + exact paper-style local sliced Wasserstein matching",
        "metric": "Cosine Prototype + Paper-Style Local Sliced Wasserstein",
    },
    "spifmax_local_papersw": {
        "display_name": "SPIFMAX-LocalPaperSW",
        "architecture": "SPIF core + consistency/decorr/sparse regularization, with exact paper-style local sliced Wasserstein matching",
        "metric": "Cosine Prototype + Paper-Style Local Sliced Wasserstein",
    },
    "spifharmce": {
        "display_name": "SPIF-HMambaCE",
        "architecture": "SPIF stable/variant encoder -> gated stable tokens -> global-only residual Mamba refinement + attentive pooling -> local paper-SW -> adaptive residual fusion",
        "metric": "Mamba Global Prototype + Paper-Style Local Sliced Wasserstein",
    },
    "spifharmmax": {
        "display_name": "SPIF-HMambaMAX",
        "architecture": "SPIF-HMambaCE + branch-specific CE and lightweight factorization / gate regularization",
        "metric": "Mamba Global Prototype + Paper-Style Local Sliced Wasserstein",
    },
    "spifmambace": {
        "display_name": "SPIFMambaCE",
        "architecture": "SPIF stable/variant factorization -> raw coarse stable gate -> residual official Mamba refinement -> final stable gate -> global prototype + local top-r partial matching",
        "metric": "Cosine Prototype + Partial Token Matching",
    },
    "spifmambamax": {
        "display_name": "SPIFMambaMAX",
        "architecture": "Adjusted SPIF-Mamba core + consistency/decorr/raw-gate-sparse/final-gate-sparse regularization",
        "metric": "Cosine Prototype + Partial Token Matching",
    },
    "spifmambaadjce": {
        "display_name": "SPIFMambaCE",
        "architecture": "SPIF stable/variant factorization -> raw coarse stable gate -> residual official Mamba refinement -> final stable gate -> global prototype + local top-r partial matching",
        "metric": "Cosine Prototype + Partial Token Matching",
    },
    "spifmambaadjmax": {
        "display_name": "SPIFMambaMAX",
        "architecture": "Adjusted SPIF-Mamba core + consistency/decorr/raw-gate-sparse/final-gate-sparse regularization",
        "metric": "Cosine Prototype + Partial Token Matching",
    },
    "spifv2ce": {
        "display_name": "SPIFv2CE",
        "architecture": "Shared bottleneck stable/variant split -> competitive token gate -> global prototype anchor + mutual local partial matching -> bounded residual local correction",
        "metric": "Cosine Prototype + Mutual Partial Token Matching",
    },
    "spifv2max": {
        "display_name": "SPIFv2MAX",
        "architecture": "SPIF-v2 core + consistency/decorr/competitive-gate regularization",
        "metric": "Cosine Prototype + Mutual Partial Token Matching",
    },
    "spifv3_cstn": {
        "display_name": "SPIFv3_CSTN",
        "architecture": "Unordered local descriptors -> canonical transport atoms -> permutation-aware support set aggregation -> sliced Wasserstein matching",
        "metric": "Canonical Atom Sliced Wasserstein Distance",
    },
    "spifv3_cstn_papersw": {
        "display_name": "SPIFv3_CSTN_PaperSW",
        "architecture": "Unordered local descriptors -> canonical transport atoms -> permutation-aware support set aggregation -> paper-style sliced Wasserstein matching",
        "metric": "Canonical Atom Paper-Style Sliced Wasserstein Distance",
    },
    "warn": {
        "display_name": "PARS-Net",
        "architecture": "ResNet12/Conv64F -> Transformer tokens -> support basis distillation -> SW prior retrieval -> prior-completed reconstruction subspace",
        "metric": "Prior-Augmented Reconstruction Subspace",
    },
    "pars_net": {
        "display_name": "PARS-Net",
        "architecture": "ResNet12/Conv64F -> Transformer tokens -> support basis distillation -> SW prior retrieval -> prior-completed reconstruction subspace",
        "metric": "Prior-Augmented Reconstruction Subspace",
    },
    "hrot_fsl": {
        "display_name": "HROT-FSL / Episode-Conditioned Optimal Transport J-FSL",
        "paper_name": "Episode-Conditioned Mass-Response Transport for Shot-Decomposed Few-Shot Learning",
        "architecture": "Backbone spatial tokens -> optional Euclidean projector -> Poincare-ball embedding -> balanced/unbalanced relational transport, with J_ECOT, J_ECOT_M2/CRS-M2, J_ECOT_CARE, J_ECOT_NNCS, CP_ECOT, and J_NCET fixed-budget support",
        "metric": "Hyperbolic Relational Optimal Transport",
    },
    "m2": {
        "display_name": "J-ECOT-M2 / SB-ECOT",
        "paper_name": "Single-Budget Episode-Calibrated Optimal Transport",
        "architecture": "Backbone spatial tokens -> shot-decomposed local measures -> single rho=0.80 KL-relaxed UOT -> threshold-calibrated evidence -> episode-calibrated shot pooling",
        "metric": "Single-Budget Episode-Calibrated UOT",
    },
    "m2_uot": {
        "display_name": "J-ECOT-M2-UOT",
        "paper_name": "Single-Budget Episode-Calibrated Optimal Transport",
        "architecture": "Alias for M2 with rho=0.80 KL-relaxed unbalanced OT",
        "metric": "Single-Budget Episode-Calibrated UOT",
    },
    "m2_full_ot": {
        "display_name": "J-ECOT-M2-FullOT",
        "paper_name": "Single-Budget Episode-Calibrated Optimal Transport",
        "architecture": "M2 counterfactual with the same shot-decomposed evidence scorer but balanced full-mass OT at rho=1.0",
        "metric": "Single-Budget Episode-Calibrated Balanced OT",
    },
    "jecot_m2": {
        "display_name": "J-ECOT-M2 / SB-ECOT",
        "paper_name": "Single-Budget Episode-Calibrated Optimal Transport",
        "architecture": "Alias for M2 with rho=0.80 KL-relaxed unbalanced OT",
        "metric": "Single-Budget Episode-Calibrated UOT",
    },
    "jecot_m2_uot": {
        "display_name": "J-ECOT-M2-UOT",
        "paper_name": "Single-Budget Episode-Calibrated Optimal Transport",
        "architecture": "Alias for M2 with rho=0.80 KL-relaxed unbalanced OT",
        "metric": "Single-Budget Episode-Calibrated UOT",
    },
    "jecot_m2_full_ot": {
        "display_name": "J-ECOT-M2-FullOT",
        "paper_name": "Single-Budget Episode-Calibrated Optimal Transport",
        "architecture": "Alias for the M2 balanced full-mass OT counterfactual at rho=1.0",
        "metric": "Single-Budget Episode-Calibrated Balanced OT",
    },
    "ours": {
        "display_name": "Ours",
        "paper_name": "Ours",
        "architecture": "Backbone local descriptors -> EGSM marginals -> single-budget mass-removed UOT -> episode-calibrated shot pooling; contribution ablations test full OT, no EGSM, and GAP descriptor cost + UOT + EGSM",
        "metric": "EGSM-Calibrated Single-Budget Token UOT",
    },
    "ours_final": {
        "display_name": "Ours-Final",
        "paper_name": "Ours-Final",
        "architecture": "Backbone local descriptors -> single-budget rho=0.8 UOT -> threshold-mass score T*M-C -> episode-calibrated shot pooling; EGSM is disabled by default",
        "metric": "Single-Budget Threshold-Mass Token UOT",
    },
    "ours_final_verified_uot": {
        "display_name": "Ours-Final Verified-UOT",
        "paper_name": "Neighborhood-Verified Evidence UOT",
        "architecture": "Backbone local descriptors -> single-budget rho=0.8 UOT -> neighborhood-verified positive evidence score -> episode-calibrated shot pooling",
        "metric": "Neighborhood-Verified Threshold-Mass Token UOT",
    },
    "ours_final_partial_ot": {
        "display_name": "Ours-Final-PartialOT",
        "paper_name": "Ours-Final Partial OT Ablation",
        "architecture": "Ours-Final ablation with fast partial OT at rho=0.8 and cost-per-transported-mass scoring; EGSM is disabled by default",
        "metric": "Fast Partial OT Cost-Per-Mass",
    },
    "ours_cpm": {
        "display_name": "Ours-CPM",
        "paper_name": "Ours (Cost-Per-Mass Multi-Budget)",
        "architecture": "Backbone spatial tokens -> EGSM marginals -> 3-budget UOT (rho=0.7,0.8,0.9) -> cost/mass scoring with detached denominator -> episode budget policy -> shot pooling",
        "metric": "Multi-Budget Cost-Per-Mass UOT",
    },
    "sg_pot": {
        "display_name": "SG-POT",
        "paper_name": "Saliency-Guided Partial Optimal Transport",
        "architecture": "Backbone local descriptors -> cross-conditioned saliency marginals -> adaptive mass fraction -> entropic partial OT -> dual evidence score (cost + residual Gini) -> shot pooling",
        "metric": "Saliency-Guided Entropic Partial Wasserstein Transport",
    },
    "ec_mrot": {
        "display_name": "EC-MROT",
        "paper_name": "Episode-Conditioned Mass-Response Optimal Transport",
        "architecture": "Backbone spatial tokens -> fixed-budget UOT mass-response path -> anchor-free path centering -> class-competitive signed entropic response functional",
        "metric": "Episode-Conditioned Mass-Response Optimal Transport",
    },
    "crj_fsl": {
        "display_name": "CRJ-FSL",
        "architecture": "J-FSL fixed-budget shot-decomposed UOT -> consensus-robust lower-confidence shot aggregation with diagnostics",
        "metric": "Consensus-Robust Threshold-Calibrated UOT",
    },
    "fgwuot_fsl": {
        "display_name": "FGWUOT-FSL",
        "architecture": "Backbone spatial tokens -> FGW-UOT per-shot matching -> J-style log-mean-exp shot pooling",
        "metric": "FGW Unbalanced OT with J-style Shot Pooling",
    },
    "cfuget": {
        "display_name": "RC-UOT",
        "paper_name": "Rival-Calibrated Unbalanced Evidence Transport",
        "architecture": "Backbone spatial tokens -> fixed-budget UOT -> rival-class coherent evidence gate -> threshold-mass evidence score",
        "metric": "Rival-Calibrated Coherent UOT Evidence",
    },
    "jsc_wdro": {
        "display_name": "JSC-WDRO",
        "architecture": "Backbone spatial tokens -> POT fixed-support Wasserstein/UOT barycenter per class -> adaptive WDRO radius -> POT/native robust query-class OT score",
        "metric": "Barycentric Wasserstein DRO Distance",
    },
    "ubt_fsl": {
        "display_name": "UBT-FSL",
        "architecture": "Backbone spatial tokens -> uncertain barycentric class object -> optional query ambiguity correction -> robust class transport scoring",
        "metric": "Uncertain Barycentric Transport",
    },
    "cbcr_fsl": {
        "display_name": "UBT-FSL (CBCR alias)",
        "architecture": "Backward-compatible alias for UBT-FSL: uncertain barycentric class object with robust class transport scoring",
        "metric": "Uncertain Barycentric Transport",
    },
    "transport_prior_replay_mamba_net": {
        "display_name": "TPR-MambaNet",
        "architecture": "Conv64F -> multi-atom support prior calibration -> replay-controlled query reader -> trajectory transport head",
        "metric": "Transport Prior Replay",
    },
    "transport_evidence_mamba_net": {
        "display_name": "TEM-Mamba",
        "architecture": "Conv64F -> token serialization -> prefix transport evidence -> selective evidence reader -> shared scalar scorer",
        "metric": "Transport Evidence Matching",
    },
}


def _build_sgpot_kwargs(args) -> dict:
    """Build constructor kwargs for SGPOT from CLI args."""
    return {
        "ours_ablation": "full",
        "pot_epsilon": float(getattr(args, "pot_epsilon", 0.05)),
        "pot_iterations": int(getattr(args, "pot_iterations", 100)),
        "pot_s_min": float(getattr(args, "pot_s_min", 0.3)),
        "pot_s_max": float(getattr(args, "pot_s_max", 0.8)),
        "pot_entropy_reg": float(getattr(args, "pot_entropy_reg", 0.05)),
        "pot_entropy_target": float(getattr(args, "pot_entropy_target", 0.7)),
        "pot_ablate_uniform_marginals": _bool_flag(
            getattr(args, "pot_ablate_uniform_marginals", "false"), default=False,
        ),
        "pot_ablate_fixed_s": (
            float(args.pot_ablate_fixed_s)
            if getattr(args, "pot_ablate_fixed_s", None) is not None
            else None
        ),
        "pot_ablate_cost_only_score": _bool_flag(
            getattr(args, "pot_ablate_cost_only_score", "false"), default=False,
        ),
        "pot_ablate_no_entropy_reg": _bool_flag(
            getattr(args, "pot_ablate_no_entropy_reg", "false"), default=False,
        ),
    }


def get_model_choices():
    return list(MODEL_REGISTRY.keys())


def get_model_metadata(model_name):
    if model_name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model: {model_name}")
    return MODEL_REGISTRY[model_name]


PAPER_BASELINE_BACKBONE_MODELS = frozenset(
    {
        "feat",
        "deepemd",
        "evidence_deepemd",
        "dice_emd",
        "transport_recon_emd",
        "tardis_emd",
        "can",
        "frn",
        "deepbdc",
    }
)
RESNET12_ONLY_MODELS = frozenset({"transport_recon_emd", "tardis_emd"})
HIGH_DIM_FEWSHOT_BACKBONES = frozenset({"resnet12", "fsl_mamba"})
EBOT_MODEL_NAMES = frozenset({"evidence_budget_ot", "evidence_budgeted_ot", "ebot", "ebot_scalogram"})
MM_SPOT_MODEL_NAMES = frozenset({"mm_spot_fsl"})
PARE_FSL_MODEL_NAMES = frozenset({"pare_fsl"})


def model_supports_fewshot_backbone(model_name, backbone_name):
    model_name = str(model_name)
    backbone_name = str(backbone_name).lower()
    if model_name in RESNET12_ONLY_MODELS:
        return backbone_name == "resnet12"
    if backbone_name != "fsl_mamba":
        return True
    return model_name not in PAPER_BASELINE_BACKBONE_MODELS


def fewshot_backbone_output_dim(backbone_name, fsl_mamba_output_dim=320):
    backbone_name = str(backbone_name).lower()
    if backbone_name == "fsl_mamba":
        return int(fsl_mamba_output_dim)
    if backbone_name in HIGH_DIM_FEWSHOT_BACKBONES:
        return 640
    return 64


def resolve_hrot_token_dim(args, default: int = 128) -> int:
    """Resolve transport token width; explicit ``hrot_token_dim`` wins over wide flag."""
    explicit = getattr(args, "hrot_token_dim", None)
    if explicit is not None:
        return int(explicit)
    if _bool_flag(getattr(args, "hrot_projector_wide", "false"), default=False):
        return int(getattr(args, "hrot_projector_wide_dim", 192))
    fallback = getattr(args, "token_dim", None)
    return int(fallback) if fallback is not None else int(default)


def resolve_fewshot_backbone(args):
    backbone = str(getattr(args, "fewshot_backbone", "resnet12")).lower()
    if backbone == "default":
        return "resnet12"
    if backbone == "slim_mamba":
        return "fsl_mamba"
    if backbone not in {"resnet12", "conv64f", "fsl_mamba"}:
        raise ValueError(f"Unsupported fewshot_backbone: {backbone}")
    return backbone


def build_model_from_args(args):
    validate_dmuot_scope(args)
    external_smnet_models = {
        "uscmamba",
        "conv64f_token_sw_metric_net",
        "class_memory_scan_mamba_net",
        "episodic_selective_scan_mamba_net",
        "permutation_robust_class_memory_mamba_net",
        "transport_prior_replay_mamba_net",
        "transport_evidence_mamba_net",
    }
    if args.model in external_smnet_models:
        raise NotImplementedError(
            f"{args.model} is benchmarked from the external smnet implementation. "
            "Use run_all_experiments.py to launch it from pulse_fewshot."
        )
    device = getattr(args, "device", "cuda")
    image_size = getattr(args, "image_size", 64)
    fewshot_backbone = resolve_fewshot_backbone(args)
    if not model_supports_fewshot_backbone(args.model, fewshot_backbone):
        if str(args.model) in RESNET12_ONLY_MODELS:
            raise ValueError(
                f"{args.model} requires fewshot_backbone=resnet12 to keep 640-dim dense maps"
            )
        raise ValueError(
            f"fewshot_backbone={fewshot_backbone} is only supported for SPIF/new repository models, "
            f"not the paper baseline `{args.model}`"
        )

    if args.model == "protonet":
        ProtoNet = _load_symbol("net.protonet", "ProtoNet")
        return ProtoNet(image_size=image_size, device=device)
    if args.model == "rada_fsl":
        RADAFSL = _load_symbol("net.rada_fsl", "RADAFSL")
        fsl_mamba_output_dim = int(getattr(args, "rada_fsl_mamba_output_dim", 320) or 320)
        hidden_dim = int(
            getattr(
                args,
                "rada_feat_dim",
                fewshot_backbone_output_dim(fewshot_backbone, fsl_mamba_output_dim=fsl_mamba_output_dim),
            )
            or fewshot_backbone_output_dim(fewshot_backbone, fsl_mamba_output_dim=fsl_mamba_output_dim)
        )
        disp_clamp_max = getattr(args, "rada_disp_clamp_max", None)
        if disp_clamp_max is not None and float(disp_clamp_max) <= 0.0:
            disp_clamp_max = None
        return RADAFSL(
            in_channels=3,
            hidden_dim=hidden_dim,
            backbone_name=fewshot_backbone,
            image_size=image_size,
            resnet12_drop_rate=0.0,
            resnet12_dropblock_size=5,
            fsl_mamba_base_dim=int(getattr(args, "rada_fsl_mamba_base_dim", 48) or 48),
            fsl_mamba_output_dim=fsl_mamba_output_dim,
            fsl_mamba_drop_path=float(getattr(args, "rada_fsl_mamba_drop_path", 0.02)),
            fsl_mamba_perturb_sigma=float(getattr(args, "rada_fsl_mamba_perturb_sigma", 0.0)),
            tau_r=float(getattr(args, "rada_tau_r", 0.5)),
            lambda_proto=float(getattr(args, "rada_lambda_proto", 0.7)),
            gamma_disp=float(getattr(args, "rada_gamma_disp", 0.5)),
            eps=float(getattr(args, "rada_eps", 1e-6)),
            reliability_head=str(getattr(args, "rada_reliability_head", "linear")),
            reliability_hidden_dim=getattr(args, "rada_reliability_hidden_dim", None),
            l2_normalize=_bool_flag(getattr(args, "rada_l2_normalize", "true"), default=True),
            entropy_reg_weight=float(getattr(args, "rada_entropy_reg_weight", 0.0)),
            use_reliability=_bool_flag(getattr(args, "rada_use_reliability", "true"), default=True),
            use_dispersion_metric=_bool_flag(
                getattr(args, "rada_use_dispersion_metric", "true"),
                default=True,
            ),
            query_conditioned=_bool_flag(getattr(args, "rada_query_conditioned", "true"), default=True),
            use_residual_anchor=_bool_flag(
                getattr(args, "rada_use_residual_anchor", "true"),
                default=True,
            ),
            use_shrinkage=_bool_flag(getattr(args, "rada_use_shrinkage", "true"), default=True),
            disp_clamp_max=disp_clamp_max,
            use_evidence_bound=_bool_flag(getattr(args, "rada_use_evidence_bound", "true"), default=True),
            evidence_temperature=float(getattr(args, "rada_evidence_temperature", 1.0)),
            min_reliability_mix=float(getattr(args, "rada_min_reliability_mix", 0.25)),
            dispersion_inflation=float(getattr(args, "rada_dispersion_inflation", 0.25)),
        )
    if args.model == "covamnet":
        CovaMNet = _load_symbol("net.covamnet", "CovaMNet")
        return CovaMNet(device=device)
    if args.model == "matchingnet":
        MatchingNet = _load_symbol("net.matchingnet", "MatchingNet")
        return MatchingNet(image_size=image_size, device=device)
    if args.model == "relationnet":
        RelationNet = _load_symbol("net.relationnet", "RelationNet")
        return RelationNet(image_size=image_size, device=device)
    if args.model == "dn4":
        DN4 = _load_symbol("net.dn4", "DN4")
        return DN4(k_neighbors=3, device=device)
    if args.model == "feat":
        FEAT = _load_symbol("net.feat", "FEAT")
        aux_balance = 0.01 if getattr(args, "shot_num", 1) == 1 else 0.1
        aux_temperature = 64.0 if getattr(args, "shot_num", 1) == 1 else 32.0
        return FEAT(
            image_size=image_size,
            temperature=64.0,
            temperature2=aux_temperature,
            aux_balance=aux_balance,
            fewshot_backbone=fewshot_backbone,
            device=device,
        )
    if args.model == "deepemd":
        DeepEMD = _load_symbol("net.deepemd", "DeepEMD")
        return DeepEMD(
            image_size=image_size,
            temperature=12.5,
            solver=getattr(args, "deepemd_solver", "sinkhorn"),
            qpth_form=getattr(args, "deepemd_qpth_form", "L2"),
            qpth_l2_strength=float(getattr(args, "deepemd_qpth_l2_strength", 1e-6)),
            sinkhorn_reg=float(getattr(args, "deepemd_sinkhorn_reg", 0.05)),
            sinkhorn_iterations=int(getattr(args, "deepemd_sinkhorn_iterations", 20)),
            sinkhorn_tolerance=float(getattr(args, "deepemd_sinkhorn_tolerance", 1e-6)),
            uot_tau_q=float(getattr(args, "deepemd_uot_tau_q", 0.5)),
            uot_tau_c=float(getattr(args, "deepemd_uot_tau_c", 0.5)),
            uot_score_normalize=_bool_flag(
                getattr(args, "deepemd_uot_score_normalize", "false"),
                default=False,
            ),
            partial_mass_fraction=float(getattr(args, "deepemd_partial_mass_fraction", 0.5)),
            partial_transport_mass=getattr(args, "deepemd_partial_transport_mass", None),
            partial_score_normalize=_bool_flag(
                getattr(args, "deepemd_partial_score_normalize", "true"),
                default=True,
            ),
            partial_backend=str(getattr(args, "deepemd_partial_backend", "fast")),
            partial_exact=_bool_flag(getattr(args, "deepemd_partial_exact", "false"), default=False),
            eps=float(getattr(args, "deepemd_eps", 1e-8)),
            sfc_lr=float(getattr(args, "deepemd_sfc_lr", 0.1)),
            sfc_update_step=int(getattr(args, "deepemd_sfc_update_step", 15)),
            sfc_bs=int(getattr(args, "deepemd_sfc_bs", 4)),
            fewshot_backbone=fewshot_backbone,
            device=device,
        )
    if args.model == "evidence_deepemd":
        EvidenceDeepEMD = _load_symbol("net.evidence_deepemd", "EvidenceDeepEMD")
        return EvidenceDeepEMD(
            image_size=image_size,
            temperature=12.5,
            solver=getattr(args, "deepemd_solver", "sinkhorn"),
            qpth_form=getattr(args, "deepemd_qpth_form", "L2"),
            qpth_l2_strength=float(getattr(args, "deepemd_qpth_l2_strength", 1e-6)),
            sinkhorn_reg=float(getattr(args, "deepemd_sinkhorn_reg", 0.05)),
            sinkhorn_iterations=int(getattr(args, "deepemd_sinkhorn_iterations", 20)),
            sinkhorn_tolerance=float(getattr(args, "deepemd_sinkhorn_tolerance", 1e-6)),
            uot_tau_q=float(getattr(args, "deepemd_uot_tau_q", 0.5)),
            uot_tau_c=float(getattr(args, "deepemd_uot_tau_c", 0.5)),
            uot_score_normalize=_bool_flag(
                getattr(args, "deepemd_uot_score_normalize", "false"),
                default=False,
            ),
            partial_mass_fraction=float(getattr(args, "deepemd_partial_mass_fraction", 0.5)),
            partial_transport_mass=getattr(args, "deepemd_partial_transport_mass", None),
            partial_score_normalize=_bool_flag(
                getattr(args, "deepemd_partial_score_normalize", "true"),
                default=True,
            ),
            partial_backend=str(getattr(args, "deepemd_partial_backend", "fast")),
            partial_exact=_bool_flag(getattr(args, "deepemd_partial_exact", "false"), default=False),
            eps=float(getattr(args, "deepemd_eps", 1e-8)),
            sfc_lr=float(getattr(args, "deepemd_sfc_lr", 0.1)),
            sfc_update_step=int(getattr(args, "deepemd_sfc_update_step", 15)),
            sfc_bs=int(getattr(args, "deepemd_sfc_bs", 4)),
            fewshot_backbone=fewshot_backbone,
            use_evidence_weight=_bool_flag(
                _first_attr(
                    args,
                    "use_evidence_weight",
                    "evidence_deepemd_use_evidence_weight",
                    default="true",
                ),
                default=True,
            ),
            use_shot_reliability=_bool_flag(
                _first_attr(
                    args,
                    "use_shot_reliability",
                    "evidence_deepemd_use_shot_reliability",
                    default="true",
                ),
                default=True,
            ),
            evidence_eps=float(
                _first_attr(args, "evidence_eps", "evidence_deepemd_evidence_eps", default=1e-3)
            ),
            consensus_scale=float(
                _first_attr(args, "consensus_scale", "evidence_deepemd_consensus_scale", default=5.0)
            ),
            shot_temperature=float(
                _first_attr(args, "shot_temperature", "evidence_deepemd_shot_temperature", default=1.0)
            ),
            debug_evidence=_bool_flag(
                _first_attr(args, "debug_evidence", "evidence_deepemd_debug_evidence", default="false"),
                default=False,
            ),
            device=device,
        )
    if args.model in EBOT_MODEL_NAMES:
        EvidenceBudgetedOTFewShot = _load_symbol("net.evidence_budget_ot", "EvidenceBudgetedOTFewShot")
        hidden_dim = fewshot_backbone_output_dim(fewshot_backbone)
        return EvidenceBudgetedOTFewShot(
            in_channels=3,
            hidden_dim=hidden_dim,
            proj_dim=int(_first_attr(args, "ebot_proj_dim", "evidence_budget_ot_proj_dim", default=256)),
            reliability_hidden_dim=int(getattr(args, "ebot_reliability_hidden_dim", 128)),
            reliability_dropout=float(getattr(args, "ebot_reliability_dropout", 0.1)),
            backbone_name=fewshot_backbone,
            image_size=image_size,
            resnet12_drop_rate=0.0,
            resnet12_dropblock_size=5,
            sinkhorn_epsilon=float(_first_attr(args, "ebot_sinkhorn_epsilon", "ebot_sinkhorn_eps", default=0.05)),
            sinkhorn_iters=int(_first_attr(args, "ebot_sinkhorn_iters", "ebot_sinkhorn_iterations", default=50)),
            dustbin_cost=float(getattr(args, "ebot_dustbin_cost", 0.7)),
            learnable_dustbin_cost=_bool_flag(
                getattr(args, "ebot_learnable_dustbin_cost", "true"),
                default=True,
            ),
            alpha_unmatched=float(getattr(args, "ebot_alpha_unmatched", 0.10)),
            min_budget=float(getattr(args, "ebot_min_budget", 0.15)),
            max_budget=float(getattr(args, "ebot_max_budget", 0.95)),
            gate_temperature=float(getattr(args, "ebot_gate_temperature", 1.0)),
            use_scalogram_priors=_bool_flag(getattr(args, "ebot_use_scalogram_priors", "true"), default=True),
            use_energy_prior=_bool_flag(getattr(args, "ebot_use_energy_prior", "true"), default=True),
            use_gradient_prior=_bool_flag(getattr(args, "ebot_use_gradient_prior", "true"), default=True),
            use_tf_coords=_bool_flag(getattr(args, "ebot_use_tf_coords", "true"), default=True),
            log_power_alpha=float(getattr(args, "ebot_log_power_alpha", 10.0)),
            prior_norm=str(getattr(args, "ebot_prior_norm", "mean")),
            use_cross_reference=_bool_flag(getattr(args, "ebot_use_cross_reference", "true"), default=True),
            use_dustbin=_bool_flag(getattr(args, "ebot_use_dustbin", "true"), default=True),
            use_evidence_budget=_bool_flag(getattr(args, "ebot_use_evidence_budget", "true"), default=True),
            use_kshot_reweighting=_bool_flag(getattr(args, "ebot_use_kshot_reweighting", "true"), default=True),
            lambda_score=float(getattr(args, "ebot_lambda_score", 1.0)),
            lambda_mass=float(getattr(args, "ebot_lambda_mass", 0.5)),
            lambda_unmatched=float(getattr(args, "ebot_lambda_unmatched", 0.5)),
            score_scale=float(getattr(args, "ebot_score_scale", 12.5)),
            learnable_score_scale=_bool_flag(
                getattr(args, "ebot_learnable_score_scale", "false"),
                default=False,
            ),
            symmetric_matching=_bool_flag(getattr(args, "ebot_symmetric_matching", "false"), default=False),
            use_uncertainty_prior=_bool_flag(getattr(args, "ebot_use_uncertainty_prior", "false"), default=False),
            use_aux_loss=_bool_flag(getattr(args, "ebot_use_aux_loss", "false"), default=False),
            budget_floor=float(getattr(args, "ebot_budget_floor", 0.15)),
            budget_ceiling=float(getattr(args, "ebot_budget_ceiling", 0.95)),
            weight_budget_low=float(getattr(args, "ebot_weight_budget_low", 0.01)),
            weight_budget_high=float(getattr(args, "ebot_weight_budget_high", 0.001)),
            eps=float(getattr(args, "ebot_eps", 1e-6)),
        )
    if args.model in PARE_FSL_MODEL_NAMES:
        PAREFSL = _load_symbol("net.pare_fsl", "PAREFSL")
        hidden_dim = fewshot_backbone_output_dim(fewshot_backbone)
        return PAREFSL(
            in_channels=3,
            hidden_dim=hidden_dim,
            token_dim=int(getattr(args, "token_dim", 128) or 128),
            backbone_name=fewshot_backbone,
            image_size=image_size,
            resnet12_drop_rate=0.0,
            resnet12_dropblock_size=5,
            sinkhorn_epsilon=float(getattr(args, "pare_sinkhorn_epsilon", 0.04)),
            sinkhorn_iterations=int(getattr(args, "pare_sinkhorn_iterations", 80)),
            sinkhorn_tolerance=float(getattr(args, "pare_sinkhorn_tolerance", 1e-6)),
            score_scale=float(getattr(args, "pare_score_scale", 16.0)),
            marginal_mode=str(getattr(args, "pare_marginal_mode", "egsm")),
            egsm_hidden_dim=int(getattr(args, "pare_egsm_hidden_dim", 32)),
            egsm_kappa_min=float(getattr(args, "pare_egsm_kappa_min", 0.05)),
            egsm_kappa_max=float(getattr(args, "pare_egsm_kappa_max", 0.35)),
            egsm_candidate_tau_q=float(getattr(args, "pare_egsm_candidate_tau_q", 1.0)),
            egsm_candidate_tau_b=float(getattr(args, "pare_egsm_candidate_tau_b", 1.0)),
            enable_learned_alpha=_bool_flag(getattr(args, "pare_enable_learned_alpha", "true"), default=True),
            alpha_min=float(getattr(args, "pare_alpha_min", 0.30)),
            alpha_max=float(getattr(args, "pare_alpha_max", 0.70)),
            alpha_prior=float(getattr(args, "pare_alpha_prior", 0.50)),
            fixed_alpha=getattr(args, "pare_fixed_alpha", None),
            alpha_reg_lambda=float(getattr(args, "pare_alpha_reg_lambda", 0.01)),
            alpha_head_hidden_dim=int(getattr(args, "pare_alpha_head_hidden_dim", 32)),
            shot_pooling=str(getattr(args, "pare_shot_pooling", "discrepancy")),
            shot_temperature_init=float(getattr(args, "pare_shot_temperature", 1.0)),
            eps=float(getattr(args, "pare_eps", 1e-8)),
        )
    if args.model in MM_SPOT_MODEL_NAMES:
        MMSPOTFSL = _load_symbol("net.mm_spot_fsl", "MMSPOTFSL")
        hidden_dim = fewshot_backbone_output_dim(fewshot_backbone)
        return MMSPOTFSL(
            in_channels=3,
            hidden_dim=hidden_dim,
            token_dim=int(getattr(args, "token_dim", 128) or 128),
            backbone_name=fewshot_backbone,
            image_size=image_size,
            resnet12_drop_rate=0.0,
            resnet12_dropblock_size=5,
            spot_mass_bank=str(getattr(args, "spot_mass_bank", "0.3,0.4,0.5")),
            sinkhorn_epsilon=float(getattr(args, "spot_sinkhorn_epsilon", 0.05)),
            sinkhorn_iterations=int(getattr(args, "spot_sinkhorn_iterations", 120)),
            sinkhorn_tolerance=float(getattr(args, "spot_sinkhorn_tolerance", 1e-6)),
            score_scale=float(getattr(args, "spot_score_scale", 16.0)),
            mass_aggregation=str(getattr(args, "spot_mass_aggregation", "softmax")),
            mass_temperature=float(getattr(args, "spot_mass_temperature", 1.0)),
            partial_backend=str(getattr(args, "spot_partial_backend", "fast")),
            use_controller=_bool_flag(getattr(args, "spot_use_controller", "false"), default=False),
            controller_hidden_dim=int(getattr(args, "spot_controller_hidden_dim", 32)),
            proto_weight=float(getattr(args, "spot_proto_weight", 0.0)),
            support_merge_mode=str(getattr(args, "spot_support_merge_mode", "concat")),
            return_transport_plan=_bool_flag(
                getattr(args, "spot_return_transport_plan", "false"),
                default=False,
            ),
            eps=float(getattr(args, "spot_eps", 1e-8)),
        )
    if args.model == "dice_emd":
        DICEEMD = _load_symbol("net.dice_emd", "DICEEMD")
        return DICEEMD(
            image_size=image_size,
            temperature=12.5,
            solver=getattr(args, "deepemd_solver", "sinkhorn"),
            qpth_form=getattr(args, "deepemd_qpth_form", "L2"),
            qpth_l2_strength=float(getattr(args, "deepemd_qpth_l2_strength", 1e-6)),
            sinkhorn_reg=float(getattr(args, "deepemd_sinkhorn_reg", 0.05)),
            sinkhorn_iterations=int(getattr(args, "deepemd_sinkhorn_iterations", 20)),
            sinkhorn_tolerance=float(getattr(args, "deepemd_sinkhorn_tolerance", 1e-6)),
            eps=float(getattr(args, "deepemd_eps", 1e-8)),
            sfc_lr=float(getattr(args, "deepemd_sfc_lr", 0.1)),
            sfc_update_step=int(getattr(args, "deepemd_sfc_update_step", 15)),
            sfc_bs=int(getattr(args, "deepemd_sfc_bs", 4)),
            fewshot_backbone=fewshot_backbone,
            lambda_disc=float(_first_attr(args, "dice_emd_lambda_disc", "lambda_disc", default=0.2)),
            tau_comp=float(_first_attr(args, "dice_emd_tau_comp", "tau_comp", default=0.1)),
            use_softmin_distance=_bool_flag(
                _first_attr(args, "dice_emd_use_softmin_distance", "use_softmin_distance", default="true"),
                default=True,
            ),
            tau_softmin=float(_first_attr(args, "dice_emd_tau_softmin", "tau_softmin", default=0.1)),
            debug_transport=_bool_flag(
                _first_attr(args, "dice_emd_debug_transport", "debug_transport", default="false"),
                default=False,
            ),
            lambda_flow=float(_first_attr(args, "dice_emd_lambda_flow", "lambda_flow", default=0.0)),
            device=device,
        )
    if args.model in {"transport_recon_emd", "tardis_emd"}:
        TransportReconEMD = _load_symbol("net.transport_recon_emd", "TransportReconEMD")
        return TransportReconEMD(
            image_size=image_size,
            fewshot_backbone=fewshot_backbone,
            tau_shot=float(
                _first_attr(
                    args,
                    "transport_recon_emd_tau_shot",
                    "tardis_emd_tau_shot",
                    "tau_shot",
                    default=0.2,
                )
            ),
            kshot_mode=str(
                _first_attr(
                    args,
                    "transport_recon_emd_kshot_mode",
                    "tardis_emd_kshot_mode",
                    "kshot_mode",
                    default="query_condense",
                )
            ),
            sinkhorn_reg=float(getattr(args, "deepemd_sinkhorn_reg", 0.05)),
            sinkhorn_iter=int(getattr(args, "deepemd_sinkhorn_iterations", 20)),
            sinkhorn_tolerance=float(getattr(args, "deepemd_sinkhorn_tolerance", 1e-6)),
            lambda_emd=float(
                _first_attr(
                    args,
                    "transport_recon_emd_lambda_emd",
                    "tardis_emd_lambda_emd",
                    "lambda_emd",
                    default=0.3,
                )
            ),
            lambda_rec=float(
                _first_attr(
                    args,
                    "transport_recon_emd_lambda_rec",
                    "tardis_emd_lambda_rec",
                    "lambda_rec",
                    default=1.0,
                )
            ),
            lambda_struct=float(
                _first_attr(
                    args,
                    "transport_recon_emd_lambda_struct",
                    "tardis_emd_lambda_struct",
                    "lambda_struct",
                    default=0.1,
                )
            ),
            score_scale=float(
                _first_attr(
                    args,
                    "transport_recon_emd_score_scale",
                    "tardis_emd_score_scale",
                    "score_scale",
                    default=8.0,
                )
            ),
            use_sfc=_bool_flag(
                _first_attr(
                    args,
                    "transport_recon_emd_use_sfc",
                    "tardis_emd_use_sfc",
                    "use_sfc",
                    default="false",
                ),
                default=False,
            ),
            normalize_reconstruction=_bool_flag(
                _first_attr(
                    args,
                    "transport_recon_emd_normalize_reconstruction",
                    "tardis_emd_normalize_reconstruction",
                    "normalize_reconstruction",
                    default="true",
                ),
                default=True,
            ),
            debug_shapes=_bool_flag(
                _first_attr(
                    args,
                    "transport_recon_emd_debug_shapes",
                    "tardis_emd_debug_shapes",
                    "debug_shapes",
                    default="false",
                ),
                default=False,
            ),
            eps=float(getattr(args, "deepemd_eps", 1e-8)),
            device=device,
        )
    if args.model == "can":
        CrossAttentionNet = _load_symbol("net.can", "CrossAttentionNet")
        return CrossAttentionNet(
            image_size=image_size,
            way_num=getattr(args, "way_num", 4),
            cam_temperature=0.025,
            loss_weight=0.5,
            reduction_ratio=6,
            fewshot_backbone=fewshot_backbone,
            device=device,
        )
    if args.model == "frn":
        FRN = _load_symbol("net.frn", "FRN")
        return FRN(image_size=image_size, fewshot_backbone=fewshot_backbone, device=device)
    if args.model == "deepbdc":
        DeepBDC = _load_symbol("net.deepbdc", "DeepBDC")
        reduce_dim = fewshot_backbone_output_dim(fewshot_backbone)
        return DeepBDC(
            image_size=image_size,
            reduce_dim=reduce_dim,
            fewshot_backbone=fewshot_backbone,
            device=device,
        )
    if args.model == "maml":
        MAMLNet = _load_symbol("net.maml", "MAMLNet")
        return MAMLNet(
            image_size=image_size,
            inner_lr=0.01,
            inner_steps=5,
            first_order=not getattr(args, "maml_second_order", False),
            max_way_num=32,
            device=device,
        )
    if args.model == "hierarchical_episodic_ssm_net":
        HierarchicalEpisodicSSMNet = _load_symbol(
            "net.hierarchical_episodic_ssm_net",
            "HierarchicalEpisodicSSMNet",
        )
        return HierarchicalEpisodicSSMNet(
            in_channels=3,
            hidden_dim=64,
            token_dim=getattr(args, "hierarchical_token_dim", 64) or 64,
            ssm_state_dim=getattr(args, "ssm_state_dim", 16),
            temperature=float(getattr(args, "hierarchical_temperature", 16.0)),
            backbone_name=fewshot_backbone,
            image_size=image_size,
            resnet12_drop_rate=0.0,
            resnet12_dropblock_size=5,
            use_sw=_bool_flag(getattr(args, "hierarchical_use_sw", "true"), default=True),
            sw_weight=float(getattr(args, "hierarchical_sw_weight", 0.25)),
            sw_num_projections=int(getattr(args, "hierarchical_sw_num_projections", 64)),
            sw_p=float(getattr(args, "hierarchical_sw_p", 2.0)),
            sw_normalize=_bool_flag(getattr(args, "hierarchical_sw_normalize", "true"), default=True),
            token_merge_mode=getattr(args, "hierarchical_token_merge_mode", "concat"),
            hierarchical_token_depth=int(getattr(args, "hierarchical_token_depth", 1)),
            hierarchical_shot_depth=int(getattr(args, "hierarchical_shot_depth", 1)),
            device=device,
        )
    if args.model == "hierarchical_consensus_slot_mamba_net":
        HierarchicalConsensusSlotMambaNet = _load_symbol(
            "net.hierarchical_consensus_slot_mamba_net",
            "HierarchicalConsensusSlotMambaNet",
        )
        return HierarchicalConsensusSlotMambaNet(
            in_channels=3,
            hidden_dim=64,
            token_dim=getattr(args, "hcsm_token_dim", getattr(args, "token_dim", 128)) or 128,
            ssm_state_dim=getattr(args, "ssm_state_dim", 16),
            temperature=float(getattr(args, "hcsm_temperature", 16.0)),
            backbone_name=fewshot_backbone,
            image_size=image_size,
            resnet12_drop_rate=0.0,
            resnet12_dropblock_size=5,
            use_sw=_bool_flag(getattr(args, "hcsm_use_sw", "true"), default=True),
            sw_weight=float(getattr(args, "hcsm_sw_weight", 0.25)),
            sw_num_projections=int(getattr(args, "hcsm_sw_num_projections", 64)),
            sw_p=float(getattr(args, "hcsm_sw_p", 2.0)),
            sw_normalize=_bool_flag(getattr(args, "hcsm_sw_normalize", "true"), default=True),
            intra_image_depth=int(getattr(args, "hcsm_intra_image_depth", 1)),
            num_write_tokens=int(getattr(args, "hcsm_num_write_tokens", 4)),
            num_transport_tokens=int(getattr(args, "hcsm_num_transport_tokens", 12)),
            num_consensus_slots=int(getattr(args, "hcsm_num_consensus_slots", 4)),
            pool_heads=int(getattr(args, "hcsm_pool_heads", 4)),
            slot_mamba_depth=int(getattr(args, "hcsm_slot_mamba_depth", 1)),
            mamba_d_conv=int(getattr(args, "hcsm_mamba_d_conv", 4)),
            mamba_expand=int(getattr(args, "hcsm_mamba_expand", 2)),
            mamba_dropout=float(getattr(args, "hcsm_mamba_dropout", 0.0)),
            sw_merged_weight=1.0,
            sw_shot_weight=0.0,
            use_shot_refiner=_bool_flag(getattr(args, "hcsm_use_shot_refiner", "true"), default=True),
            use_reliability_residual=_bool_flag(
                getattr(args, "hcsm_use_reliability_residual", "true"), default=True
            ),
            use_token_adapter=_bool_flag(getattr(args, "hcsm_use_token_adapter", "true"), default=True),
            use_local_gate=_bool_flag(getattr(args, "hcsm_use_local_gate", "true"), default=True),
        )
    if args.model == "hierarchical_support_sw_net":
        HierarchicalSupportSlicedWassersteinNet = _load_symbol(
            "net.hierarchical_support_sw_net",
            "HierarchicalSupportSlicedWassersteinNet",
        )
        return HierarchicalSupportSlicedWassersteinNet(
            in_channels=3,
            hidden_dim=64,
            token_weight_hidden=int(getattr(args, "hssw_token_weight_hidden", 128)),
            token_weight_temperature=float(getattr(args, "hssw_token_weight_temperature", 1.0)),
            token_weight_mix_alpha=float(getattr(args, "hssw_token_weight_mix_alpha", 1.0)),
            token_mass_reg_weight=float(getattr(args, "hssw_token_mass_reg_weight", 0.0)),
            multiscale_grids=_parse_csv_ints(getattr(args, "hssw_multiscale_grids", "4,1"), default=(4, 1)),
            multiscale_mass_budgets=_parse_csv_floats(
                getattr(args, "hssw_multiscale_mass_budgets", "0.6,0.3,0.1"),
                default=(0.6, 0.3, 0.1),
            ),
            weighted_sw=_bool_flag(getattr(args, "hssw_weighted_sw", "true"), default=True),
            normalize_tokens_before_sw=_bool_flag(getattr(args, "hssw_token_l2norm", "true"), default=True),
            backbone_name=fewshot_backbone,
            image_size=image_size,
            resnet12_drop_rate=0.0,
            resnet12_dropblock_size=5,
            train_num_projections=int(getattr(args, "hssw_train_num_projections", 32)),
            eval_num_projections=int(getattr(args, "hssw_eval_num_projections", 64)),
            sw_p=float(getattr(args, "hssw_sw_p", 2.0)),
            train_projection_mode=str(getattr(args, "hssw_train_projection_mode", "resample")),
            eval_projection_mode=str(getattr(args, "hssw_eval_projection_mode", "fixed")),
            eval_num_repeats=int(getattr(args, "hssw_eval_num_repeats", 1)),
            projection_seed=int(getattr(args, "hssw_projection_seed", 7)),
            pairwise_chunk_size=(
                None
                if int(getattr(args, "hssw_pairwise_chunk_size", 0)) <= 0
                else int(getattr(args, "hssw_pairwise_chunk_size", 0))
            ),
            tau=float(getattr(args, "hssw_tau", 0.2)),
            lambda_cons=float(getattr(args, "hssw_lambda_cons", 0.1)),
            lambda_red=float(getattr(args, "hssw_lambda_red", 0.02)),
            learn_logit_scale=_bool_flag(getattr(args, "hssw_learn_logit_scale", "false"), default=False),
            logit_scale_init=float(getattr(args, "hssw_logit_scale_init", 10.0)),
            logit_scale_max=float(getattr(args, "hssw_logit_scale_max", 100.0)),
            eps=float(getattr(args, "hssw_eps", 1e-6)),
        )
    if args.model in TA_DSW_MODEL_NAMES:
        TADSW = _load_symbol("net.ta_dsw", "TaskAdaptiveDistributionalSlicedWassersteinNet")
        hidden_dim = fewshot_backbone_output_dim(fewshot_backbone)
        return TADSW(
            in_channels=3,
            hidden_dim=hidden_dim,
            token_dim=int(getattr(args, "ta_dsw_token_dim", 128)),
            backbone_name=fewshot_backbone,
            image_size=image_size,
            resnet12_drop_rate=0.0,
            resnet12_dropblock_size=5,
            num_slices=int(getattr(args, "ta_dsw_num_slices", 64)),
            sw_p=float(getattr(args, "ta_dsw_sw_p", 2.0)),
            temperature=float(getattr(args, "ta_dsw_temperature", 0.1)),
            task_hidden_dim=int(getattr(args, "ta_dsw_hidden_dim", 256)),
            token_weight_hidden_dim=int(getattr(args, "ta_dsw_token_weight_hidden_dim", 64)),
            token_weight_uniform_mix=float(getattr(args, "ta_dsw_token_weight_uniform_mix", 0.2)),
            num_quantiles=int(getattr(args, "ta_dsw_num_quantiles", 256)),
            normalize_tokens=_bool_flag(getattr(args, "ta_dsw_normalize_tokens", "true"), default=True),
            train_projection_mode=str(getattr(args, "ta_dsw_train_projection_mode", "deterministic")),
            eval_projection_mode=str(getattr(args, "ta_dsw_eval_projection_mode", "fixed")),
            projection_seed=int(getattr(args, "ta_dsw_projection_seed", 7)),
            shot_aggregation=str(getattr(args, "ta_dsw_shot_aggregation", "softmin")),
            shot_softmin_beta=float(getattr(args, "ta_dsw_shot_softmin_beta", 5.0)),
            proto_scale_init=float(getattr(args, "ta_dsw_proto_scale_init", 10.0)),
            sw_scale_init=(
                None
                if getattr(args, "ta_dsw_sw_scale_init", None) is None
                else float(getattr(args, "ta_dsw_sw_scale_init"))
            ),
            learnable_scales=_bool_flag(getattr(args, "ta_dsw_learnable_scales", "true"), default=True),
            eps=float(getattr(args, "ta_dsw_eps", 1e-8)),
        )
    if args.model == "sc_lfi":
        SupportConditionedLatentFlowInferenceNet = _load_symbol(
            "net.sc_lfi",
            "SupportConditionedLatentFlowInferenceNet",
        )
        return SupportConditionedLatentFlowInferenceNet(
            in_channels=3,
            hidden_dim=64,
            latent_dim=int(getattr(args, "latent_dim", 64)),
            context_dim=int(getattr(args, "sc_lfi_context_dim", 128)),
            context_hidden_dim=(
                None
                if getattr(args, "sc_lfi_context_hidden_dim", None) is None
                else int(getattr(args, "sc_lfi_context_hidden_dim"))
            ),
            latent_hidden_dim=(
                None
                if getattr(args, "sc_lfi_latent_hidden_dim", None) is None
                else int(getattr(args, "sc_lfi_latent_hidden_dim"))
            ),
            flow_hidden_dim=int(getattr(args, "sc_lfi_flow_hidden_dim", 128)),
            flow_time_embedding_dim=int(getattr(args, "sc_lfi_flow_time_embedding_dim", 32)),
            backbone_name=fewshot_backbone,
            image_size=image_size,
            resnet12_drop_rate=0.0,
            resnet12_dropblock_size=5,
            class_context_type=str(getattr(args, "class_context_type", "deepsets")),
            flow_conditioning_type=str(getattr(args, "flow_conditioning_type", "concat")),
            distance_type=str(getattr(args, "distance_type", "sw")),
            use_global_proto_branch=_bool_flag(getattr(args, "use_global_proto_branch", "false"), default=False),
            use_flow_branch=_bool_flag(getattr(args, "use_flow_branch", "true"), default=True),
            use_align_loss=_bool_flag(getattr(args, "use_align_loss", "true"), default=True),
            use_smooth_loss=_bool_flag(getattr(args, "use_smooth_loss", "false"), default=False),
            num_flow_particles=int(getattr(args, "num_flow_particles", 16)),
            num_flow_integration_steps=int(getattr(args, "sc_lfi_num_flow_integration_steps", 8)),
            fm_time_schedule=str(getattr(args, "fm_time_schedule", "uniform")),
            score_temperature=float(getattr(args, "score_temperature", 8.0)),
            proto_branch_weight=float(getattr(args, "sc_lfi_proto_branch_weight", 0.2)),
            lambda_fm=float(getattr(args, "sc_lfi_lambda_fm", 0.05)),
            lambda_align=float(getattr(args, "sc_lfi_lambda_align", 0.1)),
            lambda_smooth=float(getattr(args, "sc_lfi_lambda_smooth", 0.0)),
            distance_normalize_inputs=_bool_flag(
                getattr(args, "sc_lfi_distance_normalize_inputs", "true"),
                default=True,
            ),
            distance_sw_num_projections=int(getattr(args, "sc_lfi_distance_sw_num_projections", 64)),
            distance_sw_p=float(getattr(args, "sc_lfi_distance_sw_p", 2.0)),
            distance_projection_seed=int(getattr(args, "sc_lfi_distance_projection_seed", 7)),
            sinkhorn_epsilon=float(getattr(args, "sc_lfi_sinkhorn_epsilon", 0.1)),
            sinkhorn_iterations=int(getattr(args, "sc_lfi_sinkhorn_iterations", 50)),
            context_num_memory_tokens=int(getattr(args, "sc_lfi_context_num_memory_tokens", 4)),
            context_num_heads=int(getattr(args, "sc_lfi_context_num_heads", 4)),
            context_ffn_multiplier=int(getattr(args, "sc_lfi_context_ffn_multiplier", 2)),
            eval_particle_seed=int(getattr(args, "sc_lfi_eval_particle_seed", 7)),
            eps=float(getattr(args, "sc_lfi_eps", 1e-6)),
        )
    if args.model == "sc_lfi_v2":
        SupportConditionedLatentFlowInferenceNetV2 = _load_symbol(
            "net.sc_lfi_v2",
            "SupportConditionedLatentFlowInferenceNetV2",
        )
        return SupportConditionedLatentFlowInferenceNetV2(
            in_channels=3,
            hidden_dim=64,
            latent_dim=int(getattr(args, "latent_dim", 64)),
            context_dim=int(getattr(args, "sc_lfi_v2_context_dim", 128)),
            context_hidden_dim=(
                None
                if getattr(args, "sc_lfi_v2_context_hidden_dim", None) is None
                else int(getattr(args, "sc_lfi_v2_context_hidden_dim"))
            ),
            latent_hidden_dim=(
                None
                if getattr(args, "sc_lfi_v2_latent_hidden_dim", None) is None
                else int(getattr(args, "sc_lfi_v2_latent_hidden_dim"))
            ),
            mass_hidden_dim=(
                None
                if getattr(args, "sc_lfi_v2_mass_hidden_dim", None) is None
                else int(getattr(args, "sc_lfi_v2_mass_hidden_dim"))
            ),
            mass_temperature=float(getattr(args, "sc_lfi_v2_mass_temperature", 1.0)),
            memory_size=int(getattr(args, "sc_lfi_v2_memory_size", 4)),
            memory_num_heads=int(getattr(args, "sc_lfi_v2_memory_num_heads", 4)),
            memory_ffn_multiplier=int(getattr(args, "sc_lfi_v2_memory_ffn_multiplier", 2)),
            flow_hidden_dim=int(getattr(args, "sc_lfi_v2_flow_hidden_dim", 128)),
            flow_time_embedding_dim=int(getattr(args, "sc_lfi_v2_flow_time_embedding_dim", 32)),
            flow_memory_num_heads=int(getattr(args, "sc_lfi_v2_flow_memory_num_heads", 4)),
            backbone_name=fewshot_backbone,
            image_size=image_size,
            resnet12_drop_rate=0.0,
            resnet12_dropblock_size=5,
            flow_conditioning_type=str(getattr(args, "sc_lfi_v2_flow_conditioning_type", "film")),
            use_global_proto_branch=_bool_flag(
                getattr(args, "sc_lfi_v2_use_global_proto_branch", "false"),
                default=False,
            ),
            use_flow_branch=_bool_flag(getattr(args, "sc_lfi_v2_use_flow_branch", "true"), default=True),
            use_align_loss=_bool_flag(getattr(args, "sc_lfi_v2_use_align_loss", "true"), default=True),
            use_margin_loss=_bool_flag(getattr(args, "sc_lfi_v2_use_margin_loss", "true"), default=True),
            use_smooth_loss=_bool_flag(getattr(args, "sc_lfi_v2_use_smooth_loss", "false"), default=False),
            train_num_flow_particles=int(getattr(args, "sc_lfi_v2_train_num_flow_particles", 8)),
            eval_num_flow_particles=int(getattr(args, "sc_lfi_v2_eval_num_flow_particles", 16)),
            train_num_integration_steps=int(getattr(args, "sc_lfi_v2_train_num_integration_steps", 8)),
            eval_num_integration_steps=int(getattr(args, "sc_lfi_v2_eval_num_integration_steps", 16)),
            solver_type=str(getattr(args, "sc_lfi_v2_solver_type", "heun")),
            fm_time_schedule=str(getattr(args, "fm_time_schedule", "uniform")),
            score_temperature=float(getattr(args, "score_temperature", 8.0)),
            proto_branch_weight=float(getattr(args, "sc_lfi_v2_proto_branch_weight", 0.15)),
            lambda_fm=float(getattr(args, "sc_lfi_v2_lambda_fm", 0.05)),
            lambda_align=float(getattr(args, "sc_lfi_v2_lambda_align", 0.1)),
            lambda_margin=float(getattr(args, "sc_lfi_v2_lambda_margin", 0.1)),
            lambda_support_fit=float(getattr(args, "sc_lfi_v2_lambda_support_fit", 0.1)),
            lambda_smooth=float(getattr(args, "sc_lfi_v2_lambda_smooth", 0.0)),
            margin_value=float(getattr(args, "sc_lfi_v2_margin_value", 0.1)),
            support_mix_min=float(getattr(args, "sc_lfi_v2_support_mix_min", 0.45)),
            support_mix_max=float(getattr(args, "sc_lfi_v2_support_mix_max", 0.8)),
            score_train_num_projections=int(getattr(args, "sc_lfi_v2_score_train_num_projections", 64)),
            score_eval_num_projections=int(getattr(args, "sc_lfi_v2_score_eval_num_projections", 128)),
            score_sw_p=float(getattr(args, "sc_lfi_v2_score_sw_p", 2.0)),
            score_normalize_inputs=_bool_flag(
                getattr(args, "sc_lfi_v2_score_normalize_inputs", "true"),
                default=True,
            ),
            score_train_projection_mode=str(getattr(args, "sc_lfi_v2_score_train_projection_mode", "resample")),
            score_eval_projection_mode=str(getattr(args, "sc_lfi_v2_score_eval_projection_mode", "fixed")),
            score_eval_num_repeats=int(getattr(args, "sc_lfi_v2_score_eval_num_repeats", 1)),
            score_projection_seed=int(getattr(args, "sc_lfi_v2_score_projection_seed", 7)),
            align_distance_type=str(getattr(args, "sc_lfi_v2_align_distance_type", "weighted_entropic_ot")),
            align_train_num_projections=int(getattr(args, "sc_lfi_v2_align_train_num_projections", 64)),
            align_eval_num_projections=int(getattr(args, "sc_lfi_v2_align_eval_num_projections", 128)),
            align_sw_p=float(getattr(args, "sc_lfi_v2_align_sw_p", 2.0)),
            align_normalize_inputs=_bool_flag(
                getattr(args, "sc_lfi_v2_align_normalize_inputs", "true"),
                default=True,
            ),
            align_train_projection_mode=str(getattr(args, "sc_lfi_v2_align_train_projection_mode", "resample")),
            align_eval_projection_mode=str(getattr(args, "sc_lfi_v2_align_eval_projection_mode", "fixed")),
            align_eval_num_repeats=int(getattr(args, "sc_lfi_v2_align_eval_num_repeats", 1)),
            align_projection_seed=int(getattr(args, "sc_lfi_v2_align_projection_seed", 11)),
            sinkhorn_epsilon=float(getattr(args, "sc_lfi_v2_sinkhorn_epsilon", 0.05)),
            sinkhorn_iterations=int(getattr(args, "sc_lfi_v2_sinkhorn_iterations", 80)),
            sinkhorn_cost_power=float(getattr(args, "sc_lfi_v2_sinkhorn_cost_power", 2.0)),
            eval_particle_seed=int(getattr(args, "sc_lfi_v2_eval_particle_seed", 7)),
            eps=float(getattr(args, "sc_lfi_v2_eps", 1e-8)),
        )
    if args.model == "sc_lfi_v3":
        SupportConditionedLatentFlowInferenceNetV3 = _load_symbol(
            "net.sc_lfi_v3",
            "SupportConditionedLatentFlowInferenceNetV3",
        )
        return SupportConditionedLatentFlowInferenceNetV3(
            in_channels=3,
            hidden_dim=64,
            latent_dim=int(getattr(args, "latent_dim", 64)),
            context_dim=int(getattr(args, "sc_lfi_v3_context_dim", 128)),
            context_hidden_dim=(
                None
                if getattr(args, "sc_lfi_v3_context_hidden_dim", None) is None
                else int(getattr(args, "sc_lfi_v3_context_hidden_dim"))
            ),
            latent_hidden_dim=(
                None
                if getattr(args, "sc_lfi_v3_latent_hidden_dim", None) is None
                else int(getattr(args, "sc_lfi_v3_latent_hidden_dim"))
            ),
            mass_hidden_dim=(
                None
                if getattr(args, "sc_lfi_v3_mass_hidden_dim", None) is None
                else int(getattr(args, "sc_lfi_v3_mass_hidden_dim"))
            ),
            mass_temperature=float(getattr(args, "sc_lfi_v3_mass_temperature", 1.0)),
            memory_size=int(getattr(args, "sc_lfi_v3_memory_size", 4)),
            memory_num_heads=int(getattr(args, "sc_lfi_v3_memory_num_heads", 4)),
            memory_ffn_multiplier=int(getattr(args, "sc_lfi_v3_memory_ffn_multiplier", 2)),
            prior_num_atoms=int(getattr(args, "sc_lfi_v3_prior_num_atoms", 4)),
            prior_scale=float(getattr(args, "sc_lfi_v3_prior_scale", 0.6)),
            episode_num_heads=int(getattr(args, "sc_lfi_v3_episode_num_heads", 4)),
            alpha_hidden_dim=(
                None
                if getattr(args, "sc_lfi_v3_alpha_hidden_dim", None) is None
                else int(getattr(args, "sc_lfi_v3_alpha_hidden_dim"))
            ),
            alpha_shot_scale=float(getattr(args, "sc_lfi_v3_alpha_shot_scale", 1.25)),
            alpha_uncertainty_scale=float(getattr(args, "sc_lfi_v3_alpha_uncertainty_scale", 1.0)),
            flow_hidden_dim=int(getattr(args, "sc_lfi_v3_flow_hidden_dim", 128)),
            flow_time_embedding_dim=int(getattr(args, "sc_lfi_v3_flow_time_embedding_dim", 32)),
            flow_memory_num_heads=int(getattr(args, "sc_lfi_v3_flow_memory_num_heads", 4)),
            backbone_name=fewshot_backbone,
            image_size=image_size,
            resnet12_drop_rate=0.0,
            resnet12_dropblock_size=5,
            flow_conditioning_type=str(getattr(args, "sc_lfi_v3_flow_conditioning_type", "film")),
            use_transport_flow=_bool_flag(getattr(args, "sc_lfi_v3_use_transport_flow", "true"), default=True),
            use_prior_measure=_bool_flag(getattr(args, "sc_lfi_v3_use_prior_measure", "true"), default=True),
            use_episode_adapter=_bool_flag(getattr(args, "sc_lfi_v3_use_episode_adapter", "true"), default=True),
            use_query_reweighting=_bool_flag(
                getattr(args, "sc_lfi_v3_use_query_reweighting", "true"),
                default=True,
            ),
            use_support_barycenter_only=_bool_flag(
                getattr(args, "sc_lfi_v3_use_support_barycenter_only", "false"),
                default=False,
            ),
            use_global_proto_branch=_bool_flag(
                getattr(args, "sc_lfi_v3_use_global_proto_branch", "false"),
                default=False,
            ),
            use_align_loss=_bool_flag(getattr(args, "sc_lfi_v3_use_align_loss", "true"), default=True),
            use_margin_loss=_bool_flag(getattr(args, "sc_lfi_v3_use_margin_loss", "true"), default=True),
            train_num_integration_steps=int(getattr(args, "sc_lfi_v3_train_num_integration_steps", 4)),
            eval_num_integration_steps=int(getattr(args, "sc_lfi_v3_eval_num_integration_steps", 8)),
            solver_type=str(getattr(args, "sc_lfi_v3_solver_type", "heun")),
            fm_time_schedule=str(getattr(args, "fm_time_schedule", "uniform")),
            score_temperature=float(getattr(args, "score_temperature", 8.0)),
            query_reweight_temperature=float(getattr(args, "sc_lfi_v3_query_reweight_temperature", 1.0)),
            proto_branch_weight=float(getattr(args, "sc_lfi_v3_proto_branch_weight", 0.1)),
            lambda_fm=float(getattr(args, "sc_lfi_v3_lambda_fm", 0.035)),
            lambda_align=float(getattr(args, "sc_lfi_v3_lambda_align", 0.05)),
            lambda_margin=float(getattr(args, "sc_lfi_v3_lambda_margin", 0.12)),
            lambda_reg=float(getattr(args, "sc_lfi_v3_lambda_reg", 0.08)),
            margin_value=float(getattr(args, "sc_lfi_v3_margin_value", 0.12)),
            support_entropy_target_ratio=float(getattr(args, "sc_lfi_v3_support_entropy_target_ratio", 0.58)),
            query_entropy_target_ratio=float(getattr(args, "sc_lfi_v3_query_entropy_target_ratio", 0.45)),
            relevance_entropy_target_ratio=float(getattr(args, "sc_lfi_v3_relevance_entropy_target_ratio", 0.4)),
            score_train_num_projections=int(getattr(args, "sc_lfi_v3_score_train_num_projections", 64)),
            score_eval_num_projections=int(getattr(args, "sc_lfi_v3_score_eval_num_projections", 128)),
            score_sw_p=float(getattr(args, "sc_lfi_v3_score_sw_p", 2.0)),
            score_normalize_inputs=_bool_flag(
                getattr(args, "sc_lfi_v3_score_normalize_inputs", "true"),
                default=True,
            ),
            score_train_projection_mode=str(getattr(args, "sc_lfi_v3_score_train_projection_mode", "resample")),
            score_eval_projection_mode=str(getattr(args, "sc_lfi_v3_score_eval_projection_mode", "fixed")),
            score_eval_num_repeats=int(getattr(args, "sc_lfi_v3_score_eval_num_repeats", 1)),
            score_projection_seed=int(getattr(args, "sc_lfi_v3_score_projection_seed", 7)),
            align_distance_type=str(getattr(args, "sc_lfi_v3_align_distance_type", "weighted_entropic_ot")),
            align_train_num_projections=int(getattr(args, "sc_lfi_v3_align_train_num_projections", 64)),
            align_eval_num_projections=int(getattr(args, "sc_lfi_v3_align_eval_num_projections", 128)),
            align_sw_p=float(getattr(args, "sc_lfi_v3_align_sw_p", 2.0)),
            align_normalize_inputs=_bool_flag(
                getattr(args, "sc_lfi_v3_align_normalize_inputs", "true"),
                default=True,
            ),
            align_train_projection_mode=str(getattr(args, "sc_lfi_v3_align_train_projection_mode", "resample")),
            align_eval_projection_mode=str(getattr(args, "sc_lfi_v3_align_eval_projection_mode", "fixed")),
            align_eval_num_repeats=int(getattr(args, "sc_lfi_v3_align_eval_num_repeats", 1)),
            align_projection_seed=int(getattr(args, "sc_lfi_v3_align_projection_seed", 11)),
            sinkhorn_epsilon=float(getattr(args, "sc_lfi_v3_sinkhorn_epsilon", 0.08)),
            sinkhorn_iterations=int(getattr(args, "sc_lfi_v3_sinkhorn_iterations", 100)),
            sinkhorn_cost_power=float(getattr(args, "sc_lfi_v3_sinkhorn_cost_power", 2.0)),
            eps=float(getattr(args, "sc_lfi_v3_eps", 1e-8)),
        )
    if args.model == "sc_lfi_v4":
        SupportConditionedLatentFlowInferenceNetV4 = _load_symbol(
            "net.sc_lfi_v4",
            "SupportConditionedLatentFlowInferenceNetV4",
        )
        return SupportConditionedLatentFlowInferenceNetV4(
            in_channels=3,
            hidden_dim=64,
            latent_dim=int(getattr(args, "latent_dim", 64)),
            context_dim=int(getattr(args, "sc_lfi_v4_context_dim", 128)),
            context_hidden_dim=(
                None
                if getattr(args, "sc_lfi_v4_context_hidden_dim", None) is None
                else int(getattr(args, "sc_lfi_v4_context_hidden_dim"))
            ),
            latent_hidden_dim=(
                None
                if getattr(args, "sc_lfi_v4_latent_hidden_dim", None) is None
                else int(getattr(args, "sc_lfi_v4_latent_hidden_dim"))
            ),
            mass_hidden_dim=(
                None
                if getattr(args, "sc_lfi_v4_mass_hidden_dim", None) is None
                else int(getattr(args, "sc_lfi_v4_mass_hidden_dim"))
            ),
            mass_temperature=float(getattr(args, "sc_lfi_v4_mass_temperature", 1.0)),
            shot_memory_size=int(getattr(args, "sc_lfi_v4_shot_memory_size", 2)),
            class_memory_size=int(getattr(args, "sc_lfi_v4_class_memory_size", 4)),
            memory_num_heads=int(getattr(args, "sc_lfi_v4_memory_num_heads", 4)),
            memory_ffn_multiplier=int(getattr(args, "sc_lfi_v4_memory_ffn_multiplier", 2)),
            prior_num_atoms=int(getattr(args, "sc_lfi_v4_prior_num_atoms", 4)),
            global_prior_size=int(getattr(args, "sc_lfi_v4_global_prior_size", 16)),
            prior_scale=float(getattr(args, "sc_lfi_v4_prior_scale", 0.25)),
            episode_num_heads=int(getattr(args, "sc_lfi_v4_episode_num_heads", 4)),
            alpha_hidden_dim=(
                None
                if getattr(args, "sc_lfi_v4_alpha_hidden_dim", None) is None
                else int(getattr(args, "sc_lfi_v4_alpha_hidden_dim"))
            ),
            shrinkage_kappa=float(getattr(args, "sc_lfi_v4_shrinkage_kappa", 2.0)),
            alpha_uncertainty_scale=float(getattr(args, "sc_lfi_v4_alpha_uncertainty_scale", 1.0)),
            use_shot_barycenter_only=_bool_flag(
                getattr(args, "sc_lfi_v4_use_shot_barycenter_only", "false"),
                default=False,
            ),
            flow_hidden_dim=int(getattr(args, "sc_lfi_v4_flow_hidden_dim", 128)),
            flow_time_embedding_dim=int(getattr(args, "sc_lfi_v4_flow_time_embedding_dim", 32)),
            flow_memory_num_heads=int(getattr(args, "sc_lfi_v4_flow_memory_num_heads", 4)),
            backbone_name=fewshot_backbone,
            image_size=image_size,
            resnet12_drop_rate=0.0,
            resnet12_dropblock_size=5,
            flow_conditioning_type=str(getattr(args, "sc_lfi_v4_flow_conditioning_type", "film")),
            use_transport_flow=_bool_flag(getattr(args, "sc_lfi_v4_use_transport_flow", "true"), default=True),
            use_prior_measure=_bool_flag(getattr(args, "sc_lfi_v4_use_prior_measure", "true"), default=True),
            use_episode_adapter=_bool_flag(getattr(args, "sc_lfi_v4_use_episode_adapter", "true"), default=True),
            use_metric_conditioning=_bool_flag(
                getattr(args, "sc_lfi_v4_use_metric_conditioning", "true"),
                default=True,
            ),
            use_align_loss=_bool_flag(getattr(args, "sc_lfi_v4_use_align_loss", "true"), default=True),
            use_margin_loss=_bool_flag(getattr(args, "sc_lfi_v4_use_margin_loss", "true"), default=True),
            use_loo_loss=_bool_flag(getattr(args, "sc_lfi_v4_use_loo_loss", "true"), default=True),
            train_num_integration_steps=int(getattr(args, "sc_lfi_v4_train_num_integration_steps", 4)),
            eval_num_integration_steps=int(getattr(args, "sc_lfi_v4_eval_num_integration_steps", 8)),
            solver_type=str(getattr(args, "sc_lfi_v4_solver_type", "heun")),
            fm_time_schedule=str(getattr(args, "fm_time_schedule", "uniform")),
            score_temperature=float(getattr(args, "score_temperature", 8.0)),
            lambda_fm=float(getattr(args, "sc_lfi_v4_lambda_fm", 0.03)),
            lambda_align=float(getattr(args, "sc_lfi_v4_lambda_align", 0.05)),
            lambda_margin=float(getattr(args, "sc_lfi_v4_lambda_margin", 0.1)),
            lambda_loo=float(getattr(args, "sc_lfi_v4_lambda_loo", 0.1)),
            lambda_reg=float(getattr(args, "sc_lfi_v4_lambda_reg", 0.05)),
            margin_value=float(getattr(args, "sc_lfi_v4_margin_value", 0.1)),
            support_token_entropy_target_ratio=float(
                getattr(args, "sc_lfi_v4_support_token_entropy_target_ratio", 0.55)
            ),
            query_entropy_target_ratio=float(getattr(args, "sc_lfi_v4_query_entropy_target_ratio", 0.45)),
            shot_entropy_target_ratio=float(getattr(args, "sc_lfi_v4_shot_entropy_target_ratio", 0.55)),
            prior_entropy_target_ratio=float(getattr(args, "sc_lfi_v4_prior_entropy_target_ratio", 0.45)),
            score_distance_type=str(getattr(args, "sc_lfi_v4_score_distance_type", "weighted_entropic_ot")),
            score_train_num_projections=int(getattr(args, "sc_lfi_v4_score_train_num_projections", 64)),
            score_eval_num_projections=int(getattr(args, "sc_lfi_v4_score_eval_num_projections", 128)),
            score_sw_p=float(getattr(args, "sc_lfi_v4_score_sw_p", 2.0)),
            score_normalize_inputs=_bool_flag(
                getattr(args, "sc_lfi_v4_score_normalize_inputs", "true"),
                default=True,
            ),
            score_train_projection_mode=str(getattr(args, "sc_lfi_v4_score_train_projection_mode", "resample")),
            score_eval_projection_mode=str(getattr(args, "sc_lfi_v4_score_eval_projection_mode", "fixed")),
            score_eval_num_repeats=int(getattr(args, "sc_lfi_v4_score_eval_num_repeats", 1)),
            score_projection_seed=int(getattr(args, "sc_lfi_v4_score_projection_seed", 7)),
            score_sinkhorn_epsilon=float(getattr(args, "sc_lfi_v4_score_sinkhorn_epsilon", 0.05)),
            score_sinkhorn_iterations=int(getattr(args, "sc_lfi_v4_score_sinkhorn_iterations", 60)),
            score_sinkhorn_cost_power=float(getattr(args, "sc_lfi_v4_score_sinkhorn_cost_power", 2.0)),
            align_distance_type=str(getattr(args, "sc_lfi_v4_align_distance_type", "weighted_entropic_ot")),
            align_train_num_projections=int(getattr(args, "sc_lfi_v4_align_train_num_projections", 64)),
            align_eval_num_projections=int(getattr(args, "sc_lfi_v4_align_eval_num_projections", 128)),
            align_sw_p=float(getattr(args, "sc_lfi_v4_align_sw_p", 2.0)),
            align_normalize_inputs=_bool_flag(
                getattr(args, "sc_lfi_v4_align_normalize_inputs", "true"),
                default=True,
            ),
            align_train_projection_mode=str(getattr(args, "sc_lfi_v4_align_train_projection_mode", "resample")),
            align_eval_projection_mode=str(getattr(args, "sc_lfi_v4_align_eval_projection_mode", "fixed")),
            align_eval_num_repeats=int(getattr(args, "sc_lfi_v4_align_eval_num_repeats", 1)),
            align_projection_seed=int(getattr(args, "sc_lfi_v4_align_projection_seed", 11)),
            sinkhorn_epsilon=float(getattr(args, "sc_lfi_v4_sinkhorn_epsilon", 0.05)),
            sinkhorn_iterations=int(getattr(args, "sc_lfi_v4_sinkhorn_iterations", 80)),
            sinkhorn_cost_power=float(getattr(args, "sc_lfi_v4_sinkhorn_cost_power", 2.0)),
            eps=float(getattr(args, "sc_lfi_v4_eps", 1e-8)),
        )
    if args.model in {"spifce", "spifmax"}:
        SPIFCE, SPIFMAX = _load_symbols("net.spif", "SPIFCE", "SPIFMAX")
        spif_cls = SPIFMAX if args.model == "spifmax" else SPIFCE
        stable_dim = int(getattr(args, "spif_stable_dim", 64) or 64)
        variant_dim = int(getattr(args, "spif_variant_dim", stable_dim) or stable_dim)
        gate_hidden = int(getattr(args, "spif_gate_hidden", max(1, stable_dim // 4)) or max(1, stable_dim // 4))
        learnable_alpha = (
            _bool_flag(getattr(args, "spif_learnable_alpha", "true"), default=True)
            if args.model == "spifmax"
            else _bool_flag(getattr(args, "spif_learnable_alpha", "false"), default=False)
        )
        return spif_cls(
            in_channels=3,
            hidden_dim=64,
            stable_dim=stable_dim,
            variant_dim=variant_dim,
            gate_hidden=gate_hidden,
            top_r=int(getattr(args, "spif_top_r", 3)),
            alpha_init=float(getattr(args, "spif_alpha_init", 0.7)),
            learnable_alpha=learnable_alpha,
            gate_on=_bool_flag(getattr(args, "spif_gate_on", "true"), default=True),
            factorization_on=_bool_flag(getattr(args, "spif_factorization_on", "true"), default=True),
            global_only=_bool_flag(getattr(args, "spif_global_only", "false"), default=False),
            local_only=_bool_flag(getattr(args, "spif_local_only", "false"), default=False),
            token_l2norm=_bool_flag(getattr(args, "spif_token_l2norm", "true"), default=True),
            consistency_weight=float(getattr(args, "spif_consistency_weight", 0.1)),
            decorr_weight=float(getattr(args, "spif_decorr_weight", 0.01)),
            sparse_weight=float(getattr(args, "spif_sparse_weight", 0.001)),
            consistency_dropout=float(getattr(args, "spif_consistency_dropout", 0.1)),
            backbone_name=fewshot_backbone,
            image_size=image_size,
            resnet12_drop_rate=0.0,
            resnet12_dropblock_size=5,
        )
    if args.model == "spif_rdp":
        SPIFRDP = _load_symbol("net.spif_rdp", "SPIFRDP")
        stable_dim = int(getattr(args, "spif_stable_dim", 64) or 64)
        variant_dim = int(getattr(args, "spif_variant_dim", stable_dim) or stable_dim)
        gate_hidden = int(getattr(args, "spif_gate_hidden", max(1, stable_dim // 4)) or max(1, stable_dim // 4))
        lambda_init = float(getattr(args, "spif_rdp_lambda_init", 1e-3))
        fsl_mamba_base_dim = int(getattr(args, "spif_rdp_fsl_mamba_base_dim", 48) or 48)
        fsl_mamba_output_dim = int(getattr(args, "spif_rdp_fsl_mamba_output_dim", 320) or 320)
        hidden_dim = fewshot_backbone_output_dim(fewshot_backbone, fsl_mamba_output_dim=fsl_mamba_output_dim)
        return SPIFRDP(
            in_channels=3,
            hidden_dim=hidden_dim,
            stable_dim=stable_dim,
            variant_dim=variant_dim,
            gate_hidden=gate_hidden,
            top_r=int(getattr(args, "spif_rdp_top_r", 5)),
            gate_on=_bool_flag(getattr(args, "spif_gate_on", "true"), default=True),
            factorization_on=_bool_flag(getattr(args, "spif_factorization_on", "true"), default=True),
            global_only=_bool_flag(getattr(args, "spif_global_only", "false"), default=False),
            local_only=_bool_flag(getattr(args, "spif_local_only", "false"), default=False),
            token_l2norm=_bool_flag(getattr(args, "spif_token_l2norm", "true"), default=True),
            consistency_weight=float(getattr(args, "spif_rdp_consistency_weight", 0.0)),
            decorr_weight=float(getattr(args, "spif_rdp_decorr_weight", 0.0)),
            sparse_weight=float(getattr(args, "spif_rdp_sparse_weight", 0.0)),
            consistency_dropout=float(getattr(args, "spif_consistency_dropout", 0.1)),
            rdp_lambda_init=lambda_init,
            rdp_alpha_init=float(getattr(args, "spif_rdp_alpha_init", 1.0)),
            rdp_tau_init=float(getattr(args, "spif_rdp_tau_init", 10.0)),
            rdp_gamma_init=float(getattr(args, "spif_rdp_gamma_init", 1.0)),
            rdp_variance_floor=float(getattr(args, "spif_rdp_variance_floor", 1e-2)),
            rdp_compact_loss_weight=float(getattr(args, "spif_rdp_compact_loss_weight", 0.1)),
            rdp_sep_loss_weight=float(getattr(args, "spif_rdp_sep_loss_weight", 0.05)),
            rdp_sep_margin=float(getattr(args, "spif_rdp_sep_margin", 0.5)),
            rdp_eps=float(getattr(args, "spif_rdp_eps", 1e-6)),
            rdp_beta_init=float(
                getattr(
                    args,
                    "spif_rdp_beta_init",
                    getattr(args, "spif_rdp_beta0_init", 0.5),
                )
            ),
            backbone_name=fewshot_backbone,
            image_size=image_size,
            resnet12_drop_rate=0.0,
            resnet12_dropblock_size=5,
            fsl_mamba_base_dim=fsl_mamba_base_dim,
            fsl_mamba_output_dim=fsl_mamba_output_dim,
            fsl_mamba_drop_path=float(getattr(args, "spif_rdp_fsl_mamba_drop_path", 0.02)),
            fsl_mamba_perturb_sigma=float(getattr(args, "spif_rdp_fsl_mamba_perturb_sigma", 0.0)),
            rdp_use_fsl_mamba_global_prior=_bool_flag(
                getattr(args, "spif_rdp_use_fsl_mamba_global_prior", "true"),
                default=True,
            ),
            rdp_fsl_mamba_global_mix_init=float(
                getattr(args, "spif_rdp_fsl_mamba_global_mix_init", 0.35)
            ),
        )
    if args.model == "spif_ota":
        SPIFOTA = _load_symbol("net.spif_ota", "SPIFOTA")
        return SPIFOTA(
            in_channels=3,
            hidden_dim=64,
            transport_dim=int(getattr(args, "spif_ota_transport_dim", None) or getattr(args, "token_dim", None) or 128),
            projector_hidden_dim=(
                None
                if getattr(args, "spif_ota_projector_hidden_dim", None) is None
                else int(getattr(args, "spif_ota_projector_hidden_dim"))
            ),
            mass_hidden_dim=(
                None
                if getattr(args, "spif_ota_mass_hidden_dim", None) is None
                else int(getattr(args, "spif_ota_mass_hidden_dim"))
            ),
            mass_temperature=float(getattr(args, "spif_ota_mass_temperature", 1.0)),
            sinkhorn_epsilon=float(getattr(args, "spif_ota_sinkhorn_epsilon", 0.05)),
            sinkhorn_iterations=int(getattr(args, "spif_ota_sinkhorn_iterations", 60)),
            shot_hidden_dim=int(getattr(args, "spif_ota_shot_hidden_dim", 32)),
            shot_aggregation=str(getattr(args, "spif_ota_shot_aggregation", "learned")),
            position_cost_weight=float(getattr(args, "spif_ota_position_cost_weight", 0.0)),
            use_column_bias=_bool_flag(getattr(args, "spif_ota_use_column_bias", "true"), default=True),
            lambda_mass=float(getattr(args, "spif_ota_lambda_mass", 0.0)),
            lambda_consistency=float(getattr(args, "spif_ota_lambda_consistency", 0.0)),
            backbone_name=fewshot_backbone,
            image_size=image_size,
            resnet12_drop_rate=0.0,
            resnet12_dropblock_size=5,
            eps=float(getattr(args, "spif_ota_eps", 1e-6)),
        )
    if args.model == "spif_otccls":
        SPIFOTCCLS = _load_symbol("net.spif_otccls", "SPIFOTCCLS")
        return SPIFOTCCLS(
            in_channels=3,
            hidden_dim=640,
            feature_dim=int(getattr(args, "spif_otccls_feature_dim", 256)),
            gate_hidden=int(getattr(args, "spif_otccls_gate_hidden", 128)),
            num_projections=int(getattr(args, "spif_otccls_num_projections", 64)),
            num_quantiles=int(getattr(args, "spif_otccls_num_quantiles", 100)),
            tau_g_init=float(getattr(args, "spif_otccls_tau_g_init", 10.0)),
            tau_l_init=float(getattr(args, "spif_otccls_tau_l_init", 1.0)),
            beta_init=float(getattr(args, "spif_otccls_beta_init", 0.5)),
            alpha_init=float(getattr(args, "spif_otccls_alpha_init", 1.0)),
            compact_loss_weight=float(getattr(args, "spif_otccls_compact_loss_weight", 0.1)),
            decorr_loss_weight=float(getattr(args, "spif_otccls_decorr_loss_weight", 0.05)),
            entropy_loss_weight=float(getattr(args, "spif_otccls_entropy_loss_weight", 0.01)),
            global_only=_bool_flag(getattr(args, "spif_global_only", "false"), default=False),
            local_only=_bool_flag(getattr(args, "spif_local_only", "false"), default=False),
            shot_agg=str(getattr(args, "spif_otccls_shot_agg", "softmax")),
            shot_softmax_beta=float(getattr(args, "spif_otccls_shot_softmax_beta", 10.0)),
            swd_backend=str(getattr(args, "spif_otccls_swd_backend", "pot")),
            eps=float(getattr(args, "spif_otccls_eps", 1e-6)),
            backbone_name=fewshot_backbone,
            image_size=image_size,
            resnet12_drop_rate=float(getattr(args, "spif_otccls_backbone_drop_rate", 0.05)),
            resnet12_dropblock_size=int(getattr(args, "spif_otccls_backbone_dropblock_size", 5)),
        )
    if args.model == "spifce_shot":
        SPIFCEShot = _load_symbol("net.spif_shot", "SPIFCEShot")
        stable_dim = int(getattr(args, "spif_stable_dim", 64) or 64)
        variant_dim = int(getattr(args, "spif_variant_dim", stable_dim) or stable_dim)
        gate_hidden = int(getattr(args, "spif_gate_hidden", max(1, stable_dim // 4)) or max(1, stable_dim // 4))
        return SPIFCEShot(
            in_channels=3,
            hidden_dim=64,
            stable_dim=stable_dim,
            variant_dim=variant_dim,
            gate_hidden=gate_hidden,
            top_r=int(getattr(args, "spif_shot_top_r", 4)),
            alpha_init=float(getattr(args, "spif_alpha_init", 0.7)),
            learnable_alpha=_bool_flag(getattr(args, "spif_learnable_alpha", "false"), default=False),
            gate_on=_bool_flag(getattr(args, "spif_gate_on", "true"), default=True),
            factorization_on=_bool_flag(getattr(args, "spif_factorization_on", "true"), default=True),
            global_only=_bool_flag(getattr(args, "spif_global_only", "false"), default=False),
            local_only=_bool_flag(getattr(args, "spif_local_only", "false"), default=False),
            token_l2norm=_bool_flag(getattr(args, "spif_token_l2norm", "true"), default=True),
            consistency_weight=0.0,
            decorr_weight=0.0,
            sparse_weight=0.0,
            consistency_dropout=float(getattr(args, "spif_consistency_dropout", 0.1)),
            spif_shot_agg=str(getattr(args, "spif_shot_agg", "softmax")),
            spif_shot_softmax_beta=float(getattr(args, "spif_shot_softmax_beta", 10.0)),
            backbone_name=fewshot_backbone,
            image_size=image_size,
            resnet12_drop_rate=0.0,
            resnet12_dropblock_size=5,
        )
    if args.model == "spifce_asym_shot":
        SPIFCEAsymShot = _load_symbol("net.spifce_asym_shot", "SPIFCEAsymShot")
        stable_dim = int(getattr(args, "spif_stable_dim", 64) or 64)
        variant_dim = int(getattr(args, "spif_variant_dim", stable_dim) or stable_dim)
        gate_hidden = int(getattr(args, "spif_gate_hidden", max(1, stable_dim // 4)) or max(1, stable_dim // 4))
        local_dim_raw = getattr(args, "spif_asym_local_dim", None)
        return SPIFCEAsymShot(
            in_channels=3,
            hidden_dim=64,
            stable_dim=stable_dim,
            variant_dim=variant_dim,
            local_dim=None if local_dim_raw is None else int(local_dim_raw),
            gate_hidden=gate_hidden,
            top_r=int(getattr(args, "spif_asym_top_r", 4)),
            alpha_init=float(getattr(args, "spif_alpha_init", 0.7)),
            learnable_alpha=_bool_flag(getattr(args, "spif_learnable_alpha", "false"), default=False),
            gate_on=_bool_flag(getattr(args, "spif_gate_on", "true"), default=True),
            factorization_on=_bool_flag(getattr(args, "spif_factorization_on", "true"), default=True),
            global_only=_bool_flag(getattr(args, "spif_global_only", "false"), default=False),
            local_only=_bool_flag(getattr(args, "spif_local_only", "false"), default=False),
            token_l2norm=_bool_flag(getattr(args, "spif_token_l2norm", "true"), default=True),
            spif_asym_shot_agg=str(getattr(args, "spif_asym_shot_agg", "softmax")),
            spif_asym_shot_softmax_beta=float(getattr(args, "spif_asym_shot_softmax_beta", 10.0)),
            spif_asym_fusion_mode=str(getattr(args, "spif_asym_fusion_mode", "margin_adaptive")),
            spif_asym_fusion_kappa=float(getattr(args, "spif_asym_fusion_kappa", 8.0)),
            spif_asym_global_scale=float(getattr(args, "spif_asym_global_scale", 1.0)),
            spif_asym_local_scale=float(getattr(args, "spif_asym_local_scale", 1.0)),
            spif_asym_local_score_mode=str(getattr(args, "spif_asym_local_score_mode", "bidirectional")),
            spif_asym_share_local_head=_bool_flag(getattr(args, "spif_asym_share_local_head", "false"), default=False),
            backbone_name=fewshot_backbone,
            image_size=image_size,
            resnet12_drop_rate=0.0,
            resnet12_dropblock_size=5,
        )
    if args.model == "spifaeb":
        SPIFAEB = _load_symbol("net.spif_aeb", "SPIFAEB")
        stable_dim = int(getattr(args, "spif_stable_dim", 64) or 64)
        variant_dim = int(getattr(args, "spif_variant_dim", stable_dim) or stable_dim)
        gate_hidden = int(getattr(args, "spif_gate_hidden", max(1, stable_dim // 4)) or max(1, stable_dim // 4))
        aeb_hidden = getattr(args, "aeb_hidden", None)
        return SPIFAEB(
            in_channels=3,
            hidden_dim=64,
            stable_dim=stable_dim,
            variant_dim=variant_dim,
            gate_hidden=gate_hidden,
            top_r=int(getattr(args, "spif_top_r", 3)),
            alpha_init=float(getattr(args, "spif_alpha_init", 0.7)),
            learnable_alpha=_bool_flag(getattr(args, "spif_learnable_alpha", "false"), default=False),
            gate_on=_bool_flag(getattr(args, "spif_gate_on", "true"), default=True),
            factorization_on=_bool_flag(getattr(args, "spif_factorization_on", "true"), default=True),
            global_only=_bool_flag(getattr(args, "spif_global_only", "false"), default=False),
            local_only=_bool_flag(getattr(args, "spif_local_only", "false"), default=False),
            token_l2norm=_bool_flag(getattr(args, "spif_token_l2norm", "true"), default=True),
            consistency_weight=0.0,
            decorr_weight=0.0,
            sparse_weight=0.0,
            consistency_dropout=float(getattr(args, "spif_consistency_dropout", 0.1)),
            aeb_hidden=None if aeb_hidden is None else int(aeb_hidden),
            aeb_beta=float(getattr(args, "aeb_beta", 1.0)),
            aeb_eps=float(getattr(args, "aeb_eps", 1e-6)),
            backbone_name=fewshot_backbone,
            image_size=image_size,
            resnet12_drop_rate=0.0,
            resnet12_dropblock_size=5,
        )
    if args.model == "spifaeb_shot":
        SPIFAEBShot = _load_symbol("net.spif_aeb_shot", "SPIFAEBShot")
        stable_dim = int(getattr(args, "spif_stable_dim", 64) or 64)
        variant_dim = int(getattr(args, "spif_variant_dim", stable_dim) or stable_dim)
        gate_hidden = int(getattr(args, "spif_gate_hidden", max(1, stable_dim // 4)) or max(1, stable_dim // 4))
        aeb_hidden = getattr(args, "aeb_hidden", None)
        return SPIFAEBShot(
            in_channels=3,
            hidden_dim=64,
            stable_dim=stable_dim,
            variant_dim=variant_dim,
            gate_hidden=gate_hidden,
            top_r=int(getattr(args, "spif_top_r", 3)),
            alpha_init=float(getattr(args, "spif_alpha_init", 0.7)),
            learnable_alpha=_bool_flag(getattr(args, "spif_learnable_alpha", "false"), default=False),
            gate_on=_bool_flag(getattr(args, "spif_gate_on", "true"), default=True),
            factorization_on=_bool_flag(getattr(args, "spif_factorization_on", "true"), default=True),
            global_only=_bool_flag(getattr(args, "spif_global_only", "false"), default=False),
            local_only=_bool_flag(getattr(args, "spif_local_only", "false"), default=False),
            token_l2norm=_bool_flag(getattr(args, "spif_token_l2norm", "true"), default=True),
            consistency_weight=0.0,
            decorr_weight=0.0,
            sparse_weight=0.0,
            consistency_dropout=float(getattr(args, "spif_consistency_dropout", 0.1)),
            aeb_hidden=None if aeb_hidden is None else int(aeb_hidden),
            aeb_beta=float(getattr(args, "aeb_beta", 1.0)),
            aeb_eps=float(getattr(args, "aeb_eps", 1e-6)),
            aeb_shot_agg=str(getattr(args, "aeb_shot_agg", "softmax")),
            aeb_shot_softmax_beta=float(getattr(args, "aeb_shot_softmax_beta", 10.0)),
            backbone_name=fewshot_backbone,
            image_size=image_size,
            resnet12_drop_rate=0.0,
            resnet12_dropblock_size=5,
        )
    if args.model == "spifaeb_v2":
        SPIFAEBV2 = _load_symbol("net.spif_aeb_v2", "SPIFAEBV2")
        stable_dim = int(getattr(args, "spif_stable_dim", 64) or 64)
        variant_dim = int(getattr(args, "spif_variant_dim", stable_dim) or stable_dim)
        gate_hidden = int(getattr(args, "spif_gate_hidden", max(1, stable_dim // 4)) or max(1, stable_dim // 4))
        local_dim_raw = getattr(args, "aeb_v2_local_dim", None)
        aeb_v2_hidden = getattr(args, "aeb_v2_hidden", None)
        return SPIFAEBV2(
            in_channels=3,
            hidden_dim=64,
            stable_dim=stable_dim,
            variant_dim=variant_dim,
            local_dim=None if local_dim_raw is None else int(local_dim_raw),
            gate_hidden=gate_hidden,
            alpha_init=float(getattr(args, "spif_alpha_init", 0.7)),
            learnable_alpha=_bool_flag(getattr(args, "spif_learnable_alpha", "false"), default=False),
            gate_on=_bool_flag(getattr(args, "spif_gate_on", "true"), default=True),
            factorization_on=_bool_flag(getattr(args, "spif_factorization_on", "true"), default=True),
            global_only=_bool_flag(getattr(args, "spif_global_only", "false"), default=False),
            local_only=_bool_flag(getattr(args, "spif_local_only", "false"), default=False),
            token_l2norm=_bool_flag(getattr(args, "spif_token_l2norm", "true"), default=True),
            aeb_v2_hidden=None if aeb_v2_hidden is None else int(aeb_v2_hidden),
            aeb_v2_min_budget=float(getattr(args, "aeb_v2_min_budget", 0.2)),
            aeb_v2_max_budget=float(getattr(args, "aeb_v2_max_budget", 0.85)),
            aeb_v2_rank_temperature=float(getattr(args, "aeb_v2_rank_temperature", 0.35)),
            aeb_v2_global_scale=float(getattr(args, "aeb_v2_global_scale", 1.0)),
            aeb_v2_local_scale=float(getattr(args, "aeb_v2_local_scale", 1.0)),
            aeb_v2_cond_global_weight=float(getattr(args, "aeb_v2_cond_global_weight", 0.35)),
            aeb_v2_cond_global_temperature=float(getattr(args, "aeb_v2_cond_global_temperature", 8.0)),
            aeb_v2_cond_global_use_gate_prior=_bool_flag(
                getattr(args, "aeb_v2_cond_global_use_gate_prior", "true"),
                default=True,
            ),
            aeb_v2_query_class_attention_on=_bool_flag(
                getattr(args, "aeb_v2_query_class_attention_on", "true"),
                default=True,
            ),
            aeb_v2_query_class_attention_temperature=float(
                getattr(args, "aeb_v2_query_class_attention_temperature", 8.0)
            ),
            aeb_v2_query_class_attention_use_gate_prior=_bool_flag(
                getattr(args, "aeb_v2_query_class_attention_use_gate_prior", "true"),
                default=True,
            ),
            aeb_v2_anchor_top_r=int(getattr(args, "aeb_v2_anchor_top_r", 4)),
            aeb_v2_anchor_mix=float(getattr(args, "aeb_v2_anchor_mix", 0.35)),
            aeb_v2_local_mix_mode=str(getattr(args, "aeb_v2_local_mix_mode", "prior_adaptive")),
            aeb_v2_adaptive_mix_min=float(getattr(args, "aeb_v2_adaptive_mix_min", 0.2)),
            aeb_v2_adaptive_mix_max=float(getattr(args, "aeb_v2_adaptive_mix_max", 0.8)),
            aeb_v2_adaptive_mix_scale=float(getattr(args, "aeb_v2_adaptive_mix_scale", 6.0)),
            aeb_v2_local_residual_scale=float(getattr(args, "aeb_v2_local_residual_scale", 0.25)),
            aeb_v2_budget_prior_scale=float(getattr(args, "aeb_v2_budget_prior_scale", 6.0)),
            aeb_v2_budget_residual_scale=float(getattr(args, "aeb_v2_budget_residual_scale", 0.15)),
            aeb_v2_budget_rank_weight=float(getattr(args, "aeb_v2_budget_rank_weight", 0.1)),
            aeb_v2_budget_rank_margin=float(getattr(args, "aeb_v2_budget_rank_margin", 0.05)),
            aeb_v2_budget_residual_reg_weight=float(getattr(args, "aeb_v2_budget_residual_reg_weight", 0.05)),
            aeb_v2_support_coverage_temperature=float(
                getattr(args, "aeb_v2_support_coverage_temperature", 8.0)
            ),
            aeb_v2_coverage_bonus_weight=float(getattr(args, "aeb_v2_coverage_bonus_weight", 0.0)),
            aeb_v2_competition_weight=float(getattr(args, "aeb_v2_competition_weight", 1.0)),
            aeb_v2_competition_temperature=float(getattr(args, "aeb_v2_competition_temperature", 8.0)),
            aeb_v2_local_margin_weight=float(getattr(args, "aeb_v2_local_margin_weight", 0.0)),
            aeb_v2_local_margin_target=float(getattr(args, "aeb_v2_local_margin_target", 0.5)),
            aeb_v2_anchor_consistency_weight=float(getattr(args, "aeb_v2_anchor_consistency_weight", 0.02)),
            aeb_v2_fusion_mode=str(getattr(args, "aeb_v2_fusion_mode", "fixed")),
            aeb_v2_fusion_kappa=float(getattr(args, "aeb_v2_fusion_kappa", 8.0)),
            aeb_v2_local_score_mode=str(getattr(args, "aeb_v2_local_score_mode", "query_to_support")),
            aeb_v2_share_local_head=_bool_flag(getattr(args, "aeb_v2_share_local_head", "false"), default=False),
            aeb_v2_query_gate_weighting=_bool_flag(
                getattr(args, "aeb_v2_query_gate_weighting", "false"),
                default=False,
            ),
            aeb_v2_detach_budget_context=_bool_flag(
                getattr(args, "aeb_v2_detach_budget_context", "true"),
                default=True,
            ),
            aeb_v2_detach_local_backbone=_bool_flag(
                getattr(args, "aeb_v2_detach_local_backbone", "true"),
                default=True,
            ),
            aeb_v2_global_ce_weight=float(getattr(args, "aeb_v2_global_ce_weight", 0.0)),
            aeb_v2_local_ce_weight=float(getattr(args, "aeb_v2_local_ce_weight", 0.05)),
            aeb_v2_eps=float(getattr(args, "aeb_v2_eps", 1e-6)),
            backbone_name=fewshot_backbone,
            image_size=image_size,
            resnet12_drop_rate=0.0,
            resnet12_dropblock_size=5,
        )
    if args.model == "spifaeb_v3":
        SPIFAEBv3 = _load_symbol("net.spifaeb_v3", "SPIFAEBv3")
        stable_dim = int(getattr(args, "spif_stable_dim", 64) or 64)
        variant_dim = int(getattr(args, "spif_variant_dim", stable_dim) or stable_dim)
        gate_hidden = int(getattr(args, "spif_gate_hidden", max(1, stable_dim // 4)) or max(1, stable_dim // 4))
        local_dim_raw = getattr(args, "aeb_v2_local_dim", None)
        aeb_v2_hidden = getattr(args, "aeb_v2_hidden", None)
        return SPIFAEBv3(
            in_channels=3,
            hidden_dim=64,
            stable_dim=stable_dim,
            variant_dim=variant_dim,
            local_dim=None if local_dim_raw is None else int(local_dim_raw),
            gate_hidden=gate_hidden,
            alpha_init=float(getattr(args, "spif_alpha_init", 0.7)),
            learnable_alpha=_bool_flag(getattr(args, "spif_learnable_alpha", "false"), default=False),
            gate_on=_bool_flag(getattr(args, "spif_gate_on", "true"), default=True),
            factorization_on=_bool_flag(getattr(args, "spif_factorization_on", "true"), default=True),
            global_only=_bool_flag(getattr(args, "spif_global_only", "false"), default=False),
            local_only=_bool_flag(getattr(args, "spif_local_only", "false"), default=False),
            token_l2norm=_bool_flag(getattr(args, "spif_token_l2norm", "true"), default=True),
            aeb_v2_hidden=None if aeb_v2_hidden is None else int(aeb_v2_hidden),
            aeb_v2_min_budget=float(getattr(args, "aeb_v2_min_budget", 0.2)),
            aeb_v2_max_budget=float(getattr(args, "aeb_v2_max_budget", 0.85)),
            aeb_v2_rank_temperature=float(getattr(args, "aeb_v2_rank_temperature", 0.35)),
            aeb_v2_global_scale=float(getattr(args, "aeb_v2_global_scale", 1.0)),
            aeb_v2_local_scale=float(getattr(args, "aeb_v2_local_scale", 1.0)),
            aeb_v2_cond_global_weight=float(getattr(args, "aeb_v2_cond_global_weight", 0.35)),
            aeb_v2_cond_global_temperature=float(getattr(args, "aeb_v2_cond_global_temperature", 8.0)),
            aeb_v2_cond_global_use_gate_prior=_bool_flag(
                getattr(args, "aeb_v2_cond_global_use_gate_prior", "true"),
                default=True,
            ),
            aeb_v2_query_class_attention_on=_bool_flag(
                getattr(args, "aeb_v2_query_class_attention_on", "true"),
                default=True,
            ),
            aeb_v2_query_class_attention_temperature=float(
                getattr(args, "aeb_v2_query_class_attention_temperature", 8.0)
            ),
            aeb_v2_query_class_attention_use_gate_prior=_bool_flag(
                getattr(args, "aeb_v2_query_class_attention_use_gate_prior", "true"),
                default=True,
            ),
            aeb_v2_anchor_top_r=int(getattr(args, "aeb_v2_anchor_top_r", 4)),
            aeb_v2_anchor_mix=float(getattr(args, "aeb_v2_anchor_mix", 0.35)),
            aeb_v2_local_mix_mode=str(getattr(args, "aeb_v2_local_mix_mode", "prior_adaptive")),
            aeb_v2_adaptive_mix_min=float(getattr(args, "aeb_v2_adaptive_mix_min", 0.2)),
            aeb_v2_adaptive_mix_max=float(getattr(args, "aeb_v2_adaptive_mix_max", 0.8)),
            aeb_v2_adaptive_mix_scale=float(getattr(args, "aeb_v2_adaptive_mix_scale", 6.0)),
            aeb_v2_local_residual_scale=float(getattr(args, "aeb_v2_local_residual_scale", 0.25)),
            aeb_v2_budget_prior_scale=float(getattr(args, "aeb_v2_budget_prior_scale", 6.0)),
            aeb_v2_budget_residual_scale=float(getattr(args, "aeb_v2_budget_residual_scale", 0.15)),
            aeb_v2_budget_rank_weight=float(getattr(args, "aeb_v2_budget_rank_weight", 0.1)),
            aeb_v2_budget_rank_margin=float(getattr(args, "aeb_v2_budget_rank_margin", 0.05)),
            aeb_v2_budget_residual_reg_weight=float(getattr(args, "aeb_v2_budget_residual_reg_weight", 0.05)),
            aeb_v2_support_coverage_temperature=float(
                getattr(args, "aeb_v2_support_coverage_temperature", 8.0)
            ),
            aeb_v2_coverage_bonus_weight=float(getattr(args, "aeb_v2_coverage_bonus_weight", 0.0)),
            aeb_v2_competition_weight=float(getattr(args, "aeb_v2_competition_weight", 1.0)),
            aeb_v2_competition_temperature=float(getattr(args, "aeb_v2_competition_temperature", 8.0)),
            aeb_v2_local_margin_weight=float(getattr(args, "aeb_v2_local_margin_weight", 0.0)),
            aeb_v2_local_margin_target=float(getattr(args, "aeb_v2_local_margin_target", 0.5)),
            aeb_v2_anchor_consistency_weight=float(getattr(args, "aeb_v2_anchor_consistency_weight", 0.02)),
            aeb_v2_fusion_mode=str(getattr(args, "aeb_v2_fusion_mode", "fixed")),
            aeb_v2_fusion_kappa=float(getattr(args, "aeb_v2_fusion_kappa", 8.0)),
            aeb_v2_local_score_mode=str(getattr(args, "aeb_v2_local_score_mode", "query_to_support")),
            aeb_v2_share_local_head=_bool_flag(getattr(args, "aeb_v2_share_local_head", "false"), default=False),
            aeb_v2_query_gate_weighting=_bool_flag(
                getattr(args, "aeb_v2_query_gate_weighting", "false"),
                default=False,
            ),
            aeb_v2_detach_budget_context=_bool_flag(
                getattr(args, "aeb_v2_detach_budget_context", "true"),
                default=True,
            ),
            aeb_v2_detach_local_backbone=_bool_flag(
                getattr(args, "aeb_v2_detach_local_backbone", "true"),
                default=True,
            ),
            aeb_v2_global_ce_weight=float(
                getattr(
                    args,
                    "spifaeb_v3_global_ce_weight",
                    getattr(args, "aeb_v2_global_ce_weight", 0.0),
                )
            ),
            aeb_v2_local_ce_weight=float(getattr(args, "aeb_v2_local_ce_weight", 0.05)),
            aeb_v2_eps=float(getattr(args, "aeb_v2_eps", 1e-6)),
            spifaeb_v3_use_structured_global=_bool_flag(
                getattr(args, "spifaeb_v3_use_structured_global", "true"),
                default=True,
            ),
            spifaeb_v3_sigma_min=float(getattr(args, "spifaeb_v3_sigma_min", 0.05)),
            spifaeb_v3_use_reliability=_bool_flag(
                getattr(args, "spifaeb_v3_use_reliability", "true"),
                default=True,
            ),
            spifaeb_v3_reliability_detach=_bool_flag(
                getattr(args, "spifaeb_v3_reliability_detach", "true"),
                default=True,
            ),
            spifaeb_v3_beta_align=float(getattr(args, "spifaeb_v3_beta_align", 1.0)),
            spifaeb_v3_beta_dev=float(getattr(args, "spifaeb_v3_beta_dev", 0.5)),
            spifaeb_v3_beta_rel=float(getattr(args, "spifaeb_v3_beta_rel", 0.2)),
            spifaeb_v3_learnable_betas=_bool_flag(
                getattr(args, "spifaeb_v3_learnable_betas", "false"),
                default=False,
            ),
            backbone_name=fewshot_backbone,
            image_size=image_size,
            resnet12_drop_rate=0.0,
            resnet12_dropblock_size=5,
        )
    if args.model in {"spifce_local_sw", "spifmax_local_sw"}:
        SPIFCELocalSW, SPIFMAXLocalSW = _load_symbols(
            "net.spif_local_sw",
            "SPIFCELocalSW",
            "SPIFMAXLocalSW",
        )
        spif_cls = SPIFMAXLocalSW if args.model == "spifmax_local_sw" else SPIFCELocalSW
        stable_dim = int(getattr(args, "spif_stable_dim", 64) or 64)
        variant_dim = int(getattr(args, "spif_variant_dim", stable_dim) or stable_dim)
        gate_hidden = int(getattr(args, "spif_gate_hidden", max(1, stable_dim // 4)) or max(1, stable_dim // 4))
        learnable_alpha = (
            _bool_flag(getattr(args, "spif_learnable_alpha", "true"), default=True)
            if args.model == "spifmax_local_sw"
            else _bool_flag(getattr(args, "spif_learnable_alpha", "false"), default=False)
        )
        return spif_cls(
            in_channels=3,
            hidden_dim=64,
            stable_dim=stable_dim,
            variant_dim=variant_dim,
            gate_hidden=gate_hidden,
            top_r=int(getattr(args, "spif_top_r", 3)),
            alpha_init=float(getattr(args, "spif_alpha_init", 0.7)),
            learnable_alpha=learnable_alpha,
            gate_on=_bool_flag(getattr(args, "spif_gate_on", "true"), default=True),
            factorization_on=_bool_flag(getattr(args, "spif_factorization_on", "true"), default=True),
            global_only=_bool_flag(getattr(args, "spif_global_only", "false"), default=False),
            local_only=_bool_flag(getattr(args, "spif_local_only", "false"), default=False),
            token_l2norm=_bool_flag(getattr(args, "spif_token_l2norm", "true"), default=True),
            consistency_weight=float(getattr(args, "spif_consistency_weight", 0.1)),
            decorr_weight=float(getattr(args, "spif_decorr_weight", 0.01)),
            sparse_weight=float(getattr(args, "spif_sparse_weight", 0.001)),
            consistency_dropout=float(getattr(args, "spif_consistency_dropout", 0.1)),
            local_sw_num_projections=int(getattr(args, "spif_local_sw_num_projections", 64)),
            local_sw_p=float(getattr(args, "spif_local_sw_p", 2.0)),
            local_sw_normalize=_bool_flag(getattr(args, "spif_local_sw_normalize", "true"), default=True),
            local_sw_score_scale=float(getattr(args, "spif_local_sw_score_scale", 1.0)),
            backbone_name=fewshot_backbone,
            image_size=image_size,
            resnet12_drop_rate=0.0,
            resnet12_dropblock_size=5,
        )
    if args.model in {"spifce_local_papersw", "spifmax_local_papersw"}:
        SPIFCELocalPaperSW, SPIFMAXLocalPaperSW = _load_symbols(
            "net.spif_local_papersw",
            "SPIFCELocalPaperSW",
            "SPIFMAXLocalPaperSW",
        )
        spif_cls = SPIFMAXLocalPaperSW if args.model == "spifmax_local_papersw" else SPIFCELocalPaperSW
        stable_dim = int(getattr(args, "spif_stable_dim", 64) or 64)
        variant_dim = int(getattr(args, "spif_variant_dim", stable_dim) or stable_dim)
        gate_hidden = int(getattr(args, "spif_gate_hidden", max(1, stable_dim // 4)) or max(1, stable_dim // 4))
        consistency_weight = 0.0 if args.model == "spifce_local_papersw" else float(
            getattr(args, "spif_consistency_weight", 0.1)
        )
        decorr_weight = 0.0 if args.model == "spifce_local_papersw" else float(
            getattr(args, "spif_decorr_weight", 0.01)
        )
        sparse_weight = 0.0 if args.model == "spifce_local_papersw" else float(
            getattr(args, "spif_sparse_weight", 0.001)
        )
        learnable_alpha = (
            _bool_flag(getattr(args, "spif_learnable_alpha", "true"), default=True)
            if args.model == "spifmax_local_papersw"
            else _bool_flag(getattr(args, "spif_learnable_alpha", "false"), default=False)
        )
        return spif_cls(
            in_channels=3,
            hidden_dim=64,
            stable_dim=stable_dim,
            variant_dim=variant_dim,
            gate_hidden=gate_hidden,
            top_r=int(getattr(args, "spif_top_r", 3)),
            alpha_init=float(getattr(args, "spif_alpha_init", 0.7)),
            learnable_alpha=learnable_alpha,
            gate_on=_bool_flag(getattr(args, "spif_gate_on", "true"), default=True),
            factorization_on=_bool_flag(getattr(args, "spif_factorization_on", "true"), default=True),
            global_only=_bool_flag(getattr(args, "spif_global_only", "false"), default=False),
            local_only=_bool_flag(getattr(args, "spif_local_only", "false"), default=False),
            token_l2norm=_bool_flag(getattr(args, "spif_token_l2norm", "true"), default=True),
            consistency_weight=consistency_weight,
            decorr_weight=decorr_weight,
            sparse_weight=sparse_weight,
            consistency_dropout=float(getattr(args, "spif_consistency_dropout", 0.1)),
            local_papersw_train_num_projections=int(getattr(args, "spif_papersw_train_num_projections", 128)),
            local_papersw_eval_num_projections=int(getattr(args, "spif_papersw_eval_num_projections", 512)),
            local_papersw_p=float(getattr(args, "spif_papersw_p", 2.0)),
            local_papersw_normalize_inputs=_bool_flag(
                getattr(args, "spif_papersw_normalize_inputs", "false"),
                default=False,
            ),
            local_papersw_train_projection_mode=str(
                getattr(args, "spif_papersw_train_projection_mode", "resample")
            ),
            local_papersw_eval_projection_mode=str(
                getattr(args, "spif_papersw_eval_projection_mode", "fixed")
            ),
            local_papersw_eval_num_repeats=int(getattr(args, "spif_papersw_eval_num_repeats", 1)),
            local_papersw_score_scale=float(getattr(args, "spif_papersw_score_scale", 8.0)),
            local_papersw_projection_seed=int(getattr(args, "spif_papersw_projection_seed", 7)),
            local_papersw_shot_agg=str(getattr(args, "spif_papersw_shot_agg", "pooled")),
            local_papersw_shot_softmin_beta=float(getattr(args, "spif_papersw_shot_softmin_beta", 10.0)),
            local_papersw_debug=_bool_flag(getattr(args, "spif_papersw_debug", "false"), default=False),
            backbone_name=fewshot_backbone,
            image_size=image_size,
            resnet12_drop_rate=0.0,
            resnet12_dropblock_size=5,
        )
    if args.model in {"spifmambace", "spifmambamax", "spifmambaadjce", "spifmambaadjmax"}:
        SPIFMambaCE, SPIFMambaMAX = _load_symbols("net.spif_mamba", "SPIFMambaCE", "SPIFMambaMAX")
        spif_cls = SPIFMambaMAX if args.model in {"spifmambamax", "spifmambaadjmax"} else SPIFMambaCE
        stable_dim = int(getattr(args, "spif_stable_dim", 64) or 64)
        variant_dim = int(getattr(args, "spif_variant_dim", stable_dim) or stable_dim)
        gate_hidden = int(getattr(args, "spif_gate_hidden", max(1, stable_dim // 4)) or max(1, stable_dim // 4))
        consistency_weight = 0.0 if args.model in {"spifmambace", "spifmambaadjce"} else float(
            getattr(args, "spif_adj_consistency_weight", 0.1)
        )
        decorr_weight = 0.0 if args.model in {"spifmambace", "spifmambaadjce"} else float(
            getattr(args, "spif_adj_decorr_weight", 0.01)
        )
        sparse_raw_weight = 0.0 if args.model in {"spifmambace", "spifmambaadjce"} else float(
            getattr(args, "spif_adj_sparse_raw_weight", 5e-4)
        )
        sparse_final_weight = 0.0 if args.model in {"spifmambace", "spifmambaadjce"} else float(
            getattr(args, "spif_adj_sparse_final_weight", 5e-4)
        )
        learnable_alpha = (
            _bool_flag(getattr(args, "spif_learnable_alpha", "true"), default=True)
            if args.model in {"spifmambamax", "spifmambaadjmax"}
            else _bool_flag(getattr(args, "spif_learnable_alpha", "false"), default=False)
        )
        return spif_cls(
            in_channels=3,
            hidden_dim=64,
            stable_dim=stable_dim,
            variant_dim=variant_dim,
            gate_hidden=gate_hidden,
            top_r=int(getattr(args, "spif_top_r", 3)),
            alpha_init=float(getattr(args, "spif_alpha_init", 0.7)),
            learnable_alpha=learnable_alpha,
            factorization_on=_bool_flag(getattr(args, "spif_factorization_on", "true"), default=True),
            global_only=_bool_flag(getattr(args, "spif_global_only", "false"), default=False),
            local_only=_bool_flag(getattr(args, "spif_local_only", "false"), default=False),
            token_l2norm=_bool_flag(getattr(args, "spif_token_l2norm", "true"), default=True),
            use_raw_gate=_bool_flag(getattr(args, "spif_adj_use_raw_gate", "true"), default=True),
            use_final_gate=_bool_flag(getattr(args, "spif_adj_use_final_gate", "true"), default=True),
            beta_init=float(getattr(args, "spif_adj_beta_init", 0.1)),
            learnable_beta=_bool_flag(getattr(args, "spif_adj_learnable_beta", "true"), default=True),
            consistency_weight=consistency_weight,
            decorr_weight=decorr_weight,
            sparse_raw_weight=sparse_raw_weight,
            sparse_final_weight=sparse_final_weight,
            consistency_dropout=float(getattr(args, "spif_adj_consistency_dropout", 0.1)),
            mamba_state_dim=int(getattr(args, "ssm_state_dim", 16)),
            mamba_depth=int(getattr(args, "spif_mamba_depth", 1)),
            mamba_ffn_multiplier=int(getattr(args, "spif_mamba_ffn_multiplier", 2)),
            mamba_d_conv=int(getattr(args, "spif_mamba_d_conv", 4)),
            mamba_expand=int(getattr(args, "spif_mamba_expand", 2)),
            mamba_dropout=float(getattr(args, "spif_mamba_dropout", 0.0)),
            backbone_name=fewshot_backbone,
            image_size=image_size,
            resnet12_drop_rate=0.0,
            resnet12_dropblock_size=5,
        )
    if args.model in {"spifharmce", "spifharmmax"}:
        SPIFHarmCE, SPIFHarmMAX = _load_symbols("net.spif_harmonic", "SPIFHarmCE", "SPIFHarmMAX")
        spif_cls = SPIFHarmMAX if args.model == "spifharmmax" else SPIFHarmCE
        stable_dim = int(getattr(args, "spif_stable_dim", 64) or 64)
        variant_dim = int(getattr(args, "spif_variant_dim", stable_dim) or stable_dim)
        gate_hidden = int(getattr(args, "spif_gate_hidden", max(1, stable_dim // 4)) or max(1, stable_dim // 4))
        return spif_cls(
            in_channels=3,
            hidden_dim=64,
            stable_dim=stable_dim,
            variant_dim=variant_dim,
            gate_hidden=gate_hidden,
            gate_on=_bool_flag(getattr(args, "spif_gate_on", "true"), default=True),
            factorization_on=_bool_flag(getattr(args, "spif_factorization_on", "true"), default=True),
            global_only=_bool_flag(getattr(args, "spif_global_only", "false"), default=False),
            local_only=_bool_flag(getattr(args, "spif_local_only", "false"), default=False),
            token_l2norm=_bool_flag(getattr(args, "spif_token_l2norm", "true"), default=True),
            global_scale=float(getattr(args, "spif_harm_global_scale", 16.0)),
            fusion_hidden_dim=int(getattr(args, "spif_harm_fusion_hidden_dim", 32)),
            mamba_state_dim=int(getattr(args, "ssm_state_dim", 16)),
            mamba_depth=int(getattr(args, "spif_mamba_depth", 1)),
            mamba_ffn_multiplier=int(getattr(args, "spif_mamba_ffn_multiplier", 2)),
            mamba_d_conv=int(getattr(args, "spif_mamba_d_conv", 4)),
            mamba_expand=int(getattr(args, "spif_mamba_expand", 2)),
            mamba_dropout=float(getattr(args, "spif_mamba_dropout", 0.0)),
            beta_init=float(getattr(args, "spif_harm_beta_init", 0.1)),
            learnable_beta=_bool_flag(getattr(args, "spif_harm_learnable_beta", "true"), default=True),
            use_final_gate=_bool_flag(getattr(args, "spif_harm_use_final_gate", "true"), default=True),
            local_papersw_train_num_projections=int(getattr(args, "spif_papersw_train_num_projections", 128)),
            local_papersw_eval_num_projections=int(getattr(args, "spif_papersw_eval_num_projections", 512)),
            local_papersw_p=float(getattr(args, "spif_papersw_p", 2.0)),
            local_papersw_normalize_inputs=_bool_flag(
                getattr(args, "spif_papersw_normalize_inputs", "false"),
                default=False,
            ),
            local_papersw_train_projection_mode=str(
                getattr(args, "spif_papersw_train_projection_mode", "resample")
            ),
            local_papersw_eval_projection_mode=str(
                getattr(args, "spif_papersw_eval_projection_mode", "fixed")
            ),
            local_papersw_eval_num_repeats=int(getattr(args, "spif_papersw_eval_num_repeats", 1)),
            local_papersw_score_scale=float(getattr(args, "spif_harm_local_score_scale", 8.0)),
            local_papersw_projection_seed=int(getattr(args, "spif_papersw_projection_seed", 7)),
            factor_consistency_weight=float(getattr(args, "spif_harm_factor_consistency_weight", 0.0)),
            factor_decorr_weight=float(getattr(args, "spif_harm_factor_decorr_weight", 0.0)),
            factor_sparse_weight=float(getattr(args, "spif_harm_factor_sparse_weight", 0.0)),
            factor_consistency_dropout=float(getattr(args, "spif_harm_factor_consistency_dropout", 0.1)),
            final_sparse_weight=float(getattr(args, "spif_harm_final_sparse_weight", 0.0)),
            global_ce_weight=float(getattr(args, "spif_harm_global_ce_weight", 0.0)),
            local_ce_weight=float(getattr(args, "spif_harm_local_ce_weight", 0.0)),
            backbone_name=fewshot_backbone,
            image_size=image_size,
            resnet12_drop_rate=0.0,
            resnet12_dropblock_size=5,
        )
    if args.model in {"spifv2ce", "spifv2max"}:
        SPIFv2CE, SPIFv2MAX = _load_symbols("net.spif_v2", "SPIFv2CE", "SPIFv2MAX")
        spif_cls = SPIFv2MAX if args.model == "spifv2max" else SPIFv2CE
        shared_dim = int(getattr(args, "spifv2_shared_dim", 64) or 64)
        stable_dim = int(getattr(args, "spifv2_stable_dim", 32) or 32)
        variant_dim = int(getattr(args, "spifv2_variant_dim", 64) or 64)
        gate_hidden = int(getattr(args, "spifv2_gate_hidden", max(1, stable_dim // 4)) or max(1, stable_dim // 4))
        learnable_lambda_local = (
            _bool_flag(getattr(args, "spifv2_learnable_lambda_local", "true"), default=True)
            if args.model == "spifv2max"
            else _bool_flag(getattr(args, "spifv2_learnable_lambda_local", "false"), default=False)
        )
        learnable_alpha = (
            _bool_flag(getattr(args, "spifv2_learnable_alpha", "true"), default=True)
            if args.model == "spifv2max"
            else _bool_flag(getattr(args, "spifv2_learnable_alpha", "false"), default=False)
        )
        consistency_weight = 0.0 if args.model == "spifv2ce" else float(getattr(args, "spifv2_consistency_weight", 0.05))
        decorr_weight = 0.0 if args.model == "spifv2ce" else float(getattr(args, "spifv2_decorr_weight", 0.005))
        gate_reg_weight = 0.0 if args.model == "spifv2ce" else float(getattr(args, "spifv2_gate_reg_weight", 5e-4))
        return spif_cls(
            in_channels=3,
            hidden_dim=64,
            shared_dim=shared_dim,
            stable_dim=stable_dim,
            variant_dim=variant_dim,
            gate_hidden=gate_hidden,
            top_r=int(getattr(args, "spifv2_top_r", 3)),
            gate_type=str(getattr(args, "spifv2_gate_type", "sparsemax")),
            use_shared_bottleneck=_bool_flag(getattr(args, "spifv2_use_shared_bottleneck", "true"), default=True),
            mutual_local=_bool_flag(getattr(args, "spifv2_mutual_local", "true"), default=True),
            fusion_mode=str(getattr(args, "spifv2_fusion_mode", "residual")),
            lambda_local_init=float(getattr(args, "spifv2_lambda_local_init", 0.2)),
            learnable_lambda_local=learnable_lambda_local,
            alpha_init=float(getattr(args, "spifv2_alpha_init", 0.7)),
            learnable_alpha=learnable_alpha,
            global_only=_bool_flag(getattr(args, "spifv2_global_only", "false"), default=False),
            local_only=_bool_flag(getattr(args, "spifv2_local_only", "false"), default=False),
            token_l2norm=_bool_flag(getattr(args, "spifv2_token_l2norm", "true"), default=True),
            consistency_weight=consistency_weight,
            decorr_weight=decorr_weight,
            gate_reg_weight=gate_reg_weight,
            consistency_dropout=float(getattr(args, "spifv2_consistency_dropout", 0.1)),
            backbone_name=fewshot_backbone,
            image_size=image_size,
            resnet12_drop_rate=0.0,
            resnet12_dropblock_size=5,
        )
    if args.model == "spifv3_cstn":
        CanonicalSetTransportNet = _load_symbol("net.spif_v3_cstn", "CanonicalSetTransportNet")
        return CanonicalSetTransportNet(
            in_channels=3,
            hidden_dim=64,
            d_model=getattr(args, "d_model", None) or getattr(args, "token_dim", None),
            temperature=float(getattr(args, "spifv3_temperature", 16.0)),
            backbone_name=fewshot_backbone,
            image_size=image_size,
            resnet12_drop_rate=0.0,
            resnet12_dropblock_size=5,
            num_atoms=int(getattr(args, "num_atoms", 8)),
            num_sw_projections=int(getattr(args, "num_sw_projections", 64)),
            assignment_temperature=float(getattr(args, "assignment_temperature", 1.0)),
            set_pool_heads=int(getattr(args, "cstn_set_pool_heads", 4)),
            use_atom_entropy_reg=_bool_flag(getattr(args, "use_atom_entropy_reg", "false"), default=False),
            atom_entropy_reg_weight=float(getattr(args, "atom_entropy_reg_weight", 0.01)),
            use_support_consistency_reg=_bool_flag(
                getattr(args, "use_support_consistency_reg", "false"),
                default=False,
            ),
            support_consistency_reg_weight=float(getattr(args, "support_consistency_reg_weight", 0.01)),
        )
    if args.model == "spifv3_cstn_papersw":
        CanonicalSetTransportPaperSWNet = _load_symbol(
            "net.spif_v3_cstn_papersw",
            "CanonicalSetTransportPaperSWNet",
        )
        return CanonicalSetTransportPaperSWNet(
            in_channels=3,
            hidden_dim=64,
            d_model=getattr(args, "d_model", None) or getattr(args, "token_dim", None),
            temperature=float(getattr(args, "spifv3_temperature", 16.0)),
            backbone_name=fewshot_backbone,
            image_size=image_size,
            resnet12_drop_rate=0.0,
            resnet12_dropblock_size=5,
            num_atoms=int(getattr(args, "num_atoms", 8)),
            assignment_temperature=float(getattr(args, "assignment_temperature", 1.0)),
            set_pool_heads=int(getattr(args, "cstn_set_pool_heads", 4)),
            use_atom_entropy_reg=_bool_flag(getattr(args, "use_atom_entropy_reg", "false"), default=False),
            atom_entropy_reg_weight=float(getattr(args, "atom_entropy_reg_weight", 0.01)),
            use_support_consistency_reg=_bool_flag(
                getattr(args, "use_support_consistency_reg", "false"),
                default=False,
            ),
            support_consistency_reg_weight=float(getattr(args, "support_consistency_reg_weight", 0.01)),
            train_num_projections=int(getattr(args, "spif_papersw_train_num_projections", 128)),
            eval_num_projections=int(getattr(args, "spif_papersw_eval_num_projections", 512)),
            p=float(getattr(args, "spif_papersw_p", 2.0)),
            normalize_inputs=_bool_flag(getattr(args, "spif_papersw_normalize_inputs", "false"), default=False),
            train_projection_mode=str(getattr(args, "spif_papersw_train_projection_mode", "resample")),
            eval_projection_mode=str(getattr(args, "spif_papersw_eval_projection_mode", "fixed")),
            eval_num_repeats=int(getattr(args, "spif_papersw_eval_num_repeats", 1)),
            projection_seed=int(getattr(args, "spif_papersw_projection_seed", 7)),
        )
    if args.model == "crj_fsl":
        CRJFSL = _load_symbol("net.crj_fsl", "CRJFSL")
        hidden_dim = fewshot_backbone_output_dim(fewshot_backbone)
        return CRJFSL(
            in_channels=3,
            hidden_dim=hidden_dim,
            token_dim=int(
                _first_attr(args, "crj_token_dim", "hrot_token_dim", "token_dim", default=128) or 128
            ),
            use_raw_backbone_tokens=_bool_flag(
                _first_attr(args, "crj_use_raw_backbone_tokens", "hrot_use_raw_backbone_tokens", default="false"),
                default=False,
            ),
            backbone_name=fewshot_backbone,
            image_size=image_size,
            resnet12_drop_rate=0.0,
            resnet12_dropblock_size=5,
            eam_hidden_dim=int(getattr(args, "hrot_eam_hidden_dim", 256)),
            curvature_init=float(getattr(args, "hrot_curvature_init", 1.0)),
            projection_scale=float(getattr(args, "hrot_projection_scale", 0.1)),
            token_temperature=float(getattr(args, "hrot_token_temperature", 0.1)),
            score_scale=float(getattr(args, "hrot_score_scale", 16.0)),
            tau_q=float(getattr(args, "hrot_tau_q", 0.5)),
            tau_c=float(getattr(args, "hrot_tau_c", 0.5)),
            sinkhorn_epsilon=float(getattr(args, "hrot_sinkhorn_epsilon", 0.1)),
            sinkhorn_iterations=int(
                _first_attr(args, "hrot_sinkhorn_iterations", "hrot_sinkhorn_iters", default=60)
            ),
            sinkhorn_tolerance=float(getattr(args, "hrot_sinkhorn_tolerance", 1e-5)),
            fixed_mass=float(getattr(args, "hrot_fixed_mass", 0.8)),
            min_mass=float(getattr(args, "hrot_min_mass", 0.1)),
            mass_bonus_init=float(getattr(args, "hrot_mass_bonus_init", 1.0)),
            transport_cost_threshold_init=getattr(args, "hrot_transport_cost_threshold_init", None),
            lambda_rho=float(getattr(args, "hrot_lambda_rho", 0.01)),
            rho_target=float(getattr(args, "hrot_rho_target", 0.8)),
            lambda_rho_rank=float(getattr(args, "hrot_lambda_rho_rank", 0.05)),
            rho_rank_margin=float(getattr(args, "hrot_rho_rank_margin", 0.05)),
            rho_rank_temperature=float(getattr(args, "hrot_rho_rank_temperature", 0.05)),
            lambda_curvature=float(getattr(args, "hrot_lambda_curvature", 0.0)),
            min_curvature=float(getattr(args, "hrot_min_curvature", 0.05)),
            structure_cost_init=float(getattr(args, "hrot_structure_cost_init", 0.05)),
            ground_cost=str(getattr(args, "hrot_ground_cost", "auto")),
            eam_mode=str(getattr(args, "hrot_eam_mode", "compact")),
            compact_eam_prior_mix=float(getattr(args, "hrot_compact_eam_prior_mix", 0.5)),
            normalize_euclidean_tokens=_bool_flag(
                getattr(args, "hrot_normalize_euclidean_tokens", "true"),
                default=True,
            ),
            normalize_rho=_bool_flag(getattr(args, "hrot_normalize_rho", "false"), default=False),
            eval_use_float64=_bool_flag(getattr(args, "hrot_eval_use_float64", "true"), default=True),
            hyperbolic_backend=str(getattr(args, "hrot_hyperbolic_backend", "auto")),
            ot_backend=str(getattr(args, "hrot_ot_backend", "native")),
            eps=float(getattr(args, "hrot_eps", 1e-6)),
            crj_pool_gamma=float(getattr(args, "crj_pool_gamma", 0.65)),
            crj_variance_penalty=float(getattr(args, "crj_variance_penalty", 0.25)),
            crj_min_shots=int(getattr(args, "crj_min_shots", 2)),
            crj_trim_fraction=float(getattr(args, "crj_trim_fraction", 0.0)),
            crj_clip_logit=float(getattr(args, "crj_clip_logit", 0.0)),
        )
    if args.model in HROT_FSL_MODEL_NAMES:
        is_ours_final_model = args.model in OURS_FINAL_MODEL_NAMES
        is_ours_final_verified_uot_model = args.model in OURS_FINAL_VERIFIED_UOT_MODEL_NAMES
        is_ours_final_partial_ot_model = args.model in OURS_FINAL_PARTIAL_OT_MODEL_NAMES
        is_ours_entrypoint_model = args.model in OURS_ENTRYPOINT_MODEL_NAMES
        is_m2_model = args.model in M2_MODEL_NAMES
        is_m2_like_model = args.model in M2_LIKE_MODEL_NAMES
        is_m2_full_ot = args.model in M2_FULL_OT_MODEL_NAMES
        is_ours_cpm_model = args.model in OURS_CPM_MODEL_NAMES
        is_sgpot_model = args.model in SGPOT_MODEL_NAMES
        HROTFSL = (
            _load_symbol("net.sg_pot", "SGPOT")
            if is_sgpot_model
            else
            _load_symbol("net.ours_cpm", "OursCPM")
            if is_ours_cpm_model
            else
            _load_symbol("net.ours", "OursM2")
            if is_ours_entrypoint_model
            else
            _load_symbol("net.jecot_m2", "JECOTM2")
            if is_m2_model
            else _load_symbol("net.hrot_fsl", "HROTFSL")
        )
        hidden_dim = fewshot_backbone_output_dim(fewshot_backbone)
        m2_default_rho = 1.0 if is_m2_full_ot else 0.80
        ecot_rho_bank = getattr(args, "hrot_ecot_rho_bank", None)
        ecot_base_rho = getattr(args, "hrot_ecot_base_rho", None)
        if is_m2_like_model and not is_ours_cpm_model and ecot_rho_bank is None:
            ecot_rho_bank = f"{m2_default_rho:.6g}"
        if is_m2_like_model and not is_ours_cpm_model and ecot_base_rho is None:
            ecot_base_rho = m2_default_rho
        ecot_transport_mode = getattr(args, "hrot_ecot_transport_mode", None)
        if is_m2_like_model and not is_ours_cpm_model and ecot_transport_mode is None:
            ecot_transport_mode = "balanced" if is_m2_full_ot else "unbalanced"
        ecot_enable_egsm_flag = _bool_flag(resolve_hrot_ecot_enable_egsm(args), default=False)
        default_m2_ablate_threshold_mass = "false" if is_ours_final_model else (
            "true" if is_m2_like_model else "false"
        )
        return HROTFSL(
            in_channels=3,
            hidden_dim=hidden_dim,
            token_dim=resolve_hrot_token_dim(args, default=128),
            use_raw_backbone_tokens=_bool_flag(
                getattr(args, "hrot_use_raw_backbone_tokens", "false"),
                default=False,
            ),
            backbone_name=fewshot_backbone,
            image_size=image_size,
            resnet12_drop_rate=0.0,
            resnet12_dropblock_size=5,
            variant="J_ECOT_M2" if is_m2_like_model else str(getattr(args, "hrot_variant", "E")),
            **(
                _build_sgpot_kwargs(args)
                if is_sgpot_model
                else
                {
                    "cpm_ablation": str(getattr(args, "cpm_ablation", "full")),
                    "cpm_alpha": float(getattr(args, "cpm_alpha", 1.0)),
                    "cpm_rho_bank": getattr(args, "cpm_rho_bank", None),
                }
                if is_ours_cpm_model
                else
                {
                    "ours_ablation": str(getattr(args, "ours_ablation", "full")),
                    "use_differential_mode": _bool_flag(
                        getattr(args, "use_differential_mode", "false"),
                        default=False,
                    ),
                    "dm_alpha": float(getattr(args, "dm_alpha", 0.0)),
                    "dm_debug": str(getattr(args, "dm_debug", "auto")),
                    "dm_debug_dir": str(getattr(args, "dm_debug_dir", "results/dmt_debug")),
                    "dm_debug_max_episodes": int(getattr(args, "dm_debug_max_episodes", 5)),
                    **(resolve_ours_final_dmuot_config(args) if is_ours_final_model else {}),
                    **(resolve_ours_final_evidence_config(args) if is_ours_final_model else {}),
                    **(
                        {
                            "enable_verified_uot_score": True,
                            "verified_uot_beta": float(getattr(args, "verified_uot_beta", 0.75)),
                            "verified_uot_tau": float(getattr(args, "verified_uot_tau", 0.10)),
                            "verified_uot_ratio_threshold": float(
                                getattr(args, "verified_uot_ratio_threshold", 0.35)
                            ),
                            "verified_uot_kernel_size": int(getattr(args, "verified_uot_kernel_size", 3)),
                        }
                        if is_ours_final_model
                        and (
                            is_ours_final_verified_uot_model
                            or _bool_flag(getattr(args, "enable_verified_uot_score", False), default=False)
                        )
                        else {}
                    ),
                    **(
                        {
                            "enable_global_residual_score": True,
                            "global_residual_mode": str(
                                getattr(args, "global_residual_mode", "residual")
                            ),
                            "global_residual_weight": float(
                                getattr(args, "global_residual_weight", 0.10)
                            ),
                        }
                        if is_ours_final_model
                        and _bool_flag(getattr(args, "enable_global_residual_score", False), default=False)
                        else {}
                    ),
                    **(
                        {
                            "enable_multiscale_ot": True,
                            "multiscale_pool_sizes": str(
                                getattr(args, "multiscale_pool_sizes", "original,2x2,1x1")
                            ),
                            "multiscale_per_scale_T": _bool_flag(
                                getattr(args, "multiscale_per_scale_T", False),
                                default=False,
                            ),
                        }
                        if is_ours_final_model
                        and _bool_flag(getattr(args, "enable_multiscale_ot", False), default=False)
                        else {}
                    ),
                    **(
                        {
                            "enable_mspta": True,
                            "mspta_scales": getattr(args, "mspta_scales", (1, 2, 3)),
                            "mspta_mass_mode": str(getattr(args, "mspta_mass_mode", "compact_area")),
                            "mspta_normalize": _bool_flag(
                                getattr(args, "mspta_normalize", True),
                                default=True,
                            ),
                            "mspta_proj_dim": int(getattr(args, "mspta_proj_dim", 0)),
                            "mspta_compact_floor": float(getattr(args, "mspta_compact_floor", 0.05)),
                            "mspta_vertical_suppression": float(
                                getattr(args, "mspta_vertical_suppression", 1.0)
                            ),
                            "mspta_saliency_cost_discount": float(
                                getattr(args, "mspta_saliency_cost_discount", 0.10)
                            ),
                            "mspta_guidance_source": str(getattr(args, "mspta_guidance_source", "mixed")),
                            "mspta_learnable_weights": _bool_flag(
                                getattr(args, "mspta_learnable_weights", False),
                                default=False,
                            ),
                        }
                        if is_ours_final_model
                        and _bool_flag(getattr(args, "enable_mspta", False), default=False)
                        else {}
                    ),
                    **(
                        {
                            "enable_context_enrichment": True,
                            "context_kernel_sizes": str(
                                getattr(args, "context_kernel_sizes", "3,5")
                            ),
                            "context_fusion": str(
                                getattr(args, "context_fusion", "weighted_sum")
                            ),
                            "context_gate_max": float(
                                getattr(args, "context_gate_max", 1.0)
                            ),
                            "context_change_max": float(
                                getattr(args, "context_change_max", 0.0)
                            ),
                        }
                        if is_ours_final_model
                        and _bool_flag(getattr(args, "enable_context_enrichment", False), default=False)
                        else {}
                    ),
                    **(
                        {
                            "context_debug": True,
                            "context_debug_dir": str(
                                getattr(args, "context_debug_dir", "results/context_debug")
                            ),
                            "context_debug_max_episodes": int(
                                getattr(args, "context_debug_max_episodes", 3)
                            ),
                        }
                        if is_ours_final_model
                        and _bool_flag(getattr(args, "context_debug", False), default=False)
                        else {}
                    ),
                    **(
                        {
                            "enable_structural_augmentation": True,
                            "struct_dim": int(getattr(args, "struct_dim", 16)),
                        }
                        if is_ours_final_model
                        and _bool_flag(getattr(args, "enable_structural_augmentation", False), default=False)
                        else {}
                    ),
                    **(
                        {
                            "enable_region_structural_uot": True,
                            "region_uot_grid_size": str(getattr(args, "region_uot_grid_size", "3x3")),
                            "region_uot_strength": float(getattr(args, "region_uot_strength", 0.08)),
                            "region_uot_fgw_alpha": float(getattr(args, "region_uot_fgw_alpha", 0.35)),
                            "region_uot_sinkhorn_epsilon": float(
                                getattr(args, "region_uot_sinkhorn_epsilon", 0.08)
                            ),
                            "region_uot_fgw_iters": int(getattr(args, "region_uot_fgw_iters", 4)),
                            "region_uot_sinkhorn_iters": int(
                                getattr(args, "region_uot_sinkhorn_iters", 40)
                            ),
                            "region_uot_topk": int(getattr(args, "region_uot_topk", 3)),
                            "region_uot_fine_gate_quantile": float(
                                getattr(args, "region_uot_fine_gate_quantile", 0.35)
                            ),
                            "region_uot_min_confidence": float(
                                getattr(args, "region_uot_min_confidence", 0.10)
                            ),
                            "region_uot_importance_temperature": float(
                                getattr(args, "region_uot_importance_temperature", 0.50)
                            ),
                        }
                        if is_ours_final_model
                        and _bool_flag(getattr(args, "enable_region_structural_uot", False), default=False)
                        else {}
                    ),
                    **(
                        {
                            "enable_adaptive_region_uot": True,
                            "adaptive_region_num_slots": int(
                                getattr(args, "adaptive_region_num_slots", 4)
                            ),
                            "adaptive_region_context_kernels": str(
                                getattr(args, "adaptive_region_context_kernels", "1,3,5")
                            ),
                            "adaptive_region_cost_discount": float(
                                getattr(args, "adaptive_region_cost_discount", 0.12)
                            ),
                            "adaptive_region_mass_mix": float(
                                getattr(args, "adaptive_region_mass_mix", 0.60)
                            ),
                            "adaptive_region_sinkhorn_epsilon": float(
                                getattr(args, "adaptive_region_sinkhorn_epsilon", 0.08)
                            ),
                            "adaptive_region_sinkhorn_iters": int(
                                getattr(args, "adaptive_region_sinkhorn_iters", 30)
                            ),
                            "adaptive_region_fine_gate_quantile": float(
                                getattr(args, "adaptive_region_fine_gate_quantile", 0.40)
                            ),
                            "adaptive_region_temperature_min": float(
                                getattr(args, "adaptive_region_temperature_min", 0.35)
                            ),
                            "adaptive_region_temperature_max": float(
                                getattr(args, "adaptive_region_temperature_max", 1.25)
                            ),
                            "adaptive_region_init_gate": float(
                                getattr(args, "adaptive_region_init_gate", 0.05)
                            ),
                        }
                        if is_ours_final_model
                        and _bool_flag(getattr(args, "enable_adaptive_region_uot", False), default=False)
                        else {}
                    ),
                    **(
                        {
                            "enable_pulse_region_uot": True,
                            "pulse_region_kernel_size": int(getattr(args, "pulse_region_kernel_size", 5)),
                            "pulse_region_cost_weight": float(getattr(args, "pulse_region_cost_weight", 0.35)),
                            "pulse_saliency_mass_mix": float(getattr(args, "pulse_saliency_mass_mix", 0.50)),
                            "pulse_saliency_cost_discount": float(
                                getattr(args, "pulse_saliency_cost_discount", 0.10)
                            ),
                            "pulse_saliency_image_weight": float(
                                getattr(args, "pulse_saliency_image_weight", 0.45)
                            ),
                            "pulse_saliency_feature_weight": float(
                                getattr(args, "pulse_saliency_feature_weight", 0.35)
                            ),
                            "pulse_saliency_contrast_weight": float(
                                getattr(args, "pulse_saliency_contrast_weight", 0.20)
                            ),
                            "pulse_region_train_strength": float(
                                getattr(args, "pulse_region_train_strength", 1.0)
                            ),
                            "pulse_region_eval_strength": float(
                                getattr(args, "pulse_region_eval_strength", 1.0)
                            ),
                            "pulse_region_multishot_train_strength": getattr(
                                args, "pulse_region_multishot_train_strength", None
                            ),
                            "pulse_region_multishot_eval_strength": getattr(
                                args, "pulse_region_multishot_eval_strength", None
                            ),
                            "pulse_region_train_schedule": str(
                                getattr(args, "pulse_region_train_schedule", "constant")
                            ),
                            "pulse_support_consensus_weight": float(
                                getattr(args, "pulse_support_consensus_weight", 0.0)
                            ),
                            "pulse_support_consensus_beta": float(
                                getattr(args, "pulse_support_consensus_beta", 5.0)
                            ),
                            "pulse_support_consensus_eta": float(
                                getattr(args, "pulse_support_consensus_eta", 0.05)
                            ),
                            "pulse_evidence_score": _bool_flag(
                                getattr(args, "pulse_evidence_score", False),
                                default=False,
                            ),
                            "pulse_evidence_score_mode": str(
                                getattr(args, "pulse_evidence_score_mode", "replace")
                            ),
                            "pulse_evidence_score_mix": float(
                                getattr(args, "pulse_evidence_score_mix", 1.0)
                            ),
                            "pulse_evidence_mass_weight": float(
                                getattr(args, "pulse_evidence_mass_weight", 1.0)
                            ),
                            "pulse_evidence_cost_weight": float(
                                getattr(args, "pulse_evidence_cost_weight", 1.0)
                            ),
                            "pulse_background_penalty": float(
                                getattr(args, "pulse_background_penalty", 0.25)
                            ),
                            "enable_discriminative_uot": _bool_flag(
                                getattr(args, "enable_discriminative_uot", False),
                                default=False,
                            ),
                            "discriminative_uot_tau": float(
                                getattr(args, "discriminative_uot_tau", 0.05)
                            ),
                            "discriminative_uot_margin": float(
                                getattr(args, "discriminative_uot_margin", 0.02)
                            ),
                            "discriminative_uot_mix": float(
                                getattr(args, "discriminative_uot_mix", 1.0)
                            ),
                            "discriminative_uot_background_penalty": float(
                                getattr(args, "discriminative_uot_background_penalty", 0.25)
                            ),
                            "discriminative_uot_mass_weight": float(
                                getattr(args, "discriminative_uot_mass_weight", 1.0)
                            ),
                            "discriminative_uot_cost_weight": float(
                                getattr(args, "discriminative_uot_cost_weight", 1.0)
                            ),
                            "pulse_discriminative_evidence": _bool_flag(
                                getattr(args, "pulse_discriminative_evidence", False),
                                default=False,
                            ),
                            "pulse_discriminative_tau": float(
                                getattr(args, "pulse_discriminative_tau", 0.05)
                            ),
                            "pulse_discriminative_margin": float(
                                getattr(args, "pulse_discriminative_margin", 0.02)
                            ),
                            "pulse_discriminative_mix": float(
                                getattr(args, "pulse_discriminative_mix", 1.0)
                            ),
                        }
                        if is_ours_final_model
                        and _bool_flag(getattr(args, "enable_pulse_region_uot", False), default=False)
                        else {}
                    ),
                    **(
                        {
                            "enable_discriminative_uot": True,
                            "discriminative_uot_tau": float(
                                getattr(args, "discriminative_uot_tau", 0.05)
                            ),
                            "discriminative_uot_margin": float(
                                getattr(args, "discriminative_uot_margin", 0.02)
                            ),
                            "discriminative_uot_mix": float(
                                getattr(args, "discriminative_uot_mix", 1.0)
                            ),
                            "discriminative_uot_background_penalty": float(
                                getattr(args, "discriminative_uot_background_penalty", 0.25)
                            ),
                            "discriminative_uot_mass_weight": float(
                                getattr(args, "discriminative_uot_mass_weight", 1.0)
                            ),
                            "discriminative_uot_cost_weight": float(
                                getattr(args, "discriminative_uot_cost_weight", 1.0)
                            ),
                        }
                        if is_ours_final_model
                        and _bool_flag(getattr(args, "enable_discriminative_uot", False), default=False)
                        else {}
                    ),
                    **(
                        {
                            "enable_label_ot": True,
                            "label_ot_epsilon": float(getattr(args, "label_ot_epsilon", 0.75)),
                            "label_ot_iterations": int(getattr(args, "label_ot_iterations", 30)),
                            "label_ot_mix": float(getattr(args, "label_ot_mix", 1.0)),
                            "label_ot_min_queries_per_class": int(
                                getattr(args, "label_ot_min_queries_per_class", 2)
                            ),
                            "label_ot_min_column_imbalance": float(
                                getattr(args, "label_ot_min_column_imbalance", 0.0)
                            ),
                            "label_ot_max_bias": float(getattr(args, "label_ot_max_bias", 2.0)),
                        }
                        if is_ours_final_model
                        and _bool_flag(getattr(args, "enable_label_ot", False), default=False)
                        else {}
                    ),
                    **(
                        {
                            "enable_pot_guide": True,
                            "pot_guide_s": float(getattr(args, "pot_guide_s", 0.5)),
                            "pot_guide_adaptive_s": _bool_flag(
                                getattr(args, "pot_guide_adaptive_s", False),
                                default=False,
                            ),
                            "pot_guide_s_min": float(getattr(args, "pot_guide_s_min", 0.2)),
                            "pot_guide_s_max": float(getattr(args, "pot_guide_s_max", 0.8)),
                            "pot_guide_epsilon": float(getattr(args, "pot_guide_epsilon", 0.05)),
                            "pot_guide_max_iter": int(getattr(args, "pot_guide_max_iter", 50)),
                        }
                        if is_ours_final_model
                        and _bool_flag(getattr(args, "enable_pot_guide", False), default=False)
                        else {}
                    ),
                }
                if is_ours_entrypoint_model
                else {}
            ),
            eam_hidden_dim=int(getattr(args, "hrot_eam_hidden_dim", 256)),
            curvature_init=float(getattr(args, "hrot_curvature_init", 1.0)),
            projection_scale=float(getattr(args, "hrot_projection_scale", 0.1)),
            token_temperature=float(getattr(args, "hrot_token_temperature", 0.1)),
            score_scale=float(getattr(args, "hrot_score_scale", 16.0)),
            tau_q=float(getattr(args, "hrot_tau_q", 0.5)),
            tau_c=float(getattr(args, "hrot_tau_c", 0.5)),
            sinkhorn_epsilon=float(getattr(args, "hrot_sinkhorn_epsilon", 0.1)),
            sinkhorn_iterations=int(
                _first_attr(args, "hrot_sinkhorn_iterations", "hrot_sinkhorn_iters", default=60)
            ),
            sinkhorn_tolerance=float(getattr(args, "hrot_sinkhorn_tolerance", 1e-5)),
            fixed_mass=float(getattr(args, "hrot_fixed_mass", 0.8)),
            egtw_tau=float(getattr(args, "hrot_egtw_tau", 1.0)),
            egtw_lambda=float(getattr(args, "hrot_egtw_lambda", 1.0)),
            egtw_eps=float(getattr(args, "hrot_egtw_eps", 1e-8)),
            egtw_attention_temperature=float(getattr(args, "hrot_egtw_attention_temperature", 0.2)),
            egtw_support_similarity_weight=float(getattr(args, "hrot_egtw_support_similarity_weight", 0.5)),
            egtw_uniform_mix=float(getattr(args, "hrot_egtw_uniform_mix", 0.25)),
            egtw_detach_masses=_bool_flag(getattr(args, "hrot_egtw_detach_masses", "true"), default=True),
            egtw_learn_tau=_bool_flag(getattr(args, "hrot_egtw_learn_tau", "false"), default=False),
            egtw_learn_lambda=_bool_flag(getattr(args, "hrot_egtw_learn_lambda", "false"), default=False),
            pre_transport_shot_pool=_bool_flag(
                getattr(args, "hrot_pre_transport_shot_pool", "false"),
                default=False,
            ),
            pre_transport_shot_pool_mode=str(getattr(args, "hrot_pre_transport_shot_pool_mode", "mean")),
            hrot_tsw_enable=_bool_flag(getattr(args, "hrot_tsw_enable", "false"), default=False),
            hrot_tsw_share_gate=_bool_flag(getattr(args, "hrot_tsw_share_gate", "true"), default=True),
            ecot_rho_bank=ecot_rho_bank,
            ecot_base_rho=ecot_base_rho,
            ecot_budget_tau=float(getattr(args, "hrot_ecot_budget_tau", 1.0)),
            ecot_max_lambda=float(getattr(args, "hrot_ecot_max_lambda", 1.0)),
            ecot_lambda_init=float(getattr(args, "hrot_ecot_lambda_init", -8.0)),
            ecot_controller_hidden=int(getattr(args, "hrot_ecot_controller_hidden", 32)),
            ecot_uniform_budget_policy=_bool_flag(
                getattr(args, "hrot_ecot_uniform_budget_policy", "false"),
                default=False,
            ),
            ecot_enable_tau_shot=_bool_flag(
                getattr(args, "hrot_ecot_enable_tau_shot", "true"),
                default=True,
            ),
            ecot_tau_shot_min=float(getattr(args, "hrot_ecot_tau_shot_min", 0.5)),
            ecot_tau_shot_max=float(getattr(args, "hrot_ecot_tau_shot_max", 2.0)),
            ecot_enable_threshold_offset=_bool_flag(
                getattr(args, "hrot_ecot_enable_threshold_offset", "false"),
                default=False,
            ),
            ecot_m2_per_shot_threshold=_bool_flag(
                getattr(args, "hrot_ecot_m2_per_shot_threshold", "false"),
                default=False,
            ),
            ecot_m2_pst_hidden=int(getattr(args, "hrot_ecot_m2_pst_hidden", 32)),
            ecot_m2_ablate_threshold_mass=_bool_flag(
                (
                    "false"
                    if is_ours_final_partial_ot_model
                    else getattr(args, "hrot_ecot_m2_ablate_threshold_mass", default_m2_ablate_threshold_mass)
                ),
                default=(not is_ours_final_model and is_m2_like_model),
            ),
            ecot_m2_cost_per_mass_score=_bool_flag(
                (
                    "true"
                    if is_ours_final_partial_ot_model
                    else getattr(args, "hrot_ecot_m2_cost_per_mass_score", "false")
                ),
                default=False,
            ),
            ecot_m2_cost_per_mass_alpha=float(getattr(args, "hrot_ecot_m2_cost_per_mass_alpha", 1.0)),
            ecot_m2_cost_per_mass_detach_mass=(
                (False if getattr(args, "model", None) == CANONICAL_M2_MODEL_NAME else True)
                if getattr(args, "hrot_ecot_m2_cost_per_mass_detach_mass", None) is None
                else _bool_flag(getattr(args, "hrot_ecot_m2_cost_per_mass_detach_mass"), default=False)
            ),
            ecot_m2_mass_score_mode=str(getattr(args, "hrot_ecot_m2_mass_score_mode", "standard")),
            ecot_m2_consensus_mass_alpha=float(
                getattr(args, "hrot_ecot_m2_consensus_mass_alpha", 1.0)
            ),
            ecot_m2_mass_reward_beta=float(getattr(args, "hrot_ecot_m2_mass_reward_beta", 1.0)),
            ecot_m2_mass_reward_shot_scaling=str(
                getattr(args, "hrot_ecot_m2_mass_reward_shot_scaling", "none")
            ),
            ecot_m2_use_swts=_bool_flag(
                getattr(args, "hrot_ecot_m2_use_swts", "false"),
                default=False,
            ),
            ecot_m2_swts_temp=float(getattr(args, "hrot_ecot_m2_swts_temp", 1.0)),
            ecot_m2_use_aqm=_bool_flag(
                getattr(args, "hrot_ecot_m2_use_aqm", "false"),
                default=False,
            ),
            ecot_m2_tau_aqm=float(getattr(args, "hrot_ecot_m2_tau_aqm", 1.0)),
            use_mncr=_bool_flag(getattr(args, "hrot_use_mncr", "false"), default=False),
            mncr_temperature=float(getattr(args, "hrot_mncr_temperature", 0.5)),
            mncr_lam=float(getattr(args, "hrot_mncr_lam", 0.5)),
            ecot_identity_reg=float(getattr(args, "hrot_ecot_identity_reg", 1e-4)),
            ecot_policy_entropy_reg=float(getattr(args, "hrot_ecot_policy_entropy_reg", 1e-3)),
            ecot_consensus_tau_mode=str(getattr(args, "hrot_ecot_consensus_tau_mode", "fixed")),
            ecot_consensus_tau=float(getattr(args, "hrot_ecot_consensus_tau", 1.0)),
            ecot_enable_nncs_marginal=(
                None
                if getattr(args, "hrot_ecot_enable_nncs_marginal", None) is None
                else _bool_flag(getattr(args, "hrot_ecot_enable_nncs_marginal"), default=False)
            ),
            ecot_nncs_beta=float(getattr(args, "hrot_ecot_nncs_beta", 5.0)),
            ecot_nncs_eta=float(getattr(args, "hrot_ecot_nncs_eta", 0.05)),
            ecot_nncs_detach=_bool_flag(getattr(args, "hrot_ecot_nncs_detach", "true"), default=True),
            ecot_nncs_distance=str(getattr(args, "hrot_ecot_nncs_distance", "auto")),
            ecot_nncs_min_sigma=float(getattr(args, "hrot_ecot_nncs_min_sigma", 1e-6)),
            ecot_enable_crs_marginal=_bool_flag(
                getattr(args, "hrot_ecot_enable_crs_marginal", "false"),
                default=False,
            ),
            ecot_crs_use_cross_ref=_bool_flag(
                getattr(args, "hrot_ecot_crs_use_cross_ref", "true"),
                default=True,
            ),
            ecot_crs_use_ssm=_bool_flag(getattr(args, "hrot_ecot_crs_use_ssm", "true"), default=True),
            ecot_crs_eta_init=float(getattr(args, "hrot_ecot_crs_eta_init", 0.30)),
            ecot_crs_lambda_cr_init=float(getattr(args, "hrot_ecot_crs_lambda_cr_init", 0.50)),
            ecot_crs_tau_ssm_init=float(getattr(args, "hrot_ecot_crs_tau_ssm_init", 0.70)),
            ecot_crs_entropy_reg=float(getattr(args, "hrot_ecot_crs_entropy_reg", 0.0)),
            ecot_crs_side=str(getattr(args, "hrot_ecot_crs_side", "support")),
            ecot_crs_ssm_type=str(getattr(args, "hrot_ecot_crs_ssm_type", "auto")),
            ecot_enable_mea_marginal=_bool_flag(
                getattr(args, "hrot_ecot_enable_mea_marginal", "false"),
                default=False,
            ),
            ecot_mea_eta_init=float(getattr(args, "hrot_ecot_mea_eta_init", 0.35)),
            ecot_mea_temperature_init=float(getattr(args, "hrot_ecot_mea_temperature_init", 0.70)),
            ecot_mea_entropy_reg=float(getattr(args, "hrot_ecot_mea_entropy_reg", 0.0)),
            ecot_enable_ccdm_marginal=_bool_flag(
                getattr(args, "hrot_ecot_enable_ccdm_marginal", "false"),
                default=False,
            ),
            ecot_ccdm_tau_q_init=float(getattr(args, "hrot_ecot_ccdm_tau_q_init", 0.50)),
            ecot_ccdm_tau_b_init=float(getattr(args, "hrot_ecot_ccdm_tau_b_init", 0.50)),
            ecot_ccdm_entropy_reg=float(getattr(args, "hrot_ecot_ccdm_entropy_reg", 0.0)),
            ecot_ccdm_entropy_shot_weight=float(getattr(args, "hrot_ecot_ccdm_entropy_shot_weight", 0.0)),
            ecot_enable_egsm=ecot_enable_egsm_flag,
            ecot_egsm_hidden_dim=int(getattr(args, "hrot_ecot_egsm_hidden_dim", 32)),
            ecot_egsm_candidate_tau_q=float(getattr(args, "hrot_ecot_egsm_candidate_tau_q", 1.0)),
            ecot_egsm_candidate_tau_b=float(getattr(args, "hrot_ecot_egsm_candidate_tau_b", 1.0)),
            ecot_egsm_kappa_min=float(getattr(args, "hrot_ecot_egsm_kappa_min", 0.05)),
            ecot_egsm_kappa_max=float(getattr(args, "hrot_ecot_egsm_kappa_max", 0.35)),
            ecot_egsm_adaptive_rho=_bool_flag(
                getattr(args, "hrot_ecot_egsm_adaptive_rho", "false"), default=False,
            ),
            ecot_egsm_rho_delta_max=float(getattr(args, "hrot_ecot_egsm_rho_delta_max", 0.15)),
            ecot_egsm_rho_grad_clip=float(getattr(args, "hrot_ecot_egsm_rho_grad_clip", 1.0)),
            ecot_egsm_rho_reg_lambda=float(getattr(args, "hrot_ecot_egsm_rho_reg_lambda", 0.01)),
            ecot_enable_ccem_marginal=_bool_flag(
                getattr(args, "hrot_ecot_enable_ccem_marginal", "false"),
                default=False,
            ),
            ecot_ccem_uniform_mix=float(getattr(args, "hrot_ecot_ccem_uniform_mix", 0.05)),
            ecot_ccem_tau_q=float(getattr(args, "hrot_ecot_ccem_tau_q", 0.25)),
            ecot_ccem_tau_s=float(getattr(args, "hrot_ecot_ccem_tau_s", 0.25)),
            ecot_transport_mode=ecot_transport_mode,
            ecot_enable_noise_sink=_bool_flag(
                getattr(args, "hrot_ecot_enable_noise_sink", "false"),
                default=False,
            ),
            ecot_noise_sink_cost_init=float(getattr(args, "hrot_ecot_noise_sink_cost_init", 1.0)),
            ecot_noise_sink_score_penalty=float(getattr(args, "hrot_ecot_noise_sink_score_penalty", 0.0)),
            ecot_episode_feature_normalize=_bool_flag(
                getattr(args, "hrot_ecot_episode_feature_normalize", "false"),
                default=False,
            ),
            ecot_episode_feature_norm_eps=float(getattr(args, "hrot_ecot_episode_feature_norm_eps", 1e-6)),
            hrot_amp_marginals=_bool_flag(
                getattr(args, "hrot_amp_marginals", "false"),
                default=False,
            ),
            hrot_amp_marginals_tau=float(getattr(args, "hrot_amp_marginals_tau", 1.0)),
            hrot_token_center=_bool_flag(
                getattr(args, "hrot_token_center", "false"),
                default=False,
            ),
            hrot_token_center_query=_bool_flag(
                getattr(args, "hrot_token_center_query", "true"),
                default=True,
            ),
            ncet_mix_init=float(getattr(args, "hrot_ncet_mix_init", 0.25)),
            ncet_real_penalty_init=float(getattr(args, "hrot_ncet_real_penalty_init", 0.25)),
            ncet_null_penalty_init=float(getattr(args, "hrot_ncet_null_penalty_init", 0.05)),
            ncet_sink_cost_init=float(getattr(args, "hrot_ncet_sink_cost_init", 1.0)),
            hlm_min_mass=float(getattr(args, "hrot_hlm_min_mass", 0.1)),
            hlm_init_mass=float(getattr(args, "hrot_hlm_init_mass", 0.8)),
            hlm_budget_mode=str(getattr(args, "hrot_hlm_budget_mode", "cost")),
            hlm_token_mode=str(getattr(args, "hrot_hlm_token_mode", "cost")),
            hlm_token_tau=float(getattr(args, "hrot_hlm_token_tau", 0.25)),
            hlm_budget_hidden_dim=int(getattr(args, "hrot_hlm_budget_hidden_dim", 64)),
            hlm_token_hidden_dim=int(getattr(args, "hrot_hlm_token_hidden_dim", 64)),
            hlm_detach_cost_features=_bool_flag(
                getattr(args, "hrot_hlm_detach_cost_features", "true"),
                default=True,
            ),
            hlm_allow_mass_grad=_bool_flag(getattr(args, "hrot_hlm_allow_mass_grad", "true"), default=True),
            hlm_lambda_rho=float(getattr(args, "hrot_hlm_lambda_rho", 0.0)),
            hlm_lambda_eff=float(getattr(args, "hrot_hlm_lambda_eff", 0.0)),
            hlm_eff_margin=float(getattr(args, "hrot_hlm_eff_margin", 0.05)),
            hlm_lambda_shot_cov=float(getattr(args, "hrot_hlm_lambda_shot_cov", 0.0)),
            hlm_shot_entropy_floor=float(getattr(args, "hrot_hlm_shot_entropy_floor", 0.35)),
            hlm_return_diagnostics=_bool_flag(
                getattr(args, "hrot_hlm_return_diagnostics", "true"),
                default=True,
            ),
            min_mass=float(getattr(args, "hrot_min_mass", 0.1)),
            mass_bonus_init=float(getattr(args, "hrot_mass_bonus_init", 1.0)),
            transport_cost_threshold_init=getattr(args, "hrot_transport_cost_threshold_init", None),
            lambda_rho=float(getattr(args, "hrot_lambda_rho", 0.01)),
            rho_target=float(getattr(args, "hrot_rho_target", 0.8)),
            lambda_rho_rank=float(getattr(args, "hrot_lambda_rho_rank", 0.05)),
            rho_rank_margin=float(getattr(args, "hrot_rho_rank_margin", 0.05)),
            rho_rank_temperature=float(getattr(args, "hrot_rho_rank_temperature", 0.05)),
            lambda_curvature=float(getattr(args, "hrot_lambda_curvature", 0.0)),
            min_curvature=float(getattr(args, "hrot_min_curvature", 0.05)),
            structure_cost_init=float(getattr(args, "hrot_structure_cost_init", 0.05)),
            ground_cost=str(getattr(args, "hrot_ground_cost", "auto")),
            eam_mode=str(getattr(args, "hrot_eam_mode", "compact")),
            compact_eam_prior_mix=float(getattr(args, "hrot_compact_eam_prior_mix", 0.5)),
            normalize_euclidean_tokens=_bool_flag(
                getattr(args, "hrot_normalize_euclidean_tokens", "true"),
                default=True,
            ),
            normalize_rho=_bool_flag(getattr(args, "hrot_normalize_rho", "false"), default=False),
            eval_use_float64=_bool_flag(getattr(args, "hrot_eval_use_float64", "true"), default=True),
            hyperbolic_backend=str(getattr(args, "hrot_hyperbolic_backend", "auto")),
            ot_backend=(
                "partial"
                if is_ours_final_partial_ot_model
                else str(getattr(args, "hrot_ot_backend", "native"))
            ),
            cost_margin_aux_weight=float(getattr(args, "hrot_cost_margin_aux_weight", 0.0)),
            cost_margin_aux_margin=float(getattr(args, "hrot_cost_margin_aux_margin", 0.02)),
            care_enable_fwec=_bool_flag(getattr(args, "care_enable_fwec", "true"), default=True),
            care_enable_qesm=_bool_flag(getattr(args, "care_enable_qesm", "true"), default=True),
            care_enable_mdr=_bool_flag(getattr(args, "care_enable_mdr", "true"), default=True),
            care_fwec_eps=float(getattr(args, "care_fwec_eps", 1e-6)),
            care_fwec_w_clamp_min=float(getattr(args, "care_fwec_w_clamp_min", 0.1)),
            care_fwec_w_clamp_max=float(getattr(args, "care_fwec_w_clamp_max", 10.0)),
            care_qesm_tau_e=float(getattr(args, "care_qesm_tau_e", 1.0)),
            care_mdr_margin=float(getattr(args, "care_mdr_margin", 0.05)),
            care_mdr_lambda=float(getattr(args, "care_mdr_lambda", 0.1)),
            use_cata=_auto_bool_flag(getattr(args, "hrot_use_cata", "auto"), default=False),
            cata_num_anchors=int(getattr(args, "hrot_cata_num_anchors", 8)),
            cata_num_heads=int(getattr(args, "hrot_cata_num_heads", 4)),
            cata_attn_dropout=float(getattr(args, "hrot_cata_attn_dropout", 0.0)),
            projector_use_mlp=_bool_flag(getattr(args, "hrot_projector_mlp", "false"), default=False),
            projector_mlp_hidden_dim=getattr(args, "hrot_projector_mlp_hidden_dim", None),
            projector_use_residual=_bool_flag(
                getattr(args, "hrot_projector_residual", "false"),
                default=False,
            ),
            eps=float(getattr(args, "hrot_eps", 1e-6)),
        )
    if args.model == "ec_mrot":
        ECMROT = _load_symbol("net.ec_mrot", "ECMROT")
        hidden_dim = fewshot_backbone_output_dim(fewshot_backbone)
        mass_response_grid = _first_attr(
            args,
            "ec_mrot_mass_response_grid",
            "ec_mrot_response_grid",
            "hrot_ecot_rho_bank",
            default="0.40,0.55,0.70,0.85,0.95",
        )
        base_budget = _first_attr(args, "ec_mrot_base_budget", "hrot_ecot_base_rho", default=0.70)
        budget_tau = _first_attr(args, "ec_mrot_budget_tau", "hrot_ecot_budget_tau", default=1.0)
        return ECMROT(
            in_channels=3,
            hidden_dim=hidden_dim,
            token_dim=int(getattr(args, "hrot_token_dim", getattr(args, "token_dim", 128)) or 128),
            use_raw_backbone_tokens=_bool_flag(
                getattr(args, "hrot_use_raw_backbone_tokens", "false"),
                default=False,
            ),
            backbone_name=fewshot_backbone,
            image_size=image_size,
            resnet12_drop_rate=0.0,
            resnet12_dropblock_size=5,
            mass_response_grid=mass_response_grid,
            base_budget=float(base_budget),
            response_mode=str(getattr(args, "ec_mrot_response_mode", "anchor_free_functional")),
            learn_response_grid=_bool_flag(
                getattr(args, "ec_mrot_learn_response_grid", "false"),
                default=False,
            ),
            budget_prior=str(getattr(args, "ec_mrot_budget_prior", "uniform")),
            budget_kl_reg=float(getattr(args, "ec_mrot_budget_kl_reg", 0.0)),
            budget_entropy_reg=float(getattr(args, "ec_mrot_budget_entropy_reg", 0.0)),
            budget_tau=float(budget_tau),
            homotopy_schedule=str(getattr(args, "ec_mrot_homotopy_schedule", "none")),
            grid_spacing_reg=float(getattr(args, "ec_mrot_grid_spacing_reg", 0.0)),
            response_strength_init=float(getattr(args, "ec_mrot_response_strength_init", 0.35)),
            response_strength_max=float(getattr(args, "ec_mrot_response_strength_max", 1.0)),
            response_temperature=float(getattr(args, "ec_mrot_response_temperature", 1.0)),
            competitive_center=_bool_flag(getattr(args, "ec_mrot_competitive_center", "true"), default=True),
            local_response_gain=float(getattr(args, "ec_mrot_local_response_gain", 2.0)),
            episode_prior_gain=float(getattr(args, "ec_mrot_episode_prior_gain", 0.10)),
            margin_temperature=float(getattr(args, "ec_mrot_margin_temperature", 1.0)),
            stability_temperature=float(getattr(args, "ec_mrot_stability_temperature", 1.0)),
            residual_gate_floor=float(getattr(args, "ec_mrot_residual_gate_floor", 0.15)),
            eam_hidden_dim=int(getattr(args, "hrot_eam_hidden_dim", 256)),
            curvature_init=float(getattr(args, "hrot_curvature_init", 1.0)),
            projection_scale=float(getattr(args, "hrot_projection_scale", 0.1)),
            token_temperature=float(getattr(args, "hrot_token_temperature", 0.1)),
            score_scale=float(getattr(args, "hrot_score_scale", 16.0)),
            tau_q=float(getattr(args, "hrot_tau_q", 0.5)),
            tau_c=float(getattr(args, "hrot_tau_c", 0.5)),
            sinkhorn_epsilon=float(getattr(args, "hrot_sinkhorn_epsilon", 0.1)),
            sinkhorn_iterations=int(
                _first_attr(args, "hrot_sinkhorn_iterations", "hrot_sinkhorn_iters", default=60)
            ),
            sinkhorn_tolerance=float(getattr(args, "hrot_sinkhorn_tolerance", 1e-5)),
            fixed_mass=float(getattr(args, "hrot_fixed_mass", 0.8)),
            egtw_tau=float(getattr(args, "hrot_egtw_tau", 1.0)),
            egtw_lambda=float(getattr(args, "hrot_egtw_lambda", 1.0)),
            egtw_eps=float(getattr(args, "hrot_egtw_eps", 1e-8)),
            egtw_attention_temperature=float(getattr(args, "hrot_egtw_attention_temperature", 0.2)),
            egtw_support_similarity_weight=float(getattr(args, "hrot_egtw_support_similarity_weight", 0.5)),
            egtw_uniform_mix=float(getattr(args, "hrot_egtw_uniform_mix", 0.25)),
            egtw_detach_masses=_bool_flag(getattr(args, "hrot_egtw_detach_masses", "true"), default=True),
            egtw_learn_tau=_bool_flag(getattr(args, "hrot_egtw_learn_tau", "false"), default=False),
            egtw_learn_lambda=_bool_flag(getattr(args, "hrot_egtw_learn_lambda", "false"), default=False),
            pre_transport_shot_pool=_bool_flag(
                getattr(args, "hrot_pre_transport_shot_pool", "false"),
                default=False,
            ),
            pre_transport_shot_pool_mode=str(getattr(args, "hrot_pre_transport_shot_pool_mode", "mean")),
            ecot_rho_bank=mass_response_grid,
            ecot_base_rho=float(base_budget),
            ecot_budget_tau=float(budget_tau),
            ecot_max_lambda=float(getattr(args, "hrot_ecot_max_lambda", 1.0)),
            ecot_lambda_init=float(getattr(args, "hrot_ecot_lambda_init", -8.0)),
            ecot_controller_hidden=int(getattr(args, "hrot_ecot_controller_hidden", 32)),
            ecot_uniform_budget_policy=_bool_flag(
                getattr(args, "hrot_ecot_uniform_budget_policy", "false"),
                default=False,
            ),
            ecot_enable_tau_shot=_bool_flag(
                getattr(args, "hrot_ecot_enable_tau_shot", "true"),
                default=True,
            ),
            ecot_tau_shot_min=float(getattr(args, "hrot_ecot_tau_shot_min", 0.5)),
            ecot_tau_shot_max=float(getattr(args, "hrot_ecot_tau_shot_max", 2.0)),
            ecot_enable_threshold_offset=_bool_flag(
                getattr(args, "hrot_ecot_enable_threshold_offset", "false"),
                default=False,
            ),
            ecot_m2_per_shot_threshold=_bool_flag(
                getattr(args, "hrot_ecot_m2_per_shot_threshold", "false"),
                default=False,
            ),
            ecot_m2_pst_hidden=int(getattr(args, "hrot_ecot_m2_pst_hidden", 32)),
            ecot_m2_ablate_threshold_mass=_bool_flag(
                getattr(args, "hrot_ecot_m2_ablate_threshold_mass", "false"),
                default=False,
            ),
            ecot_m2_cost_per_mass_score=_bool_flag(
                getattr(args, "hrot_ecot_m2_cost_per_mass_score", "false"),
                default=False,
            ),
            ecot_m2_cost_per_mass_alpha=float(getattr(args, "hrot_ecot_m2_cost_per_mass_alpha", 1.0)),
            ecot_m2_cost_per_mass_detach_mass=(
                True
                if getattr(args, "hrot_ecot_m2_cost_per_mass_detach_mass", None) is None
                else _bool_flag(getattr(args, "hrot_ecot_m2_cost_per_mass_detach_mass"), default=True)
            ),
            ecot_m2_mass_score_mode=str(getattr(args, "hrot_ecot_m2_mass_score_mode", "standard")),
            ecot_m2_consensus_mass_alpha=float(
                getattr(args, "hrot_ecot_m2_consensus_mass_alpha", 1.0)
            ),
            ecot_m2_mass_reward_beta=float(getattr(args, "hrot_ecot_m2_mass_reward_beta", 1.0)),
            ecot_m2_mass_reward_shot_scaling=str(
                getattr(args, "hrot_ecot_m2_mass_reward_shot_scaling", "none")
            ),
            ecot_m2_use_swts=_bool_flag(getattr(args, "hrot_ecot_m2_use_swts", "false"), default=False),
            ecot_m2_swts_temp=float(getattr(args, "hrot_ecot_m2_swts_temp", 1.0)),
            ecot_m2_use_aqm=_bool_flag(getattr(args, "hrot_ecot_m2_use_aqm", "false"), default=False),
            ecot_m2_tau_aqm=float(getattr(args, "hrot_ecot_m2_tau_aqm", 1.0)),
            use_mncr=_bool_flag(getattr(args, "hrot_use_mncr", "false"), default=False),
            mncr_temperature=float(getattr(args, "hrot_mncr_temperature", 0.5)),
            mncr_lam=float(getattr(args, "hrot_mncr_lam", 0.5)),
            ecot_identity_reg=float(getattr(args, "hrot_ecot_identity_reg", 1e-4)),
            ecot_policy_entropy_reg=float(getattr(args, "hrot_ecot_policy_entropy_reg", 1e-3)),
            ecot_consensus_tau_mode=str(getattr(args, "hrot_ecot_consensus_tau_mode", "fixed")),
            ecot_consensus_tau=float(getattr(args, "hrot_ecot_consensus_tau", 1.0)),
            hlm_min_mass=float(getattr(args, "hrot_hlm_min_mass", 0.1)),
            hlm_init_mass=float(getattr(args, "hrot_hlm_init_mass", 0.8)),
            hlm_budget_mode=str(getattr(args, "hrot_hlm_budget_mode", "cost")),
            hlm_token_mode=str(getattr(args, "hrot_hlm_token_mode", "cost")),
            hlm_token_tau=float(getattr(args, "hrot_hlm_token_tau", 0.25)),
            hlm_budget_hidden_dim=int(getattr(args, "hrot_hlm_budget_hidden_dim", 64)),
            hlm_token_hidden_dim=int(getattr(args, "hrot_hlm_token_hidden_dim", 64)),
            hlm_detach_cost_features=_bool_flag(
                getattr(args, "hrot_hlm_detach_cost_features", "true"),
                default=True,
            ),
            hlm_allow_mass_grad=_bool_flag(getattr(args, "hrot_hlm_allow_mass_grad", "true"), default=True),
            hlm_lambda_rho=float(getattr(args, "hrot_hlm_lambda_rho", 0.0)),
            hlm_lambda_eff=float(getattr(args, "hrot_hlm_lambda_eff", 0.0)),
            hlm_eff_margin=float(getattr(args, "hrot_hlm_eff_margin", 0.05)),
            hlm_lambda_shot_cov=float(getattr(args, "hrot_hlm_lambda_shot_cov", 0.0)),
            hlm_shot_entropy_floor=float(getattr(args, "hrot_hlm_shot_entropy_floor", 0.35)),
            hlm_return_diagnostics=_bool_flag(
                getattr(args, "hrot_hlm_return_diagnostics", "true"),
                default=True,
            ),
            min_mass=float(getattr(args, "hrot_min_mass", 0.1)),
            mass_bonus_init=float(getattr(args, "hrot_mass_bonus_init", 1.0)),
            transport_cost_threshold_init=getattr(args, "hrot_transport_cost_threshold_init", None),
            lambda_rho=float(getattr(args, "hrot_lambda_rho", 0.01)),
            rho_target=float(getattr(args, "hrot_rho_target", 0.8)),
            lambda_rho_rank=float(getattr(args, "hrot_lambda_rho_rank", 0.05)),
            rho_rank_margin=float(getattr(args, "hrot_rho_rank_margin", 0.05)),
            rho_rank_temperature=float(getattr(args, "hrot_rho_rank_temperature", 0.05)),
            lambda_curvature=float(getattr(args, "hrot_lambda_curvature", 0.0)),
            min_curvature=float(getattr(args, "hrot_min_curvature", 0.05)),
            structure_cost_init=float(getattr(args, "hrot_structure_cost_init", 0.05)),
            ground_cost=str(getattr(args, "hrot_ground_cost", "auto")),
            eam_mode=str(getattr(args, "hrot_eam_mode", "compact")),
            compact_eam_prior_mix=float(getattr(args, "hrot_compact_eam_prior_mix", 0.5)),
            normalize_euclidean_tokens=_bool_flag(
                getattr(args, "hrot_normalize_euclidean_tokens", "true"),
                default=True,
            ),
            normalize_rho=_bool_flag(getattr(args, "hrot_normalize_rho", "false"), default=False),
            eval_use_float64=_bool_flag(getattr(args, "hrot_eval_use_float64", "true"), default=True),
            hyperbolic_backend=str(getattr(args, "hrot_hyperbolic_backend", "auto")),
            ot_backend=str(getattr(args, "hrot_ot_backend", "native")),
            eps=float(getattr(args, "hrot_eps", 1e-6)),
        )
    if args.model == "fgwuot_fsl":
        FGWUOTFewShot = _load_symbol("net.fgwuot_fsl", "FGWUOTFewShot")
        hidden_dim = fewshot_backbone_output_dim(fewshot_backbone)
        return FGWUOTFewShot(
            in_channels=3,
            hidden_dim=hidden_dim,
            token_dim=int(getattr(args, "fgwuot_token_dim", 128)),
            backbone_name=fewshot_backbone,
            image_size=image_size,
            tau=float(getattr(args, "fgwuot_tau", 0.5)),
            eps_sinkhorn=float(getattr(args, "fgwuot_eps_sinkhorn", 0.1)),
            fgw_iters=int(getattr(args, "fgwuot_fgw_iters", 8)),
            sinkhorn_iters=int(getattr(args, "fgwuot_sinkhorn_iters", 60)),
            sinkhorn_tol=float(getattr(args, "fgwuot_sinkhorn_tol", 1e-5)),
            alpha_init=float(getattr(args, "fgwuot_alpha_init", 0.5)),
            score_scale_init=float(getattr(args, "fgwuot_score_scale_init", 16.0)),
            rho_head_hidden=int(getattr(args, "fgwuot_rho_head_hidden", 32)),
            lambda_rho=float(getattr(args, "fgwuot_lambda_rho", 0.01)),
            rho_target=float(getattr(args, "fgwuot_rho_target", 0.8)),
            normalize_tokens=_bool_flag(
                getattr(args, "fgwuot_normalize_tokens", "true"), default=True),
            shot_aggregation=str(getattr(args, "fgwuot_shot_aggregation", "j_logmeanexp")),
            eps=float(getattr(args, "fgwuot_eps", 1e-8)),
        )
    if args.model == "cfuget":
        CFUGETFewShot = _load_symbol("net.cfuget", "CFUGETFewShot")
        hidden_dim = fewshot_backbone_output_dim(fewshot_backbone)
        return CFUGETFewShot(
            in_channels=3,
            hidden_dim=hidden_dim,
            token_dim=int(getattr(args, "cfuget_token_dim", 128)),
            backbone_name=fewshot_backbone,
            image_size=image_size,
            rho=float(getattr(args, "cfuget_rho", 0.8)),
            tau=float(getattr(args, "cfuget_tau", 0.5)),
            eps_sinkhorn=float(getattr(args, "cfuget_eps_sinkhorn", 0.08)),
            fgw_iters=int(getattr(args, "cfuget_fgw_iters", 1)),
            sinkhorn_iters=int(getattr(args, "cfuget_sinkhorn_iters", 50)),
            sinkhorn_tol=float(getattr(args, "cfuget_sinkhorn_tol", 1e-5)),
            alpha_init=float(getattr(args, "cfuget_alpha_init", 1e-6)),
            score_scale_init=float(getattr(args, "cfuget_score_scale_init", 16.0)),
            threshold_init=float(getattr(args, "cfuget_threshold_init", 0.5)),
            rival_temperature=float(getattr(args, "cfuget_rival_temperature", 0.07)),
            rival_margin=float(getattr(args, "cfuget_rival_margin", 0.02)),
            coherent_temperature=float(getattr(args, "cfuget_coherent_temperature", 0.10)),
            spatial_structure_weight=float(getattr(args, "cfuget_spatial_structure_weight", 0.0)),
            mass_weight=float(getattr(args, "cfuget_mass_weight", 1.0)),
            cost_weight=float(getattr(args, "cfuget_cost_weight", 1.0)),
            normalize_tokens=_bool_flag(getattr(args, "cfuget_normalize_tokens", "true"), default=True),
            structure_detach=_bool_flag(getattr(args, "cfuget_structure_detach", "false"), default=False),
            rival_detach=_bool_flag(getattr(args, "cfuget_rival_detach", "true"), default=True),
            ablation_mode=str(getattr(args, "cfuget_ablation_mode", "full")),
            shot_aggregation=str(getattr(args, "cfuget_shot_aggregation", "j_logmeanexp")),
            eps=float(getattr(args, "cfuget_eps", 1e-8)),
        )
    if args.model == "jsc_wdro":
        JSCWDRO = _load_symbol("net.jsc_wdro", "JSCWDRO")
        hidden_dim = fewshot_backbone_output_dim(fewshot_backbone)
        return JSCWDRO(
            in_channels=3,
            hidden_dim=hidden_dim,
            token_dim=int(getattr(args, "jsc_wdro_token_dim", getattr(args, "token_dim", 128)) or 128),
            backbone_name=fewshot_backbone,
            image_size=image_size,
            resnet12_drop_rate=0.0,
            resnet12_dropblock_size=5,
            sinkhorn_epsilon=float(getattr(args, "jsc_wdro_sinkhorn_epsilon", 0.05)),
            sinkhorn_iterations=int(getattr(args, "jsc_wdro_sinkhorn_iterations", 80)),
            sinkhorn_tolerance=float(getattr(args, "jsc_wdro_sinkhorn_tolerance", 1e-5)),
            barycenter_iterations=int(getattr(args, "jsc_wdro_barycenter_iterations", 40)),
            barycenter_tolerance=float(getattr(args, "jsc_wdro_barycenter_tolerance", 1e-5)),
            barycenter_transport=str(getattr(args, "jsc_wdro_barycenter_transport", "unbalanced")),
            barycenter_tau=float(getattr(args, "jsc_wdro_barycenter_tau", 0.5)),
            barycenter_method=str(getattr(args, "jsc_wdro_barycenter_method", "sinkhorn")),
            tau_q=float(getattr(args, "jsc_wdro_tau_q", 0.5)),
            tau_c=float(getattr(args, "jsc_wdro_tau_c", 0.5)),
            query_transport=str(getattr(args, "jsc_wdro_query_transport", "balanced")),
            score_scale=float(getattr(args, "jsc_wdro_score_scale", 16.0)),
            epsilon_alpha_init=float(getattr(args, "jsc_wdro_epsilon_alpha_init", 0.05)),
            epsilon_beta_init=float(getattr(args, "jsc_wdro_epsilon_beta_init", 0.25)),
            epsilon_floor_init=float(getattr(args, "jsc_wdro_epsilon_floor_init", 1e-4)),
            learn_epsilon=_bool_flag(getattr(args, "jsc_wdro_learn_epsilon", "true"), default=True),
            epsilon_dimension=getattr(args, "jsc_wdro_epsilon_dimension", None),
            epsilon_reg_weight=float(getattr(args, "jsc_wdro_epsilon_reg_weight", 0.0)),
            normalize_tokens=_bool_flag(getattr(args, "jsc_wdro_normalize_tokens", "true"), default=True),
            cost_power=float(getattr(args, "jsc_wdro_cost_power", 2.0)),
            normalize_unbalanced_cost=_bool_flag(
                getattr(args, "jsc_wdro_normalize_unbalanced_cost", "true"),
                default=True,
            ),
            unbalanced_score_mode=str(getattr(args, "jsc_wdro_unbalanced_score_mode", "uot_objective")),
            token_weighting=str(getattr(args, "jsc_wdro_token_weighting", "uniform")),
            token_weight_temperature=float(getattr(args, "jsc_wdro_token_weight_temperature", 0.25)),
            use_competitive_diagnostics=_bool_flag(
                getattr(args, "jsc_wdro_use_competitive_diagnostics", "true"),
                default=True,
            ),
            competitive_temperature=float(getattr(args, "jsc_wdro_competitive_temperature", 0.1)),
            profile=_bool_flag(getattr(args, "jsc_wdro_profile", "false"), default=False),
            ot_backend=str(getattr(args, "jsc_wdro_ot_backend", "pot")),
            eps=float(getattr(args, "jsc_wdro_eps", 1e-8)),
        )
    if args.model in {"ubt_fsl", "cbcr_fsl"}:
        UBTFSL = _load_symbol("net.ubt_fsl", "UBTFSL")
        hidden_dim = fewshot_backbone_output_dim(fewshot_backbone)
        primary_prefix = "cbcr_fsl" if args.model == "cbcr_fsl" else "ubt_fsl"
        fallback_prefix = "ubt_fsl" if primary_prefix == "cbcr_fsl" else "cbcr_fsl"

        def _method_arg(suffix, default=None, *extra_names):
            return _first_attr(
                args,
                f"{primary_prefix}_{suffix}",
                f"{fallback_prefix}_{suffix}",
                *extra_names,
                default=default,
            )

        return UBTFSL(
            in_channels=3,
            hidden_dim=hidden_dim,
            token_dim=int(_method_arg("token_dim", 128, "token_dim") or 128),
            backbone_name=fewshot_backbone,
            image_size=image_size,
            resnet12_drop_rate=0.0,
            resnet12_dropblock_size=5,
            sinkhorn_epsilon=float(_method_arg("sinkhorn_epsilon", 0.05)),
            sinkhorn_iterations=int(_method_arg("sinkhorn_iterations", 80)),
            sinkhorn_tolerance=float(_method_arg("sinkhorn_tolerance", 1e-5)),
            barycenter_iterations=int(_method_arg("barycenter_iterations", 40)),
            barycenter_tolerance=float(_method_arg("barycenter_tolerance", 1e-5)),
            barycenter_method=str(_method_arg("barycenter_method", "mixture")),
            alpha=float(_method_arg("alpha", 0.3)),
            beta=float(_method_arg("beta", 0.1)),
            tau=float(_method_arg("query_competition_temperature", None) or _method_arg("tau", 0.5)),
            rho=float(_method_arg("rho", 1.0)),
            score_scale=float(_method_arg("score_scale", 8.0)),
            normalize_tokens=_bool_flag(_method_arg("normalize_tokens", "true"), default=True),
            cost_power=float(_method_arg("cost_power", 2.0)),
            profile=_bool_flag(_method_arg("profile", "false"), default=False),
            transport_backend=str(_method_arg("transport_backend", "unbalanced_ot")),
            scoring=str(_method_arg("scoring", "robust_transport_envelope")),
            use_query_competition=_bool_flag(
                _method_arg("use_query_competition", "true"),
                default=True,
            ),
            query_competition_temperature=float(
                _method_arg("query_competition_temperature", None) or _method_arg("tau", 0.5)
            ),
            ablation_mode=str(_method_arg("ablation_mode", "ubt_full")),
            ot_backend=str(_method_arg("solver_backend", None) or _method_arg("ot_backend", "native")),
            eps=float(_method_arg("eps", 1e-8)),
        )
    if args.model in {"warn", "pars_net"}:
        WassersteinRoutedAttentiveReconstructionNet = _load_symbol(
            "net.warn",
            "WassersteinRoutedAttentiveReconstructionNet",
        )
        return WassersteinRoutedAttentiveReconstructionNet(
            in_channels=3,
            hidden_dim=64,
            token_dim=int(getattr(args, "warn_token_dim", getattr(args, "token_dim", 128)) or 128),
            backbone_name=fewshot_backbone,
            image_size=image_size,
            resnet12_drop_rate=0.0,
            resnet12_dropblock_size=5,
            transformer_depth=int(getattr(args, "warn_transformer_depth", 1)),
            num_heads=int(getattr(args, "warn_num_heads", 4)),
            ffn_multiplier=int(getattr(args, "warn_ffn_multiplier", 4)),
            num_support_basis_tokens=int(getattr(args, "warn_num_basis_tokens", 8)),
            num_readout_basis_tokens=int(
                getattr(args, "warn_num_readout_basis_tokens", getattr(args, "warn_num_basis_tokens", 8))
            ),
            num_prior_subspaces=int(getattr(args, "warn_num_prior_subspaces", 16)),
            num_prior_atoms=int(getattr(args, "warn_num_prior_atoms", 8)),
            attn_dropout=float(getattr(args, "warn_attn_dropout", 0.0)),
            sw_num_projections=int(getattr(args, "warn_sw_num_projections", 64)),
            sw_p=float(getattr(args, "warn_sw_p", 2.0)),
            sw_normalize=_bool_flag(getattr(args, "warn_sw_normalize", "true"), default=True),
            prior_temperature=float(getattr(args, "warn_prior_temperature", 1.0)),
            score_scale=float(getattr(args, "warn_score_scale", 16.0)),
            diversity_weight=float(getattr(args, "warn_diversity_weight", 0.0)),
            recon_lambda_init=float(getattr(args, "warn_recon_lambda_init", 0.1)),
        )
    CosineNet = _load_symbol("net.cosine", "CosineNet")
    return CosineNet(image_size=image_size, device=device)
