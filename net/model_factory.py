"""Model registry and builders for pulse_fewshot benchmarks."""

from importlib import import_module


def _bool_flag(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).lower() == "true"


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
        "display_name": "HROT-FSL",
        "architecture": "Backbone spatial tokens -> Euclidean projector -> Poincare-ball embedding -> balanced/unbalanced relational transport with episode-adaptive, token-reliable mass",
        "metric": "Hyperbolic Relational Optimal Transport",
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


def get_model_choices():
    return list(MODEL_REGISTRY.keys())


def get_model_metadata(model_name):
    if model_name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model: {model_name}")
    return MODEL_REGISTRY[model_name]


PAPER_BASELINE_BACKBONE_MODELS = frozenset({"feat", "deepemd", "can", "frn", "deepbdc"})
HIGH_DIM_FEWSHOT_BACKBONES = frozenset({"resnet12", "fsl_mamba"})


def model_supports_fewshot_backbone(model_name, backbone_name):
    model_name = str(model_name)
    backbone_name = str(backbone_name).lower()
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
        raise ValueError(
            f"fewshot_backbone={fewshot_backbone} is only supported for SPIF/new repository models, "
            f"not the paper baseline `{args.model}`"
        )

    if args.model == "protonet":
        ProtoNet = _load_symbol("net.protonet", "ProtoNet")
        return ProtoNet(image_size=image_size, device=device)
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
            sfc_lr=float(getattr(args, "deepemd_sfc_lr", 0.1)),
            sfc_update_step=int(getattr(args, "deepemd_sfc_update_step", 15)),
            sfc_bs=int(getattr(args, "deepemd_sfc_bs", 4)),
            fewshot_backbone=fewshot_backbone,
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
    if args.model == "hrot_fsl":
        HROTFSL = _load_symbol("net.hrot_fsl", "HROTFSL")
        hidden_dim = fewshot_backbone_output_dim(fewshot_backbone)
        return HROTFSL(
            in_channels=3,
            hidden_dim=hidden_dim,
            token_dim=int(getattr(args, "hrot_token_dim", getattr(args, "token_dim", 128)) or 128),
            backbone_name=fewshot_backbone,
            image_size=image_size,
            resnet12_drop_rate=0.0,
            resnet12_dropblock_size=5,
            variant=str(getattr(args, "hrot_variant", "E")),
            eam_hidden_dim=int(getattr(args, "hrot_eam_hidden_dim", 256)),
            curvature_init=float(getattr(args, "hrot_curvature_init", 1.0)),
            projection_scale=float(getattr(args, "hrot_projection_scale", 0.1)),
            token_temperature=float(getattr(args, "hrot_token_temperature", 0.1)),
            score_scale=float(getattr(args, "hrot_score_scale", 16.0)),
            tau_q=float(getattr(args, "hrot_tau_q", 0.5)),
            tau_c=float(getattr(args, "hrot_tau_c", 0.5)),
            sinkhorn_epsilon=float(getattr(args, "hrot_sinkhorn_epsilon", 0.1)),
            sinkhorn_iterations=int(getattr(args, "hrot_sinkhorn_iterations", 60)),
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
