"""Model registry and builders for pulse_fewshot benchmarks."""

from net.can import CrossAttentionNet
from net.adcmamba_sw import ADCMambaSWNet
from net.adchot import ADCHOTNet
from net.cosine import CosineNet
from net.covamnet import CovaMNet
from net.deepbdc import DeepBDC
from net.deepemd import DeepEMD
from net.dn4 import DN4
from net.feat import FEAT
from net.frn import FRN
from net.hierarchical_consensus_slot_mamba_net import HierarchicalConsensusSlotMambaNet
from net.hierarchical_episodic_ssm_net import HierarchicalEpisodicSSMNet
from net.maml import MAMLNet
from net.mann import MANNNet
from net.matchingnet import MatchingNet
from net.protonet import ProtoNet
from net.relationnet import RelationNet
from net.spif import SPIFCE, SPIFMAX
from net.warn import WassersteinRoutedAttentiveReconstructionNet


def _bool_flag(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).lower() == "true"


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
    "mann": {
        "display_name": "MANN",
        "architecture": "Conv64F embedding + external memory controller + least-used access write/read",
        "metric": "External-Memory Episodic Readout",
    },
    "adchot": {
        "display_name": "ADC-HOT",
        "architecture": "ResNet12 backbone pretraining + paper-style adaptive distribution calibration with hierarchical OT",
        "metric": "Calibrated Logistic Regression",
    },
    "adcmamba_sw": {
        "display_name": "ADC-Mamba-SW",
        "architecture": "Old ADC-HOT prior calibration + official Mamba token encoder + POT sliced Wasserstein prior trust",
        "metric": "Calibrated Gaussian + Sliced Wasserstein Distance",
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
    "spifce": {
        "display_name": "SPIFCE",
        "architecture": "Stable/variant token factorization -> stable gate -> global prototype + local top-r partial matching",
        "metric": "Cosine Prototype + Partial Token Matching",
    },
    "spifmax": {
        "display_name": "SPIFMAX",
        "architecture": "SPIF core + consistency/decorr/sparse regularization",
        "metric": "Cosine Prototype + Partial Token Matching",
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


def resolve_fewshot_backbone(args):
    backbone = str(getattr(args, "fewshot_backbone", "resnet12")).lower()
    if backbone == "default":
        return "resnet12"
    if backbone not in {"resnet12", "conv64f"}:
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

    if args.model == "protonet":
        return ProtoNet(image_size=image_size, device=device)
    if args.model == "covamnet":
        return CovaMNet(device=device)
    if args.model == "matchingnet":
        return MatchingNet(image_size=image_size, device=device)
    if args.model == "relationnet":
        return RelationNet(image_size=image_size, device=device)
    if args.model == "dn4":
        return DN4(k_neighbors=3, device=device)
    if args.model == "feat":
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
        return DeepEMD(
            image_size=image_size,
            temperature=12.5,
            solver=getattr(args, "deepemd_solver", "opencv"),
            qpth_form=getattr(args, "deepemd_qpth_form", "L2"),
            qpth_l2_strength=float(getattr(args, "deepemd_qpth_l2_strength", 1e-6)),
            sfc_lr=float(getattr(args, "deepemd_sfc_lr", 0.1)),
            sfc_update_step=int(getattr(args, "deepemd_sfc_update_step", 100)),
            sfc_bs=int(getattr(args, "deepemd_sfc_bs", 4)),
            fewshot_backbone=fewshot_backbone,
            device=device,
        )
    if args.model == "can":
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
        return FRN(image_size=image_size, fewshot_backbone=fewshot_backbone, device=device)
    if args.model == "deepbdc":
        reduce_dim = 640 if fewshot_backbone == "resnet12" else 64
        return DeepBDC(
            image_size=image_size,
            reduce_dim=reduce_dim,
            fewshot_backbone=fewshot_backbone,
            device=device,
        )
    if args.model == "maml":
        return MAMLNet(
            image_size=image_size,
            inner_lr=0.01,
            inner_steps=5,
            first_order=not getattr(args, "maml_second_order", False),
            max_way_num=32,
            device=device,
        )
    if args.model == "mann":
        return MANNNet(
            image_size=image_size,
            way_num=getattr(args, "way_num", 4),
            cell_size=200,
            memory_slots=128,
            memory_dim=40,
            num_reads=4,
            gamma=0.95,
            device=device,
        )
    if args.model == "adchot":
        return ADCHOTNet(
            image_size=image_size,
            way_num=getattr(args, "way_num", 4),
            beta=0.8,
            alpha=0.21,
            lambd=0.3,
            fewshot_backbone=fewshot_backbone,
            device=device,
        )
    if args.model == "adcmamba_sw":
        return ADCMambaSWNet(
            image_size=image_size,
            way_num=getattr(args, "way_num", 4),
            token_dim=getattr(args, "token_dim", 128) or 128,
            state_dim=getattr(args, "ssm_state_dim", 16),
            mamba_depth=getattr(args, "ssm_depth", 2),
            alpha=0.21,
            lambd=0.3,
            ema=0.05,
            sw_num_projections=getattr(args, "sw_num_projections", 64),
            sw_weight=getattr(args, "sw_weight", 1.0),
            transport_temperature=getattr(args, "transport_temperature", 6.0),
            fewshot_backbone=fewshot_backbone,
            device=device,
        )
    if args.model == "hierarchical_episodic_ssm_net":
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
    if args.model in {"spifce", "spifmax"}:
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
    if args.model in {"warn", "pars_net"}:
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
    return CosineNet(image_size=image_size, device=device)
