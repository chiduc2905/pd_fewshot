"""Few-shot benchmark training and evaluation entrypoint."""

import csv
import os
import argparse
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
import wandb

try:
    from thop import profile, clever_format
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False
    print("Warning: thop not installed. Run 'pip install thop' for FLOPs calculation.")

from dataset import load_dataset
from dataloader.dataloader import FewshotDataset
from function.function import (
    RelationLoss,
    plot_confusion_matrix,
    plot_tsne,
    plot_training_curves,
    seed_func,
)
from net.model_factory import build_model_from_args, get_model_choices, get_model_metadata, resolve_fewshot_backbone


def _bool_flag(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).lower() == "true"


def resolve_runtime_device(args):
    if not torch.cuda.is_available():
        return "cpu"
    gpu_id = int(getattr(args, "gpu_id", 0))
    if gpu_id < 0:
        return "cpu"
    device_count = torch.cuda.device_count()
    if gpu_id >= device_count:
        raise ValueError(f"gpu_id={gpu_id} out of range. Available CUDA devices: 0..{device_count - 1}")
    return f"cuda:{gpu_id}"


def get_args():
    parser = argparse.ArgumentParser(description="PD Scalogram Few-shot Benchmark")

    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/mnt/disk2/nhatnc/res/scalogram_fewshot/proposed_model/smnet/scalogram_27_1",
    )
    parser.add_argument("--path_weights", type=str, default="checkpoints/")
    parser.add_argument("--path_results", type=str, default="results/")
    parser.add_argument("--weights", type=str, default=None, help="Checkpoint for testing or weight initialization")
    parser.add_argument("--resume", type=str, default=None, help="Resumable training checkpoint (*.ckpt)")
    parser.add_argument("--dataset_name", type=str, default="knee_aug_split")

    parser.add_argument("--model", type=str, default="covamnet", choices=get_model_choices())
    parser.add_argument(
        "--maml_second_order",
        dest="maml_second_order",
        action="store_true",
        help="Use paper-style second-order MAML gradients (default).",
    )
    parser.add_argument(
        "--maml_first_order",
        dest="maml_second_order",
        action="store_false",
        help="Use first-order MAML approximation for faster benchmarking.",
    )
    parser.set_defaults(maml_second_order=True)
    parser.add_argument("--fewshot_backbone", type=str, default="resnet12", choices=["default", "resnet12", "conv64f"])

    parser.add_argument("--way_num", type=int, default=4)
    parser.add_argument("--shot_num", type=int, default=1)
    parser.add_argument("--query_num", type=int, default=None, help="Legacy: same queries for all splits")
    parser.add_argument("--query_num_train", type=int, default=1)
    parser.add_argument("--query_num_val", type=int, default=1)
    parser.add_argument("--query_num_test", type=int, default=1)
    parser.add_argument(
        "--selected_classes",
        type=str,
        default=None,
        help='Comma-separated class indices to use (e.g. "0,1"). If None, use all classes.',
    )
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--token_dim", type=int, default=128)
    parser.add_argument("--ssm_state_dim", type=int, default=16)
    parser.add_argument("--ssm_depth", type=int, default=2)
    parser.add_argument("--sw_num_projections", type=int, default=64)
    parser.add_argument("--sw_weight", type=float, default=1.0)
    parser.add_argument("--transport_temperature", type=float, default=6.0)
    parser.add_argument("--hierarchical_token_dim", type=int, default=64)
    parser.add_argument("--hierarchical_token_depth", type=int, default=1)
    parser.add_argument("--hierarchical_shot_depth", type=int, default=1)
    parser.add_argument("--hierarchical_temperature", type=float, default=16.0)
    parser.add_argument("--hierarchical_use_sw", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--hierarchical_sw_weight", type=float, default=0.25)
    parser.add_argument("--hierarchical_sw_num_projections", type=int, default=64)
    parser.add_argument("--hierarchical_sw_p", type=float, default=2.0)
    parser.add_argument("--hierarchical_sw_normalize", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--hierarchical_token_merge_mode", type=str, default="concat", choices=["concat", "mean"])
    parser.add_argument("--hcsm_token_dim", type=int, default=128)
    parser.add_argument("--hcsm_temperature", type=float, default=16.0)
    parser.add_argument("--hcsm_intra_image_depth", type=int, default=1)
    parser.add_argument("--hcsm_num_write_tokens", type=int, default=4)
    parser.add_argument("--hcsm_num_transport_tokens", type=int, default=12)
    parser.add_argument("--hcsm_num_consensus_slots", type=int, default=4)
    parser.add_argument("--hcsm_pool_heads", type=int, default=4)
    parser.add_argument("--hcsm_use_sw", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--hcsm_sw_weight", type=float, default=0.25)
    parser.add_argument("--hcsm_sw_num_projections", type=int, default=64)
    parser.add_argument("--hcsm_sw_p", type=float, default=2.0)
    parser.add_argument("--hcsm_sw_normalize", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--hcsm_slot_mamba_depth", type=int, default=1)
    parser.add_argument("--hcsm_mamba_d_conv", type=int, default=4)
    parser.add_argument("--hcsm_mamba_expand", type=int, default=2)
    parser.add_argument("--hcsm_mamba_dropout", type=float, default=0.0)
    parser.add_argument("--hcsm_sw_merged_weight", type=float, default=1.0)
    parser.add_argument("--hcsm_sw_shot_weight", type=float, default=1.0)
    parser.add_argument("--hcsm_use_shot_refiner", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--hcsm_use_reliability_residual", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--hcsm_use_token_adapter", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--hcsm_use_local_gate", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--spif_stable_dim", type=int, default=64)
    parser.add_argument("--spif_variant_dim", type=int, default=64)
    parser.add_argument("--spif_gate_hidden", type=int, default=16)
    parser.add_argument("--spif_top_r", type=int, default=3)
    parser.add_argument("--spif_alpha_init", type=float, default=0.7)
    parser.add_argument("--spif_gate_on", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--spif_factorization_on", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--spif_global_only", type=str, default="false", choices=["true", "false"])
    parser.add_argument("--spif_local_only", type=str, default="false", choices=["true", "false"])
    parser.add_argument("--spif_learnable_alpha", type=str, default=None, choices=["true", "false"])
    parser.add_argument("--spif_token_l2norm", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--spif_consistency_weight", type=float, default=0.1)
    parser.add_argument("--spif_decorr_weight", type=float, default=0.01)
    parser.add_argument("--spif_sparse_weight", type=float, default=0.001)
    parser.add_argument("--spif_consistency_dropout", type=float, default=0.1)
    parser.add_argument("--warn_token_dim", type=int, default=128)
    parser.add_argument("--warn_transformer_depth", type=int, default=1)
    parser.add_argument("--warn_num_heads", type=int, default=4)
    parser.add_argument("--warn_ffn_multiplier", type=int, default=4)
    parser.add_argument("--warn_num_basis_tokens", type=int, default=8)
    parser.add_argument("--warn_num_readout_basis_tokens", type=int, default=8)
    parser.add_argument("--warn_num_prior_subspaces", type=int, default=16)
    parser.add_argument("--warn_num_prior_atoms", type=int, default=8)
    parser.add_argument("--warn_attn_dropout", type=float, default=0.0)
    parser.add_argument("--warn_sw_num_projections", type=int, default=64)
    parser.add_argument("--warn_sw_p", type=float, default=2.0)
    parser.add_argument("--warn_sw_normalize", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--warn_prior_temperature", type=float, default=1.0)
    parser.add_argument("--warn_score_scale", type=float, default=16.0)
    parser.add_argument("--warn_diversity_weight", type=float, default=0.0)
    parser.add_argument("--warn_recon_lambda_init", type=float, default=0.1)

    parser.add_argument(
        "--training_samples",
        type=int,
        default=None,
        help="Total training samples (must be divisible by way_num)",
    )
    parser.add_argument("--episode_num_train", type=int, default=200)
    parser.add_argument("--episode_num_val", type=int, default=300)
    parser.add_argument("--episode_num_test", type=int, default=300)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", type=str, default="false", choices=["true", "false"])
    parser.add_argument("--persistent_workers", type=str, default="false", choices=["true", "false"])
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--cudnn_deterministic", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--cudnn_benchmark", type=str, default="false", choices=["true", "false"])
    parser.add_argument("--gpu_id", type=int, default=0, help="CUDA device id (negative to force CPU)")
    parser.add_argument("--lr", type=float, default=5e-4, help="Base learning rate")
    parser.add_argument("--step_size", type=int, default=10, help="StepLR decay interval in epochs")
    parser.add_argument("--gamma", type=float, default=0.5, help="StepLR multiplicative decay factor")
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--train_augment", type=str, default="false", choices=["true", "false"])
    parser.add_argument("--time_shift_max", type=int, default=4)
    parser.add_argument("--time_shift_prob", type=float, default=0.5)
    parser.add_argument("--amp_scale_min", type=float, default=0.9)
    parser.add_argument("--amp_scale_max", type=float, default=1.1)
    parser.add_argument("--amp_scale_prob", type=float, default=0.5)
    parser.add_argument("--time_mask_width", type=int, default=4)
    parser.add_argument("--time_mask_prob", type=float, default=0.25)
    parser.add_argument("--freq_mask_width", type=int, default=4)
    parser.add_argument("--freq_mask_prob", type=float, default=0.25)
    parser.add_argument("--grad_clip", type=float, default=2.0)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--save_misclf_report", action="store_true")
    parser.add_argument("--misclf_topk", type=int, default=30)
    parser.add_argument("--save_last_checkpoint", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--skip_final_test", type=str, default="false", choices=["true", "false"])
    parser.add_argument("--deepemd_train_sfc", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--deepemd_fast_val", type=str, default="false", choices=["true", "false"])
    parser.add_argument("--deepemd_test_exact", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--deepemd_test_sfc", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--deepemd_solver", type=str, default="opencv", choices=["opencv", "qpth", "linprog", "sinkhorn"])
    parser.add_argument("--deepemd_qpth_form", type=str, default="L2", choices=["QP", "L2"])
    parser.add_argument("--deepemd_qpth_l2_strength", type=float, default=1e-6)
    parser.add_argument("--deepemd_sfc_lr", type=float, default=0.1)
    parser.add_argument("--deepemd_sfc_update_step", type=int, default=100)
    parser.add_argument("--deepemd_sfc_bs", type=int, default=4)

    parser.add_argument("--project", type=str, default="pulse_fewshot")

    return parser.parse_args()


def get_model(args):
    args.fewshot_backbone = resolve_fewshot_backbone(args)
    model = build_model_from_args(args)
    # Keep model placement centralized because not every benchmark module
    # moves itself to args.device inside its constructor.
    model = model.to(torch.device(args.device))
    meta = get_model_metadata(args.model)
    print("\nModel Config:")
    print(f"  model: {meta['display_name']}")
    print(f"  architecture: {meta['architecture']}")
    if args.model == "maml":
        print(f"  maml: inner_lr=0.01, inner_steps=5, second_order={args.maml_second_order}")
    if args.model == "deepemd":
        print(
            "  deepemd: official-style EMD/SFC "
            f"(solver={getattr(args, 'deepemd_solver', 'opencv')}, "
            f"qpth_form={getattr(args, 'deepemd_qpth_form', 'L2')}, "
            f"sfc_lr={getattr(args, 'deepemd_sfc_lr', 0.1)}, "
            f"sfc_steps={getattr(args, 'deepemd_sfc_update_step', 100)}, "
            f"sfc_bs={getattr(args, 'deepemd_sfc_bs', 4)})"
        )
    if args.model == "adchot":
        print("  adchot: supervised base-class pretraining + fitted base statistics + OT calibration")
    if args.model == "adcmamba_sw":
        print(
            "  adcmamba_sw: official mamba_ssm token encoder + POT sliced Wasserstein distance "
            f"(token_dim={getattr(args, 'token_dim', 128)}, "
            f"d_state={getattr(args, 'ssm_state_dim', 16)}, "
            f"depth={getattr(args, 'ssm_depth', 2)}, "
            f"sw_proj={getattr(args, 'sw_num_projections', 64)}, "
            f"sw_weight={getattr(args, 'sw_weight', 1.0)}, "
            f"transport_temperature={getattr(args, 'transport_temperature', 6.0)})"
        )
    if args.model == "hierarchical_episodic_ssm_net":
        print(
            "  hierarchical_episodic_ssm_net: token-level SSM -> shot-level memory SSM -> "
            "hierarchical matcher + SW "
            f"(token_dim={getattr(args, 'hierarchical_token_dim', 64)}, "
            f"d_state={getattr(args, 'ssm_state_dim', 16)}, "
            f"token_depth={getattr(args, 'hierarchical_token_depth', 1)}, "
            f"shot_depth={getattr(args, 'hierarchical_shot_depth', 1)}, "
            f"sw={getattr(args, 'hierarchical_use_sw', 'true')}, "
            f"sw_weight={getattr(args, 'hierarchical_sw_weight', 0.25)})"
        )
    if args.model == "hierarchical_consensus_slot_mamba_net":
        print(
            "  hierarchical_consensus_slot_mamba_net: intra-image Mamba -> set memory pooling -> "
            "reliability-coupled matcher + SW "
            f"(token_dim={getattr(args, 'hcsm_token_dim', 128)}, "
            f"d_state={getattr(args, 'ssm_state_dim', 16)}, "
            f"intra_depth={getattr(args, 'hcsm_intra_image_depth', 1)}, "
            f"slots={getattr(args, 'hcsm_num_consensus_slots', 4)}, "
            f"sw={getattr(args, 'hcsm_use_sw', 'true')}, "
            f"sw_weight={getattr(args, 'hcsm_sw_weight', 0.25)})"
        )
    if args.model in {"spifce", "spifmax"}:
        print(
            "  spif: stable/variant token factorization -> stable evidence gate -> "
            "global prototypes + local top-r partial matching "
            f"(stable_dim={getattr(args, 'spif_stable_dim', 64)}, "
            f"variant_dim={getattr(args, 'spif_variant_dim', 64)}, "
            f"top_r={getattr(args, 'spif_top_r', 3)}, "
            f"gate={getattr(args, 'spif_gate_on', 'true')}, "
            f"factorization={getattr(args, 'spif_factorization_on', 'true')}, "
            f"global_only={getattr(args, 'spif_global_only', 'false')}, "
            f"local_only={getattr(args, 'spif_local_only', 'false')})"
        )
    if args.model in {"warn", "pars_net"}:
        print(
            "  pars_net: Transformer tokens -> support basis distillation -> SW prior retrieval -> "
            "prior-completed reconstruction subspace "
            f"(token_dim={getattr(args, 'warn_token_dim', 128)}, "
            f"depth={getattr(args, 'warn_transformer_depth', 1)}, "
            f"basis={getattr(args, 'warn_num_basis_tokens', 8)}, "
            f"prior_slots={getattr(args, 'warn_num_prior_subspaces', 16)}, "
            f"sw_proj={getattr(args, 'warn_sw_num_projections', 64)}, "
            f"prior_temp={getattr(args, 'warn_prior_temperature', 1.0)})"
        )
    return model


def build_train_augment_cfg(args):
    return {
        "time_shift_max": args.time_shift_max,
        "time_shift_prob": args.time_shift_prob,
        "amp_scale_min": args.amp_scale_min,
        "amp_scale_max": args.amp_scale_max,
        "amp_scale_prob": args.amp_scale_prob,
        "time_mask_width": args.time_mask_width,
        "time_mask_prob": args.time_mask_prob,
        "freq_mask_width": args.freq_mask_width,
        "freq_mask_prob": args.freq_mask_prob,
    }


def _pin_memory_enabled(args) -> bool:
    return _bool_flag(getattr(args, "pin_memory", "false"), default=False) and str(getattr(args, "device", "")).startswith(
        "cuda"
    )


def _persistent_workers_enabled(args) -> bool:
    return max(0, int(getattr(args, "num_workers", 0))) > 0 and _bool_flag(
        getattr(args, "persistent_workers", "false"),
        default=False,
    )


def _build_loader_kwargs(args, batch_size, shuffle, generator=None, worker_seed=None):
    num_workers = max(0, int(getattr(args, "num_workers", 0)))
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": _pin_memory_enabled(args),
    }
    if generator is not None:
        kwargs["generator"] = generator
    if num_workers > 0:
        kwargs["persistent_workers"] = _persistent_workers_enabled(args)
        prefetch_factor = int(getattr(args, "prefetch_factor", 2))
        if prefetch_factor > 0:
            kwargs["prefetch_factor"] = prefetch_factor
        if worker_seed is not None:
            kwargs["worker_init_fn"] = lambda worker_id: seed_func(worker_seed + worker_id)
    return kwargs


def configure_cudnn_runtime(args):
    # Keep runtime deterministic by default for reproducible few-shot benchmarks.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def unwrap_model_state(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model_state", "state_dict", "params"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
    return checkpoint


def strip_module_prefix(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    if not any(key.startswith("module.") for key in state_dict):
        return state_dict
    return {
        key[len("module.") :] if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }


def load_model_weights(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = strip_module_prefix(unwrap_model_state(checkpoint))
    model.load_state_dict(state_dict)
    return checkpoint


def get_checkpoint_stem(args):
    samples_suffix = f"{args.training_samples}samples" if args.training_samples else "all"
    return os.path.join(
        args.path_weights,
        f"{args.dataset_name}_{args.model}_{samples_suffix}_{args.shot_num}shot",
    )


def get_best_model_path(args):
    return f"{get_checkpoint_stem(args)}_final.pth"


def get_last_checkpoint_path(args):
    return f"{get_checkpoint_stem(args)}_last.ckpt"


def forward_scores(net, query, support, args, train=False, phase=None, query_targets=None, support_targets=None):
    phase = phase or ("train" if train else "eval")
    if args.model == "maml":
        smoothing = args.label_smoothing if train else 0.0
        return net(query, support, label_smoothing=smoothing)
    if args.model == "deepemd":
        if phase == "train":
            refine_proto = _bool_flag(args.deepemd_train_sfc, default=True)
            return net(query, support, refine_proto=refine_proto, exact=False)
        if phase == "val":
            fast_val = _bool_flag(args.deepemd_fast_val, default=False)
            return net(query, support, refine_proto=not fast_val, exact=not fast_val)
        if phase == "test":
            exact = _bool_flag(args.deepemd_test_exact, default=True)
            refine_proto = _bool_flag(args.deepemd_test_sfc, default=True)
            return net(query, support, refine_proto=refine_proto, exact=exact)
        return net(query, support)
    if args.model in {"feat", "can"}:
        return net(query, support, query_targets=query_targets, support_targets=support_targets)
    return net(query, support)


def compute_loss(scores, targets, args, relation_loss_fn, train=False):
    if isinstance(scores, dict):
        if "loss" in scores:
            return scores["loss"]
        logits = scores["logits"]
        base_loss = compute_loss(logits, targets, args, relation_loss_fn, train=train)
        aux_loss = scores.get("aux_loss")
        return base_loss if aux_loss is None else base_loss + aux_loss

    if args.model == "relationnet":
        return relation_loss_fn(scores, targets)
    smoothing = args.label_smoothing if train else 0.0
    return F.cross_entropy(scores, targets, label_smoothing=smoothing)


def extract_logits(scores):
    if isinstance(scores, dict):
        return scores["logits"]
    return scores


def _dataset_conditioned_ready(net):
    ready = getattr(net, "base_stats_ready", None)
    if ready is None:
        return False
    if isinstance(ready, torch.Tensor):
        return bool(ready.item())
    return bool(ready)


def prepare_dataset_conditioned_model(net, train_X, train_y, args, force=False):
    if not hasattr(net, "fit_base_statistics"):
        return
    if not force and _dataset_conditioned_ready(net):
        return
    print("Fitting ADC-HOT base statistics from the training split ...")
    fit_batch_size = max(16, min(64, len(train_X)))
    net.fit_base_statistics(train_X, train_y, batch_size=fit_batch_size)


def train_loop(net, train_X, train_y, val_X, val_y, args):
    device = torch.device(args.device)
    relation_loss_fn = RelationLoss().to(device)

    optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.StepLR(
        optimizer,
        step_size=args.step_size,
        gamma=args.gamma,
    )

    train_augment = _bool_flag(args.train_augment, default=False)
    augment_cfg = build_train_augment_cfg(args)
    history = {
        "train_acc": [],
        "val_acc": [],
        "train_loss": [],
        "val_loss": [],
    }
    best_acc = 0.0
    start_epoch = 1

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=args.device)
        state_dict = strip_module_prefix(unwrap_model_state(checkpoint))
        net.load_state_dict(state_dict)

        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            optimizer_state = checkpoint.get("optimizer_state")
            scheduler_state = checkpoint.get("scheduler_state")
            if optimizer_state is not None:
                optimizer.load_state_dict(optimizer_state)
            if scheduler is not None and scheduler_state is not None:
                scheduler.load_state_dict(scheduler_state)
            history = checkpoint.get("history", history)
            best_acc = float(checkpoint.get("best_acc", 0.0))
            start_epoch = int(checkpoint.get("epoch", 0)) + 1
            print(f"Resuming training from {args.resume} at epoch {start_epoch}/{args.num_epochs}")
        else:
            print(f"Loaded weights from {args.resume}; checkpoint has no optimizer state, restarting from epoch 1")
    elif args.weights:
        load_model_weights(net, args.weights, args.device)
        print(f"Initialized model weights from {args.weights}")

    val_seed = args.seed + 1

    print(
        f"Training protocol: scheduler=StepLR(step_size={args.step_size}, gamma={args.gamma}), "
        f"lr={args.lr:.2e}, label_smoothing={args.label_smoothing:.3f}, "
        f"train_augment={train_augment}, train_seed=args.seed+epoch, val_seed={val_seed}, "
        f"num_workers={args.num_workers}, pin_memory={_pin_memory_enabled(args)}, "
        f"persistent_workers={_persistent_workers_enabled(args)}, "
        f"cudnn(deterministic={torch.backends.cudnn.deterministic}, benchmark={torch.backends.cudnn.benchmark})"
    )
    if train_augment:
        print(f"Augmentation config: {augment_cfg}")

    if start_epoch > args.num_epochs:
        print(f"Checkpoint already reached epoch {start_epoch - 1}; skipping training loop.")
        return best_acc, history

    for epoch in range(start_epoch, args.num_epochs + 1):
        train_seed = args.seed + epoch
        train_ds = FewshotDataset(
            train_X,
            train_y,
            args.episode_num_train,
            args.way_num,
            args.shot_num,
            args.query_num_train,
            seed=train_seed,
            augment=train_augment,
            augment_cfg=augment_cfg,
        )
        train_gen = torch.Generator()
        train_gen.manual_seed(train_seed)
        train_loader = DataLoader(
            train_ds,
            **_build_loader_kwargs(
                args,
                batch_size=args.batch_size,
                shuffle=True,
                generator=train_gen,
                worker_seed=train_seed,
            ),
        )

        net.train()
        total_loss = 0.0
        train_correct = 0
        train_total = 0
        non_blocking = _pin_memory_enabled(args) and device.type == "cuda"

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.num_epochs}")
        for query, q_labels, support, support_labels in pbar:
            optimizer.zero_grad()
            current_lr = optimizer.param_groups[0]["lr"]

            batch_size = query.shape[0]
            channels, height, width = query.shape[2], query.shape[3], query.shape[4]
            support = support.view(batch_size, args.way_num, args.shot_num, channels, height, width).to(
                device,
                non_blocking=non_blocking,
            )
            query = query.to(device, non_blocking=non_blocking)
            targets = q_labels.view(-1).to(device)
            support_targets = support_labels.view(batch_size, args.way_num, args.shot_num).to(device)

            scores = forward_scores(
                net,
                query,
                support,
                args,
                train=True,
                phase="train",
                query_targets=targets,
                support_targets=support_targets,
            )
            loss = compute_loss(scores, targets, args, relation_loss_fn, train=True)
            logits = extract_logits(scores)

            with torch.no_grad():
                preds = logits.argmax(dim=1)
                train_correct += (preds == targets).sum().item()
                train_total += targets.size(0)

            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{current_lr:.2e}")

        scheduler.step()

        train_acc = train_correct / train_total if train_total > 0 else 0.0

        val_ds = FewshotDataset(
            val_X,
            val_y,
            args.episode_num_val,
            args.way_num,
            args.shot_num,
            args.query_num_val,
            seed=val_seed,
        )
        val_loader = DataLoader(
            val_ds,
            **_build_loader_kwargs(
                args,
                batch_size=1,
                shuffle=False,
                worker_seed=val_seed,
            ),
        )

        val_acc, val_loss = evaluate(net, val_loader, args)
        avg_loss = total_loss / len(train_loader)

        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["train_loss"].append(avg_loss)
        history["val_loss"].append(val_loss if val_loss else 0.0)

        train_val_gap = train_acc - val_acc
        print(
            f"Epoch {epoch}: Loss={avg_loss:.4f}, Train={train_acc:.4f}, "
            f"Val={val_acc:.4f} (gap={train_val_gap:+.4f})"
        )

        wandb.log(
            {
                "epoch": epoch,
                "loss/train": avg_loss,
                "loss/val": val_loss,
                "accuracy/train": train_acc,
                "accuracy/val": val_acc,
                "train_val_gap": train_val_gap,
                "lr": current_lr,
            }
        )

        if val_acc > best_acc:
            best_acc = val_acc
            print(f"  -> New best val: {val_acc:.4f}")
            wandb.run.summary["best_val_acc"] = best_acc

            final_path = get_best_model_path(args)
            torch.save(net.state_dict(), final_path)
            print(f"  Saved best model to {final_path}")

        if _bool_flag(args.save_last_checkpoint, default=True):
            last_path = get_last_checkpoint_path(args)
            checkpoint = {
                "epoch": epoch,
                "model_state": net.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_acc": best_acc,
                "history": history,
                "args": vars(args),
            }
            torch.save(checkpoint, last_path)

    samples_str = f"{args.training_samples}samples" if args.training_samples else "allsamples"
    curves_path = os.path.join(
        args.path_results,
        f"training_{args.dataset_name}_{args.model}_{samples_str}_{args.shot_num}shot",
    )
    try:
        plot_training_curves(history, curves_path)
        if os.path.exists(f"{curves_path}_curves.png"):
            wandb.log({"training_curves": wandb.Image(f"{curves_path}_curves.png")})
    except RuntimeError as exc:
        print(f"Skipping training curves: {exc}")

    print(f"Best Validation Accuracy: {best_acc:.4f}")
    return best_acc, history


def train_adchot_loop(net, train_X, train_y, val_X, val_y, args):
    device = torch.device(args.device)
    history = {
        "train_acc": [],
        "val_acc": [],
        "train_loss": [],
        "val_loss": [],
    }

    optimizer = optim.AdamW(net.pretrain_parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.StepLR(
        optimizer,
        step_size=args.step_size,
        gamma=args.gamma,
    )

    best_acc = 0.0
    start_epoch = 1

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=args.device)
        state_dict = strip_module_prefix(unwrap_model_state(checkpoint))
        net.load_state_dict(state_dict)

        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            optimizer_state = checkpoint.get("optimizer_state")
            scheduler_state = checkpoint.get("scheduler_state")
            if optimizer_state is not None:
                optimizer.load_state_dict(optimizer_state)
            if scheduler_state is not None:
                scheduler.load_state_dict(scheduler_state)
            history = checkpoint.get("history", history)
            best_acc = float(checkpoint.get("best_acc", 0.0))
            start_epoch = int(checkpoint.get("epoch", 0)) + 1
            print(f"Resuming ADC-HOT training from {args.resume} at epoch {start_epoch}/{args.num_epochs}")
        else:
            print(f"Loaded ADC-HOT weights from {args.resume}; restarting pretraining from epoch 1")
    elif args.weights:
        load_model_weights(net, args.weights, args.device)
        print(f"Initialized ADC-HOT model weights from {args.weights}")

    supervised_batch_size = max(16, min(64, len(train_X)))
    train_loader = DataLoader(
        TensorDataset(train_X, train_y),
        **_build_loader_kwargs(
            args,
            batch_size=supervised_batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(args.seed),
            worker_seed=args.seed,
        ),
    )
    val_seed = args.seed + 1

    print(
        "ADC-HOT protocol: supervised backbone pretraining on base classes, "
        "then fit base statistics and evaluate episodes with the paper-style calibration head."
    )

    if start_epoch > args.num_epochs:
        print(f"Checkpoint already reached epoch {start_epoch - 1}; skipping ADC-HOT training loop.")
        prepare_dataset_conditioned_model(net, train_X, train_y, args)
        return best_acc, history

    for epoch in range(start_epoch, args.num_epochs + 1):
        net.train()
        total_loss = 0.0
        train_correct = 0
        train_total = 0
        current_lr = optimizer.param_groups[0]["lr"]

        non_blocking = _pin_memory_enabled(args) and device.type == "cuda"
        pbar = tqdm(train_loader, desc=f"ADC-HOT Epoch {epoch}/{args.num_epochs}")
        for images, labels in pbar:
            optimizer.zero_grad()
            images = images.to(device, non_blocking=non_blocking)
            labels = labels.to(device)

            logits = net.pretrain_logits(images)
            loss = F.cross_entropy(logits, labels, label_smoothing=args.label_smoothing)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(net.pretrain_parameters(), args.grad_clip)
            optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{current_lr:.2e}")

        scheduler.step()
        train_acc = train_correct / train_total if train_total > 0 else 0.0
        avg_loss = total_loss / max(1, len(train_loader))

        prepare_dataset_conditioned_model(net, train_X, train_y, args, force=True)

        val_ds = FewshotDataset(
            val_X,
            val_y,
            args.episode_num_val,
            args.way_num,
            args.shot_num,
            args.query_num_val,
            seed=val_seed,
        )
        val_loader = DataLoader(
            val_ds,
            **_build_loader_kwargs(
                args,
                batch_size=1,
                shuffle=False,
                worker_seed=val_seed,
            ),
        )
        val_acc, val_loss = evaluate(net, val_loader, args)

        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["train_loss"].append(avg_loss)
        history["val_loss"].append(val_loss if val_loss else 0.0)

        train_val_gap = train_acc - val_acc
        print(
            f"Epoch {epoch}: SupLoss={avg_loss:.4f}, SupTrain={train_acc:.4f}, "
            f"EpisodicVal={val_acc:.4f} (gap={train_val_gap:+.4f})"
        )

        wandb.log(
            {
                "epoch": epoch,
                "loss/train": avg_loss,
                "loss/val": val_loss,
                "accuracy/train": train_acc,
                "accuracy/val": val_acc,
                "train_val_gap": train_val_gap,
                "lr": current_lr,
            }
        )

        if val_acc > best_acc:
            best_acc = val_acc
            print(f"  -> New best val: {val_acc:.4f}")
            wandb.run.summary["best_val_acc"] = best_acc
            final_path = get_best_model_path(args)
            torch.save(net.state_dict(), final_path)
            print(f"  Saved best model to {final_path}")

        if _bool_flag(args.save_last_checkpoint, default=True):
            last_path = get_last_checkpoint_path(args)
            checkpoint = {
                "epoch": epoch,
                "model_state": net.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_acc": best_acc,
                "history": history,
                "args": vars(args),
            }
            torch.save(checkpoint, last_path)

    samples_str = f"{args.training_samples}samples" if args.training_samples else "allsamples"
    curves_path = os.path.join(
        args.path_results,
        f"training_{args.dataset_name}_{args.model}_{samples_str}_{args.shot_num}shot",
    )
    try:
        plot_training_curves(history, curves_path)
        if os.path.exists(f"{curves_path}_curves.png"):
            wandb.log({"training_curves": wandb.Image(f"{curves_path}_curves.png")})
    except RuntimeError as exc:
        print(f"Skipping training curves: {exc}")

    print(f"Best Validation Accuracy: {best_acc:.4f}")
    return best_acc, history


def evaluate(net, loader, args):
    device = torch.device(args.device)
    relation_loss_fn = RelationLoss().to(device)
    net.eval()
    correct, total = 0, 0
    total_loss = 0.0
    num_batches = 0
    non_blocking = _pin_memory_enabled(args) and device.type == "cuda"

    with torch.no_grad():
        for query, q_labels, support, support_labels in loader:
            batch_size = query.shape[0]
            channels, height, width = query.shape[2], query.shape[3], query.shape[4]
            shot_num = support.shape[1] // args.way_num

            support = support.view(batch_size, args.way_num, shot_num, channels, height, width).to(
                device,
                non_blocking=non_blocking,
            )
            query = query.to(device, non_blocking=non_blocking)
            targets = q_labels.view(-1).to(device)
            support_targets = support_labels.view(batch_size, args.way_num, shot_num).to(device)

            scores = forward_scores(
                net,
                query,
                support,
                args,
                train=False,
                phase="val",
                query_targets=targets,
                support_targets=support_targets,
            )
            logits = extract_logits(scores)
            preds = logits.argmax(dim=1)

            correct += (preds == targets).sum().item()
            total += targets.size(0)

            loss = compute_loss(scores, targets, args, relation_loss_fn, train=False)
            total_loss += loss.item()
            num_batches += 1

    acc = correct / total if total > 0 else 0.0
    avg_loss = total_loss / num_batches if num_batches > 0 else None
    return acc, avg_loss


def calculate_p_value(acc, baseline, n):
    from scipy.stats import norm

    if n <= 0:
        return 1.0
    z = (acc - baseline) / np.sqrt(baseline * (1 - baseline) / n)
    return 2 * norm.sf(abs(z))


def extract_model_features(net, x):
    if hasattr(net, "extract_features"):
        feat = net.extract_features(x)
    elif hasattr(net, "encode"):
        feat = net.encode(x)
    elif hasattr(net, "encoder"):
        feat = net.encoder(x)
    else:
        raise AttributeError("Model does not expose extract_features/encode/encoder")

    if feat.dim() == 4:
        feat = F.adaptive_avg_pool2d(feat, 1).view(feat.size(0), -1)
    elif feat.dim() > 2:
        feat = feat.view(feat.size(0), -1)
    return F.normalize(feat, p=2, dim=-1)


def test_final(net, loader, args, test_X=None, test_y=None, test_file_paths=None):
    import time

    device = torch.device(args.device)
    non_blocking = _pin_memory_enabled(args) and device.type == "cuda"
    num_episodes = len(loader)
    meta = get_model_metadata(args.model)
    print(f"\n{'=' * 60}")
    print(f"Final Test: {meta['display_name']} | {args.dataset_name} | {args.shot_num}-shot")
    print(f"{num_episodes} episodes x {args.way_num} classes x {args.query_num_test} query")
    print("=" * 60)

    net.eval()
    all_preds, all_targets = [], []
    episode_accuracies = []
    episode_times = []
    query_seen = Counter()
    query_mis = Counter()
    query_pair = defaultdict(Counter)

    with torch.no_grad():
        for batch in tqdm(loader, desc="Testing"):
            if len(batch) == 6:
                query, q_labels, support, support_labels, q_indices, _ = batch
                q_indices_np = q_indices.view(-1).cpu().numpy()
            else:
                query, q_labels, support, support_labels = batch
                q_indices_np = None

            start_time = time.perf_counter()

            batch_size, _, channels, height, width = query.shape
            support = support.view(batch_size, args.way_num, args.shot_num, channels, height, width).to(
                device,
                non_blocking=non_blocking,
            )
            query = query.to(device, non_blocking=non_blocking)
            targets = q_labels.view(-1).to(device)
            support_targets = support_labels.view(batch_size, args.way_num, args.shot_num).to(device)

            scores = forward_scores(
                net,
                query,
                support,
                args,
                train=False,
                phase="test",
                query_targets=targets,
                support_targets=support_targets,
            )
            logits = extract_logits(scores)
            preds = logits.argmax(dim=1)

            end_time = time.perf_counter()
            episode_times.append((end_time - start_time) * 1000)
            episode_accuracies.append((preds == targets).float().mean().item())

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

            if q_indices_np is not None:
                preds_np = preds.cpu().numpy()
                targets_np = targets.cpu().numpy()
                for idx_i, true_i, pred_i in zip(q_indices_np, targets_np, preds_np):
                    idx_i = int(idx_i)
                    true_i = int(true_i)
                    pred_i = int(pred_i)
                    query_seen[idx_i] += 1
                    if pred_i != true_i:
                        query_mis[idx_i] += 1
                        query_pair[idx_i][(true_i, pred_i)] += 1

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    episode_accuracies = np.array(episode_accuracies)
    episode_times = np.array(episode_times)

    acc_mean = episode_accuracies.mean()
    acc_std = episode_accuracies.std()
    acc_worst = episode_accuracies.min()
    acc_best = episode_accuracies.max()
    acc_ci95 = 1.96 * acc_std / np.sqrt(len(episode_accuracies))
    time_mean = episode_times.mean()
    time_std = episode_times.std()

    prec, rec, f1, _ = precision_recall_fscore_support(
        all_targets,
        all_preds,
        labels=list(range(args.way_num)),
        average="macro",
        zero_division=0,
    )
    p_val = calculate_p_value(acc_mean, 1.0 / args.way_num, len(all_targets))

    print(f"\n{'=' * 60}")
    print("ACCURACY METRICS")
    print("=" * 60)
    print(f"  Mean Accuracy : {acc_mean * 100:.2f} +/- {acc_ci95 * 100:.2f}% (95% CI)")
    print(f"  Std Deviation : {acc_std * 100:.2f}%")
    print(f"  Worst-case    : {acc_worst * 100:.2f}%")
    print(f"  Best-case     : {acc_best * 100:.2f}%")
    print(f"  Precision     : {prec:.4f}")
    print(f"  Recall        : {rec:.4f}")
    print(f"  F1-Score      : {f1:.4f}")
    print(f"  p-value       : {p_val:.2e}")
    print(f"\nInference Time  : {time_mean:.2f} +/- {time_std:.2f} ms/episode")

    wandb.log(
        {
            "test_accuracy_mean": acc_mean,
            "test_accuracy_std": acc_std,
            "test_accuracy_ci95": acc_ci95,
            "test_accuracy_worst": acc_worst,
            "test_accuracy_best": acc_best,
            "test_precision": prec,
            "test_recall": rec,
            "test_f1": f1,
            "test_p_value": p_val,
            "inference_time_mean_ms": time_mean,
            "inference_time_std_ms": time_std,
        }
    )

    wandb.run.summary["test_accuracy_mean"] = acc_mean
    wandb.run.summary["test_accuracy_ci95"] = acc_ci95

    samples_str = f"_{args.training_samples}samples" if args.training_samples else "_allsamples"
    cm_base = os.path.join(
        args.path_results,
        f"confusion_matrix_{args.dataset_name}_{args.model}_{samples_str.strip('_')}_{args.shot_num}shot",
    )
    try:
        plot_confusion_matrix(all_targets, all_preds, args.way_num, cm_base, class_names=args.class_names)
        if os.path.exists(f"{cm_base}_2col.png"):
            wandb.log({"confusion_matrix": wandb.Image(f"{cm_base}_2col.png")})
    except RuntimeError as exc:
        print(f"Skipping confusion matrix plot: {exc}")

    if args.save_misclf_report and query_seen:
        pair_totals = Counter()
        true_totals = Counter()
        for true_i, pred_i in zip(all_targets.tolist(), all_preds.tolist()):
            true_totals[int(true_i)] += 1
            if int(true_i) != int(pred_i):
                pair_totals[(int(true_i), int(pred_i))] += 1

        pair_path = os.path.join(
            args.path_results,
            f"misclass_pairs_{args.dataset_name}_{args.model}_{samples_str.strip('_')}_{args.shot_num}shot.txt",
        )
        with open(pair_path, "w") as handle:
            handle.write("Most common confusion pairs (True -> Pred)\n")
            handle.write("-" * 70 + "\n")
            for (true_i, pred_i), count in pair_totals.most_common():
                denom = max(1, true_totals[true_i])
                rate = count / denom
                true_name = args.class_names[true_i] if true_i < len(args.class_names) else f"Class{true_i}"
                pred_name = args.class_names[pred_i] if pred_i < len(args.class_names) else f"Class{pred_i}"
                handle.write(f"{true_name} -> {pred_name}: {count} ({rate:.2%} of true {true_name})\n")

        rows = []
        for idx_i, seen in query_seen.items():
            mis = query_mis.get(idx_i, 0)
            if mis <= 0:
                continue
            top_pair, top_pair_cnt = query_pair[idx_i].most_common(1)[0]
            true_i, pred_i = top_pair
            file_path = f"index_{idx_i}"
            if test_file_paths is not None and 0 <= idx_i < len(test_file_paths):
                file_path = test_file_paths[idx_i]

            rows.append(
                {
                    "file_index": idx_i,
                    "file_path": file_path,
                    "mis_count": mis,
                    "seen_count": seen,
                    "mis_rate": mis / max(1, seen),
                    "top_confusion_true": args.class_names[true_i],
                    "top_confusion_pred": args.class_names[pred_i],
                    "top_confusion_count": top_pair_cnt,
                }
            )

        rows.sort(key=lambda item: (-item["mis_count"], -item["mis_rate"], item["file_index"]))
        if args.misclf_topk > 0:
            rows = rows[: args.misclf_topk]

        mis_path = os.path.join(
            args.path_results,
            f"misclassified_files_{args.dataset_name}_{args.model}_{samples_str.strip('_')}_{args.shot_num}shot.csv",
        )
        with open(mis_path, "w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "file_index",
                    "file_path",
                    "mis_count",
                    "seen_count",
                    "mis_rate",
                    "top_confusion_true",
                    "top_confusion_pred",
                    "top_confusion_count",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

        print(f"Saved confusion-pair summary: {pair_path}")
        print(f"Saved per-file misclassification report: {mis_path}")

    if test_X is not None and test_y is not None:
        print(f"\n{'=' * 60}")
        print("Extracting features for t-SNE from unique test samples")
        print(f"Test set size: {len(test_X)} samples ({len(test_X) // args.way_num}/class)")
        print("=" * 60)

        with torch.no_grad():
            test_X_device = test_X.to(device)
            test_y_np = test_y.cpu().numpy()
            batch_size = 32
            all_features = []

            for start in range(0, len(test_X), batch_size):
                batch_X = test_X_device[start : start + batch_size]
                feat = extract_model_features(net, batch_X)
                all_features.append(feat.cpu().numpy())

            features = np.vstack(all_features)
            print(f"Extracted {len(features)} unique features (shape: {features.shape})")

            tsne_path = os.path.join(
                args.path_results,
                f"tsne_{args.dataset_name}_{args.model}_{samples_str.strip('_')}_{args.shot_num}shot",
            )
            try:
                plot_tsne(features, test_y_np, args.way_num, tsne_path, class_names=args.class_names)
                if os.path.exists(f"{tsne_path}_tsne.png"):
                    wandb.log({"tsne_plot": wandb.Image(f"{tsne_path}_tsne.png")})
            except RuntimeError as exc:
                print(f"Skipping t-SNE plot: {exc}")

    txt_path = os.path.join(
        args.path_results,
        f"results_{args.dataset_name}_{args.model}_{samples_str.strip('_')}_{args.shot_num}shot.txt",
    )
    with open(txt_path, "w") as handle:
        handle.write(f"Model: {meta['display_name']} ({args.model})\n")
        handle.write(f"Dataset: {args.dataset_name}\n")
        handle.write(f"Shot: {args.shot_num}\n")
        handle.write(f"Training Samples: {args.training_samples if args.training_samples else 'All'}\n")
        handle.write("-" * 40 + "\n")
        handle.write(f"Accuracy : {acc_mean:.4f} +/- {acc_std:.4f}\n")
        handle.write(f"Worst-case : {acc_worst:.4f}\n")
        handle.write(f"Best-case : {acc_best:.4f}\n")
        handle.write(f"Precision : {prec:.4f}\n")
        handle.write(f"Recall : {rec:.4f}\n")
        handle.write(f"F1-Score : {f1:.4f}\n")
        handle.write(f"Inference Time: {time_mean:.2f} +/- {time_std:.2f} ms/episode\n")
    print(f"Results saved to {txt_path}")


def calculate_flops(model, input_size=(3, 64, 64), device="cuda"):
    if not THOP_AVAILABLE:
        return None, None
    try:
        dummy_input = torch.randn(1, *input_size).to(device)
        if hasattr(model, "encoder"):
            macs, params = profile(model.encoder, inputs=(dummy_input,), verbose=False)
            return macs, params
    except Exception as exc:
        print(f"Warning: Could not calculate FLOPs: {exc}")
    return None, None


def log_model_parameters(model, model_name, device="cuda", image_size=64):
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)

    log_dict = {
        "model/total_parameters": total_params,
        "model/trainable_parameters": trainable_params,
    }

    macs, flops_params = calculate_flops(model, input_size=(3, image_size, image_size), device=device)
    if macs is not None:
        flops = macs * 2
        macs_readable, params_readable = clever_format([macs, flops_params], "%.2f")
        flops_readable = clever_format([flops], "%.2f")[0]
        log_dict.update(
            {
                "model/macs": macs,
                "model/flops": flops,
                "model/macs_readable": macs_readable,
                "model/flops_readable": flops_readable,
                "model/profile_params_readable": params_readable,
            }
        )
        print(f"MACs (encoder): {macs_readable}")
        print(f"FLOPs (encoder): {flops_readable}")

    wandb.log(log_dict)
    wandb.run.summary["model_total_params"] = total_params
    wandb.run.summary["model_trainable_params"] = trainable_params
    print(f"\nModel Parameters: {total_params:,} (trainable: {trainable_params:,})")


def main():
    args = get_args()
    args.device = resolve_runtime_device(args)
    args.fewshot_backbone = resolve_fewshot_backbone(args)
    model_meta = get_model_metadata(args.model)

    if args.query_num is not None:
        args.query_num_train = args.query_num
        args.query_num_val = args.query_num
        args.query_num_test = args.query_num

    print(f"\n{'=' * 72}")
    print(model_meta["display_name"])
    print("=" * 72)
    print(f"Mode        : {args.mode}")
    print(f"Config      : {args.model} | {args.shot_num}-shot | {args.num_epochs} epochs | device={args.device}")
    print(f"Dataset     : {args.dataset_path} ({args.dataset_name})")
    print(f"Architecture: {model_meta['architecture']}")
    print(f"Backbone    : {args.fewshot_backbone}")

    samples_str = f"{args.training_samples}samples" if args.training_samples else "all"
    run_name = f"{args.model}_{args.dataset_name}_{samples_str}_{args.shot_num}shot"
    config = vars(args).copy()
    config["architecture"] = model_meta["architecture"]
    config["distance_metric"] = model_meta["metric"]

    wandb.init(
        project=args.project,
        config=config,
        name=run_name,
        group=f"{args.model}_{args.dataset_name}",
        job_type=args.mode,
    )

    seed_func(args.seed)
    configure_cudnn_runtime(args)
    print(
        f"Runtime     : workers={args.num_workers}, pin_memory={_pin_memory_enabled(args)}, "
        f"persistent_workers={_persistent_workers_enabled(args)}, "
        f"cudnn(det={torch.backends.cudnn.deterministic}, bench={torch.backends.cudnn.benchmark})"
    )

    os.makedirs(args.path_weights, exist_ok=True)
    os.makedirs(args.path_results, exist_ok=True)

    dataset = load_dataset(args.dataset_path, image_size=args.image_size)

    def to_tensor(images, labels):
        return torch.from_numpy(images.astype(np.float32)), torch.from_numpy(labels).long()

    train_X, train_y = to_tensor(dataset.X_train, dataset.y_train)
    val_X, val_y = to_tensor(dataset.X_val, dataset.y_val)
    test_X, test_y = to_tensor(dataset.X_test, dataset.y_test)
    train_file_paths = [path for path, _ in getattr(dataset, "train_files", [])] if hasattr(dataset, "train_files") else None
    val_file_paths = [path for path, _ in getattr(dataset, "val_files", [])] if hasattr(dataset, "val_files") else None
    test_file_paths = [path for path, _ in getattr(dataset, "test_files", [])] if hasattr(dataset, "test_files") else None

    pretty_map = {
        "surface": "Surface",
        "internal": "Internal",
        "corona": "Corona",
        "notpd": "NotPD",
        "nopd": "NotPD",
    }
    dataset_classes = list(getattr(dataset, "classes", []))
    if dataset_classes:
        all_class_names = [pretty_map.get(class_name.lower(), class_name) for class_name in dataset_classes]
    else:
        num_classes = int(len(torch.unique(train_y)))
        all_class_names = [f"Class{i}" for i in range(num_classes)]

    if args.selected_classes:
        selected = [int(class_id.strip()) for class_id in args.selected_classes.split(",")]
        if any(class_id < 0 or class_id >= len(all_class_names) for class_id in selected):
            raise ValueError(f"selected_classes={selected} out of range for classes={all_class_names}")
        print(f"\nUsing only selected classes: {selected}")
        args.class_names = [all_class_names[class_id] for class_id in selected]
        args.way_num = len(selected)

        def filter_classes(images, labels, selected_classes, file_paths=None):
            mask = torch.zeros(len(labels), dtype=torch.bool)
            for class_id in selected_classes:
                mask |= labels == class_id

            filtered_images = images[mask]
            filtered_labels = labels[mask]
            label_map = {old: new for new, old in enumerate(selected_classes)}
            remapped_labels = torch.tensor([label_map[label.item()] for label in filtered_labels])
            filtered_paths = None
            if file_paths is not None and len(file_paths) == len(labels):
                filtered_paths = [path for path, keep in zip(file_paths, mask.tolist()) if keep]
            return filtered_images, remapped_labels, filtered_paths

        train_X, train_y, train_file_paths = filter_classes(train_X, train_y, selected, train_file_paths)
        val_X, val_y, val_file_paths = filter_classes(val_X, val_y, selected, val_file_paths)
        test_X, test_y, test_file_paths = filter_classes(test_X, test_y, selected, test_file_paths)
        print(f"Train: {len(train_X)}, Val: {len(val_X)}, Test: {len(test_X)}")
    else:
        args.class_names = all_class_names
        if args.way_num != len(args.class_names):
            print(
                f"way_num={args.way_num} does not match dataset classes={len(args.class_names)}. "
                f"Using way_num={len(args.class_names)}."
            )
            args.way_num = len(args.class_names)

    wandb.config.update(
        {
            "way_num": args.way_num,
            "query_num_train": args.query_num_train,
            "query_num_val": args.query_num_val,
            "query_num_test": args.query_num_test,
        },
        allow_val_change=True,
    )

    if args.training_samples:
        if args.training_samples % args.way_num != 0:
            raise ValueError(
                f"training_samples ({args.training_samples}) must be divisible by way_num ({args.way_num}) "
                "for balanced class sampling."
            )
        per_class = args.training_samples // args.way_num
        sample_images, sample_labels = [], []
        for class_id in range(args.way_num):
            indices = (train_y == class_id).nonzero(as_tuple=True)[0]
            if len(indices) < per_class:
                raise ValueError(f"Class {class_id}: need {per_class}, have {len(indices)}")
            generator = torch.Generator().manual_seed(args.seed)
            perm = torch.randperm(len(indices), generator=generator)[:per_class]
            sample_images.append(train_X[indices[perm]])
            sample_labels.append(train_y[indices[perm]])
        train_X = torch.cat(sample_images)
        train_y = torch.cat(sample_labels)
        print(f"Using {args.training_samples} training samples ({per_class}/class)")

    test_ds = FewshotDataset(
        test_X,
        test_y,
        args.episode_num_test,
        args.way_num,
        args.shot_num,
        args.query_num_test,
        args.seed,
        return_indices=args.save_misclf_report,
    )
    test_loader = DataLoader(
        test_ds,
        **_build_loader_kwargs(
            args,
            batch_size=1,
            shuffle=False,
            worker_seed=args.seed,
        ),
    )

    net = get_model(args)
    log_model_parameters(net, args.model, device=args.device, image_size=args.image_size)

    if args.mode == "train":
        if args.model == "adchot":
            train_adchot_loop(net, train_X, train_y, val_X, val_y, args)
        else:
            train_loop(net, train_X, train_y, val_X, val_y, args)
        if not _bool_flag(args.skip_final_test, default=False):
            path = get_best_model_path(args)
            print(f"Testing with best checkpoint: {path}")
            load_model_weights(net, path, args.device)
            prepare_dataset_conditioned_model(net, train_X, train_y, args)
            test_final(net, test_loader, args, test_X=test_X, test_y=test_y, test_file_paths=test_file_paths)
    else:
        if args.weights:
            load_model_weights(net, args.weights, args.device)
            prepare_dataset_conditioned_model(net, train_X, train_y, args)
            test_final(net, test_loader, args, test_X=test_X, test_y=test_y, test_file_paths=test_file_paths)
        else:
            print("Error: Please specify --weights for test mode")

    wandb.finish()


if __name__ == "__main__":
    main()
