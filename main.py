"""Few-shot benchmark training and evaluation entrypoint."""

import csv
import os
import argparse
import re
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
import wandb
import matplotlib.pyplot as plt

try:
    from thop import profile, clever_format
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False
    print("Warning: thop not installed. Run 'pip install thop' for FLOPs calculation.")

from dataset import load_dataset, load_external_scalogram_split
from dataloader.dataloader import FewshotDataset, RobustFewshotDataset
from function.function import (
    ContrastiveLoss,
    RelationLoss,
    plot_confusion_matrix,
    plot_tsne,
    plot_training_curves,
    seed_func,
)
from net.model_factory import (
    HROT_FSL_MODEL_NAMES,
    M2_FULL_OT_MODEL_NAMES,
    M2_MODEL_NAMES,
    build_model_from_args,
    get_model_choices,
    get_model_metadata,
    resolve_fewshot_backbone,
)
from visualization import (
    compute_support_episode_distribution,
    export_dataset_noise_profile,
    export_episode_q1_figure,
    export_support_distribution_figure,
)


def _bool_flag(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).lower() == "true"


PAPER_BASELINE_MODELS = {
    "cosine",
    "protonet",
    "covamnet",
    "matchingnet",
    "relationnet",
    "dn4",
    "feat",
    "deepemd",
    "evidence_deepemd",
    "can",
    "frn",
    "deepbdc",
    "maml",
}


def model_uses_label_smoothing(model_name):
    return str(model_name) not in PAPER_BASELINE_MODELS


def effective_label_smoothing(args, train=False):
    if not train:
        return 0.0
    if not model_uses_label_smoothing(getattr(args, "model", "")):
        return 0.0
    return float(getattr(args, "label_smoothing", 0.0))


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


def apply_optional_env_overrides(args):
    return args


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
    parser.add_argument(
        "--fewshot_backbone",
        type=str,
        default="resnet12",
        choices=["default", "resnet12", "conv64f", "fsl_mamba", "slim_mamba"],
    )

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
    parser.add_argument("--image_size", type=int, default=84)
    parser.add_argument("--token_dim", type=int, default=128)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--ssm_state_dim", type=int, default=16)
    parser.add_argument("--ssm_depth", type=int, default=2)
    parser.add_argument("--sw_num_projections", type=int, default=64)
    parser.add_argument("--sw_weight", type=float, default=1.0)
    parser.add_argument("--transport_temperature", type=float, default=6.0)
    parser.add_argument("--use_global_proto_branch", type=str, default="false", choices=["true", "false"])
    parser.add_argument("--use_flow_branch", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--use_align_loss", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--use_smooth_loss", type=str, default="false", choices=["true", "false"])
    parser.add_argument("--distance_type", type=str, default="sw", choices=["sw", "entropic_ot"])
    parser.add_argument(
        "--class_context_type",
        type=str,
        default="deepsets",
        choices=["deepsets", "lightweight_set_transformer"],
    )
    parser.add_argument("--flow_conditioning_type", type=str, default="concat", choices=["concat", "film"])
    parser.add_argument("--num_flow_particles", type=int, default=16)
    parser.add_argument("--fm_time_schedule", type=str, default="uniform", choices=["uniform", "beta"])
    parser.add_argument("--score_temperature", type=float, default=8.0)
    parser.add_argument("--sc_lfi_context_dim", type=int, default=128)
    parser.add_argument("--sc_lfi_context_hidden_dim", type=int, default=None)
    parser.add_argument("--sc_lfi_latent_hidden_dim", type=int, default=None)
    parser.add_argument("--sc_lfi_flow_hidden_dim", type=int, default=128)
    parser.add_argument("--sc_lfi_flow_time_embedding_dim", type=int, default=32)
    parser.add_argument("--sc_lfi_num_flow_integration_steps", type=int, default=8)
    parser.add_argument("--sc_lfi_proto_branch_weight", type=float, default=0.2)
    parser.add_argument("--sc_lfi_lambda_fm", type=float, default=0.05)
    parser.add_argument("--sc_lfi_lambda_align", type=float, default=0.1)
    parser.add_argument("--sc_lfi_lambda_smooth", type=float, default=0.0)
    parser.add_argument("--sc_lfi_distance_normalize_inputs", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--sc_lfi_distance_sw_num_projections", type=int, default=64)
    parser.add_argument("--sc_lfi_distance_sw_p", type=float, default=2.0)
    parser.add_argument("--sc_lfi_distance_projection_seed", type=int, default=7)
    parser.add_argument("--sc_lfi_sinkhorn_epsilon", type=float, default=0.1)
    parser.add_argument("--sc_lfi_sinkhorn_iterations", type=int, default=50)
    parser.add_argument("--sc_lfi_context_num_memory_tokens", type=int, default=4)
    parser.add_argument("--sc_lfi_context_num_heads", type=int, default=4)
    parser.add_argument("--sc_lfi_context_ffn_multiplier", type=int, default=2)
    parser.add_argument("--sc_lfi_eval_particle_seed", type=int, default=7)
    parser.add_argument("--sc_lfi_eps", type=float, default=1e-6)
    parser.add_argument("--sc_lfi_v2_context_dim", type=int, default=128)
    parser.add_argument("--sc_lfi_v2_context_hidden_dim", type=int, default=None)
    parser.add_argument("--sc_lfi_v2_latent_hidden_dim", type=int, default=None)
    parser.add_argument("--sc_lfi_v2_mass_hidden_dim", type=int, default=None)
    parser.add_argument("--sc_lfi_v2_mass_temperature", type=float, default=1.0)
    parser.add_argument("--sc_lfi_v2_memory_size", type=int, default=4)
    parser.add_argument("--sc_lfi_v2_memory_num_heads", type=int, default=4)
    parser.add_argument("--sc_lfi_v2_memory_ffn_multiplier", type=int, default=2)
    parser.add_argument("--sc_lfi_v2_flow_hidden_dim", type=int, default=128)
    parser.add_argument("--sc_lfi_v2_flow_time_embedding_dim", type=int, default=32)
    parser.add_argument("--sc_lfi_v2_flow_memory_num_heads", type=int, default=4)
    parser.add_argument("--sc_lfi_v2_flow_conditioning_type", type=str, default="film", choices=["concat", "film"])
    parser.add_argument("--sc_lfi_v2_use_global_proto_branch", type=str, default="false", choices=["true", "false"])
    parser.add_argument("--sc_lfi_v2_use_flow_branch", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--sc_lfi_v2_use_align_loss", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--sc_lfi_v2_use_margin_loss", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--sc_lfi_v2_use_smooth_loss", type=str, default="false", choices=["true", "false"])
    parser.add_argument("--sc_lfi_v2_train_num_flow_particles", type=int, default=8)
    parser.add_argument("--sc_lfi_v2_eval_num_flow_particles", type=int, default=16)
    parser.add_argument("--sc_lfi_v2_train_num_integration_steps", type=int, default=8)
    parser.add_argument("--sc_lfi_v2_eval_num_integration_steps", type=int, default=16)
    parser.add_argument("--sc_lfi_v2_solver_type", type=str, default="heun", choices=["euler", "heun"])
    parser.add_argument("--sc_lfi_v2_proto_branch_weight", type=float, default=0.15)
    parser.add_argument("--sc_lfi_v2_lambda_fm", type=float, default=0.05)
    parser.add_argument("--sc_lfi_v2_lambda_align", type=float, default=0.1)
    parser.add_argument("--sc_lfi_v2_lambda_margin", type=float, default=0.1)
    parser.add_argument("--sc_lfi_v2_lambda_support_fit", type=float, default=0.1)
    parser.add_argument("--sc_lfi_v2_lambda_smooth", type=float, default=0.0)
    parser.add_argument("--sc_lfi_v2_margin_value", type=float, default=0.1)
    parser.add_argument("--sc_lfi_v2_support_mix_min", type=float, default=0.45)
    parser.add_argument("--sc_lfi_v2_support_mix_max", type=float, default=0.8)
    parser.add_argument("--sc_lfi_v2_score_train_num_projections", type=int, default=64)
    parser.add_argument("--sc_lfi_v2_score_eval_num_projections", type=int, default=128)
    parser.add_argument("--sc_lfi_v2_score_sw_p", type=float, default=2.0)
    parser.add_argument("--sc_lfi_v2_score_normalize_inputs", type=str, default="true", choices=["true", "false"])
    parser.add_argument(
        "--sc_lfi_v2_score_train_projection_mode",
        type=str,
        default="resample",
        choices=["resample", "fixed"],
    )
    parser.add_argument(
        "--sc_lfi_v2_score_eval_projection_mode",
        type=str,
        default="fixed",
        choices=["resample", "fixed"],
    )
    parser.add_argument("--sc_lfi_v2_score_eval_num_repeats", type=int, default=1)
    parser.add_argument("--sc_lfi_v2_score_projection_seed", type=int, default=7)
    parser.add_argument(
        "--sc_lfi_v2_align_distance_type",
        type=str,
        default="weighted_entropic_ot",
        choices=["weighted_sw", "weighted_entropic_ot"],
    )
    parser.add_argument("--sc_lfi_v2_align_train_num_projections", type=int, default=64)
    parser.add_argument("--sc_lfi_v2_align_eval_num_projections", type=int, default=128)
    parser.add_argument("--sc_lfi_v2_align_sw_p", type=float, default=2.0)
    parser.add_argument("--sc_lfi_v2_align_normalize_inputs", type=str, default="true", choices=["true", "false"])
    parser.add_argument(
        "--sc_lfi_v2_align_train_projection_mode",
        type=str,
        default="resample",
        choices=["resample", "fixed"],
    )
    parser.add_argument(
        "--sc_lfi_v2_align_eval_projection_mode",
        type=str,
        default="fixed",
        choices=["resample", "fixed"],
    )
    parser.add_argument("--sc_lfi_v2_align_eval_num_repeats", type=int, default=1)
    parser.add_argument("--sc_lfi_v2_align_projection_seed", type=int, default=11)
    parser.add_argument("--sc_lfi_v2_sinkhorn_epsilon", type=float, default=0.05)
    parser.add_argument("--sc_lfi_v2_sinkhorn_iterations", type=int, default=80)
    parser.add_argument("--sc_lfi_v2_sinkhorn_cost_power", type=float, default=2.0)
    parser.add_argument("--sc_lfi_v2_eval_particle_seed", type=int, default=7)
    parser.add_argument("--sc_lfi_v2_eps", type=float, default=1e-8)
    parser.add_argument("--sc_lfi_v3_context_dim", type=int, default=128)
    parser.add_argument("--sc_lfi_v3_context_hidden_dim", type=int, default=None)
    parser.add_argument("--sc_lfi_v3_latent_hidden_dim", type=int, default=None)
    parser.add_argument("--sc_lfi_v3_mass_hidden_dim", type=int, default=None)
    parser.add_argument("--sc_lfi_v3_mass_temperature", type=float, default=1.0)
    parser.add_argument("--sc_lfi_v3_memory_size", type=int, default=4)
    parser.add_argument("--sc_lfi_v3_memory_num_heads", type=int, default=4)
    parser.add_argument("--sc_lfi_v3_memory_ffn_multiplier", type=int, default=2)
    parser.add_argument("--sc_lfi_v3_prior_num_atoms", type=int, default=4)
    parser.add_argument("--sc_lfi_v3_prior_scale", type=float, default=0.6)
    parser.add_argument("--sc_lfi_v3_episode_num_heads", type=int, default=4)
    parser.add_argument("--sc_lfi_v3_alpha_hidden_dim", type=int, default=None)
    parser.add_argument("--sc_lfi_v3_alpha_shot_scale", type=float, default=1.25)
    parser.add_argument("--sc_lfi_v3_alpha_uncertainty_scale", type=float, default=1.0)
    parser.add_argument("--sc_lfi_v3_flow_hidden_dim", type=int, default=128)
    parser.add_argument("--sc_lfi_v3_flow_time_embedding_dim", type=int, default=32)
    parser.add_argument("--sc_lfi_v3_flow_memory_num_heads", type=int, default=4)
    parser.add_argument("--sc_lfi_v3_flow_conditioning_type", type=str, default="film", choices=["concat", "film"])
    parser.add_argument("--sc_lfi_v3_use_transport_flow", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--sc_lfi_v3_use_prior_measure", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--sc_lfi_v3_use_episode_adapter", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--sc_lfi_v3_use_query_reweighting", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--sc_lfi_v3_use_support_barycenter_only", type=str, default="false", choices=["true", "false"])
    parser.add_argument("--sc_lfi_v3_use_global_proto_branch", type=str, default="false", choices=["true", "false"])
    parser.add_argument("--sc_lfi_v3_use_align_loss", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--sc_lfi_v3_use_margin_loss", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--sc_lfi_v3_train_num_integration_steps", type=int, default=4)
    parser.add_argument("--sc_lfi_v3_eval_num_integration_steps", type=int, default=8)
    parser.add_argument("--sc_lfi_v3_solver_type", type=str, default="heun", choices=["euler", "heun"])
    parser.add_argument("--sc_lfi_v3_query_reweight_temperature", type=float, default=1.0)
    parser.add_argument("--sc_lfi_v3_proto_branch_weight", type=float, default=0.1)
    parser.add_argument("--sc_lfi_v3_lambda_fm", type=float, default=0.035)
    parser.add_argument("--sc_lfi_v3_lambda_align", type=float, default=0.05)
    parser.add_argument("--sc_lfi_v3_lambda_margin", type=float, default=0.12)
    parser.add_argument("--sc_lfi_v3_lambda_reg", type=float, default=0.08)
    parser.add_argument("--sc_lfi_v3_margin_value", type=float, default=0.12)
    parser.add_argument("--sc_lfi_v3_support_entropy_target_ratio", type=float, default=0.58)
    parser.add_argument("--sc_lfi_v3_query_entropy_target_ratio", type=float, default=0.45)
    parser.add_argument("--sc_lfi_v3_relevance_entropy_target_ratio", type=float, default=0.4)
    parser.add_argument("--sc_lfi_v3_score_train_num_projections", type=int, default=64)
    parser.add_argument("--sc_lfi_v3_score_eval_num_projections", type=int, default=128)
    parser.add_argument("--sc_lfi_v3_score_sw_p", type=float, default=2.0)
    parser.add_argument("--sc_lfi_v3_score_normalize_inputs", type=str, default="true", choices=["true", "false"])
    parser.add_argument(
        "--sc_lfi_v3_score_train_projection_mode",
        type=str,
        default="resample",
        choices=["resample", "fixed"],
    )
    parser.add_argument(
        "--sc_lfi_v3_score_eval_projection_mode",
        type=str,
        default="fixed",
        choices=["resample", "fixed"],
    )
    parser.add_argument("--sc_lfi_v3_score_eval_num_repeats", type=int, default=1)
    parser.add_argument("--sc_lfi_v3_score_projection_seed", type=int, default=7)
    parser.add_argument(
        "--sc_lfi_v3_align_distance_type",
        type=str,
        default="weighted_entropic_ot",
        choices=["weighted_sw", "weighted_entropic_ot"],
    )
    parser.add_argument("--sc_lfi_v3_align_train_num_projections", type=int, default=64)
    parser.add_argument("--sc_lfi_v3_align_eval_num_projections", type=int, default=128)
    parser.add_argument("--sc_lfi_v3_align_sw_p", type=float, default=2.0)
    parser.add_argument("--sc_lfi_v3_align_normalize_inputs", type=str, default="true", choices=["true", "false"])
    parser.add_argument(
        "--sc_lfi_v3_align_train_projection_mode",
        type=str,
        default="resample",
        choices=["resample", "fixed"],
    )
    parser.add_argument(
        "--sc_lfi_v3_align_eval_projection_mode",
        type=str,
        default="fixed",
        choices=["resample", "fixed"],
    )
    parser.add_argument("--sc_lfi_v3_align_eval_num_repeats", type=int, default=1)
    parser.add_argument("--sc_lfi_v3_align_projection_seed", type=int, default=11)
    parser.add_argument("--sc_lfi_v3_sinkhorn_epsilon", type=float, default=0.08)
    parser.add_argument("--sc_lfi_v3_sinkhorn_iterations", type=int, default=100)
    parser.add_argument("--sc_lfi_v3_sinkhorn_cost_power", type=float, default=2.0)
    parser.add_argument("--sc_lfi_v3_eps", type=float, default=1e-8)
    parser.add_argument("--sc_lfi_v4_context_dim", type=int, default=128)
    parser.add_argument("--sc_lfi_v4_context_hidden_dim", type=int, default=None)
    parser.add_argument("--sc_lfi_v4_latent_hidden_dim", type=int, default=None)
    parser.add_argument("--sc_lfi_v4_mass_hidden_dim", type=int, default=None)
    parser.add_argument("--sc_lfi_v4_mass_temperature", type=float, default=1.0)
    parser.add_argument("--sc_lfi_v4_shot_memory_size", type=int, default=2)
    parser.add_argument("--sc_lfi_v4_class_memory_size", type=int, default=4)
    parser.add_argument("--sc_lfi_v4_memory_num_heads", type=int, default=4)
    parser.add_argument("--sc_lfi_v4_memory_ffn_multiplier", type=int, default=2)
    parser.add_argument("--sc_lfi_v4_prior_num_atoms", type=int, default=4)
    parser.add_argument("--sc_lfi_v4_global_prior_size", type=int, default=16)
    parser.add_argument("--sc_lfi_v4_prior_scale", type=float, default=0.25)
    parser.add_argument("--sc_lfi_v4_episode_num_heads", type=int, default=4)
    parser.add_argument("--sc_lfi_v4_alpha_hidden_dim", type=int, default=None)
    parser.add_argument("--sc_lfi_v4_shrinkage_kappa", type=float, default=2.0)
    parser.add_argument("--sc_lfi_v4_alpha_uncertainty_scale", type=float, default=1.0)
    parser.add_argument("--sc_lfi_v4_use_shot_barycenter_only", type=str, default="false", choices=["true", "false"])
    parser.add_argument("--sc_lfi_v4_flow_hidden_dim", type=int, default=128)
    parser.add_argument("--sc_lfi_v4_flow_time_embedding_dim", type=int, default=32)
    parser.add_argument("--sc_lfi_v4_flow_memory_num_heads", type=int, default=4)
    parser.add_argument("--sc_lfi_v4_flow_conditioning_type", type=str, default="film", choices=["concat", "film"])
    parser.add_argument("--sc_lfi_v4_use_transport_flow", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--sc_lfi_v4_use_prior_measure", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--sc_lfi_v4_use_episode_adapter", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--sc_lfi_v4_use_metric_conditioning", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--sc_lfi_v4_use_align_loss", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--sc_lfi_v4_use_margin_loss", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--sc_lfi_v4_use_loo_loss", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--sc_lfi_v4_train_num_integration_steps", type=int, default=4)
    parser.add_argument("--sc_lfi_v4_eval_num_integration_steps", type=int, default=8)
    parser.add_argument("--sc_lfi_v4_solver_type", type=str, default="heun", choices=["euler", "heun"])
    parser.add_argument("--sc_lfi_v4_lambda_fm", type=float, default=0.03)
    parser.add_argument("--sc_lfi_v4_lambda_align", type=float, default=0.05)
    parser.add_argument("--sc_lfi_v4_lambda_margin", type=float, default=0.1)
    parser.add_argument("--sc_lfi_v4_lambda_loo", type=float, default=0.1)
    parser.add_argument("--sc_lfi_v4_lambda_reg", type=float, default=0.05)
    parser.add_argument("--sc_lfi_v4_margin_value", type=float, default=0.1)
    parser.add_argument("--sc_lfi_v4_support_token_entropy_target_ratio", type=float, default=0.55)
    parser.add_argument("--sc_lfi_v4_query_entropy_target_ratio", type=float, default=0.45)
    parser.add_argument("--sc_lfi_v4_shot_entropy_target_ratio", type=float, default=0.55)
    parser.add_argument("--sc_lfi_v4_prior_entropy_target_ratio", type=float, default=0.45)
    parser.add_argument(
        "--sc_lfi_v4_score_distance_type",
        type=str,
        default="weighted_entropic_ot",
        choices=["weighted_sw", "weighted_entropic_ot"],
    )
    parser.add_argument("--sc_lfi_v4_score_train_num_projections", type=int, default=64)
    parser.add_argument("--sc_lfi_v4_score_eval_num_projections", type=int, default=128)
    parser.add_argument("--sc_lfi_v4_score_sw_p", type=float, default=2.0)
    parser.add_argument("--sc_lfi_v4_score_normalize_inputs", type=str, default="true", choices=["true", "false"])
    parser.add_argument(
        "--sc_lfi_v4_score_train_projection_mode",
        type=str,
        default="resample",
        choices=["resample", "fixed"],
    )
    parser.add_argument(
        "--sc_lfi_v4_score_eval_projection_mode",
        type=str,
        default="fixed",
        choices=["resample", "fixed"],
    )
    parser.add_argument("--sc_lfi_v4_score_eval_num_repeats", type=int, default=1)
    parser.add_argument("--sc_lfi_v4_score_projection_seed", type=int, default=7)
    parser.add_argument("--sc_lfi_v4_score_sinkhorn_epsilon", type=float, default=0.05)
    parser.add_argument("--sc_lfi_v4_score_sinkhorn_iterations", type=int, default=60)
    parser.add_argument("--sc_lfi_v4_score_sinkhorn_cost_power", type=float, default=2.0)
    parser.add_argument(
        "--sc_lfi_v4_align_distance_type",
        type=str,
        default="weighted_entropic_ot",
        choices=["weighted_sw", "weighted_entropic_ot"],
    )
    parser.add_argument("--sc_lfi_v4_align_train_num_projections", type=int, default=64)
    parser.add_argument("--sc_lfi_v4_align_eval_num_projections", type=int, default=128)
    parser.add_argument("--sc_lfi_v4_align_sw_p", type=float, default=2.0)
    parser.add_argument("--sc_lfi_v4_align_normalize_inputs", type=str, default="true", choices=["true", "false"])
    parser.add_argument(
        "--sc_lfi_v4_align_train_projection_mode",
        type=str,
        default="resample",
        choices=["resample", "fixed"],
    )
    parser.add_argument(
        "--sc_lfi_v4_align_eval_projection_mode",
        type=str,
        default="fixed",
        choices=["resample", "fixed"],
    )
    parser.add_argument("--sc_lfi_v4_align_eval_num_repeats", type=int, default=1)
    parser.add_argument("--sc_lfi_v4_align_projection_seed", type=int, default=11)
    parser.add_argument("--sc_lfi_v4_sinkhorn_epsilon", type=float, default=0.05)
    parser.add_argument("--sc_lfi_v4_sinkhorn_iterations", type=int, default=80)
    parser.add_argument("--sc_lfi_v4_sinkhorn_cost_power", type=float, default=2.0)
    parser.add_argument("--sc_lfi_v4_eps", type=float, default=1e-8)
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
    parser.add_argument("--hssw_token_weight_hidden", type=int, default=128)
    parser.add_argument("--hssw_token_weight_temperature", type=float, default=1.0)
    parser.add_argument("--hssw_token_weight_mix_alpha", type=float, default=1.0)
    parser.add_argument("--hssw_token_mass_reg_weight", type=float, default=0.0)
    parser.add_argument("--hssw_multiscale_grids", type=str, default="4,1")
    parser.add_argument("--hssw_multiscale_mass_budgets", type=str, default="0.6,0.3,0.1")
    parser.add_argument("--hssw_token_l2norm", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--hssw_weighted_sw", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--hssw_train_num_projections", type=int, default=32)
    parser.add_argument("--hssw_eval_num_projections", type=int, default=64)
    parser.add_argument("--hssw_sw_p", type=float, default=2.0)
    parser.add_argument(
        "--hssw_train_projection_mode",
        type=str,
        default="resample",
        choices=["resample", "fixed"],
    )
    parser.add_argument(
        "--hssw_eval_projection_mode",
        type=str,
        default="fixed",
        choices=["resample", "fixed"],
    )
    parser.add_argument("--hssw_eval_num_repeats", type=int, default=1)
    parser.add_argument("--hssw_projection_seed", type=int, default=7)
    parser.add_argument("--hssw_pairwise_chunk_size", type=int, default=0)
    parser.add_argument("--hssw_tau", type=float, default=0.2)
    parser.add_argument("--hssw_lambda_cons", type=float, default=0.1)
    parser.add_argument("--hssw_lambda_red", type=float, default=0.02)
    parser.add_argument("--hssw_learn_logit_scale", type=str, default="false", choices=["true", "false"])
    parser.add_argument("--hssw_logit_scale_init", type=float, default=10.0)
    parser.add_argument("--hssw_logit_scale_max", type=float, default=100.0)
    parser.add_argument("--hssw_eps", type=float, default=1e-6)
    parser.add_argument("--spif_stable_dim", type=int, default=64)
    parser.add_argument("--spif_variant_dim", type=int, default=64)
    parser.add_argument("--spif_gate_hidden", type=int, default=16)
    parser.add_argument("--spif_top_r", type=int, default=3)
    parser.add_argument("--spif_shot_top_r", type=int, default=4)
    parser.add_argument("--spif_asym_top_r", type=int, default=4)
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
    parser.add_argument(
        "--spif_shot_agg",
        type=str,
        default="softmax",
        choices=["pooled", "mean", "softmax"],
    )
    parser.add_argument("--spif_shot_softmax_beta", type=float, default=10.0)
    parser.add_argument("--spif_asym_local_dim", type=int, default=None)
    parser.add_argument(
        "--spif_asym_shot_agg",
        type=str,
        default="softmax",
        choices=["pooled", "mean", "softmax"],
    )
    parser.add_argument("--spif_asym_shot_softmax_beta", type=float, default=10.0)
    parser.add_argument(
        "--spif_asym_fusion_mode",
        type=str,
        default="margin_adaptive",
        choices=["fixed", "margin_adaptive"],
    )
    parser.add_argument("--spif_asym_fusion_kappa", type=float, default=8.0)
    parser.add_argument("--spif_asym_global_scale", type=float, default=1.0)
    parser.add_argument("--spif_asym_local_scale", type=float, default=1.0)
    parser.add_argument(
        "--spif_asym_local_score_mode",
        type=str,
        default="bidirectional",
        choices=["query_to_support", "bidirectional"],
    )
    parser.add_argument("--spif_asym_share_local_head", type=str, default="false", choices=["true", "false"])
    parser.add_argument("--aeb_hidden", "--spif_aeb_hidden", dest="aeb_hidden", type=int, default=None)
    parser.add_argument("--aeb_beta", "--spif_aeb_beta", dest="aeb_beta", type=float, default=1.0)
    parser.add_argument("--aeb_eps", "--spif_aeb_eps", dest="aeb_eps", type=float, default=1e-6)
    parser.add_argument(
        "--aeb_shot_agg",
        "--spif_aeb_shot_agg",
        dest="aeb_shot_agg",
        type=str,
        default="softmax",
        choices=["pooled", "mean", "softmax"],
    )
    parser.add_argument(
        "--aeb_shot_softmax_beta",
        "--spif_aeb_shot_softmax_beta",
        dest="aeb_shot_softmax_beta",
        type=float,
        default=10.0,
    )
    parser.add_argument("--aeb_v2_hidden", type=int, default=None)
    parser.add_argument("--aeb_v2_local_dim", type=int, default=None)
    parser.add_argument("--aeb_v2_min_budget", type=float, default=0.2)
    parser.add_argument("--aeb_v2_max_budget", type=float, default=0.85)
    parser.add_argument("--aeb_v2_rank_temperature", type=float, default=0.35)
    parser.add_argument("--aeb_v2_global_scale", type=float, default=1.0)
    parser.add_argument("--aeb_v2_local_scale", type=float, default=1.0)
    parser.add_argument("--aeb_v2_cond_global_weight", type=float, default=0.35)
    parser.add_argument("--aeb_v2_cond_global_temperature", type=float, default=8.0)
    parser.add_argument("--aeb_v2_cond_global_use_gate_prior", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--aeb_v2_query_class_attention_on", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--aeb_v2_query_class_attention_temperature", type=float, default=8.0)
    parser.add_argument(
        "--aeb_v2_query_class_attention_use_gate_prior",
        type=str,
        default="true",
        choices=["true", "false"],
    )
    parser.add_argument("--aeb_v2_anchor_top_r", type=int, default=4)
    parser.add_argument("--aeb_v2_anchor_mix", type=float, default=0.35)
    parser.add_argument(
        "--aeb_v2_local_mix_mode",
        type=str,
        default="prior_adaptive",
        choices=["fixed", "prior_adaptive"],
    )
    parser.add_argument("--aeb_v2_adaptive_mix_min", type=float, default=0.2)
    parser.add_argument("--aeb_v2_adaptive_mix_max", type=float, default=0.8)
    parser.add_argument("--aeb_v2_adaptive_mix_scale", type=float, default=6.0)
    parser.add_argument("--aeb_v2_local_residual_scale", type=float, default=0.25)
    parser.add_argument("--aeb_v2_budget_prior_scale", type=float, default=6.0)
    parser.add_argument("--aeb_v2_budget_residual_scale", type=float, default=0.15)
    parser.add_argument("--aeb_v2_budget_rank_weight", type=float, default=0.1)
    parser.add_argument("--aeb_v2_budget_rank_margin", type=float, default=0.05)
    parser.add_argument("--aeb_v2_budget_residual_reg_weight", type=float, default=0.05)
    parser.add_argument("--aeb_v2_support_coverage_temperature", type=float, default=8.0)
    parser.add_argument("--aeb_v2_coverage_bonus_weight", type=float, default=0.0)
    parser.add_argument("--aeb_v2_competition_weight", type=float, default=1.0)
    parser.add_argument("--aeb_v2_competition_temperature", type=float, default=8.0)
    parser.add_argument("--aeb_v2_local_margin_weight", type=float, default=0.0)
    parser.add_argument("--aeb_v2_local_margin_target", type=float, default=0.5)
    parser.add_argument("--aeb_v2_anchor_consistency_weight", type=float, default=0.02)
    parser.add_argument(
        "--aeb_v2_fusion_mode",
        type=str,
        default="fixed",
        choices=["fixed", "margin_adaptive", "residual"],
    )
    parser.add_argument("--aeb_v2_fusion_kappa", type=float, default=8.0)
    parser.add_argument(
        "--aeb_v2_local_score_mode",
        type=str,
        default="query_to_support",
        choices=["query_to_support", "bidirectional"],
    )
    parser.add_argument("--aeb_v2_share_local_head", type=str, default="false", choices=["true", "false"])
    parser.add_argument("--aeb_v2_query_gate_weighting", type=str, default="false", choices=["true", "false"])
    parser.add_argument("--aeb_v2_detach_budget_context", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--aeb_v2_detach_local_backbone", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--aeb_v2_global_ce_weight", type=float, default=0.0)
    parser.add_argument("--aeb_v2_local_ce_weight", type=float, default=0.05)
    parser.add_argument("--aeb_v2_eps", type=float, default=1e-6)
    parser.add_argument("--spifaeb_v3_use_structured_global", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--spifaeb_v3_sigma_min", type=float, default=0.05)
    parser.add_argument("--spifaeb_v3_use_reliability", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--spifaeb_v3_reliability_detach", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--spifaeb_v3_beta_align", type=float, default=1.0)
    parser.add_argument("--spifaeb_v3_beta_dev", type=float, default=0.5)
    parser.add_argument("--spifaeb_v3_beta_rel", type=float, default=0.2)
    parser.add_argument("--spifaeb_v3_learnable_betas", type=str, default="false", choices=["true", "false"])
    parser.add_argument("--spifaeb_v3_global_ce_weight", type=float, default=0.0)
    parser.add_argument("--spif_rdp_top_r", type=int, default=5)
    parser.add_argument("--spif_rdp_lambda_init", type=float, default=1e-3)
    parser.add_argument("--spif_rdp_alpha_init", type=float, default=1.0)
    parser.add_argument("--spif_rdp_tau_init", type=float, default=10.0)
    parser.add_argument("--spif_rdp_gamma_init", type=float, default=1.0)
    parser.add_argument("--spif_rdp_beta_init", "--spif_rdp_beta0_init", dest="spif_rdp_beta_init", type=float, default=0.5)
    parser.add_argument("--spif_rdp_variance_floor", type=float, default=1e-2)
    parser.add_argument("--spif_rdp_compact_loss_weight", type=float, default=0.1)
    parser.add_argument("--spif_rdp_sep_loss_weight", type=float, default=0.05)
    parser.add_argument("--spif_rdp_sep_margin", type=float, default=0.5)
    parser.add_argument("--spif_rdp_eps", type=float, default=1e-6)
    parser.add_argument("--spif_rdp_consistency_weight", type=float, default=0.0)
    parser.add_argument("--spif_rdp_decorr_weight", type=float, default=0.0)
    parser.add_argument("--spif_rdp_sparse_weight", type=float, default=0.0)
    parser.add_argument("--spif_rdp_fsl_mamba_base_dim", type=int, default=48)
    parser.add_argument("--spif_rdp_fsl_mamba_output_dim", type=int, default=320)
    parser.add_argument("--spif_rdp_fsl_mamba_drop_path", type=float, default=0.02)
    parser.add_argument("--spif_rdp_fsl_mamba_perturb_sigma", type=float, default=0.0)
    parser.add_argument("--spif_rdp_use_fsl_mamba_global_prior", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--spif_rdp_fsl_mamba_global_mix_init", type=float, default=0.35)
    parser.add_argument("--rada_feat_dim", type=int, default=None)
    parser.add_argument("--rada_tau_r", type=float, default=0.5)
    parser.add_argument("--rada_lambda_proto", type=float, default=0.7)
    parser.add_argument("--rada_gamma_disp", type=float, default=0.5)
    parser.add_argument("--rada_eps", type=float, default=1e-6)
    parser.add_argument("--rada_reliability_head", type=str, default="linear", choices=["linear", "mlp"])
    parser.add_argument("--rada_reliability_hidden_dim", type=int, default=16)
    parser.add_argument("--rada_l2_normalize", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--rada_entropy_reg_weight", type=float, default=0.0)
    parser.add_argument("--rada_use_reliability", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--rada_use_dispersion_metric", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--rada_query_conditioned", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--rada_use_residual_anchor", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--rada_use_shrinkage", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--rada_disp_clamp_max", type=float, default=0.0)
    parser.add_argument("--rada_use_evidence_bound", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--rada_evidence_temperature", type=float, default=1.0)
    parser.add_argument("--rada_min_reliability_mix", type=float, default=0.25)
    parser.add_argument("--rada_dispersion_inflation", type=float, default=0.25)
    parser.add_argument("--rada_fsl_mamba_base_dim", type=int, default=48)
    parser.add_argument("--rada_fsl_mamba_output_dim", type=int, default=320)
    parser.add_argument("--rada_fsl_mamba_drop_path", type=float, default=0.02)
    parser.add_argument("--rada_fsl_mamba_perturb_sigma", type=float, default=0.0)
    parser.add_argument("--spif_ota_transport_dim", type=int, default=None)
    parser.add_argument("--spif_ota_projector_hidden_dim", type=int, default=None)
    parser.add_argument("--spif_ota_mass_hidden_dim", type=int, default=None)
    parser.add_argument("--spif_ota_mass_temperature", type=float, default=1.0)
    parser.add_argument("--spif_ota_sinkhorn_epsilon", type=float, default=0.05)
    parser.add_argument("--spif_ota_sinkhorn_iterations", type=int, default=60)
    parser.add_argument("--spif_ota_shot_hidden_dim", type=int, default=32)
    parser.add_argument(
        "--spif_ota_shot_aggregation",
        type=str,
        default="learned",
        choices=["learned", "mean"],
    )
    parser.add_argument("--spif_ota_position_cost_weight", type=float, default=0.0)
    parser.add_argument("--spif_ota_use_column_bias", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--spif_ota_lambda_mass", type=float, default=0.0)
    parser.add_argument("--spif_ota_lambda_consistency", type=float, default=0.0)
    parser.add_argument("--spif_ota_eps", type=float, default=1e-6)
    parser.add_argument("--ebot_proj_dim", type=int, default=256)
    parser.add_argument("--ebot_reliability_hidden_dim", type=int, default=128)
    parser.add_argument("--ebot_reliability_dropout", type=float, default=0.1)
    parser.add_argument("--ebot_sinkhorn_epsilon", "--ebot_sinkhorn_eps", dest="ebot_sinkhorn_epsilon", type=float, default=0.05)
    parser.add_argument("--ebot_sinkhorn_iters", "--ebot_sinkhorn_iterations", dest="ebot_sinkhorn_iters", type=int, default=50)
    parser.add_argument("--ebot_dustbin_cost", type=float, default=0.7)
    parser.add_argument("--ebot_learnable_dustbin_cost", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--ebot_alpha_unmatched", type=float, default=0.10)
    parser.add_argument("--ebot_min_budget", type=float, default=0.15)
    parser.add_argument("--ebot_max_budget", type=float, default=0.95)
    parser.add_argument("--ebot_gate_temperature", type=float, default=1.0)
    parser.add_argument("--ebot_use_scalogram_priors", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--ebot_use_energy_prior", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--ebot_use_gradient_prior", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--ebot_use_tf_coords", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--ebot_log_power_alpha", type=float, default=10.0)
    parser.add_argument("--ebot_prior_norm", type=str, default="mean", choices=["mean", "zscore", "none"])
    parser.add_argument("--ebot_use_cross_reference", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--ebot_use_dustbin", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--ebot_use_evidence_budget", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--ebot_use_kshot_reweighting", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--ebot_lambda_score", type=float, default=1.0)
    parser.add_argument("--ebot_lambda_mass", type=float, default=0.5)
    parser.add_argument("--ebot_lambda_unmatched", type=float, default=0.5)
    parser.add_argument("--ebot_score_scale", type=float, default=12.5)
    parser.add_argument("--ebot_learnable_score_scale", type=str, default="false", choices=["true", "false"])
    parser.add_argument("--ebot_symmetric_matching", type=str, default="false", choices=["true", "false"])
    parser.add_argument("--ebot_use_uncertainty_prior", type=str, default="false", choices=["true", "false"])
    parser.add_argument("--ebot_use_aux_loss", type=str, default="false", choices=["true", "false"])
    parser.add_argument("--ebot_budget_floor", type=float, default=0.15)
    parser.add_argument("--ebot_budget_ceiling", type=float, default=0.95)
    parser.add_argument("--ebot_weight_budget_low", type=float, default=0.01)
    parser.add_argument("--ebot_weight_budget_high", type=float, default=0.001)
    parser.add_argument("--ebot_eps", type=float, default=1e-6)
    parser.add_argument("--spif_otccls_feature_dim", type=int, default=256)
    parser.add_argument("--spif_otccls_gate_hidden", type=int, default=128)
    parser.add_argument("--spif_otccls_num_projections", type=int, default=64)
    parser.add_argument("--spif_otccls_num_quantiles", type=int, default=100)
    parser.add_argument("--spif_otccls_tau_g_init", type=float, default=10.0)
    parser.add_argument("--spif_otccls_tau_l_init", type=float, default=1.0)
    parser.add_argument("--spif_otccls_beta_init", type=float, default=0.5)
    parser.add_argument("--spif_otccls_alpha_init", type=float, default=1.0)
    parser.add_argument("--spif_otccls_compact_loss_weight", type=float, default=0.1)
    parser.add_argument("--spif_otccls_decorr_loss_weight", type=float, default=0.05)
    parser.add_argument("--spif_otccls_entropy_loss_weight", type=float, default=0.01)
    parser.add_argument("--spif_otccls_shot_agg", type=str, default="softmax", choices=["pooled", "mean", "softmax"])
    parser.add_argument("--spif_otccls_shot_softmax_beta", type=float, default=10.0)
    parser.add_argument("--spif_otccls_swd_backend", type=str, default="pot", choices=["pot", "quantile", "auto"])
    parser.add_argument("--spif_otccls_eps", type=float, default=1e-6)
    parser.add_argument("--spif_otccls_backbone_drop_rate", type=float, default=0.05)
    parser.add_argument("--spif_otccls_backbone_dropblock_size", type=int, default=5)
    parser.add_argument("--spif_local_sw_num_projections", type=int, default=64)
    parser.add_argument("--spif_local_sw_p", type=float, default=2.0)
    parser.add_argument("--spif_local_sw_normalize", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--spif_local_sw_score_scale", type=float, default=1.0)
    parser.add_argument("--spif_papersw_train_num_projections", type=int, default=128)
    parser.add_argument("--spif_papersw_eval_num_projections", type=int, default=512)
    parser.add_argument("--spif_papersw_p", type=float, default=2.0)
    parser.add_argument("--spif_papersw_normalize_inputs", type=str, default="false", choices=["true", "false"])
    parser.add_argument(
        "--spif_papersw_train_projection_mode",
        type=str,
        default="resample",
        choices=["resample", "fixed"],
    )
    parser.add_argument(
        "--spif_papersw_eval_projection_mode",
        type=str,
        default="fixed",
        choices=["resample", "fixed"],
    )
    parser.add_argument("--spif_papersw_eval_num_repeats", type=int, default=1)
    parser.add_argument("--spif_papersw_score_scale", type=float, default=8.0)
    parser.add_argument("--spif_papersw_projection_seed", type=int, default=7)
    parser.add_argument(
        "--spif_papersw_shot_agg",
        type=str,
        default="pooled",
        choices=["pooled", "mean", "softmin"],
    )
    parser.add_argument("--spif_papersw_shot_softmin_beta", type=float, default=10.0)
    parser.add_argument("--spif_papersw_debug", type=str, default="false", choices=["true", "false"])
    parser.add_argument("--spif_harm_global_scale", type=float, default=16.0)
    parser.add_argument("--spif_harm_fusion_hidden_dim", type=int, default=32)
    parser.add_argument("--spif_harm_local_score_scale", type=float, default=8.0)
    parser.add_argument("--spif_harm_factor_consistency_weight", type=float, default=0.0)
    parser.add_argument("--spif_harm_factor_decorr_weight", type=float, default=0.0)
    parser.add_argument("--spif_harm_factor_sparse_weight", type=float, default=0.0)
    parser.add_argument("--spif_harm_factor_consistency_dropout", type=float, default=0.1)
    parser.add_argument("--spif_harm_final_sparse_weight", type=float, default=0.0)
    parser.add_argument("--spif_harm_global_ce_weight", type=float, default=0.0)
    parser.add_argument("--spif_harm_local_ce_weight", type=float, default=0.0)
    parser.add_argument("--spif_harm_beta_init", type=float, default=0.1)
    parser.add_argument("--spif_harm_learnable_beta", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--spif_harm_use_final_gate", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--spif_mamba_depth", type=int, default=1)
    parser.add_argument("--spif_mamba_ffn_multiplier", type=int, default=2)
    parser.add_argument("--spif_mamba_d_conv", type=int, default=4)
    parser.add_argument("--spif_mamba_expand", type=int, default=2)
    parser.add_argument("--spif_mamba_dropout", type=float, default=0.0)
    parser.add_argument("--spif_mamba_gate_position", type=str, default="after", choices=["after", "before"])
    parser.add_argument("--spif_adj_use_raw_gate", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--spif_adj_use_final_gate", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--spif_adj_beta_init", type=float, default=0.1)
    parser.add_argument("--spif_adj_learnable_beta", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--spif_adj_consistency_weight", type=float, default=0.1)
    parser.add_argument("--spif_adj_decorr_weight", type=float, default=0.01)
    parser.add_argument("--spif_adj_sparse_raw_weight", type=float, default=5e-4)
    parser.add_argument("--spif_adj_sparse_final_weight", type=float, default=5e-4)
    parser.add_argument("--spif_adj_consistency_dropout", type=float, default=0.1)
    parser.add_argument("--spifv2_shared_dim", type=int, default=64)
    parser.add_argument("--spifv2_stable_dim", type=int, default=32)
    parser.add_argument("--spifv2_variant_dim", type=int, default=64)
    parser.add_argument("--spifv2_gate_hidden", type=int, default=None)
    parser.add_argument("--spifv2_top_r", type=int, default=3)
    parser.add_argument("--spifv2_gate_type", type=str, default="sparsemax", choices=["softmax", "sparsemax", "sigmoid"])
    parser.add_argument("--spifv2_use_shared_bottleneck", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--spifv2_mutual_local", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--spifv2_fusion_mode", type=str, default="residual", choices=["residual", "linear"])
    parser.add_argument("--spifv2_lambda_local_init", type=float, default=0.2)
    parser.add_argument("--spifv2_learnable_lambda_local", type=str, default=None, choices=["true", "false"])
    parser.add_argument("--spifv2_alpha_init", type=float, default=0.7)
    parser.add_argument("--spifv2_learnable_alpha", type=str, default=None, choices=["true", "false"])
    parser.add_argument("--spifv2_global_only", type=str, default="false", choices=["true", "false"])
    parser.add_argument("--spifv2_local_only", type=str, default="false", choices=["true", "false"])
    parser.add_argument("--spifv2_token_l2norm", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--spifv2_consistency_weight", type=float, default=0.05)
    parser.add_argument("--spifv2_decorr_weight", type=float, default=0.005)
    parser.add_argument("--spifv2_gate_reg_weight", type=float, default=5e-4)
    parser.add_argument("--spifv2_consistency_dropout", type=float, default=0.1)
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--spifv3_temperature", type=float, default=16.0)
    parser.add_argument("--num_atoms", type=int, default=8)
    parser.add_argument("--num_sw_projections", type=int, default=64)
    parser.add_argument("--assignment_temperature", type=float, default=1.0)
    parser.add_argument("--cstn_set_pool_heads", type=int, default=4)
    parser.add_argument("--use_atom_entropy_reg", type=str, default="false", choices=["true", "false"])
    parser.add_argument("--atom_entropy_reg_weight", type=float, default=0.01)
    parser.add_argument("--use_support_consistency_reg", type=str, default="false", choices=["true", "false"])
    parser.add_argument("--support_consistency_reg_weight", type=float, default=0.01)
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
    parser.add_argument("--hrot_token_dim", type=int, default=None)
    parser.add_argument("--hrot_use_raw_backbone_tokens", type=str, default="false", choices=["true", "false"])
    parser.add_argument(
        "--hrot_variant",
        type=str,
        default="E",
        choices=[
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
            "I",
            "J",
            "J_EGTW",
            "JE",
            "J_ECOT",
            "J_ECOT_M2",
            "J_ECOT_NNCS",
            "J_ECOT_CARE",
            "CARE",
            "CARE_UOT",
            "CARE-UOT",
            "CP_ECOT",
            "J_NCET",
            "NCET",
            "J_HLM",
            "K",
            "L",
            "M",
            "N",
            "O",
            "P",
            "Q",
            "R",
            "S",
            "T",
            "U",
            "V",
        ],
    )
    parser.add_argument("--hrot_eam_hidden_dim", type=int, default=256)
    parser.add_argument("--hrot_curvature_init", type=float, default=1.0)
    parser.add_argument("--hrot_projection_scale", type=float, default=0.1)
    parser.add_argument("--hrot_token_temperature", type=float, default=0.1)
    parser.add_argument("--hrot_score_scale", type=float, default=16.0)
    parser.add_argument("--hrot_tau_q", type=float, default=0.5)
    parser.add_argument("--hrot_tau_c", type=float, default=0.5)
    parser.add_argument("--hrot_sinkhorn_epsilon", type=float, default=0.1)
    parser.add_argument(
        "--hrot_sinkhorn_iterations",
        "--hrot_sinkhorn_iters",
        dest="hrot_sinkhorn_iterations",
        type=int,
        default=60,
    )
    parser.add_argument("--hrot_sinkhorn_tolerance", type=float, default=1e-5)
    parser.add_argument("--hrot_fixed_mass", type=float, default=0.8)
    parser.add_argument("--hrot_egtw_tau", type=float, default=1.0)
    parser.add_argument("--hrot_egtw_lambda", type=float, default=1.0)
    parser.add_argument("--hrot_egtw_eps", type=float, default=1e-8)
    parser.add_argument("--hrot_egtw_attention_temperature", type=float, default=0.2)
    parser.add_argument("--hrot_egtw_support_similarity_weight", type=float, default=0.5)
    parser.add_argument("--hrot_egtw_uniform_mix", type=float, default=0.25)
    parser.add_argument("--hrot_egtw_detach_masses", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--hrot_egtw_learn_tau", type=str, default="false", choices=["true", "false"])
    parser.add_argument("--hrot_egtw_learn_lambda", type=str, default="false", choices=["true", "false"])
    parser.add_argument(
        "--hrot_pre_transport_shot_pool",
        "--ecot_pre_transport_shot_pool",
        dest="hrot_pre_transport_shot_pool",
        type=str,
        default="false",
        choices=["true", "false"],
    )
    parser.add_argument(
        "--hrot_tsw_enable",
        type=str,
        default="false",
        choices=["true", "false"],
        help="Enable legacy Token Saliency Reweighting on the HROT cost matrix; use only for ablation/negative control",
    )
    parser.add_argument(
        "--hrot_tsw_share_gate",
        type=str,
        default="true",
        choices=["true", "false"],
        help="Share the TSW gate between query and support tokens",
    )
    parser.add_argument(
        "--hrot_ecot_rho_bank",
        "--ecot_rho_bank",
        "--ec_mrot_mass_response_grid",
        "--ec_mrot_response_grid",
        dest="hrot_ecot_rho_bank",
        type=str,
        default=None,
    )
    parser.add_argument("--hrot_ecot_base_rho", "--ec_mrot_base_budget", dest="hrot_ecot_base_rho", type=float, default=None)
    parser.add_argument("--hrot_ecot_budget_tau", "--ec_mrot_budget_tau", dest="hrot_ecot_budget_tau", type=float, default=1.0)
    parser.add_argument("--hrot_ecot_max_lambda", type=float, default=1.0)
    parser.add_argument(
        "--hrot_ecot_lambda_init",
        "--ecot_lambda_init",
        dest="hrot_ecot_lambda_init",
        type=float,
        default=-8.0,
    )
    parser.add_argument("--hrot_ecot_controller_hidden", type=int, default=32)
    parser.add_argument(
        "--hrot_ecot_uniform_budget_policy",
        "--ecot_uniform_budget_policy",
        dest="hrot_ecot_uniform_budget_policy",
        type=str,
        default="false",
        choices=["true", "false"],
    )
    parser.add_argument(
        "--hrot_ecot_enable_tau_shot",
        "--ecot_enable_tau_shot",
        dest="hrot_ecot_enable_tau_shot",
        type=str,
        default="true",
        choices=["true", "false"],
    )
    parser.add_argument("--hrot_ecot_tau_shot_min", type=float, default=0.5)
    parser.add_argument("--hrot_ecot_tau_shot_max", type=float, default=2.0)
    parser.add_argument("--hrot_ecot_enable_threshold_offset", type=str, default="false", choices=["true", "false"])
    parser.add_argument(
        "--hrot_ecot_m2_ablate_threshold_mass",
        "--m2_ablate_T",
        type=str,
        default="true",
        choices=["true", "false"],
        help="J_ECOT_M2 only: score uses -transport_cost only (zeros the T * mass term); default on.",
    )
    parser.add_argument(
        "--hrot_ecot_m2_cost_per_mass_score",
        type=str,
        default="false",
        choices=["true", "false"],
        help=(
            "J_ECOT_M2 only: replace T*m - cost with -alpha * cost / (mass + hrot_eps) "
            "(average transport cost per unit mass; default off; requires --hrot_ecot_m2_ablate_threshold_mass false; "
            "mutually exclusive with --hrot_ecot_m2_ablate_threshold_mass). "
            "Logits are ~1/rho larger in magnitude than T*m-cost for typical rho; consider lowering --hrot_score_scale if softmax saturates."
        ),
    )
    parser.add_argument(
        "--hrot_ecot_m2_cost_per_mass_alpha",
        type=float,
        default=1.0,
        help="J_ECOT_M2 only: alpha in -alpha * transport_cost / (transported_mass + eps) when cost-per-mass scoring is enabled.",
    )
    parser.add_argument(
        "--hrot_ecot_m2_cost_per_mass_detach_mass",
        type=str,
        default="true",
        choices=["true", "false"],
        help=(
            "J_ECOT_M2 only (with cost-per-mass score): use transported_mass.detach() in the denominator so gradients "
            "do not shrink mass toward eps; default true."
        ),
    )
    parser.add_argument(
        "--hrot_ecot_m2_use_swts",
        type=str,
        default="false",
        choices=["true", "false"],
        help=(
            "J_ECOT_M2 mass-removed score only: Self-Weighted Transport Score from transport plan P and cost D "
            "(requires ablate T*m and not cost-per-mass)."
        ),
    )
    parser.add_argument(
        "--hrot_ecot_m2_swts_temp",
        type=float,
        default=1.0,
        help="J_ECOT_M2 SWTS: softmax temperature over query marginals q(r); large values recover uniform weights (~ -C).",
    )
    parser.add_argument(
        "--hrot_ecot_identity_reg",
        "--ecot_identity_reg",
        dest="hrot_ecot_identity_reg",
        type=float,
        default=1e-4,
    )
    parser.add_argument("--hrot_ecot_policy_entropy_reg", type=float, default=1e-3)
    parser.add_argument("--hrot_ecot_consensus_tau_mode", type=str, default="fixed", choices=["fixed", "sqrt"])
    parser.add_argument("--hrot_ecot_consensus_tau", type=float, default=1.0)
    parser.add_argument(
        "--hrot_ecot_enable_nncs_marginal",
        type=str,
        default=None,
        choices=["true", "false"],
    )
    parser.add_argument("--hrot_ecot_nncs_beta", type=float, default=5.0)
    parser.add_argument("--hrot_ecot_nncs_eta", type=float, default=0.05)
    parser.add_argument("--hrot_ecot_nncs_detach", type=str, default="true", choices=["true", "false"])
    parser.add_argument(
        "--hrot_ecot_nncs_distance",
        type=str,
        default="auto",
        choices=["auto", "euclidean", "cosine"],
    )
    parser.add_argument("--hrot_ecot_nncs_min_sigma", type=float, default=1e-6)
    parser.add_argument(
        "--hrot_ecot_enable_crs_marginal",
        type=str,
        default="false",
        choices=["true", "false"],
    )
    parser.add_argument(
        "--hrot_ecot_crs_use_cross_ref",
        type=str,
        default="true",
        choices=["true", "false"],
    )
    parser.add_argument(
        "--hrot_ecot_crs_use_ssm",
        type=str,
        default="true",
        choices=["true", "false"],
    )
    parser.add_argument("--hrot_ecot_crs_eta_init", type=float, default=0.30)
    parser.add_argument("--hrot_ecot_crs_lambda_cr_init", type=float, default=0.50)
    parser.add_argument("--hrot_ecot_crs_tau_ssm_init", type=float, default=0.70)
    parser.add_argument("--hrot_ecot_crs_entropy_reg", type=float, default=0.0)
    parser.add_argument(
        "--hrot_ecot_crs_side",
        type=str,
        default="support",
        choices=["support", "both"],
    )
    parser.add_argument(
        "--hrot_ecot_crs_ssm_type",
        type=str,
        default="auto",
        choices=["auto", "simple", "mamba"],
    )
    parser.add_argument(
        "--hrot_ecot_enable_mea_marginal",
        type=str,
        default="false",
        choices=["true", "false"],
    )
    parser.add_argument("--hrot_ecot_mea_eta_init", type=float, default=0.35)
    parser.add_argument("--hrot_ecot_mea_temperature_init", type=float, default=0.70)
    parser.add_argument("--hrot_ecot_mea_entropy_reg", type=float, default=0.0)
    parser.add_argument(
        "--hrot_ecot_enable_ccdm_marginal",
        type=str,
        default="false",
        choices=["true", "false"],
    )
    parser.add_argument("--hrot_ecot_ccdm_tau_q_init", type=float, default=0.50)
    parser.add_argument("--hrot_ecot_ccdm_tau_b_init", type=float, default=0.50)
    parser.add_argument("--hrot_ecot_ccdm_entropy_reg", type=float, default=0.0)
    parser.add_argument("--hrot_ecot_ccdm_entropy_shot_weight", type=float, default=0.0)
    parser.add_argument(
        "--hrot_ecot_transport_mode",
        type=str,
        default=None,
        choices=["unbalanced", "uot", "unbalanced_ot", "balanced", "ot", "balanced_ot", "full_ot"],
        help="Transport family for ECOT/M2 ablations; M2 uses UOT and m2_full_ot uses balanced full OT by default.",
    )
    parser.add_argument(
        "--hrot_ecot_enable_noise_sink",
        type=str,
        default="false",
        choices=["true", "false"],
    )
    parser.add_argument("--hrot_ecot_noise_sink_cost_init", type=float, default=1.0)
    parser.add_argument("--hrot_ecot_noise_sink_score_penalty", type=float, default=0.0)
    parser.add_argument(
        "--hrot_ecot_episode_feature_normalize",
        type=str,
        default="false",
        choices=["true", "false"],
        help="J_ECOT_M2 only: z-score projected tokens over the full episode (query+support) before L2; default off.",
    )
    parser.add_argument(
        "--hrot_ecot_episode_feature_norm_eps",
        type=float,
        default=1e-6,
        help="Epsilon added to per-dim std for J_ECOT_M2 episode token normalization.",
    )
    parser.add_argument(
        "--hrot_amp_marginals",
        type=str,
        default="false",
        choices=["true", "false"],
    )
    parser.add_argument("--hrot_amp_marginals_tau", type=float, default=1.0)
    parser.add_argument(
        "--hrot_token_center",
        type=str,
        default="false",
        choices=["true", "false"],
    )
    parser.add_argument(
        "--hrot_token_center_query",
        type=str,
        default="true",
        choices=["true", "false"],
    )
    parser.add_argument("--care_enable_fwec", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--care_enable_qesm", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--care_enable_mdr", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--care_fwec_eps", type=float, default=1e-6)
    parser.add_argument("--care_fwec_w_clamp_min", type=float, default=0.1)
    parser.add_argument("--care_fwec_w_clamp_max", type=float, default=10.0)
    parser.add_argument("--care_qesm_tau_e", type=float, default=1.0)
    parser.add_argument("--care_mdr_margin", type=float, default=0.05)
    parser.add_argument("--care_mdr_lambda", type=float, default=0.1)
    parser.add_argument("--hrot_ncet_mix_init", type=float, default=0.25)
    parser.add_argument("--hrot_ncet_real_penalty_init", type=float, default=0.25)
    parser.add_argument("--hrot_ncet_null_penalty_init", type=float, default=0.05)
    parser.add_argument("--hrot_ncet_sink_cost_init", type=float, default=1.0)
    parser.add_argument(
        "--ec_mrot_response_mode",
        type=str,
        default="anchor_free_functional",
        choices=["anchor_free_functional", "counterfactual_residual", "discrete_quadrature"],
    )
    parser.add_argument("--ec_mrot_learn_response_grid", type=str, default="false", choices=["true", "false"])
    parser.add_argument("--ec_mrot_budget_prior", type=str, default="uniform", choices=["uniform", "base_anchored"])
    parser.add_argument("--ec_mrot_budget_kl_reg", type=float, default=0.0)
    parser.add_argument("--ec_mrot_budget_entropy_reg", type=float, default=0.0)
    parser.add_argument("--ec_mrot_homotopy_schedule", type=str, default="none", choices=["none", "linear", "cosine"])
    parser.add_argument("--ec_mrot_grid_spacing_reg", type=float, default=0.0)
    parser.add_argument("--ec_mrot_response_strength_init", type=float, default=0.35)
    parser.add_argument("--ec_mrot_response_strength_max", type=float, default=1.0)
    parser.add_argument("--ec_mrot_response_temperature", type=float, default=1.0)
    parser.add_argument("--ec_mrot_competitive_center", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--ec_mrot_local_response_gain", type=float, default=2.0)
    parser.add_argument("--ec_mrot_episode_prior_gain", type=float, default=0.10)
    parser.add_argument("--ec_mrot_margin_temperature", type=float, default=1.0)
    parser.add_argument("--ec_mrot_stability_temperature", type=float, default=1.0)
    parser.add_argument("--ec_mrot_residual_gate_floor", type=float, default=0.15)
    # Deprecated compatibility only: old J_HLM configs may still pass these,
    # but they no longer activate learned shot or token mass.
    parser.add_argument("--hrot_hlm_min_mass", type=float, default=0.1)
    parser.add_argument("--hrot_hlm_init_mass", type=float, default=0.8)
    parser.add_argument(
        "--hrot_hlm_budget_mode",
        type=str,
        default="cost",
        choices=["geodesic", "cost", "hybrid"],
    )
    parser.add_argument(
        "--hrot_hlm_token_mode",
        type=str,
        default="cost",
        choices=["uniform", "cost", "hybrid"],
    )
    parser.add_argument("--hrot_hlm_token_tau", type=float, default=0.25)
    parser.add_argument("--hrot_hlm_budget_hidden_dim", type=int, default=64)
    parser.add_argument("--hrot_hlm_token_hidden_dim", type=int, default=64)
    parser.add_argument("--hrot_hlm_detach_cost_features", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--hrot_hlm_allow_mass_grad", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--hrot_hlm_lambda_rho", type=float, default=0.0)
    parser.add_argument("--hrot_hlm_lambda_eff", type=float, default=0.0)
    parser.add_argument("--hrot_hlm_eff_margin", type=float, default=0.05)
    parser.add_argument("--hrot_hlm_lambda_shot_cov", type=float, default=0.0)
    parser.add_argument("--hrot_hlm_shot_entropy_floor", type=float, default=0.35)
    parser.add_argument("--hrot_hlm_return_diagnostics", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--hrot_min_mass", type=float, default=0.1)
    parser.add_argument("--hrot_mass_bonus_init", type=float, default=1.0)
    parser.add_argument(
        "--hrot_transport_cost_threshold_init",
        "--jecot_m2_T_init",
        type=float,
        default=None,
        dest="hrot_transport_cost_threshold_init",
        metavar="T0",
        help=(
            "Initial learned transport cost threshold T (positive). Used by HROT variants with cost-threshold "
            "scoring, including J-ECOT-M2 (default: T*m term ablated unless --hrot_ecot_m2_ablate_threshold_mass false; "
            "cost/mass score via --hrot_ecot_m2_cost_per_mass_score true). "
            "Stored as softplus(raw). "
            "If omitted, T0 = hrot_mass_bonus_init / hrot_score_scale (defaults 1.0/16 ≈ 0.0625)."
        ),
    )
    parser.add_argument("--hrot_lambda_rho", type=float, default=0.01)
    parser.add_argument("--hrot_rho_target", type=float, default=0.8)
    parser.add_argument("--hrot_lambda_rho_rank", type=float, default=0.05)
    parser.add_argument("--hrot_rho_rank_margin", type=float, default=0.05)
    parser.add_argument("--hrot_rho_rank_temperature", type=float, default=0.05)
    parser.add_argument("--hrot_lambda_curvature", type=float, default=0.0)
    parser.add_argument("--hrot_min_curvature", type=float, default=0.05)
    parser.add_argument("--hrot_structure_cost_init", type=float, default=0.05)
    parser.add_argument("--hrot_ground_cost", type=str, default="auto", choices=["auto", "euclidean", "cosine"])
    parser.add_argument("--hrot_eam_mode", type=str, default="compact", choices=["legacy", "compact"])
    parser.add_argument("--hrot_compact_eam_prior_mix", type=float, default=0.5)
    parser.add_argument("--hrot_normalize_euclidean_tokens", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--hrot_normalize_rho", type=str, default="false", choices=["true", "false"])
    parser.add_argument("--hrot_eval_use_float64", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--hrot_hyperbolic_backend", type=str, default="auto", choices=["auto", "geoopt", "native"])
    parser.add_argument(
        "--hrot_ot_backend",
        type=str,
        default="native",
        choices=[
            "auto",
            "native",
            "pot",
            "partial",
            "partial_ot",
            "partial_sinkhorn",
            "partial_pot",
            "pot_partial",
            "partial_exact",
            "exact_partial",
        ],
    )
    parser.add_argument("--hrot_eps", type=float, default=1e-6)
    parser.add_argument(
        "--jecot_m2_val_query_csv",
        type=str,
        default="",
        help="J-ECOT-M2 only: if non-empty, append per-query validation diagnostics to this CSV path on each evaluate() call.",
    )
    parser.add_argument(
        "--jecot_m2_val_save_hist_fig",
        type=str,
        default="true",
        choices=["true", "false"],
        help="J-ECOT-M2 only: save true_avg_cost histogram with threshold line under path_results each validation epoch.",
    )
    parser.add_argument("--crj_token_dim", type=int, default=None)
    parser.add_argument("--crj_use_raw_backbone_tokens", type=str, default=None, choices=["true", "false"])
    parser.add_argument("--crj_pool_gamma", type=float, default=0.65)
    parser.add_argument("--crj_variance_penalty", type=float, default=0.25)
    parser.add_argument("--crj_min_shots", type=int, default=2)
    parser.add_argument("--crj_trim_fraction", type=float, default=0.0)
    parser.add_argument("--crj_clip_logit", type=float, default=0.0)
    parser.add_argument("--fgwuot_token_dim", type=int, default=128)
    parser.add_argument("--fgwuot_tau", type=float, default=0.5)
    parser.add_argument("--fgwuot_eps_sinkhorn", type=float, default=0.1)
    parser.add_argument("--fgwuot_fgw_iters", type=int, default=8)
    parser.add_argument("--fgwuot_sinkhorn_iters", type=int, default=60)
    parser.add_argument("--fgwuot_sinkhorn_tol", type=float, default=1e-5)
    parser.add_argument("--fgwuot_alpha_init", type=float, default=0.5)
    parser.add_argument("--fgwuot_score_scale_init", type=float, default=16.0)
    parser.add_argument("--fgwuot_rho_head_hidden", type=int, default=32)
    parser.add_argument("--fgwuot_lambda_rho", type=float, default=0.01)
    parser.add_argument("--fgwuot_rho_target", type=float, default=0.8)
    parser.add_argument("--fgwuot_normalize_tokens", type=str, default="true", choices=["true", "false"])
    parser.add_argument(
        "--fgwuot_shot_aggregation",
        type=str,
        default="j_logmeanexp",
        choices=["mean", "j_logmeanexp"],
    )
    parser.add_argument("--fgwuot_eps", type=float, default=1e-8)
    parser.add_argument("--jsc_wdro_token_dim", type=int, default=None)
    parser.add_argument("--jsc_wdro_score_scale", type=float, default=16.0)
    parser.add_argument("--jsc_wdro_sinkhorn_epsilon", type=float, default=0.05)
    parser.add_argument("--jsc_wdro_sinkhorn_iterations", type=int, default=80)
    parser.add_argument("--jsc_wdro_sinkhorn_tolerance", type=float, default=1e-5)
    parser.add_argument("--jsc_wdro_barycenter_iterations", type=int, default=40)
    parser.add_argument("--jsc_wdro_barycenter_tolerance", type=float, default=1e-5)
    parser.add_argument(
        "--jsc_wdro_barycenter_transport",
        type=str,
        default="unbalanced",
        choices=["balanced", "unbalanced"],
    )
    parser.add_argument("--jsc_wdro_barycenter_tau", type=float, default=0.5)
    parser.add_argument("--jsc_wdro_barycenter_method", type=str, default="sinkhorn")
    parser.add_argument("--jsc_wdro_tau_q", type=float, default=0.5)
    parser.add_argument("--jsc_wdro_tau_c", type=float, default=0.5)
    parser.add_argument("--jsc_wdro_query_transport", type=str, default="balanced", choices=["balanced", "unbalanced"])
    parser.add_argument("--jsc_wdro_epsilon_alpha_init", type=float, default=0.05)
    parser.add_argument("--jsc_wdro_epsilon_beta_init", type=float, default=0.25)
    parser.add_argument("--jsc_wdro_epsilon_floor_init", type=float, default=1e-4)
    parser.add_argument("--jsc_wdro_learn_epsilon", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--jsc_wdro_epsilon_dimension", type=int, default=None)
    parser.add_argument("--jsc_wdro_epsilon_reg_weight", type=float, default=0.0)
    parser.add_argument("--jsc_wdro_normalize_tokens", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--jsc_wdro_cost_power", type=float, default=2.0)
    parser.add_argument("--jsc_wdro_normalize_unbalanced_cost", type=str, default="true", choices=["true", "false"])
    parser.add_argument(
        "--jsc_wdro_unbalanced_score_mode",
        type=str,
        default="uot_objective",
        choices=["uot_objective", "normalized_transport", "transport"],
    )
    parser.add_argument(
        "--jsc_wdro_token_weighting",
        type=str,
        default="uniform",
        choices=["uniform", "cross_reference"],
    )
    parser.add_argument("--jsc_wdro_token_weight_temperature", type=float, default=0.25)
    parser.add_argument("--jsc_wdro_use_competitive_diagnostics", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--jsc_wdro_competitive_temperature", type=float, default=0.1)
    parser.add_argument("--jsc_wdro_profile", type=str, default="false", choices=["true", "false"])
    parser.add_argument("--jsc_wdro_ot_backend", type=str, default="pot", choices=["auto", "native", "pot"])
    parser.add_argument("--jsc_wdro_eps", type=float, default=1e-8)
    parser.add_argument("--ubt_fsl_token_dim", type=int, default=None)
    parser.add_argument("--ubt_fsl_sinkhorn_epsilon", type=float, default=None)
    parser.add_argument("--ubt_fsl_sinkhorn_iterations", type=int, default=None)
    parser.add_argument("--ubt_fsl_sinkhorn_tolerance", type=float, default=None)
    parser.add_argument("--ubt_fsl_barycenter_iterations", type=int, default=None)
    parser.add_argument("--ubt_fsl_barycenter_tolerance", type=float, default=None)
    parser.add_argument("--ubt_fsl_barycenter_method", type=str, default=None)
    parser.add_argument("--ubt_fsl_alpha", type=float, default=None)
    parser.add_argument("--ubt_fsl_beta", type=float, default=None)
    parser.add_argument("--ubt_fsl_tau", type=float, default=None)
    parser.add_argument("--ubt_fsl_query_competition_temperature", type=float, default=None)
    parser.add_argument("--ubt_fsl_rho", type=float, default=None)
    parser.add_argument("--ubt_fsl_score_scale", type=float, default=None)
    parser.add_argument("--ubt_fsl_normalize_tokens", type=str, default=None, choices=["true", "false"])
    parser.add_argument("--ubt_fsl_cost_power", type=float, default=None)
    parser.add_argument("--ubt_fsl_profile", type=str, default=None, choices=["true", "false"])
    parser.add_argument(
        "--ubt_fsl_transport_backend",
        type=str,
        default=None,
        choices=["balanced", "balanced_ot", "unbalanced", "unbalanced_ot", "uot"],
    )
    parser.add_argument(
        "--ubt_fsl_scoring",
        type=str,
        default=None,
        choices=["robust_transport_envelope", "negative_transport_distance", "raw_ot_score"],
    )
    parser.add_argument("--ubt_fsl_use_query_competition", type=str, default=None, choices=["true", "false"])
    parser.add_argument(
        "--ubt_fsl_ablation_mode",
        type=str,
        default=None,
        choices=[
            "deepemd_pairwise",
            "barycentric_only",
            "uncertain_barycentric",
            "ubt_full",
            "fixed_radius",
            "dynamic_radius",
            "balanced_ot",
            "unbalanced_ot",
            "no_beta_floor",
            "tau_sensitivity",
            "alpha_sensitivity",
        ],
    )
    parser.add_argument("--ubt_fsl_solver_backend", type=str, default=None, choices=["auto", "native", "pot"])
    parser.add_argument("--ubt_fsl_ot_backend", type=str, default=None, choices=["auto", "native", "pot"])
    parser.add_argument("--ubt_fsl_eps", type=float, default=None)
    parser.add_argument("--cbcr_fsl_token_dim", type=int, default=None)
    parser.add_argument("--cbcr_fsl_sinkhorn_epsilon", type=float, default=None)
    parser.add_argument("--cbcr_fsl_sinkhorn_iterations", type=int, default=None)
    parser.add_argument("--cbcr_fsl_sinkhorn_tolerance", type=float, default=None)
    parser.add_argument("--cbcr_fsl_barycenter_iterations", type=int, default=None)
    parser.add_argument("--cbcr_fsl_barycenter_tolerance", type=float, default=None)
    parser.add_argument("--cbcr_fsl_barycenter_method", type=str, default=None)
    parser.add_argument("--cbcr_fsl_alpha", type=float, default=None)
    parser.add_argument("--cbcr_fsl_beta", type=float, default=None)
    parser.add_argument("--cbcr_fsl_tau", type=float, default=None)
    parser.add_argument("--cbcr_fsl_rho", type=float, default=None)
    parser.add_argument("--cbcr_fsl_score_scale", type=float, default=None)
    parser.add_argument("--cbcr_fsl_normalize_tokens", type=str, default=None, choices=["true", "false"])
    parser.add_argument("--cbcr_fsl_cost_power", type=float, default=None)
    parser.add_argument("--cbcr_fsl_profile", type=str, default=None, choices=["true", "false"])
    parser.add_argument("--cbcr_fsl_ot_backend", type=str, default=None, choices=["auto", "native", "pot"])
    parser.add_argument(
        "--cbcr_fsl_transport_backend",
        type=str,
        default=None,
        choices=["balanced", "balanced_ot", "unbalanced", "unbalanced_ot", "uot"],
    )
    parser.add_argument("--cbcr_fsl_scoring", type=str, default=None)
    parser.add_argument("--cbcr_fsl_use_query_competition", type=str, default=None, choices=["true", "false"])
    parser.add_argument("--cbcr_fsl_ablation_mode", type=str, default=None)
    parser.add_argument("--cbcr_fsl_eps", type=float, default=None)

    parser.add_argument(
        "--training_samples",
        type=int,
        default=None,
        help="Total training samples (must be divisible by way_num)",
    )
    parser.add_argument("--episode_num_train", type=int, default=200)
    parser.add_argument("--episode_num_val", type=int, default=300)
    parser.add_argument("--episode_num_test", type=int, default=400)
    parser.add_argument(
        "--test_protocol",
        type=str,
        default="auto",
        choices=["auto", "clean", "robust", "noise"],
        help="Final-test protocol. auto uses --noise_test_root first, then robust pools, otherwise clean test.",
    )
    parser.add_argument(
        "--noise_test_root",
        type=str,
        default=None,
        help="Root containing external SNR test folders such as test_snr5db_rf_1_15mhz.",
    )
    parser.add_argument(
        "--noise_test_splits",
        type=str,
        default="auto",
        help="Comma-separated SNR split folders to test, or auto to use all test_snr* folders.",
    )
    parser.add_argument(
        "--model_selection_split",
        type=str,
        default="val",
        choices=["val", "test"],
        help="Split used for per-epoch model selection and best-checkpoint tracking.",
    )
    parser.add_argument(
        "--merge_val_into_train",
        type=str,
        default="false",
        choices=["true", "false"],
        help="If true, concatenate the current val split into the train pool before episodic sampling.",
    )
    parser.add_argument(
        "--train_episode_seed_offset",
        type=int,
        default=0,
        help="Offset added to --seed before deriving epoch-specific train episode seeds.",
    )
    parser.add_argument(
        "--selection_episode_seed_offset",
        type=int,
        default=1,
        help="Offset added to --seed before deriving epoch-specific selection episode seeds.",
    )
    parser.add_argument(
        "--selection_episode_seed_mode",
        type=str,
        default="per_epoch",
        choices=["fixed", "per_epoch"],
        help="Use one fixed selection seed for all epochs or advance it with epoch index.",
    )
    parser.add_argument(
        "--final_test_episode_seed_offset",
        type=int,
        default=0,
        help="Offset added to --seed for the final fixed test episode set.",
    )
    parser.add_argument(
        "--final_test_seed",
        type=int,
        default=None,
        help=(
            "Absolute seed for final-test episode sampling. "
            "When set, this overrides --seed + --final_test_episode_seed_offset."
        ),
    )
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
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["step", "cosine"])
    parser.add_argument("--step_size", type=int, default=10, help="StepLR decay interval in epochs")
    parser.add_argument("--gamma", type=float, default=0.5, help="StepLR multiplicative decay factor")
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--warmup_start_factor", type=float, default=0.1)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument(
        "--classification_loss",
        type=str,
        default="cross_entropy",
        choices=["cross_entropy", "repo_contrastive"],
        help="Main classification loss. 'repo_contrastive' matches the CE-equivalent ContrastiveLoss used by the referenced Mahalanobis repos.",
    )
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
    parser.add_argument("--grad_clip", type=float, default=0.0)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--save_misclf_report", action="store_true")
    parser.add_argument("--misclf_topk", type=int, default=30)
    parser.add_argument("--export_q1_report", type=str, default="false", choices=["true", "false"])
    parser.add_argument("--q1_num_episodes", type=int, default=8)
    parser.add_argument("--q1_queries_per_episode", type=int, default=1)
    parser.add_argument("--q1_misclassified_only", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--export_dataset_profile", type=str, default="false", choices=["true", "false"])
    parser.add_argument("--save_last_checkpoint", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--skip_final_test", type=str, default="false", choices=["true", "false"])
    parser.add_argument("--deepemd_train_sfc", type=str, default="false", choices=["true", "false"])
    parser.add_argument("--deepemd_train_sfc_update_step", type=int, default=None)
    parser.add_argument("--deepemd_train_sfc_bs", type=int, default=None)
    parser.add_argument("--deepemd_fast_val", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--deepemd_test_exact", type=str, default="false", choices=["true", "false"])
    parser.add_argument("--deepemd_test_sfc", type=str, default="false", choices=["true", "false"])
    parser.add_argument(
        "--deepemd_solver",
        type=str,
        default="sinkhorn",
        choices=[
            "opencv",
            "qpth",
            "linprog",
            "sinkhorn",
            "uot",
            "unbalanced",
            "unbalanced_ot",
            "sinkhorn_unbalanced",
            "partial",
            "partial_ot",
            "partial_sinkhorn",
        ],
    )
    parser.add_argument("--deepemd_qpth_form", type=str, default="L2", choices=["QP", "L2"])
    parser.add_argument("--deepemd_qpth_l2_strength", type=float, default=1e-6)
    parser.add_argument("--deepemd_sinkhorn_reg", type=float, default=0.05)
    parser.add_argument("--deepemd_sinkhorn_iterations", type=int, default=20)
    parser.add_argument("--deepemd_sinkhorn_tolerance", type=float, default=1e-6)
    parser.add_argument("--deepemd_uot_tau_q", type=float, default=0.5)
    parser.add_argument("--deepemd_uot_tau_c", type=float, default=0.5)
    parser.add_argument("--deepemd_uot_score_normalize", type=str, default="false", choices=["true", "false"])
    parser.add_argument("--deepemd_partial_mass_fraction", type=float, default=0.5)
    parser.add_argument("--deepemd_partial_transport_mass", type=float, default=None)
    parser.add_argument("--deepemd_partial_score_normalize", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--deepemd_partial_backend", type=str, default="native", choices=["native", "pot"])
    parser.add_argument("--deepemd_partial_exact", type=str, default="false", choices=["true", "false"])
    parser.add_argument("--deepemd_eps", type=float, default=1e-8)
    parser.add_argument("--deepemd_sfc_lr", type=float, default=0.1)
    parser.add_argument("--deepemd_sfc_update_step", type=int, default=15)
    parser.add_argument("--deepemd_sfc_bs", type=int, default=4)
    parser.add_argument("--use_evidence_weight", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--use_shot_reliability", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--evidence_eps", type=float, default=1e-3)
    parser.add_argument("--consensus_scale", type=float, default=5.0)
    parser.add_argument("--shot_temperature", type=float, default=1.0)
    parser.add_argument("--debug_evidence", type=str, default="false", choices=["true", "false"])
    parser.add_argument("--dice_emd_lambda_disc", "--lambda_disc", dest="dice_emd_lambda_disc", type=float, default=0.2)
    parser.add_argument("--dice_emd_tau_comp", "--tau_comp", dest="dice_emd_tau_comp", type=float, default=0.1)
    parser.add_argument(
        "--dice_emd_use_softmin_distance",
        "--use_softmin_distance",
        dest="dice_emd_use_softmin_distance",
        type=str,
        default="true",
        choices=["true", "false"],
    )
    parser.add_argument("--dice_emd_tau_softmin", "--tau_softmin", dest="dice_emd_tau_softmin", type=float, default=0.1)
    parser.add_argument(
        "--dice_emd_debug_transport",
        "--debug_transport",
        dest="dice_emd_debug_transport",
        type=str,
        default="false",
        choices=["true", "false"],
    )
    parser.add_argument("--dice_emd_lambda_flow", "--lambda_flow", dest="dice_emd_lambda_flow", type=float, default=0.0)
    parser.add_argument(
        "--transport_recon_emd_tau_shot",
        "--tardis_emd_tau_shot",
        dest="transport_recon_emd_tau_shot",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--transport_recon_emd_kshot_mode",
        "--tardis_emd_kshot_mode",
        dest="transport_recon_emd_kshot_mode",
        type=str,
        default="query_condense",
        choices=["query_condense", "mean_condense"],
    )
    parser.add_argument(
        "--transport_recon_emd_lambda_emd",
        "--tardis_emd_lambda_emd",
        dest="transport_recon_emd_lambda_emd",
        type=float,
        default=0.3,
    )
    parser.add_argument(
        "--transport_recon_emd_lambda_rec",
        "--tardis_emd_lambda_rec",
        dest="transport_recon_emd_lambda_rec",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--transport_recon_emd_lambda_struct",
        "--tardis_emd_lambda_struct",
        dest="transport_recon_emd_lambda_struct",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--transport_recon_emd_score_scale",
        "--tardis_emd_score_scale",
        dest="transport_recon_emd_score_scale",
        type=float,
        default=8.0,
    )
    parser.add_argument(
        "--transport_recon_emd_debug_shapes",
        "--tardis_emd_debug_shapes",
        dest="transport_recon_emd_debug_shapes",
        type=str,
        default="false",
        choices=["true", "false"],
    )
    parser.add_argument(
        "--transport_recon_emd_normalize_reconstruction",
        "--tardis_emd_normalize_reconstruction",
        dest="transport_recon_emd_normalize_reconstruction",
        type=str,
        default="true",
        choices=["true", "false"],
    )

    parser.add_argument("--project", type=str, default="pulse_fewshot")
    parser.add_argument("--experiment_tag", type=str, default="")

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
        train_sfc_steps = getattr(args, "deepemd_train_sfc_update_step", None)
        train_sfc_bs = getattr(args, "deepemd_train_sfc_bs", None)
        train_sfc_overrides = []
        if train_sfc_steps is not None:
            train_sfc_overrides.append(f"train_sfc_steps={train_sfc_steps}")
        if train_sfc_bs is not None:
            train_sfc_overrides.append(f"train_sfc_bs={train_sfc_bs}")
        override_suffix = ""
        if train_sfc_overrides:
            override_suffix = ", " + ", ".join(train_sfc_overrides)
        print(
            "  deepemd: fast benchmark OT "
            f"(train_solver={getattr(args, 'deepemd_solver', 'sinkhorn')}, "
            f"train_sfc={getattr(args, 'deepemd_train_sfc', 'false')}, "
            f"fast_val={getattr(args, 'deepemd_fast_val', 'true')}, "
            f"test_exact={getattr(args, 'deepemd_test_exact', 'false')}, "
            f"test_sfc={getattr(args, 'deepemd_test_sfc', 'false')}, "
            f"qpth_form={getattr(args, 'deepemd_qpth_form', 'L2')}, "
            f"sinkhorn_reg={getattr(args, 'deepemd_sinkhorn_reg', 0.05)}, "
            f"sinkhorn_iters={getattr(args, 'deepemd_sinkhorn_iterations', 20)}, "
            f"uot_tau=({getattr(args, 'deepemd_uot_tau_q', 0.5)}, "
            f"{getattr(args, 'deepemd_uot_tau_c', 0.5)}), "
            f"partial_mass_fraction={getattr(args, 'deepemd_partial_mass_fraction', 0.5)}, "
            f"partial_backend={getattr(args, 'deepemd_partial_backend', 'native')}, "
            f"sfc_lr={getattr(args, 'deepemd_sfc_lr', 0.1)}, "
            f"sfc_steps={getattr(args, 'deepemd_sfc_update_step', 15)}, "
            f"sfc_bs={getattr(args, 'deepemd_sfc_bs', 4)}{override_suffix})"
        )
    if args.model == "evidence_deepemd":
        print(
            "  evidence_deepemd: DeepEMD balanced OT with evidence masses and shot aggregation "
            f"(train_solver={getattr(args, 'deepemd_solver', 'sinkhorn')}, "
            f"use_evidence_weight={getattr(args, 'use_evidence_weight', 'true')}, "
            f"use_shot_reliability={getattr(args, 'use_shot_reliability', 'true')}, "
            f"evidence_eps={getattr(args, 'evidence_eps', 1e-3)}, "
            f"consensus_scale={getattr(args, 'consensus_scale', 5.0)}, "
            f"shot_temperature={getattr(args, 'shot_temperature', 1.0)}, "
            f"debug_evidence={getattr(args, 'debug_evidence', 'false')}, "
            f"fast_val={getattr(args, 'deepemd_fast_val', 'true')}, "
            f"test_exact={getattr(args, 'deepemd_test_exact', 'false')}, "
            f"test_sfc={getattr(args, 'deepemd_test_sfc', 'false')})"
        )
    if args.model == "dice_emd":
        print(
            "  dice_emd: DeepEMD balanced Sinkhorn/EMD with class-competitive evidence cost "
            f"(lambda_disc={getattr(args, 'dice_emd_lambda_disc', 0.2)}, "
            f"tau_comp={getattr(args, 'dice_emd_tau_comp', 0.1)}, "
            f"use_softmin_distance={getattr(args, 'dice_emd_use_softmin_distance', 'true')}, "
            f"tau_softmin={getattr(args, 'dice_emd_tau_softmin', 0.1)}, "
            f"debug_transport={getattr(args, 'dice_emd_debug_transport', 'false')}, "
            f"lambda_flow={getattr(args, 'dice_emd_lambda_flow', 0.0)}, "
            f"solver={getattr(args, 'deepemd_solver', 'sinkhorn')})"
        )
    if args.model in {"transport_recon_emd", "tardis_emd"}:
        print(
            f"  {args.model}: DeepEMD dense ResNet12 tokens + balanced transport alignment "
            f"(tau_shot={getattr(args, 'transport_recon_emd_tau_shot', 0.2)}, "
            f"kshot_mode={getattr(args, 'transport_recon_emd_kshot_mode', 'query_condense')}, "
            f"sinkhorn_reg={getattr(args, 'deepemd_sinkhorn_reg', 0.05)}, "
            f"sinkhorn_iters={getattr(args, 'deepemd_sinkhorn_iterations', 20)}, "
            f"lambda_emd={getattr(args, 'transport_recon_emd_lambda_emd', 0.3)}, "
            f"lambda_rec={getattr(args, 'transport_recon_emd_lambda_rec', 1.0)}, "
            f"lambda_struct={getattr(args, 'transport_recon_emd_lambda_struct', 0.1)}, "
            f"score_scale={getattr(args, 'transport_recon_emd_score_scale', 8.0)}, "
            f"normalize_reconstruction={getattr(args, 'transport_recon_emd_normalize_reconstruction', 'true')}, "
            f"debug_shapes={getattr(args, 'transport_recon_emd_debug_shapes', 'false')})"
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
    if args.model == "spifce_shot":
        print(
            "  spifce_shot: stable/variant token factorization -> stable evidence gate -> "
            "global prototypes + shot-preserving local top-r partial matching "
            f"(stable_dim={getattr(args, 'spif_stable_dim', 64)}, "
            f"variant_dim={getattr(args, 'spif_variant_dim', 64)}, "
            f"top_r={getattr(args, 'spif_shot_top_r', 4)}, "
            f"shot_agg={getattr(args, 'spif_shot_agg', 'softmax')}, "
            f"shot_softmax_beta={getattr(args, 'spif_shot_softmax_beta', 10.0)}, "
            f"gate={getattr(args, 'spif_gate_on', 'true')}, "
            f"factorization={getattr(args, 'spif_factorization_on', 'true')}, "
            f"global_only={getattr(args, 'spif_global_only', 'false')}, "
            f"local_only={getattr(args, 'spif_local_only', 'false')})"
        )
        print(
            "  note: `shot_agg=pooled` reproduces the original shot-collapsed SPIFCE local matcher "
            "for the same top-r, while `mean` and `softmax` preserve support-shot structure."
        )
    if args.model == "spifce_asym_shot":
        print(
            "  spifce_asym_shot: gated stable global prototype branch + separate local token head + "
            "shot-preserving local top-r matching + adaptive fusion "
            f"(stable_dim={getattr(args, 'spif_stable_dim', 64)}, "
            f"variant_dim={getattr(args, 'spif_variant_dim', 64)}, "
            f"local_dim={getattr(args, 'spif_asym_local_dim', None) or getattr(args, 'spif_stable_dim', 64)}, "
            f"top_r={getattr(args, 'spif_asym_top_r', 4)}, "
            f"shot_agg={getattr(args, 'spif_asym_shot_agg', 'softmax')}, "
            f"shot_softmax_beta={getattr(args, 'spif_asym_shot_softmax_beta', 10.0)}, "
            f"fusion_mode={getattr(args, 'spif_asym_fusion_mode', 'margin_adaptive')}, "
            f"fusion_kappa={getattr(args, 'spif_asym_fusion_kappa', 8.0)}, "
            f"global_scale={getattr(args, 'spif_asym_global_scale', 1.0)}, "
            f"local_scale={getattr(args, 'spif_asym_local_scale', 1.0)}, "
            f"local_score_mode={getattr(args, 'spif_asym_local_score_mode', 'bidirectional')}, "
            f"share_local_head={getattr(args, 'spif_asym_share_local_head', 'false')}, "
            f"gate={getattr(args, 'spif_gate_on', 'true')}, "
            f"factorization={getattr(args, 'spif_factorization_on', 'true')}, "
            f"global_only={getattr(args, 'spif_global_only', 'false')}, "
            f"local_only={getattr(args, 'spif_local_only', 'false')})"
        )
        print(
            "  note: global branch reuses SPIF stable gating, local branch gets its own token head, "
            "and `bidirectional` local score averages query->support and support->query top-r correspondence "
            "before shot aggregation."
        )
    if args.model == "spifaeb":
        aeb_hidden = getattr(args, "aeb_hidden", None)
        stable_dim = int(getattr(args, "spif_stable_dim", 64) or 64)
        print(
            "  spifaeb: stable/variant token factorization -> stable evidence gate -> "
            "global prototypes + adaptive evidence-budget local matching "
            f"(stable_dim={stable_dim}, "
            f"variant_dim={getattr(args, 'spif_variant_dim', 64)}, "
            f"aeb_hidden={aeb_hidden if aeb_hidden is not None else max(16, stable_dim // 2)}, "
            f"aeb_beta={getattr(args, 'aeb_beta', 1.0)}, "
            f"aeb_eps={getattr(args, 'aeb_eps', 1e-6)}, "
            f"gate={getattr(args, 'spif_gate_on', 'true')}, "
            f"factorization={getattr(args, 'spif_factorization_on', 'true')}, "
            f"global_only={getattr(args, 'spif_global_only', 'false')}, "
            f"local_only={getattr(args, 'spif_local_only', 'false')})"
        )
        print(
            "  note: this variant preserves the original global cosine branch and replaces only "
            "the local top-r matcher, so `spif_top_r` is accepted for compatibility but unused."
        )
    if args.model == "spifaeb_shot":
        aeb_hidden = getattr(args, "aeb_hidden", None)
        stable_dim = int(getattr(args, "spif_stable_dim", 64) or 64)
        print(
            "  spifaeb_shot: stable/variant token factorization -> stable evidence gate -> "
            "global prototypes + shot-preserving adaptive evidence-budget local matching "
            f"(stable_dim={stable_dim}, "
            f"variant_dim={getattr(args, 'spif_variant_dim', 64)}, "
            f"aeb_hidden={aeb_hidden if aeb_hidden is not None else max(16, stable_dim // 2)}, "
            f"aeb_beta={getattr(args, 'aeb_beta', 1.0)}, "
            f"aeb_eps={getattr(args, 'aeb_eps', 1e-6)}, "
            f"shot_agg={getattr(args, 'aeb_shot_agg', 'softmax')}, "
            f"shot_softmax_beta={getattr(args, 'aeb_shot_softmax_beta', 10.0)}, "
            f"gate={getattr(args, 'spif_gate_on', 'true')}, "
            f"factorization={getattr(args, 'spif_factorization_on', 'true')}, "
            f"global_only={getattr(args, 'spif_global_only', 'false')}, "
            f"local_only={getattr(args, 'spif_local_only', 'false')})"
        )
        print(
            "  note: `shot_agg=pooled` reproduces the original shot-collapsed AEB local matcher, "
            "while `mean` and `softmax` preserve support-shot structure."
        )
    if args.model == "spif_rdp":
        stable_dim = int(getattr(args, "spif_stable_dim", 64) or 64)
        backbone_name = getattr(args, "fewshot_backbone", "resnet12")
        print(
            "  spif_rdp: stable/variant SPIF encoder + reliability-weighted class prototype + "
            "diagonal variance floor + reliability-modulated global distance + Sinkhorn-OT local residual "
            f"(stable_dim={stable_dim}, "
            f"variant_dim={getattr(args, 'spif_variant_dim', 64)}, "
            f"backbone={backbone_name}, "
            f"top_r={getattr(args, 'spif_rdp_top_r', 5)}, "
            f"lambda_init={getattr(args, 'spif_rdp_lambda_init', 1e-3)}, "
            f"alpha_init={getattr(args, 'spif_rdp_alpha_init', 1.0)}, "
            f"tau_init={getattr(args, 'spif_rdp_tau_init', 10.0)}, "
            f"variance_floor={getattr(args, 'spif_rdp_variance_floor', 1e-2)}, "
            f"compact_loss_weight={getattr(args, 'spif_rdp_compact_loss_weight', 0.1)}, "
            f"sep_loss_weight={getattr(args, 'spif_rdp_sep_loss_weight', 0.05)}, "
            f"sep_margin={getattr(args, 'spif_rdp_sep_margin', 0.5)}, "
            f"consistency_weight={getattr(args, 'spif_rdp_consistency_weight', 0.0)}, "
            f"decorr_weight={getattr(args, 'spif_rdp_decorr_weight', 0.0)}, "
            f"sparse_weight={getattr(args, 'spif_rdp_sparse_weight', 0.0)}, "
            f"gate={getattr(args, 'spif_gate_on', 'true')}, "
            f"factorization={getattr(args, 'spif_factorization_on', 'true')})"
        )
        print(
            "  note: `spif_rdp_top_r` is accepted only for backward compatibility. "
            "`spif_global_only` and `spif_local_only` still work. "
            "`spif_rdp_beta0_init` remains accepted as a legacy alias of `spif_rdp_beta_init`."
        )
    if args.model == "rada_fsl":
        print(
            "  rada_fsl: backbone pooled embeddings -> query-conditioned reliability weights -> "
            "residual prototypes -> diagonal dispersion-aware metric "
            f"(feat_dim={getattr(args, 'rada_feat_dim', None) or 'backbone'}, "
            f"tau_r={getattr(args, 'rada_tau_r', 0.5)}, "
            f"lambda_proto={getattr(args, 'rada_lambda_proto', 0.7)}, "
            f"gamma_disp={getattr(args, 'rada_gamma_disp', 0.5)}, "
            f"reliability_head={getattr(args, 'rada_reliability_head', 'linear')}, "
            f"reliability_hidden_dim={getattr(args, 'rada_reliability_hidden_dim', 16)}, "
            f"l2_normalize={getattr(args, 'rada_l2_normalize', 'true')}, "
            f"use_reliability={getattr(args, 'rada_use_reliability', 'true')}, "
            f"use_dispersion_metric={getattr(args, 'rada_use_dispersion_metric', 'true')}, "
            f"query_conditioned={getattr(args, 'rada_query_conditioned', 'true')}, "
            f"use_residual_anchor={getattr(args, 'rada_use_residual_anchor', 'true')}, "
            f"use_shrinkage={getattr(args, 'rada_use_shrinkage', 'true')}, "
            f"use_evidence_bound={getattr(args, 'rada_use_evidence_bound', 'true')}, "
            f"evidence_temperature={getattr(args, 'rada_evidence_temperature', 1.0)}, "
            f"min_reliability_mix={getattr(args, 'rada_min_reliability_mix', 0.25)}, "
            f"dispersion_inflation={getattr(args, 'rada_dispersion_inflation', 0.25)}, "
            f"entropy_reg_weight={getattr(args, 'rada_entropy_reg_weight', 0.0)})"
        )
        print(
            "  note: default training remains episodic cross-entropy only; "
            "evidence-bound stabilizes shot reliability without an extra branch; "
            "`rada_entropy_reg_weight` is optional and off by default."
        )
    if args.model == "spif_ota":
        print(
            "  spif_ota: single-branch transport projector + physics-aware token masses + "
            "batched entropic OT + shot-wise evidence aggregation "
            f"(transport_dim={getattr(args, 'spif_ota_transport_dim', None) or getattr(args, 'token_dim', None) or 128}, "
            f"mass_hidden={getattr(args, 'spif_ota_mass_hidden_dim', None)}, "
            f"mass_temp={getattr(args, 'spif_ota_mass_temperature', 1.0)}, "
            f"sinkhorn_eps={getattr(args, 'spif_ota_sinkhorn_epsilon', 0.05)}, "
            f"sinkhorn_iters={getattr(args, 'spif_ota_sinkhorn_iterations', 60)}, "
            f"shot_hidden={getattr(args, 'spif_ota_shot_hidden_dim', 32)}, "
            f"shot_agg={getattr(args, 'spif_ota_shot_aggregation', 'learned')}, "
            f"position_cost={getattr(args, 'spif_ota_position_cost_weight', 0.0)}, "
            f"use_column_bias={getattr(args, 'spif_ota_use_column_bias', 'true')}, "
            f"lambda_mass={getattr(args, 'spif_ota_lambda_mass', 0.0)}, "
            f"lambda_consistency={getattr(args, 'spif_ota_lambda_consistency', 0.0)})"
        )
    if args.model in {"evidence_budget_ot", "evidence_budgeted_ot", "ebot", "ebot_scalogram"}:
        print(
            "  ebot: scalogram priors -> reliability evidence budget -> "
            "dustbin Sinkhorn OT -> reliability-weighted k-shot aggregation "
            f"(proj_dim={getattr(args, 'ebot_proj_dim', 256)}, "
            f"sinkhorn_eps={getattr(args, 'ebot_sinkhorn_epsilon', 0.05)}, "
            f"sinkhorn_iters={getattr(args, 'ebot_sinkhorn_iters', 50)}, "
            f"dustbin_cost={getattr(args, 'ebot_dustbin_cost', 0.7)}, "
            f"learnable_dustbin={getattr(args, 'ebot_learnable_dustbin_cost', 'true')}, "
            f"budget=({getattr(args, 'ebot_min_budget', 0.15)}, {getattr(args, 'ebot_max_budget', 0.95)}), "
            f"score_scale={getattr(args, 'ebot_score_scale', 12.5)}, "
            f"priors={getattr(args, 'ebot_use_scalogram_priors', 'true')}, "
            f"cross_ref={getattr(args, 'ebot_use_cross_reference', 'true')}, "
            f"kshot_reweight={getattr(args, 'ebot_use_kshot_reweighting', 'true')}, "
            f"symmetric={getattr(args, 'ebot_symmetric_matching', 'false')})"
        )
    if args.model == "spif_otccls":
        print(
            "  spif_otccls: ResNet12 RGB backbone + feature-wise stable/variant gating + "
            "energy-weighted sliced Wasserstein global branch + confusability-conditioned local branch "
            f"(feature_dim={getattr(args, 'spif_otccls_feature_dim', 256)}, "
            f"gate_hidden={getattr(args, 'spif_otccls_gate_hidden', 128)}, "
            f"num_projections={getattr(args, 'spif_otccls_num_projections', 64)}, "
            f"num_quantiles={getattr(args, 'spif_otccls_num_quantiles', 100)}, "
            f"tau_g_init={getattr(args, 'spif_otccls_tau_g_init', 10.0)}, "
            f"tau_l_init={getattr(args, 'spif_otccls_tau_l_init', 1.0)}, "
            f"beta_init={getattr(args, 'spif_otccls_beta_init', 0.5)}, "
            f"alpha_init={getattr(args, 'spif_otccls_alpha_init', 1.0)}, "
            f"compact_loss_weight={getattr(args, 'spif_otccls_compact_loss_weight', 0.1)}, "
            f"decorr_loss_weight={getattr(args, 'spif_otccls_decorr_loss_weight', 0.05)}, "
            f"entropy_loss_weight={getattr(args, 'spif_otccls_entropy_loss_weight', 0.01)}, "
            f"shot_agg={getattr(args, 'spif_otccls_shot_agg', 'softmax')}, "
            f"shot_softmax_beta={getattr(args, 'spif_otccls_shot_softmax_beta', 10.0)}, "
            f"swd_backend={getattr(args, 'spif_otccls_swd_backend', 'pot')}, "
            f"backbone_drop_rate={getattr(args, 'spif_otccls_backbone_drop_rate', 0.05)}, "
            f"global_only={getattr(args, 'spif_global_only', 'false')}, "
            f"local_only={getattr(args, 'spif_local_only', 'false')})"
        )
    if args.model == "spifaeb_v2":
        stable_dim = int(getattr(args, "spif_stable_dim", 64) or 64)
        print(
            "  spifaeb_v2: stable/variant global branch + stable-anchor residual local head + "
            "evidence-profile-conditioned retention controller + monotonic rank-budget local matching "
            f"(stable_dim={stable_dim}, "
            f"variant_dim={getattr(args, 'spif_variant_dim', 64)}, "
            f"local_dim={getattr(args, 'aeb_v2_local_dim', None) or stable_dim}, "
            f"budget_hidden={getattr(args, 'aeb_v2_hidden', None) or max(16, stable_dim // 2)}, "
            f"budget_range=({getattr(args, 'aeb_v2_min_budget', 0.2)}, {getattr(args, 'aeb_v2_max_budget', 0.85)}), "
            f"rank_temperature={getattr(args, 'aeb_v2_rank_temperature', 0.35)}, "
            f"cond_global_weight={getattr(args, 'aeb_v2_cond_global_weight', 0.35)}, "
            f"cond_global_temperature={getattr(args, 'aeb_v2_cond_global_temperature', 8.0)}, "
            f"cond_global_use_gate_prior={getattr(args, 'aeb_v2_cond_global_use_gate_prior', 'true')}, "
            f"query_class_attention_on={getattr(args, 'aeb_v2_query_class_attention_on', 'true')}, "
            f"query_class_attention_temperature={getattr(args, 'aeb_v2_query_class_attention_temperature', 8.0)}, "
            f"query_class_attention_use_gate_prior={getattr(args, 'aeb_v2_query_class_attention_use_gate_prior', 'true')}, "
            f"anchor_top_r={getattr(args, 'aeb_v2_anchor_top_r', 4)}, "
            f"anchor_mix={getattr(args, 'aeb_v2_anchor_mix', 0.35)}, "
            f"local_mix_mode={getattr(args, 'aeb_v2_local_mix_mode', 'prior_adaptive')}, "
            f"adaptive_mix_range=({getattr(args, 'aeb_v2_adaptive_mix_min', 0.2)}, {getattr(args, 'aeb_v2_adaptive_mix_max', 0.8)}), "
            f"adaptive_mix_scale={getattr(args, 'aeb_v2_adaptive_mix_scale', 6.0)}, "
            f"local_residual_scale={getattr(args, 'aeb_v2_local_residual_scale', 0.25)}, "
            f"budget_prior_scale={getattr(args, 'aeb_v2_budget_prior_scale', 6.0)}, "
            f"budget_residual_scale={getattr(args, 'aeb_v2_budget_residual_scale', 0.15)}, "
            f"budget_rank_weight={getattr(args, 'aeb_v2_budget_rank_weight', 0.1)}, "
            f"budget_rank_margin={getattr(args, 'aeb_v2_budget_rank_margin', 0.05)}, "
            f"budget_residual_reg_weight={getattr(args, 'aeb_v2_budget_residual_reg_weight', 0.05)}, "
            f"support_coverage_temperature={getattr(args, 'aeb_v2_support_coverage_temperature', 8.0)}, "
            f"coverage_bonus_weight={getattr(args, 'aeb_v2_coverage_bonus_weight', 0.0)}, "
            f"competition_weight={getattr(args, 'aeb_v2_competition_weight', 1.0)}, "
            f"competition_temperature={getattr(args, 'aeb_v2_competition_temperature', 8.0)}, "
            f"local_margin_weight={getattr(args, 'aeb_v2_local_margin_weight', 0.0)}, "
            f"local_margin_target={getattr(args, 'aeb_v2_local_margin_target', 0.5)}, "
            f"anchor_consistency_weight={getattr(args, 'aeb_v2_anchor_consistency_weight', 0.02)}, "
            f"fusion_mode={getattr(args, 'aeb_v2_fusion_mode', 'fixed')}, "
            f"fusion_kappa={getattr(args, 'aeb_v2_fusion_kappa', 8.0)}, "
            f"local_score_mode={getattr(args, 'aeb_v2_local_score_mode', 'query_to_support')}, "
            f"global_scale={getattr(args, 'aeb_v2_global_scale', 1.0)}, "
            f"local_scale={getattr(args, 'aeb_v2_local_scale', 1.0)}, "
            f"global_ce_weight={getattr(args, 'aeb_v2_global_ce_weight', 0.0)}, "
            f"local_ce_weight={getattr(args, 'aeb_v2_local_ce_weight', 0.05)}, "
            f"share_local_head={getattr(args, 'aeb_v2_share_local_head', 'false')}, "
            f"query_gate_weighting={getattr(args, 'aeb_v2_query_gate_weighting', 'false')}, "
            f"detach_budget_context={getattr(args, 'aeb_v2_detach_budget_context', 'true')}, "
            f"detach_local_backbone={getattr(args, 'aeb_v2_detach_local_backbone', 'true')}, "
            f"gate={getattr(args, 'spif_gate_on', 'true')}, "
            f"factorization={getattr(args, 'spif_factorization_on', 'true')}, "
            f"global_only={getattr(args, 'spif_global_only', 'false')}, "
            f"local_only={getattr(args, 'spif_local_only', 'false')})"
        )
        print(
            "  note: this variant removes the old threshold+fallback local operator; the predicted budget now "
            "controls a monotonic soft top-k over local evidence that is explicitly contrasted against competing "
            "classes, and branch-specific CE keeps global/local heads semantically aligned."
        )
    if args.model == "spifaeb_v3":
        stable_dim = int(getattr(args, "spif_stable_dim", 64) or 64)
        print(
            "  spifaeb_v3: stable/variant SPIF encoder + structured evidence global head "
            "(class center + scalar dispersion + support reliability) + unchanged v2 local branch "
            f"(stable_dim={stable_dim}, "
            f"variant_dim={getattr(args, 'spif_variant_dim', 64)}, "
            f"local_dim={getattr(args, 'aeb_v2_local_dim', None) or stable_dim}, "
            f"use_structured_global={getattr(args, 'spifaeb_v3_use_structured_global', 'true')}, "
            f"sigma_min={getattr(args, 'spifaeb_v3_sigma_min', 0.05)}, "
            f"use_reliability={getattr(args, 'spifaeb_v3_use_reliability', 'true')}, "
            f"reliability_detach={getattr(args, 'spifaeb_v3_reliability_detach', 'true')}, "
            f"beta_align={getattr(args, 'spifaeb_v3_beta_align', 1.0)}, "
            f"beta_dev={getattr(args, 'spifaeb_v3_beta_dev', 0.5)}, "
            f"beta_rel={getattr(args, 'spifaeb_v3_beta_rel', 0.2)}, "
            f"learnable_betas={getattr(args, 'spifaeb_v3_learnable_betas', 'false')}, "
            f"global_scale={getattr(args, 'aeb_v2_global_scale', 1.0)}, "
            f"local_scale={getattr(args, 'aeb_v2_local_scale', 1.0)}, "
            f"fusion_mode={getattr(args, 'aeb_v2_fusion_mode', 'fixed')}, "
            f"global_ce_weight={getattr(args, 'spifaeb_v3_global_ce_weight', 0.0)}, "
            f"local_ce_weight={getattr(args, 'aeb_v2_local_ce_weight', 0.05)}, "
            f"gate={getattr(args, 'spif_gate_on', 'true')}, "
            f"factorization={getattr(args, 'spif_factorization_on', 'true')}, "
            f"global_only={getattr(args, 'spif_global_only', 'false')}, "
            f"local_only={getattr(args, 'spif_local_only', 'false')})"
        )
        print(
            "  note: v3 replaces the old cosine/conditioned-cosine global scorer; the v2 local matcher, "
            "retention controller, and fusion path remain intact. `aeb_v2_cond_global_*` flags are ignored."
        )
    if args.model in {"spifce_local_sw", "spifmax_local_sw"}:
        print(
            "  spif_local_sw: stable/variant token factorization -> stable evidence gate -> "
            "global cosine prototypes + local token-set sliced Wasserstein matching "
            f"(stable_dim={getattr(args, 'spif_stable_dim', 64)}, "
            f"variant_dim={getattr(args, 'spif_variant_dim', 64)}, "
            f"sw_proj={getattr(args, 'spif_local_sw_num_projections', 64)}, "
            f"sw_p={getattr(args, 'spif_local_sw_p', 2.0)}, "
            f"sw_normalize={getattr(args, 'spif_local_sw_normalize', 'true')}, "
            f"sw_scale={getattr(args, 'spif_local_sw_score_scale', 1.0)}, "
            f"gate={getattr(args, 'spif_gate_on', 'true')}, "
            f"factorization={getattr(args, 'spif_factorization_on', 'true')}, "
            f"global_only={getattr(args, 'spif_global_only', 'false')}, "
            f"local_only={getattr(args, 'spif_local_only', 'false')})"
        )
        print(
            "  note: this variant keeps the original global cosine branch and replaces only "
            "the local matcher, so `spif_top_r` is unused here."
        )
    if args.model in {"spifce_local_papersw", "spifmax_local_papersw"}:
        print(
            "  spif_local_papersw: stable/variant token factorization -> stable evidence gate -> "
            "global cosine prototypes + exact paper-style local sliced Wasserstein matching "
            f"(stable_dim={getattr(args, 'spif_stable_dim', 64)}, "
            f"variant_dim={getattr(args, 'spif_variant_dim', 64)}, "
            f"train_proj={getattr(args, 'spif_papersw_train_num_projections', 128)}, "
            f"eval_proj={getattr(args, 'spif_papersw_eval_num_projections', 512)}, "
            f"sw_p={getattr(args, 'spif_papersw_p', 2.0)}, "
            f"normalize_inputs={getattr(args, 'spif_papersw_normalize_inputs', 'false')}, "
            f"train_proj_mode={getattr(args, 'spif_papersw_train_projection_mode', 'resample')}, "
            f"eval_proj_mode={getattr(args, 'spif_papersw_eval_projection_mode', 'fixed')}, "
            f"eval_repeats={getattr(args, 'spif_papersw_eval_num_repeats', 1)}, "
            f"sw_scale={getattr(args, 'spif_papersw_score_scale', 8.0)}, "
            f"shot_agg={getattr(args, 'spif_papersw_shot_agg', 'pooled')}, "
            f"shot_softmin_beta={getattr(args, 'spif_papersw_shot_softmin_beta', 10.0)}, "
            f"gate={getattr(args, 'spif_gate_on', 'true')}, "
            f"factorization={getattr(args, 'spif_factorization_on', 'true')}, "
            f"global_only={getattr(args, 'spif_global_only', 'false')}, "
            f"local_only={getattr(args, 'spif_local_only', 'false')})"
        )
        print(
            "  note: `shot_agg=pooled` reproduces the old shot-collapsed local paper-SW baseline, "
            "while `mean` and `softmin` keep support shots separate until after transport."
        )
    if args.model in {"spifharmce", "spifharmmax"}:
        print(
            "  spif_harmonic: stable/variant token factorization -> gated stable tokens -> "
            "global-only residual Mamba refinement + attentive pooling -> local paper-SW -> adaptive residual fusion "
            f"(stable_dim={getattr(args, 'spif_stable_dim', 64)}, "
            f"variant_dim={getattr(args, 'spif_variant_dim', 64)}, "
            f"global_scale={getattr(args, 'spif_harm_global_scale', 16.0)}, "
            f"mamba_state={getattr(args, 'ssm_state_dim', 16)}, "
            f"mamba_depth={getattr(args, 'spif_mamba_depth', 1)}, "
            f"beta_init={getattr(args, 'spif_harm_beta_init', 0.1)}, "
            f"learnable_beta={getattr(args, 'spif_harm_learnable_beta', 'true')}, "
            f"use_final_gate={getattr(args, 'spif_harm_use_final_gate', 'true')}, "
            f"fusion_hidden_dim={getattr(args, 'spif_harm_fusion_hidden_dim', 32)}, "
            f"local_scale={getattr(args, 'spif_harm_local_score_scale', 8.0)}, "
            f"train_proj={getattr(args, 'spif_papersw_train_num_projections', 128)}, "
            f"eval_proj={getattr(args, 'spif_papersw_eval_num_projections', 512)}, "
            f"global_only={getattr(args, 'spif_global_only', 'false')}, "
            f"local_only={getattr(args, 'spif_local_only', 'false')})"
        )
        if args.model == "spifharmmax":
            print(
                "  spif_harmonic_aux: "
                f"global_ce={getattr(args, 'spif_harm_global_ce_weight', 0.0)}, "
                f"local_ce={getattr(args, 'spif_harm_local_ce_weight', 0.0)}, "
                f"factor_consistency={getattr(args, 'spif_harm_factor_consistency_weight', 0.0)}, "
                f"factor_decorr={getattr(args, 'spif_harm_factor_decorr_weight', 0.0)}, "
                f"factor_sparse={getattr(args, 'spif_harm_factor_sparse_weight', 0.0)}, "
                f"final_sparse={getattr(args, 'spif_harm_final_sparse_weight', 0.0)}"
            )
    if args.model in {"spifmambace", "spifmambamax", "spifmambaadjce", "spifmambaadjmax"}:
        print(
            "  spif_mamba_adjusted: stable/variant factorization -> raw coarse gate -> residual official Mamba "
            "refinement -> final stable gate -> global prototypes + local top-r partial matching "
            f"(stable_dim={getattr(args, 'spif_stable_dim', 64)}, "
            f"variant_dim={getattr(args, 'spif_variant_dim', 64)}, "
            f"top_r={getattr(args, 'spif_top_r', 3)}, "
            f"raw_gate={getattr(args, 'spif_adj_use_raw_gate', 'true')}, "
            f"final_gate={getattr(args, 'spif_adj_use_final_gate', 'true')}, "
            f"beta_init={getattr(args, 'spif_adj_beta_init', 0.1)}, "
            f"learnable_beta={getattr(args, 'spif_adj_learnable_beta', 'true')}, "
            f"mamba_depth={getattr(args, 'spif_mamba_depth', 1)}, "
            f"mamba_ffn_multiplier={getattr(args, 'spif_mamba_ffn_multiplier', 2)}, "
            f"mamba_d_conv={getattr(args, 'spif_mamba_d_conv', 4)}, "
            f"mamba_expand={getattr(args, 'spif_mamba_expand', 2)}, "
            f"mamba_dropout={getattr(args, 'spif_mamba_dropout', 0.0)}, "
            f"factorization={getattr(args, 'spif_factorization_on', 'true')}, "
            f"global_only={getattr(args, 'spif_global_only', 'false')}, "
            f"local_only={getattr(args, 'spif_local_only', 'false')})"
        )
    if args.model in {"spifv2ce", "spifv2max"}:
        print(
            "  spif_v2: shared bottleneck stable/variant split -> competitive gate -> "
            "global prototype anchor + mutual local partial matching -> bounded residual local correction "
            f"(shared_dim={getattr(args, 'spifv2_shared_dim', 64)}, "
            f"stable_dim={getattr(args, 'spifv2_stable_dim', 32)}, "
            f"variant_dim={getattr(args, 'spifv2_variant_dim', 64)}, "
            f"gate_type={getattr(args, 'spifv2_gate_type', 'sparsemax')}, "
            f"top_r={getattr(args, 'spifv2_top_r', 3)}, "
            f"mutual_local={getattr(args, 'spifv2_mutual_local', 'true')}, "
            f"fusion={getattr(args, 'spifv2_fusion_mode', 'residual')}, "
            f"lambda_local_init={getattr(args, 'spifv2_lambda_local_init', 0.2)}, "
            f"shared_bottleneck={getattr(args, 'spifv2_use_shared_bottleneck', 'true')}, "
            f"global_only={getattr(args, 'spifv2_global_only', 'false')}, "
            f"local_only={getattr(args, 'spifv2_local_only', 'false')})"
        )
        if args.model == "spifv2max":
            print(
                "  spif_v2_aux: "
                f"consistency={getattr(args, 'spifv2_consistency_weight', 0.05)}, "
                f"decorr={getattr(args, 'spifv2_decorr_weight', 0.005)}, "
                f"gate_reg={getattr(args, 'spifv2_gate_reg_weight', 5e-4)}, "
                f"consistency_dropout={getattr(args, 'spifv2_consistency_dropout', 0.1)}"
            )
    if args.model == "spifv3_cstn":
        print(
            "  spif_v3_cstn: unordered local descriptors -> canonical transport atoms -> "
            "permutation-aware support set pooling -> sliced Wasserstein matching "
            f"(d_model={getattr(args, 'd_model', None) or getattr(args, 'token_dim', 128) or 64}, "
            f"num_atoms={getattr(args, 'num_atoms', 8)}, "
            f"num_sw_projections={getattr(args, 'num_sw_projections', 64)}, "
            f"assignment_temperature={getattr(args, 'assignment_temperature', 1.0)}, "
            f"temperature={getattr(args, 'spifv3_temperature', 16.0)}, "
            f"set_pool_heads={getattr(args, 'cstn_set_pool_heads', 4)})"
        )
        print(
            "  spif_v3_cstn_aux: "
            f"atom_entropy_reg={getattr(args, 'use_atom_entropy_reg', 'false')} "
            f"(weight={getattr(args, 'atom_entropy_reg_weight', 0.01)}), "
            f"support_consistency_reg={getattr(args, 'use_support_consistency_reg', 'false')} "
            f"(weight={getattr(args, 'support_consistency_reg_weight', 0.01)})"
        )
    if args.model == "spifv3_cstn_papersw":
        print(
            "  spif_v3_cstn_papersw: unordered local descriptors -> canonical transport atoms -> "
            "permutation-aware support set pooling -> paper-style sliced Wasserstein matching "
            f"(d_model={getattr(args, 'd_model', None) or getattr(args, 'token_dim', 128) or 64}, "
            f"num_atoms={getattr(args, 'num_atoms', 8)}, "
            f"train_num_projections={getattr(args, 'spif_papersw_train_num_projections', 128)}, "
            f"eval_num_projections={getattr(args, 'spif_papersw_eval_num_projections', 512)}, "
            f"p={getattr(args, 'spif_papersw_p', 2.0)}, "
            f"normalize_inputs={getattr(args, 'spif_papersw_normalize_inputs', 'false')}, "
            f"assignment_temperature={getattr(args, 'assignment_temperature', 1.0)}, "
            f"temperature={getattr(args, 'spifv3_temperature', 16.0)}, "
            f"set_pool_heads={getattr(args, 'cstn_set_pool_heads', 4)})"
        )
        print(
            "  spif_v3_cstn_papersw_aux: "
            f"atom_entropy_reg={getattr(args, 'use_atom_entropy_reg', 'false')} "
            f"(weight={getattr(args, 'atom_entropy_reg_weight', 0.01)}), "
            f"support_consistency_reg={getattr(args, 'use_support_consistency_reg', 'false')} "
            f"(weight={getattr(args, 'support_consistency_reg_weight', 0.01)}), "
            f"train_projection_mode={getattr(args, 'spif_papersw_train_projection_mode', 'resample')}, "
            f"eval_projection_mode={getattr(args, 'spif_papersw_eval_projection_mode', 'fixed')}, "
            f"eval_num_repeats={getattr(args, 'spif_papersw_eval_num_repeats', 1)}"
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
    if args.model in HROT_FSL_MODEL_NAMES:
        hrot_raw_tokens = _bool_flag(getattr(args, "hrot_use_raw_backbone_tokens", "false"), default=False)
        hrot_token_dim = "raw_backbone" if hrot_raw_tokens else (
            getattr(args, "hrot_token_dim", None) or getattr(args, "token_dim", 128)
        )
        hrot_variant = "J_ECOT_M2" if args.model in M2_MODEL_NAMES else getattr(args, "hrot_variant", "E")
        hrot_ecot_rho_bank = getattr(args, "hrot_ecot_rho_bank", None)
        if args.model in M2_MODEL_NAMES and hrot_ecot_rho_bank is None:
            hrot_ecot_rho_bank = "1.0" if args.model in M2_FULL_OT_MODEL_NAMES else "0.80"
        if hrot_ecot_rho_bank is None:
            normalized_hrot_variant = str(hrot_variant).strip().upper().replace("-", "").replace("_", "")
            if normalized_hrot_variant == "CPECOT":
                hrot_ecot_rho_bank = "0.50,0.80,0.95"
            elif normalized_hrot_variant in {
                "JECOTM2",
                "ECOTM2",
                "M2JECOT",
                "SBECOT",
                "JECOTNNCS",
                "ECOTNNCS",
                "JECOTCARE",
                "CAREUOT",
            }:
                hrot_ecot_rho_bank = "0.80"
            else:
                hrot_ecot_rho_bank = "0.45,0.60,0.75,0.80,0.90"
        hrot_ecot_base_rho = getattr(args, "hrot_ecot_base_rho", None)
        if args.model in M2_MODEL_NAMES and hrot_ecot_base_rho is None:
            hrot_ecot_base_rho = 1.0 if args.model in M2_FULL_OT_MODEL_NAMES else 0.80
        ecot_transport_mode = getattr(args, "hrot_ecot_transport_mode", None)
        if ecot_transport_mode is None:
            ecot_transport_mode = "balanced" if args.model in M2_FULL_OT_MODEL_NAMES else "unbalanced"
        hrot_label = "m2" if args.model in M2_MODEL_NAMES else "hrot_fsl"
        hrot_path_text = (
            "shot-decomposed single-budget episode-calibrated transport "
            if args.model in M2_MODEL_NAMES
            else "Poincare-ball geometry -> balanced/unbalanced relational transport "
        )
        print(
            f"  {hrot_label}: backbone spatial tokens -> optional Euclidean projector -> "
            f"{hrot_path_text}"
            f"(variant={hrot_variant}, "
            f"token_dim={hrot_token_dim}, "
            f"raw_backbone_tokens={getattr(args, 'hrot_use_raw_backbone_tokens', 'false')}, "
            f"eam_hidden={getattr(args, 'hrot_eam_hidden_dim', 256)}, "
            f"curvature_init={getattr(args, 'hrot_curvature_init', 1.0)}, "
            f"proj_scale={getattr(args, 'hrot_projection_scale', 0.1)}, "
            f"token_temp={getattr(args, 'hrot_token_temperature', 0.1)}, "
            f"score_scale={getattr(args, 'hrot_score_scale', 16.0)}, "
            f"tau_q={getattr(args, 'hrot_tau_q', 0.5)}, "
            f"tau_c={getattr(args, 'hrot_tau_c', 0.5)}, "
            f"sinkhorn_eps={getattr(args, 'hrot_sinkhorn_epsilon', 0.1)}, "
            f"sinkhorn_iters={getattr(args, 'hrot_sinkhorn_iterations', 60)}, "
            f"hyp_backend={getattr(args, 'hrot_hyperbolic_backend', 'auto')}, "
            f"ot_backend={getattr(args, 'hrot_ot_backend', 'native')}, "
            f"fixed_mass={getattr(args, 'hrot_fixed_mass', 0.8)}, "
            f"egtw_tau={getattr(args, 'hrot_egtw_tau', 1.0)}, "
            f"egtw_lambda={getattr(args, 'hrot_egtw_lambda', 1.0)}, "
            f"egtw_attn_temp={getattr(args, 'hrot_egtw_attention_temperature', 0.2)}, "
            f"egtw_support_sim={getattr(args, 'hrot_egtw_support_similarity_weight', 0.5)}, "
            f"egtw_uniform_mix={getattr(args, 'hrot_egtw_uniform_mix', 0.25)}, "
            f"egtw_detach={getattr(args, 'hrot_egtw_detach_masses', 'true')}, "
            f"pre_transport_shot_pool={getattr(args, 'hrot_pre_transport_shot_pool', 'false')}, "
            f"tsw_enable={getattr(args, 'hrot_tsw_enable', 'false')}, "
            f"tsw_share_gate={getattr(args, 'hrot_tsw_share_gate', 'true')}, "
            f"ecot_rho_bank={hrot_ecot_rho_bank}, "
            f"ecot_base_rho={hrot_ecot_base_rho}, "
            f"ecot_budget_tau={getattr(args, 'hrot_ecot_budget_tau', 1.0)}, "
            f"ecot_lambda_init={getattr(args, 'hrot_ecot_lambda_init', -8.0)}, "
            f"ecot_controller_hidden={getattr(args, 'hrot_ecot_controller_hidden', 32)}, "
            f"ecot_uniform_budget_policy={getattr(args, 'hrot_ecot_uniform_budget_policy', 'false')}, "
            f"ecot_tau_shot={getattr(args, 'hrot_ecot_enable_tau_shot', 'true')}, "
            f"ecot_consensus_tau_mode={getattr(args, 'hrot_ecot_consensus_tau_mode', 'fixed')}, "
            f"ecot_consensus_tau={getattr(args, 'hrot_ecot_consensus_tau', 1.0)}, "
            f"ecot_nncs_enable={getattr(args, 'hrot_ecot_enable_nncs_marginal', None)}, "
            f"ecot_nncs_beta={getattr(args, 'hrot_ecot_nncs_beta', 5.0)}, "
            f"ecot_nncs_eta={getattr(args, 'hrot_ecot_nncs_eta', 0.05)}, "
            f"ecot_nncs_detach={getattr(args, 'hrot_ecot_nncs_detach', 'true')}, "
            f"ecot_nncs_distance={getattr(args, 'hrot_ecot_nncs_distance', 'auto')}, "
            f"ecot_nncs_min_sigma={getattr(args, 'hrot_ecot_nncs_min_sigma', 1e-6)}, "
            f"ecot_crs_enable={getattr(args, 'hrot_ecot_enable_crs_marginal', 'false')}, "
            f"ecot_crs_cross_ref={getattr(args, 'hrot_ecot_crs_use_cross_ref', 'true')}, "
            f"ecot_crs_ssm={getattr(args, 'hrot_ecot_crs_use_ssm', 'true')}, "
            f"ecot_crs_side={getattr(args, 'hrot_ecot_crs_side', 'support')}, "
            f"ecot_mea_enable={getattr(args, 'hrot_ecot_enable_mea_marginal', 'false')}, "
            f"ecot_mea_eta={getattr(args, 'hrot_ecot_mea_eta_init', 0.35)}, "
            f"ecot_mea_temp={getattr(args, 'hrot_ecot_mea_temperature_init', 0.70)}, "
            f"ecot_ccdm_enable={getattr(args, 'hrot_ecot_enable_ccdm_marginal', 'false')}, "
            f"ecot_ccdm_tau_q={getattr(args, 'hrot_ecot_ccdm_tau_q_init', 0.50)}, "
            f"ecot_ccdm_tau_b={getattr(args, 'hrot_ecot_ccdm_tau_b_init', 0.50)}, "
            f"ecot_ccdm_ew={getattr(args, 'hrot_ecot_ccdm_entropy_shot_weight', 0.0)}, "
            f"ecot_transport_mode={ecot_transport_mode}, "
            f"ecot_noise_sink={getattr(args, 'hrot_ecot_enable_noise_sink', 'false')}, "
            f"ecot_noise_sink_cost={getattr(args, 'hrot_ecot_noise_sink_cost_init', 1.0)}, "
            f"ecot_episode_feat_norm={getattr(args, 'hrot_ecot_episode_feature_normalize', 'false')}, "
            f"ecot_episode_feat_norm_eps={getattr(args, 'hrot_ecot_episode_feature_norm_eps', 1e-6)}, "
            f"amp_marginals={getattr(args, 'hrot_amp_marginals', 'false')}, "
            f"amp_marginals_tau={getattr(args, 'hrot_amp_marginals_tau', 1.0)}, "
            f"token_center={getattr(args, 'hrot_token_center', 'false')}, "
            f"token_center_query={getattr(args, 'hrot_token_center_query', 'true')}, "
            f"care_fwec={getattr(args, 'care_enable_fwec', 'true')}, "
            f"care_qesm={getattr(args, 'care_enable_qesm', 'true')}, "
            f"care_mdr={getattr(args, 'care_enable_mdr', 'true')}, "
            f"care_mdr_lambda={getattr(args, 'care_mdr_lambda', 0.1)}, "
            f"ncet_mix_init={getattr(args, 'hrot_ncet_mix_init', 0.25)}, "
            f"ncet_real_penalty_init={getattr(args, 'hrot_ncet_real_penalty_init', 0.25)}, "
            f"ncet_null_penalty_init={getattr(args, 'hrot_ncet_null_penalty_init', 0.05)}, "
            f"ncet_sink_cost_init={getattr(args, 'hrot_ncet_sink_cost_init', 1.0)}, "
            f"transport_cost_threshold={getattr(args, 'hrot_transport_cost_threshold_init', None)}, "
            f"lambda_rho={getattr(args, 'hrot_lambda_rho', 0.01)}, "
            f"lambda_rho_rank={getattr(args, 'hrot_lambda_rho_rank', 0.05)}, "
            f"rho_rank_margin={getattr(args, 'hrot_rho_rank_margin', 0.05)}, "
            f"rho_rank_temperature={getattr(args, 'hrot_rho_rank_temperature', 0.05)}, "
            f"structure_cost_init={getattr(args, 'hrot_structure_cost_init', 0.05)}, "
            f"ground_cost={getattr(args, 'hrot_ground_cost', 'auto')}, "
            f"eam_mode={getattr(args, 'hrot_eam_mode', 'compact')}, "
            f"compact_eam_prior_mix={getattr(args, 'hrot_compact_eam_prior_mix', 0.5)}, "
            f"normalize_rho={getattr(args, 'hrot_normalize_rho', 'false')})"
        )
    if args.model == "ec_mrot":
        hrot_raw_tokens = _bool_flag(getattr(args, "hrot_use_raw_backbone_tokens", "false"), default=False)
        hrot_token_dim = "raw_backbone" if hrot_raw_tokens else (
            getattr(args, "hrot_token_dim", None) or getattr(args, "token_dim", 128)
        )
        mass_response_grid = getattr(args, "hrot_ecot_rho_bank", None) or "0.40,0.55,0.70,0.85,0.95"
        base_budget = getattr(args, "hrot_ecot_base_rho", None)
        if base_budget is None:
            base_budget = 0.70
        print(
            "  ec_mrot: backbone spatial tokens -> mass-response UOT quadrature -> "
            "anchor-free competitive response functional -> "
            "entropic support-risk aggregation "
            f"(token_dim={hrot_token_dim}, "
            f"raw_backbone_tokens={getattr(args, 'hrot_use_raw_backbone_tokens', 'false')}, "
            f"response_mode={getattr(args, 'ec_mrot_response_mode', 'anchor_free_functional')}, "
            f"mass_response_grid={mass_response_grid}, "
            f"legacy_controller_base_budget={base_budget}, "
            f"budget_tau={getattr(args, 'hrot_ecot_budget_tau', 1.0)}, "
            f"homotopy_lambda_init={getattr(args, 'hrot_ecot_lambda_init', -8.0)}, "
            f"identity_reg={getattr(args, 'hrot_ecot_identity_reg', 1e-4)}, "
            f"uniform_measure={getattr(args, 'hrot_ecot_uniform_budget_policy', 'false')}, "
            f"support_risk_temperature={getattr(args, 'hrot_ecot_enable_tau_shot', 'true')}, "
            f"learn_response_grid={getattr(args, 'ec_mrot_learn_response_grid', 'false')}, "
            f"budget_prior={getattr(args, 'ec_mrot_budget_prior', 'uniform')}, "
            f"budget_kl_reg={getattr(args, 'ec_mrot_budget_kl_reg', 0.0)}, "
            f"budget_entropy_reg={getattr(args, 'ec_mrot_budget_entropy_reg', 0.0)}, "
            f"homotopy_schedule={getattr(args, 'ec_mrot_homotopy_schedule', 'none')}, "
            f"response_strength_init={getattr(args, 'ec_mrot_response_strength_init', 0.35)}, "
            f"response_temperature={getattr(args, 'ec_mrot_response_temperature', 1.0)}, "
            f"competitive_center={getattr(args, 'ec_mrot_competitive_center', 'true')}, "
            f"local_response_gain={getattr(args, 'ec_mrot_local_response_gain', 2.0)}, "
            f"episode_prior_gain={getattr(args, 'ec_mrot_episode_prior_gain', 0.10)})"
        )
    if args.model == "crj_fsl":
        crj_raw_tokens = _bool_flag(
            getattr(args, "crj_use_raw_backbone_tokens", getattr(args, "hrot_use_raw_backbone_tokens", "false")),
            default=False,
        )
        crj_token_dim = "raw_backbone" if crj_raw_tokens else (
            getattr(args, "crj_token_dim", None)
            or getattr(args, "hrot_token_dim", None)
            or getattr(args, "token_dim", 128)
        )
        print(
            "  crj_fsl: J-FSL fixed-budget shot-decomposed UOT -> "
            "consensus-robust lower-confidence shot aggregation "
            f"(token_dim={crj_token_dim}, "
            f"fixed_mass={getattr(args, 'hrot_fixed_mass', 0.8)}, "
            f"score_scale={getattr(args, 'hrot_score_scale', 16.0)}, "
            f"sinkhorn_eps={getattr(args, 'hrot_sinkhorn_epsilon', 0.1)}, "
            f"sinkhorn_iters={getattr(args, 'hrot_sinkhorn_iterations', 60)}, "
            f"ground_cost={getattr(args, 'hrot_ground_cost', 'auto')}, "
            f"pool_gamma={getattr(args, 'crj_pool_gamma', 0.65)}, "
            f"variance_penalty={getattr(args, 'crj_variance_penalty', 0.25)}, "
            f"min_shots={getattr(args, 'crj_min_shots', 2)}, "
            f"trim_fraction={getattr(args, 'crj_trim_fraction', 0.0)}, "
            f"clip_logit={getattr(args, 'crj_clip_logit', 0.0)})"
        )
    if args.model == "fgwuot_fsl":
        print(
            "  fgwuot_fsl: FGW-UOT over spatial tokens -> J-style shot pooling "
            f"(token_dim={getattr(args, 'fgwuot_token_dim', 128)}, "
            f"shot_agg={getattr(args, 'fgwuot_shot_aggregation', 'j_logmeanexp')}, "
            f"alpha_init={getattr(args, 'fgwuot_alpha_init', 0.5)}, "
            f"tau={getattr(args, 'fgwuot_tau', 0.5)}, "
            f"sinkhorn_eps={getattr(args, 'fgwuot_eps_sinkhorn', 0.1)}, "
            f"fgw_iters={getattr(args, 'fgwuot_fgw_iters', 8)})"
        )
    if args.model == "jsc_wdro":
        print(
            "  jsc_wdro: backbone spatial tokens -> class Wasserstein barycenter -> "
            "adaptive WDRO radius -> robust query-class OT score "
            f"(token_dim={getattr(args, 'jsc_wdro_token_dim', None) or getattr(args, 'token_dim', 128)}, "
            f"score_scale={getattr(args, 'jsc_wdro_score_scale', 16.0)}, "
            f"sinkhorn_eps={getattr(args, 'jsc_wdro_sinkhorn_epsilon', 0.05)}, "
            f"sinkhorn_iters={getattr(args, 'jsc_wdro_sinkhorn_iterations', 80)}, "
            f"bary_iters={getattr(args, 'jsc_wdro_barycenter_iterations', 40)}, "
            f"bary_transport={getattr(args, 'jsc_wdro_barycenter_transport', 'unbalanced')}, "
            f"bary_tau={getattr(args, 'jsc_wdro_barycenter_tau', 0.5)}, "
            f"query_transport={getattr(args, 'jsc_wdro_query_transport', 'balanced')}, "
            f"epsilon_alpha={getattr(args, 'jsc_wdro_epsilon_alpha_init', 0.05)}, "
            f"epsilon_beta={getattr(args, 'jsc_wdro_epsilon_beta_init', 0.25)}, "
            f"epsilon_reg={getattr(args, 'jsc_wdro_epsilon_reg_weight', 0.0)}, "
            f"unbalanced_score={getattr(args, 'jsc_wdro_unbalanced_score_mode', 'uot_objective')}, "
            f"token_weighting={getattr(args, 'jsc_wdro_token_weighting', 'uniform')}, "
            f"ot_backend={getattr(args, 'jsc_wdro_ot_backend', 'pot')})"
        )
    if args.model in {"ubt_fsl", "cbcr_fsl"}:
        primary_prefix = "cbcr_fsl" if args.model == "cbcr_fsl" else "ubt_fsl"
        fallback_prefix = "ubt_fsl" if primary_prefix == "cbcr_fsl" else "cbcr_fsl"

        def _method_arg(suffix, default=None):
            value = getattr(args, f"{primary_prefix}_{suffix}", None)
            if value is not None:
                return value
            value = getattr(args, f"{fallback_prefix}_{suffix}", None)
            return default if value is None else value

        print(
            f"  {args.model}: UBT-FSL uncertain barycentric class object -> "
            "optional query ambiguity correction -> robust class transport scoring "
            f"(token_dim={_method_arg('token_dim', None) or getattr(args, 'token_dim', 128)}, "
            f"sinkhorn_eps={_method_arg('sinkhorn_epsilon', 0.05)}, "
            f"sinkhorn_iters={_method_arg('sinkhorn_iterations', 80)}, "
            f"bary_iters={_method_arg('barycenter_iterations', 40)}, "
            f"alpha={_method_arg('alpha', 0.3)}, "
            f"beta={_method_arg('beta', 0.1)}, "
            f"query_tau={_method_arg('query_competition_temperature', None) or _method_arg('tau', 0.5)}, "
            f"rho={_method_arg('rho', 1.0)}, "
            f"score_scale={_method_arg('score_scale', 8.0)}, "
            f"bary_method={_method_arg('barycenter_method', 'mixture')}, "
            f"scoring={_method_arg('scoring', 'robust_transport_envelope')}, "
            f"use_query_competition={_method_arg('use_query_competition', 'true')}, "
            f"transport_backend={_method_arg('transport_backend', 'unbalanced_ot')}, "
            f"solver_backend={_method_arg('solver_backend', None) or _method_arg('ot_backend', 'native')}, "
            f"ablation={_method_arg('ablation_mode', 'ubt_full')})"
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
    deterministic = _bool_flag(getattr(args, "cudnn_deterministic", "true"), default=True)
    benchmark = _bool_flag(getattr(args, "cudnn_benchmark", "false"), default=False)

    if deterministic:
        # Required by CUDA/cuBLAS for bitwise-reproducible GEMM paths when deterministic
        # algorithms are enabled.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        benchmark = False

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark
    try:
        torch.use_deterministic_algorithms(deterministic, warn_only=True)
    except Exception as exc:
        print(f"Warning: could not set deterministic algorithms={deterministic}: {exc}")


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


def get_checkpoint_args_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        return {}
    raw_args = checkpoint.get("args")
    if isinstance(raw_args, argparse.Namespace):
        return vars(raw_args)
    if isinstance(raw_args, dict):
        return raw_args
    return {}


def infer_hrot_variant_from_state_dict(state_dict, checkpoint_args=None):
    checkpoint_args = checkpoint_args or {}
    checkpoint_variant = checkpoint_args.get("hrot_variant")
    if checkpoint_variant is not None:
        normalized = str(checkpoint_variant).strip().upper().replace("-", "").replace("_", "")
        if normalized == "RL":
            normalized = "T"
        if normalized in {"W", "NCET", "JNCET", "NUISANCEDECOUPLED", "JNUISANCE"}:
            normalized = "J_NCET"
        if normalized in {"JEGTW", "JE"}:
            normalized = "JE"
        if normalized in {"JECOTNNCS", "ECOTNNCS", "NNCSJECOT"}:
            normalized = "J_ECOT_NNCS"
        if normalized in {"JECOTM2", "ECOTM2", "M2JECOT", "SBECOT", "JECOTSINGLE", "JECOTSINGLEBUDGET"}:
            normalized = "J_ECOT_M2"
        if normalized in {"JECOTCARE", "CAREUOT", "CARE", "JCAREUOT"}:
            normalized = "J_ECOT_CARE"
        if normalized in {"JECOT", "ECOT"}:
            normalized = "J_ECOT"
        if normalized in {"CPECOT"}:
            normalized = "CP_ECOT"
        if normalized in {"JHLM", "JHIERMASS", "JHMASS", "JTAHM"}:
            normalized = "J_ECOT"
        if normalized in {
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
            "I",
            "J",
            "JE",
            "J_ECOT",
            "J_ECOT_M2",
            "J_ECOT_NNCS",
            "J_ECOT_CARE",
            "CP_ECOT",
            "J_NCET",
            "K",
            "L",
            "M",
            "N",
            "O",
            "P",
            "Q",
            "R",
            "S",
            "T",
            "U",
            "V",
        }:
            return normalized

    if not isinstance(state_dict, dict):
        return None

    keys = set(state_dict.keys())

    def has_param(prefix):
        return any(key == prefix or key.startswith(f"{prefix}.") for key in keys)

    eam_in_dim = None
    eam_weight = state_dict.get("eam.network.0.weight")
    if torch.is_tensor(eam_weight) and eam_weight.dim() == 2:
        eam_in_dim = int(eam_weight.shape[1])

    if "w_q" in keys or "raw_tau_token" in keys:
        return "V"
    if "raw_ncet_mix" in keys or "raw_ncet_real_penalty" in keys:
        return "J_NCET"
    if "raw_ecot_lambda" in keys or has_param("episode_controller"):
        return "J_ECOT"
    if has_param("hierarchical_transport_mass"):
        return "J_ECOT"
    if "raw_egtw_tau" in keys or "raw_egtw_lambda" in keys:
        return "JE"
    if has_param("q_eam"):
        if "raw_structure_cost_weight" in keys:
            if "raw_transport_cost_threshold" in keys:
                return "R"
            return "T"
        return "Q"
    if has_param("euclidean_eam") and has_param("reduced_geodesic_eam"):
        return "M"
    if "raw_transport_cost_threshold" in keys:
        if has_param("q_shot_pool_scorer") and not has_param("q_threshold_scorer"):
            return "S"
        if "raw_token_temperature" in keys and not has_param("q_eam"):
            return "P"
        if eam_in_dim == 8:
            return "O"
        if eam_in_dim == 3:
            return "L"
        if eam_in_dim == 4:
            return "H"
        return "J"
    if eam_in_dim == 4:
        return "K"
    return "E"


def infer_hrot_arch_overrides_from_state_dict(state_dict, checkpoint_args=None):
    checkpoint_args = checkpoint_args or {}
    overrides = {}
    variant = infer_hrot_variant_from_state_dict(state_dict, checkpoint_args=checkpoint_args)
    if variant is not None:
        overrides["hrot_variant"] = variant
        for egtw_key in (
            "hrot_egtw_tau",
            "hrot_egtw_lambda",
            "hrot_egtw_eps",
            "hrot_egtw_attention_temperature",
            "hrot_egtw_support_similarity_weight",
            "hrot_egtw_uniform_mix",
            "hrot_egtw_detach_masses",
            "hrot_egtw_learn_tau",
            "hrot_egtw_learn_lambda",
        ):
            if checkpoint_args.get(egtw_key) is not None:
                overrides[egtw_key] = checkpoint_args[egtw_key]
        for ecot_key in (
            "hrot_pre_transport_shot_pool",
            "hrot_ecot_rho_bank",
            "hrot_ecot_base_rho",
            "hrot_ecot_budget_tau",
            "hrot_ecot_max_lambda",
            "hrot_ecot_lambda_init",
            "hrot_ecot_controller_hidden",
            "hrot_ecot_uniform_budget_policy",
            "hrot_ecot_enable_tau_shot",
            "hrot_ecot_tau_shot_min",
            "hrot_ecot_tau_shot_max",
            "hrot_ecot_enable_threshold_offset",
            "hrot_ecot_m2_ablate_threshold_mass",
            "hrot_ecot_m2_cost_per_mass_score",
            "hrot_ecot_m2_cost_per_mass_alpha",
            "hrot_ecot_m2_cost_per_mass_detach_mass",
            "hrot_ecot_m2_use_swts",
            "hrot_ecot_m2_swts_temp",
            "hrot_ecot_identity_reg",
            "hrot_ecot_policy_entropy_reg",
            "hrot_ecot_consensus_tau_mode",
            "hrot_ecot_consensus_tau",
            "hrot_ecot_enable_nncs_marginal",
            "hrot_ecot_nncs_beta",
            "hrot_ecot_nncs_eta",
            "hrot_ecot_nncs_detach",
            "hrot_ecot_nncs_distance",
            "hrot_ecot_nncs_min_sigma",
            "hrot_ecot_enable_crs_marginal",
            "hrot_ecot_crs_use_cross_ref",
            "hrot_ecot_crs_use_ssm",
            "hrot_ecot_crs_eta_init",
            "hrot_ecot_crs_lambda_cr_init",
            "hrot_ecot_crs_tau_ssm_init",
            "hrot_ecot_crs_entropy_reg",
            "hrot_ecot_crs_side",
            "hrot_ecot_crs_ssm_type",
            "hrot_ecot_enable_mea_marginal",
            "hrot_ecot_mea_eta_init",
            "hrot_ecot_mea_temperature_init",
            "hrot_ecot_mea_entropy_reg",
            "hrot_ecot_enable_ccdm_marginal",
            "hrot_ecot_ccdm_tau_q_init",
            "hrot_ecot_ccdm_tau_b_init",
            "hrot_ecot_ccdm_entropy_reg",
            "hrot_ecot_ccdm_entropy_shot_weight",
            "hrot_ecot_enable_noise_sink",
            "hrot_ecot_noise_sink_cost_init",
            "hrot_ecot_noise_sink_score_penalty",
            "hrot_ecot_episode_feature_normalize",
            "hrot_ecot_episode_feature_norm_eps",
            "hrot_amp_marginals",
            "hrot_amp_marginals_tau",
            "hrot_token_center",
            "hrot_token_center_query",
            "care_enable_fwec",
            "care_enable_qesm",
            "care_enable_mdr",
            "care_fwec_eps",
            "care_fwec_w_clamp_min",
            "care_fwec_w_clamp_max",
            "care_qesm_tau_e",
            "care_mdr_margin",
            "care_mdr_lambda",
        ):
            if checkpoint_args.get(ecot_key) is not None:
                overrides[ecot_key] = checkpoint_args[ecot_key]
        if (
            "hrot_ecot_enable_crs_marginal" not in overrides
            and any(key.startswith("crs_marginal.") for key in state_dict)
        ):
            overrides["hrot_ecot_enable_crs_marginal"] = "true"
        if (
            "hrot_ecot_enable_mea_marginal" not in overrides
            and any(key.startswith("mea_marginal.") for key in state_dict)
        ):
            overrides["hrot_ecot_enable_mea_marginal"] = "true"
        if (
            "hrot_ecot_enable_ccdm_marginal" not in overrides
            and any(key.startswith("ccdm_marginal.") for key in state_dict)
        ):
            overrides["hrot_ecot_enable_ccdm_marginal"] = "true"
        for ncet_key in (
            "hrot_ncet_mix_init",
            "hrot_ncet_real_penalty_init",
            "hrot_ncet_null_penalty_init",
            "hrot_ncet_sink_cost_init",
        ):
            if checkpoint_args.get(ncet_key) is not None:
                overrides[ncet_key] = checkpoint_args[ncet_key]
        if checkpoint_args.get("hrot_eam_mode") is not None:
            overrides["hrot_eam_mode"] = str(checkpoint_args["hrot_eam_mode"]).strip().lower().replace("-", "_")
        elif variant == "H":
            overrides["hrot_eam_mode"] = "legacy"
        if checkpoint_args.get("hrot_compact_eam_prior_mix") is not None:
            overrides["hrot_compact_eam_prior_mix"] = float(checkpoint_args["hrot_compact_eam_prior_mix"])

    if "hrot_use_raw_backbone_tokens" in checkpoint_args:
        raw_flag = _bool_flag(checkpoint_args.get("hrot_use_raw_backbone_tokens"), default=False)
        overrides["hrot_use_raw_backbone_tokens"] = "true" if raw_flag else "false"
    else:
        uses_raw = not any(key.startswith("token_projector.") for key in state_dict)
        overrides["hrot_use_raw_backbone_tokens"] = "true" if uses_raw else "false"

    token_dim = None
    if checkpoint_args.get("hrot_token_dim") is not None:
        token_dim = int(checkpoint_args["hrot_token_dim"])
    elif checkpoint_args.get("token_dim") is not None:
        token_dim = int(checkpoint_args["token_dim"])
    else:
        projector_weight = state_dict.get("token_projector.1.weight")
        if torch.is_tensor(projector_weight) and projector_weight.dim() == 2:
            token_dim = int(projector_weight.shape[0])
        else:
            for vector_key in ("query_token_attention_vector", "support_token_attention_vector", "w_q", "w_s"):
                vector = state_dict.get(vector_key)
                if torch.is_tensor(vector) and vector.dim() == 1:
                    token_dim = int(vector.shape[0])
                    break
            if token_dim is None:
                eam_weight = state_dict.get("eam.network.0.weight")
                if torch.is_tensor(eam_weight) and eam_weight.dim() == 2:
                    in_dim = int(eam_weight.shape[1])
                    if in_dim >= 5 and (in_dim - 3) % 2 == 0:
                        token_dim = (in_dim - 3) // 2
    if token_dim is not None:
        overrides["hrot_token_dim"] = int(token_dim)

    eam_hidden = None
    if checkpoint_args.get("hrot_eam_hidden_dim") is not None:
        eam_hidden = int(checkpoint_args["hrot_eam_hidden_dim"])
    else:
        eam_weight = state_dict.get("eam.network.0.weight")
        if torch.is_tensor(eam_weight) and eam_weight.dim() == 2:
            eam_hidden = int(eam_weight.shape[0])
    if eam_hidden is not None:
        overrides["hrot_eam_hidden_dim"] = int(eam_hidden)

    return overrides


def maybe_autoconfigure_hrot_from_checkpoint(args):
    if getattr(args, "model", None) not in HROT_FSL_MODEL_NAMES:
        return args

    checkpoint_path = getattr(args, "resume", None) or getattr(args, "weights", None)
    if not checkpoint_path:
        return args

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = strip_module_prefix(unwrap_model_state(checkpoint))
    checkpoint_args = get_checkpoint_args_dict(checkpoint)
    overrides = infer_hrot_arch_overrides_from_state_dict(state_dict, checkpoint_args=checkpoint_args)

    changed = []
    for key, value in overrides.items():
        current = getattr(args, key, None)
        if current != value:
            setattr(args, key, value)
            changed.append((key, value))

    if changed:
        changed_text = ", ".join(f"{key}={value}" for key, value in changed)
        print(f"HROT checkpoint-aligned config from {checkpoint_path}: {changed_text}")
    return args


def apply_standalone_m2_defaults(args):
    if getattr(args, "model", None) not in M2_MODEL_NAMES:
        return args
    full_ot = args.model in M2_FULL_OT_MODEL_NAMES
    default_rho = 1.0 if full_ot else 0.80
    args.hrot_variant = "J_ECOT_M2"
    if getattr(args, "hrot_ecot_rho_bank", None) is None:
        args.hrot_ecot_rho_bank = f"{default_rho:.6g}"
    if getattr(args, "hrot_ecot_base_rho", None) is None:
        args.hrot_ecot_base_rho = default_rho
    if getattr(args, "hrot_ecot_transport_mode", None) is None:
        args.hrot_ecot_transport_mode = "balanced" if full_ot else "unbalanced"
    return args


def load_model_weights(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = strip_module_prefix(unwrap_model_state(checkpoint))
    model.load_state_dict(state_dict)
    return checkpoint


def get_checkpoint_stem(args):
    samples_suffix = f"{args.training_samples}samples" if args.training_samples else "all"
    tag_suffix = f"_{args.experiment_tag}" if getattr(args, "experiment_tag", "") else ""
    return os.path.join(
        args.path_weights,
        f"{args.dataset_name}_{args.model}_{samples_suffix}_{args.shot_num}shot{tag_suffix}",
    )


def get_best_model_path(args):
    return f"{get_checkpoint_stem(args)}_final.pth"


def get_last_checkpoint_path(args):
    return f"{get_checkpoint_stem(args)}_last.ckpt"


def get_model_selection_split(args) -> str:
    return str(getattr(args, "model_selection_split", "val")).lower()


def build_episode_seed(args, split_name: str, epoch: int | None = None) -> int:
    split_name = str(split_name).lower()
    if split_name == "train":
        offset = int(getattr(args, "train_episode_seed_offset", 0))
        use_epoch = True
    elif split_name in {"selection", "select", "eval", "val", "test"}:
        offset = int(getattr(args, "selection_episode_seed_offset", 1))
        use_epoch = str(getattr(args, "selection_episode_seed_mode", "fixed")).lower() == "per_epoch"
    elif split_name in {"final_test", "final"}:
        explicit_seed = getattr(args, "final_test_seed", None)
        if explicit_seed is not None:
            return int(explicit_seed)
        offset = int(getattr(args, "final_test_episode_seed_offset", 0))
        use_epoch = False
    else:
        raise ValueError(f"Unsupported split_name={split_name}")

    seed = int(getattr(args, "seed", 0)) + offset
    if use_epoch and epoch is not None:
        seed += int(epoch)
    return seed


def get_selection_protocol(args):
    split = get_model_selection_split(args)
    if split == "test":
        return split, int(args.episode_num_test), int(args.query_num_test)
    return split, int(args.episode_num_val), int(args.query_num_val)


def concat_split_tensors(primary_X, primary_y, secondary_X, secondary_y):
    if secondary_X is None or secondary_y is None or len(secondary_X) == 0:
        return primary_X, primary_y
    if primary_X is None or primary_y is None or len(primary_X) == 0:
        return secondary_X, secondary_y
    return torch.cat([primary_X, secondary_X], dim=0), torch.cat([primary_y, secondary_y], dim=0)


def infer_source_ids(file_paths):
    if file_paths is None:
        return None
    return [os.path.splitext(os.path.basename(path))[0] for path in file_paths]


def sanitize_result_tag(value):
    tag = re.sub(r"[^A-Za-z0-9]+", "_", str(value).strip()).strip("_").lower()
    return tag or "test"


def snr_sort_key(split_name):
    name = str(split_name).lower()
    match = re.search(r"snr(?:_minus)?(\d+)db", name)
    if not match:
        return (1, name)
    value = int(match.group(1))
    if "snr_minus" in name:
        value = -value
    return (0, value, name)


def split_has_class_dirs(split_path):
    if not os.path.isdir(split_path):
        return False
    expected = {"surface", "internal", "corona", "notpd", "nopd", "not_pd"}
    child_dirs = {
        name.lower()
        for name in os.listdir(split_path)
        if os.path.isdir(os.path.join(split_path, name))
    }
    return bool(child_dirs & expected)


def discover_noise_test_splits(noise_test_root, requested_splits="auto"):
    if not noise_test_root:
        return []
    root = os.path.abspath(noise_test_root)
    if not os.path.isdir(root):
        raise ValueError(f"noise_test_root not found: {root}")

    requested = str(requested_splits or "auto").strip()
    if requested.lower() == "auto":
        names = [
            name
            for name in os.listdir(root)
            if os.path.isdir(os.path.join(root, name))
            and name.lower().startswith("test_snr")
            and split_has_class_dirs(os.path.join(root, name))
        ]
        if not names:
            names = [
                name
                for name in os.listdir(root)
                if os.path.isdir(os.path.join(root, name))
                and split_has_class_dirs(os.path.join(root, name))
            ]
        names = sorted(names, key=snr_sort_key)
    else:
        names = [name.strip() for name in requested.split(",") if name.strip()]

    splits = []
    for name in names:
        split_path = os.path.join(root, name)
        if not os.path.isdir(split_path):
            raise ValueError(f"Noise test split not found: {split_path}")
        if not split_has_class_dirs(split_path):
            raise ValueError(f"Noise test split has no class folders: {split_path}")
        splits.append((name, split_path))
    return splits


def load_noise_test_pools(args, dataset):
    splits = discover_noise_test_splits(
        getattr(args, "noise_test_root", None),
        getattr(args, "noise_test_splits", "auto"),
    )
    pools = {}
    for split_name, split_path in splits:
        images, labels, files = load_external_scalogram_split(
            split_path,
            transform=dataset.transform,
            classes=getattr(dataset, "classes", None),
        )
        if len(images) == 0:
            raise ValueError(f"Noise test split is empty: {split_path}")
        pools[split_name] = {
            "images": torch.from_numpy(images.astype(np.float32)),
            "labels": torch.from_numpy(labels).long(),
            "files": [path for path, _ in files],
            "path": split_path,
        }
        print(f"Loaded noise test split {split_name}: {len(images)} samples")
    return pools


def get_dataset_split_arrays(dataset, split_name):
    images = getattr(dataset, f"X_{split_name}", None)
    labels = getattr(dataset, f"y_{split_name}", None)
    files = getattr(dataset, f"{split_name}_files", None)
    if images is None or labels is None:
        return None
    return {
        "images": images,
        "labels": labels,
        "files": [path for path, _ in files] if files is not None else None,
    }


def get_effective_test_protocol(args, dataset):
    requested = str(getattr(args, "test_protocol", "auto")).lower()
    has_robust = bool(getattr(dataset, "has_hrot_robust_protocol", False))
    has_noise_root = bool(getattr(args, "noise_test_root", None))
    if requested == "auto":
        if has_noise_root:
            return "noise"
        return "robust" if has_robust else "clean"
    if requested == "noise" and not has_noise_root:
        raise ValueError("--test_protocol noise requires --noise_test_root.")
    if requested == "robust" and not has_robust:
        raise ValueError(
            "--test_protocol robust requires an HROT robust dataset with "
            "test_1shot_support_snr10, test_1shot_query_snr10, "
            "test_5shot_support_outlier_snr0, and test_5shot_query_snr10."
        )
    return requested


def get_test_protocol_suffix(args):
    test_name = str(getattr(args, "current_test_name", "") or "").strip()
    if test_name:
        return f"_{sanitize_result_tag(test_name)}"
    protocol = str(getattr(args, "effective_test_protocol", getattr(args, "test_protocol", "clean"))).lower()
    if protocol in {"auto", "clean", ""}:
        return ""
    return f"_{sanitize_result_tag(protocol)}"


def build_robust_test_dataset(args, clean_X, clean_y, clean_file_paths, robust_pools, return_indices=False):
    query_pool_name = "test_1shot_query_snr10" if args.shot_num == 1 else "test_5shot_query_snr10"
    query_pool = robust_pools.get(query_pool_name)
    if query_pool is None:
        raise ValueError(f"Missing robust query pool: {query_pool_name}")

    common_kwargs = {
        "episode_num": args.episode_num_test,
        "way_num": args.way_num,
        "shot_num": args.shot_num,
        "query_num": args.query_num_test,
        "seed": build_episode_seed(args, "final_test"),
        "query_data": query_pool["images"],
        "query_labels": query_pool["labels"],
        "query_source_ids": infer_source_ids(query_pool["files"]),
        "return_indices": return_indices,
    }

    if args.shot_num == 1:
        support_pool_name = "test_1shot_support_snr10"
        support_pool = robust_pools.get(support_pool_name)
        if support_pool is None:
            raise ValueError(f"Missing robust support pool: {support_pool_name}")
        return RobustFewshotDataset(
            support_data=support_pool["images"],
            support_labels=support_pool["labels"],
            support_source_ids=infer_source_ids(support_pool["files"]),
            **common_kwargs,
        )

    if args.shot_num == 5:
        outlier_pool_name = "test_5shot_support_outlier_snr0"
        outlier_pool = robust_pools.get(outlier_pool_name)
        if outlier_pool is None:
            raise ValueError(f"Missing robust support-outlier pool: {outlier_pool_name}")
        return RobustFewshotDataset(
            clean_data=clean_X,
            clean_labels=clean_y,
            clean_source_ids=infer_source_ids(clean_file_paths),
            outlier_data=outlier_pool["images"],
            outlier_labels=outlier_pool["labels"],
            outlier_source_ids=infer_source_ids(outlier_pool["files"]),
            **common_kwargs,
        )

    raise ValueError(f"HROT robust test supports only 1-shot and 5-shot, got {args.shot_num}-shot.")


def forward_scores(
    net,
    query,
    support,
    args,
    train=False,
    phase=None,
    query_targets=None,
    support_targets=None,
    collect_diagnostics=False,
):
    phase = phase or ("train" if train else "eval")
    if args.model == "maml":
        smoothing = effective_label_smoothing(args, train=train)
        return net(query, support, label_smoothing=smoothing)
    if args.model in {"deepemd", "evidence_deepemd"}:
        if phase == "train":
            refine_proto = _bool_flag(args.deepemd_train_sfc, default=False)
            return net(
                query,
                support,
                refine_proto=refine_proto,
                exact=False,
                sfc_update_step_override=getattr(args, "deepemd_train_sfc_update_step", None),
                sfc_bs_override=getattr(args, "deepemd_train_sfc_bs", None),
            )
        if phase == "val":
            fast_val = _bool_flag(args.deepemd_fast_val, default=True)
            return net(query, support, refine_proto=not fast_val, exact=not fast_val)
        if phase == "test":
            exact = _bool_flag(args.deepemd_test_exact, default=False)
            refine_proto = _bool_flag(args.deepemd_test_sfc, default=False)
            return net(query, support, refine_proto=refine_proto, exact=exact)
        return net(query, support)
    if args.model == "dice_emd":
        if phase == "train":
            refine_proto = _bool_flag(args.deepemd_train_sfc, default=False)
            return net(
                query,
                support,
                refine_proto=refine_proto,
                exact=False,
                sfc_update_step_override=getattr(args, "deepemd_train_sfc_update_step", None),
                sfc_bs_override=getattr(args, "deepemd_train_sfc_bs", None),
                return_aux=collect_diagnostics,
                query_targets=query_targets,
                support_targets=support_targets,
            )
        if phase == "val":
            fast_val = _bool_flag(args.deepemd_fast_val, default=True)
            return net(
                query,
                support,
                refine_proto=not fast_val,
                exact=not fast_val,
                return_aux=collect_diagnostics,
                query_targets=query_targets,
                support_targets=support_targets,
            )
        if phase == "test":
            exact = _bool_flag(args.deepemd_test_exact, default=False)
            refine_proto = _bool_flag(args.deepemd_test_sfc, default=False)
            return net(
                query,
                support,
                refine_proto=refine_proto,
                exact=exact,
                return_aux=collect_diagnostics,
                query_targets=query_targets,
                support_targets=support_targets,
            )
        return net(query, support, return_aux=collect_diagnostics)
    if args.model in {"transport_recon_emd", "tardis_emd"}:
        return net(
            query,
            support,
            return_aux=collect_diagnostics,
            query_targets=query_targets,
            support_targets=support_targets,
        )
    if args.model in {"feat", "can"}:
        return net(query, support, query_targets=query_targets, support_targets=support_targets)
    if args.model == "hierarchical_support_sw_net":
        return net(query, support, return_aux=collect_diagnostics)
    if args.model == "sc_lfi":
        return net(query, support, return_aux=collect_diagnostics)
    if args.model == "sc_lfi_v2":
        return net(
            query,
            support,
            query_targets=query_targets,
            support_targets=support_targets,
            return_aux=collect_diagnostics,
        )
    if args.model == "sc_lfi_v3":
        return net(
            query,
            support,
            query_targets=query_targets,
            support_targets=support_targets,
            return_aux=collect_diagnostics,
        )
    if args.model == "sc_lfi_v4":
        return net(
            query,
            support,
            query_targets=query_targets,
            support_targets=support_targets,
            return_aux=collect_diagnostics,
        )
    if args.model == "spifaeb_v2":
        return net(
            query,
            support,
            query_targets=query_targets,
            support_targets=support_targets,
            return_aux=collect_diagnostics,
        )
    if args.model == "spifaeb_v3":
        return net(
            query,
            support,
            query_targets=query_targets,
            support_targets=support_targets,
            return_aux=collect_diagnostics,
        )
    if args.model == "spif_ota":
        return net(
            query,
            support,
            query_targets=query_targets,
            support_targets=support_targets,
            return_aux=collect_diagnostics,
        )
    if args.model in {"evidence_budget_ot", "evidence_budgeted_ot", "ebot", "ebot_scalogram"}:
        return net(
            query,
            support,
            query_targets=query_targets,
            support_targets=support_targets,
            return_aux=collect_diagnostics,
        )
    if args.model in HROT_FSL_MODEL_NAMES or args.model == "ec_mrot":
        return net(
            query,
            support,
            query_targets=query_targets,
            support_targets=support_targets,
            return_aux=collect_diagnostics,
        )
    if args.model == "crj_fsl":
        return net(
            query,
            support,
            query_targets=query_targets,
            support_targets=support_targets,
            return_aux=collect_diagnostics,
        )
    if args.model == "fgwuot_fsl":
        return net(
            query,
            support,
            query_targets=query_targets,
            support_targets=support_targets,
            return_aux=collect_diagnostics,
        )
    if args.model == "jsc_wdro":
        return net(
            query,
            support,
            query_targets=query_targets,
            support_targets=support_targets,
            return_aux=collect_diagnostics,
        )
    if args.model in {"ubt_fsl", "cbcr_fsl"}:
        return net(
            query,
            support,
            query_targets=query_targets,
            support_targets=support_targets,
            return_aux=collect_diagnostics,
        )
    if args.model == "rada_fsl":
        return net(query, support, return_aux=collect_diagnostics)
    if args.model in {
        "spifce",
        "spifmax",
        "spifce_shot",
        "spifce_asym_shot",
        "spifaeb",
        "spifaeb_shot",
        "spif_rdp",
        "spif_otccls",
    }:
        return net(query, support, return_aux=collect_diagnostics)
    if getattr(net, "requires_query_targets", False):
        return net(query, support, query_targets=query_targets, support_targets=support_targets)
    return net(query, support)


def compute_standard_classification_loss(scores, targets, args, contrastive_loss_fn, train=False):
    if getattr(args, "classification_loss", "cross_entropy") == "repo_contrastive":
        return contrastive_loss_fn(scores, targets)
    smoothing = effective_label_smoothing(args, train=train)
    return F.cross_entropy(scores, targets, label_smoothing=smoothing)


def compute_loss(scores, targets, args, relation_loss_fn, contrastive_loss_fn, train=False):
    if isinstance(scores, dict):
        if "loss" in scores:
            return scores["loss"]
        logits = scores["logits"]
        base_loss = compute_loss(logits, targets, args, relation_loss_fn, contrastive_loss_fn, train=train)
        aux_loss = scores.get("aux_loss")
        return base_loss if aux_loss is None else base_loss + aux_loss

    if args.model == "relationnet":
        return relation_loss_fn(scores, targets)
    return compute_standard_classification_loss(scores, targets, args, contrastive_loss_fn, train=train)


def extract_logits(scores):
    if isinstance(scores, dict):
        return scores["logits"]
    return scores


def compute_loss_breakdown(scores, targets, args, relation_loss_fn, contrastive_loss_fn, train=False):
    total_loss = compute_loss(scores, targets, args, relation_loss_fn, contrastive_loss_fn, train=train)
    logits = extract_logits(scores)

    if isinstance(scores, dict) and "loss" in scores:
        cls_loss = (
            compute_loss(logits, targets, args, relation_loss_fn, contrastive_loss_fn, train=train)
            if "logits" in scores
            else total_loss
        )
        aux_loss = total_loss - cls_loss if "logits" in scores else None
        return total_loss, cls_loss, aux_loss

    cls_loss = compute_loss(logits, targets, args, relation_loss_fn, contrastive_loss_fn, train=train)
    aux_loss = scores.get("aux_loss") if isinstance(scores, dict) else None
    return total_loss, cls_loss, aux_loss


def _scalar_metric(value):
    if value is None:
        return None
    if isinstance(value, (float, int)):
        scalar = float(value)
        return scalar if np.isfinite(scalar) else None
    if torch.is_tensor(value):
        if value.numel() == 0:
            return None
        scalar = float(value.detach().mean().item())
        return scalar if np.isfinite(scalar) else None
    return None


def _masked_scalar_metric(values, mask):
    if values is None or mask is None:
        return None
    if not torch.is_tensor(values) or not torch.is_tensor(mask):
        return None
    if values.numel() == 0 or mask.numel() == 0 or not bool(mask.any().item()):
        return None
    return _scalar_metric(values[mask])


def summarize_prediction_diagnostics(logits, targets):
    with torch.no_grad():
        probs = torch.softmax(logits, dim=1)
        true_probs = probs.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)
        true_logits = logits.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)
        if logits.shape[1] > 1:
            mask = F.one_hot(targets, num_classes=logits.shape[1]).bool()
            max_other_logits = logits.masked_fill(mask, float("-inf")).max(dim=1).values
            margin = true_logits - max_other_logits
        else:
            margin = torch.zeros_like(true_logits)
        max_probs, predictions = probs.max(dim=1)
        correct = predictions.eq(targets)
        return {
            "pred_acc": _scalar_metric(correct.float()),
            "true_prob": _scalar_metric(true_probs),
            "max_prob": _scalar_metric(max_probs),
            "correct_true_prob": _masked_scalar_metric(true_probs, correct),
            "wrong_true_prob": _masked_scalar_metric(true_probs, ~correct),
            "correct_max_prob": _masked_scalar_metric(max_probs, correct),
            "wrong_max_prob": _masked_scalar_metric(max_probs, ~correct),
            "logit_margin": _scalar_metric(margin),
            "margin_pos_rate": _scalar_metric((margin > 0.0).float()),
            "correct_logit_margin": _masked_scalar_metric(margin, correct),
            "wrong_logit_margin": _masked_scalar_metric(margin, ~correct),
        }


def summarize_distance_diagnostics(total_distance, targets):
    with torch.no_grad():
        true_distance = total_distance.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)
        if total_distance.shape[1] > 1:
            mask = F.one_hot(targets, num_classes=total_distance.shape[1]).bool()
            best_negative_distance = total_distance.masked_fill(mask, float("inf")).min(dim=1).values
            distance_gap = best_negative_distance - true_distance
        else:
            best_negative_distance = torch.zeros_like(true_distance)
            distance_gap = torch.zeros_like(true_distance)
        return {
            "true_distance": _scalar_metric(true_distance),
            "best_negative_distance": _scalar_metric(best_negative_distance),
            "distance_gap": _scalar_metric(distance_gap),
            "distance_gap_pos_rate": _scalar_metric((distance_gap > 0.0).float()),
            "correct_distance_gap": _masked_scalar_metric(distance_gap, distance_gap > 0.0),
            "wrong_distance_gap": _masked_scalar_metric(distance_gap, distance_gap <= 0.0),
        }


def summarize_class_distance_tensor(distances, targets, prefix):
    with torch.no_grad():
        true_distance = distances.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)
        if distances.shape[1] > 1:
            mask = F.one_hot(targets, num_classes=distances.shape[1]).bool()
            best_negative_distance = distances.masked_fill(mask, float("inf")).min(dim=1).values
            distance_gap = best_negative_distance - true_distance
        else:
            best_negative_distance = torch.zeros_like(true_distance)
            distance_gap = torch.zeros_like(true_distance)
        return {
            f"{prefix}_true_distance": _scalar_metric(true_distance),
            f"{prefix}_best_negative_distance": _scalar_metric(best_negative_distance),
            f"{prefix}_distance_gap": _scalar_metric(distance_gap),
            f"{prefix}_distance_gap_pos_rate": _scalar_metric((distance_gap > 0.0).float()),
            f"{prefix}_correct_distance_gap": _masked_scalar_metric(distance_gap, distance_gap > 0.0),
            f"{prefix}_wrong_distance_gap": _masked_scalar_metric(distance_gap, distance_gap <= 0.0),
        }


def summarize_class_score_tensor(scores, targets, prefix):
    with torch.no_grad():
        true_score = scores.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)
        if scores.shape[1] > 1:
            mask = F.one_hot(targets, num_classes=scores.shape[1]).bool()
            best_negative_score = scores.masked_fill(mask, float("-inf")).max(dim=1).values
            score_gap = true_score - best_negative_score
        else:
            best_negative_score = torch.zeros_like(true_score)
            score_gap = torch.zeros_like(true_score)
        return {
            f"{prefix}_true_score": _scalar_metric(true_score),
            f"{prefix}_best_negative_score": _scalar_metric(best_negative_score),
            f"{prefix}_score_gap": _scalar_metric(score_gap),
        }


def summarize_budget_tensor(values, targets, prefix):
    with torch.no_grad():
        true_value = values.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)
        if values.shape[1] > 1:
            mask = F.one_hot(targets, num_classes=values.shape[1]).bool()
            best_negative_value = values.masked_fill(mask, float("-inf")).max(dim=1).values
            value_gap = true_value - best_negative_value
        else:
            best_negative_value = torch.zeros_like(true_value)
            value_gap = torch.zeros_like(true_value)
        return {
            f"{prefix}_true": _scalar_metric(true_value),
            f"{prefix}_best_negative": _scalar_metric(best_negative_value),
            f"{prefix}_gap": _scalar_metric(value_gap),
        }


def summarize_episode_class_tensor(values, targets, prefix):
    with torch.no_grad():
        if values.dim() == 1:
            values = values.unsqueeze(0)
        if values.dim() != 2:
            return {}
        batch_size, way_num = values.shape
        if targets.numel() % batch_size != 0:
            return {}
        episode_targets = targets.reshape(batch_size, -1)
        true_value = values.gather(dim=1, index=episode_targets).reshape(-1)
        if way_num > 1:
            mask = F.one_hot(episode_targets, num_classes=way_num).bool()
            expanded = values.unsqueeze(1).expand(-1, episode_targets.shape[1], -1)
            best_negative_value = expanded.masked_fill(mask, float("-inf")).max(dim=-1).values.reshape(-1)
            value_gap = true_value - best_negative_value
        else:
            best_negative_value = torch.zeros_like(true_value)
            value_gap = torch.zeros_like(true_value)
        return {
            f"{prefix}_true": _scalar_metric(true_value),
            f"{prefix}_best_negative": _scalar_metric(best_negative_value),
            f"{prefix}_gap": _scalar_metric(value_gap),
        }


def summarize_score_diagnostics(scores, logits, targets, cls_loss=None, aux_loss=None):
    metrics = summarize_prediction_diagnostics(logits, targets)
    metrics["cls_loss"] = _scalar_metric(cls_loss)
    metrics["aux_loss"] = _scalar_metric(aux_loss)

    if not isinstance(scores, dict):
        return metrics

    total_distance = scores.get("total_distance")
    if torch.is_tensor(total_distance) and total_distance.dim() == 2 and total_distance.shape[0] == targets.shape[0]:
        metrics.update(summarize_distance_diagnostics(total_distance, targets))

    query_class_distance = scores.get("query_class_distance")
    if (
        torch.is_tensor(query_class_distance)
        and query_class_distance.dim() == 2
        and query_class_distance.shape[0] == targets.shape[0]
    ):
        metrics.update(summarize_class_distance_tensor(query_class_distance, targets, prefix="query_class"))

    epsilon = scores.get("epsilon")
    if torch.is_tensor(epsilon):
        metrics.update(summarize_episode_class_tensor(epsilon, targets, prefix="epsilon"))

    support_dispersion = scores.get("support_dispersion")
    if torch.is_tensor(support_dispersion):
        metrics.update(summarize_episode_class_tensor(support_dispersion, targets, prefix="support_dispersion"))

    transported_mass = scores.get("transported_mass")
    if torch.is_tensor(transported_mass) and transported_mass.dim() == 2 and transported_mass.shape[0] == targets.shape[0]:
        metrics.update(summarize_budget_tensor(transported_mass, targets, prefix="transport_mass"))

    weighted_unmatched_mass = scores.get("weighted_unmatched_mass")
    if (
        torch.is_tensor(weighted_unmatched_mass)
        and weighted_unmatched_mass.dim() == 2
        and weighted_unmatched_mass.shape[0] == targets.shape[0]
    ):
        metrics.update(summarize_class_distance_tensor(weighted_unmatched_mass, targets, prefix="unmatched_mass"))

    global_scores = scores.get("global_scores")
    if torch.is_tensor(global_scores) and global_scores.dim() == 2 and global_scores.shape[0] == targets.shape[0]:
        metrics.update(summarize_class_score_tensor(global_scores, targets, prefix="global"))

    component_score_keys = {
        "raw_logits": "raw",
        "score_emd": "emd",
        "score_rec": "recon",
        "score_struct": "struct",
    }
    for score_key, prefix in component_score_keys.items():
        component_scores = scores.get(score_key)
        if (
            torch.is_tensor(component_scores)
            and component_scores.dim() == 2
            and component_scores.shape[0] == targets.shape[0]
        ):
            metrics.update(summarize_class_score_tensor(component_scores, targets, prefix=prefix))

    local_scores = scores.get("local_scores")
    if torch.is_tensor(local_scores) and local_scores.dim() == 2 and local_scores.shape[0] == targets.shape[0]:
        metrics.update(summarize_class_score_tensor(local_scores, targets, prefix="local"))
        if torch.is_tensor(global_scores) and global_scores.shape == local_scores.shape:
            metrics["global_local_agreement"] = _scalar_metric(
                global_scores.argmax(dim=1).eq(local_scores.argmax(dim=1)).float().mean()
            )
    crj_score_tensors = {
        "crj_vanilla_logits": "crj_vanilla",
        "crj_robust_lcb": "crj_lcb",
        "crj_logit_delta": "crj_delta",
    }
    for score_key, prefix in crj_score_tensors.items():
        score_tensor = scores.get(score_key)
        if torch.is_tensor(score_tensor) and score_tensor.dim() == 2 and score_tensor.shape[0] == targets.shape[0]:
            metrics.update(summarize_class_score_tensor(score_tensor, targets, prefix=prefix))

    crj_budget_tensors = {
        "crj_shot_score_std": "crj_shot_std",
        "crj_consensus_confidence": "crj_consensus",
        "crj_effective_shots": "crj_effective_shots",
    }
    for budget_key, prefix in crj_budget_tensors.items():
        budget_tensor = scores.get(budget_key)
        if (
            torch.is_tensor(budget_tensor)
            and budget_tensor.dim() == 2
            and budget_tensor.shape[0] == targets.shape[0]
        ):
            metrics.update(summarize_budget_tensor(budget_tensor, targets, prefix=prefix))
    anchor_local_scores = scores.get("anchor_local_scores")
    if torch.is_tensor(anchor_local_scores) and anchor_local_scores.dim() == 2 and anchor_local_scores.shape[0] == targets.shape[0]:
        metrics.update(summarize_class_score_tensor(anchor_local_scores, targets, prefix="anchor_local"))
    adaptive_local_scores = scores.get("adaptive_local_scores")
    if (
        torch.is_tensor(adaptive_local_scores)
        and adaptive_local_scores.dim() == 2
        and adaptive_local_scores.shape[0] == targets.shape[0]
    ):
        metrics.update(summarize_class_score_tensor(adaptive_local_scores, targets, prefix="adaptive_local"))
    raw_local_scores = scores.get("raw_local_scores")
    if torch.is_tensor(raw_local_scores) and raw_local_scores.dim() == 2 and raw_local_scores.shape[0] == targets.shape[0]:
        metrics.update(summarize_class_score_tensor(raw_local_scores, targets, prefix="raw_local"))
    competitive_local_scores = scores.get("competitive_local_scores")
    if (
        torch.is_tensor(competitive_local_scores)
        and competitive_local_scores.dim() == 2
        and competitive_local_scores.shape[0] == targets.shape[0]
    ):
        metrics.update(summarize_class_score_tensor(competitive_local_scores, targets, prefix="competitive_local"))

    rho = scores.get("rho")
    if torch.is_tensor(rho) and rho.dim() == 2 and rho.shape[0] == targets.shape[0]:
        metrics.update(summarize_budget_tensor(rho, targets, prefix="budget"))
    rho_prior = scores.get("rho_prior")
    if torch.is_tensor(rho_prior) and rho_prior.dim() == 2 and rho_prior.shape[0] == targets.shape[0]:
        metrics.update(summarize_budget_tensor(rho_prior, targets, prefix="budget_prior"))
    rho_residual = scores.get("rho_residual")
    if torch.is_tensor(rho_residual) and rho_residual.dim() == 2 and rho_residual.shape[0] == targets.shape[0]:
        metrics.update(summarize_budget_tensor(rho_residual, targets, prefix="budget_residual"))
    evidence_sharpness = scores.get("evidence_sharpness")
    if (
        torch.is_tensor(evidence_sharpness)
        and evidence_sharpness.dim() == 2
        and evidence_sharpness.shape[0] == targets.shape[0]
    ):
        metrics.update(summarize_budget_tensor(evidence_sharpness, targets, prefix="evidence_sharpness"))
    evidence_advantage = scores.get("evidence_advantage")
    if (
        torch.is_tensor(evidence_advantage)
        and evidence_advantage.dim() == 2
        and evidence_advantage.shape[0] == targets.shape[0]
    ):
        metrics.update(summarize_budget_tensor(evidence_advantage, targets, prefix="evidence_advantage"))

    active_match_counts = scores.get("active_match_counts")
    if torch.is_tensor(active_match_counts) and active_match_counts.dim() == 2 and active_match_counts.shape[0] == targets.shape[0]:
        metrics.update(summarize_budget_tensor(active_match_counts, targets, prefix="active_matches"))

    fallback_row_fraction = scores.get("fallback_row_fraction")
    if (
        torch.is_tensor(fallback_row_fraction)
        and fallback_row_fraction.dim() == 2
        and fallback_row_fraction.shape[0] == targets.shape[0]
    ):
        metrics.update(summarize_budget_tensor(fallback_row_fraction, targets, prefix="fallback_fraction"))

    retained_fraction = scores.get("retained_fraction")
    if torch.is_tensor(retained_fraction) and retained_fraction.dim() == 2 and retained_fraction.shape[0] == targets.shape[0]:
        metrics.update(summarize_budget_tensor(retained_fraction, targets, prefix="retained_fraction"))

    shot_weights = scores.get("shot_aggregation_weights")
    if torch.is_tensor(shot_weights) and shot_weights.dim() == 3:
        shot_entropy = -(shot_weights.clamp_min(1e-12) * shot_weights.clamp_min(1e-12).log()).sum(dim=-1)
        metrics["mean_shot_weight_entropy"] = _scalar_metric(shot_entropy)

    extra_metric_keys = {
        "mean_shot_distance",
        "mean_gamma_entropy",
        "mean_consistency_penalty",
        "mean_redundancy_penalty",
        "logit_scale",
        "score_scale",
        "mass_regularization",
        "mean_query_mass_entropy",
        "mean_support_mass_entropy",
        "mean_budget",
        "mean_gate",
        "alpha",
        "global_margin",
        "local_margin",
        "global_branch_ce",
        "local_branch_ce",
        "budget_rank_loss",
        "rho_rank_loss",
        "budget_residual_reg_loss",
        "anchor_consistency_loss",
        "local_anchor_mix",
        "mean_reliability",
        "beta_eff",
        "beta0",
        "rho_bar",
        "compact_loss",
        "decorr_loss",
        "entropy_loss",
        "separation_loss",
        "factorization_aux_loss",
        "lambda_value",
        "alpha_value",
        "tau_value",
        "tau_local_value",
        "support_energy_entropy",
        "mean_attention_entropy",
        "alpha_entropy",
        "alpha_max_mean",
        "reliability_mix",
        "effective_support_size",
        "dispersion_inflation",
        "dispersion_mean",
        "dispersion_min",
        "prototype_shift_norm",
        "budget_low_loss",
        "budget_high_loss",
        "dustbin_cost",
        "matched_mass",
        "sinkhorn_error",
        "unmatched_mass",
        "weighted_unmatched_mass",
        "mean_crj_logit_delta",
        "mean_crj_shot_std",
        "mean_crj_effective_shots",
        "mean_crj_pool_entropy",
        "crj_pool_gamma",
        "crj_config_pool_gamma",
        "crj_effective_gamma",
        "crj_active",
        "crj_shot_num",
        "crj_variance_penalty",
        "crj_min_shots",
        "crj_trim_fraction",
        "crj_clip_logit",
        "ec_mrot_homotopy_lambda",
        "ec_mrot_response_strength",
        "ec_mrot_budget_kl",
        "ec_mrot_budget_kl_loss",
        "ec_mrot_budget_entropy",
        "ec_mrot_budget_entropy_loss",
        "ec_mrot_local_budget_entropy",
        "ec_mrot_response_grid_spacing_loss",
        "ec_mrot_aux_loss",
    }
    for key in extra_metric_keys:
        scalar = _scalar_metric(scores.get(key))
        if scalar is not None:
            metrics[key] = scalar
    return metrics


def accumulate_diagnostics(metric_sums, metrics, weight):
    if weight <= 0:
        return
    for key, value in metrics.items():
        if value is None:
            continue
        metric_sums[key] += float(value) * float(weight)


def finalize_diagnostics(metric_sums, total_weight):
    if total_weight <= 0:
        return {}
    return {key: value / float(total_weight) for key, value in metric_sums.items()}


def _unwrap_data_parallel(net):
    return net.module if hasattr(net, "module") else net


def _jecot_m2_transport_eps(net) -> float:
    core = _unwrap_data_parallel(net)
    return float(getattr(core, "eps", 1e-8))


def init_jecot_m2_validation_accumulator():
    return {
        "count": 0,
        "sum_true_mass": 0.0,
        "sum_best_wrong_mass": 0.0,
        "sum_true_cost": 0.0,
        "sum_best_wrong_cost": 0.0,
        "sum_true_avg_cost": 0.0,
        "sum_best_wrong_avg_cost": 0.0,
        "sum_true_logit": 0.0,
        "sum_best_wrong_logit": 0.0,
        "sum_margin": 0.0,
        "wrong_count": 0,
        "failure_wrong_count": 0,
        "true_avg_cost_samples": [],
    }


def accumulate_jecot_m2_validation_batch(
    scores: dict,
    logits: torch.Tensor,
    targets: torch.Tensor,
    preds: torch.Tensor,
    acc: dict,
    eps: float,
) -> None:
    transport_cost = scores.get("transport_cost")
    transport_mass = scores.get("transported_mass")
    if (
        not torch.is_tensor(transport_cost)
        or not torch.is_tensor(transport_mass)
        or transport_cost.dim() != 2
        or transport_mass.dim() != 2
        or logits.shape != transport_cost.shape
    ):
        return
    way = transport_mass.shape[1]
    if way < 2:
        return
    with torch.no_grad():
        y = targets.long()
        M = transport_mass
        C = transport_cost
        S = logits
        mass_y = M.gather(1, y.unsqueeze(1)).squeeze(1)
        cost_y = C.gather(1, y.unsqueeze(1)).squeeze(1)
        logit_y = S.gather(1, y.unsqueeze(1)).squeeze(1)
        mask = F.one_hot(y, num_classes=way).bool()
        mass_wrong = M.masked_fill(mask, float("-inf"))
        bw_mass = mass_wrong.max(dim=1).values
        logits_wrong = S.masked_fill(mask, float("-inf"))
        bw_logit = logits_wrong.max(dim=1).values
        runner = logits_wrong.argmax(dim=1)
        bw_cost = C.gather(1, runner.unsqueeze(1)).squeeze(1)
        runner_mass = M.gather(1, runner.unsqueeze(1)).squeeze(1)
        avg_y = cost_y / (mass_y + eps)
        bw_avg_cost = bw_cost / (runner_mass + eps)
        margin = logit_y - bw_logit
        failure_case = (mass_y > bw_mass) & (logit_y < bw_logit)
        wrong = preds.ne(y)
        n = int(y.numel())
        acc["count"] += n
        acc["sum_true_mass"] += float(mass_y.sum().item())
        acc["sum_best_wrong_mass"] += float(bw_mass.sum().item())
        acc["sum_true_cost"] += float(cost_y.sum().item())
        acc["sum_best_wrong_cost"] += float(bw_cost.sum().item())
        acc["sum_true_avg_cost"] += float(avg_y.sum().item())
        acc["sum_best_wrong_avg_cost"] += float(bw_avg_cost.sum().item())
        acc["sum_true_logit"] += float(logit_y.sum().item())
        acc["sum_best_wrong_logit"] += float(bw_logit.sum().item())
        acc["sum_margin"] += float(margin.sum().item())
        w = wrong
        acc["wrong_count"] += int(w.sum().item())
        acc["failure_wrong_count"] += int((failure_case & w).sum().item())
        acc["true_avg_cost_samples"].extend(avg_y.detach().float().cpu().numpy().tolist())


def finalize_jecot_m2_validation_accumulator(acc: dict) -> dict:
    n = int(acc["count"])
    out: dict = {}
    if n <= 0:
        return out
    out["m2_mean_true_mass"] = acc["sum_true_mass"] / n
    out["m2_mean_best_wrong_mass"] = acc["sum_best_wrong_mass"] / n
    out["m2_mean_true_cost"] = acc["sum_true_cost"] / n
    out["m2_mean_best_wrong_cost"] = acc["sum_best_wrong_cost"] / n
    out["m2_mean_true_avg_cost"] = acc["sum_true_avg_cost"] / n
    out["m2_mean_best_wrong_avg_cost"] = acc["sum_best_wrong_avg_cost"] / n
    out["m2_mean_true_logit"] = acc["sum_true_logit"] / n
    out["m2_mean_best_wrong_logit"] = acc["sum_best_wrong_logit"] / n
    out["m2_mean_logit_margin"] = acc["sum_margin"] / n
    wn = int(acc["wrong_count"])
    out["m2_failure_case_rate_among_wrong"] = (
        float(acc["failure_wrong_count"]) / float(wn) if wn > 0 else float("nan")
    )
    return out


def _jecot_m2_query_log_rows(
    scores: dict,
    logits: torch.Tensor,
    targets: torch.Tensor,
    preds: torch.Tensor,
    eps: float,
    global_offset: int,
) -> list[dict]:
    transport_cost = scores.get("transport_cost")
    transport_mass = scores.get("transported_mass")
    if (
        not torch.is_tensor(transport_cost)
        or not torch.is_tensor(transport_mass)
        or transport_cost.dim() != 2
        or transport_mass.dim() != 2
        or logits.shape != transport_cost.shape
    ):
        return []
    way = transport_mass.shape[1]
    if way < 2:
        return []
    rows: list[dict] = []
    with torch.no_grad():
        y = targets.long()
        M = transport_mass
        C = transport_cost
        S = logits
        mass_y = M.gather(1, y.unsqueeze(1)).squeeze(1)
        cost_y = C.gather(1, y.unsqueeze(1)).squeeze(1)
        logit_y = S.gather(1, y.unsqueeze(1)).squeeze(1)
        mask = F.one_hot(y, num_classes=way).bool()
        mass_wrong = M.masked_fill(mask, float("-inf"))
        bw_mass = mass_wrong.max(dim=1).values
        logits_wrong = S.masked_fill(mask, float("-inf"))
        bw_logit = logits_wrong.max(dim=1).values
        runner = logits_wrong.argmax(dim=1)
        bw_cost = C.gather(1, runner.unsqueeze(1)).squeeze(1)
        runner_mass = M.gather(1, runner.unsqueeze(1)).squeeze(1)
        avg_y = cost_y / (mass_y + eps)
        bw_avg_cost = bw_cost / (runner_mass + eps)
        margin = logit_y - bw_logit
        failure_case = ((mass_y > bw_mass) & (logit_y < bw_logit)).bool()
        for i in range(y.numel()):
            rows.append(
                {
                    "query_idx": global_offset + i,
                    "y": int(y[i].item()),
                    "pred": int(preds[i].item()),
                    "true_mass": float(mass_y[i].item()),
                    "best_wrong_mass": float(bw_mass[i].item()),
                    "true_cost": float(cost_y[i].item()),
                    "best_wrong_cost": float(bw_cost[i].item()),
                    "true_avg_cost": float(avg_y[i].item()),
                    "best_wrong_avg_cost": float(bw_avg_cost[i].item()),
                    "true_logit": float(logit_y[i].item()),
                    "best_wrong_logit": float(bw_logit[i].item()),
                    "margin": float(margin[i].item()),
                    "failure_case": int(failure_case[i].item()),
                }
            )
    return rows


def save_jecot_m2_validation_histogram(true_avg_costs: np.ndarray, threshold: float, save_path: str) -> None:
    if true_avg_costs.size == 0 or not np.isfinite(threshold):
        return
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(true_avg_costs[np.isfinite(true_avg_costs)], bins=40, color="#2E86AB", alpha=0.85, edgecolor="white")
    ax.axvline(threshold, color="#E94F37", linewidth=2.0, label=f"T={threshold:.4g}")
    ax.set_xlabel("True-class avg cost C_y / (M_y + eps)")
    ax.set_ylabel("Count")
    ax.set_title("J-ECOT-M2 validation: true avg cost vs learned threshold")
    ax.legend()
    ax.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_jecot_m2_threshold_curve(history_thresholds: list, save_path: str, eval_label: str = "Validation") -> None:
    if not history_thresholds:
        return
    fig, ax = plt.subplots(figsize=(6, 3.5))
    epochs = range(1, len(history_thresholds) + 1)
    ax.plot(epochs, history_thresholds, color="#2E86AB", marker="o", markersize=3, linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Transport cost threshold T")
    ax.set_title(f"J-ECOT-M2 learned T over epochs ({eval_label})")
    ax.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def prefix_diagnostics(metrics, prefix):
    return {f"{prefix}{key}": value for key, value in metrics.items()}


def format_diagnostic_summary(metrics):
    if not metrics:
        return ""
    ordered_keys = [
        "cls_loss",
        "aux_loss",
        "pred_acc",
        "true_prob",
        "max_prob",
        "correct_true_prob",
        "wrong_true_prob",
        "correct_max_prob",
        "wrong_max_prob",
        "logit_margin",
        "margin_pos_rate",
        "correct_logit_margin",
        "wrong_logit_margin",
        "raw_true_score",
        "raw_best_negative_score",
        "raw_score_gap",
        "emd_true_score",
        "emd_best_negative_score",
        "emd_score_gap",
        "recon_true_score",
        "recon_best_negative_score",
        "recon_score_gap",
        "struct_true_score",
        "struct_best_negative_score",
        "struct_score_gap",
        "global_true_score",
        "global_best_negative_score",
        "global_score_gap",
        "local_true_score",
        "local_best_negative_score",
        "local_score_gap",
        "crj_vanilla_true_score",
        "crj_vanilla_best_negative_score",
        "crj_vanilla_score_gap",
        "crj_lcb_true_score",
        "crj_lcb_best_negative_score",
        "crj_lcb_score_gap",
        "crj_delta_true_score",
        "crj_delta_best_negative_score",
        "crj_delta_score_gap",
        "raw_local_true_score",
        "raw_local_best_negative_score",
        "raw_local_score_gap",
        "competitive_local_true_score",
        "competitive_local_best_negative_score",
        "competitive_local_score_gap",
        "adaptive_local_true_score",
        "adaptive_local_best_negative_score",
        "adaptive_local_score_gap",
        "global_local_agreement",
        "budget_true",
        "budget_best_negative",
        "budget_gap",
        "budget_prior_true",
        "budget_prior_best_negative",
        "budget_prior_gap",
        "budget_residual_true",
        "budget_residual_best_negative",
        "budget_residual_gap",
        "evidence_advantage_true",
        "evidence_advantage_best_negative",
        "evidence_advantage_gap",
        "evidence_sharpness_true",
        "evidence_sharpness_best_negative",
        "evidence_sharpness_gap",
        "active_matches_true",
        "active_matches_best_negative",
        "active_matches_gap",
        "fallback_fraction_true",
        "fallback_fraction_best_negative",
        "fallback_fraction_gap",
        "retained_fraction_true",
        "retained_fraction_best_negative",
        "retained_fraction_gap",
        "crj_shot_std_true",
        "crj_shot_std_best_negative",
        "crj_shot_std_gap",
        "crj_consensus_true",
        "crj_consensus_best_negative",
        "crj_consensus_gap",
        "crj_effective_shots_true",
        "crj_effective_shots_best_negative",
        "crj_effective_shots_gap",
        "mean_crj_logit_delta",
        "mean_crj_shot_std",
        "mean_crj_effective_shots",
        "mean_crj_pool_entropy",
        "crj_pool_gamma",
        "crj_config_pool_gamma",
        "crj_effective_gamma",
        "crj_active",
        "crj_shot_num",
        "crj_variance_penalty",
        "crj_min_shots",
        "mean_budget",
        "mean_gate",
        "alpha",
        "mean_reliability",
        "beta_eff",
        "beta0",
        "rho_bar",
        "global_margin",
        "local_margin",
        "global_branch_ce",
        "local_branch_ce",
        "compact_loss",
        "decorr_loss",
        "entropy_loss",
        "separation_loss",
        "factorization_aux_loss",
        "lambda_value",
        "alpha_value",
        "tau_value",
        "tau_local_value",
        "support_energy_entropy",
        "mean_attention_entropy",
        "mean_shot_weight_entropy",
        "alpha_entropy",
        "alpha_max_mean",
        "reliability_mix",
        "effective_support_size",
        "dispersion_inflation",
        "dispersion_mean",
        "dispersion_min",
        "prototype_shift_norm",
        "true_distance",
        "best_negative_distance",
        "distance_gap",
        "distance_gap_pos_rate",
        "correct_distance_gap",
        "wrong_distance_gap",
        "query_class_true_distance",
        "query_class_best_negative_distance",
        "query_class_distance_gap",
        "query_class_distance_gap_pos_rate",
        "query_class_correct_distance_gap",
        "query_class_wrong_distance_gap",
        "epsilon_true",
        "epsilon_best_negative",
        "epsilon_gap",
        "support_dispersion_true",
        "support_dispersion_best_negative",
        "support_dispersion_gap",
        "transport_mass_true",
        "transport_mass_best_negative",
        "transport_mass_gap",
        "m2_mean_true_mass",
        "m2_mean_best_wrong_mass",
        "m2_mean_true_cost",
        "m2_mean_best_wrong_cost",
        "m2_mean_true_avg_cost",
        "m2_mean_best_wrong_avg_cost",
        "m2_mean_true_logit",
        "m2_mean_best_wrong_logit",
        "m2_mean_logit_margin",
        "m2_failure_case_rate_among_wrong",
        "mean_shot_distance",
        "mean_gamma_entropy",
        "mean_consistency_penalty",
        "mean_redundancy_penalty",
        "logit_scale",
    ]
    chunks = []
    emitted = set()
    for key in ordered_keys:
        if key in metrics:
            chunks.append(f"{key}={metrics[key]:.4f}")
            emitted.add(key)
    for key in sorted(metrics.keys()):
        if key in emitted:
            continue
        chunks.append(f"{key}={metrics[key]:.4f}")
    return ", ".join(chunks)


def build_scheduler(optimizer, args):
    scheduler_name = getattr(args, "scheduler", "cosine")
    if scheduler_name == "step":
        return lr_scheduler.StepLR(
            optimizer,
            step_size=args.step_size,
            gamma=args.gamma,
        )

    warmup_epochs = max(0, int(getattr(args, "warmup_epochs", 0)))
    warmup_start_factor = float(getattr(args, "warmup_start_factor", 0.1))
    warmup_start_factor = min(max(warmup_start_factor, 1e-4), 1.0)
    min_lr = float(getattr(args, "min_lr", 1e-6))
    total_epochs = max(1, int(args.num_epochs))

    if warmup_epochs > 0 and total_epochs > warmup_epochs:
        warmup = lr_scheduler.LinearLR(
            optimizer,
            start_factor=warmup_start_factor,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, total_epochs - warmup_epochs),
            eta_min=min_lr,
        )
        return lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs],
        )

    if warmup_epochs > 0:
        return lr_scheduler.LinearLR(
            optimizer,
            start_factor=warmup_start_factor,
            end_factor=1.0,
            total_iters=max(1, warmup_epochs),
        )

    return lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_epochs,
        eta_min=min_lr,
    )


def describe_scheduler(args):
    scheduler_name = getattr(args, "scheduler", "cosine")
    if scheduler_name == "step":
        return f"StepLR(step_size={args.step_size}, gamma={args.gamma})"
    warmup_epochs = max(0, int(getattr(args, "warmup_epochs", 0)))
    warmup_start_factor = min(max(float(getattr(args, "warmup_start_factor", 0.1)), 1e-4), 1.0)
    cosine_desc = f"CosineAnnealingLR(T_max={args.num_epochs}, eta_min={getattr(args, 'min_lr', 1e-6):.1e})"
    if warmup_epochs <= 0:
        return cosine_desc
    return (
        f"LinearLR(start_factor={warmup_start_factor:.1e}, total_iters={warmup_epochs}) -> "
        f"CosineAnnealingLR(T_max={max(1, args.num_epochs - warmup_epochs)}, eta_min={getattr(args, 'min_lr', 1e-6):.1e})"
    )


def maybe_load_scheduler_state(scheduler, scheduler_state, checkpoint_path):
    if scheduler is None or scheduler_state is None:
        return
    try:
        scheduler.load_state_dict(scheduler_state)
    except Exception as exc:
        print(f"Warning: failed to load scheduler state from {checkpoint_path}: {exc}")


def maybe_set_model_training_progress(net, progress):
    module = net.module if hasattr(net, "module") else net
    setter = getattr(module, "set_homotopy_progress", None)
    if setter is not None:
        setter(progress)


def train_loop(net, train_X, train_y, selection_X, selection_y, args):
    device = torch.device(args.device)
    relation_loss_fn = RelationLoss().to(device)
    contrastive_loss_fn = ContrastiveLoss().to(device)

    optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler(optimizer, args)

    train_augment = _bool_flag(args.train_augment, default=False)
    augment_cfg = build_train_augment_cfg(args)
    history = {
        "train_acc": [],
        "val_acc": [],
        "train_loss": [],
        "val_loss": [],
    }
    if args.model in M2_MODEL_NAMES:
        history["m2_transport_cost_threshold"] = []
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
            maybe_load_scheduler_state(scheduler, scheduler_state, args.resume)
            history = checkpoint.get("history", history)
            best_acc = float(checkpoint.get("best_acc", 0.0))
            start_epoch = int(checkpoint.get("epoch", 0)) + 1
            print(f"Resuming training from {args.resume} at epoch {start_epoch}/{args.num_epochs}")
        else:
            print(f"Loaded weights from {args.resume}; checkpoint has no optimizer state, restarting from epoch 1")
    elif args.weights:
        load_model_weights(net, args.weights, args.device)
        print(f"Initialized model weights from {args.weights}")

    if args.model in M2_MODEL_NAMES:
        history.setdefault("m2_transport_cost_threshold", [])

    selection_split, selection_episode_num, selection_query_num = get_selection_protocol(args)
    train_smoothing = effective_label_smoothing(args, train=True)
    smoothing_policy = "enabled for repo-new models only" if train_smoothing > 0 else "disabled for paper baselines"

    print(
        f"Training protocol: scheduler={describe_scheduler(args)}, "
        f"lr={args.lr:.2e}, classification_loss={args.classification_loss}, "
        f"label_smoothing=config={args.label_smoothing:.3f}, effective={train_smoothing:.3f} ({smoothing_policy}), "
        f"train_augment={train_augment}, selection_split={selection_split}, "
        f"train_seed=args.seed+train_episode_seed_offset+epoch, "
        f"selection_seed=args.seed+selection_episode_seed_offset"
        f"{'+epoch' if str(getattr(args, 'selection_episode_seed_mode', 'fixed')).lower() == 'per_epoch' else ''}, "
        f"episodes(train/select)={args.episode_num_train}/{selection_episode_num}, "
        f"query(train/select)={args.query_num_train}/{selection_query_num}, "
        f"num_workers={args.num_workers}, pin_memory={_pin_memory_enabled(args)}, "
        f"persistent_workers={_persistent_workers_enabled(args)}, "
        f"cudnn(deterministic={torch.backends.cudnn.deterministic}, benchmark={torch.backends.cudnn.benchmark})"
    )
    if args.classification_loss != "cross_entropy" and train_smoothing > 0:
        print("Warning: label_smoothing is ignored when classification_loss=repo_contrastive.")
    if train_augment:
        print(f"Augmentation config: {augment_cfg}")

    if start_epoch > args.num_epochs:
        print(f"Checkpoint already reached epoch {start_epoch - 1}; skipping training loop.")
        return best_acc, history

    for epoch in range(start_epoch, args.num_epochs + 1):
        denom = max(args.num_epochs - 1, 1)
        maybe_set_model_training_progress(net, (epoch - 1) / float(denom))
        train_seed = build_episode_seed(args, "train", epoch=epoch)
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
        train_diag_sums = defaultdict(float)
        non_blocking = _pin_memory_enabled(args) and device.type == "cuda"

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.num_epochs}")
        for query, q_labels, support, support_labels in pbar:
            optimizer.zero_grad(set_to_none=True)
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
                collect_diagnostics=True,
            )
            loss, cls_loss, aux_loss = compute_loss_breakdown(
                scores,
                targets,
                args,
                relation_loss_fn,
                contrastive_loss_fn,
                train=True,
            )
            logits = extract_logits(scores)

            with torch.no_grad():
                preds = logits.argmax(dim=1)
                train_correct += (preds == targets).sum().item()
                train_total += targets.size(0)
                batch_metrics = summarize_score_diagnostics(
                    scores=scores,
                    logits=logits,
                    targets=targets,
                    cls_loss=cls_loss,
                    aux_loss=aux_loss,
                )
                accumulate_diagnostics(train_diag_sums, batch_metrics, weight=targets.size(0))

            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{current_lr:.2e}")

        scheduler.step()

        train_acc = train_correct / train_total if train_total > 0 else 0.0
        train_diag = finalize_diagnostics(train_diag_sums, train_total)

        selection_seed = build_episode_seed(args, "selection", epoch=epoch)
        selection_ds = FewshotDataset(
            selection_X,
            selection_y,
            selection_episode_num,
            args.way_num,
            args.shot_num,
            selection_query_num,
            seed=selection_seed,
        )
        selection_loader = DataLoader(
            selection_ds,
            **_build_loader_kwargs(
                args,
                batch_size=1,
                shuffle=False,
                worker_seed=selection_seed,
            ),
        )

        m2_val_aux: dict = {}
        selection_acc, selection_loss, selection_diag = evaluate(
            net,
            selection_loader,
            args,
            phase=selection_split,
            m2_aux_out=m2_val_aux if args.model in M2_MODEL_NAMES else None,
        )
        avg_loss = total_loss / len(train_loader)

        history["train_acc"].append(train_acc)
        history["val_acc"].append(selection_acc)
        history["train_loss"].append(avg_loss)
        history["val_loss"].append(selection_loss if selection_loss else 0.0)
        if args.model in M2_MODEL_NAMES:
            t_cur = m2_val_aux.get("transport_cost_threshold")
            history["m2_transport_cost_threshold"].append(
                float(t_cur) if t_cur is not None and np.isfinite(t_cur) else float("nan")
            )

        train_selection_gap = train_acc - selection_acc
        print(
            f"Epoch {epoch}: Loss={avg_loss:.4f}, Train={train_acc:.4f}, "
            f"{selection_split.title()}={selection_acc:.4f} (gap={train_selection_gap:+.4f})"
        )
        train_diag_summary = format_diagnostic_summary(train_diag)
        if train_diag_summary:
            print(f"  TrainDiag: {train_diag_summary}")
        selection_diag_summary = format_diagnostic_summary(selection_diag)
        if selection_diag_summary:
            print(f"  {selection_split.title()}Diag: {selection_diag_summary}")

        wandb_payload = {
            "epoch": epoch,
            "loss/train": avg_loss,
            "accuracy/train": train_acc,
            "lr": current_lr,
        }
        wandb_payload[f"loss/{selection_split}"] = selection_loss
        wandb_payload[f"accuracy/{selection_split}"] = selection_acc
        wandb_payload[f"train_{selection_split}_gap"] = train_selection_gap
        wandb_payload.update(prefix_diagnostics(train_diag, "diag/train_"))
        wandb_payload.update(prefix_diagnostics(selection_diag, f"diag/{selection_split}_"))
        if args.model in M2_MODEL_NAMES:
            thr = m2_val_aux.get("transport_cost_threshold")
            if thr is not None and np.isfinite(thr):
                wandb_payload["jecot_m2/transport_cost_threshold"] = float(thr)
            hist = m2_val_aux.get("true_avg_cost_hist")
            if isinstance(hist, np.ndarray) and hist.size > 0:
                wandb_payload["jecot_m2/val_true_avg_cost_hist"] = wandb.Histogram(hist[np.isfinite(hist)])
            if _bool_flag(getattr(args, "jecot_m2_val_save_hist_fig", "true"), default=True):
                samples_str = f"{args.training_samples}samples" if args.training_samples else "allsamples"
                tag_suffix = f"_{args.experiment_tag}" if getattr(args, "experiment_tag", "") else ""
                hist_path = os.path.join(
                    args.path_results,
                    f"m2_val_avg_cost_hist_{args.dataset_name}_{args.model}_{samples_str}_{args.shot_num}shot"
                    f"{tag_suffix}_ep{epoch:04d}.png",
                )
                try:
                    if thr is not None and isinstance(hist, np.ndarray):
                        save_jecot_m2_validation_histogram(hist, float(thr), hist_path)
                        wandb_payload["jecot_m2/val_avg_cost_hist_fig"] = wandb.Image(hist_path)
                except (RuntimeError, ValueError, OSError) as exc:
                    print(f"Skipping M2 validation histogram: {exc}")
        wandb.log(wandb_payload)

        if selection_acc > best_acc:
            best_acc = selection_acc
            print(f"  -> New best {selection_split}: {selection_acc:.4f}")
            wandb.run.summary[f"best_{selection_split}_acc"] = best_acc

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
    tag_suffix = f"_{args.experiment_tag}" if getattr(args, "experiment_tag", "") else ""
    curves_path = os.path.join(
        args.path_results,
        f"training_{args.dataset_name}_{args.model}_{samples_str}_{args.shot_num}shot{tag_suffix}",
    )
    try:
        plot_training_curves(history, curves_path, eval_label=selection_split.title())
        if os.path.exists(f"{curves_path}_curves.png"):
            wandb.log({"training_curves": wandb.Image(f"{curves_path}_curves.png")})
    except RuntimeError as exc:
        print(f"Skipping training curves: {exc}")
    if args.model in M2_MODEL_NAMES and history.get("m2_transport_cost_threshold"):
        t_path = f"{curves_path}_m2_threshold.png"
        try:
            save_jecot_m2_threshold_curve(
                history["m2_transport_cost_threshold"],
                t_path,
                eval_label=selection_split.title(),
            )
            if os.path.exists(t_path):
                wandb.log({"jecot_m2/threshold_curve": wandb.Image(t_path)})
        except (RuntimeError, ValueError, OSError) as exc:
            print(f"Skipping M2 threshold curve: {exc}")

    print(f"Best {selection_split.title()} Accuracy: {best_acc:.4f}")
    return best_acc, history


def evaluate(net, loader, args, phase="val", m2_aux_out: dict | None = None):
    device = torch.device(args.device)
    relation_loss_fn = RelationLoss().to(device)
    contrastive_loss_fn = ContrastiveLoss().to(device)
    net.eval()
    correct, total = 0, 0
    total_loss = 0.0
    num_batches = 0
    diag_sums = defaultdict(float)
    non_blocking = _pin_memory_enabled(args) and device.type == "cuda"
    m2_active = args.model in M2_MODEL_NAMES
    m2_acc = init_jecot_m2_validation_accumulator() if m2_active else None
    m2_eps = _jecot_m2_transport_eps(net) if m2_active else 1e-8
    m2_csv_path = str(getattr(args, "jecot_m2_val_query_csv", "") or "").strip()
    m2_csv_rows: list[dict] | None = [] if (m2_active and m2_csv_path) else None
    m2_query_offset = 0

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
                phase=phase,
                query_targets=targets,
                support_targets=support_targets,
                collect_diagnostics=True,
            )
            logits = extract_logits(scores)
            preds = logits.argmax(dim=1)

            correct += (preds == targets).sum().item()
            total += targets.size(0)

            loss, cls_loss, aux_loss = compute_loss_breakdown(
                scores,
                targets,
                args,
                relation_loss_fn,
                contrastive_loss_fn,
                train=False,
            )
            total_loss += loss.item()
            num_batches += 1
            batch_metrics = summarize_score_diagnostics(
                scores=scores,
                logits=logits,
                targets=targets,
                cls_loss=cls_loss,
                aux_loss=aux_loss,
            )
            accumulate_diagnostics(diag_sums, batch_metrics, weight=targets.size(0))
            if m2_acc is not None:
                accumulate_jecot_m2_validation_batch(
                    scores,
                    logits,
                    targets,
                    preds,
                    m2_acc,
                    m2_eps,
                )
                if m2_csv_rows is not None:
                    m2_csv_rows.extend(
                        _jecot_m2_query_log_rows(
                            scores,
                            logits,
                            targets,
                            preds,
                            m2_eps,
                            m2_query_offset,
                        )
                    )
                    m2_query_offset += int(targets.numel())

    acc = correct / total if total > 0 else 0.0
    avg_loss = total_loss / num_batches if num_batches > 0 else None
    diag = finalize_diagnostics(diag_sums, total)
    if m2_acc is not None:
        diag.update(finalize_jecot_m2_validation_accumulator(m2_acc))
        core = _unwrap_data_parallel(net)
        thr = None
        if hasattr(core, "transport_cost_threshold"):
            thr = float(core.transport_cost_threshold.detach().float().mean().item())
        hist = np.asarray(m2_acc["true_avg_cost_samples"], dtype=np.float64)
        if m2_aux_out is not None:
            m2_aux_out["transport_cost_threshold"] = thr
            m2_aux_out["true_avg_cost_hist"] = hist
        if m2_csv_path and m2_csv_rows:
            os.makedirs(os.path.dirname(m2_csv_path) or ".", exist_ok=True)
            write_rows_csv(m2_csv_path, m2_csv_rows)
    return acc, avg_loss, diag


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


def write_rows_csv(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key in seen:
                continue
            seen.add(key)
            fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_final(net, loader, args, test_X=None, test_y=None, test_file_paths=None):
    import time

    device = torch.device(args.device)
    non_blocking = _pin_memory_enabled(args) and device.type == "cuda"
    num_episodes = len(loader)
    meta = get_model_metadata(args.model)
    current_test_name = str(getattr(args, "current_test_name", "") or "").strip()
    print(f"\n{'=' * 60}")
    print(f"Final Test: {meta['display_name']} | {args.dataset_name} | {args.shot_num}-shot")
    if current_test_name:
        print(f"Test Split: {current_test_name}")
    print(f"{num_episodes} episodes x {args.way_num} classes x {args.query_num_test} query")
    print("=" * 60)

    net.eval()
    all_preds, all_targets = [], []
    episode_accuracies = []
    episode_times = []
    query_seen = Counter()
    query_mis = Counter()
    query_pair = defaultdict(Counter)
    q1_enabled = _bool_flag(getattr(args, "export_q1_report", "false"), default=False)
    q1_limit = max(0, int(getattr(args, "q1_num_episodes", 0)))
    q1_queries_per_episode = max(1, int(getattr(args, "q1_queries_per_episode", 1)))
    q1_misclassified_only = _bool_flag(getattr(args, "q1_misclassified_only", "true"), default=True)
    q1_rows = []
    q1_figure_paths = []
    support_distribution_rows = []
    support_summary_sums = defaultdict(float)
    support_summary_count = 0.0
    test_diag_sums = defaultdict(float)
    test_diag_total = 0.0
    exported_q1 = 0
    collect_test_diagnostics = q1_enabled or args.model == "crj_fsl"

    with torch.no_grad():
        for episode_idx, batch in enumerate(tqdm(loader, desc="Testing")):
            if len(batch) == 6:
                query, q_labels, support, support_labels, q_indices, support_indices = batch
                q_indices_np = q_indices.view(-1).cpu().numpy()
            else:
                query, q_labels, support, support_labels = batch
                q_indices_np = None
                support_indices = None

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
                collect_diagnostics=collect_test_diagnostics,
            )
            logits = extract_logits(scores)
            preds = logits.argmax(dim=1)

            end_time = time.perf_counter()
            episode_times.append((end_time - start_time) * 1000)
            episode_accuracies.append((preds == targets).float().mean().item())

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            batch_metrics = summarize_score_diagnostics(scores=scores, logits=logits, targets=targets)
            accumulate_diagnostics(test_diag_sums, batch_metrics, weight=targets.size(0))
            test_diag_total += float(targets.size(0))

            if args.shot_num > 1:
                support_rows_episode, support_summary = compute_support_episode_distribution(
                    support[0].detach().cpu(),
                    class_names=args.class_names,
                    episode_index=episode_idx,
                    outputs=scores if isinstance(scores, dict) else None,
                )
                support_distribution_rows.extend(support_rows_episode)
                for key, value in support_summary.items():
                    support_summary_sums[key] += float(value)
                support_summary_count += 1.0

            if q1_enabled and exported_q1 < q1_limit:
                if q1_misclassified_only:
                    selected_query_indices = preds.ne(targets).nonzero(as_tuple=True)[0].tolist()
                else:
                    selected_query_indices = list(range(int(preds.shape[0])))
                selected_query_indices = selected_query_indices[:q1_queries_per_episode]

                if selected_query_indices:
                    q1_query_paths = None
                    if q_indices_np is not None and test_file_paths is not None:
                        q1_query_paths = []
                        for local_query_idx in range(len(q_indices_np)):
                            dataset_idx = int(q_indices_np[local_query_idx])
                            if 0 <= dataset_idx < len(test_file_paths):
                                q1_query_paths.append(test_file_paths[dataset_idx])
                            else:
                                q1_query_paths.append(None)
                    q1_base = os.path.join(
                        args.path_results,
                        f"q1_focus_{args.dataset_name}_{args.model}_{args.shot_num}shot_ep{episode_idx:04d}{get_test_protocol_suffix(args)}.png",
                    )
                    q1_rows.extend(
                        export_episode_q1_figure(
                            model=net,
                            outputs=scores if isinstance(scores, dict) else {"logits": logits},
                            query_images=query[0].detach().cpu(),
                            support_images=support[0].detach().cpu(),
                            logits=logits.detach().cpu(),
                            preds=preds.detach().cpu(),
                            targets=targets.detach().cpu(),
                            class_names=args.class_names,
                            save_path=q1_base,
                            episode_index=episode_idx,
                            query_indices=selected_query_indices,
                            query_file_paths=q1_query_paths,
                        )
                    )
                    q1_figure_paths.append(q1_base)
                    if args.shot_num > 1:
                        support_fig_path = os.path.join(
                            args.path_results,
                            f"q1_support_distribution_{args.dataset_name}_{args.model}_{args.shot_num}shot_ep{episode_idx:04d}{get_test_protocol_suffix(args)}.png",
                        )
                        export_support_distribution_figure(
                            support[0].detach().cpu(),
                            class_names=args.class_names,
                            episode_index=episode_idx,
                            outputs=scores if isinstance(scores, dict) else None,
                            save_path=support_fig_path,
                        )
                    exported_q1 += 1

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
    if support_summary_count > 0.0:
        print("\nSUPPORT-SHOT DIAGNOSTICS")
        print("=" * 60)
        print(f"  Mean pairwise distance : {support_summary_sums['support_pairwise_distance_mean'] / support_summary_count:.4f}")
        print(f"  Max outlier score      : {support_summary_sums['support_outlier_score_max'] / support_summary_count:.4f}")
        print(f"  Mean entropy std       : {support_summary_sums['support_entropy_std_mean'] / support_summary_count:.4f}")
    test_diag = finalize_diagnostics(test_diag_sums, test_diag_total)
    test_diag_summary = format_diagnostic_summary(test_diag)
    if test_diag_summary:
        print(f"  TestDiag: {test_diag_summary}")

    test_metrics = {
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
        **prefix_diagnostics(test_diag, "diag/test_"),
    }
    if current_test_name:
        metric_scope = sanitize_result_tag(current_test_name)
        test_metrics.update(
            {
                f"noise/{metric_scope}/accuracy_mean": acc_mean,
                f"noise/{metric_scope}/accuracy_ci95": acc_ci95,
                f"noise/{metric_scope}/f1": f1,
                f"noise/{metric_scope}/inference_time_mean_ms": time_mean,
            }
        )
    wandb.log(test_metrics)

    wandb.run.summary["test_accuracy_mean"] = acc_mean
    wandb.run.summary["test_accuracy_ci95"] = acc_ci95
    wandb.run.summary["model_selection_split"] = get_model_selection_split(args)
    wandb.run.summary["merge_val_into_train"] = _bool_flag(getattr(args, "merge_val_into_train", "false"), default=False)
    wandb.run.summary["test_protocol"] = getattr(args, "effective_test_protocol", getattr(args, "test_protocol", "clean"))
    if current_test_name:
        summary_scope = sanitize_result_tag(current_test_name)
        wandb.run.summary[f"noise/{summary_scope}/accuracy_mean"] = acc_mean
        wandb.run.summary[f"noise/{summary_scope}/accuracy_ci95"] = acc_ci95

    support_summary = {}
    if support_summary_count > 0.0:
        support_summary = {
            key: value / support_summary_count
            for key, value in support_summary_sums.items()
        }
        wandb.log({f"support/{key}": value for key, value in support_summary.items()})

    samples_str = f"_{args.training_samples}samples" if args.training_samples else "_allsamples"
    tag_suffix = f"_{args.experiment_tag}" if getattr(args, "experiment_tag", "") else ""
    protocol_suffix = get_test_protocol_suffix(args)
    cm_base = os.path.join(
        args.path_results,
        f"confusion_matrix_{args.dataset_name}_{args.model}_{samples_str.strip('_')}_{args.shot_num}shot{tag_suffix}{protocol_suffix}",
    )
    try:
        plot_confusion_matrix(all_targets, all_preds, args.way_num, cm_base, class_names=args.class_names)
        if os.path.exists(f"{cm_base}_2col.png"):
            wandb.log({"confusion_matrix": wandb.Image(f"{cm_base}_2col.png")})
    except RuntimeError as exc:
        print(f"Skipping confusion matrix plot: {exc}")

    if q1_rows:
        q1_csv_path = os.path.join(
            args.path_results,
            f"q1_focus_summary_{args.dataset_name}_{args.model}_{samples_str.strip('_')}_{args.shot_num}shot{tag_suffix}{protocol_suffix}.csv",
        )
        write_rows_csv(q1_csv_path, q1_rows)
        q1_signal_mean = float(np.mean([row["pred_signal_focus_ratio"] for row in q1_rows]))
        q1_background_mean = float(np.mean([row["pred_background_focus_ratio"] for row in q1_rows]))
        wandb.log(
            {
                "q1/pred_signal_focus_ratio_mean": q1_signal_mean,
                "q1/pred_background_focus_ratio_mean": q1_background_mean,
            }
        )
        for idx, figure_path in enumerate(q1_figure_paths[: min(3, len(q1_figure_paths))]):
            if os.path.exists(figure_path):
                wandb.log({f"q1/example_{idx}": wandb.Image(figure_path)})
        print(f"Saved Q1 focus summary: {q1_csv_path}")
        print(f"Saved {len(q1_figure_paths)} Q1 figure(s)")

    if support_distribution_rows:
        support_csv_path = os.path.join(
            args.path_results,
            f"support_episode_distribution_{args.dataset_name}_{args.model}_{samples_str.strip('_')}_{args.shot_num}shot{tag_suffix}{protocol_suffix}.csv",
        )
        write_rows_csv(support_csv_path, support_distribution_rows)
        print(f"Saved support-shot distribution summary: {support_csv_path}")

    if args.save_misclf_report and query_seen:
        pair_totals = Counter()
        true_totals = Counter()
        for true_i, pred_i in zip(all_targets.tolist(), all_preds.tolist()):
            true_totals[int(true_i)] += 1
            if int(true_i) != int(pred_i):
                pair_totals[(int(true_i), int(pred_i))] += 1

        pair_path = os.path.join(
            args.path_results,
            f"misclass_pairs_{args.dataset_name}_{args.model}_{samples_str.strip('_')}_{args.shot_num}shot{tag_suffix}{protocol_suffix}.txt",
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
            f"misclassified_files_{args.dataset_name}_{args.model}_{samples_str.strip('_')}_{args.shot_num}shot{tag_suffix}{protocol_suffix}.csv",
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
                f"tsne_{args.dataset_name}_{args.model}_{samples_str.strip('_')}_{args.shot_num}shot{tag_suffix}{protocol_suffix}",
            )
            try:
                plot_tsne(features, test_y_np, args.way_num, tsne_path, class_names=args.class_names)
                if os.path.exists(f"{tsne_path}_tsne.png"):
                    wandb.log({"tsne_plot": wandb.Image(f"{tsne_path}_tsne.png")})
            except RuntimeError as exc:
                print(f"Skipping t-SNE plot: {exc}")

    txt_path = os.path.join(
        args.path_results,
        f"results_{args.dataset_name}_{args.model}_{samples_str.strip('_')}_{args.shot_num}shot{tag_suffix}{protocol_suffix}.txt",
    )
    with open(txt_path, "w") as handle:
        handle.write(f"Model: {meta['display_name']} ({args.model})\n")
        handle.write(f"Dataset: {args.dataset_name}\n")
        handle.write(f"Shot: {args.shot_num}\n")
        handle.write(f"Training Samples: {args.training_samples if args.training_samples else 'All'}\n")
        handle.write(
            f"Test Protocol: {getattr(args, 'effective_test_protocol', getattr(args, 'test_protocol', 'clean'))}\n"
        )
        if current_test_name:
            handle.write(f"Test Split: {current_test_name}\n")
        if getattr(args, "noise_test_root", None):
            handle.write(f"Noise Test Root: {args.noise_test_root}\n")
        handle.write(f"Model Selection Split: {get_model_selection_split(args)}\n")
        handle.write(
            f"Merged Val Into Train: {_bool_flag(getattr(args, 'merge_val_into_train', 'false'), default=False)}\n"
        )
        handle.write("-" * 40 + "\n")
        handle.write(f"Accuracy : {acc_mean:.4f} +/- {acc_std:.4f}\n")
        handle.write(f"Worst-case : {acc_worst:.4f}\n")
        handle.write(f"Best-case : {acc_best:.4f}\n")
        handle.write(f"Precision : {prec:.4f}\n")
        handle.write(f"Recall : {rec:.4f}\n")
        handle.write(f"F1-Score : {f1:.4f}\n")
        handle.write(f"Inference Time: {time_mean:.2f} +/- {time_std:.2f} ms/episode\n")
        if support_summary:
            handle.write("-" * 40 + "\n")
            handle.write(f"Support Pairwise Distance Mean: {support_summary['support_pairwise_distance_mean']:.4f}\n")
            handle.write(f"Support Outlier Score Max: {support_summary['support_outlier_score_max']:.4f}\n")
            handle.write(f"Support Entropy Std Mean: {support_summary['support_entropy_std_mean']:.4f}\n")
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


def log_model_parameters(model, model_name, device="cuda", image_size=84):
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
    args = apply_optional_env_overrides(args)
    args = maybe_autoconfigure_hrot_from_checkpoint(args)
    args = apply_standalone_m2_defaults(args)
    if _bool_flag(getattr(args, "cudnn_deterministic", "true"), default=True):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    args.device = resolve_runtime_device(args)
    args.fewshot_backbone = resolve_fewshot_backbone(args)
    model_meta = get_model_metadata(args.model)

    if args.query_num is not None:
        args.query_num_train = args.query_num
        args.query_num_val = args.query_num
        args.query_num_test = args.query_num

    selection_split = get_model_selection_split(args)
    merge_val_into_train = _bool_flag(getattr(args, "merge_val_into_train", "false"), default=False)
    if merge_val_into_train and selection_split == "val":
        raise ValueError("merge_val_into_train=true is incompatible with model_selection_split=val.")

    print(f"\n{'=' * 72}")
    print(model_meta["display_name"])
    print("=" * 72)
    print(f"Mode        : {args.mode}")
    print(f"Config      : {args.model} | {args.shot_num}-shot | {args.num_epochs} epochs | device={args.device}")
    print(f"Dataset     : {args.dataset_path} ({args.dataset_name})")
    print(f"Architecture: {model_meta['architecture']}")
    print(f"Backbone    : {args.fewshot_backbone}")
    print(f"Selection   : split={selection_split}, merge_val_into_train={merge_val_into_train}")

    samples_str = f"{args.training_samples}samples" if args.training_samples else "all"
    tag_suffix = f"_{args.experiment_tag}" if getattr(args, "experiment_tag", "") else ""
    run_name = f"{args.model}_{args.dataset_name}_{samples_str}_{args.shot_num}shot{tag_suffix}"
    config = vars(args).copy()
    config["architecture"] = model_meta["architecture"]
    config["distance_metric"] = model_meta["metric"]
    config["selection_split"] = selection_split
    config["merge_val_into_train"] = merge_val_into_train

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
    args.effective_test_protocol = get_effective_test_protocol(args, dataset)
    print(f"Final Test  : protocol={args.effective_test_protocol}")

    def to_tensor(images, labels):
        return torch.from_numpy(images.astype(np.float32)), torch.from_numpy(labels).long()

    train_X, train_y = to_tensor(dataset.X_train, dataset.y_train)
    val_X, val_y = to_tensor(dataset.X_val, dataset.y_val)
    test_X, test_y = to_tensor(dataset.X_test, dataset.y_test)
    train_file_paths = [path for path, _ in getattr(dataset, "train_files", [])] if hasattr(dataset, "train_files") else None
    val_file_paths = [path for path, _ in getattr(dataset, "val_files", [])] if hasattr(dataset, "val_files") else None
    test_file_paths = [path for path, _ in getattr(dataset, "test_files", [])] if hasattr(dataset, "test_files") else None
    robust_pools = {}
    for split_name in getattr(dataset, "robust_test_splits", []):
        split_arrays = get_dataset_split_arrays(dataset, split_name)
        if split_arrays is None:
            continue
        split_X, split_y = to_tensor(split_arrays["images"], split_arrays["labels"])
        robust_pools[split_name] = {
            "images": split_X,
            "labels": split_y,
            "files": split_arrays["files"],
        }
    noise_pools = load_noise_test_pools(args, dataset) if args.effective_test_protocol == "noise" else {}

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
        for split_name, pool in robust_pools.items():
            pool_X, pool_y, pool_files = filter_classes(
                pool["images"],
                pool["labels"],
                selected,
                pool["files"],
            )
            robust_pools[split_name] = {
                "images": pool_X,
                "labels": pool_y,
                "files": pool_files,
            }
        for split_name, pool in noise_pools.items():
            pool_X, pool_y, pool_files = filter_classes(
                pool["images"],
                pool["labels"],
                selected,
                pool["files"],
            )
            noise_pools[split_name] = {
                **pool,
                "images": pool_X,
                "labels": pool_y,
                "files": pool_files,
            }
        print(f"Train: {len(train_X)}, Val: {len(val_X)}, Test: {len(test_X)}")
    else:
        args.class_names = all_class_names
        if args.way_num != len(args.class_names):
            print(
                f"way_num={args.way_num} does not match dataset classes={len(args.class_names)}. "
                f"Using way_num={len(args.class_names)}."
            )
            args.way_num = len(args.class_names)

    if _bool_flag(getattr(args, "export_dataset_profile", "false"), default=False):
        profile_stem = f"{args.dataset_name}_{args.model}_{samples_str}_{args.shot_num}shot{tag_suffix}".strip("_")
        profile_splits = {
            "train": (train_X, train_y, train_file_paths),
            "val": (val_X, val_y, val_file_paths),
            "test": (test_X, test_y, test_file_paths),
        }
        for split_name, pool in robust_pools.items():
            profile_splits[split_name] = (pool["images"], pool["labels"], pool["files"])
        for split_name, pool in noise_pools.items():
            profile_splits[split_name] = (pool["images"], pool["labels"], pool["files"])
        profile_payload = export_dataset_noise_profile(
            profile_splits,
            class_names=args.class_names,
            save_dir=args.path_results,
            run_stem=profile_stem,
        )
        dataset_metrics = {
            f"dataset/{metric_name}": metric_value
            for metric_name, metric_value in profile_payload["metrics"].items()
        }
        if dataset_metrics:
            wandb.log(dataset_metrics)
        if os.path.exists(profile_payload["figure_path"]):
            wandb.log({"dataset_noise_profile": wandb.Image(profile_payload["figure_path"])})
        print(f"Saved dataset profile figure: {profile_payload['figure_path']}")
        print(f"Saved dataset profile summary: {profile_payload['summary_csv_path']}")

    wandb.config.update(
        {
            "way_num": args.way_num,
            "query_num_train": args.query_num_train,
            "query_num_val": args.query_num_val,
            "query_num_test": args.query_num_test,
            "test_protocol": args.effective_test_protocol,
            "noise_test_root": getattr(args, "noise_test_root", None),
            "noise_test_splits": list(noise_pools.keys()),
            "selection_split": selection_split,
            "merge_val_into_train": merge_val_into_train,
        },
        allow_val_change=True,
    )

    if merge_val_into_train:
        train_X, train_y = concat_split_tensors(train_X, train_y, val_X, val_y)
        if train_file_paths is not None or val_file_paths is not None:
            train_file_paths = list(train_file_paths or []) + list(val_file_paths or [])
        print(f"Merged validation data into training pool: Train={len(train_X)}, Test={len(test_X)}")

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

    final_test_seed = build_episode_seed(args, "final_test")
    print(f"Final Test  : episode_seed={final_test_seed}")
    final_test_runs = []
    return_indices = args.save_misclf_report or _bool_flag(getattr(args, "export_q1_report", "false"), default=False)

    if args.effective_test_protocol == "noise":
        if not noise_pools:
            raise ValueError("--test_protocol noise did not find any SNR test split.")
        for split_name, pool in noise_pools.items():
            test_ds = FewshotDataset(
                pool["images"],
                pool["labels"],
                args.episode_num_test,
                args.way_num,
                args.shot_num,
                args.query_num_test,
                final_test_seed,
                return_indices=return_indices,
            )
            test_loader = DataLoader(
                test_ds,
                **_build_loader_kwargs(
                    args,
                    batch_size=1,
                    shuffle=False,
                    worker_seed=final_test_seed,
                ),
            )
            final_test_runs.append(
                {
                    "name": split_name,
                    "loader": test_loader,
                    "test_X": pool["images"],
                    "test_y": pool["labels"],
                    "test_file_paths": pool["files"],
                }
            )
        print(
            "Noise final test: "
            f"{len(final_test_runs)} SNR split(s): {', '.join(run['name'] for run in final_test_runs)}"
        )
    elif args.effective_test_protocol == "robust":
        query_pool_name = "test_1shot_query_snr10" if args.shot_num == 1 else "test_5shot_query_snr10"
        query_pool = robust_pools.get(query_pool_name)
        if query_pool is None:
            raise ValueError(f"Missing robust query pool: {query_pool_name}")
        test_ds = build_robust_test_dataset(
            args,
            clean_X=test_X,
            clean_y=test_y,
            clean_file_paths=test_file_paths,
            robust_pools=robust_pools,
            return_indices=return_indices,
        )
        test_loader = DataLoader(
            test_ds,
            **_build_loader_kwargs(
                args,
                batch_size=1,
                shuffle=False,
                worker_seed=final_test_seed,
            ),
        )
        final_test_runs.append(
            {
                "name": "",
                "loader": test_loader,
                "test_X": query_pool["images"],
                "test_y": query_pool["labels"],
                "test_file_paths": query_pool["files"],
            }
        )
        print(
            "Robust final test: "
            f"{args.shot_num}-shot, query_pool={query_pool_name}, "
            f"query_samples={len(query_pool['images'])}"
        )
    else:
        test_ds = FewshotDataset(
            test_X,
            test_y,
            args.episode_num_test,
            args.way_num,
            args.shot_num,
            args.query_num_test,
            final_test_seed,
            return_indices=return_indices,
        )
        test_loader = DataLoader(
            test_ds,
            **_build_loader_kwargs(
                args,
                batch_size=1,
                shuffle=False,
                worker_seed=final_test_seed,
            ),
        )
        final_test_runs.append(
            {
                "name": "",
                "loader": test_loader,
                "test_X": test_X,
                "test_y": test_y,
                "test_file_paths": test_file_paths,
            }
        )

    if selection_split == "test":
        selection_X, selection_y = test_X, test_y
    else:
        selection_X, selection_y = val_X, val_y

    net = get_model(args)
    log_model_parameters(net, args.model, device=args.device, image_size=args.image_size)

    if args.mode == "train":
        train_loop(net, train_X, train_y, selection_X, selection_y, args)
        if not _bool_flag(args.skip_final_test, default=False):
            path = get_best_model_path(args)
            print(f"Testing with best checkpoint: {path}")
            load_model_weights(net, path, args.device)
            for run in final_test_runs:
                args.current_test_name = run["name"]
                test_final(
                    net,
                    run["loader"],
                    args,
                    test_X=run["test_X"],
                    test_y=run["test_y"],
                    test_file_paths=run["test_file_paths"],
                )
    else:
        if args.weights:
            load_model_weights(net, args.weights, args.device)
            for run in final_test_runs:
                args.current_test_name = run["name"]
                test_final(
                    net,
                    run["loader"],
                    args,
                    test_X=run["test_X"],
                    test_y=run["test_y"],
                    test_file_paths=run["test_file_paths"],
                )
        else:
            print("Error: Please specify --weights for test mode")

    wandb.finish()


if __name__ == "__main__":
    main()
