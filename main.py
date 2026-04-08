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
from torch.utils.data import DataLoader
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
    ContrastiveLoss,
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


PAPER_BASELINE_MODELS = {
    "cosine",
    "protonet",
    "covamnet",
    "matchingnet",
    "relationnet",
    "dn4",
    "feat",
    "deepemd",
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
    parser.add_argument("--query_num_train", type=int, default=5)
    parser.add_argument("--query_num_val", type=int, default=10)
    parser.add_argument("--query_num_test", type=int, default=10)
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
    parser.add_argument("--hrot_variant", type=str, default="E", choices=["A", "B", "C", "D", "E"])
    parser.add_argument("--hrot_eam_hidden_dim", type=int, default=256)
    parser.add_argument("--hrot_curvature_init", type=float, default=1.0)
    parser.add_argument("--hrot_projection_scale", type=float, default=0.1)
    parser.add_argument("--hrot_score_scale", type=float, default=16.0)
    parser.add_argument("--hrot_tau_q", type=float, default=0.5)
    parser.add_argument("--hrot_tau_c", type=float, default=0.5)
    parser.add_argument("--hrot_sinkhorn_epsilon", type=float, default=0.1)
    parser.add_argument("--hrot_sinkhorn_iterations", type=int, default=60)
    parser.add_argument("--hrot_sinkhorn_tolerance", type=float, default=1e-5)
    parser.add_argument("--hrot_fixed_mass", type=float, default=0.8)
    parser.add_argument("--hrot_min_mass", type=float, default=0.1)
    parser.add_argument("--hrot_mass_bonus_init", type=float, default=1.0)
    parser.add_argument("--hrot_lambda_rho", type=float, default=0.01)
    parser.add_argument("--hrot_rho_target", type=float, default=0.8)
    parser.add_argument("--hrot_lambda_curvature", type=float, default=0.0)
    parser.add_argument("--hrot_min_curvature", type=float, default=0.05)
    parser.add_argument("--hrot_normalize_euclidean_tokens", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--hrot_eval_use_float64", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--hrot_hyperbolic_backend", type=str, default="auto", choices=["auto", "geoopt", "native"])
    parser.add_argument("--hrot_ot_backend", type=str, default="native", choices=["auto", "native", "pot"])
    parser.add_argument("--hrot_eps", type=float, default=1e-6)

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
        default="fixed",
        choices=["fixed", "per_epoch"],
        help="Use one fixed selection seed for all epochs or advance it with epoch index.",
    )
    parser.add_argument(
        "--final_test_episode_seed_offset",
        type=int,
        default=0,
        help="Offset added to --seed for the final fixed test episode set.",
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
    parser.add_argument("--label_smoothing", type=float, default=0.1)
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
        print(
            "  deepemd: official-style EMD/SFC "
            f"(solver={getattr(args, 'deepemd_solver', 'opencv')}, "
            f"qpth_form={getattr(args, 'deepemd_qpth_form', 'L2')}, "
            f"sfc_lr={getattr(args, 'deepemd_sfc_lr', 0.1)}, "
            f"sfc_steps={getattr(args, 'deepemd_sfc_update_step', 100)}, "
            f"sfc_bs={getattr(args, 'deepemd_sfc_bs', 4)})"
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
    if args.model == "hrot_fsl":
        print(
            "  hrot_fsl: backbone spatial tokens -> Euclidean projector -> "
            "Poincare-ball geometry -> balanced/unbalanced relational transport "
            f"(variant={getattr(args, 'hrot_variant', 'E')}, "
            f"token_dim={getattr(args, 'hrot_token_dim', None) or getattr(args, 'token_dim', 128)}, "
            f"eam_hidden={getattr(args, 'hrot_eam_hidden_dim', 256)}, "
            f"curvature_init={getattr(args, 'hrot_curvature_init', 1.0)}, "
            f"proj_scale={getattr(args, 'hrot_projection_scale', 0.1)}, "
            f"score_scale={getattr(args, 'hrot_score_scale', 16.0)}, "
            f"tau_q={getattr(args, 'hrot_tau_q', 0.5)}, "
            f"tau_c={getattr(args, 'hrot_tau_c', 0.5)}, "
            f"sinkhorn_eps={getattr(args, 'hrot_sinkhorn_epsilon', 0.1)}, "
            f"sinkhorn_iters={getattr(args, 'hrot_sinkhorn_iterations', 60)}, "
            f"hyp_backend={getattr(args, 'hrot_hyperbolic_backend', 'auto')}, "
            f"ot_backend={getattr(args, 'hrot_ot_backend', 'native')}, "
            f"fixed_mass={getattr(args, 'hrot_fixed_mass', 0.8)}, "
            f"lambda_rho={getattr(args, 'hrot_lambda_rho', 0.01)})"
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
    if args.model == "hrot_fsl":
        return net(
            query,
            support,
            query_targets=query_targets,
            support_targets=support_targets,
            return_aux=collect_diagnostics,
        )
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
        max_probs = probs.max(dim=1).values
        return {
            "true_prob": _scalar_metric(true_probs),
            "max_prob": _scalar_metric(max_probs),
            "logit_margin": _scalar_metric(margin),
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


def summarize_score_diagnostics(scores, logits, targets, cls_loss=None, aux_loss=None):
    metrics = summarize_prediction_diagnostics(logits, targets)
    metrics["cls_loss"] = _scalar_metric(cls_loss)
    metrics["aux_loss"] = _scalar_metric(aux_loss)

    if not isinstance(scores, dict):
        return metrics

    total_distance = scores.get("total_distance")
    if torch.is_tensor(total_distance) and total_distance.dim() == 2 and total_distance.shape[0] == targets.shape[0]:
        metrics.update(summarize_distance_diagnostics(total_distance, targets))

    global_scores = scores.get("global_scores")
    if torch.is_tensor(global_scores) and global_scores.dim() == 2 and global_scores.shape[0] == targets.shape[0]:
        metrics.update(summarize_class_score_tensor(global_scores, targets, prefix="global"))

    local_scores = scores.get("local_scores")
    if torch.is_tensor(local_scores) and local_scores.dim() == 2 and local_scores.shape[0] == targets.shape[0]:
        metrics.update(summarize_class_score_tensor(local_scores, targets, prefix="local"))
        if torch.is_tensor(global_scores) and global_scores.shape == local_scores.shape:
            metrics["global_local_agreement"] = _scalar_metric(
                global_scores.argmax(dim=1).eq(local_scores.argmax(dim=1)).float().mean()
            )
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


def prefix_diagnostics(metrics, prefix):
    return {f"{prefix}{key}": value for key, value in metrics.items()}


def format_diagnostic_summary(metrics):
    if not metrics:
        return ""
    ordered_keys = [
        "cls_loss",
        "aux_loss",
        "true_prob",
        "max_prob",
        "logit_margin",
        "global_true_score",
        "global_best_negative_score",
        "global_score_gap",
        "local_true_score",
        "local_best_negative_score",
        "local_score_gap",
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
        "true_distance",
        "best_negative_distance",
        "distance_gap",
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

        selection_acc, selection_loss, selection_diag = evaluate(
            net,
            selection_loader,
            args,
            phase=selection_split,
        )
        avg_loss = total_loss / len(train_loader)

        history["train_acc"].append(train_acc)
        history["val_acc"].append(selection_acc)
        history["train_loss"].append(avg_loss)
        history["val_loss"].append(selection_loss if selection_loss else 0.0)

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

    print(f"Best {selection_split.title()} Accuracy: {best_acc:.4f}")
    return best_acc, history


def evaluate(net, loader, args, phase="val"):
    device = torch.device(args.device)
    relation_loss_fn = RelationLoss().to(device)
    contrastive_loss_fn = ContrastiveLoss().to(device)
    net.eval()
    correct, total = 0, 0
    total_loss = 0.0
    num_batches = 0
    diag_sums = defaultdict(float)
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

    acc = correct / total if total > 0 else 0.0
    avg_loss = total_loss / num_batches if num_batches > 0 else None
    return acc, avg_loss, finalize_diagnostics(diag_sums, total)


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
    wandb.run.summary["model_selection_split"] = get_model_selection_split(args)
    wandb.run.summary["merge_val_into_train"] = _bool_flag(getattr(args, "merge_val_into_train", "false"), default=False)

    samples_str = f"_{args.training_samples}samples" if args.training_samples else "_allsamples"
    tag_suffix = f"_{args.experiment_tag}" if getattr(args, "experiment_tag", "") else ""
    cm_base = os.path.join(
        args.path_results,
        f"confusion_matrix_{args.dataset_name}_{args.model}_{samples_str.strip('_')}_{args.shot_num}shot{tag_suffix}",
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
            f"misclass_pairs_{args.dataset_name}_{args.model}_{samples_str.strip('_')}_{args.shot_num}shot{tag_suffix}.txt",
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
                f"tsne_{args.dataset_name}_{args.model}_{samples_str.strip('_')}_{args.shot_num}shot{tag_suffix}",
            )
            try:
                plot_tsne(features, test_y_np, args.way_num, tsne_path, class_names=args.class_names)
                if os.path.exists(f"{tsne_path}_tsne.png"):
                    wandb.log({"tsne_plot": wandb.Image(f"{tsne_path}_tsne.png")})
            except RuntimeError as exc:
                print(f"Skipping t-SNE plot: {exc}")

    txt_path = os.path.join(
        args.path_results,
        f"results_{args.dataset_name}_{args.model}_{samples_str.strip('_')}_{args.shot_num}shot{tag_suffix}.txt",
    )
    with open(txt_path, "w") as handle:
        handle.write(f"Model: {meta['display_name']} ({args.model})\n")
        handle.write(f"Dataset: {args.dataset_name}\n")
        handle.write(f"Shot: {args.shot_num}\n")
        handle.write(f"Training Samples: {args.training_samples if args.training_samples else 'All'}\n")
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
    test_ds = FewshotDataset(
        test_X,
        test_y,
        args.episode_num_test,
        args.way_num,
        args.shot_num,
        args.query_num_test,
        final_test_seed,
        return_indices=args.save_misclf_report,
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
            test_final(net, test_loader, args, test_X=test_X, test_y=test_y, test_file_paths=test_file_paths)
    else:
        if args.weights:
            load_model_weights(net, args.weights, args.device)
            test_final(net, test_loader, args, test_X=test_X, test_y=test_y, test_file_paths=test_file_paths)
        else:
            print("Error: Please specify --weights for test mode")

    wandb.finish()


if __name__ == "__main__":
    main()
