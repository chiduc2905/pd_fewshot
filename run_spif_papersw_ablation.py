"""Run dedicated ablations for the SPIF local paper-SW family.

This script intentionally targets only the new paper-style SW variants and keeps
the older local-SW experiment path untouched.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import run_all_experiments as base_runner


DEFAULT_MODELS = ["spifce_local_papersw"]
DEFAULT_VARIANTS = [
    "paper_default",
    "proj_budget_low",
    "proj_budget_high",
    "train_fixed_proj",
    "normalized_inputs",
    "eval_repeat4",
    "score_scale4",
    "score_scale16",
]


ABLATION_VARIANTS = {
    "paper_default": {
        "spif_papersw_train_num_projections": "128",
        "spif_papersw_eval_num_projections": "512",
        "spif_papersw_train_projection_mode": "resample",
        "spif_papersw_eval_projection_mode": "fixed",
        "spif_papersw_eval_num_repeats": "1",
        "spif_papersw_normalize_inputs": "false",
        "spif_papersw_score_scale": "8.0",
    },
    "proj_budget_low": {
        "spif_papersw_train_num_projections": "64",
        "spif_papersw_eval_num_projections": "256",
        "spif_papersw_train_projection_mode": "resample",
        "spif_papersw_eval_projection_mode": "fixed",
        "spif_papersw_eval_num_repeats": "1",
        "spif_papersw_normalize_inputs": "false",
        "spif_papersw_score_scale": "8.0",
    },
    "proj_budget_high": {
        "spif_papersw_train_num_projections": "256",
        "spif_papersw_eval_num_projections": "1024",
        "spif_papersw_train_projection_mode": "resample",
        "spif_papersw_eval_projection_mode": "fixed",
        "spif_papersw_eval_num_repeats": "1",
        "spif_papersw_normalize_inputs": "false",
        "spif_papersw_score_scale": "8.0",
    },
    "train_fixed_proj": {
        "spif_papersw_train_num_projections": "128",
        "spif_papersw_eval_num_projections": "512",
        "spif_papersw_train_projection_mode": "fixed",
        "spif_papersw_eval_projection_mode": "fixed",
        "spif_papersw_eval_num_repeats": "1",
        "spif_papersw_normalize_inputs": "false",
        "spif_papersw_score_scale": "8.0",
    },
    "normalized_inputs": {
        "spif_papersw_train_num_projections": "128",
        "spif_papersw_eval_num_projections": "512",
        "spif_papersw_train_projection_mode": "resample",
        "spif_papersw_eval_projection_mode": "fixed",
        "spif_papersw_eval_num_repeats": "1",
        "spif_papersw_normalize_inputs": "true",
        "spif_papersw_score_scale": "8.0",
    },
    "eval_repeat4": {
        "spif_papersw_train_num_projections": "128",
        "spif_papersw_eval_num_projections": "512",
        "spif_papersw_train_projection_mode": "resample",
        "spif_papersw_eval_projection_mode": "fixed",
        "spif_papersw_eval_num_repeats": "4",
        "spif_papersw_normalize_inputs": "false",
        "spif_papersw_score_scale": "8.0",
    },
    "score_scale4": {
        "spif_papersw_train_num_projections": "128",
        "spif_papersw_eval_num_projections": "512",
        "spif_papersw_train_projection_mode": "resample",
        "spif_papersw_eval_projection_mode": "fixed",
        "spif_papersw_eval_num_repeats": "1",
        "spif_papersw_normalize_inputs": "false",
        "spif_papersw_score_scale": "4.0",
    },
    "score_scale16": {
        "spif_papersw_train_num_projections": "128",
        "spif_papersw_eval_num_projections": "512",
        "spif_papersw_train_projection_mode": "resample",
        "spif_papersw_eval_projection_mode": "fixed",
        "spif_papersw_eval_num_repeats": "1",
        "spif_papersw_normalize_inputs": "false",
        "spif_papersw_score_scale": "16.0",
    },
}


def get_args():
    parser = argparse.ArgumentParser(description="Run SPIF local paper-SW ablations")
    parser.add_argument("--project", type=str, default="pulse_fewshot_papersw_ablation")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/mnt/disk2/nhatnc/res/scalogram_fewshot/proposed_model/smnet/scalogram_27_1",
    )
    parser.add_argument("--dataset_name", type=str, default="knee_aug_split")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shot_num", type=int, default=None, choices=[1, 5])
    parser.add_argument("--mode_id", type=int, default=None, choices=[1, 2, 3, 4])
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(DEFAULT_MODELS),
        help="Comma-separated model names from the paper-SW family",
    )
    parser.add_argument(
        "--variants",
        type=str,
        default=",".join(DEFAULT_VARIANTS),
        help="Comma-separated ablation variant names",
    )
    parser.add_argument("--fewshot_backbone", type=str, default="default", choices=["default", "resnet12", "conv64f"])
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--pin_memory", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--persistent_workers", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--use_val_protocol", type=str, default="false", choices=["true", "false"])
    return parser.parse_args()


def _bool_flag(value: str) -> bool:
    return str(value).lower() == "true"


def _get_target_script(use_val_protocol: bool) -> str:
    script_name = "main_val_protocol.py" if use_val_protocol else "main.py"
    return str(Path(__file__).resolve().parent / script_name)


def _samples_from_mode_id(mode_id: int | None) -> list[int | None]:
    if mode_id is None:
        return [base_runner.EXPERIMENT_MODES[idx] for idx in sorted(base_runner.EXPERIMENT_MODES)]
    return [base_runner.EXPERIMENT_MODES[mode_id]]


def _build_base_command(args, model: str, shot: int, samples: int | None, variant_name: str) -> list[str]:
    cmd = [
        sys.executable,
        _get_target_script(_bool_flag(args.use_val_protocol)),
        "--model",
        model,
        "--shot_num",
        str(shot),
        "--way_num",
        "4",
        "--query_num_train",
        str(base_runner.TRAIN_QUERY_NUM),
        "--query_num_val",
        str(base_runner.EVAL_QUERY_NUM),
        "--query_num_test",
        str(base_runner.EVAL_QUERY_NUM),
        "--image_size",
        "64",
        "--mode",
        "train",
        "--project",
        args.project,
        "--dataset_path",
        args.dataset_path,
        "--dataset_name",
        f"{args.dataset_name}_{variant_name}",
        "--path_weights",
        "checkpoints/",
        "--path_results",
        "results/",
        "--num_epochs",
        "100",
        "--lr",
        "5e-4",
        "--step_size",
        "10",
        "--gamma",
        "0.5",
        "--weight_decay",
        "5e-4",
        "--grad_clip",
        "0.0",
        "--label_smoothing",
        "0.0",
        "--train_augment",
        "false",
        "--time_shift_prob",
        "0.0",
        "--amp_scale_prob",
        "0.0",
        "--time_mask_prob",
        "0.0",
        "--freq_mask_prob",
        "0.0",
        "--episode_num_train",
        "200",
        "--episode_num_val",
        "300",
        "--episode_num_test",
        "300",
        "--seed",
        str(args.seed),
        "--gpu_id",
        str(args.gpu_id),
        "--num_workers",
        str(args.num_workers),
        "--pin_memory",
        args.pin_memory,
        "--persistent_workers",
        args.persistent_workers,
        "--prefetch_factor",
        str(args.prefetch_factor),
        "--cudnn_deterministic",
        base_runner.FIXED_CUDNN_DETERMINISTIC,
        "--cudnn_benchmark",
        base_runner.FIXED_CUDNN_BENCHMARK,
        "--deepemd_fast_val",
        "false",
    ]
    if samples is not None:
        cmd.extend(["--training_samples", str(samples)])
    if args.fewshot_backbone != "default":
        cmd.extend(["--fewshot_backbone", args.fewshot_backbone])
    return cmd


def _apply_variant_overrides(cmd: list[str], overrides: dict[str, str]) -> list[str]:
    for key, value in overrides.items():
        cmd.extend([f"--{key}", str(value)])
    return cmd


def run_variant(args, model: str, shot: int, samples: int | None, variant_name: str) -> bool:
    overrides = ABLATION_VARIANTS[variant_name]
    cmd = _build_base_command(args, model=model, shot=shot, samples=samples, variant_name=variant_name)
    _apply_variant_overrides(cmd, overrides)

    print(f"\n{'=' * 88}")
    print("SPIF Paper-SW Ablation")
    print("=" * 88)
    print(f"Model       : {model}")
    print(f"Variant     : {variant_name}")
    print(f"Shot        : {shot}")
    print(f"Samples     : {samples if samples is not None else 'All'}")
    print(f"Backbone    : {args.fewshot_backbone}")
    print(f"ValProtocol : {_bool_flag(args.use_val_protocol)}")
    print(f"Overrides   : {overrides}")
    print("=" * 88)

    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as exc:
        print(f"Error: {exc}")
        return False


def main():
    args = get_args()
    requested_models = [model.strip() for model in args.models.split(",") if model.strip()]
    requested_variants = [variant.strip() for variant in args.variants.split(",") if variant.strip()]
    valid_models = set(base_runner.get_model_choices())
    invalid_models = [model for model in requested_models if model not in valid_models]
    if invalid_models:
        raise ValueError(f"Unsupported models: {invalid_models}")
    invalid_variants = [variant for variant in requested_variants if variant not in ABLATION_VARIANTS]
    if invalid_variants:
        raise ValueError(f"Unsupported variants: {invalid_variants}. Valid: {sorted(ABLATION_VARIANTS)}")

    shots = [args.shot_num] if args.shot_num is not None else base_runner.SHOTS_DEFAULT
    samples_list = _samples_from_mode_id(args.mode_id)

    total = len(requested_models) * len(requested_variants) * len(shots) * len(samples_list)
    done = 0
    success = 0
    for model in requested_models:
        for variant_name in requested_variants:
            for shot in shots:
                for samples in samples_list:
                    done += 1
                    ok = run_variant(args, model=model, shot=shot, samples=samples, variant_name=variant_name)
                    success += int(ok)
                    print(f"Progress    : {done}/{total} | success={success}")

    print(f"\nFinished SPIF paper-SW ablations: {success}/{total} runs succeeded.")


if __name__ == "__main__":
    main()
