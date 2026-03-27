"""Run pulse_fewshot benchmark experiments with a unified training protocol."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from net.model_factory import get_model_choices, get_model_metadata


SAMPLES_LIST = [60, 160, 240, None]
SHOTS_DEFAULT = [1, 5]
TRAIN_QUERY_NUM = 1
EVAL_QUERY_NUM = 1
FIXED_CUDNN_DETERMINISTIC = "true"
FIXED_CUDNN_BENCHMARK = "false"


def get_args():
    parser = argparse.ArgumentParser(description="Run all pulse_fewshot benchmark experiments")
    parser.add_argument("--project", type=str, default="pulse_fewshot", help="WandB project name")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/mnt/disk2/nhatnc/res/scalogram_fewshot/proposed_model/smnet/scalogram_27_1",
        help="Path to dataset",
    )
    parser.add_argument("--dataset_name", type=str, default="knee_aug_split", help="Dataset name for logging")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--shot_num", type=int, default=None, choices=[1, 5], help="Optional fixed shot number")
    parser.add_argument("--mode_id", type=int, default=None, choices=[1, 2, 3, 4], help="Optional sample mode")
    parser.add_argument(
        "--models",
        type=str,
        default="covamnet,protonet,cosine,relationnet,matchingnet,mann,dn4,feat,deepemd,adchot,adcmamba_sw,maml,can,frn,deepbdc,hierarchical_episodic_ssm_net,hierarchical_consensus_slot_mamba_net",
        help="Comma-separated model registry names to run",
    )
    parser.add_argument("--fewshot_backbone", type=str, default="default", choices=["default", "resnet12", "conv64f"])
    parser.add_argument("--gpu_id", type=int, default=0, help="CUDA device id to pass through")
    parser.add_argument("--num_workers", type=int, default=8, help="DataLoader workers to pass through to main.py")
    parser.add_argument("--pin_memory", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--persistent_workers", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--cudnn_deterministic", type=str, default="false", choices=["true", "false"])
    parser.add_argument("--cudnn_benchmark", type=str, default="true", choices=["true", "false"])
    return parser.parse_args()


EXPERIMENT_MODES = {
    1: 60,
    2: 160,
    3: 240,
    4: None,
}

EXTERNAL_SMNET_MODELS = {
    "uscmamba",
    "conv64f_token_sw_metric_net",
    "class_memory_scan_mamba_net",
    "episodic_selective_scan_mamba_net",
    "permutation_robust_class_memory_mamba_net",
    "transport_prior_replay_mamba_net",
    "transport_evidence_mamba_net",
}


def get_smnet_root():
    return (Path(__file__).resolve().parent.parent / "proposed_model" / "smnet").resolve()


def run_experiment(
    model,
    shot,
    samples,
    dataset_path,
    dataset_name,
    project,
    seed,
    gpu_id,
    fewshot_backbone,
    num_workers,
    pin_memory,
    persistent_workers,
    prefetch_factor,
):
    print(f"\n{'=' * 72}")
    print("Experiment")
    print('=' * 72)
    print(f"Model       : {model}")
    print(f"Shot        : {shot}")
    print(f"Samples     : {samples if samples else 'All'}")
    print(f"Backbone    : {fewshot_backbone}")
    print(
        f"Runtime     : workers={num_workers}, pin_memory={pin_memory}, "
        f"persistent_workers={persistent_workers}, "
        f"cudnn(det={FIXED_CUDNN_DETERMINISTIC}, bench={FIXED_CUDNN_BENCHMARK})"
    )
    print("=" * 72)

    use_external_smnet = model in EXTERNAL_SMNET_MODELS
    grad_clip = "0.0" if model == "hierarchical_consensus_slot_mamba_net" else "0.0"
    target_script = str(get_smnet_root() / "main.py") if use_external_smnet else "main.py"
    cmd = [
        sys.executable,
        target_script,
        "--model",
        model,
        "--shot_num",
        str(shot),
        "--way_num",
        "4",
        "--query_num_train",
        str(TRAIN_QUERY_NUM),
        "--query_num_val",
        str(EVAL_QUERY_NUM),
        "--query_num_test",
        str(EVAL_QUERY_NUM),
        "--image_size",
        "64",
        "--mode",
        "train",
        "--project",
        project,
        "--dataset_path",
        dataset_path,
        "--dataset_name",
        dataset_name,
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
        grad_clip,
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
        str(seed),
        "--gpu_id",
        str(gpu_id),
        "--num_workers",
        str(num_workers),
        "--pin_memory",
        pin_memory,
        "--persistent_workers",
        persistent_workers,
        "--prefetch_factor",
        str(prefetch_factor),
        "--cudnn_deterministic",
        FIXED_CUDNN_DETERMINISTIC,
        "--cudnn_benchmark",
        FIXED_CUDNN_BENCHMARK,
    ]

    if samples is not None:
        cmd.extend(["--training_samples", str(samples)])
    if fewshot_backbone != "default":
        cmd.extend(["--fewshot_backbone", fewshot_backbone])
    if not use_external_smnet:
        cmd.extend(["--deepemd_fast_val", "false"])
        if model == "deepemd" and shot > 1:
            cmd.extend(
                [
                    "--deepemd_train_sfc",
                    "true",
                    "--deepemd_test_sfc",
                    "true",
                    "--deepemd_test_exact",
                    "true",
                ]
            )

    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as exc:
        print(f"Error: {exc}")
        return False


def generate_comparison_charts(dataset_name, shots, models):
    import re

    try:
        from function.function import plot_model_comparison_bar
    except ImportError:
        print("Warning: Could not import plot function, skipping charts")
        return

    results_dir = "results/"
    model_display_names = {model_name: get_model_metadata(model_name)["display_name"] for model_name in models}

    for samples in SAMPLES_LIST:
        samples_str = f"{samples}samples" if samples is not None else "allsamples"
        model_results = {}

        for model in models:
            display_name = model_display_names[model]
            model_results[display_name] = {}

            for shot in shots:
                result_file = os.path.join(
                    results_dir,
                    f"results_{dataset_name}_{model}_{samples_str}_{shot}shot.txt",
                )
                if os.path.exists(result_file):
                    with open(result_file, "r") as handle:
                        content = handle.read()
                    match = re.search(r"Accuracy\s*:\s*([\d.]+)\s*(?:\+/-|±)", content)
                    if match:
                        model_results[display_name][f"{shot}shot"] = float(match.group(1))

        model_results = {
            name: scores
            for name, scores in model_results.items()
            if all(f"{shot}shot" in scores for shot in shots)
        }
        if not model_results:
            continue

        training_samples = samples if samples is not None else "All"
        save_path = os.path.join(results_dir, f"model_comparison_{dataset_name}_{samples_str}.png")
        plot_model_comparison_bar(model_results, training_samples, save_path)
        print(f"Model comparison chart saved to {save_path}")


def main():
    args = get_args()
    shots = [args.shot_num] if args.shot_num is not None else SHOTS_DEFAULT
    requested_models = [model.strip() for model in args.models.split(",") if model.strip()]
    valid_models = set(get_model_choices())
    invalid_models = [model for model in requested_models if model not in valid_models]
    if invalid_models:
        raise ValueError(f"Unsupported models: {invalid_models}. Valid choices: {sorted(valid_models)}")

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    if args.mode_id is not None:
        samples = EXPERIMENT_MODES[args.mode_id]
        experiments = [(model, samples, shot) for model in requested_models for shot in shots]
        print("=" * 72)
        print("pulse_fewshot - Single Experiment")
        print("=" * 72)
        print(f"Mode        : {args.mode_id} ({samples if samples else 'All'} samples)")
        print(f"Models      : {', '.join(requested_models)}")
        print(f"Shots       : {', '.join(map(str, shots))}")
        print(f"Backbone    : {args.fewshot_backbone}")
        print(f"Dataset     : {args.dataset_path} ({args.dataset_name})")
        print(f"GPU         : {args.gpu_id}")
        print(
            f"Runtime     : workers={args.num_workers}, pin_memory={args.pin_memory}, "
            f"persistent_workers={args.persistent_workers}, "
            f"cudnn(det={FIXED_CUDNN_DETERMINISTIC}, bench={FIXED_CUDNN_BENCHMARK})"
        )
        print(
            "Overrides   : "
            f"lr=5e-4, grad_clip={'1.0 for consensus, else 0.0'}, "
            "label_smoothing=0, augment=off, masks=off, DeepEMD 5-shot=SFC on"
        )
        print(f"Total       : {len(experiments)} experiment(s)")
        print("=" * 72)
    else:
        experiments = [
            (model, samples, shot)
            for model in requested_models
            for samples in SAMPLES_LIST
            for shot in shots
        ]
        print("=" * 72)
        print("pulse_fewshot - Full Experiment Suite")
        print("=" * 72)
        print("Mode Mapping:")
        for mode_id, sample_count in EXPERIMENT_MODES.items():
            print(f"  Mode {mode_id}: {sample_count if sample_count else 'All'} samples")
        print(f"Models      : {', '.join(requested_models)}")
        print(f"Shots       : {', '.join(f'{shot}-shot' for shot in shots)}")
        print(f"Backbone    : {args.fewshot_backbone}")
        print(f"Dataset     : {args.dataset_path} ({args.dataset_name})")
        print(f"GPU         : {args.gpu_id}")
        print(
            f"Runtime     : workers={args.num_workers}, pin_memory={args.pin_memory}, "
            f"persistent_workers={args.persistent_workers}, "
            f"cudnn(det={FIXED_CUDNN_DETERMINISTIC}, bench={FIXED_CUDNN_BENCHMARK})"
        )
        print(
            "Overrides   : "
            f"lr=5e-4, grad_clip={'1.0 for consensus, else 0.0'}, "
            "label_smoothing=0, augment=off, masks=off, DeepEMD 5-shot=SFC on"
        )
        print(f"Total       : {len(experiments)} experiments")
        print("=" * 72)

    success_count = 0
    failed_experiments = []

    for idx, (model, samples, shot) in enumerate(experiments, 1):
        print(f"\n[{idx}/{len(experiments)}]", end=" ")
        success = run_experiment(
            model=model,
            shot=shot,
            samples=samples,
            dataset_path=args.dataset_path,
            dataset_name=args.dataset_name,
            project=args.project,
            seed=args.seed,
            gpu_id=args.gpu_id,
            fewshot_backbone=args.fewshot_backbone,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            persistent_workers=args.persistent_workers,
            prefetch_factor=args.prefetch_factor,
        )
        if success:
            success_count += 1
        else:
            failed_experiments.append(f"{model}_{shot}shot_{samples if samples else 'all'}samples")

    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"Total: {len(experiments)}")
    print(f"Success: {success_count}")
    print(f"Failed: {len(failed_experiments)}")
    if failed_experiments:
        print("\nFailed experiments:")
        for experiment in failed_experiments:
            print(f"  - {experiment}")

    print("\n" + "=" * 60)
    print("Generating comparison charts...")
    print("=" * 60)
    generate_comparison_charts(args.dataset_name, shots, requested_models)
    print("\nAll experiments completed!")


if __name__ == "__main__":
    main()
