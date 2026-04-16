"""Run pulse_fewshot experiments with validation-only MLFork-style final evaluation.

This leaves ``run_all_experiments.py`` untouched and reuses its orchestration.
Internal pulse_fewshot models run via ``main_val_protocol.py``.
External SMNet models run via the SMNet-side ``main_val_protocol.py``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import run_all_experiments as base_runner


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
    spif_global_only,
    spif_local_only,
    passthrough_args=None,
    variant_args=None,
    experiment_tag=None,
    experiment_label=None,
):
    applied_backbone = base_runner.resolve_backbone_for_model(model, fewshot_backbone)

    print(f"\n{'=' * 72}")
    print("Experiment")
    print("=" * 72)
    print(f"Model       : {model}")
    if experiment_label:
        print(f"Variant     : {experiment_label}")
    if experiment_tag:
        print(f"Tag         : {experiment_tag}")
    print(f"Shot        : {shot}")
    print(f"Samples     : {samples if samples else 'All'}")
    print(f"Backbone    : {applied_backbone}")
    if applied_backbone != fewshot_backbone and fewshot_backbone != "default":
        print(f"Backbone Req: {fewshot_backbone} (skipped for this model)")
    if model.startswith("spif"):
        print(f"SPIF Ablate : global_only={spif_global_only}, local_only={spif_local_only}")
    print("Final Eval  : split=val, protocol=mlfork")
    print(
        f"Runtime     : workers={num_workers}, pin_memory={pin_memory}, "
        f"persistent_workers={persistent_workers}, "
        f"cudnn(det={base_runner.FIXED_CUDNN_DETERMINISTIC}, bench={base_runner.FIXED_CUDNN_BENCHMARK})"
    )
    print("=" * 72)

    use_external_smnet = model in base_runner.EXTERNAL_SMNET_MODELS
    grad_clip = "0.0"
    if use_external_smnet:
        target_script = str(base_runner.get_smnet_root() / "main_val_protocol.py")
    else:
        target_script = str(Path(__file__).resolve().parent / "main_val_protocol.py")

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
        str(base_runner.TRAIN_QUERY_NUM),
        "--query_num_val",
        str(base_runner.EVAL_QUERY_NUM),
        "--query_num_test",
        str(base_runner.EVAL_QUERY_NUM),
        "--image_size",
        "84",
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
        base_runner.FIXED_CUDNN_DETERMINISTIC,
        "--cudnn_benchmark",
        base_runner.FIXED_CUDNN_BENCHMARK,
    ]

    if samples is not None:
        cmd.extend(["--training_samples", str(samples)])
    if applied_backbone != "default":
        cmd.extend(["--fewshot_backbone", applied_backbone])
    if not use_external_smnet:
        cmd.extend(["--deepemd_fast_val", "true"])
        if model.startswith("spif"):
            cmd.extend(
                [
                    "--spif_global_only",
                    spif_global_only,
                    "--spif_local_only",
                    spif_local_only,
                ]
            )
        if model == "deepemd" and shot > 1:
            cmd.extend(
                [
                    "--deepemd_solver",
                    base_runner.DEEPEMD_5SHOT_TRAIN_SOLVER,
                    "--deepemd_train_sfc",
                    base_runner.DEEPEMD_5SHOT_TRAIN_SFC,
                    "--deepemd_train_sfc_update_step",
                    str(base_runner.DEEPEMD_5SHOT_TRAIN_SFC_STEPS),
                    "--deepemd_train_sfc_bs",
                    str(base_runner.DEEPEMD_5SHOT_TRAIN_SFC_BS),
                    "--deepemd_test_sfc",
                    base_runner.DEEPEMD_5SHOT_TEST_SFC,
                    "--deepemd_test_exact",
                    base_runner.DEEPEMD_5SHOT_TEST_EXACT,
                ]
            )
    if passthrough_args:
        cmd.extend(passthrough_args)
    if variant_args:
        cmd.extend(variant_args)
    if experiment_tag:
        cmd.extend(["--experiment_tag", experiment_tag])

    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as exc:
        print(f"Error: {exc}")
        return False


base_runner.run_experiment = run_experiment


if __name__ == "__main__":
    base_runner.main()
