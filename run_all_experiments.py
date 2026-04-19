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
TRAIN_EPISODES_PER_EPOCH = 130
TEST_EPISODES_PER_EPOCH = 150
FIXED_CUDNN_DETERMINISTIC = "true"
FIXED_CUDNN_BENCHMARK = "false"
DEEPEMD_5SHOT_TRAIN_SOLVER = "sinkhorn"
DEEPEMD_5SHOT_TRAIN_SFC = "false"
DEEPEMD_5SHOT_TRAIN_SFC_STEPS = 15
DEEPEMD_5SHOT_TRAIN_SFC_BS = 20
DEEPEMD_5SHOT_TEST_EXACT = "false"
DEEPEMD_5SHOT_TEST_SFC = "false"


def get_args():
    parser = argparse.ArgumentParser(description="Run all pulse_fewshot benchmark experiments")
    parser.add_argument("--project", type=str, default="pulse_fewshot", help="WandB project name")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/workspace/pd_fewshot/scalogram_27_1_hrot_robust",
        help="Path to dataset",
    )
    parser.add_argument("--dataset_name", type=str, default="knee_aug_split", help="Dataset name for logging")
    parser.add_argument(
        "--test_protocol",
        type=str,
        default="auto",
        choices=["auto", "clean", "robust"],
        help="Final-test protocol. auto uses robust test pools when the dataset provides them.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--shot_num", type=int, default=None, choices=[1, 5], help="Optional fixed shot number")
    parser.add_argument("--mode_id", type=int, default=None, choices=[1, 2, 3, 4], help="Optional sample mode")
    parser.add_argument(
        "--models",
        type=str,
        default="covamnet,protonet,cosine,relationnet,matchingnet,dn4,feat,deepemd,maml,can,frn,deepbdc,hierarchical_episodic_ssm_net,hierarchical_consensus_slot_mamba_net",
        help="Comma-separated model registry names to run",
    )
    parser.add_argument(
        "--fewshot_backbone",
        type=str,
        default="default",
        choices=["default", "resnet12", "conv64f", "fsl_mamba", "slim_mamba"],
    )
    parser.add_argument("--gpu_id", type=int, default=0, help="CUDA device id to pass through")
    parser.add_argument("--num_workers", type=int, default=8, help="DataLoader workers to pass through to main.py")
    parser.add_argument("--pin_memory", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--persistent_workers", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--cudnn_deterministic", type=str, default="false", choices=["true", "false"])
    parser.add_argument("--cudnn_benchmark", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--spif_global_only", type=str, default="false", choices=["true", "false"])
    parser.add_argument("--spif_local_only", type=str, default="false", choices=["true", "false"])
    parser.add_argument(
        "--spifce_ablation_suite",
        type=str,
        default="none",
        choices=["none", "contrib", "contrib_detailed"],
        help="Expand SPIFCE contribution ablations with tagged runs",
    )
    args, passthrough_args = parser.parse_known_args()
    args.passthrough_args = passthrough_args
    return args


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

FSL_MAMBA_COMPATIBLE_MODELS = {
    "hierarchical_episodic_ssm_net",
    "hierarchical_consensus_slot_mamba_net",
    "hierarchical_support_sw_net",
    "sc_lfi",
    "sc_lfi_v2",
    "sc_lfi_v3",
    "sc_lfi_v4",
    "spifv3_cstn",
    "spifv3_cstn_papersw",
    "warn",
    "pars_net",
    "hrot_fsl",
    "jsc_wdro",
}


def resolve_backbone_for_model(model: str, requested_backbone: str) -> str:
    requested_backbone = str(requested_backbone).lower()
    if requested_backbone == "slim_mamba":
        requested_backbone = "fsl_mamba"
    if requested_backbone != "fsl_mamba":
        return requested_backbone
    if model in EXTERNAL_SMNET_MODELS:
        return "default"
    if model.startswith("spif") or model in FSL_MAMBA_COMPATIBLE_MODELS:
        return "fsl_mamba"
    return "default"


def get_smnet_root():
    return (Path(__file__).resolve().parent.parent / "proposed_model" / "smnet").resolve()


def get_pulse_fewshot_root():
    return Path(__file__).resolve().parent


def dataset_has_hrot_robust_layout(dataset_path):
    root = Path(dataset_path)
    required = [
        "train",
        "val",
        "test_clean",
        "test_1shot_support_snr10",
        "test_1shot_query_snr10",
        "test_5shot_support_outlier_snr0",
        "test_5shot_query_snr10",
    ]
    return all((root / name).is_dir() for name in required)


def resolve_test_protocol(requested_protocol, dataset_path):
    requested_protocol = str(requested_protocol).lower()
    if requested_protocol == "auto":
        return "robust" if dataset_has_hrot_robust_layout(dataset_path) else "clean"
    return requested_protocol


def result_protocol_suffix(test_protocol):
    return "_robust" if str(test_protocol).lower() == "robust" else ""


def build_spifce_ablation_variants(suite_name):
    base_variants = [
        {
            "tag": "base_shared_proto",
            "label": "Base shared prototype",
            "extra_args": [
                "--spif_factorization_on",
                "false",
                "--spif_gate_on",
                "false",
                "--spif_global_only",
                "true",
                "--spif_local_only",
                "false",
            ],
        },
        {
            "tag": "factor_only",
            "label": "Factorization only",
            "extra_args": [
                "--spif_factorization_on",
                "true",
                "--spif_gate_on",
                "false",
                "--spif_global_only",
                "true",
                "--spif_local_only",
                "false",
            ],
        },
        {
            "tag": "gate_only",
            "label": "Gate only",
            "extra_args": [
                "--spif_factorization_on",
                "false",
                "--spif_gate_on",
                "true",
                "--spif_global_only",
                "true",
                "--spif_local_only",
                "false",
            ],
        },
        {
            "tag": "local_module_only",
            "label": "Local module only",
            "extra_args": [
                "--spif_factorization_on",
                "false",
                "--spif_gate_on",
                "false",
                "--spif_global_only",
                "false",
                "--spif_local_only",
                "false",
            ],
        },
        {
            "tag": "factor_gate",
            "label": "Factorization + gate",
            "extra_args": [
                "--spif_factorization_on",
                "true",
                "--spif_gate_on",
                "true",
                "--spif_global_only",
                "true",
                "--spif_local_only",
                "false",
            ],
        },
        {
            "tag": "factor_local",
            "label": "Factorization + local",
            "extra_args": [
                "--spif_factorization_on",
                "true",
                "--spif_gate_on",
                "false",
                "--spif_global_only",
                "false",
                "--spif_local_only",
                "false",
            ],
        },
        {
            "tag": "gate_local",
            "label": "Gate + local",
            "extra_args": [
                "--spif_factorization_on",
                "false",
                "--spif_gate_on",
                "true",
                "--spif_global_only",
                "false",
                "--spif_local_only",
                "false",
            ],
        },
        {
            "tag": "full",
            "label": "Full SPIFCE",
            "extra_args": [
                "--spif_factorization_on",
                "true",
                "--spif_gate_on",
                "true",
                "--spif_global_only",
                "false",
                "--spif_local_only",
                "false",
            ],
        },
    ]
    if suite_name != "contrib_detailed":
        return base_variants

    return base_variants + [
        {
            "tag": "shared_local_branch_only",
            "label": "Shared local branch only",
            "extra_args": [
                "--spif_factorization_on",
                "false",
                "--spif_gate_on",
                "false",
                "--spif_global_only",
                "false",
                "--spif_local_only",
                "true",
            ],
        },
        {
            "tag": "factor_local_branch_only",
            "label": "Factorized local branch only",
            "extra_args": [
                "--spif_factorization_on",
                "true",
                "--spif_gate_on",
                "false",
                "--spif_global_only",
                "false",
                "--spif_local_only",
                "true",
            ],
        },
        {
            "tag": "gate_local_branch_only",
            "label": "Gated local branch only",
            "extra_args": [
                "--spif_factorization_on",
                "false",
                "--spif_gate_on",
                "true",
                "--spif_global_only",
                "false",
                "--spif_local_only",
                "true",
            ],
        },
        {
            "tag": "full_local_branch_only",
            "label": "Full local branch only",
            "extra_args": [
                "--spif_factorization_on",
                "true",
                "--spif_gate_on",
                "true",
                "--spif_global_only",
                "false",
                "--spif_local_only",
                "true",
            ],
        },
    ]


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
    test_protocol,
    passthrough_args=None,
    variant_args=None,
    experiment_tag=None,
    experiment_label=None,
):
    applied_backbone = resolve_backbone_for_model(model, fewshot_backbone)

    print(f"\n{'=' * 72}")
    print("Experiment")
    print('=' * 72)
    print(f"Model       : {model}")
    if experiment_label:
        print(f"Variant     : {experiment_label}")
    if experiment_tag:
        print(f"Tag         : {experiment_tag}")
    print(f"Shot        : {shot}")
    print(f"Samples     : {samples if samples else 'All'}")
    print(f"Backbone    : {applied_backbone}")
    print(f"Test Proto  : {test_protocol}")
    print("Protocol    : selection=val, merge_val_into_train=false, episodes(train/val/test)=130/150/150")
    if applied_backbone != fewshot_backbone and fewshot_backbone != "default":
        print(f"Backbone Req: {fewshot_backbone} (skipped for this model)")
    if model.startswith("spif"):
        print(f"SPIF Ablate : global_only={spif_global_only}, local_only={spif_local_only}")
    print(
        f"Runtime     : workers={num_workers}, pin_memory={pin_memory}, "
        f"persistent_workers={persistent_workers}, "
        f"cudnn(det={FIXED_CUDNN_DETERMINISTIC}, bench={FIXED_CUDNN_BENCHMARK})"
    )
    print("=" * 72)

    use_external_smnet = model in EXTERNAL_SMNET_MODELS
    grad_clip = "0.0"
    target_script = (
        str(get_smnet_root() / "main.py")
        if use_external_smnet
        else str(get_pulse_fewshot_root() / "main.py")
    )
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
        "--scheduler",
        "cosine",
        "--step_size",
        "10",
        "--gamma",
        "0.5",
        "--warmup_epochs",
        "5",
        "--min_lr",
        "1e-6",
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
        str(TRAIN_EPISODES_PER_EPOCH),
        "--episode_num_val",
        str(TEST_EPISODES_PER_EPOCH),
        "--episode_num_test",
        str(TEST_EPISODES_PER_EPOCH),
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
    if applied_backbone != "default":
        cmd.extend(["--fewshot_backbone", applied_backbone])
    if not use_external_smnet:
        cmd.extend(["--test_protocol", test_protocol])
        cmd.extend(["--deepemd_fast_val", "true"])
        cmd.extend(
            [
                "--model_selection_split",
                "val",
                "--merge_val_into_train",
                "false",
                "--selection_episode_seed_mode",
                "per_epoch",
                "--selection_episode_seed_offset",
                "100000",
                "--final_test_episode_seed_offset",
                "200000",
            ]
        )
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
                    DEEPEMD_5SHOT_TRAIN_SOLVER,
                    "--deepemd_train_sfc",
                    DEEPEMD_5SHOT_TRAIN_SFC,
                    "--deepemd_train_sfc_update_step",
                    str(DEEPEMD_5SHOT_TRAIN_SFC_STEPS),
                    "--deepemd_train_sfc_bs",
                    str(DEEPEMD_5SHOT_TRAIN_SFC_BS),
                    "--deepemd_test_sfc",
                    DEEPEMD_5SHOT_TEST_SFC,
                    "--deepemd_test_exact",
                    DEEPEMD_5SHOT_TEST_EXACT,
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


def generate_comparison_charts(dataset_name, shots, models, test_protocol="clean"):
    import re

    try:
        from function.function import plot_model_comparison_bar
    except ImportError:
        print("Warning: Could not import plot function, skipping charts")
        return

    results_dir = "results/"
    model_display_names = {model_name: get_model_metadata(model_name)["display_name"] for model_name in models}
    protocol_suffix = result_protocol_suffix(test_protocol)

    for samples in SAMPLES_LIST:
        samples_str = f"{samples}samples" if samples is not None else "allsamples"
        model_results = {}

        for model in models:
            display_name = model_display_names[model]
            model_results[display_name] = {}

            for shot in shots:
                result_file = os.path.join(
                    results_dir,
                    f"results_{dataset_name}_{model}_{samples_str}_{shot}shot{protocol_suffix}.txt",
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
        save_path = os.path.join(results_dir, f"model_comparison_{dataset_name}_{samples_str}{protocol_suffix}.png")
        plot_model_comparison_bar(model_results, training_samples, save_path)
        print(f"Model comparison chart saved to {save_path}")


def main():
    args = get_args()
    if args.spif_global_only == "true" and args.spif_local_only == "true":
        raise ValueError("`--spif_global_only` and `--spif_local_only` cannot both be true.")
    effective_test_protocol = resolve_test_protocol(args.test_protocol, args.dataset_path)
    if effective_test_protocol == "robust" and not dataset_has_hrot_robust_layout(args.dataset_path):
        raise ValueError(
            "--test_protocol robust requires train/val/test_clean and all HROT robust test pools."
        )
    shots = [args.shot_num] if args.shot_num is not None else SHOTS_DEFAULT
    requested_models = [model.strip() for model in args.models.split(",") if model.strip()]
    valid_models = set(get_model_choices())
    invalid_models = [model for model in requested_models if model not in valid_models]
    if invalid_models:
        raise ValueError(f"Unsupported models: {invalid_models}. Valid choices: {sorted(valid_models)}")

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    if args.spifce_ablation_suite != "none":
        if requested_models != ["spifce"]:
            raise ValueError(
                "`--spifce_ablation_suite` currently supports only `--models spifce` "
                f"(got {requested_models})"
            )
        ablation_variants = build_spifce_ablation_variants(args.spifce_ablation_suite)
        samples_list = [EXPERIMENT_MODES[args.mode_id]] if args.mode_id is not None else SAMPLES_LIST
        experiments = [
            {
                "model": "spifce",
                "samples": samples,
                "shot": shot,
                "variant_args": variant["extra_args"],
                "experiment_tag": variant["tag"],
                "experiment_label": variant["label"],
            }
            for variant in ablation_variants
            for samples in samples_list
            for shot in shots
        ]
        print("=" * 72)
        print("pulse_fewshot - SPIFCE Contribution Ablation Suite")
        print("=" * 72)
        if args.mode_id is None:
            print("Mode Mapping:")
            for mode_id, sample_count in EXPERIMENT_MODES.items():
                print(f"  Mode {mode_id}: {sample_count if sample_count else 'All'} samples")
        else:
            samples = EXPERIMENT_MODES[args.mode_id]
            print(f"Mode        : {args.mode_id} ({samples if samples else 'All'} samples)")
        print(f"Models      : {', '.join(requested_models)}")
        print(f"Shots       : {', '.join(f'{shot}-shot' for shot in shots)}")
        print(f"Backbone    : {args.fewshot_backbone}")
        print(f"Ablation    : {args.spifce_ablation_suite}")
        print("Variants    :")
        for variant in ablation_variants:
            print(f"  - {variant['tag']}: {variant['label']}")
        print(f"Dataset     : {args.dataset_path} ({args.dataset_name})")
        print(f"GPU         : {args.gpu_id}")
        print(
            f"Runtime     : workers={args.num_workers}, pin_memory={args.pin_memory}, "
            f"persistent_workers={args.persistent_workers}, "
            f"cudnn(det={FIXED_CUDNN_DETERMINISTIC}, bench={FIXED_CUDNN_BENCHMARK})"
        )
        print(
            "Overrides   : "
            "scheduler=cosine(warmup=5, warmup_start=0.1, eta_min=1e-6), "
            "lr=5e-4, grad_clip=0.0, label_smoothing=0.0, "
            "query(train/val/test)=1/1/1, episodes(train/val/test)=130/150/150, selection=val, merge_val_into_train=false, "
            "augment=off, masks=off, "
            f"DeepEMD 5-shot=train solver={DEEPEMD_5SHOT_TRAIN_SOLVER}, "
            f"train SFC={DEEPEMD_5SHOT_TRAIN_SFC}, "
            f"train SFC steps={DEEPEMD_5SHOT_TRAIN_SFC_STEPS}, "
            f"train SFC bs={DEEPEMD_5SHOT_TRAIN_SFC_BS}, "
            f"test exact={DEEPEMD_5SHOT_TEST_EXACT}, test SFC={DEEPEMD_5SHOT_TEST_SFC}"
        )
        if args.passthrough_args:
            print(f"Forwarded   : {' '.join(args.passthrough_args)}")
        print(f"Test Proto  : {effective_test_protocol}")
        print(f"Total       : {len(experiments)} ablation experiment(s)")
        print("=" * 72)
    elif args.mode_id is not None:
        samples = EXPERIMENT_MODES[args.mode_id]
        experiments = [
            {
                "model": model,
                "samples": samples,
                "shot": shot,
                "variant_args": None,
                "experiment_tag": None,
                "experiment_label": None,
            }
            for model in requested_models
            for shot in shots
        ]
        print("=" * 72)
        print("pulse_fewshot - Single Experiment")
        print("=" * 72)
        print(f"Mode        : {args.mode_id} ({samples if samples else 'All'} samples)")
        print(f"Models      : {', '.join(requested_models)}")
        print(f"Shots       : {', '.join(map(str, shots))}")
        print(f"Backbone    : {args.fewshot_backbone}")
        if any(model.startswith("spif") for model in requested_models):
            print(f"SPIF Ablate : global_only={args.spif_global_only}, local_only={args.spif_local_only}")
        print(f"Dataset     : {args.dataset_path} ({args.dataset_name})")
        print(f"Test Proto  : {effective_test_protocol}")
        print(f"GPU         : {args.gpu_id}")
        print(
            f"Runtime     : workers={args.num_workers}, pin_memory={args.pin_memory}, "
            f"persistent_workers={args.persistent_workers}, "
            f"cudnn(det={FIXED_CUDNN_DETERMINISTIC}, bench={FIXED_CUDNN_BENCHMARK})"
        )
        print(
            "Overrides   : "
            "scheduler=cosine(warmup=5, warmup_start=0.1, eta_min=1e-6), "
            "lr=5e-4, grad_clip=0.0, label_smoothing=0.0, "
            "query(train/val/test)=1/1/1, episodes(train/val/test)=130/150/150, selection=val, merge_val_into_train=false, "
            "augment=off, masks=off, "
            f"DeepEMD 5-shot=train solver={DEEPEMD_5SHOT_TRAIN_SOLVER}, "
            f"train SFC={DEEPEMD_5SHOT_TRAIN_SFC}, "
            f"train SFC steps={DEEPEMD_5SHOT_TRAIN_SFC_STEPS}, "
            f"train SFC bs={DEEPEMD_5SHOT_TRAIN_SFC_BS}, "
            f"test exact={DEEPEMD_5SHOT_TEST_EXACT}, test SFC={DEEPEMD_5SHOT_TEST_SFC}"
        )
        if args.passthrough_args:
            print(f"Forwarded   : {' '.join(args.passthrough_args)}")
        print(f"Total       : {len(experiments)} experiment(s)")
        print("=" * 72)
    else:
        experiments = [
            {
                "model": model,
                "samples": samples,
                "shot": shot,
                "variant_args": None,
                "experiment_tag": None,
                "experiment_label": None,
            }
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
        if any(model.startswith("spif") for model in requested_models):
            print(f"SPIF Ablate : global_only={args.spif_global_only}, local_only={args.spif_local_only}")
        print(f"Dataset     : {args.dataset_path} ({args.dataset_name})")
        print(f"Test Proto  : {effective_test_protocol}")
        print(f"GPU         : {args.gpu_id}")
        print(
            f"Runtime     : workers={args.num_workers}, pin_memory={args.pin_memory}, "
            f"persistent_workers={args.persistent_workers}, "
            f"cudnn(det={FIXED_CUDNN_DETERMINISTIC}, bench={FIXED_CUDNN_BENCHMARK})"
        )
        print(
            "Overrides   : "
            "scheduler=cosine(warmup=5, warmup_start=0.1, eta_min=1e-6), "
            "lr=5e-4, grad_clip=0.0, label_smoothing=0.0, "
            "query(train/val/test)=1/1/1, episodes(train/val/test)=130/150/150, selection=val, merge_val_into_train=false, "
            "augment=off, masks=off, "
            f"DeepEMD 5-shot=train solver={DEEPEMD_5SHOT_TRAIN_SOLVER}, "
            f"train SFC={DEEPEMD_5SHOT_TRAIN_SFC}, "
            f"train SFC steps={DEEPEMD_5SHOT_TRAIN_SFC_STEPS}, "
            f"train SFC bs={DEEPEMD_5SHOT_TRAIN_SFC_BS}, "
            f"test exact={DEEPEMD_5SHOT_TEST_EXACT}, test SFC={DEEPEMD_5SHOT_TEST_SFC}"
        )
        if args.passthrough_args:
            print(f"Forwarded   : {' '.join(args.passthrough_args)}")
        print(f"Total       : {len(experiments)} experiments")
        print("=" * 72)

    success_count = 0
    failed_experiments = []

    for idx, experiment in enumerate(experiments, 1):
        model = experiment["model"]
        samples = experiment["samples"]
        shot = experiment["shot"]
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
            spif_global_only=args.spif_global_only,
            spif_local_only=args.spif_local_only,
            test_protocol=effective_test_protocol,
            passthrough_args=args.passthrough_args,
            variant_args=experiment["variant_args"],
            experiment_tag=experiment["experiment_tag"],
            experiment_label=experiment["experiment_label"],
        )
        if success:
            success_count += 1
        else:
            tag_suffix = f"_{experiment['experiment_tag']}" if experiment["experiment_tag"] else ""
            failed_experiments.append(f"{model}_{shot}shot_{samples if samples else 'all'}samples{tag_suffix}")

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

    if args.spifce_ablation_suite == "none":
        print("\n" + "=" * 60)
        print("Generating comparison charts...")
        print("=" * 60)
        generate_comparison_charts(args.dataset_name, shots, requested_models, effective_test_protocol)
    else:
        print("\n" + "=" * 60)
        print("Skipping standard comparison charts for tagged ablation runs.")
        print("=" * 60)
    print("\nAll experiments completed!")


if __name__ == "__main__":
    main()
