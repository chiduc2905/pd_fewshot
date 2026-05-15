"""Run pulse_fewshot benchmark experiments with a unified training protocol."""

import argparse
from datetime import datetime
import os
import re
import shlex
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
DEFAULT_FINAL_TEST_SEED = 200042
FIXED_CUDNN_DETERMINISTIC = "true"
FIXED_CUDNN_BENCHMARK = "false"
DEEPEMD_5SHOT_TRAIN_SOLVER = "sinkhorn"
DEEPEMD_5SHOT_TRAIN_SFC = "false"
DEEPEMD_5SHOT_TRAIN_SFC_STEPS = 15
DEEPEMD_5SHOT_TRAIN_SFC_BS = 20
DEEPEMD_5SHOT_TEST_EXACT = "false"
DEEPEMD_5SHOT_TEST_SFC = "false"


def default_dataset_path():
    local = Path(__file__).resolve().parent.parent.parent / "dataset" / "scalogram_27_1"
    return str(local) if local.is_dir() else "/workspace/pd_fewshot/scalogram_27_1"


def default_noise_test_root():
    local = (
        Path(__file__).resolve().parent.parent.parent
        / "dataset"
        / "scalogram_27_1_pd_noise_benchmark_test_moderate"
    )
    if local.is_dir():
        return str(local)
    workspace = Path("/workspace/pd_fewshot/scalogram_27_1_pd_noise_benchmark_test_moderate")
    return str(workspace) if workspace.is_dir() else None


def log_cli_command(args, log_path="results/cli_commands.log"):
    """Append the exact launcher CLI with timestamp and WandB project."""
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
    command = shlex.join([sys.executable, *sys.argv])
    project = str(getattr(args, "project", ""))
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as handle:
        handle.write("=" * 88 + "\n")
        handle.write(f"timestamp      : {timestamp}\n")
        handle.write(f"wandb_project  : {project}\n")
        handle.write(f"cwd            : {Path.cwd()}\n")
        handle.write(f"command        : {command}\n")

    print(f"CLI command log: {path}")


def get_args():
    # Disable prefix matching so flags like `--mode test` are forwarded to main.py
    # instead of colliding with `--mode_id` / `--models`.
    parser = argparse.ArgumentParser(
        description="Run all pulse_fewshot benchmark experiments",
        allow_abbrev=False,
    )
    parser.add_argument("--project", type=str, default="pulse_fewshot", help="WandB project name")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=default_dataset_path(),
        help="Path to clean train/val/test dataset",
    )
    parser.add_argument("--dataset_name", type=str, default="knee_aug_split", help="Dataset name for logging")
    parser.add_argument(
        "--test_protocol",
        type=str,
        default="auto",
        choices=["auto", "clean", "noise"],
        help="Final-test protocol. auto uses --noise_test_root when SNR test folders are available.",
    )
    parser.add_argument(
        "--noise_test_root",
        type=str,
        default=default_noise_test_root(),
        help="Root containing SNR benchmark test folders such as test_snr5db_rf_1_15mhz.",
    )
    parser.add_argument(
        "--noise_test_splits",
        type=str,
        default="auto",
        help="Comma-separated SNR split folders to test, or auto to use all test_snr* folders.",
    )
    parser.add_argument(
        "--extra_test_protocols",
        type=str,
        default="none",
        help=(
            "Comma-separated extra final-test protocols to run test-only from the "
            "trained checkpoint after the primary protocol. Extra protocols use "
            "--final_test_seed only; --final_test_seeds applies to the primary protocol."
        ),
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--final_test_seed",
        type=int,
        default=DEFAULT_FINAL_TEST_SEED,
        help=(
            "Fixed final-test episode seed passed to main.py. "
            "Default keeps the previous seed-42 benchmark episodes."
        ),
    )
    parser.add_argument(
        "--final_test_seeds",
        type=str,
        default=None,
        help=(
            "Comma-separated final-test episode seeds. The first seed is used for "
            "the normal post-train final test; remaining seeds run test-only from "
            "the saved final checkpoint."
        ),
    )
    parser.add_argument("--shot_num", type=int, default=None, choices=[1, 5], help="Optional fixed shot number")
    parser.add_argument("--mode_id", type=int, default=None, choices=[1, 2, 3, 4], help="Optional sample mode")
    parser.add_argument(
        "--mode_ids",
        type=str,
        default=None,
        help=(
            "Comma-separated mode ids (e.g. 3,4). Runs all combinations of models × listed modes × shots. "
            "Mutually exclusive with --mode_id."
        ),
    )
    parser.add_argument(
        "--models",
        type=str,
        default="covamnet,protonet,cosine,relationnet,matchingnet,dn4,feat,deepemd,maml,can,frn,deepbdc,hierarchical_episodic_ssm_net,hierarchical_consensus_slot_mamba_net,fgwuot_fsl",
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
    parser.add_argument(
        "--ours_ablation_suite",
        type=str,
        default="none",
        choices=["none", "contrib", "no_egsm_gap"],
        help=(
            "Expand Ours contribution ablations with tagged runs. "
            "contrib=all four (full, full_ot, no_egsm, gap); "
            "no_egsm_gap=only no_egsm and gap."
        ),
    )
    parser.add_argument(
        "--m2_ablate_T",
        type=str,
        default="true",
        choices=["true", "false"],
        help="M2 only: forwards --hrot_ecot_m2_ablate_threshold_mass (drops T * mass); default true.",
    )
    parser.add_argument(
        "--m2_cost_per_mass",
        type=str,
        default="false",
        choices=["true", "false"],
        help="M2 only: forwards --hrot_ecot_m2_cost_per_mass_score true (cost/mass score; mutually exclusive with m2_ablate_T).",
    )
    args, passthrough_args = parser.parse_known_args()
    extra = list(passthrough_args)
    ablate_t = str(getattr(args, "m2_ablate_T", "true")).lower() == "true"
    cost_pm = str(getattr(args, "m2_cost_per_mass", "false")).lower() == "true"
    if cost_pm:
        extra += [
            "--hrot_ecot_m2_cost_per_mass_score",
            "true",
            "--hrot_ecot_m2_ablate_threshold_mass",
            "false",
        ]
    elif ablate_t:
        extra += [
            "--hrot_ecot_m2_ablate_threshold_mass",
            "true",
            "--hrot_ecot_m2_cost_per_mass_score",
            "false",
        ]
    else:
        extra += [
            "--hrot_ecot_m2_ablate_threshold_mass",
            "false",
            "--hrot_ecot_m2_cost_per_mass_score",
            "false",
        ]
    args.passthrough_args = extra
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
    "m2",
    "ours",
    "m2_uot",
    "m2_full_ot",
    "jecot_m2",
    "jecot_m2_uot",
    "jecot_m2_full_ot",
    "crj_fsl",
    "fgwuot_fsl",
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
    expected = {"surface", "internal", "corona", "notpd", "nopd", "not_pd"}
    if not Path(split_path).is_dir():
        return False
    child_dirs = {
        child.name.lower()
        for child in Path(split_path).iterdir()
        if child.is_dir()
    }
    return bool(child_dirs & expected)


def discover_noise_test_splits(noise_test_root, requested_splits="auto"):
    if not noise_test_root:
        return []
    root = Path(noise_test_root)
    if not root.is_dir():
        return []

    requested = str(requested_splits or "auto").strip()
    if requested.lower() == "auto":
        names = [
            child.name
            for child in root.iterdir()
            if child.is_dir()
            and child.name.lower().startswith("test_snr")
            and split_has_class_dirs(child)
        ]
        if not names:
            names = [
                child.name
                for child in root.iterdir()
                if child.is_dir() and split_has_class_dirs(child)
            ]
        return sorted(names, key=snr_sort_key)
    return [name.strip() for name in requested.split(",") if name.strip()]


def dataset_has_training_layout(dataset_path):
    root = Path(dataset_path)
    return (
        (root / "train").is_dir()
        and (root / "val").is_dir()
        and ((root / "test").is_dir() or (root / "test_clean").is_dir())
    )


def dataset_has_noise_benchmark_layout(dataset_path):
    return bool(discover_noise_test_splits(dataset_path, "auto"))


def infer_training_dataset_path(noise_test_root):
    root = Path(noise_test_root)
    name = root.name
    base_name = name.split("_pd_noise_benchmark_test")[0]
    if not base_name or base_name == name:
        return None
    candidate = root.parent / base_name
    return candidate if dataset_has_training_layout(candidate) else None


def resolve_test_protocol(requested_protocol, noise_test_root, noise_test_splits="auto"):
    requested_protocol = str(requested_protocol).lower()
    if requested_protocol == "auto":
        return "noise" if discover_noise_test_splits(noise_test_root, noise_test_splits) else "clean"
    if requested_protocol == "noise" and not discover_noise_test_splits(noise_test_root, noise_test_splits):
        raise ValueError("--test_protocol noise requires --noise_test_root with at least one SNR test folder.")
    return requested_protocol


def result_protocol_suffix(test_protocol, test_split_name=None):
    if test_split_name:
        return f"_{sanitize_result_tag(test_split_name)}"
    test_protocol = str(test_protocol).lower()
    if test_protocol in {"auto", "clean", ""}:
        return ""
    return f"_{sanitize_result_tag(test_protocol)}"


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


def build_ours_ablation_variants():
    return [
        {
            "tag": "ours_full",
            "label": "Full Ours: local descriptors + UOT + EGSM",
            "extra_args": ["--ours_ablation", "full"],
        },
        {
            "tag": "ours_full_ot",
            "label": "Balanced full OT instead of UOT",
            "extra_args": ["--ours_ablation", "full_ot"],
        },
        {
            "tag": "ours_no_egsm",
            "label": "Ours without EGSM marginals",
            "extra_args": ["--ours_ablation", "no_egsm"],
        },
        {
            "tag": "ours_gap",
            "label": "GAP descriptor cost + UOT + EGSM instead of local-descriptor cost",
            "extra_args": ["--ours_ablation", "gap"],
        },
    ]


def build_ours_no_egsm_gap_variants():
    """Subset: EGSM-off control and GAP-descriptor control (both tagged)."""
    all_variants = build_ours_ablation_variants()
    keep = {"ours_no_egsm", "ours_gap"}
    return [v for v in all_variants if v["tag"] in keep]


def build_final_checkpoint_path(dataset_name, model, samples, shot, experiment_tag=None):
    samples_suffix = f"{samples}samples" if samples is not None else "all"
    tag_suffix = f"_{experiment_tag}" if experiment_tag else ""
    return os.path.join(
        "checkpoints",
        f"{dataset_name}_{model}_{samples_suffix}_{shot}shot{tag_suffix}_final.pth",
    )


def set_cli_option(cmd, flag, value):
    updated = []
    replaced = False
    i = 0
    while i < len(cmd):
        if cmd[i] == flag:
            updated.extend([flag, str(value)])
            i += 2
            replaced = True
        else:
            updated.append(cmd[i])
            i += 1
    if not replaced:
        updated.extend([flag, str(value)])
    return updated


def remove_cli_option(cmd, flag):
    updated = []
    i = 0
    while i < len(cmd):
        if cmd[i] == flag:
            i += 2
        else:
            updated.append(cmd[i])
            i += 1
    return updated


def run_experiment(
    model,
    shot,
    samples,
    dataset_path,
    dataset_name,
    project,
    seed,
    final_test_seed,
    gpu_id,
    fewshot_backbone,
    num_workers,
    pin_memory,
    persistent_workers,
    prefetch_factor,
    spif_global_only,
    spif_local_only,
    test_protocol,
    noise_test_root,
    noise_test_splits,
    passthrough_args=None,
    variant_args=None,
    experiment_tag=None,
    experiment_label=None,
    extra_final_test_seeds=None,
    extra_test_protocols=None,
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
    if str(test_protocol).lower() == "noise":
        print(f"Noise Root  : {noise_test_root}")
        print(f"Noise Splits: {', '.join(discover_noise_test_splits(noise_test_root, noise_test_splits))}")
    print("Protocol    : selection=val, merge_val_into_train=false, episodes(train/val/test)=130/150/150")
    extra_test_protocols = list(extra_test_protocols or [])
    if extra_test_protocols:
        print(f"Extra Tests : {', '.join(extra_test_protocols)} (seed={final_test_seed})")
    if extra_final_test_seeds:
        seed_text = ",".join(str(seed) for seed in [final_test_seed, *extra_final_test_seeds])
        print(f"Test Seeds  : final_test_seeds={seed_text}")
    else:
        print(f"Test Seed   : final_test_seed={final_test_seed}")
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
        if str(test_protocol).lower() == "noise":
            cmd.extend(["--noise_test_root", noise_test_root])
            cmd.extend(["--noise_test_splits", noise_test_splits])
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
                "--final_test_seed",
                str(final_test_seed),
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
    except subprocess.CalledProcessError as exc:
        print(f"Error: {exc}")
        return False

    if extra_final_test_seeds or extra_test_protocols:
        if use_external_smnet:
            print("Warning: extra final tests are not implemented for external smnet models; skipping.")
            return True

        checkpoint_path = build_final_checkpoint_path(
            dataset_name=dataset_name,
            model=model,
            samples=samples,
            shot=shot,
            experiment_tag=experiment_tag,
        )
        if not os.path.exists(checkpoint_path):
            print(f"Error: trained checkpoint not found for extra tests: {checkpoint_path}")
            return False

        for extra_seed in extra_final_test_seeds:
            seed_tag = f"testseed{extra_seed}"
            if experiment_tag:
                seed_tag = f"{experiment_tag}_{seed_tag}"
            test_cmd = list(cmd)
            test_cmd = set_cli_option(test_cmd, "--mode", "test")
            test_cmd = set_cli_option(test_cmd, "--weights", checkpoint_path)
            test_cmd = set_cli_option(test_cmd, "--final_test_seed", str(extra_seed))
            test_cmd = set_cli_option(test_cmd, "--experiment_tag", seed_tag)

            print(f"\nExtra final test: seed={extra_seed}, checkpoint={checkpoint_path}")
            try:
                subprocess.run(test_cmd, check=True)
            except subprocess.CalledProcessError as exc:
                print(f"Error: {exc}")
                return False

        for extra_protocol in extra_test_protocols:
            protocol_tag = f"{sanitize_result_tag(extra_protocol)}_testseed{final_test_seed}"
            if experiment_tag:
                protocol_tag = f"{experiment_tag}_{protocol_tag}"
            test_cmd = list(cmd)
            test_cmd = set_cli_option(test_cmd, "--mode", "test")
            test_cmd = set_cli_option(test_cmd, "--weights", checkpoint_path)
            test_cmd = set_cli_option(test_cmd, "--final_test_seed", str(final_test_seed))
            test_cmd = set_cli_option(test_cmd, "--test_protocol", extra_protocol)
            test_cmd = set_cli_option(test_cmd, "--experiment_tag", protocol_tag)
            if str(extra_protocol).lower() == "noise":
                test_cmd = set_cli_option(test_cmd, "--noise_test_root", noise_test_root)
                test_cmd = set_cli_option(test_cmd, "--noise_test_splits", noise_test_splits)
            else:
                test_cmd = remove_cli_option(test_cmd, "--noise_test_root")
                test_cmd = remove_cli_option(test_cmd, "--noise_test_splits")

            print(
                f"\nExtra {extra_protocol} final test: "
                f"seed={final_test_seed}, checkpoint={checkpoint_path}"
            )
            try:
                subprocess.run(test_cmd, check=True)
            except subprocess.CalledProcessError as exc:
                print(f"Error: {exc}")
                return False

    return True


def generate_comparison_charts(dataset_name, shots, models, test_protocol="clean", test_split_names=None):
    import re

    try:
        from function.function import plot_model_comparison_bar
    except ImportError:
        print("Warning: Could not import plot function, skipping charts")
        return

    results_dir = "results/"
    model_display_names = {model_name: get_model_metadata(model_name)["display_name"] for model_name in models}
    if test_split_names:
        for test_split_name in test_split_names:
            protocol_suffix = result_protocol_suffix(test_protocol, test_split_name)
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
                            match = re.search(r"Accuracy\s*:\s*([\d.]+)\s*(?:\+/-|Â±)", content)
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
                save_path = os.path.join(
                    results_dir,
                    f"model_comparison_{dataset_name}_{samples_str}{protocol_suffix}.png",
                )
                plot_model_comparison_bar(model_results, training_samples, save_path)
                print(f"Model comparison chart saved to {save_path}")
        return
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


def parse_mode_ids(mode_ids_str):
    if not mode_ids_str or not str(mode_ids_str).strip():
        return None
    parsed = []
    for token in str(mode_ids_str).split(","):
        token = token.strip()
        if not token:
            continue
        mid = int(token)
        if mid not in EXPERIMENT_MODES:
            raise ValueError(f"Invalid mode id {mid} in --mode_ids (expected one of {sorted(EXPERIMENT_MODES)})")
        parsed.append(mid)
    if not parsed:
        return None
    return parsed


def parse_final_test_seeds(seed_list_str):
    if seed_list_str is None or not str(seed_list_str).strip():
        return None
    parsed = []
    for token in str(seed_list_str).split(","):
        token = token.strip()
        if not token:
            continue
        parsed.append(int(token))
    if not parsed:
        return None
    return parsed


def parse_extra_test_protocols(protocols_str):
    if protocols_str is None or not str(protocols_str).strip():
        return []
    parsed = []
    for token in str(protocols_str).split(","):
        protocol = token.strip().lower()
        if not protocol or protocol == "none":
            continue
        if protocol == "auto":
            raise ValueError("--extra_test_protocols accepts concrete protocols: clean or noise.")
        if protocol not in {"clean", "noise"}:
            raise ValueError(
                f"Invalid extra test protocol '{protocol}' in --extra_test_protocols "
                "(expected clean or noise)."
            )
        if protocol not in parsed:
            parsed.append(protocol)
    return parsed


def format_final_test_seed_line(args):
    extra_seeds = list(getattr(args, "extra_final_test_seeds", []) or [])
    if extra_seeds:
        seeds = [int(args.final_test_seed), *[int(seed) for seed in extra_seeds]]
        return f"Test Seeds  : final_test_seeds={','.join(str(seed) for seed in seeds)}"
    return f"Test Seed   : final_test_seed={args.final_test_seed}"


def print_extra_test_protocol_line(args):
    protocols = list(getattr(args, "extra_test_protocols", []) or [])
    if protocols:
        print(f"Extra Tests : {','.join(protocols)} with final_test_seed={args.final_test_seed}")


def main():
    args = get_args()
    final_test_seeds = parse_final_test_seeds(getattr(args, "final_test_seeds", None))
    if final_test_seeds is None:
        final_test_seeds = [int(args.final_test_seed)]
    args.final_test_seed = int(final_test_seeds[0])
    args.extra_final_test_seeds = [int(seed) for seed in final_test_seeds[1:]]
    requested_extra_test_protocols = parse_extra_test_protocols(getattr(args, "extra_test_protocols", None))
    parsed_mode_ids = parse_mode_ids(getattr(args, "mode_ids", None))
    if args.mode_id is not None and parsed_mode_ids is not None:
        raise ValueError("Use either --mode_id or --mode_ids, not both.")
    if args.spif_global_only == "true" and args.spif_local_only == "true":
        raise ValueError("`--spif_global_only` and `--spif_local_only` cannot both be true.")
    if dataset_has_noise_benchmark_layout(args.dataset_path) and not dataset_has_training_layout(args.dataset_path):
        args.noise_test_root = args.dataset_path
        inferred_dataset_path = infer_training_dataset_path(args.dataset_path)
        if inferred_dataset_path is None:
            raise ValueError(
                "--dataset_path points to a noise benchmark root. "
                "Pass the clean train/val/test dataset with --dataset_path, "
                "or place the matching clean dataset beside the noise root."
            )
        print(
            "Interpreting --dataset_path as --noise_test_root; "
            f"using clean dataset: {inferred_dataset_path}"
        )
        args.dataset_path = str(inferred_dataset_path)
    effective_test_protocol = resolve_test_protocol(
        args.test_protocol,
        args.noise_test_root,
        args.noise_test_splits,
    )
    noise_test_split_names = (
        discover_noise_test_splits(args.noise_test_root, args.noise_test_splits)
        if effective_test_protocol == "noise"
        else []
    )
    extra_test_protocols = [
        protocol for protocol in requested_extra_test_protocols if protocol != effective_test_protocol
    ]
    if "noise" in extra_test_protocols:
        extra_noise_splits = discover_noise_test_splits(args.noise_test_root, args.noise_test_splits)
        if not extra_noise_splits:
            raise ValueError(
                "--extra_test_protocols noise requires --noise_test_root with at least one SNR test folder."
            )
        if not noise_test_split_names:
            noise_test_split_names = extra_noise_splits
    args.extra_test_protocols = extra_test_protocols
    shots = [args.shot_num] if args.shot_num is not None else SHOTS_DEFAULT
    requested_models = [model.strip() for model in args.models.split(",") if model.strip()]
    valid_models = set(get_model_choices())
    invalid_models = [model for model in requested_models if model not in valid_models]
    if invalid_models:
        raise ValueError(f"Unsupported models: {invalid_models}. Valid choices: {sorted(valid_models)}")

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    log_cli_command(args)

    if args.spifce_ablation_suite != "none" and args.ours_ablation_suite != "none":
        raise ValueError("Use only one ablation suite at a time: --spifce_ablation_suite or --ours_ablation_suite.")

    if args.ours_ablation_suite != "none":
        if requested_models != ["ours"]:
            raise ValueError(
                "`--ours_ablation_suite` currently supports only `--models ours` "
                f"(got {requested_models})"
            )
        ablation_variants = (
            build_ours_no_egsm_gap_variants()
            if args.ours_ablation_suite == "no_egsm_gap"
            else build_ours_ablation_variants()
        )
        samples_list = (
            [EXPERIMENT_MODES[args.mode_id]]
            if args.mode_id is not None
            else ([EXPERIMENT_MODES[mid] for mid in parsed_mode_ids] if parsed_mode_ids else SAMPLES_LIST)
        )
        experiments = [
            {
                "model": "ours",
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
        print("pulse_fewshot - Ours Contribution Ablation Suite")
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
        print(f"Ablation    : {args.ours_ablation_suite}")
        print("Variants    :")
        for variant in ablation_variants:
            print(f"  - {variant['tag']}: {variant['label']}")
        print(f"Dataset     : {args.dataset_path} ({args.dataset_name})")
        print(f"Test Proto  : {effective_test_protocol}")
        print(format_final_test_seed_line(args))
        print_extra_test_protocol_line(args)
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
            "augment=off, masks=off, Ours full=local_descriptors+UOT(rho=0.8)+EGSM, "
            f"DeepEMD 5-shot=train solver={DEEPEMD_5SHOT_TRAIN_SOLVER}, "
            f"train SFC={DEEPEMD_5SHOT_TRAIN_SFC}, "
            f"train SFC steps={DEEPEMD_5SHOT_TRAIN_SFC_STEPS}, "
            f"train SFC bs={DEEPEMD_5SHOT_TRAIN_SFC_BS}, "
            f"test exact={DEEPEMD_5SHOT_TEST_EXACT}, test SFC={DEEPEMD_5SHOT_TEST_SFC}"
        )
        if args.passthrough_args:
            print(f"Forwarded   : {' '.join(args.passthrough_args)}")
        if effective_test_protocol == "noise":
            print(f"Noise Root  : {args.noise_test_root}")
            print(f"Noise Splits: {', '.join(noise_test_split_names)}")
        print(f"Total       : {len(experiments)} ablation experiment(s)")
        print("=" * 72)
    elif args.spifce_ablation_suite != "none":
        if requested_models != ["spifce"]:
            raise ValueError(
                "`--spifce_ablation_suite` currently supports only `--models spifce` "
                f"(got {requested_models})"
            )
        ablation_variants = build_spifce_ablation_variants(args.spifce_ablation_suite)
        samples_list = (
            [EXPERIMENT_MODES[args.mode_id]]
            if args.mode_id is not None
            else ([EXPERIMENT_MODES[mid] for mid in parsed_mode_ids] if parsed_mode_ids else SAMPLES_LIST)
        )
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
        print(format_final_test_seed_line(args))
        print_extra_test_protocol_line(args)
        if effective_test_protocol == "noise":
            print(f"Noise Root  : {args.noise_test_root}")
            print(f"Noise Splits: {', '.join(noise_test_split_names)}")
        print(f"Total       : {len(experiments)} ablation experiment(s)")
        print("=" * 72)
    elif parsed_mode_ids is not None:
        experiments = [
            {
                "model": model,
                "samples": EXPERIMENT_MODES[mode_id],
                "shot": shot,
                "variant_args": None,
                "experiment_tag": None,
                "experiment_label": None,
            }
            for model in requested_models
            for mode_id in parsed_mode_ids
            for shot in shots
        ]
        print("=" * 72)
        print("pulse_fewshot - Multi-Mode Experiment")
        print("=" * 72)
        print("Modes       : " + ", ".join(f"{mid} ({EXPERIMENT_MODES[mid] if EXPERIMENT_MODES[mid] else 'All'} samples)" for mid in parsed_mode_ids))
        print(f"Models      : {', '.join(requested_models)}")
        print(f"Shots       : {', '.join(map(str, shots))}")
        print(f"Backbone    : {args.fewshot_backbone}")
        if any(model.startswith("spif") for model in requested_models):
            print(f"SPIF Ablate : global_only={args.spif_global_only}, local_only={args.spif_local_only}")
        print(f"Dataset     : {args.dataset_path} ({args.dataset_name})")
        print(f"Test Proto  : {effective_test_protocol}")
        print(format_final_test_seed_line(args))
        print_extra_test_protocol_line(args)
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
        if effective_test_protocol == "noise":
            print(f"Noise Root  : {args.noise_test_root}")
            print(f"Noise Splits: {', '.join(noise_test_split_names)}")
        print(f"Total       : {len(experiments)} experiment(s)")
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
        print(format_final_test_seed_line(args))
        print_extra_test_protocol_line(args)
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
        if effective_test_protocol == "noise":
            print(f"Noise Root  : {args.noise_test_root}")
            print(f"Noise Splits: {', '.join(noise_test_split_names)}")
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
        print(format_final_test_seed_line(args))
        print_extra_test_protocol_line(args)
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
        if effective_test_protocol == "noise":
            print(f"Noise Root  : {args.noise_test_root}")
            print(f"Noise Splits: {', '.join(noise_test_split_names)}")
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
            final_test_seed=args.final_test_seed,
            gpu_id=args.gpu_id,
            fewshot_backbone=args.fewshot_backbone,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            persistent_workers=args.persistent_workers,
            prefetch_factor=args.prefetch_factor,
            spif_global_only=args.spif_global_only,
            spif_local_only=args.spif_local_only,
            test_protocol=effective_test_protocol,
            noise_test_root=args.noise_test_root,
            noise_test_splits=args.noise_test_splits,
            passthrough_args=args.passthrough_args,
            variant_args=experiment["variant_args"],
            experiment_tag=experiment["experiment_tag"],
            experiment_label=experiment["experiment_label"],
            extra_final_test_seeds=args.extra_final_test_seeds,
            extra_test_protocols=args.extra_test_protocols,
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
        generate_comparison_charts(
            args.dataset_name,
            shots,
            requested_models,
            effective_test_protocol,
            noise_test_split_names if effective_test_protocol == "noise" else None,
        )
    else:
        print("\n" + "=" * 60)
        print("Skipping standard comparison charts for tagged ablation runs.")
        print("=" * 60)
    print("\nAll experiments completed!")


if __name__ == "__main__":
    main()
