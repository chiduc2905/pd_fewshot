"""Run pulse_fewshot benchmark experiments with a unified training protocol."""

import argparse
from datetime import datetime
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path, PurePosixPath

from net.model_factory import get_model_choices, get_model_metadata


SAMPLES_LIST = [60, 160, 240, None]
# Ours contribution ablations: only these training-set sizes (ignores --mode_id / --mode_ids).
OURS_ABLATION_SAMPLE_COUNTS = [60, 240]
OURS_FINAL_SAMPLE_COUNTS = SAMPLES_LIST
OURS_FINAL_MODEL_NAME = "ours_final"
OURS_FINAL_PARTIAL_OT_MODEL_NAME = "ours_final_partial_ot"
CFUGET_MODEL_NAME = "cfuget"
SHOTS_DEFAULT = [1, 5]
TRAIN_QUERY_NUM = 1
EVAL_QUERY_NUM = 1
TRAIN_EPISODES_PER_EPOCH = 130
TEST_EPISODES_PER_EPOCH = 150
DEFAULT_FINAL_TEST_SEED = 200042
DEFAULT_SELECTION_BASE_SEED = 42
DEFAULT_SELECTION_EPISODE_SEED_OFFSET = 100000
FIXED_CUDNN_DETERMINISTIC = "true"
FIXED_CUDNN_BENCHMARK = "false"
DEEPEMD_5SHOT_TRAIN_SOLVER = "sinkhorn"
DEEPEMD_5SHOT_TRAIN_SFC = "false"
DEEPEMD_5SHOT_TRAIN_SFC_STEPS = 15
DEEPEMD_5SHOT_TRAIN_SFC_BS = 20
DEEPEMD_5SHOT_TEST_EXACT = "false"
DEEPEMD_5SHOT_TEST_SFC = "false"
PAPER_BASELINE_MODELS = [
    "cosine",
    "protonet",
    "relationnet",
    "dn4",
    "feat",
    "can",
    "frn",
    "covamnet",
    "deepbdc",
    "deepemd",
]
MODEL_GROUP_ALIASES = {
    "paper_baselines": PAPER_BASELINE_MODELS,
    "baseline_papers": PAPER_BASELINE_MODELS,
    "image_baselines": PAPER_BASELINE_MODELS,
    "table_baselines": PAPER_BASELINE_MODELS,
}
DEFAULT_DATASET_ROOT = PurePosixPath("/workspace/pd_fewshot")
BASE_SCALOGRAM_DATASET = "scalogram_27_1"
MOTHERWAVE_DATASET_FOLDERS = ("scalogram_27_1_cgau4", "scalogram_27_1_fbsp3")
NOISE_BENCHMARK_SUFFIX = "_pd_noise_benchmark_test_moderate"


def default_dataset_path(dataset_folder=BASE_SCALOGRAM_DATASET):
    return str(DEFAULT_DATASET_ROOT / dataset_folder)


def default_noise_test_root(dataset_folder=BASE_SCALOGRAM_DATASET):
    return str(DEFAULT_DATASET_ROOT / f"{dataset_folder}{NOISE_BENCHMARK_SUFFIX}")


def motherwave_dataset_name(base_dataset_name, dataset_folder):
    suffix_prefix = f"{BASE_SCALOGRAM_DATASET}_"
    suffix = dataset_folder[len(suffix_prefix):] if dataset_folder.startswith(suffix_prefix) else dataset_folder
    base = str(base_dataset_name or "").strip()
    if not base or base == dataset_folder or base.endswith(f"_{suffix}"):
        return base or dataset_folder
    return f"{base}_{suffix}"


def dataset_root_for_child_paths(dataset_path):
    dataset_path = str(dataset_path)
    if dataset_path.startswith("/"):
        path = PurePosixPath(dataset_path)
    else:
        path = Path(dataset_path)
    known_dataset_names = {BASE_SCALOGRAM_DATASET, *MOTHERWAVE_DATASET_FOLDERS}
    if path.name in known_dataset_names or path.name.endswith(NOISE_BENCHMARK_SUFFIX):
        return path.parent
    return path


def build_dataset_specs(args):
    if str(getattr(args, "test_motherwave_datasets", "false")).lower() != "true":
        return [
            {
                "path": str(args.dataset_path),
                "name": str(args.dataset_name),
                "noise_test_root": args.noise_test_root,
            }
        ]

    dataset_root = dataset_root_for_child_paths(args.dataset_path)
    return [
        {
            "path": str(dataset_root / dataset_folder),
            "name": motherwave_dataset_name(args.dataset_name, dataset_folder),
            "noise_test_root": str(dataset_root / f"{dataset_folder}{NOISE_BENCHMARK_SUFFIX}"),
        }
        for dataset_folder in MOTHERWAVE_DATASET_FOLDERS
    ]


def print_dataset_plan(dataset_specs):
    if len(dataset_specs) == 1:
        spec = dataset_specs[0]
        print(f"Dataset     : {spec['path']} ({spec['name']})")
        return
    print("Datasets    :")
    for spec in dataset_specs:
        print(f"  - {spec['path']} ({spec['name']})")


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
        "--test_motherwave_datasets",
        "--motherwave_datasets",
        dest="test_motherwave_datasets",
        type=str,
        default="false",
        choices=["true", "false"],
        help=(
            "Run the same experiment plan on the alternate mother-wave scalogram datasets "
            "scalogram_27_1_cgau4 and scalogram_27_1_fbsp3 under /workspace/pd_fewshot."
        ),
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
    parser.add_argument(
        "--seed",
        type=str,
        default="42",
        help="Random training seed, or comma-separated seeds such as 44,45,46",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated training seeds. Overrides --seed when provided.",
    )
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
        "--selection_seed",
        type=int,
        default=DEFAULT_SELECTION_BASE_SEED,
        help=(
            "Base seed used for val/model-selection episodes. "
            "Default keeps validation episodes identical to the original --seed 42 run."
        ),
    )
    parser.add_argument(
        "--final_test_seeds",
        type=str,
        default=None,
        help=(
            "Comma-separated final-test episode seeds. The first seed is used for "
            "the normal post-train final test; remaining seeds run test-only from "
            "the saved best checkpoint."
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
        help=(
            "Comma-separated model registry names to run. "
            "Aliases: paper_baselines/image_baselines for cosine, protonet, relationnet, "
            "dn4, feat, can, frn, covamnet, deepbdc, deepemd."
        ),
    )
    parser.add_argument(
        "--experiment_tag",
        type=str,
        default=None,
        help=(
            "Optional base tag for result/checkpoint filenames. With multiple training seeds, "
            "use {seed} in the tag or the runner appends _seed<seed> automatically."
        ),
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
    parser.add_argument(
        "--skip_existing",
        type=str,
        default="false",
        choices=["true", "false"],
        help=(
            "Skip an experiment/test when all expected result files already exist. "
            "If primary clean results exist but requested extra tests are missing, "
            "reuse the tagged best checkpoint for test-only runs."
        ),
    )
    parser.add_argument(
        "--result_artifacts",
        type=str,
        default="figures_only",
        choices=["figures_only", "all"],
        help="Forwarded to main.py. figures_only keeps results_*.txt and only saves confusion matrix + t-SNE figures.",
    )
    parser.add_argument(
        "--log_cli_command",
        type=str,
        default="false",
        choices=["true", "false"],
        help="Append this launcher CLI to results/cli_commands.log.",
    )
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
            "Expand Ours contribution ablations with tagged runs (training samples: 60 and 240 only). "
            "EGSM-off variant runs first. "
            "contrib=no_egsm, full, full_ot, gap; no_egsm_gap=no_egsm and gap only."
        ),
    )
    parser.add_argument(
        "--ours_final_ablation_suite",
        type=str,
        default="none",
        choices=[
            "none",
            "contrib",
            "rho_grid",
            "complete",
            "partial_ot",
            "tau_shot_off",
            "pooling",
            "dmuot",
            "mass_on",
            "pst_dm",
            "evidence",
            "pulse_region",
        ],
        help=(
            "Expand Ours-Final runs with tagged variants. "
            "contrib=full, ccem_uot, partial_ot, full_ot, gap, mass_off; "
            "partial_ot=only the fast Partial OT + cost-per-mass variant; "
            "rho_grid=rho 0.6,0.7,0.8,0.9 at 60/5-shot and 240/1-shot unless "
            "--ours_final_rho_values is set; "
            "complete=contrib over all modes/shots plus restricted rho_grid; "
            "tau_shot_off=Ours-Final with ECOT tau-shot pooling disabled; "
            "pooling=class-pooled support before OT plus fixed shot pooling; "
            "dmuot=token-g, cost modulation, and discriminative marginal sweeps; "
            "mass_on=shot-consensus threshold-mass score variants; "
            "evidence=cost-derived evidence marginal ablation (query/support/both/rival); "
            "pulse_region=baseline Ours-Final plus pulse cost guidance and pulse evidence-score controls."
        ),
    )
    parser.add_argument(
        "--ours_final_rho_values",
        type=str,
        default=None,
        help=(
            "Comma-separated rho values for --ours_final_ablation_suite rho_grid/complete "
            "(e.g. 0.9,0.95). Defaults to 0.6,0.7,0.8,0.9."
        ),
    )
    parser.add_argument(
        "--ours_final_ablation_variants",
        type=str,
        default="all",
        help=(
            "Comma-separated Ours-Final variant subset for --ours_final_ablation_suite. "
            "Aliases: full/original/uot, ccem_uot/evidence, full_ot/ot/balanced_ot, "
            "partial_ot/fast_partial_ot, gap, mass_off, class_pooled, fixed_shot_pooling, "
            "tau_shot_off, rho_<value>. "
            "Default all keeps the suite's normal variants."
        ),
    )
    parser.add_argument(
        "--cfuget_ablation_suite",
        type=str,
        default="none",
        choices=["none", "contrib", "runtime", "complete"],
        help=(
            "Expand RC-UOT runs with tagged variants. "
            "contrib=full, no_rival, no_coherent_score, cost_only; "
            "runtime=solver/evidence-temperature checks; complete=contrib+runtime."
        ),
    )
    parser.add_argument(
        "--cfuget_ablation_variants",
        type=str,
        default="all",
        help=(
            "Comma-separated RC-UOT variant subset for --cfuget_ablation_suite. "
            "Aliases: full, no_rival, no_coherent_score, cost_only, sink30, sink70, "
            "tau005, tau010, margin0, margin005. "
            "Default all keeps the suite's normal variants."
        ),
    )
    parser.add_argument(
        "--spot_ablation_suite",
        type=str,
        default="none",
        choices=["none", "core", "runtime", "sweep", "controller", "complete"],
        help=(
            "Expand MM-SPOT-FSL multi-mass POT runs with tagged variants. "
            "core=single s=0.4 vs K=3 mass bank; runtime=K/iteration cost checks; "
            "sweep=mass/epsilon/aggregation ablations; controller=learned mass selector ablation; "
            "complete=core+runtime+sweep."
        ),
    )
    parser.add_argument(
        "--spot_ablation_variants",
        type=str,
        default="all",
        help=(
            "Comma-separated MM-SPOT-FSL variant subset for --spot_ablation_suite. "
            "Aliases: default/k3, single/s04, iter60, iter120, k5, eps003, eps005, eps010, "
            "mean, controller. Default all keeps the suite's normal variants."
        ),
    )
    parser.add_argument(
        "--pare_ablation_suite",
        type=str,
        default="none",
        choices=["none", "core", "runtime", "alpha", "complete"],
        help=(
            "Expand PARE-FSL runs with tagged variants (learned alpha, not mass banks). "
            "core=default EGSM+learned alpha vs uniform vs fixed alpha; "
            "runtime=sinkhorn iteration budget; alpha=alpha range / reg ablations; "
            "complete=core+runtime."
        ),
    )
    parser.add_argument(
        "--sgpot_ablation_suite",
        type=str,
        default="none",
        choices=["none", "core", "hparam", "complete"],
        help=(
            "Expand SG-POT runs with tagged ablation variants. "
            "core=full vs uniform_marginals vs fixed_s vs cost_only_score vs no_entropy_reg; "
            "hparam=mass fraction range + epsilon + entropy reg weight sweeps; "
            "complete=core+hparam."
        ),
    )
    parser.add_argument(
        "--sgpot_ablation_variants",
        type=str,
        default="all",
        help=(
            "Comma-separated SG-POT variant subset for --sgpot_ablation_suite. "
            "Default all keeps the suite's normal variants."
        ),
    )
    parser.add_argument(
        "--pare_ablation_variants",
        type=str,
        default="all",
        help=(
            "Comma-separated PARE-FSL variant subset for --pare_ablation_suite. "
            "Aliases: default, uniform, fixed_alpha, iter60, iter80, iter120, "
            "eps002, eps004, eps006, shot_mean, alpha_wide, alpha_noreg. "
            "Default all keeps the suite's normal variants."
        ),
    )
    parser.add_argument(
        "--export_uot_evidence_figure",
        type=str,
        default="auto",
        choices=["auto", "true", "false"],
        help=(
            "Forwarded to main.py to export UOT/Partial-OT transport evidence and "
            "support comparison figures during final test. "
            "auto follows main.py defaults, which enable one-class paper figures for Ours-Final and full_ot."
        ),
    )
    parser.add_argument("--uot_evidence_num_episodes", type=int, default=1)
    parser.add_argument("--uot_evidence_queries_per_episode", type=int, default=1)
    parser.add_argument("--uot_evidence_correct_only", type=str, default="true", choices=["true", "false"])
    parser.add_argument(
        "--uot_evidence_visual_style",
        type=str,
        default="auto",
        choices=["auto", "paper", "legacy"],
        help="Forwarded to main.py. auto uses paper for Ours-Final and legacy for other models.",
    )
    parser.add_argument(
        "--m2_ablate_T",
        type=str,
        default="auto",
        choices=["auto", "true", "false"],
        help=(
            "M2-family only: forwards --hrot_ecot_m2_ablate_threshold_mass when explicitly true/false. "
            "auto leaves the model default in main.py (ours_final keeps T * mass on)."
        ),
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
    ablate_t_raw = str(getattr(args, "m2_ablate_T", "auto")).lower()
    cost_pm = str(getattr(args, "m2_cost_per_mass", "false")).lower() == "true"
    if cost_pm:
        extra += [
            "--hrot_ecot_m2_cost_per_mass_score",
            "true",
            "--hrot_ecot_m2_ablate_threshold_mass",
            "false",
        ]
    elif ablate_t_raw == "true":
        extra += [
            "--hrot_ecot_m2_ablate_threshold_mass",
            "true",
            "--hrot_ecot_m2_cost_per_mass_score",
            "false",
        ]
    elif ablate_t_raw == "false":
        extra += [
            "--hrot_ecot_m2_ablate_threshold_mass",
            "false",
            "--hrot_ecot_m2_cost_per_mass_score",
            "false",
        ]
    if "--result_artifacts" not in extra:
        extra += ["--result_artifacts", args.result_artifacts]
    if args.result_artifacts == "figures_only":
        if "--save_last_checkpoint" not in extra:
            extra += ["--save_last_checkpoint", "false"]
        if "--jecot_m2_val_save_hist_fig" not in extra:
            extra += ["--jecot_m2_val_save_hist_fig", "false"]
    export_uot_evidence = str(getattr(args, "export_uot_evidence_figure", "auto")).lower()
    if export_uot_evidence in {"true", "false"}:
        extra += ["--export_uot_evidence_figure", export_uot_evidence]
    if export_uot_evidence == "true":
        extra += [
            "--uot_evidence_num_episodes",
            str(args.uot_evidence_num_episodes),
            "--uot_evidence_queries_per_episode",
            str(args.uot_evidence_queries_per_episode),
            "--uot_evidence_correct_only",
            str(args.uot_evidence_correct_only),
        ]
    visual_style = str(getattr(args, "uot_evidence_visual_style", "auto")).lower()
    if visual_style != "auto":
        extra += ["--uot_evidence_visual_style", visual_style]
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
    OURS_FINAL_MODEL_NAME,
    OURS_FINAL_PARTIAL_OT_MODEL_NAME,
    "mm_spot_fsl",
    "pare_fsl",
    "m2_uot",
    "m2_full_ot",
    "jecot_m2",
    "jecot_m2_uot",
    "jecot_m2_full_ot",
    "crj_fsl",
    "fgwuot_fsl",
    CFUGET_MODEL_NAME,
    "jsc_wdro",
    "sg_pot",
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


def split_snr_db(split_name):
    name = str(split_name).lower()
    match = re.search(r"snr(?:_minus)?(\d+)db", name)
    if not match:
        return None
    value = int(match.group(1))
    return -value if "snr_minus" in name else value


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


def select_noise_snr_splits(noise_test_root, requested_splits="auto", snr_values=(15, 10, 5)):
    available = discover_noise_test_splits(noise_test_root, requested_splits)
    by_snr = {}
    for split_name in available:
        snr = split_snr_db(split_name)
        if snr is None:
            continue
        by_snr.setdefault(snr, []).append(split_name)

    selected = []
    missing = []
    for snr in snr_values:
        candidates = sorted(by_snr.get(int(snr), []), key=snr_sort_key)
        if not candidates:
            missing.append(int(snr))
            continue
        selected.append(candidates[0])

    if missing:
        raise ValueError(
            "Required noise SNR split(s) not found for "
            f"{', '.join(f'{snr}db' for snr in missing)}. "
            f"Available splits: {', '.join(available) if available else '(none)'}"
        )
    return selected


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


def build_expected_result_path(
    dataset_name,
    model,
    samples,
    shot,
    experiment_tag=None,
    test_protocol="clean",
    test_split_name=None,
):
    samples_str = f"{samples}samples" if samples is not None else "allsamples"
    tag_suffix = f"_{experiment_tag}" if experiment_tag else ""
    protocol_suffix = result_protocol_suffix(test_protocol, test_split_name)
    return os.path.join(
        "results",
        f"results_{dataset_name}_{model}_{samples_str}_{shot}shot{tag_suffix}{protocol_suffix}.txt",
    )


def expected_result_paths(
    dataset_name,
    model,
    samples,
    shot,
    experiment_tag=None,
    test_protocol="clean",
    noise_test_root=None,
    noise_test_splits="auto",
):
    if str(test_protocol).lower() == "noise":
        split_names = discover_noise_test_splits(noise_test_root, noise_test_splits)
        return [
            build_expected_result_path(
                dataset_name,
                model,
                samples,
                shot,
                experiment_tag=experiment_tag,
                test_protocol=test_protocol,
                test_split_name=split_name,
            )
            for split_name in split_names
        ]
    return [
        build_expected_result_path(
            dataset_name,
            model,
            samples,
            shot,
            experiment_tag=experiment_tag,
            test_protocol=test_protocol,
        )
    ]


def merge_extra_test_protocols(*protocol_lists):
    merged = []
    for protocols in protocol_lists:
        for protocol in protocols or []:
            normalized = str(protocol).strip().lower()
            if normalized and normalized not in merged:
                merged.append(normalized)
    return merged


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
            "tag": "ours_no_egsm",
            "label": "Ours without EGSM marginals",
            "extra_args": ["--ours_ablation", "no_egsm"],
        },
        {
            "tag": "ours_full",
            "label": "Full Ours: local descriptors + UOT + EGSM",
            "extra_args": [],
        },
        {
            "tag": "ours_full_ot",
            "label": "Balanced full OT instead of UOT",
            "extra_args": ["--ours_ablation", "full_ot"],
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


def _ours_final_base_args(
    rho="0.8",
    ablation="full",
    transport_mode="unbalanced",
    ablate_threshold_mass="false",
    fixed_mass=None,
):
    """Explicit Ours-Final defaults; variant args are appended after passthrough args."""
    fixed_mass = str(rho) if fixed_mass is None else str(fixed_mass)
    return [
        "--ours_ablation",
        str(ablation),
        "--hrot_ecot_enable_egsm",
        "false",
        "--hrot_ecot_m2_ablate_threshold_mass",
        str(ablate_threshold_mass),
        "--hrot_ecot_m2_cost_per_mass_score",
        "false",
        "--hrot_ecot_rho_bank",
        str(rho),
        "--hrot_ecot_base_rho",
        str(rho),
        "--hrot_fixed_mass",
        fixed_mass,
        "--hrot_ecot_transport_mode",
        str(transport_mode),
    ]


def _ours_final_partial_ot_args():
    """Ours-Final-compatible Partial OT ablation: fast backend with mass-normalized cost."""
    return [
        "--ours_ablation",
        "full",
        "--hrot_ecot_enable_egsm",
        "false",
        "--hrot_ecot_m2_ablate_threshold_mass",
        "false",
        "--hrot_ecot_m2_cost_per_mass_score",
        "true",
        "--hrot_ecot_rho_bank",
        "0.8",
        "--hrot_ecot_base_rho",
        "0.8",
        "--hrot_fixed_mass",
        "0.8",
        "--hrot_ecot_transport_mode",
        "unbalanced",
        "--hrot_ot_backend",
        "partial",
    ]


def build_ours_final_ablation_variants():
    base = _ours_final_base_args()
    return [
        {
            "tag": "ours_final_full",
            "checkpoint_tag": "ablation_full",
            "label": "Ours-Final: local descriptors + UOT rho=0.8 + threshold-mass score, EGSM off",
            "extra_args": base,
        },
        {
            "tag": "ours_final_ccem_uot",
            "checkpoint_tag": "ablation_ccem_uot",
            "label": (
                "Ours-Final CCEM-UOT: class-contrastive evidence marginals + UOT rho=0.8 "
                "+ explicit noise sink"
            ),
            "extra_args": base
            + [
                "--hrot_ecot_enable_ccem_marginal",
                "true",
                "--hrot_ecot_enable_noise_sink",
                "true",
                "--hrot_ecot_noise_sink_cost_init",
                "0.7",
            ],
            "mode1_noise": True,
        },
        {
            "tag": "ours_final_partial_ot",
            "checkpoint_tag": "ablation_partial_ot",
            "label": "Ours-Final ablation: fast Partial OT + cost-per-mass score",
            "model": OURS_FINAL_PARTIAL_OT_MODEL_NAME,
            "extra_args": _ours_final_partial_ot_args(),
            "mode1_noise": True,
        },
        {
            "tag": "ours_final_full_ot",
            "checkpoint_tag": "ablation_full_ot",
            "label": "Ours-Final ablation: balanced full OT replaces UOT",
            "extra_args": _ours_final_base_args(
                "1.0",
                ablation="full_ot",
                transport_mode="balanced",
                fixed_mass="0.8",
            ),
            "mode1_noise": True,
        },
        {
            "tag": "ours_final_gap",
            "checkpoint_tag": "ablation_gap",
            "label": "Ours-Final ablation: GAP feature-map descriptors replace local tokens",
            "extra_args": _ours_final_base_args(ablation="gap"),
            "mode1_noise": True,
        },
        {
            "tag": "ours_final_mass_off",
            "checkpoint_tag": "ablation_mass_off",
            "label": "Ours-Final ablation: cost-only score removes threshold-mass reward",
            "extra_args": _ours_final_base_args(ablate_threshold_mass="true"),
            "mode1_noise": True,
        },
    ]


def build_ours_final_rho_grid_variants(rho_values=None):
    rho_values = tuple(rho_values or ("0.6", "0.7", "0.8", "0.9"))
    return [
        {
            "tag": f"ours_final_rho_{str(rho).replace('.', 'p')}",
            "checkpoint_tag": f"rho_grid_rho_{str(rho).replace('.', 'p')}",
            "label": f"Ours-Final rho={rho} single-budget UOT",
            "extra_args": _ours_final_base_args(str(rho)),
        }
        for rho in rho_values
    ]


def build_ours_final_tau_shot_off_variants():
    return [
        {
            "tag": "ours_final_tau_shot_off",
            "checkpoint_tag": "ablation_tau_shot_off",
            "label": "Ours-Final ablation: ECOT tau-shot pooling disabled",
            "extra_args": _ours_final_base_args() + ["--hrot_ecot_enable_tau_shot", "false"],
        }
    ]


def build_ours_final_pooling_variants():
    return [
        {
            "tag": "ours_final_class_pooled",
            "checkpoint_tag": "ablation_class_pooled",
            "label": "Ours-Final ablation: class-pooled support tokens before OT",
            "extra_args": _ours_final_base_args()
            + [
                "--hrot_pre_transport_shot_pool",
                "true",
                "--hrot_pre_transport_shot_pool_mode",
                "concat",
            ],
        },
        {
            "tag": "ours_final_fixed_shot_pooling",
            "checkpoint_tag": "ablation_fixed_shot_pooling",
            "label": "Ours-Final ablation: fixed log-mean-exp shot pooling disables ECOT tau-shot",
            "extra_args": _ours_final_base_args() + ["--hrot_ecot_enable_tau_shot", "false"],
        },
    ]


def build_ours_final_mass_on_variants():
    base = _ours_final_base_args()
    variants = []
    for beta, tag_value in (("0.5", "0p5"), ("0.75", "0p75"), ("0.25", "0p25")):
        variants.append(
            {
                "tag": f"ours_final_mass_scaled_b{tag_value}",
                "checkpoint_tag": f"mass_scaled_b{tag_value}",
                "label": (
                    "Ours-Final mass-on: keep 1-shot mass reward, scale multi-shot "
                    f"T*M reward by beta={beta}"
                ),
                "extra_args": base
                + [
                    "--hrot_ecot_m2_mass_reward_shot_scaling",
                    "multi_shot_beta",
                    "--hrot_ecot_m2_mass_reward_beta",
                    beta,
                ],
            }
        )
    for alpha, tag_value in (("1.0", "1"), ("0.75", "0p75"), ("0.5", "0p5")):
        variants.append(
            {
                "tag": f"ours_final_mass_consensus_a{tag_value}",
                "checkpoint_tag": f"mass_consensus_a{tag_value}",
                "label": (
                    "Ours-Final mass-on: shot-consensus threshold-mass score "
                    f"with alpha={alpha}"
                ),
                "extra_args": base
                + [
                    "--hrot_ecot_m2_mass_score_mode",
                    "shot_consensus",
                    "--hrot_ecot_m2_consensus_mass_alpha",
                    alpha,
                ],
            }
        )
    return variants


def build_ours_final_pst_dm_variants():
    """Per-Shot Threshold (PST) and Discriminative Marginals (DM) ablation variants."""
    base = _ours_final_base_args()
    return [
        {
            "tag": "ours_final_pst",
            "checkpoint_tag": "pst_only",
            "label": "Ours-Final + per-shot adaptive threshold",
            "extra_args": base + ["--hrot_ecot_m2_per_shot_threshold", "true"],
            "mode1_noise": True,
        },
        {
            "tag": "ours_final_dm",
            "checkpoint_tag": "dm_only",
            "label": "Ours-Final + discriminative learned-attention marginals",
            "extra_args": base + [
                "--ours_final_dmuot_ablation",
                "exp6_learned_attn_tau_1_shot_sqrt",
            ],
            "mode1_noise": True,
        },
        {
            "tag": "ours_final_pst_dm",
            "checkpoint_tag": "pst_dm",
            "label": "Ours-Final + per-shot threshold + discriminative marginals",
            "extra_args": base + [
                "--hrot_ecot_m2_per_shot_threshold",
                "true",
                "--ours_final_dmuot_ablation",
                "exp6_learned_attn_tau_1_shot_sqrt",
            ],
            "mode1_noise": True,
        },
    ]


def build_ours_final_dmuot_variants():
    base = _ours_final_base_args()
    variants = [
        {
            "tag": "ours_final_dmuot_exp0_g_episode_mean_dist",
            "checkpoint_tag": "dmuot_exp0_g_episode_mean_dist",
            "label": "DM-UOT exp0: log token g via episode mean distance; no transport change",
            "extra_args": base + ["--ours_final_dmuot_ablation", "exp0_g_episode_mean_dist"],
        },
        {
            "tag": "ours_final_dmuot_exp0_g_token_norm_pre_l2",
            "checkpoint_tag": "dmuot_exp0_g_token_norm_pre_l2",
            "label": "DM-UOT exp0: log token g via projector pre-L2 norm; no transport change",
            "extra_args": base + ["--ours_final_dmuot_ablation", "exp0_g_token_norm_pre_l2"],
        },
    ]
    for value, tag_value in (("2.0", "2"), ("1.0", "1")):
        for strength_tag, strength_label in (
            ("shot_sqrt", "1/sqrt(shot)"),
            ("shot_inverse", "1/shot"),
        ):
            variants.append(
                {
                    "tag": f"ours_final_dmuot_exp5_tau_marg_{tag_value}_{strength_tag}",
                    "checkpoint_tag": f"dmuot_exp5_tau_marg_{tag_value}_{strength_tag}",
                    "label": (
                        "DM-UOT exp5: shot-neutralized discriminative marginals "
                        f"tau_marg={value}, strength={strength_label}"
                    ),
                    "extra_args": base
                    + [
                        "--ours_final_dmuot_ablation",
                        f"exp5_tau_marg_{tag_value}_{strength_tag}",
                    ],
                }
            )
    for value, tag_value in (
        ("0.5", "0p5"),
        ("1.0", "1"),
        ("2.0", "2"),
        ("5.0", "5"),
    ):
        variants.append(
            {
                "tag": f"ours_final_dmuot_exp1_lambda_cost_{tag_value}",
                "checkpoint_tag": f"dmuot_exp1_lambda_cost_{tag_value}",
                "label": f"DM-UOT exp1: discriminative cost modulation lambda_cost={value}",
                "extra_args": base
                + [
                    "--ours_final_dmuot_ablation",
                    f"exp1_lambda_cost_{tag_value}",
                ],
            }
        )
    for value, tag_value in (
        ("0.25", "0p25"),
        ("0.5", "0p5"),
        ("1.0", "1"),
        ("2.0", "2"),
    ):
        variants.append(
            {
                "tag": f"ours_final_dmuot_exp2_tau_marg_{tag_value}",
                "checkpoint_tag": f"dmuot_exp2_tau_marg_{tag_value}",
                "label": f"DM-UOT exp2: discriminative token marginals tau_marg={value}",
                "extra_args": base
                + [
                    "--ours_final_dmuot_ablation",
                    f"exp2_tau_marg_{tag_value}",
                ],
            }
        )
    return variants


def build_ours_final_evidence_marginal_variants():
    """Clean ablation sweep for cost-derived evidence marginals (paper Table).

    1. Ours-Final uniform marginals          (baseline)
    2. + query evidence marginals only        (query_only)
    3. + support evidence marginals only      (support_only)
    4. + query + support evidence marginals   (both)
    5. + class-discriminative rival-aware     (rival_aware)
    """
    base = _ours_final_base_args()
    variants = [
        {
            "tag": "ours_final_evidence_query_only",
            "checkpoint_tag": "evidence_query_only",
            "label": "Evidence marginals: query-side only",
            "extra_args": base + ["--ours_final_evidence_ablation", "query_only"],
        },
        {
            "tag": "ours_final_evidence_support_only",
            "checkpoint_tag": "evidence_support_only",
            "label": "Evidence marginals: support-side only",
            "extra_args": base + ["--ours_final_evidence_ablation", "support_only"],
        },
        {
            "tag": "ours_final_evidence_both",
            "checkpoint_tag": "evidence_both",
            "label": "Evidence marginals: query + support (both sides)",
            "extra_args": base + ["--ours_final_evidence_ablation", "both"],
        },
        {
            "tag": "ours_final_evidence_rival_aware",
            "checkpoint_tag": "evidence_rival_aware",
            "label": "Evidence marginals: class-discriminative rival-aware",
            "extra_args": base + ["--ours_final_evidence_ablation", "rival_aware"],
        },
    ]
    for tau in ("0.05", "0.1", "0.2", "0.5"):
        tag_tau = tau.replace(".", "p")
        variants.append(
            {
                "tag": f"ours_final_evidence_both_tau_{tag_tau}",
                "checkpoint_tag": f"evidence_both_tau_{tag_tau}",
                "label": f"Evidence marginals: both, tau_evidence={tau}",
                "extra_args": base + [
                    "--ours_final_evidence_ablation", f"both_tau_{tag_tau}",
                ],
            }
        )
    for tau_m in ("0.5", "1.0", "2.0"):
        tag_tau_m = tau_m.replace(".", "p")
        variants.append(
            {
                "tag": f"ours_final_evidence_both_tau_m_{tag_tau_m}",
                "checkpoint_tag": f"evidence_both_tau_m_{tag_tau_m}",
                "label": f"Evidence marginals: both, tau_marginal={tau_m}",
                "extra_args": base + [
                    "--ours_final_evidence_ablation", f"both_tau_m_{tag_tau_m}",
                ],
            }
        )
    for rm in ("0.0", "0.25", "0.5", "1.0"):
        tag_rm = rm.replace(".", "p")
        variants.append(
            {
                "tag": f"ours_final_evidence_rival_margin_{tag_rm}",
                "checkpoint_tag": f"evidence_rival_margin_{tag_rm}",
                "label": f"Evidence marginals: rival-aware, rival_margin={rm}",
                "extra_args": base + [
                    "--ours_final_evidence_ablation", f"rival_margin_{tag_rm}",
                ],
            }
        )
    return variants


def build_ours_final_pulse_region_variants():
    """Pulse-region comparison suite with an explicit Ours-Final baseline."""
    base = _ours_final_base_args()
    return [
        {
            "tag": "ours_final_pulse_baseline",
            "checkpoint_tag": "pulse_baseline",
            "label": "Ours-Final baseline: UOT rho=0.8 + threshold-mass score, no pulse guidance",
            "extra_args": base,
        },
        {
            "tag": "ours_final_pulse_cost",
            "checkpoint_tag": "pulse_cost",
            "label": "Pulse-region cost/marginal guidance only; evidence score disabled",
            "extra_args": base
            + [
                "--enable_pulse_region_uot",
                "--pulse_evidence_score",
                "false",
                "--pulse_discriminative_evidence",
                "false",
            ],
        },
        {
            "tag": "ours_final_pulse_score_only",
            "checkpoint_tag": "pulse_score_only",
            "label": "Pulse-region score-only evidence: keep UOT cost/marginals unchanged",
            "extra_args": base
            + [
                "--enable_pulse_region_uot",
                "--pulse_region_cost_weight",
                "0.0",
                "--pulse_saliency_mass_mix",
                "0.0",
                "--pulse_saliency_cost_discount",
                "0.0",
                "--pulse_evidence_score",
                "true",
                "--pulse_discriminative_evidence",
                "true",
            ],
        },
        {
            "tag": "ours_final_pulse_full_evidence",
            "checkpoint_tag": "pulse_full_evidence",
            "label": "Pulse-region cost/marginal guidance plus rival-aware pulse evidence score",
            "extra_args": base
            + [
                "--enable_pulse_region_uot",
                "--pulse_evidence_score",
                "true",
                "--pulse_discriminative_evidence",
                "true",
            ],
        },
    ]


def _cfuget_base_args(
    ablation_mode="full",
    fgw_iters="1",
    sinkhorn_iters="50",
    alpha_init="1e-6",
    spatial_structure_weight="0.0",
    rival_temperature="0.07",
    rival_margin="0.02",
    shot_aggregation="j_logmeanexp",
):
    """Explicit RC-UOT defaults; variant args are appended after passthrough args."""
    return [
        "--cfuget_token_dim",
        "128",
        "--cfuget_rho",
        "0.8",
        "--cfuget_tau",
        "0.5",
        "--cfuget_eps_sinkhorn",
        "0.08",
        "--cfuget_fgw_iters",
        str(fgw_iters),
        "--cfuget_sinkhorn_iters",
        str(sinkhorn_iters),
        "--cfuget_sinkhorn_tol",
        "1e-5",
        "--cfuget_alpha_init",
        str(alpha_init),
        "--cfuget_score_scale_init",
        "16.0",
        "--cfuget_threshold_init",
        "0.5",
        "--cfuget_rival_temperature",
        str(rival_temperature),
        "--cfuget_rival_margin",
        str(rival_margin),
        "--cfuget_coherent_temperature",
        "0.10",
        "--cfuget_spatial_structure_weight",
        str(spatial_structure_weight),
        "--cfuget_mass_weight",
        "1.0",
        "--cfuget_cost_weight",
        "1.0",
        "--cfuget_normalize_tokens",
        "true",
        "--cfuget_structure_detach",
        "false",
        "--cfuget_rival_detach",
        "true",
        "--cfuget_ablation_mode",
        str(ablation_mode),
        "--cfuget_shot_aggregation",
        str(shot_aggregation),
        "--cfuget_eps",
        "1e-8",
    ]


def build_cfuget_contrib_variants():
    base = _cfuget_base_args()
    return [
        {
            "tag": "cfuget_full",
            "checkpoint_tag": "ablation_full",
            "label": "RC-UOT: UOT + rival coherent evidence + threshold-mass score",
            "extra_args": base,
        },
        {
            "tag": "cfuget_no_rival",
            "checkpoint_tag": "ablation_no_rival",
            "label": "RC-UOT ablation: evidence gate without rival-class subtraction",
            "extra_args": _cfuget_base_args(ablation_mode="no_rival"),
        },
        {
            "tag": "cfuget_no_coherent_score",
            "checkpoint_tag": "ablation_no_coherent_score",
            "label": "RC-UOT ablation: score raw UOT mass-cost instead of coherent gated evidence",
            "extra_args": _cfuget_base_args(ablation_mode="no_coherent_score"),
        },
        {
            "tag": "cfuget_cost_only",
            "checkpoint_tag": "ablation_cost_only",
            "label": "RC-UOT ablation: remove threshold-mass reward from coherent score",
            "extra_args": _cfuget_base_args(ablation_mode="cost_only"),
        },
    ]


def build_cfuget_runtime_variants():
    return [
        {
            "tag": "cfuget_sink30",
            "checkpoint_tag": "runtime_sink30",
            "label": "RC-UOT runtime: Sinkhorn iters=30",
            "extra_args": _cfuget_base_args(sinkhorn_iters="30"),
        },
        {
            "tag": "cfuget_sink70",
            "checkpoint_tag": "runtime_sink70",
            "label": "RC-UOT runtime: Sinkhorn iters=70",
            "extra_args": _cfuget_base_args(sinkhorn_iters="70"),
        },
        {
            "tag": "cfuget_tau005",
            "checkpoint_tag": "runtime_tau005",
            "label": "RC-UOT evidence check: sharper rival evidence temperature=0.05",
            "extra_args": _cfuget_base_args(rival_temperature="0.05"),
        },
        {
            "tag": "cfuget_tau010",
            "checkpoint_tag": "runtime_tau010",
            "label": "RC-UOT evidence check: smoother rival evidence temperature=0.10",
            "extra_args": _cfuget_base_args(rival_temperature="0.10"),
        },
        {
            "tag": "cfuget_margin0",
            "checkpoint_tag": "runtime_margin0",
            "label": "RC-UOT evidence check: no rival margin",
            "extra_args": _cfuget_base_args(rival_margin="0.0"),
        },
        {
            "tag": "cfuget_margin005",
            "checkpoint_tag": "runtime_margin005",
            "label": "RC-UOT evidence check: stronger rival margin=0.05",
            "extra_args": _cfuget_base_args(rival_margin="0.05"),
        },
    ]


def build_cfuget_ablation_variants(suite_name):
    contrib = build_cfuget_contrib_variants()
    runtime = build_cfuget_runtime_variants()
    if suite_name == "contrib":
        return contrib
    if suite_name == "runtime":
        return runtime
    if suite_name == "complete":
        return contrib + runtime
    raise ValueError(f"Unknown C-FUGET ablation suite: {suite_name}")


MM_SPOT_MODEL_NAME = "mm_spot_fsl"


def _mm_spot_base_args(
    mass_bank="0.3,0.4,0.5",
    epsilon="0.05",
    iterations="120",
    aggregation="softmax",
    temperature="1.0",
    partial_backend="pot",
    use_controller="false",
    proto_weight="0.0",
):
    """Explicit MM-SPOT-FSL defaults; variant args are appended after passthrough args."""
    return [
        "--spot_mass_bank",
        str(mass_bank),
        "--spot_sinkhorn_epsilon",
        str(epsilon),
        "--spot_sinkhorn_iterations",
        str(iterations),
        "--spot_sinkhorn_tolerance",
        "1e-6",
        "--spot_score_scale",
        "16.0",
        "--spot_mass_aggregation",
        str(aggregation),
        "--spot_mass_temperature",
        str(temperature),
        "--spot_partial_backend",
        str(partial_backend),
        "--spot_use_controller",
        str(use_controller),
        "--spot_proto_weight",
        str(proto_weight),
    ]


def build_mm_spot_core_variants():
    return [
        {
            "tag": "mm_spot_k3_softmax",
            "checkpoint_tag": "k3_softmax",
            "label": "MM-SPOT-FSL: K=3 mass bank [0.3,0.4,0.5], softmax score aggregation",
            "extra_args": _mm_spot_base_args(),
        },
        {
            "tag": "mm_spot_single_s04",
            "checkpoint_tag": "single_s04",
            "label": "MM-SPOT-FSL control: single fixed mass s=0.4",
            "extra_args": _mm_spot_base_args(mass_bank="0.4"),
        },
        {
            "tag": "mm_spot_k3_mean",
            "checkpoint_tag": "k3_mean",
            "label": "MM-SPOT-FSL ablation: K=3 bank with mean score aggregation",
            "extra_args": _mm_spot_base_args(aggregation="mean"),
        },
    ]


def build_mm_spot_runtime_variants():
    return [
        {
            "tag": "mm_spot_k3_iter60",
            "checkpoint_tag": "k3_iter60",
            "label": "MM-SPOT-FSL runtime: K=3 mass bank, 60 POT iterations",
            "extra_args": _mm_spot_base_args(iterations="60"),
        },
        {
            "tag": "mm_spot_k3_iter120",
            "checkpoint_tag": "k3_iter120",
            "label": "MM-SPOT-FSL runtime: K=3 mass bank, 120 POT iterations",
            "extra_args": _mm_spot_base_args(iterations="120"),
        },
        {
            "tag": "mm_spot_k5_iter60",
            "checkpoint_tag": "k5_iter60",
            "label": "MM-SPOT-FSL runtime: K=5 mass bank [0.2,0.3,0.4,0.5,0.6], 60 POT iterations",
            "extra_args": _mm_spot_base_args(mass_bank="0.2,0.3,0.4,0.5,0.6", iterations="60"),
        },
        {
            "tag": "mm_spot_single_s04_iter120",
            "checkpoint_tag": "single_s04_iter120",
            "label": "MM-SPOT-FSL runtime control: single s=0.4, 120 POT iterations",
            "extra_args": _mm_spot_base_args(mass_bank="0.4", iterations="120"),
        },
    ]


def build_mm_spot_sweep_variants():
    return [
        {
            "tag": "mm_spot_single_s03",
            "checkpoint_tag": "single_s03",
            "label": "MM-SPOT-FSL mass sweep: single fixed mass s=0.3",
            "extra_args": _mm_spot_base_args(mass_bank="0.3"),
        },
        {
            "tag": "mm_spot_single_s04",
            "checkpoint_tag": "single_s04",
            "label": "MM-SPOT-FSL mass sweep: single fixed mass s=0.4",
            "extra_args": _mm_spot_base_args(mass_bank="0.4"),
        },
        {
            "tag": "mm_spot_single_s05",
            "checkpoint_tag": "single_s05",
            "label": "MM-SPOT-FSL mass sweep: single fixed mass s=0.5",
            "extra_args": _mm_spot_base_args(mass_bank="0.5"),
        },
        {
            "tag": "mm_spot_k3_eps003",
            "checkpoint_tag": "k3_eps003",
            "label": "MM-SPOT-FSL epsilon sweep: K=3 bank, epsilon=0.03",
            "extra_args": _mm_spot_base_args(epsilon="0.03"),
        },
        {
            "tag": "mm_spot_k3_eps005",
            "checkpoint_tag": "k3_eps005",
            "label": "MM-SPOT-FSL epsilon sweep: K=3 bank, epsilon=0.05",
            "extra_args": _mm_spot_base_args(epsilon="0.05"),
        },
        {
            "tag": "mm_spot_k3_eps010",
            "checkpoint_tag": "k3_eps010",
            "label": "MM-SPOT-FSL epsilon sweep: K=3 bank, epsilon=0.10",
            "extra_args": _mm_spot_base_args(epsilon="0.10"),
        },
    ]


def build_mm_spot_controller_variants():
    return [
        {
            "tag": "mm_spot_controller_k3",
            "checkpoint_tag": "controller_k3",
            "label": "MM-SPOT-FSL controller ablation: learned selector over K=3 mass bank",
            "extra_args": _mm_spot_base_args(use_controller="true"),
        }
    ]


def unique_variants_by_tag(variants):
    unique = []
    seen = set()
    for variant in variants:
        if variant["tag"] in seen:
            continue
        seen.add(variant["tag"])
        unique.append(variant)
    return unique


def build_mm_spot_ablation_variants(suite_name):
    core = build_mm_spot_core_variants()
    runtime = build_mm_spot_runtime_variants()
    sweep = build_mm_spot_sweep_variants()
    controller = build_mm_spot_controller_variants()
    if suite_name == "core":
        return core
    if suite_name == "runtime":
        return runtime
    if suite_name == "sweep":
        return sweep
    if suite_name == "controller":
        return controller
    if suite_name == "complete":
        return unique_variants_by_tag(core + runtime + sweep)
    raise ValueError(f"Unsupported MM-SPOT-FSL ablation suite: {suite_name}")


PARE_FSL_MODEL_NAME = "pare_fsl"


def _pare_base_args(
    epsilon="0.04",
    iterations="80",
    marginal_mode="egsm",
    learned_alpha="true",
    fixed_alpha=None,
    alpha_min="0.30",
    alpha_max="0.70",
    alpha_reg_lambda="0.01",
    shot_pooling="discrepancy",
    shot_temperature="1.0",
):
    """Explicit PARE-FSL defaults; variant args are appended after passthrough args."""
    args = [
        "--pare_sinkhorn_epsilon",
        str(epsilon),
        "--pare_sinkhorn_iterations",
        str(iterations),
        "--pare_sinkhorn_tolerance",
        "1e-6",
        "--pare_score_scale",
        "16.0",
        "--pare_marginal_mode",
        str(marginal_mode),
        "--pare_enable_learned_alpha",
        str(learned_alpha),
        "--pare_alpha_min",
        str(alpha_min),
        "--pare_alpha_max",
        str(alpha_max),
        "--pare_alpha_reg_lambda",
        str(alpha_reg_lambda),
        "--pare_shot_pooling",
        str(shot_pooling),
        "--pare_shot_temperature",
        str(shot_temperature),
    ]
    if fixed_alpha is not None:
        args.extend(["--pare_fixed_alpha", str(fixed_alpha)])
    return args


def build_pare_core_variants():
    return [
        {
            "tag": "pare_default",
            "checkpoint_tag": "default",
            "label": "PARE-FSL: EGSM marginals (sum=1), learned alpha, native partial OT per shot",
            "extra_args": _pare_base_args(),
        },
        {
            "tag": "pare_uniform",
            "checkpoint_tag": "uniform",
            "label": "PARE-FSL ablation: uniform marginals with learned alpha",
            "extra_args": _pare_base_args(marginal_mode="uniform"),
        },
        {
            "tag": "pare_fixed_alpha",
            "checkpoint_tag": "fixed_alpha",
            "label": "PARE-FSL control: fixed partial mass alpha=0.5 (no learned head)",
            "extra_args": _pare_base_args(learned_alpha="false", fixed_alpha="0.5"),
        },
    ]


def build_pare_runtime_variants():
    return [
        {
            "tag": "pare_iter60",
            "checkpoint_tag": "iter60",
            "label": "PARE-FSL runtime: 60 POT iterations",
            "extra_args": _pare_base_args(iterations="60"),
        },
        {
            "tag": "pare_iter80",
            "checkpoint_tag": "iter80",
            "label": "PARE-FSL runtime: 80 POT iterations (default)",
            "extra_args": _pare_base_args(iterations="80"),
        },
        {
            "tag": "pare_iter120",
            "checkpoint_tag": "iter120",
            "label": "PARE-FSL runtime: 120 POT iterations",
            "extra_args": _pare_base_args(iterations="120"),
        },
        {
            "tag": "pare_eps002",
            "checkpoint_tag": "eps002",
            "label": "PARE-FSL epsilon sweep: epsilon=0.02",
            "extra_args": _pare_base_args(epsilon="0.02"),
        },
        {
            "tag": "pare_eps006",
            "checkpoint_tag": "eps006",
            "label": "PARE-FSL epsilon sweep: epsilon=0.06",
            "extra_args": _pare_base_args(epsilon="0.06"),
        },
        {
            "tag": "pare_shot_mean",
            "checkpoint_tag": "shot_mean",
            "label": "PARE-FSL pooling ablation: mean shot pooling",
            "extra_args": _pare_base_args(shot_pooling="mean"),
        },
    ]


def build_pare_alpha_variants():
    return [
        {
            "tag": "pare_alpha_wide",
            "checkpoint_tag": "alpha_wide",
            "label": "PARE-FSL alpha ablation: wider range [0.25, 0.80]",
            "extra_args": _pare_base_args(alpha_min="0.25", alpha_max="0.80"),
        },
        {
            "tag": "pare_alpha_noreg",
            "checkpoint_tag": "alpha_noreg",
            "label": "PARE-FSL alpha ablation: disable alpha prior regularization",
            "extra_args": _pare_base_args(alpha_reg_lambda="0.0"),
        },
    ]


def build_pare_ablation_variants(suite_name):
    core = build_pare_core_variants()
    runtime = build_pare_runtime_variants()
    alpha = build_pare_alpha_variants()
    if suite_name == "core":
        return core
    if suite_name == "runtime":
        return runtime
    if suite_name == "alpha":
        return alpha
    if suite_name == "complete":
        return unique_variants_by_tag(core + runtime)
    raise ValueError(f"Unsupported PARE-FSL ablation suite: {suite_name}")


SGPOT_MODEL_NAME = "sg_pot"


def _sgpot_base_args(
    epsilon="0.05",
    iterations="100",
    s_min="0.3",
    s_max="0.8",
    entropy_reg="0.05",
    entropy_target="0.7",
    ablate_uniform_marginals="false",
    ablate_fixed_s=None,
    ablate_cost_only_score="false",
    ablate_no_entropy_reg="false",
):
    args = [
        "--pot_epsilon", str(epsilon),
        "--pot_iterations", str(iterations),
        "--pot_s_min", str(s_min),
        "--pot_s_max", str(s_max),
        "--pot_entropy_reg", str(entropy_reg),
        "--pot_entropy_target", str(entropy_target),
        "--pot_ablate_uniform_marginals", str(ablate_uniform_marginals),
        "--pot_ablate_cost_only_score", str(ablate_cost_only_score),
        "--pot_ablate_no_entropy_reg", str(ablate_no_entropy_reg),
    ]
    if ablate_fixed_s is not None:
        args.extend(["--pot_ablate_fixed_s", str(ablate_fixed_s)])
    return args


def build_sgpot_core_variants():
    return [
        {
            "tag": "sgpot_full",
            "checkpoint_tag": "full",
            "label": "SG-POT full: saliency marginals + adaptive s + dual evidence",
            "extra_args": _sgpot_base_args(),
        },
        {
            "tag": "sgpot_uniform_marginals",
            "checkpoint_tag": "uniform_marginals",
            "label": "SG-POT ablation: uniform marginals (no saliency gate)",
            "extra_args": _sgpot_base_args(ablate_uniform_marginals="true"),
        },
        {
            "tag": "sgpot_fixed_s05",
            "checkpoint_tag": "fixed_s05",
            "label": "SG-POT ablation: fixed mass fraction s=0.5 (no adaptive predictor)",
            "extra_args": _sgpot_base_args(ablate_fixed_s="0.5"),
        },
        {
            "tag": "sgpot_cost_only",
            "checkpoint_tag": "cost_only",
            "label": "SG-POT ablation: cost-only score (no residual Gini)",
            "extra_args": _sgpot_base_args(ablate_cost_only_score="true"),
        },
        {
            "tag": "sgpot_no_entropy_reg",
            "checkpoint_tag": "no_entropy_reg",
            "label": "SG-POT ablation: no saliency entropy regularization",
            "extra_args": _sgpot_base_args(ablate_no_entropy_reg="true"),
        },
    ]


def build_sgpot_hparam_variants():
    return [
        {
            "tag": "sgpot_smin02_smax09",
            "checkpoint_tag": "smin02_smax09",
            "label": "SG-POT hparam: wider mass range s_min=0.2, s_max=0.9",
            "extra_args": _sgpot_base_args(s_min="0.2", s_max="0.9"),
        },
        {
            "tag": "sgpot_smin04_smax07",
            "checkpoint_tag": "smin04_smax07",
            "label": "SG-POT hparam: narrower mass range s_min=0.4, s_max=0.7",
            "extra_args": _sgpot_base_args(s_min="0.4", s_max="0.7"),
        },
        {
            "tag": "sgpot_eps003",
            "checkpoint_tag": "eps003",
            "label": "SG-POT hparam: epsilon=0.03",
            "extra_args": _sgpot_base_args(epsilon="0.03"),
        },
        {
            "tag": "sgpot_eps010",
            "checkpoint_tag": "eps010",
            "label": "SG-POT hparam: epsilon=0.10",
            "extra_args": _sgpot_base_args(epsilon="0.10"),
        },
        {
            "tag": "sgpot_entreg01",
            "checkpoint_tag": "entreg01",
            "label": "SG-POT hparam: entropy reg weight=0.1",
            "extra_args": _sgpot_base_args(entropy_reg="0.1"),
        },
        {
            "tag": "sgpot_entreg002",
            "checkpoint_tag": "entreg002",
            "label": "SG-POT hparam: entropy reg weight=0.02",
            "extra_args": _sgpot_base_args(entropy_reg="0.02"),
        },
    ]


def build_sgpot_ablation_variants(suite_name):
    core = build_sgpot_core_variants()
    hparam = build_sgpot_hparam_variants()
    if suite_name == "core":
        return core
    if suite_name == "hparam":
        return hparam
    if suite_name == "complete":
        return unique_variants_by_tag(core + hparam)
    raise ValueError(f"Unsupported SG-POT ablation suite: {suite_name}")


def filter_sgpot_variants(variants, allowed_tags):
    if allowed_tags is None:
        return list(variants)
    return [variant for variant in variants if variant["tag"] in allowed_tags]


def restricted_ours_final_rho_grid_pairs(samples_list, shots):
    allowed_pairs = {(60, 5), (240, 1)}
    return [
        (samples, shot)
        for samples in samples_list
        for shot in shots
        if (samples, shot) in allowed_pairs
    ]


def build_best_checkpoint_path(dataset_name, model, samples, shot, checkpoint_tag=None):
    samples_suffix = f"{samples}samples" if samples is not None else "all"
    tag_suffix = f"_{sanitize_result_tag(checkpoint_tag)}" if checkpoint_tag else ""
    return os.path.join(
        "checkpoints",
        f"{dataset_name}_{model}_{samples_suffix}_{shot}shot{tag_suffix}_best.pth",
    )


def build_legacy_final_checkpoint_path(dataset_name, model, samples, shot, experiment_tag=None):
    samples_suffix = f"{samples}samples" if samples is not None else "all"
    tag_suffix = f"_{experiment_tag}" if experiment_tag else ""
    return os.path.join(
        "checkpoints",
        f"{dataset_name}_{model}_{samples_suffix}_{shot}shot{tag_suffix}_final.pth",
    )


def resolve_checkpoint_path(preferred_path, legacy_path=None):
    if os.path.exists(preferred_path):
        return preferred_path
    if legacy_path and os.path.exists(legacy_path):
        return legacy_path
    return preferred_path


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


def all_paths_exist(paths):
    return bool(paths) and all(os.path.exists(path) for path in paths)


def cli_bool_option(cmd, flag, default=False):
    if flag not in cmd:
        return default
    idx = len(cmd) - 1 - list(reversed(cmd)).index(flag)
    if idx + 1 >= len(cmd) or str(cmd[idx + 1]).startswith("--"):
        return True
    value = str(cmd[idx + 1]).strip().lower()
    if value == "auto":
        return default
    return value in {"1", "true", "yes", "y", "on"}


def uot_evidence_artifacts_exist(
    dataset_name,
    model,
    samples,
    shot,
    experiment_tag=None,
    test_protocol="clean",
    noise_test_root=None,
    noise_test_splits="auto",
):
    samples_str = f"{samples}samples" if samples is not None else "allsamples"
    tag_suffix = f"_{experiment_tag}" if experiment_tag else ""
    split_names = [None]
    if str(test_protocol).lower() == "noise":
        split_names = discover_noise_test_splits(noise_test_root, noise_test_splits)
    for split_name in split_names:
        protocol_suffix = result_protocol_suffix(test_protocol, split_name)
        pattern = (
            f"uot_evidence_{dataset_name}_{model}_{samples_str}_"
            f"{shot}shot{tag_suffix}{protocol_suffix}_ep*.png"
        )
        matches = list(Path("results").glob(pattern))
        main_matches = [
            path
            for path in matches
            if not path.stem.endswith("_transport_matrix") and not path.stem.endswith("_all_classes")
        ]
        if not main_matches:
            return False
        if model in {OURS_FINAL_MODEL_NAME, OURS_FINAL_PARTIAL_OT_MODEL_NAME}:
            matrix_pattern = (
                f"uot_evidence_{dataset_name}_{model}_{samples_str}_"
                f"{shot}shot{tag_suffix}{protocol_suffix}_ep*_transport_matrix.png"
            )
            if not any(Path("results").glob(matrix_pattern)):
                return False
    return True


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
    checkpoint_tag=None,
    experiment_label=None,
    extra_final_test_seeds=None,
    extra_test_protocols=None,
    extra_noise_test_splits=None,
    skip_existing=False,
    selection_seed=DEFAULT_SELECTION_BASE_SEED,
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
    if checkpoint_tag and checkpoint_tag != experiment_tag:
        print(f"Ckpt Tag    : {checkpoint_tag}")
    selection_offset = selection_episode_seed_offset(seed, selection_seed)
    print(f"Shot        : {shot}")
    print(f"Samples     : {samples if samples else 'All'}")
    print(f"Dataset     : {dataset_path} ({dataset_name})")
    print(f"Backbone    : {applied_backbone}")
    print(f"Train Seed  : {seed}")
    print(f"Val Seed    : {selection_seed}+{DEFAULT_SELECTION_EPISODE_SEED_OFFSET}+epoch")
    print(f"Test Proto  : {test_protocol}")
    checkpoint_path = build_best_checkpoint_path(
        dataset_name=dataset_name,
        model=model,
        samples=samples,
        shot=shot,
        checkpoint_tag=checkpoint_tag or experiment_tag,
    )
    legacy_checkpoint_path = build_legacy_final_checkpoint_path(
        dataset_name=dataset_name,
        model=model,
        samples=samples,
        shot=shot,
        experiment_tag=experiment_tag,
    )
    primary_result_paths = expected_result_paths(
        dataset_name=dataset_name,
        model=model,
        samples=samples,
        shot=shot,
        experiment_tag=experiment_tag,
        test_protocol=test_protocol,
        noise_test_root=noise_test_root,
        noise_test_splits=noise_test_splits,
    )
    print(f"Checkpoint  : {checkpoint_path}")
    if legacy_checkpoint_path != checkpoint_path and os.path.exists(legacy_checkpoint_path):
        print(f"Legacy Ckpt : {legacy_checkpoint_path}")
    if len(primary_result_paths) == 1:
        print(f"Result File : {primary_result_paths[0]}")
    else:
        print("Result Files:")
        for result_path in primary_result_paths:
            print(f"  - {result_path}")
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
    grad_clip = "1.0" if model == SGPOT_MODEL_NAME else "0.0"
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
                str(selection_offset),
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
    if checkpoint_tag:
        cmd.extend(["--checkpoint_tag", checkpoint_tag])

    if os.environ.get("PULSE_FEWSHOT_PRINT_TRAIN_CMD", "").lower() in {"1", "true", "yes"}:
        print(f"[pulse_fewshot child CLI] {shlex.join(cmd)}")

    primary_complete = all_paths_exist(primary_result_paths)
    needs_uot_evidence = cli_bool_option(
        cmd,
        "--export_uot_evidence_figure",
        default=model in {OURS_FINAL_MODEL_NAME, OURS_FINAL_PARTIAL_OT_MODEL_NAME},
    )
    primary_uot_evidence_complete = (
        not needs_uot_evidence
        or uot_evidence_artifacts_exist(
            dataset_name=dataset_name,
            model=model,
            samples=samples,
            shot=shot,
            experiment_tag=experiment_tag,
            test_protocol=test_protocol,
            noise_test_root=noise_test_root,
            noise_test_splits=noise_test_splits,
        )
    )
    if skip_existing and primary_complete:
        if needs_uot_evidence and not primary_uot_evidence_complete:
            checkpoint_for_test = resolve_checkpoint_path(checkpoint_path, legacy_checkpoint_path)
            if os.path.exists(checkpoint_for_test):
                print("Skip train  : primary result exists; running test-only to export missing UOT evidence figure.")
                test_cmd = list(cmd)
                test_cmd = set_cli_option(test_cmd, "--mode", "test")
                test_cmd = set_cli_option(test_cmd, "--weights", checkpoint_for_test)
                subprocess.run(test_cmd, check=True)
            else:
                print("Re-run      : result exists but checkpoint is missing; training is required for UOT evidence export.")
                subprocess.run(cmd, check=True)
        else:
            print("Skip        : primary result file(s) already exist; not training this experiment.")
    else:
        subprocess.run(cmd, check=True)

    if extra_final_test_seeds or extra_test_protocols:
        if use_external_smnet:
            print("Warning: extra final tests are not implemented for external smnet models; skipping.")
            return True

        checkpoint_for_test = resolve_checkpoint_path(checkpoint_path, legacy_checkpoint_path)
        if not os.path.exists(checkpoint_for_test):
            print(f"Error: trained checkpoint not found for extra tests: {checkpoint_path}")
            return False

        for extra_seed in extra_final_test_seeds:
            seed_tag = f"testseed{extra_seed}"
            if experiment_tag:
                seed_tag = f"{experiment_tag}_{seed_tag}"
            seed_result_paths = expected_result_paths(
                dataset_name=dataset_name,
                model=model,
                samples=samples,
                shot=shot,
                experiment_tag=seed_tag,
                test_protocol=test_protocol,
                noise_test_root=noise_test_root,
                noise_test_splits=noise_test_splits,
            )
            if skip_existing and all_paths_exist(seed_result_paths):
                print(f"\nSkip extra final test seed={extra_seed}: result file(s) already exist.")
                continue
            test_cmd = list(cmd)
            test_cmd = set_cli_option(test_cmd, "--mode", "test")
            test_cmd = set_cli_option(test_cmd, "--weights", checkpoint_for_test)
            test_cmd = set_cli_option(test_cmd, "--final_test_seed", str(extra_seed))
            test_cmd = set_cli_option(test_cmd, "--experiment_tag", seed_tag)

            print(f"\nExtra final test: seed={extra_seed}, checkpoint={checkpoint_for_test}")
            subprocess.run(test_cmd, check=True)

        for extra_protocol in extra_test_protocols:
            protocol_tag = f"{sanitize_result_tag(extra_protocol)}_testseed{final_test_seed}"
            if experiment_tag:
                protocol_tag = f"{experiment_tag}_{protocol_tag}"
            test_cmd = list(cmd)
            test_cmd = set_cli_option(test_cmd, "--mode", "test")
            test_cmd = set_cli_option(test_cmd, "--weights", checkpoint_for_test)
            test_cmd = set_cli_option(test_cmd, "--final_test_seed", str(final_test_seed))
            test_cmd = set_cli_option(test_cmd, "--test_protocol", extra_protocol)
            test_cmd = set_cli_option(test_cmd, "--experiment_tag", protocol_tag)
            if str(extra_protocol).lower() == "noise":
                test_cmd = set_cli_option(test_cmd, "--noise_test_root", noise_test_root)
                protocol_noise_splits = extra_noise_test_splits or noise_test_splits
                test_cmd = set_cli_option(test_cmd, "--noise_test_splits", protocol_noise_splits)
            else:
                test_cmd = remove_cli_option(test_cmd, "--noise_test_root")
                test_cmd = remove_cli_option(test_cmd, "--noise_test_splits")
            extra_result_paths = expected_result_paths(
                dataset_name=dataset_name,
                model=model,
                samples=samples,
                shot=shot,
                experiment_tag=protocol_tag,
                test_protocol=extra_protocol,
                noise_test_root=noise_test_root,
                noise_test_splits=extra_noise_test_splits or noise_test_splits,
            )
            if skip_existing and all_paths_exist(extra_result_paths):
                print(f"\nSkip extra {extra_protocol} final test: result file(s) already exist.")
                continue

            print(
                f"\nExtra {extra_protocol} final test: "
                f"seed={final_test_seed}, checkpoint={checkpoint_for_test}"
            )
            if len(extra_result_paths) == 1:
                print(f"Extra result file: {extra_result_paths[0]}")
            else:
                print("Extra result files:")
                for result_path in extra_result_paths:
                    print(f"  - {result_path}")
            subprocess.run(test_cmd, check=True)

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


def parse_training_seeds(seed_list_str):
    if seed_list_str is None or not str(seed_list_str).strip():
        return [42]
    parsed = []
    for token in str(seed_list_str).split(","):
        token = token.strip()
        if not token:
            continue
        seed = int(token)
        if seed not in parsed:
            parsed.append(seed)
    if not parsed:
        raise ValueError("--seed/--seeds must contain at least one integer seed.")
    return parsed


def parse_requested_models(models_str):
    requested = []
    for token in str(models_str or "").split(","):
        model = token.strip()
        if not model:
            continue
        alias_models = MODEL_GROUP_ALIASES.get(model.lower())
        if alias_models:
            for alias_model in alias_models:
                if alias_model not in requested:
                    requested.append(alias_model)
            continue
        if model not in requested:
            requested.append(model)
    return requested


def format_training_seed_line(training_seeds):
    if len(training_seeds) == 1:
        return f"Train Seed  : seed={training_seeds[0]}"
    return f"Train Seeds : seeds={','.join(str(seed) for seed in training_seeds)}"


def format_planned_total(experiments, training_seeds, suffix, dataset_count=1):
    return f"Total       : {len(experiments) * len(training_seeds) * dataset_count} {suffix}"


def join_result_tags(*parts):
    normalized = [str(part).strip() for part in parts if str(part or "").strip()]
    return "_".join(normalized) if normalized else None


def format_seeded_base_tag(base_tag, seed, multiple_training_seeds):
    base_tag = str(base_tag or "").strip()
    if "{seed}" in base_tag:
        return base_tag.format(seed=seed)
    if multiple_training_seeds:
        return join_result_tags(base_tag, f"seed{seed}")
    return base_tag or None


def build_effective_run_tags(experiment, base_experiment_tag, seed, multiple_training_seeds):
    base_tag = format_seeded_base_tag(base_experiment_tag, seed, multiple_training_seeds)
    variant_tag = experiment.get("experiment_tag")
    variant_checkpoint_tag = experiment.get("checkpoint_tag")
    experiment_tag = join_result_tags(base_tag, variant_tag)
    checkpoint_tag = join_result_tags(base_tag, variant_checkpoint_tag or variant_tag)
    return experiment_tag, checkpoint_tag


def selection_episode_seed_offset(train_seed, selection_seed):
    return DEFAULT_SELECTION_EPISODE_SEED_OFFSET + int(selection_seed) - int(train_seed)


def parse_rho_values(rho_values_str):
    if rho_values_str is None or not str(rho_values_str).strip():
        return None
    parsed = []
    for token in str(rho_values_str).split(","):
        token = token.strip()
        if not token:
            continue
        rho = float(token)
        if not 0.0 < rho <= 1.0:
            raise ValueError(f"Invalid rho value {rho!r} in --ours_final_rho_values (expected 0 < rho <= 1).")
        parsed.append(f"{rho:.6g}")
    if not parsed:
        return None
    return parsed


def parse_ours_final_variant_filter(variants_str):
    if variants_str is None:
        return None
    raw = str(variants_str).strip()
    if not raw or raw.lower() in {"all", "none", "*"}:
        return None

    aliases = {
        "full": "ours_final_full",
        "original": "ours_final_full",
        "orig": "ours_final_full",
        "uot": "ours_final_full",
        "ours_final": "ours_final_full",
        "ours_final_full": "ours_final_full",
        "ccem": "ours_final_ccem_uot",
        "ccem_uot": "ours_final_ccem_uot",
        "evidence": "ours_final_ccem_uot",
        "ours_final_ccem_uot": "ours_final_ccem_uot",
        "partial": "ours_final_partial_ot",
        "partial_ot": "ours_final_partial_ot",
        "fast_partial": "ours_final_partial_ot",
        "fast_partial_ot": "ours_final_partial_ot",
        "ours_final_partial_ot": "ours_final_partial_ot",
        "full_ot": "ours_final_full_ot",
        "ot": "ours_final_full_ot",
        "balanced": "ours_final_full_ot",
        "balanced_ot": "ours_final_full_ot",
        "ours_full_ot": "ours_final_full_ot",
        "ours_final_full_ot": "ours_final_full_ot",
        "gap": "ours_final_gap",
        "prototype": "ours_final_gap",
        "ours_final_gap": "ours_final_gap",
        "mass_off": "ours_final_mass_off",
        "cost_only": "ours_final_mass_off",
        "threshold_mass_off": "ours_final_mass_off",
        "ours_final_mass_off": "ours_final_mass_off",
        "class_pooled": "ours_final_class_pooled",
        "class_pool": "ours_final_class_pooled",
        "classpooled": "ours_final_class_pooled",
        "pooled": "ours_final_class_pooled",
        "pre_transport_pool": "ours_final_class_pooled",
        "ours_final_class_pooled": "ours_final_class_pooled",
        "fixed_shot_pooling": "ours_final_fixed_shot_pooling",
        "fixed_shot_pool": "ours_final_fixed_shot_pooling",
        "fixed_pooling": "ours_final_fixed_shot_pooling",
        "fixed_shot": "ours_final_fixed_shot_pooling",
        "ours_final_fixed_shot_pooling": "ours_final_fixed_shot_pooling",
        "tau_shot_off": "ours_final_tau_shot_off",
        "ours_final_tau_shot_off": "ours_final_tau_shot_off",
        "pst": "ours_final_pst",
        "per_shot_threshold": "ours_final_pst",
        "ours_final_pst": "ours_final_pst",
        "dm": "ours_final_dm",
        "disc_marginal": "ours_final_dm",
        "ours_final_dm": "ours_final_dm",
        "pst_dm": "ours_final_pst_dm",
        "ours_final_pst_dm": "ours_final_pst_dm",
        "pulse_baseline": "ours_final_pulse_baseline",
        "pulse_base": "ours_final_pulse_baseline",
        "ours_final_pulse_baseline": "ours_final_pulse_baseline",
        "pulse_cost": "ours_final_pulse_cost",
        "pulse_region": "ours_final_pulse_cost",
        "ours_final_pulse_cost": "ours_final_pulse_cost",
        "pulse_score_only": "ours_final_pulse_score_only",
        "score_only": "ours_final_pulse_score_only",
        "pulse_evidence_score": "ours_final_pulse_score_only",
        "ours_final_pulse_score_only": "ours_final_pulse_score_only",
        "pulse_full": "ours_final_pulse_full_evidence",
        "pulse_full_evidence": "ours_final_pulse_full_evidence",
        "pulse_evidence": "ours_final_pulse_full_evidence",
        "ours_final_pulse_full_evidence": "ours_final_pulse_full_evidence",
        "mass_scaled": "ours_final_mass_scaled_b0p5",
        "mass_scaled_b0p5": "ours_final_mass_scaled_b0p5",
        "ours_final_mass_scaled_b0p5": "ours_final_mass_scaled_b0p5",
        "mass_scaled_b0p75": "ours_final_mass_scaled_b0p75",
        "ours_final_mass_scaled_b0p75": "ours_final_mass_scaled_b0p75",
        "mass_scaled_b0p25": "ours_final_mass_scaled_b0p25",
        "ours_final_mass_scaled_b0p25": "ours_final_mass_scaled_b0p25",
        "mass_consensus": "ours_final_mass_consensus_a1",
        "mass_on": "ours_final_mass_scaled_b0p5",
        "mass_consensus_a1": "ours_final_mass_consensus_a1",
        "ours_final_mass_consensus_a1": "ours_final_mass_consensus_a1",
        "mass_consensus_a0p75": "ours_final_mass_consensus_a0p75",
        "ours_final_mass_consensus_a0p75": "ours_final_mass_consensus_a0p75",
        "mass_consensus_a0p5": "ours_final_mass_consensus_a0p5",
        "ours_final_mass_consensus_a0p5": "ours_final_mass_consensus_a0p5",
    }

    parsed = []
    for token in raw.split(","):
        name = token.strip().lower().replace("-", "_")
        if not name:
            continue
        if name.startswith("dmuot_") and not name.startswith("ours_final_dmuot_"):
            tag = "ours_final_" + name
        elif name.startswith("ours_final_dmuot_"):
            tag = name
        elif name.startswith("rho_") and not name.startswith("ours_final_rho_"):
            name = "ours_final_" + name
            tag = name.replace(".", "p")
        elif name.startswith("ours_final_rho_"):
            tag = name.replace(".", "p")
        else:
            tag = aliases.get(name)
        if tag is None:
            raise ValueError(
                f"Invalid --ours_final_ablation_variants token '{token}'. "
                "Use full, ccem_uot, partial_ot, full_ot, gap, mass_off, class_pooled, "
                "fixed_shot_pooling, tau_shot_off, mass_scaled_b*, mass_consensus_a*, "
                "rho_<value>, dmuot_<name>, or exact tags."
            )
        if tag not in parsed:
            parsed.append(tag)
    return set(parsed) if parsed else None


def filter_ours_final_variants(variants, allowed_tags):
    if allowed_tags is None:
        return list(variants)
    return [variant for variant in variants if variant["tag"] in allowed_tags]


def parse_cfuget_variant_filter(variants_str):
    if variants_str is None:
        return None
    raw = str(variants_str).strip()
    if not raw or raw.lower() in {"all", "none", "*"}:
        return None

    aliases = {
        "full": "cfuget_full",
        "default": "cfuget_full",
        "cfuget": "cfuget_full",
        "cfuget_full": "cfuget_full",
        "no_structure": "cfuget_no_structure",
        "nostructure": "cfuget_no_structure",
        "no_fgw": "cfuget_no_structure",
        "structure_off": "cfuget_no_structure",
        "cfuget_no_structure": "cfuget_no_structure",
        "no_rival": "cfuget_no_rival",
        "rival_off": "cfuget_no_rival",
        "gate_off": "cfuget_no_rival",
        "cfuget_no_rival": "cfuget_no_rival",
        "no_coherent": "cfuget_no_coherent_score",
        "no_coherent_score": "cfuget_no_coherent_score",
        "raw_score": "cfuget_no_coherent_score",
        "raw_mass": "cfuget_no_coherent_score",
        "cfuget_no_coherent_score": "cfuget_no_coherent_score",
        "cost_only": "cfuget_cost_only",
        "mass_off": "cfuget_cost_only",
        "cfuget_cost_only": "cfuget_cost_only",
        "feature_only": "cfuget_feature_only",
        "uot_only": "cfuget_feature_only",
        "plain_uot": "cfuget_feature_only",
        "cfuget_feature_only": "cfuget_feature_only",
        "fast": "cfuget_sink30",
        "fast_fgw1_sink30": "cfuget_sink30",
        "cfuget_fast_fgw1_sink30": "cfuget_sink30",
        "fgw1": "cfuget_sink30",
        "cfuget_fgw1": "cfuget_sink30",
        "sink30": "cfuget_sink30",
        "sinkhorn30": "cfuget_sink30",
        "cfuget_sink30": "cfuget_sink30",
        "sink70": "cfuget_sink70",
        "sinkhorn70": "cfuget_sink70",
        "cfuget_sink70": "cfuget_sink70",
        "tau005": "cfuget_tau005",
        "tau0p05": "cfuget_tau005",
        "tau_0p05": "cfuget_tau005",
        "cfuget_tau005": "cfuget_tau005",
        "tau010": "cfuget_tau010",
        "tau0p10": "cfuget_tau010",
        "tau_0p10": "cfuget_tau010",
        "cfuget_tau010": "cfuget_tau010",
        "margin0": "cfuget_margin0",
        "margin_0": "cfuget_margin0",
        "cfuget_margin0": "cfuget_margin0",
        "margin005": "cfuget_margin005",
        "margin0p05": "cfuget_margin005",
        "margin_0p05": "cfuget_margin005",
        "cfuget_margin005": "cfuget_margin005",
    }

    parsed = []
    for token in raw.split(","):
        name = token.strip().lower().replace("-", "_")
        if not name:
            continue
        tag = aliases.get(name)
        if tag is None:
            raise ValueError(
                f"Invalid --cfuget_ablation_variants token '{token}'. "
                "Use full, no_rival, no_coherent_score, cost_only, sink30, sink70, "
                "tau005, tau010, margin0, margin005, or exact tags."
            )
        if tag not in parsed:
            parsed.append(tag)
    return set(parsed) if parsed else None


def filter_cfuget_variants(variants, allowed_tags):
    if allowed_tags is None:
        return list(variants)
    return [variant for variant in variants if variant["tag"] in allowed_tags]


def parse_mm_spot_variant_filter(variants_str):
    if variants_str is None:
        return None
    raw = str(variants_str).strip()
    if not raw or raw.lower() in {"all", "none", "*"}:
        return None

    aliases = {
        "default": "mm_spot_k3_softmax",
        "k3": "mm_spot_k3_softmax",
        "bank": "mm_spot_k3_softmax",
        "bank_k3": "mm_spot_k3_softmax",
        "softmax": "mm_spot_k3_softmax",
        "mm_spot_k3_softmax": "mm_spot_k3_softmax",
        "single": "mm_spot_single_s04",
        "s04": "mm_spot_single_s04",
        "single_s04": "mm_spot_single_s04",
        "mm_spot_single_s04": "mm_spot_single_s04",
        "s03": "mm_spot_single_s03",
        "single_s03": "mm_spot_single_s03",
        "mm_spot_single_s03": "mm_spot_single_s03",
        "s05": "mm_spot_single_s05",
        "single_s05": "mm_spot_single_s05",
        "mm_spot_single_s05": "mm_spot_single_s05",
        "mean": "mm_spot_k3_mean",
        "k3_mean": "mm_spot_k3_mean",
        "mm_spot_k3_mean": "mm_spot_k3_mean",
        "iter60": "mm_spot_k3_iter60",
        "k3_iter60": "mm_spot_k3_iter60",
        "mm_spot_k3_iter60": "mm_spot_k3_iter60",
        "iter120": "mm_spot_k3_iter120",
        "k3_iter120": "mm_spot_k3_iter120",
        "mm_spot_k3_iter120": "mm_spot_k3_iter120",
        "k5": "mm_spot_k5_iter60",
        "k5_iter60": "mm_spot_k5_iter60",
        "mm_spot_k5_iter60": "mm_spot_k5_iter60",
        "single_s04_iter120": "mm_spot_single_s04_iter120",
        "mm_spot_single_s04_iter120": "mm_spot_single_s04_iter120",
        "eps003": "mm_spot_k3_eps003",
        "eps0p03": "mm_spot_k3_eps003",
        "mm_spot_k3_eps003": "mm_spot_k3_eps003",
        "eps005": "mm_spot_k3_eps005",
        "eps0p05": "mm_spot_k3_eps005",
        "mm_spot_k3_eps005": "mm_spot_k3_eps005",
        "eps010": "mm_spot_k3_eps010",
        "eps0p10": "mm_spot_k3_eps010",
        "mm_spot_k3_eps010": "mm_spot_k3_eps010",
        "controller": "mm_spot_controller_k3",
        "controller_k3": "mm_spot_controller_k3",
        "mm_spot_controller_k3": "mm_spot_controller_k3",
    }

    parsed = []
    for token in raw.split(","):
        name = token.strip().lower().replace("-", "_").replace(".", "p")
        if not name:
            continue
        tag = aliases.get(name, name if name.startswith("mm_spot_") else None)
        if tag is None:
            raise ValueError(
                f"Invalid --spot_ablation_variants token '{token}'. "
                "Use default/k3, single/s04, s03, s05, mean, iter60, iter120, "
                "k5, eps003, eps005, eps010, controller, or exact mm_spot_* tags."
            )
        if tag not in parsed:
            parsed.append(tag)
    return set(parsed) if parsed else None


def parse_pare_variant_filter(variants_str):
    if variants_str is None or str(variants_str).strip().lower() in {"", "all", "*"}:
        return None
    aliases = {
        "default": "pare_default",
        "pare_default": "pare_default",
        "uniform": "pare_uniform",
        "pare_uniform": "pare_uniform",
        "fixed": "pare_fixed_alpha",
        "fixed_alpha": "pare_fixed_alpha",
        "pare_fixed_alpha": "pare_fixed_alpha",
        "iter60": "pare_iter60",
        "pare_iter60": "pare_iter60",
        "iter80": "pare_iter80",
        "pare_iter80": "pare_iter80",
        "iter120": "pare_iter120",
        "pare_iter120": "pare_iter120",
        "eps002": "pare_eps002",
        "eps0p02": "pare_eps002",
        "pare_eps002": "pare_eps002",
        "eps006": "pare_eps006",
        "eps0p06": "pare_eps006",
        "pare_eps006": "pare_eps006",
        "shot_mean": "pare_shot_mean",
        "pare_shot_mean": "pare_shot_mean",
        "alpha_wide": "pare_alpha_wide",
        "pare_alpha_wide": "pare_alpha_wide",
        "alpha_noreg": "pare_alpha_noreg",
        "pare_alpha_noreg": "pare_alpha_noreg",
    }
    parsed = []
    for token in str(variants_str).split(","):
        name = token.strip().lower().replace("-", "_")
        if not name:
            continue
        tag = aliases.get(name, name if name.startswith("pare_") else None)
        if tag is None:
            raise ValueError(
                f"Invalid --pare_ablation_variants token '{token}'. "
                "Use aliases such as default, single, uniform, iter60, a045, eps004, "
                "shot_mean, tau010, or exact pare_* tags."
            )
        if tag not in parsed:
            parsed.append(tag)
    return set(parsed) if parsed else None


def filter_pare_variants(variants, allowed_tags):
    if allowed_tags is None:
        return list(variants)
    return [variant for variant in variants if variant["tag"] in allowed_tags]


def filter_mm_spot_variants(variants, allowed_tags):
    if allowed_tags is None:
        return list(variants)
    return [variant for variant in variants if variant["tag"] in allowed_tags]


def _parse_sgpot_variant_filter(variants_str):
    if variants_str is None or str(variants_str).strip().lower() in {"", "all", "*"}:
        return None
    parsed = []
    for token in str(variants_str).split(","):
        name = token.strip().lower().replace("-", "_")
        if not name:
            continue
        tag = name if name.startswith("sgpot_") else f"sgpot_{name}"
        if tag not in parsed:
            parsed.append(tag)
    return set(parsed) if parsed else None


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
    training_seed_arg = getattr(args, "seeds", None) if getattr(args, "seeds", None) is not None else args.seed
    training_seeds = parse_training_seeds(training_seed_arg)
    args.training_seeds = training_seeds
    args.seed = int(training_seeds[0])
    final_test_seeds = parse_final_test_seeds(getattr(args, "final_test_seeds", None))
    if final_test_seeds is None:
        final_test_seeds = [int(args.final_test_seed)]
    args.final_test_seed = int(final_test_seeds[0])
    args.extra_final_test_seeds = [int(seed) for seed in final_test_seeds[1:]]
    requested_extra_test_protocols = parse_extra_test_protocols(getattr(args, "extra_test_protocols", None))
    parsed_mode_ids = parse_mode_ids(getattr(args, "mode_ids", None))
    if args.mode_id is not None and parsed_mode_ids is not None:
        raise ValueError("Use either --mode_id or --mode_ids, not both.")
    ours_final_rho_values = parse_rho_values(getattr(args, "ours_final_rho_values", None))
    ours_final_variant_filter = parse_ours_final_variant_filter(
        getattr(args, "ours_final_ablation_variants", "all")
    )
    cfuget_variant_filter = parse_cfuget_variant_filter(
        getattr(args, "cfuget_ablation_variants", "all")
    )
    mm_spot_variant_filter = parse_mm_spot_variant_filter(
        getattr(args, "spot_ablation_variants", "all")
    )
    pare_variant_filter = parse_pare_variant_filter(
        getattr(args, "pare_ablation_variants", "all")
    )
    sgpot_variant_filter = _parse_sgpot_variant_filter(
        getattr(args, "sgpot_ablation_variants", "all")
    )
    if args.spif_global_only == "true" and args.spif_local_only == "true":
        raise ValueError("`--spif_global_only` and `--spif_local_only` cannot both be true.")
    dataset_specs = []
    for spec in build_dataset_specs(args):
        spec = dict(spec)
        if dataset_has_noise_benchmark_layout(spec["path"]) and not dataset_has_training_layout(spec["path"]):
            spec["noise_test_root"] = spec["path"]
            inferred_dataset_path = infer_training_dataset_path(spec["path"])
            if inferred_dataset_path is None:
                raise ValueError(
                    "--dataset_path points to a noise benchmark root. "
                    "Pass the clean train/val/test dataset with --dataset_path, "
                    "or place the matching clean dataset beside the noise root."
                )
            print(
                "Interpreting dataset path as --noise_test_root; "
                f"using clean dataset: {inferred_dataset_path}"
            )
            spec["path"] = str(inferred_dataset_path)

        spec["test_protocol"] = resolve_test_protocol(
            args.test_protocol,
            spec.get("noise_test_root"),
            args.noise_test_splits,
        )
        spec["noise_test_split_names"] = (
            discover_noise_test_splits(spec.get("noise_test_root"), args.noise_test_splits)
            if spec["test_protocol"] == "noise"
            else []
        )
        spec["extra_test_protocols"] = [
            protocol for protocol in requested_extra_test_protocols if protocol != spec["test_protocol"]
        ]
        if "noise" in spec["extra_test_protocols"]:
            extra_noise_splits = discover_noise_test_splits(
                spec.get("noise_test_root"),
                args.noise_test_splits,
            )
            if not extra_noise_splits:
                raise ValueError(
                    "--extra_test_protocols noise requires --noise_test_root with at least one SNR test folder."
                )
            if not spec["noise_test_split_names"]:
                spec["noise_test_split_names"] = extra_noise_splits
        dataset_specs.append(spec)

    primary_dataset_spec = dataset_specs[0]
    args.dataset_path = primary_dataset_spec["path"]
    args.dataset_name = primary_dataset_spec["name"]
    args.noise_test_root = primary_dataset_spec.get("noise_test_root")
    effective_test_protocol = primary_dataset_spec["test_protocol"]
    noise_test_split_names = primary_dataset_spec["noise_test_split_names"]
    args.extra_test_protocols = primary_dataset_spec["extra_test_protocols"]
    shots = [args.shot_num] if args.shot_num is not None else SHOTS_DEFAULT
    requested_models = parse_requested_models(args.models)
    valid_models = set(get_model_choices())
    invalid_models = [model for model in requested_models if model not in valid_models]
    if invalid_models:
        raise ValueError(f"Unsupported models: {invalid_models}. Valid choices: {sorted(valid_models)}")

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    if str(getattr(args, "log_cli_command", "false")).lower() == "true":
        log_cli_command(args)

    active_ablation_suites = [
        name
        for name, value in (
            ("--spifce_ablation_suite", args.spifce_ablation_suite),
            ("--ours_ablation_suite", args.ours_ablation_suite),
            ("--ours_final_ablation_suite", args.ours_final_ablation_suite),
            ("--cfuget_ablation_suite", args.cfuget_ablation_suite),
            ("--spot_ablation_suite", args.spot_ablation_suite),
            ("--pare_ablation_suite", args.pare_ablation_suite),
            ("--sgpot_ablation_suite", args.sgpot_ablation_suite),
        )
        if value != "none"
    ]
    if len(active_ablation_suites) > 1:
        raise ValueError(f"Use only one ablation suite at a time: {', '.join(active_ablation_suites)}.")

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
        samples_list = list(OURS_FINAL_SAMPLE_COUNTS)
        experiments = [
            {
                "model": "ours",
                "samples": samples,
                "shot": shot,
                "variant_args": variant["extra_args"],
                "experiment_tag": variant["tag"],
                "checkpoint_tag": variant.get("checkpoint_tag", variant["tag"]),
                "experiment_label": variant["label"],
            }
            for variant in ablation_variants
            for samples in samples_list
            for shot in shots
        ]
        print("=" * 72)
        print("pulse_fewshot - Ours Contribution Ablation Suite")
        print("=" * 72)
        print("Samples     : 60 and 240 only (see OURS_ABLATION_SAMPLE_COUNTS)")
        print(f"Models      : {', '.join(requested_models)}")
        print(f"Shots       : {', '.join(f'{shot}-shot' for shot in shots)}")
        print(f"Backbone    : {args.fewshot_backbone}")
        print(f"Ablation    : {args.ours_ablation_suite}")
        print("Variants    :")
        for variant in ablation_variants:
            print(f"  - {variant['tag']}: {variant['label']}")
        print_dataset_plan(dataset_specs)
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
        print(format_planned_total(experiments, training_seeds, "ablation experiment(s)", len(dataset_specs)))
        print("=" * 72)
    elif args.ours_final_ablation_suite != "none":
        suite_name = args.ours_final_ablation_suite
        allowed_requested_models = [[OURS_FINAL_MODEL_NAME]]
        if suite_name == "partial_ot":
            allowed_requested_models.append([OURS_FINAL_PARTIAL_OT_MODEL_NAME])
        if requested_models not in allowed_requested_models:
            raise ValueError(
                f"`--ours_final_ablation_suite {suite_name}` currently supports only "
                f"`--models {OURS_FINAL_MODEL_NAME}`"
                + (
                    f" or `--models {OURS_FINAL_PARTIAL_OT_MODEL_NAME}`"
                    if suite_name == "partial_ot"
                    else ""
                )
                + " "
                f"(got {requested_models})"
            )
        contrib_variants = build_ours_final_ablation_variants()
        rho_grid_variants = build_ours_final_rho_grid_variants(ours_final_rho_values)
        tau_shot_off_variants = build_ours_final_tau_shot_off_variants()
        pooling_variants = build_ours_final_pooling_variants()
        mass_on_variants = build_ours_final_mass_on_variants()
        dmuot_variants = build_ours_final_dmuot_variants()
        pst_dm_variants = build_ours_final_pst_dm_variants()
        evidence_variants = build_ours_final_evidence_marginal_variants()
        pulse_region_variants = build_ours_final_pulse_region_variants()
        if suite_name == "dmuot" and ours_final_variant_filter is None:
            dmuot_variants = [
                variant
                for variant in dmuot_variants
                if not variant["tag"].startswith("ours_final_dmuot_exp0_")
            ]
        contrib_variants = filter_ours_final_variants(contrib_variants, ours_final_variant_filter)
        rho_grid_variants = filter_ours_final_variants(rho_grid_variants, ours_final_variant_filter)
        tau_shot_off_variants = filter_ours_final_variants(tau_shot_off_variants, ours_final_variant_filter)
        pooling_variants = filter_ours_final_variants(pooling_variants, ours_final_variant_filter)
        mass_on_variants = filter_ours_final_variants(mass_on_variants, ours_final_variant_filter)
        dmuot_variants = filter_ours_final_variants(dmuot_variants, ours_final_variant_filter)
        pst_dm_variants = filter_ours_final_variants(pst_dm_variants, ours_final_variant_filter)
        evidence_variants = filter_ours_final_variants(evidence_variants, ours_final_variant_filter)
        pulse_region_variants = filter_ours_final_variants(pulse_region_variants, ours_final_variant_filter)
        if suite_name == "rho_grid":
            ablation_variants = rho_grid_variants
        elif suite_name == "tau_shot_off":
            ablation_variants = tau_shot_off_variants
        elif suite_name == "pooling":
            ablation_variants = pooling_variants
        elif suite_name == "dmuot":
            ablation_variants = dmuot_variants
        elif suite_name == "mass_on":
            ablation_variants = mass_on_variants
        elif suite_name == "pst_dm":
            ablation_variants = pst_dm_variants
        elif suite_name == "evidence":
            ablation_variants = evidence_variants
        elif suite_name == "pulse_region":
            ablation_variants = pulse_region_variants
        elif suite_name == "partial_ot":
            ablation_variants = [
                variant for variant in contrib_variants if variant["tag"] == "ours_final_partial_ot"
            ]
        elif suite_name == "complete":
            ablation_variants = contrib_variants + rho_grid_variants + pst_dm_variants + evidence_variants
        else:
            ablation_variants = contrib_variants
        if not ablation_variants:
            requested = ", ".join(sorted(ours_final_variant_filter or [])) or "all"
            raise ValueError(
                f"No Ours-Final variants selected for suite={suite_name!r} "
                f"with --ours_final_ablation_variants={requested}."
            )
        if args.mode_id is not None:
            samples_list = [EXPERIMENT_MODES[args.mode_id]]
        elif parsed_mode_ids is not None:
            samples_list = [EXPERIMENT_MODES[mode_id] for mode_id in parsed_mode_ids]
        else:
            samples_list = list(OURS_FINAL_SAMPLE_COUNTS)
        mode1_sample_count = EXPERIMENT_MODES[1]
        enable_mode1_noise_tests = (
            suite_name in {"contrib", "complete", "partial_ot"}
            and effective_test_protocol != "noise"
            and mode1_sample_count in samples_list
            and bool(discover_noise_test_splits(args.noise_test_root, args.noise_test_splits))
        )
        def include_mode1_noise(variant):
            return suite_name == "complete" or variant.get("mode1_noise", False)

        mode1_noise_splits = []
        mode1_noise_splits_arg = None
        if enable_mode1_noise_tests:
            mode1_noise_splits = select_noise_snr_splits(
                args.noise_test_root,
                args.noise_test_splits,
                snr_values=(15, 10, 5),
            )
            mode1_noise_splits_arg = ",".join(mode1_noise_splits)
        experiments = []
        if suite_name in {"contrib", "complete", "partial_ot"}:
            suite_contrib_variants = ablation_variants if suite_name == "partial_ot" else contrib_variants
            experiments.extend(
                {
                    "model": variant.get("model", OURS_FINAL_MODEL_NAME),
                    "samples": samples,
                    "shot": shot,
                    "variant_args": variant["extra_args"],
                    "experiment_tag": variant["tag"],
                    "checkpoint_tag": variant.get("checkpoint_tag", variant["tag"]),
                    "experiment_label": variant["label"],
                    "extra_test_protocols": merge_extra_test_protocols(
                        args.extra_test_protocols,
                        ["noise"]
                        if (
                            enable_mode1_noise_tests
                            and samples == mode1_sample_count
                            and include_mode1_noise(variant)
                        )
                        else [],
                    ),
                    "extra_noise_test_splits": (
                        mode1_noise_splits_arg
                        if (
                            enable_mode1_noise_tests
                            and samples == mode1_sample_count
                            and include_mode1_noise(variant)
                        )
                        else None
                    ),
                }
                for variant in suite_contrib_variants
                for samples in samples_list
                for shot in shots
            )
        if suite_name == "tau_shot_off":
            experiments.extend(
                {
                    "model": variant.get("model", OURS_FINAL_MODEL_NAME),
                    "samples": samples,
                    "shot": shot,
                    "variant_args": variant["extra_args"],
                    "experiment_tag": variant["tag"],
                    "checkpoint_tag": variant.get("checkpoint_tag", variant["tag"]),
                    "experiment_label": variant["label"],
                    "extra_test_protocols": args.extra_test_protocols,
                    "extra_noise_test_splits": None,
                }
                for variant in tau_shot_off_variants
                for samples in samples_list
                for shot in shots
            )
        if suite_name == "pooling":
            experiments.extend(
                {
                    "model": variant.get("model", OURS_FINAL_MODEL_NAME),
                    "samples": samples,
                    "shot": shot,
                    "variant_args": variant["extra_args"],
                    "experiment_tag": variant["tag"],
                    "checkpoint_tag": variant.get("checkpoint_tag", variant["tag"]),
                    "experiment_label": variant["label"],
                    "extra_test_protocols": args.extra_test_protocols,
                    "extra_noise_test_splits": None,
                }
                for variant in pooling_variants
                for samples in samples_list
                for shot in shots
            )
        if suite_name == "dmuot":
            experiments.extend(
                {
                    "model": variant.get("model", OURS_FINAL_MODEL_NAME),
                    "samples": samples,
                    "shot": shot,
                    "variant_args": variant["extra_args"],
                    "experiment_tag": variant["tag"],
                    "checkpoint_tag": variant.get("checkpoint_tag", variant["tag"]),
                    "experiment_label": variant["label"],
                    "extra_test_protocols": args.extra_test_protocols,
                    "extra_noise_test_splits": None,
                }
                for variant in dmuot_variants
                for samples in samples_list
                for shot in shots
            )
        if suite_name == "mass_on":
            experiments.extend(
                {
                    "model": variant.get("model", OURS_FINAL_MODEL_NAME),
                    "samples": samples,
                    "shot": shot,
                    "variant_args": variant["extra_args"],
                    "experiment_tag": variant["tag"],
                    "checkpoint_tag": variant.get("checkpoint_tag", variant["tag"]),
                    "experiment_label": variant["label"],
                    "extra_test_protocols": args.extra_test_protocols,
                    "extra_noise_test_splits": None,
                }
                for variant in mass_on_variants
                for samples in samples_list
                for shot in shots
            )
        if suite_name == "pulse_region":
            experiments.extend(
                {
                    "model": variant.get("model", OURS_FINAL_MODEL_NAME),
                    "samples": samples,
                    "shot": shot,
                    "variant_args": variant["extra_args"],
                    "experiment_tag": variant["tag"],
                    "checkpoint_tag": variant.get("checkpoint_tag", variant["tag"]),
                    "experiment_label": variant["label"],
                    "extra_test_protocols": args.extra_test_protocols,
                    "extra_noise_test_splits": None,
                }
                for variant in pulse_region_variants
                for samples in samples_list
                for shot in shots
            )
        if suite_name in {"pst_dm", "complete"} and pst_dm_variants:
            experiments.extend(
                {
                    "model": variant.get("model", OURS_FINAL_MODEL_NAME),
                    "samples": samples,
                    "shot": shot,
                    "variant_args": variant["extra_args"],
                    "experiment_tag": variant["tag"],
                    "checkpoint_tag": variant.get("checkpoint_tag", variant["tag"]),
                    "experiment_label": variant["label"],
                    "extra_test_protocols": args.extra_test_protocols,
                    "extra_noise_test_splits": None,
                }
                for variant in pst_dm_variants
                for samples in samples_list
                for shot in shots
            )
        if suite_name in {"rho_grid", "complete"} and rho_grid_variants:
            rho_grid_pairs = restricted_ours_final_rho_grid_pairs(samples_list, shots)
            if suite_name == "rho_grid" and not rho_grid_pairs:
                raise ValueError(
                    "Ours-Final rho_grid is restricted to 60-sample/5-shot and 240-sample/1-shot. "
                    "Requested modes/shots do not include either pair."
                )
            experiments.extend(
                {
                    "model": variant.get("model", OURS_FINAL_MODEL_NAME),
                    "samples": samples,
                    "shot": shot,
                    "variant_args": variant["extra_args"],
                    "experiment_tag": variant["tag"],
                    "checkpoint_tag": variant.get("checkpoint_tag", variant["tag"]),
                    "experiment_label": variant["label"],
                    "extra_test_protocols": args.extra_test_protocols,
                    "extra_noise_test_splits": None,
                }
                for variant in rho_grid_variants
                for samples, shot in rho_grid_pairs
            )
        print("=" * 72)
        print("pulse_fewshot - Ours-Final Ablation Suite")
        print("=" * 72)
        sample_text = ", ".join(str(sample) if sample is not None else "All" for sample in samples_list)
        print(f"Samples     : {sample_text}")
        print(f"Models      : {', '.join(requested_models)}")
        print(f"Shots       : {', '.join(f'{shot}-shot' for shot in shots)}")
        print(f"Backbone    : {args.fewshot_backbone}")
        print(f"Ablation    : {args.ours_final_ablation_suite}")
        if ours_final_variant_filter is not None:
            print(f"Variant Pick: {', '.join(sorted(ours_final_variant_filter))}")
        print("Variants    :")
        for variant in ablation_variants:
            variant_model = variant.get("model", OURS_FINAL_MODEL_NAME)
            model_text = f" [{variant_model}]" if variant_model != OURS_FINAL_MODEL_NAME else ""
            print(f"  - {variant['tag']}{model_text}: {variant['label']}")
        if suite_name in {"rho_grid", "complete"} and rho_grid_variants:
            rho_pairs_text = ", ".join(
                f"{samples} samples/{shot}-shot"
                for samples, shot in restricted_ours_final_rho_grid_pairs(samples_list, shots)
            )
            print(f"Rho Grid    : {rho_pairs_text if rho_pairs_text else '(no selected rho-grid pair)'}")
            if ours_final_rho_values:
                print(f"Rho Values  : {', '.join(ours_final_rho_values)}")
        if enable_mode1_noise_tests:
            mode1_noise_tags = [
                variant["tag"]
                for variant in contrib_variants
                if include_mode1_noise(variant)
            ]
            mode1_noise_run_count = len(mode1_noise_tags) * len(shots)
            print(
                "Mode-1 Noise : test-only after clean training for "
                f"{', '.join(mode1_noise_tags)} at {mode1_sample_count} samples "
                f"({mode1_noise_run_count} run(s))"
            )
            print(f"Noise Root  : {args.noise_test_root}")
            print(
                "Noise Splits: "
                f"{', '.join(mode1_noise_splits)}"
            )
        print_dataset_plan(dataset_specs)
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
        print(format_planned_total(experiments, training_seeds, "ablation experiment(s)", len(dataset_specs)))
        print("=" * 72)
    elif args.cfuget_ablation_suite != "none":
        if requested_models != [CFUGET_MODEL_NAME]:
            raise ValueError(
                f"`--cfuget_ablation_suite` currently supports only `--models {CFUGET_MODEL_NAME}` "
                f"(got {requested_models})"
            )
        suite_name = args.cfuget_ablation_suite
        ablation_variants = filter_cfuget_variants(
            build_cfuget_ablation_variants(suite_name),
            cfuget_variant_filter,
        )
        if not ablation_variants:
            requested = ", ".join(sorted(cfuget_variant_filter or [])) or "all"
            raise ValueError(
                f"No C-FUGET variants selected for suite={suite_name!r} "
                f"with --cfuget_ablation_variants={requested}."
            )
        if args.mode_id is not None:
            samples_list = [EXPERIMENT_MODES[args.mode_id]]
        elif parsed_mode_ids is not None:
            samples_list = [EXPERIMENT_MODES[mode_id] for mode_id in parsed_mode_ids]
        else:
            samples_list = list(OURS_FINAL_SAMPLE_COUNTS)
        experiments = [
            {
                "model": CFUGET_MODEL_NAME,
                "samples": samples,
                "shot": shot,
                "variant_args": variant["extra_args"],
                "experiment_tag": variant["tag"],
                "checkpoint_tag": variant.get("checkpoint_tag", variant["tag"]),
                "experiment_label": variant["label"],
                "extra_test_protocols": args.extra_test_protocols,
                "extra_noise_test_splits": None,
            }
            for variant in ablation_variants
            for samples in samples_list
            for shot in shots
        ]
        print("=" * 72)
        print("pulse_fewshot - RC-UOT Ablation Suite")
        print("=" * 72)
        sample_text = ", ".join(str(sample) if sample is not None else "All" for sample in samples_list)
        print(f"Samples     : {sample_text}")
        print(f"Models      : {', '.join(requested_models)}")
        print(f"Shots       : {', '.join(f'{shot}-shot' for shot in shots)}")
        print(f"Backbone    : {args.fewshot_backbone}")
        print(f"Ablation    : {args.cfuget_ablation_suite}")
        if cfuget_variant_filter is not None:
            print(f"Variant Pick: {', '.join(sorted(cfuget_variant_filter))}")
        print("Variants    :")
        for variant in ablation_variants:
            print(f"  - {variant['tag']}: {variant['label']}")
        print_dataset_plan(dataset_specs)
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
            "RC-UOT: fixed rho=0.8 UOT, rival coherent evidence gate, threshold-mass score"
        )
        if args.passthrough_args:
            print(f"Forwarded   : {' '.join(args.passthrough_args)}")
        print(f"Test Proto  : {effective_test_protocol}")
        print(format_final_test_seed_line(args))
        print_extra_test_protocol_line(args)
        if effective_test_protocol == "noise":
            print(f"Noise Root  : {args.noise_test_root}")
            print(f"Noise Splits: {', '.join(noise_test_split_names)}")
        print(format_planned_total(experiments, training_seeds, "ablation experiment(s)", len(dataset_specs)))
        print("=" * 72)
    elif args.spot_ablation_suite != "none":
        if requested_models != [MM_SPOT_MODEL_NAME]:
            raise ValueError(
                f"`--spot_ablation_suite` currently supports only `--models {MM_SPOT_MODEL_NAME}` "
                f"(got {requested_models})"
            )
        suite_name = args.spot_ablation_suite
        ablation_variants = filter_mm_spot_variants(
            build_mm_spot_ablation_variants(suite_name),
            mm_spot_variant_filter,
        )
        if not ablation_variants:
            requested = ", ".join(sorted(mm_spot_variant_filter or [])) or "all"
            raise ValueError(
                f"No MM-SPOT-FSL variants selected for suite={suite_name!r} "
                f"with --spot_ablation_variants={requested}."
            )
        if args.mode_id is not None:
            samples_list = [EXPERIMENT_MODES[args.mode_id]]
        elif parsed_mode_ids is not None:
            samples_list = [EXPERIMENT_MODES[mode_id] for mode_id in parsed_mode_ids]
        else:
            samples_list = list(OURS_FINAL_SAMPLE_COUNTS)
        experiments = [
            {
                "model": MM_SPOT_MODEL_NAME,
                "samples": samples,
                "shot": shot,
                "variant_args": variant["extra_args"],
                "experiment_tag": variant["tag"],
                "checkpoint_tag": variant.get("checkpoint_tag", variant["tag"]),
                "experiment_label": variant["label"],
                "extra_test_protocols": args.extra_test_protocols,
                "extra_noise_test_splits": None,
            }
            for variant in ablation_variants
            for samples in samples_list
            for shot in shots
        ]
        print("=" * 72)
        print("pulse_fewshot - MM-SPOT-FSL Multi-Mass POT Suite")
        print("=" * 72)
        sample_text = ", ".join(str(sample) if sample is not None else "All" for sample in samples_list)
        print(f"Samples     : {sample_text}")
        print(f"Models      : {', '.join(requested_models)}")
        print(f"Shots       : {', '.join(f'{shot}-shot' for shot in shots)}")
        print(f"Backbone    : {args.fewshot_backbone}")
        print(f"Ablation    : {args.spot_ablation_suite}")
        if mm_spot_variant_filter is not None:
            print(f"Variant Pick: {', '.join(sorted(mm_spot_variant_filter))}")
        print("Variants    :")
        for variant in ablation_variants:
            print(f"  - {variant['tag']}: {variant['label']}")
        print_dataset_plan(dataset_specs)
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
            "MM-SPOT score=-score_scale*C_s/s, full marginals, no aux loss by default"
        )
        if args.passthrough_args:
            print(f"Forwarded   : {' '.join(args.passthrough_args)}")
        print(f"Test Proto  : {effective_test_protocol}")
        print(format_final_test_seed_line(args))
        print_extra_test_protocol_line(args)
        if effective_test_protocol == "noise":
            print(f"Noise Root  : {args.noise_test_root}")
            print(f"Noise Splits: {', '.join(noise_test_split_names)}")
        print(format_planned_total(experiments, training_seeds, "ablation experiment(s)", len(dataset_specs)))
        print("=" * 72)
    elif args.pare_ablation_suite != "none":
        if requested_models != [PARE_FSL_MODEL_NAME]:
            raise ValueError(
                f"`--pare_ablation_suite` currently supports only `--models {PARE_FSL_MODEL_NAME}` "
                f"(got {requested_models})"
            )
        suite_name = args.pare_ablation_suite
        ablation_variants = filter_pare_variants(
            build_pare_ablation_variants(suite_name),
            pare_variant_filter,
        )
        if not ablation_variants:
            requested = ", ".join(sorted(pare_variant_filter or [])) or "all"
            raise ValueError(
                f"No PARE-FSL variants selected for suite={suite_name!r} "
                f"with --pare_ablation_variants={requested}."
            )
        if args.mode_id is not None:
            samples_list = [EXPERIMENT_MODES[args.mode_id]]
        elif parsed_mode_ids is not None:
            samples_list = [EXPERIMENT_MODES[mode_id] for mode_id in parsed_mode_ids]
        else:
            samples_list = list(OURS_FINAL_SAMPLE_COUNTS)
        experiments = [
            {
                "model": PARE_FSL_MODEL_NAME,
                "samples": samples,
                "shot": shot,
                "variant_args": variant["extra_args"],
                "experiment_tag": variant["tag"],
                "checkpoint_tag": variant.get("checkpoint_tag", variant["tag"]),
                "experiment_label": variant["label"],
                "extra_test_protocols": args.extra_test_protocols,
                "extra_noise_test_splits": None,
            }
            for variant in ablation_variants
            for samples in samples_list
            for shot in shots
        ]
        print("=" * 72)
        print("pulse_fewshot - PARE-FSL (learned partial mass, EGSM marginals)")
        print("=" * 72)
        sample_text = ", ".join(str(sample) if sample is not None else "All" for sample in samples_list)
        print(f"Samples     : {sample_text}")
        print(f"Models      : {', '.join(requested_models)}")
        print(f"Shots       : {', '.join(f'{shot}-shot' for shot in shots)}")
        print(f"Backbone    : {args.fewshot_backbone}")
        print(f"Ablation    : {args.pare_ablation_suite}")
        if pare_variant_filter is not None:
            print(f"Variant Pick: {', '.join(sorted(pare_variant_filter))}")
        print("Variants    :")
        for variant in ablation_variants:
            print(f"  - {variant['tag']}: {variant['label']}")
        print_dataset_plan(dataset_specs)
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
            "PARE-FSL: EGSM marginals, learned alpha per shot, one native POT, score=-scale*C/M"
        )
        if args.passthrough_args:
            print(f"Forwarded   : {' '.join(args.passthrough_args)}")
        print(f"Test Proto  : {effective_test_protocol}")
        print(format_final_test_seed_line(args))
        print_extra_test_protocol_line(args)
        if effective_test_protocol == "noise":
            print(f"Noise Root  : {args.noise_test_root}")
            print(f"Noise Splits: {', '.join(noise_test_split_names)}")
        print(format_planned_total(experiments, training_seeds, "ablation experiment(s)", len(dataset_specs)))
        print("=" * 72)
    elif args.sgpot_ablation_suite != "none":
        if requested_models != [SGPOT_MODEL_NAME]:
            raise ValueError(
                f"`--sgpot_ablation_suite` currently supports only `--models {SGPOT_MODEL_NAME}` "
                f"(got {requested_models})"
            )
        suite_name = args.sgpot_ablation_suite
        ablation_variants = filter_sgpot_variants(
            build_sgpot_ablation_variants(suite_name),
            sgpot_variant_filter,
        )
        if not ablation_variants:
            requested = ", ".join(sorted(sgpot_variant_filter or [])) or "all"
            raise ValueError(
                f"No SG-POT variants selected for suite={suite_name!r} "
                f"with --sgpot_ablation_variants={requested}."
            )
        samples_list = (
            [EXPERIMENT_MODES[args.mode_id]]
            if args.mode_id is not None
            else ([EXPERIMENT_MODES[mid] for mid in parsed_mode_ids] if parsed_mode_ids else SAMPLES_LIST)
        )
        experiments = [
            {
                "model": SGPOT_MODEL_NAME,
                "samples": samples,
                "shot": shot,
                "variant_args": variant["extra_args"],
                "experiment_tag": variant["tag"],
                "checkpoint_tag": variant.get("checkpoint_tag", variant["tag"]),
                "experiment_label": variant["label"],
                "extra_test_protocols": args.extra_test_protocols,
                "extra_noise_test_splits": None,
            }
            for variant in ablation_variants
            for samples in samples_list
            for shot in shots
        ]
        print("=" * 72)
        print("pulse_fewshot - SG-POT (saliency-guided partial optimal transport)")
        print("=" * 72)
        sample_text = ", ".join(str(sample) if sample is not None else "All" for sample in samples_list)
        print(f"Samples     : {sample_text}")
        print(f"Models      : {', '.join(requested_models)}")
        print(f"Shots       : {', '.join(f'{shot}-shot' for shot in shots)}")
        print(f"Backbone    : {args.fewshot_backbone}")
        print(f"Ablation    : {args.sgpot_ablation_suite}")
        if sgpot_variant_filter is not None:
            print(f"Variant Pick: {', '.join(sorted(sgpot_variant_filter))}")
        print("Variants    :")
        for variant in ablation_variants:
            print(f"  - {variant['tag']}: {variant['label']}")
        print_dataset_plan(dataset_specs)
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
            "SG-POT: saliency marginals, adaptive mass fraction, dual evidence score"
        )
        if args.passthrough_args:
            print(f"Forwarded   : {' '.join(args.passthrough_args)}")
        print(f"Test Proto  : {effective_test_protocol}")
        print(format_final_test_seed_line(args))
        print_extra_test_protocol_line(args)
        if effective_test_protocol == "noise":
            print(f"Noise Root  : {args.noise_test_root}")
            print(f"Noise Splits: {', '.join(noise_test_split_names)}")
        print(format_planned_total(experiments, training_seeds, "ablation experiment(s)", len(dataset_specs)))
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
        print_dataset_plan(dataset_specs)
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
        print(format_planned_total(experiments, training_seeds, "ablation experiment(s)", len(dataset_specs)))
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
        print_dataset_plan(dataset_specs)
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
        print(format_planned_total(experiments, training_seeds, "experiment(s)", len(dataset_specs)))
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
        print_dataset_plan(dataset_specs)
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
        print(format_planned_total(experiments, training_seeds, "experiment(s)", len(dataset_specs)))
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
        print_dataset_plan(dataset_specs)
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
        print(format_planned_total(experiments, training_seeds, "experiments", len(dataset_specs)))
        print("=" * 72)

    total_runs = len(dataset_specs) * len(experiments) * len(training_seeds)
    multiple_training_seeds = len(training_seeds) > 1
    multiple_datasets = len(dataset_specs) > 1
    if multiple_datasets or multiple_training_seeds or str(getattr(args, "experiment_tag", "") or "").strip():
        print("\n" + "=" * 72)
        print("Run Expansion")
        print("=" * 72)
        if multiple_datasets:
            print(f"Datasets    : {len(dataset_specs)}")
        print(format_training_seed_line(training_seeds))
        if str(getattr(args, "experiment_tag", "") or "").strip():
            print(f"Base Tag    : {args.experiment_tag}")
        print(
            f"Runs        : {len(dataset_specs)} dataset(s) x "
            f"{len(experiments)} experiment spec(s) x {len(training_seeds)} seed(s) = {total_runs}"
        )
        print("=" * 72)

    success_count = 0
    failed_experiments = []

    for dataset_index, dataset_spec in enumerate(dataset_specs):
        for seed_index, train_seed in enumerate(training_seeds):
            for experiment_index, experiment in enumerate(experiments, 1):
                idx = (
                    dataset_index * len(training_seeds) * len(experiments)
                    + seed_index * len(experiments)
                    + experiment_index
                )
                model = experiment["model"]
                samples = experiment["samples"]
                shot = experiment["shot"]
                experiment_tag, checkpoint_tag = build_effective_run_tags(
                    experiment,
                    args.experiment_tag,
                    train_seed,
                    multiple_training_seeds,
                )
                print(f"\n[{idx}/{total_runs}]", end=" ")
                success = run_experiment(
                    model=model,
                    shot=shot,
                    samples=samples,
                    dataset_path=dataset_spec["path"],
                    dataset_name=dataset_spec["name"],
                    project=args.project,
                    seed=train_seed,
                    selection_seed=args.selection_seed,
                    final_test_seed=args.final_test_seed,
                    gpu_id=args.gpu_id,
                    fewshot_backbone=args.fewshot_backbone,
                    num_workers=args.num_workers,
                    pin_memory=args.pin_memory,
                    persistent_workers=args.persistent_workers,
                    prefetch_factor=args.prefetch_factor,
                    spif_global_only=args.spif_global_only,
                    spif_local_only=args.spif_local_only,
                    test_protocol=dataset_spec["test_protocol"],
                    noise_test_root=dataset_spec.get("noise_test_root"),
                    noise_test_splits=args.noise_test_splits,
                    passthrough_args=args.passthrough_args,
                    variant_args=experiment["variant_args"],
                    experiment_tag=experiment_tag,
                    checkpoint_tag=checkpoint_tag,
                    experiment_label=experiment["experiment_label"],
                    extra_final_test_seeds=args.extra_final_test_seeds,
                    extra_test_protocols=merge_extra_test_protocols(
                        dataset_spec.get("extra_test_protocols"),
                        experiment.get("extra_test_protocols"),
                    ),
                    extra_noise_test_splits=experiment.get("extra_noise_test_splits"),
                    skip_existing=str(getattr(args, "skip_existing", "false")).lower() == "true",
                )
                if success:
                    success_count += 1
                else:
                    tag_suffix = f"_{experiment_tag}" if experiment_tag else f"_seed{train_seed}"
                    failed_experiments.append(
                        f"{dataset_spec['name']}_{model}_{shot}shot_"
                        f"{samples if samples else 'all'}samples{tag_suffix}"
                    )

    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"Total: {total_runs}")
    print(f"Success: {success_count}")
    print(f"Failed: {len(failed_experiments)}")
    if failed_experiments:
        print("\nFailed experiments:")
        for experiment in failed_experiments:
            print(f"  - {experiment}")

    if (
        args.spifce_ablation_suite == "none"
        and args.ours_ablation_suite == "none"
        and args.ours_final_ablation_suite == "none"
        and args.cfuget_ablation_suite == "none"
        and args.spot_ablation_suite == "none"
        and args.pare_ablation_suite == "none"
        and args.sgpot_ablation_suite == "none"
        and args.result_artifacts == "all"
    ):
        print("\n" + "=" * 60)
        print("Generating comparison charts...")
        print("=" * 60)
        for dataset_spec in dataset_specs:
            generate_comparison_charts(
                dataset_spec["name"],
                shots,
                requested_models,
                dataset_spec["test_protocol"],
                (
                    dataset_spec["noise_test_split_names"]
                    if dataset_spec["test_protocol"] == "noise"
                    else None
                ),
            )
    elif (
        args.spifce_ablation_suite == "none"
        and args.ours_ablation_suite == "none"
        and args.ours_final_ablation_suite == "none"
        and args.cfuget_ablation_suite == "none"
        and args.spot_ablation_suite == "none"
        and args.pare_ablation_suite == "none"
        and args.sgpot_ablation_suite == "none"
        and args.result_artifacts != "all"
    ):
        print("\n" + "=" * 60)
        print("Skipping comparison charts because result_artifacts=figures_only.")
        print("=" * 60)
    elif (
        args.spifce_ablation_suite != "none"
        or args.ours_ablation_suite != "none"
        or args.ours_final_ablation_suite != "none"
        or args.cfuget_ablation_suite != "none"
        or args.spot_ablation_suite != "none"
        or args.pare_ablation_suite != "none"
        or args.sgpot_ablation_suite != "none"
    ):
        print("\n" + "=" * 60)
        print("Skipping standard comparison charts for tagged ablation runs.")
        print("=" * 60)
    print("\nAll experiments completed!")


if __name__ == "__main__":
    main()
