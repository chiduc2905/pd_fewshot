"""Run EC-MROT ablations through the run_all_experiments protocol."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
import math
from pathlib import Path
import re
import statistics
import subprocess
from typing import Iterable

import run_all_experiments as run_all


DEFAULT_SEEDS = (42, 43, 44)
DEFAULT_MASS_RESPONSE_GRID = "0.40,0.55,0.70,0.85,0.95"
RESULTS_DIR = Path("results")
CORE_VARIANTS = ("full", "m1", "m2", "m3", "m4", "m5")


@dataclass(frozen=True)
class AblationVariant:
    name: str
    module: str
    description: str
    extra_args: tuple[str, ...] = ()


VARIANTS: tuple[AblationVariant, ...] = (
    AblationVariant("full", "Full", "Full EC-MROT"),
    AblationVariant(
        "m1",
        "M1",
        "Architectural control: pre-transport prototype pooling",
        ("--hrot_pre_transport_shot_pool", "true"),
    ),
    AblationVariant(
        "m2",
        "M2",
        "Mass-response path collapsed to a single base budget rho=0.80",
        ("--ec_mrot_mass_response_grid", "0.80", "--hrot_ecot_base_rho", "0.80"),
    ),
    AblationVariant(
        "m3",
        "M3",
        "Class-competitive response centering removed",
        ("--ec_mrot_competitive_center", "false"),
    ),
    AblationVariant(
        "m4",
        "M4",
        "Anchor-free response functional disabled",
        (
            "--ec_mrot_response_strength_init",
            "0.0005",
            "--ec_mrot_response_strength_max",
            "0.001",
            "--hrot_ecot_identity_reg",
            "0.0",
        ),
    ),
    AblationVariant(
        "m5",
        "M5",
        "Entropic support-risk aggregation fixed",
        ("--hrot_ecot_enable_tau_shot", "false"),
    ),
    AblationVariant(
        "learnable_grid",
        "OPT-grid",
        "Learnable constrained quadrature grid",
        ("--ec_mrot_learn_response_grid", "true", "--ec_mrot_grid_spacing_reg", "0.1"),
    ),
    AblationVariant(
        "no_budget_prior",
        "OPT-prior",
        "No budget-measure KL prior",
        ("--ec_mrot_budget_kl_reg", "0.0"),
    ),
    AblationVariant(
        "no_homotopy_schedule",
        "OPT-schedule",
        "No homotopy schedule",
        ("--ec_mrot_homotopy_schedule", "none"),
    ),
)

VARIANT_BY_NAME = {variant.name: variant for variant in VARIANTS}
VARIANT_ALIASES = {
    "all": ",".join(CORE_VARIANTS),
    "full_ec_mrot": "full",
    "m1_proto": "m1",
    "m1_architectural_control": "m1",
    "m2_single": "m2",
    "m2_single_budget": "m2",
    "m3_uniform": "m3",
    "m3_uniform_measure": "m3",
    "m3_no_center": "m3",
    "m3_no_competitive_center": "m3",
    "m4_no_homotopy": "m4",
    "m4_no_residual": "m4",
    "m5_fixed": "m5",
    "m5_fixed_support_risk": "m5",
}


def format_samples(samples: int | None) -> str:
    return "all" if samples is None else str(samples)


def format_samples_file(samples: int | None) -> str:
    return "allsamples" if samples is None else f"{samples}samples"


def default_n_train_text() -> str:
    return ",".join(format_samples(samples) for samples in run_all.SAMPLES_LIST)


def default_dataset_path() -> str:
    if hasattr(run_all, "default_dataset_path"):
        return run_all.default_dataset_path()
    return "/workspace/pd_fewshot/scalogram_27_1"


def default_noise_test_root() -> str | None:
    if hasattr(run_all, "default_noise_test_root"):
        return run_all.default_noise_test_root()
    return None


def parse_train_values(text: str) -> list[int | None]:
    values: list[int | None] = []
    for raw_item in text.split(","):
        item = raw_item.strip().lower()
        if not item:
            continue
        if item in {"all", "full", "none"}:
            values.append(None)
        else:
            values.append(int(item))
    return values


def parse_csv_ints(text: str, default: Iterable[int]) -> list[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    return values or list(default)


def expand_variant_names(text: str) -> list[str]:
    names: list[str] = []
    for raw_item in text.split(","):
        item = raw_item.strip().lower()
        if not item:
            continue
        item = VARIANT_ALIASES.get(item, item)
        if "," in item:
            names.extend(expand_variant_names(item))
        else:
            names.append(item)
    return names


def get_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Run EC-MROT ablations with run_all_experiments setup")
    parser.add_argument("--project", type=str, default="pulse_fewshot")
    parser.add_argument("--dataset_path", type=str, default=default_dataset_path())
    parser.add_argument("--dataset_name", type=str, default="knee_aug_split")
    parser.add_argument("--test_protocol", type=str, default="auto", choices=["auto", "clean", "noise"])
    parser.add_argument("--noise_test_root", type=str, default=default_noise_test_root())
    parser.add_argument("--noise_test_splits", type=str, default="auto")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--pin_memory", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--persistent_workers", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument(
        "--fewshot_backbone",
        type=str,
        default="default",
        choices=["default", "resnet12", "conv64f", "fsl_mamba", "slim_mamba"],
    )
    parser.add_argument("--final_test_seed", type=int, default=run_all.DEFAULT_FINAL_TEST_SEED)
    parser.add_argument("--n_train", type=str, default=default_n_train_text())
    parser.add_argument("--shots", type=str, default=",".join(map(str, run_all.SHOTS_DEFAULT)))
    parser.add_argument("--seeds", type=str, default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument(
        "--variants",
        type=str,
        default="all",
        help="Comma-separated subset of: " + ", ".join(VARIANT_BY_NAME),
    )
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--run_log", type=str, default="results/ec_mrot_ablation_runs.csv")
    parser.add_argument("--summary_csv", type=str, default="results/ec_mrot_ablation_summary.csv")
    parser.add_argument("--summary_md", type=str, default="results/ec_mrot_ablation_summary.md")
    args, passthrough = parser.parse_known_args()
    return args, passthrough


def base_ec_mrot_args() -> list[str]:
    return [
        "--ec_mrot_response_mode",
        "anchor_free_functional",
        "--hrot_pre_transport_shot_pool",
        "false",
        "--ec_mrot_mass_response_grid",
        DEFAULT_MASS_RESPONSE_GRID,
        "--hrot_ecot_lambda_init",
        "-8.0",
        "--hrot_ecot_identity_reg",
        "1e-4",
        "--hrot_ecot_uniform_budget_policy",
        "false",
        "--hrot_ecot_enable_tau_shot",
        "true",
        "--ec_mrot_learn_response_grid",
        "false",
        "--ec_mrot_budget_prior",
        "uniform",
        "--ec_mrot_budget_kl_reg",
        "0.0",
        "--ec_mrot_budget_entropy_reg",
        "0.0",
        "--ec_mrot_homotopy_schedule",
        "none",
        "--ec_mrot_response_strength_init",
        "0.35",
        "--ec_mrot_response_strength_max",
        "1.0",
        "--ec_mrot_response_temperature",
        "1.0",
        "--ec_mrot_competitive_center",
        "true",
        "--ec_mrot_local_response_gain",
        "2.0",
        "--ec_mrot_episode_prior_gain",
        "0.10",
        "--ec_mrot_margin_temperature",
        "1.0",
        "--ec_mrot_stability_temperature",
        "1.0",
        "--ec_mrot_residual_gate_floor",
        "0.15",
    ]


def result_path(dataset_name: str, samples: int | None, shot: int, tag: str, protocol: str) -> Path:
    protocol_suffix = run_all.result_protocol_suffix(protocol)
    filename = f"results_{dataset_name}_ec_mrot_{format_samples_file(samples)}_{shot}shot_{tag}{protocol_suffix}.txt"
    return RESULTS_DIR / filename


def parse_accuracy(path: Path) -> tuple[float, float, float]:
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"Accuracy\s*:\s*([0-9.]+)\s*(?:\+/-|\u00b1)\s*([0-9.]+)", text)
    if not match:
        raise ValueError(f"Could not parse Accuracy from {path}")
    acc_mean = float(match.group(1))
    acc_std = float(match.group(2))
    acc_95ci = 1.96 * acc_std / math.sqrt(float(run_all.TEST_EPISODES_PER_EPOCH))
    return acc_mean, acc_std, acc_95ci


def append_run_log(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "timestamp",
        "variant",
        "module",
        "n_train",
        "shot",
        "seed",
        "acc_mean",
        "acc_std",
        "acc_95ci",
        "result_file",
    ]
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def aggregate_rows(rows: list[dict[str, object]]) -> dict[tuple[int, str, str], dict[str, object]]:
    grouped: dict[tuple[int, str, str], list[float]] = defaultdict(list)
    modules: dict[str, str] = {}
    for row in rows:
        key = (int(row["shot"]), str(row["n_train"]), str(row["variant"]))
        grouped[key].append(float(row["acc_mean"]))
        modules[str(row["variant"])] = str(row["module"])

    summary: dict[tuple[int, str, str], dict[str, object]] = {}
    for key, values in grouped.items():
        shot, n_train, variant = key
        summary[key] = {
            "shot": shot,
            "n_train": n_train,
            "variant": variant,
            "module": modules[variant],
            "seed_count": len(values),
            "acc_seed_mean": statistics.mean(values),
            "acc_seed_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        }
    return summary


def write_summary_csv(path: Path, summary: dict[tuple[int, str, str], dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["shot", "n_train", "variant", "module", "seed_count", "acc_seed_mean", "acc_seed_std"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key in sorted(summary):
            writer.writerow(summary[key])


def write_summary_md(
    path: Path,
    summary: dict[tuple[int, str, str], dict[str, object]],
    shots: list[int],
    samples_list: list[int | None],
    variants: list[AblationVariant],
) -> None:
    sample_labels = [format_samples(samples) for samples in samples_list]
    lines = ["# EC-MROT Ablation Summary", ""]
    for shot in shots:
        lines.extend([f"## {shot}-shot", ""])
        lines.append("| Variant | " + " | ".join(sample_labels) + " |")
        lines.append("|---|" + "|".join("---:" for _ in sample_labels) + "|")
        for variant in variants:
            cells = [variant.name]
            for label in sample_labels:
                item = summary.get((shot, label, variant.name))
                cells.append(
                    "" if item is None else f"{float(item['acc_seed_mean']):.4f} +/- {float(item['acc_seed_std']):.4f}"
                )
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def fake_subprocess_run(cmd, check=False, **kwargs):
    print("DRY RUN:")
    print(subprocess.list2cmdline(cmd))
    return subprocess.CompletedProcess(cmd, 0)


def main() -> int:
    args, passthrough = get_args()
    samples_list = parse_train_values(args.n_train)
    shots = parse_csv_ints(args.shots, run_all.SHOTS_DEFAULT)
    seeds = parse_csv_ints(args.seeds, DEFAULT_SEEDS)
    variant_names = expand_variant_names(args.variants)
    unknown = [name for name in variant_names if name not in VARIANT_BY_NAME]
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}. Valid variants: {sorted(VARIANT_BY_NAME)}")
    variants = [VARIANT_BY_NAME[name] for name in variant_names]

    effective_test_protocol = run_all.resolve_test_protocol(
        args.test_protocol,
        args.noise_test_root,
        args.noise_test_splits,
    )
    noise_split_names = (
        run_all.discover_noise_test_splits(args.noise_test_root, args.noise_test_splits)
        if effective_test_protocol == "noise"
        else []
    )

    total = len(variants) * len(samples_list) * len(shots) * len(seeds)
    print("=" * 72)
    print("EC-MROT ablation suite via run_all_experiments.run_experiment")
    print("=" * 72)
    print("Protocol    : identical to run_all_experiments.py")
    print(
        "Fixed setup : "
        f"way=4, query(train/val/test)=1/1/1, "
        f"episodes(train/val/test)="
        f"{run_all.TRAIN_EPISODES_PER_EPOCH}/{run_all.TEST_EPISODES_PER_EPOCH}/"
        f"{run_all.TEST_EPISODES_PER_EPOCH}, epochs=100"
    )
    print(f"Samples     : {', '.join(format_samples(samples) for samples in samples_list)}")
    print(f"Shots       : {', '.join(f'{shot}-shot' for shot in shots)}")
    print(f"Seeds       : {', '.join(map(str, seeds))}")
    print(f"Variants    : {', '.join(variant.name for variant in variants)}")
    print(f"Backbone    : {args.fewshot_backbone}")
    print(f"Test Proto  : {args.test_protocol} -> {effective_test_protocol}")
    if effective_test_protocol == "noise":
        print(f"Noise Root  : {args.noise_test_root}")
        print(f"Noise Splits: {', '.join(noise_split_names)}")
    print(f"Total       : {total} run(s)")
    if passthrough:
        print(f"Forwarded   : {' '.join(passthrough)}")
    print("=" * 72)

    original_run = run_all.subprocess.run
    if args.dry_run:
        run_all.subprocess.run = fake_subprocess_run

    rows: list[dict[str, object]] = []
    failures: list[str] = []
    try:
        for variant in variants:
            for samples in samples_list:
                for shot in shots:
                    for seed in seeds:
                        sample_label = format_samples(samples)
                        tag = f"ecmrot_{variant.name}_n{sample_label}_{shot}shot_s{seed}"
                        variant_args = [*base_ec_mrot_args(), *variant.extra_args]
                        label = f"{variant.name} | n_train={sample_label} | {shot}-shot | seed={seed}"
                        print(f"\n{label}")

                        success = run_all.run_experiment(
                            model="ec_mrot",
                            shot=shot,
                            samples=samples,
                            dataset_path=args.dataset_path,
                            dataset_name=args.dataset_name,
                            project=args.project,
                            seed=seed,
                            final_test_seed=args.final_test_seed,
                            gpu_id=args.gpu_id,
                            fewshot_backbone=args.fewshot_backbone,
                            num_workers=args.num_workers,
                            pin_memory=args.pin_memory,
                            persistent_workers=args.persistent_workers,
                            prefetch_factor=args.prefetch_factor,
                            spif_global_only="false",
                            spif_local_only="false",
                            test_protocol=effective_test_protocol,
                            noise_test_root=args.noise_test_root,
                            noise_test_splits=args.noise_test_splits,
                            passthrough_args=passthrough,
                            variant_args=variant_args,
                            experiment_tag=tag,
                            experiment_label=f"{variant.module}: {variant.description}",
                        )
                        if args.dry_run:
                            continue
                        if not success:
                            failures.append(label)
                            if not args.continue_on_error:
                                return 1
                            continue

                        out_path = result_path(args.dataset_name, samples, shot, tag, effective_test_protocol)
                        try:
                            acc_mean, acc_std, acc_95ci = parse_accuracy(out_path)
                        except Exception as exc:
                            failures.append(f"{label} ({exc})")
                            if not args.continue_on_error:
                                raise
                            continue

                        row = {
                            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
                            "variant": variant.name,
                            "module": variant.module,
                            "n_train": sample_label,
                            "shot": shot,
                            "seed": seed,
                            "acc_mean": f"{acc_mean:.6f}",
                            "acc_std": f"{acc_std:.6f}",
                            "acc_95ci": f"{acc_95ci:.6f}",
                            "result_file": str(out_path),
                        }
                        rows.append(row)
                        append_run_log(Path(args.run_log), row)
                        print(f"Parsed accuracy: mean={acc_mean:.4f}, std={acc_std:.4f}, 95ci={acc_95ci:.4f}")
    finally:
        if args.dry_run:
            run_all.subprocess.run = original_run

    if args.dry_run:
        print("\nDry run only; no summaries were written.")
        return 0

    summary = aggregate_rows(rows)
    write_summary_csv(Path(args.summary_csv), summary)
    write_summary_md(Path(args.summary_md), summary, shots, samples_list, variants)

    print("\n" + "=" * 72)
    print("EC-MROT ablation summary")
    print("=" * 72)
    print(f"Run log     : {args.run_log}")
    print(f"Summary CSV : {args.summary_csv}")
    print(f"Summary MD  : {args.summary_md}")
    if failures:
        print("Failures:")
        for failure in failures:
            print(f"  - {failure}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
