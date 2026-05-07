"""Run J-ECOT module ablations through the exact run_all_experiments protocol."""

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
DEFAULT_RHO_BANK = "0.45,0.60,0.75,0.80,0.90"
RESULTS_DIR = Path("results")


@dataclass(frozen=True)
class AblationVariant:
    name: str
    module: str
    description: str
    extra_args: tuple[str, ...] = ()


VARIANTS: tuple[AblationVariant, ...] = (
    AblationVariant("full_jecot", "full", "Full J-ECOT reference"),
    AblationVariant(
        "abl_m1_proto",
        "M1",
        "Pool support shots before transport",
        ("--hrot_pre_transport_shot_pool", "true"),
    ),
    AblationVariant(
        "abl_m2_single",
        "M2",
        "Single base-rho expert",
        ("--hrot_ecot_rho_bank", "0.80"),
    ),
    AblationVariant(
        "abl_m2_ncet",
        "M2+",
        "Single base-rho expert with nuisance-decoupled transport",
        (
            "--hrot_variant",
            "J_NCET",
            "--hrot_fixed_mass",
            "0.80",
            "--hrot_ncet_mix_init",
            "0.25",
            "--hrot_ncet_real_penalty_init",
            "0.25",
            "--hrot_ncet_null_penalty_init",
            "0.05",
            "--hrot_ncet_sink_cost_init",
            "1.0",
        ),
    ),
    AblationVariant(
        "abl_m3_uniform",
        "M3",
        "Uniform budget policy",
        ("--hrot_ecot_uniform_budget_policy", "true"),
    ),
    AblationVariant(
        "abl_m4_noanchor",
        "M4",
        "No base-anchored identity start",
        ("--hrot_ecot_lambda_init", "8.0", "--hrot_ecot_identity_reg", "0.0"),
    ),
    AblationVariant(
        "abl_m5_meanpool",
        "M5",
        "Fixed tau-shot pooling",
        ("--hrot_ecot_enable_tau_shot", "false"),
    ),
)
VARIANT_BY_NAME = {variant.name: variant for variant in VARIANTS}


def format_samples(samples: int | None) -> str:
    return "all" if samples is None else str(samples)


def format_samples_file(samples: int | None) -> str:
    return "allsamples" if samples is None else f"{samples}samples"


def default_n_train_text() -> str:
    return ",".join(format_samples(samples) for samples in run_all.SAMPLES_LIST)


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


def get_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Run J-ECOT ablations with run_all_experiments setup")
    parser.add_argument("--project", type=str, default="pulse_fewshot")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/workspace/pd_fewshot/scalogram_27_1_hrot_robust",
    )
    parser.add_argument("--dataset_name", type=str, default="knee_aug_split")
    parser.add_argument("--test_protocol", type=str, default="auto", choices=["auto", "clean", "robust"])
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
        default=",".join(variant.name for variant in VARIANTS),
        help="Comma-separated subset of: " + ", ".join(variant.name for variant in VARIANTS),
    )
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--run_log", type=str, default="results/jecot_ablation_runs.csv")
    parser.add_argument("--summary_csv", type=str, default="results/jecot_ablation_summary.csv")
    parser.add_argument("--summary_md", type=str, default="results/jecot_ablation_summary.md")
    args, passthrough = parser.parse_known_args()
    return args, passthrough


def base_jecot_args() -> list[str]:
    return [
        "--hrot_variant",
        "J_ECOT",
        "--hrot_pre_transport_shot_pool",
        "false",
        "--hrot_ecot_rho_bank",
        DEFAULT_RHO_BANK,
        "--hrot_ecot_base_rho",
        "0.80",
        "--hrot_ecot_lambda_init",
        "-8.0",
        "--hrot_ecot_identity_reg",
        "1e-4",
        "--hrot_ecot_uniform_budget_policy",
        "false",
        "--hrot_ecot_enable_tau_shot",
        "true",
    ]


def result_path(dataset_name: str, samples: int | None, shot: int, tag: str, protocol: str) -> Path:
    protocol_suffix = run_all.result_protocol_suffix(protocol)
    filename = f"results_{dataset_name}_hrot_fsl_{format_samples_file(samples)}_{shot}shot_{tag}{protocol_suffix}.txt"
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
    lines = ["# J-ECOT Ablation Summary", ""]
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

        drops: list[str] = []
        for label in sample_labels:
            full = summary.get((shot, label, "full_jecot"))
            if full is None:
                continue
            full_acc = float(full["acc_seed_mean"])
            best: tuple[float, AblationVariant] | None = None
            for variant in variants:
                if variant.name == "full_jecot":
                    continue
                item = summary.get((shot, label, variant.name))
                if item is None:
                    continue
                drop = full_acc - float(item["acc_seed_mean"])
                if best is None or drop > best[0]:
                    best = (drop, variant)
            if best is not None:
                drops.append(f"- n_train={label}: {best[1].module} ({best[1].name}), drop={best[0]:.4f}")
        if drops:
            lines.extend(["Largest drop vs full:", *drops, ""])
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
    variant_names = [name.strip() for name in args.variants.split(",") if name.strip()]
    unknown = [name for name in variant_names if name not in VARIANT_BY_NAME]
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}. Valid variants: {sorted(VARIANT_BY_NAME)}")
    variants = [VARIANT_BY_NAME[name] for name in variant_names]

    effective_test_protocol = run_all.resolve_test_protocol(args.test_protocol, args.dataset_path)
    if effective_test_protocol == "robust" and not run_all.dataset_has_hrot_robust_layout(args.dataset_path):
        raise ValueError("--test_protocol robust requires train/val/test_clean and all HROT robust test pools.")

    total = len(variants) * len(samples_list) * len(shots) * len(seeds)
    print("=" * 72)
    print("J-ECOT ablation suite via run_all_experiments.run_experiment")
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
                        tag = f"jecot_{variant.name}_n{sample_label}_{shot}shot_s{seed}"
                        variant_args = [*base_jecot_args(), *variant.extra_args]
                        label = f"{variant.name} | n_train={sample_label} | {shot}-shot | seed={seed}"
                        print(f"\n{label}")

                        success = run_all.run_experiment(
                            model="hrot_fsl",
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
    print("J-ECOT ablation summary")
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
