"""Run CCDM marginal ablations for J-ECOT-M2 across SNR noise levels.

Variants:
  baseline   — M2 with uniform marginals (no CCDM)
  ccdm       — M2 + CCDM cross-class discriminative marginals
  ccdm_ew    — M2 + CCDM + entropy-weighted shot pooling
  mea        — M2 + MEA mutual-evidence attention marginals (reference)
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import math
from pathlib import Path
import re
import subprocess
from typing import Iterable

import run_all_experiments as run_all


DEFAULT_SEEDS = (42, 43, 44)
RESULTS_DIR = Path("results")


@dataclass(frozen=True)
class CCDMVariant:
    name: str
    model: str
    module: str
    description: str
    variant_args: list[str]


VARIANTS: tuple[CCDMVariant, ...] = (
    CCDMVariant(
        "baseline",
        "m2",
        "M2-baseline",
        "M2 with uniform marginals (no CCDM)",
        ["--hrot_ecot_enable_ccdm_marginal", "false"],
    ),
    CCDMVariant(
        "ccdm",
        "m2",
        "M2+CCDM",
        "M2 + cross-class discriminative marginals",
        [
            "--hrot_ecot_enable_ccdm_marginal", "true",
            "--hrot_ecot_ccdm_entropy_shot_weight", "0.0",
        ],
    ),
    CCDMVariant(
        "ccdm_ew",
        "m2",
        "M2+CCDM+EW",
        "M2 + CCDM + entropy-weighted shot pooling",
        [
            "--hrot_ecot_enable_ccdm_marginal", "true",
            "--hrot_ecot_ccdm_entropy_shot_weight", "1.0",
        ],
    ),
    CCDMVariant(
        "mea",
        "m2",
        "M2+MEA",
        "M2 + mutual-evidence attention marginals (reference)",
        [
            "--hrot_ecot_enable_ccdm_marginal", "false",
            "--hrot_ecot_enable_mea_marginal", "true",
        ],
    ),
)
VARIANT_BY_NAME = {v.name: v for v in VARIANTS}


def format_samples(samples: int | None) -> str:
    return "all" if samples is None else str(samples)


def format_samples_file(samples: int | None) -> str:
    return "allsamples" if samples is None else f"{samples}samples"


def default_n_train_text() -> str:
    return ",".join(format_samples(s) for s in run_all.SAMPLES_LIST)


def parse_train_values(text: str) -> list[int | None]:
    values: list[int | None] = []
    for raw in text.split(","):
        item = raw.strip().lower()
        if not item:
            continue
        values.append(None if item in {"all", "full", "none"} else int(item))
    return values


def parse_csv_ints(text: str, default: Iterable[int]) -> list[int]:
    values = [int(x.strip()) for x in text.split(",") if x.strip()]
    return values or list(default)


def result_path(
    dataset_name: str,
    model: str,
    samples: int | None,
    shot: int,
    tag: str,
    protocol: str,
    test_split_name: str | None = None,
) -> Path:
    protocol_suffix = run_all.result_protocol_suffix(protocol, test_split_name)
    filename = (
        f"results_{dataset_name}_{model}_{format_samples_file(samples)}"
        f"_{shot}shot_{tag}{protocol_suffix}.txt"
    )
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
        "timestamp", "variant", "model", "module",
        "n_train", "shot", "seed", "test_split",
        "acc_mean", "acc_std", "acc_95ci", "result_file",
    ]
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def fake_subprocess_run(cmd, check=False, **kwargs):
    print("DRY RUN:")
    print(subprocess.list2cmdline(cmd))
    return subprocess.CompletedProcess(cmd, 0)


def get_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="CCDM marginal ablation for M2 across SNR noise levels"
    )
    parser.add_argument("--project", type=str, default="pulse_fewshot")
    parser.add_argument("--dataset_path", type=str, default=run_all.default_dataset_path())
    parser.add_argument("--dataset_name", type=str, default="knee_aug_split")
    parser.add_argument("--test_protocol", type=str, default="noise", choices=["auto", "clean", "noise"])
    parser.add_argument("--noise_test_root", type=str, default=run_all.default_noise_test_root())
    parser.add_argument("--noise_test_splits", type=str, default="auto")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--pin_memory", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--persistent_workers", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument(
        "--fewshot_backbone", type=str, default="default",
        choices=["default", "resnet12", "conv64f", "fsl_mamba", "slim_mamba"],
    )
    parser.add_argument("--final_test_seed", type=int, default=run_all.DEFAULT_FINAL_TEST_SEED)
    parser.add_argument("--n_train", type=str, default=default_n_train_text())
    parser.add_argument("--shots", type=str, default=",".join(map(str, run_all.SHOTS_DEFAULT)))
    parser.add_argument("--seeds", type=str, default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument(
        "--variants", type=str,
        default=",".join(v.name for v in VARIANTS),
        help="Comma-separated subset of: " + ", ".join(v.name for v in VARIANTS),
    )
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--run_log", type=str, default="results/ccdm_ablation_runs.csv")
    args, passthrough = parser.parse_known_args()
    return args, passthrough


def main() -> int:
    args, passthrough = get_args()
    samples_list = parse_train_values(args.n_train)
    shots = parse_csv_ints(args.shots, run_all.SHOTS_DEFAULT)
    seeds = parse_csv_ints(args.seeds, DEFAULT_SEEDS)
    variant_names = [n.strip() for n in args.variants.split(",") if n.strip()]
    unknown = [n for n in variant_names if n not in VARIANT_BY_NAME]
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}. Valid: {sorted(VARIANT_BY_NAME)}")
    variants = [VARIANT_BY_NAME[n] for n in variant_names]

    effective_test_protocol = run_all.resolve_test_protocol(
        args.test_protocol, args.noise_test_root, args.noise_test_splits,
    )
    noise_split_names = (
        run_all.discover_noise_test_splits(args.noise_test_root, args.noise_test_splits)
        if effective_test_protocol == "noise"
        else []
    )

    total = len(variants) * len(samples_list) * len(shots) * len(seeds)
    print("=" * 72)
    print("CCDM marginal ablation suite for M2")
    print("=" * 72)
    print(f"Variants    : {', '.join(f'{v.name}={v.module}' for v in variants)}")
    print(f"Samples     : {', '.join(format_samples(s) for s in samples_list)}")
    print(f"Shots       : {', '.join(f'{s}-shot' for s in shots)}")
    print(f"Seeds       : {', '.join(map(str, seeds))}")
    print(f"Backbone    : {args.fewshot_backbone}")
    print(f"Test Proto  : {args.test_protocol} -> {effective_test_protocol}")
    if noise_split_names:
        print(f"SNR Splits  : {', '.join(noise_split_names)}")
    if passthrough:
        print(f"Forwarded   : {' '.join(passthrough)}")
    print(f"Total       : {total} run(s)")
    print("=" * 72)

    original_run = run_all.subprocess.run
    if args.dry_run:
        run_all.subprocess.run = fake_subprocess_run

    failures: list[str] = []
    try:
        for variant in variants:
            for samples in samples_list:
                for shot in shots:
                    for seed in seeds:
                        sample_label = format_samples(samples)
                        tag = f"ccdm_{variant.name}_n{sample_label}_{shot}shot_s{seed}"
                        label = f"{variant.name} | n_train={sample_label} | {shot}-shot | seed={seed}"
                        print(f"\n{label}")
                        success = run_all.run_experiment(
                            model=variant.model,
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
                            variant_args=variant.variant_args,
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

                        parse_targets = (
                            [
                                (
                                    split_name,
                                    result_path(
                                        args.dataset_name, variant.model, samples,
                                        shot, tag, effective_test_protocol, split_name,
                                    ),
                                )
                                for split_name in noise_split_names
                            ]
                            if effective_test_protocol == "noise"
                            else [
                                (
                                    "",
                                    result_path(
                                        args.dataset_name, variant.model, samples,
                                        shot, tag, effective_test_protocol,
                                    ),
                                )
                            ]
                        )
                        for test_split, out_path in parse_targets:
                            try:
                                acc_mean, acc_std, acc_95ci = parse_accuracy(out_path)
                            except Exception as exc:
                                failures.append(f"{label} {test_split or ''} ({exc})".strip())
                                if not args.continue_on_error:
                                    raise
                                continue
                            row = {
                                "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
                                "variant": variant.name,
                                "model": variant.model,
                                "module": variant.module,
                                "n_train": sample_label,
                                "shot": shot,
                                "seed": seed,
                                "test_split": test_split,
                                "acc_mean": f"{acc_mean:.6f}",
                                "acc_std": f"{acc_std:.6f}",
                                "acc_95ci": f"{acc_95ci:.6f}",
                                "result_file": str(out_path),
                            }
                            append_run_log(Path(args.run_log), row)
                            split_text = f", split={test_split}" if test_split else ""
                            print(
                                f"Parsed accuracy{split_text}: "
                                f"mean={acc_mean:.4f}, std={acc_std:.4f}, 95ci={acc_95ci:.4f}"
                            )
    finally:
        if args.dry_run:
            run_all.subprocess.run = original_run

    if args.dry_run:
        print("\nDry run only; no run log was written.")
        return 0

    if failures:
        print("Failures:")
        for failure in failures:
            print(f"  - {failure}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
