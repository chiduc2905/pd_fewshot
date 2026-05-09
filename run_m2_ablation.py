"""Run M2 as a standalone model suite through run_all_experiments protocol."""

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
class M2Variant:
    name: str
    model: str
    module: str
    description: str


VARIANTS: tuple[M2Variant, ...] = (
    M2Variant(
        "uot",
        "m2",
        "SB-ECOT-UOT",
        "single-budget KL-relaxed UOT at rho=0.80",
    ),
    M2Variant(
        "full_ot",
        "m2_full_ot",
        "SB-ECOT-FullOT",
        "same shot-decomposed evidence module with balanced full-mass OT at rho=1.0",
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
        values.append(None if item in {"all", "full", "none"} else int(item))
    return values


def parse_csv_ints(text: str, default: Iterable[int]) -> list[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
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
    filename = f"results_{dataset_name}_{model}_{format_samples_file(samples)}_{shot}shot_{tag}{protocol_suffix}.txt"
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
        "model",
        "module",
        "n_train",
        "shot",
        "seed",
        "test_split",
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


def fake_subprocess_run(cmd, check=False, **kwargs):
    print("DRY RUN:")
    print(subprocess.list2cmdline(cmd))
    return subprocess.CompletedProcess(cmd, 0)


def get_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Run standalone M2 UOT vs full-OT ablations")
    parser.add_argument("--project", type=str, default="pulse_fewshot")
    parser.add_argument("--dataset_path", type=str, default=run_all.default_dataset_path())
    parser.add_argument("--dataset_name", type=str, default="knee_aug_split")
    parser.add_argument("--test_protocol", type=str, default="auto", choices=["auto", "clean", "noise"])
    parser.add_argument("--noise_test_root", type=str, default=run_all.default_noise_test_root())
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
        default=",".join(variant.name for variant in VARIANTS),
        help="Comma-separated subset of: " + ", ".join(variant.name for variant in VARIANTS),
    )
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--run_log", type=str, default="results/m2_ablation_runs.csv")
    args, passthrough = parser.parse_known_args()
    return args, passthrough


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
    print("M2 standalone ablation suite via run_all_experiments.run_experiment")
    print("=" * 72)
    print("Protocol    : identical to run_all_experiments.py")
    print("Module      : SB-ECOT = shot-decomposed single-budget episode-calibrated transport")
    print(f"Variants    : {', '.join(f'{v.name}={v.model}' for v in variants)}")
    print(f"Samples     : {', '.join(format_samples(samples) for samples in samples_list)}")
    print(f"Shots       : {', '.join(f'{shot}-shot' for shot in shots)}")
    print(f"Seeds       : {', '.join(map(str, seeds))}")
    print(f"Backbone    : {args.fewshot_backbone}")
    print(f"Test Proto  : {args.test_protocol} -> {effective_test_protocol}")
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
                        tag = f"m2_{variant.name}_n{sample_label}_{shot}shot_s{seed}"
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
                            variant_args=None,
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
                                        args.dataset_name,
                                        variant.model,
                                        samples,
                                        shot,
                                        tag,
                                        effective_test_protocol,
                                        split_name,
                                    ),
                                )
                                for split_name in noise_split_names
                            ]
                            if effective_test_protocol == "noise"
                            else [
                                (
                                    "",
                                    result_path(
                                        args.dataset_name,
                                        variant.model,
                                        samples,
                                        shot,
                                        tag,
                                        effective_test_protocol,
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
