"""Run statistical-feature baselines for 60 training samples and all training samples.

Mirrors ``run_all_experiments.EXPERIMENT_MODES`` modes 1 (60) and 4 (all) only.

For a **single file** with 60 + all and 3 moderate-noise test splits (SVM/MLP/RF), use::

    python bench_statistical_classifiers_mat.py --consolidated_noise_report \\
        --dataset_path .../pulse27_1_mat_from_scalogram_27_1 \\
        --noise_test_root .../pulse27_1_mat_from_scalogram_27_1_pd_noise_benchmark_test_moderate
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Batch classical ML on pulse .mat (60 + all train)")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=str(
            Path(__file__).resolve().parent.parent.parent
            / "dataset"
            / "pulse27_1_mat_from_scalogram_27_1"
        ),
    )
    parser.add_argument("--dataset_name", type=str, default="pulse27_1_mat")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--classifiers", type=str, default="svm,mlp,rf")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument(
        "--samples",
        type=str,
        default="60,all",
        help='Comma list: e.g. "60,all" or single "240"',
    )
    parser.add_argument(
        "--noise_test_root",
        type=str,
        default=None,
        help="Forwarded to bench script (SNR benchmark root).",
    )
    parser.add_argument(
        "--noise_only",
        action="store_true",
        help="Forwarded to bench script: only evaluate noise splits.",
    )
    args, passthrough = parser.parse_known_args()

    script = Path(__file__).resolve().parent / "bench_statistical_classifiers_mat.py"
    tokens = [t.strip().lower() for t in args.samples.split(",") if t.strip()]
    runs = []
    for t in tokens:
        if t == "all":
            runs.append(None)
        else:
            runs.append(int(t))

    for training_samples in runs:
        label = "allsamples" if training_samples is None else f"{training_samples}samples"
        print("\n" + "=" * 72)
        print(f"Classical mat benchmarks: {label}")
        print("=" * 72)

        cmd = [
            sys.executable,
            str(script),
            "--dataset_path",
            args.dataset_path,
            "--dataset_name",
            args.dataset_name,
            "--seed",
            str(args.seed),
            "--classifiers",
            args.classifiers,
            "--results_dir",
            args.results_dir,
        ]
        if training_samples is not None:
            cmd.extend(["--training_samples", str(training_samples)])
        if args.noise_test_root:
            cmd.extend(["--noise_test_root", args.noise_test_root])
        if args.noise_only:
            cmd.append("--noise_only")
        cmd.extend(passthrough)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
