"""Classical ML baselines on pulse .mat data (same folder layout as scalogram).

Features (8): mean, std, variance, skewness, kurtosis, RMS, peak (max |x|), energy (sum x^2)
computed on the primary signal array in each .mat (Trace_3_VOLT / Voltage, or flattened image).

Training subset: optional balanced subsample of the train split (same rule as main.py
``--training_samples`` with torch.Generator(seed)).

Optional ``--noise_test_root``: evaluate on each ``test_snr*`` folder (same discovery rules as
``run_all_experiments`` noise protocol). Use ``--noise_only`` to skip ``dataset_path/test``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
from scipy.stats import kurtosis, skew
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from dataset import CLASS_MAP, CLASS_ORDER, resolve_class_dirs


FEATURE_NAMES = ("mean", "std", "variance", "skewness", "kurtosis", "rms", "peak", "energy")


def list_mat_files(folder: str) -> list[str]:
    return sorted(
        f
        for f in os.listdir(folder)
        if f.lower().endswith(".mat")
        and "labeled" not in f.lower()
        and "labeled:" not in f
    )


def collect_mat_split_files(split_path: str) -> list[tuple[str, int]]:
    split_files: list[tuple[str, int]] = []
    class_dirs = resolve_class_dirs(split_path)
    for class_name in CLASS_ORDER:
        class_path = class_dirs.get(class_name)
        if class_path is None:
            print(f"Warning: Class folder not found: {split_path}/{class_name}")
            continue
        files = list_mat_files(class_path)
        label = CLASS_MAP[class_name]
        split_files.extend((os.path.join(class_path, f), label) for f in files)
    return split_files


def load_signal_from_mat(mat_path: str) -> np.ndarray:
    mat = sio.loadmat(mat_path)
    keys = [k for k in mat if not k.startswith("__")]

    if "Trace_3_VOLT" in mat:
        return np.asarray(mat["Trace_3_VOLT"], dtype=np.float64).ravel()
    if "Voltage" in mat:
        return np.asarray(mat["Voltage"], dtype=np.float64).ravel()

    image_keys = ("scalogram", "cwt", "CWT", "image", "img", "S", "spectrogram")
    for k in image_keys:
        if k in mat:
            return np.asarray(mat[k], dtype=np.float64).ravel()

    numeric = []
    for k in keys:
        arr = np.asarray(mat[k], dtype=np.float64)
        if arr.size == 0:
            continue
        if arr.ndim <= 2 and arr.size >= 16:
            numeric.append(arr.ravel())
    if len(numeric) == 1:
        return numeric[0]
    raise ValueError(f"Could not resolve signal from {mat_path}; keys={keys}")


def statistical_features(signal: np.ndarray) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float64).ravel()
    if x.size == 0:
        raise ValueError("empty signal")
    m = float(np.mean(x))
    s = float(np.std(x, ddof=0))
    v = float(np.var(x, ddof=0))
    sk = float(skew(x, bias=False))
    ku = float(kurtosis(x, bias=False, fisher=True))
    if not np.isfinite(sk):
        sk = 0.0
    if not np.isfinite(ku):
        ku = 0.0
    rms = float(np.sqrt(np.mean(x**2)))
    peak = float(np.max(np.abs(x)))
    energy = float(np.sum(x**2))
    for name, val in zip(FEATURE_NAMES, (m, s, v, sk, ku, rms, peak, energy)):
        if not np.isfinite(val):
            raise ValueError(f"non-finite feature {name}={val} (check input signal)")
    return np.array([m, s, v, sk, ku, rms, peak, energy], dtype=np.float64)


def build_feature_matrix(file_label_pairs: list[tuple[str, int]]):
    X_list, y_list = [], []
    for path, label in file_label_pairs:
        sig = load_signal_from_mat(path)
        X_list.append(statistical_features(sig))
        y_list.append(label)
    return np.stack(X_list, axis=0), np.array(y_list, dtype=np.int64)


def subsample_train_balanced(
    X: np.ndarray, y: np.ndarray, training_samples: int | None, seed: int, way_num: int
):
    if training_samples is None:
        return X, y
    if training_samples % way_num != 0:
        raise ValueError(
            f"training_samples ({training_samples}) must be divisible by way_num ({way_num})"
        )
    per_class = training_samples // way_num
    X_parts, y_parts = [], []
    for class_id in range(way_num):
        idx = np.flatnonzero(y == class_id)
        if idx.size < per_class:
            raise ValueError(f"Class {class_id}: need {per_class}, have {idx.size}")
        generator = torch.Generator().manual_seed(int(seed))
        perm = torch.randperm(int(idx.size), generator=generator).numpy()[:per_class]
        chosen = idx[perm]
        X_parts.append(X[chosen])
        y_parts.append(y[chosen])
    return np.concatenate(X_parts, axis=0), np.concatenate(y_parts, axis=0)


def make_classifier(name: str, seed: int):
    name = name.strip().lower()
    if name in ("svm", "svc"):
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="rbf", gamma="scale", random_state=seed)),
            ]
        )
    if name in ("ann", "mlp", "mlpclassifier"):
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    MLPClassifier(
                        hidden_layer_sizes=(128, 64),
                        max_iter=2000,
                        random_state=seed,
                        early_stopping=True,
                        validation_fraction=0.1,
                        n_iter_no_change=30,
                    ),
                ),
            ]
        )
    if name in ("rf", "random_forest", "randomforest"):
        return RandomForestClassifier(
            n_estimators=300,
            random_state=seed,
            n_jobs=-1,
        )
    raise ValueError(f"Unknown classifier {name!r}")


def parse_classifiers(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def sanitize_result_tag(value: str) -> str:
    tag = re.sub(r"[^A-Za-z0-9]+", "_", str(value).strip()).strip("_").lower()
    return tag or "test"


def split_has_class_dirs(split_path: str) -> bool:
    expected = {"surface", "internal", "corona", "notpd", "nopd", "not_pd"}
    p = Path(split_path)
    if not p.is_dir():
        return False
    child_dirs = {child.name.lower() for child in p.iterdir() if child.is_dir()}
    return bool(child_dirs & expected)


def snr_sort_key(split_name: str):
    name = str(split_name).lower()
    match = re.search(r"snr(?:_minus)?(\d+)db", name)
    if not match:
        return (1, name)
    value = int(match.group(1))
    if "snr_minus" in name:
        value = -value
    return (0, value, name)


def discover_noise_test_splits(noise_test_root: str | None, requested_splits: str = "auto") -> list[str]:
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


def clean_stat_result_files(results_dir: str, dataset_name: str) -> list[str]:
    """Remove prior results_{dataset_name}_stat_* files (txt/json)."""
    rd = Path(results_dir)
    if not rd.is_dir():
        return []
    removed: list[str] = []
    for pattern in (f"results_{dataset_name}_stat_*.txt", f"results_{dataset_name}_stat_*.json"):
        for p in rd.glob(pattern):
            p.unlink()
            removed.append(str(p))
    return removed


def _normalize_clf_tag(clf_name: str) -> str:
    tag = clf_name.strip().lower()
    if tag in ("svc",):
        return "svm"
    if tag in ("mlpclassifier", "ann"):
        return "mlp"
    return tag


def run_consolidated_noise_report(
    dataset_path: str,
    dataset_name: str,
    noise_test_root: str,
    results_dir: str,
    seed: int,
    way_num: int,
    classifiers: str,
    noise_test_splits: str = "auto",
    report_filename: str | None = None,
    training_sizes: list[int | None] | None = None,
) -> str:
    """Train on 60 and all samples; evaluate each model on each noise split. One report file."""
    root = os.path.abspath(dataset_path)
    noise_root = os.path.abspath(noise_test_root)
    noise_split_names = discover_noise_test_splits(noise_root, noise_test_splits)
    if not noise_split_names:
        raise ValueError(f"No noise splits under {noise_root}")

    sizes = training_sizes if training_sizes is not None else [60, None]
    clf_list = parse_classifiers(classifiers)
    tags = [_normalize_clf_tag(c) for c in clf_list]

    train_files = collect_mat_split_files(os.path.join(root, "train"))
    if not train_files:
        raise ValueError(f"No training .mat under {root}/train")

    print(f"Dataset: {root}")
    print(f"Train files: {len(train_files)}")
    print(f"Noise tests: {noise_root} -> {noise_split_names}")

    X_train_full, y_train_full = build_feature_matrix(train_files)

    test_payload: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    test_counts: dict[str, int] = {}
    for split_name in noise_split_names:
        test_dir = os.path.join(noise_root, split_name)
        test_files = collect_mat_split_files(test_dir)
        if not test_files:
            raise ValueError(f"No .mat in {test_dir}")
        X_te, y_te = build_feature_matrix(test_files)
        test_payload[split_name] = (X_te, y_te)
        test_counts[split_name] = len(y_te)

    os.makedirs(results_dir, exist_ok=True)
    removed = clean_stat_result_files(results_dir, dataset_name)
    if removed:
        print(f"Removed {len(removed)} old result file(s) matching results_{dataset_name}_stat_*")

    report_path = os.path.join(
        results_dir,
        report_filename or f"results_{dataset_name}_stat_noise_moderate_benchmark.txt",
    )

    lines: list[str] = []
    lines.append("=" * 88)
    lines.append("Pulse .mat — classical ML (statistical features) — moderate noise benchmark")
    lines.append("=" * 88)
    lines.append(f"Train dataset : {root}")
    lines.append(f"Noise root    : {noise_root}")
    lines.append(f"Noise splits  : {', '.join(noise_split_names)}")
    lines.append(f"Test sizes    : {', '.join(f'{k}={v}' for k, v in test_counts.items())}")
    lines.append(f"Classifiers   : {', '.join(tags)}")
    lines.append(f"Seed          : {seed}")
    lines.append(f"Features (8)  : {', '.join(FEATURE_NAMES)}")
    lines.append(f"Full train N  : {len(y_train_full)}")
    lines.append("")

    grid: dict[str, dict[str, dict[str, float]]] = {}

    for training_samples in sizes:
        label = f"{training_samples} samples (balanced)" if training_samples else "ALL train samples"
        short = f"{training_samples}samples" if training_samples else "allsamples"
        X_tr, y_tr = subsample_train_balanced(
            X_train_full, y_train_full, training_samples, seed, way_num
        )
        print(f"\n>>> Training size: {short} (n={len(y_tr)})")
        grid[short] = {}

        for clf_name, tag in zip(clf_list, tags):
            model = make_classifier(clf_name, seed)
            model.fit(X_tr, y_tr)
            grid[short][tag] = {}
            print(f"  {tag}: fit done")

            for split_name in noise_split_names:
                X_te, y_te = test_payload[split_name]
                pred = model.predict(X_te)
                acc = float(accuracy_score(y_te, pred))
                grid[short][tag][split_name] = acc
                print(f"    {split_name}: {acc * 100:.2f}%")

        lines.append("-" * 88)
        lines.append(label)
        lines.append(f"Train n = {len(y_tr)}")
        lines.append("-" * 88)

        colw = max(len(s) for s in noise_split_names)
        lines.append("Accuracy % (overall correct on each noise test split)")
        header = f"{'model':<8}" + "".join(f"{s:>{colw + 2}}" for s in noise_split_names)
        lines.append(header)
        lines.append("-" * len(header))
        for tag in tags:
            row = f"{tag:<8}"
            for split_name in noise_split_names:
                acc = grid[short][tag][split_name]
                row += f"{acc * 100:>{colw + 2}.2f}"
            lines.append(row)
        lines.append("")

    body = "\n".join(lines).rstrip() + "\n"
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(body)

    print(f"\nSingle report written: {report_path}")
    return report_path


def main():
    parser = argparse.ArgumentParser(description="SVM / MLP / RF on pulse .mat statistical features")
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
    parser.add_argument("--training_samples", type=int, default=None, help="Total train samples (divisible by 4); omit = use all train")
    parser.add_argument("--way_num", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--classifiers", type=str, default="svm,mlp,rf")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument(
        "--noise_test_root",
        type=str,
        default=None,
        help="Root with per-SNR test folders (e.g. ..._pd_noise_benchmark_test_moderate).",
    )
    parser.add_argument(
        "--noise_test_splits",
        type=str,
        default="auto",
        help='Comma-separated split folder names, or "auto" for all test_snr* (sorted by SNR).',
    )
    parser.add_argument(
        "--noise_only",
        action="store_true",
        help="Evaluate only splits under --noise_test_root (skip dataset_path/test).",
    )
    parser.add_argument(
        "--consolidated_noise_report",
        action="store_true",
        help=(
            "Write one report: train 60 + all samples, each model × 3 noise test splits. "
            "Deletes prior results_{dataset_name}_stat_* in --results_dir. "
            "Requires --noise_test_root (or uses default …_pd_noise_benchmark_test_moderate beside dataset)."
        ),
    )
    parser.add_argument(
        "--report_filename",
        type=str,
        default=None,
        help="Output file name inside --results_dir (consolidated mode only).",
    )
    args = parser.parse_args()

    if args.consolidated_noise_report:
        default_noise = (
            Path(__file__).resolve().parent.parent.parent
            / "dataset"
            / "pulse27_1_mat_from_scalogram_27_1_pd_noise_benchmark_test_moderate"
        )
        noise_root = args.noise_test_root or (str(default_noise) if default_noise.is_dir() else None)
        if not noise_root:
            raise SystemExit(
                "consolidated_noise_report: pass --noise_test_root or place the moderate benchmark folder "
                f"next to dataset: {default_noise}"
            )
        run_consolidated_noise_report(
            dataset_path=args.dataset_path,
            dataset_name=args.dataset_name,
            noise_test_root=noise_root,
            results_dir=args.results_dir,
            seed=args.seed,
            way_num=args.way_num,
            classifiers=args.classifiers,
            noise_test_splits=args.noise_test_splits,
            report_filename=args.report_filename,
        )
        return

    root = os.path.abspath(args.dataset_path)
    train_files = collect_mat_split_files(os.path.join(root, "train"))

    noise_root = os.path.abspath(args.noise_test_root) if args.noise_test_root else None
    noise_split_names = discover_noise_test_splits(noise_root, args.noise_test_splits)

    test_jobs: list[tuple[str, str]] = []
    if not args.noise_only:
        test_jobs.append(("clean_test", os.path.join(root, "test")))
    for split_name in noise_split_names:
        test_jobs.append((split_name, os.path.join(noise_root, split_name)))

    if not test_jobs:
        raise ValueError("No test splits: set --dataset_path with test/, or pass --noise_test_root.")
    if args.noise_only and not noise_split_names:
        raise ValueError(
            "--noise_only requires a valid --noise_test_root with at least one split "
            "(check --noise_test_splits)."
        )

    print(f"Dataset: {root}")
    print(f"Train files: {len(train_files)}")
    if noise_root:
        print(f"Noise tests: {noise_root} -> {noise_split_names}")
    print(f"Test passes: {[name for name, _ in test_jobs]}")

    X_train, y_train = build_feature_matrix(train_files)

    X_train, y_train = subsample_train_balanced(
        X_train, y_train, args.training_samples, args.seed, args.way_num
    )
    print(f"After subsample: Train={len(y_train)}")

    os.makedirs(args.results_dir, exist_ok=True)
    samples_str = f"{args.training_samples}samples" if args.training_samples else "allsamples"

    summary: dict = {
        "dataset_path": root,
        "seed": args.seed,
        "training_samples": args.training_samples,
        "noise_test_root": noise_root,
        "test_evaluations": [name for name, _ in test_jobs],
    }

    for clf_name in parse_classifiers(args.classifiers):
        model = make_classifier(clf_name, args.seed)
        model.fit(X_train, y_train)
        tag = clf_name.lower()
        if tag in ("svc",):
            tag = "svm"
        if tag in ("mlpclassifier", "ann"):
            tag = "mlp"

        summary[tag] = {}
        print(f"\n========== {tag.upper()} ==========")

        for test_label, test_dir in test_jobs:
            test_files = collect_mat_split_files(test_dir)
            if not test_files:
                print(f"  [{test_label}] skip (no .mat files)")
                continue
            X_test, y_test = build_feature_matrix(test_files)
            pred = model.predict(X_test)
            acc = float(accuracy_score(y_test, pred))
            suffix = "" if test_label == "clean_test" else f"_{sanitize_result_tag(test_label)}"

            print(f"  [{test_label}] n_test={len(y_test)}  Accuracy: {acc * 100:.2f}%")
            print(classification_report(y_test, pred, target_names=CLASS_ORDER))

            result_path = os.path.join(
                args.results_dir,
                f"results_{args.dataset_name}_stat_{tag}_{samples_str}{suffix}.txt",
            )
            with open(result_path, "w", encoding="utf-8") as handle:
                handle.write(f"Classifier  : {tag}\n")
                handle.write(f"Test split  : {test_label}\n")
                handle.write(f"Test path   : {test_dir}\n")
                handle.write(f"Features    : {', '.join(FEATURE_NAMES)}\n")
                handle.write(f"Train size  : {len(y_train)}\n")
                handle.write(f"Test size   : {len(y_test)}\n")
                handle.write(f"Accuracy    : {acc * 100:.2f}%\n")
                handle.write("\n" + classification_report(y_test, pred, target_names=CLASS_ORDER))
            print(f"  Wrote {result_path}")

            summary[tag][test_label] = {"accuracy": acc, "result_file": result_path}

    meta_path = os.path.join(
        args.results_dir,
        f"results_{args.dataset_name}_stat_summary_{samples_str}.json",
    )
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"\nSummary JSON: {meta_path}")


if __name__ == "__main__":
    main()
