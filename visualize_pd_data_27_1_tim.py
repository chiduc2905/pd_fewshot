from __future__ import annotations

import argparse
import csv
import io
import math
import os
import re
import struct
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import BinaryIO

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import scipy.io as sio
except ImportError:  # The fallback reader supports the MAT files generated from CSV.
    sio = None


DEFAULT_INPUT_ROOT = Path(__file__).resolve().parent / "PD_data_27_1_mat"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "PD_data_27_1_viz_tim"

MI_INT8 = 1
MI_UINT8 = 2
MI_INT32 = 5
MI_UINT32 = 6
MI_DOUBLE = 9
MI_MATRIX = 14
MI_COMPRESSED = 15


@dataclass(frozen=True)
class Task:
    mat_path: str
    output_path: str
    input_root: str
    output_root: str
    dpi: int
    fig_width: float
    fig_height: float
    max_plot_points: int
    save_pdf: bool
    overwrite: bool


def natural_key(path: Path) -> list[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def pad8(nbytes: int) -> int:
    return (-nbytes) % 8


def read_tag(handle: BinaryIO) -> tuple[int, bytes] | None:
    raw = handle.read(8)
    if not raw:
        return None
    if len(raw) != 8:
        raise ValueError("truncated MAT tag")

    tag0, tag1 = struct.unpack("<II", raw)
    small_nbytes = tag0 >> 16
    if small_nbytes:
        dtype = tag0 & 0xFFFF
        return dtype, raw[4 : 4 + small_nbytes]

    dtype = tag0
    nbytes = tag1
    payload = handle.read(nbytes)
    if len(payload) != nbytes:
        raise ValueError("truncated MAT payload")
    if pad8(nbytes):
        handle.read(pad8(nbytes))
    return dtype, payload


def parse_mat_matrix(payload: bytes) -> tuple[str, np.ndarray] | None:
    stream = io.BytesIO(payload)

    flags = read_tag(stream)
    dims_el = read_tag(stream)
    name_el = read_tag(stream)
    real_el = read_tag(stream)
    if flags is None or dims_el is None or name_el is None or real_el is None:
        return None

    dims_dtype, dims_payload = dims_el
    name_dtype, name_payload = name_el
    real_dtype, real_payload = real_el
    if dims_dtype not in (MI_INT32, MI_UINT32) or name_dtype not in (MI_INT8, MI_UINT8):
        return None
    if real_dtype != MI_DOUBLE:
        return None

    dims = struct.unpack("<" + "i" * (len(dims_payload) // 4), dims_payload)
    if len(dims) < 2:
        return None

    name = name_payload.decode("ascii")
    arr = np.frombuffer(real_payload, dtype="<f8").copy()
    return name, arr.reshape(dims, order="F")


def load_mat_v5_double_columns(path: Path) -> dict[str, np.ndarray]:
    variables: dict[str, np.ndarray] = {}
    with path.open("rb") as handle:
        handle.read(128)
        while True:
            element = read_tag(handle)
            if element is None:
                break
            dtype, payload = element
            if dtype == MI_COMPRESSED:
                import zlib

                with io.BytesIO(zlib.decompress(payload)) as compressed_stream:
                    compressed_element = read_tag(compressed_stream)
                if compressed_element is None or compressed_element[0] != MI_MATRIX:
                    continue
                parsed = parse_mat_matrix(compressed_element[1])
            elif dtype == MI_MATRIX:
                parsed = parse_mat_matrix(payload)
            else:
                parsed = None

            if parsed is not None:
                name, arr = parsed
                variables[name] = arr
    return variables


def load_waveform(mat_path: Path) -> tuple[np.ndarray, np.ndarray]:
    if sio is not None:
        mat = sio.loadmat(mat_path)
    else:
        mat = load_mat_v5_double_columns(mat_path)

    if "Voltage" in mat and "Time" in mat:
        voltage = np.asarray(mat["Voltage"], dtype=np.float64).reshape(-1)
        time_s = np.asarray(mat["Time"], dtype=np.float64).reshape(-1)
    elif "Trace_3_VOLT" in mat and "Time_s" in mat:
        voltage = np.asarray(mat["Trace_3_VOLT"], dtype=np.float64).reshape(-1)
        time_s = np.asarray(mat["Time_s"], dtype=np.float64).reshape(-1)
    else:
        keys = sorted(k for k in mat if not k.startswith("__"))
        raise ValueError(f"unsupported MAT keys: {keys}")

    if len(voltage) != len(time_s):
        raise ValueError(f"Voltage/Time length mismatch: {len(voltage)} vs {len(time_s)}")
    if len(voltage) == 0:
        raise ValueError("empty waveform")
    return voltage, time_s


def envelope_downsample(
    time_ms: np.ndarray, voltage: np.ndarray, max_plot_points: int
) -> tuple[np.ndarray, np.ndarray]:
    n = len(voltage)
    if n <= max_plot_points or max_plot_points < 4:
        return time_ms, voltage

    bins = max(2, max_plot_points // 2)
    chunk = int(math.ceil(n / bins))
    usable = (n // chunk) * chunk
    if usable < chunk:
        return time_ms, voltage

    v_core = voltage[:usable].reshape(-1, chunk)
    min_idx = np.argmin(v_core, axis=1)
    max_idx = np.argmax(v_core, axis=1)
    pair_idx = np.stack([min_idx, max_idx], axis=1)
    pair_idx = np.sort(pair_idx, axis=1)

    base = (np.arange(v_core.shape[0]) * chunk).reshape(-1, 1)
    flat_idx = (base + pair_idx).reshape(-1)

    if usable < n:
        tail = voltage[usable:]
        tail_base = usable
        tail_idx = np.array([np.argmin(tail), np.argmax(tail)], dtype=np.int64) + tail_base
        flat_idx = np.concatenate([flat_idx, np.sort(tail_idx)])

    flat_idx = np.unique(flat_idx)
    return time_ms[flat_idx], voltage[flat_idx]


def configure_ieee_tim_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 9,
            "axes.labelsize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )


def visualize_waveform(
    voltage: np.ndarray,
    time_s: np.ndarray,
    output_path: Path,
    dpi: int,
    fig_width: float,
    fig_height: float,
    max_plot_points: int,
    save_pdf: bool,
) -> None:
    configure_ieee_tim_style()
    time_ms = time_s * 1e3
    plot_time, plot_voltage = envelope_downsample(time_ms, voltage, max_plot_points)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    ax.plot(plot_time, plot_voltage, color="black", linewidth=0.45, antialiased=True)

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude (V)")
    ax.set_xlim(float(time_ms[0]), float(time_ms[-1]))
    ax.margins(x=0)
    ax.tick_params(top=False, right=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    y_abs = float(np.nanmax(np.abs(plot_voltage)))
    if not np.isfinite(y_abs) or y_abs < 1e-12:
        y_abs = 1e-3
    ax.set_ylim(-1.05 * y_abs, 1.05 * y_abs)

    fig.tight_layout(pad=0.2)
    fig.savefig(output_path, dpi=dpi)
    if save_pdf:
        fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def collect_tasks(
    input_root: Path,
    output_root: Path,
    limit_per_folder: int | None,
    max_files: int | None,
    dpi: int,
    fig_width: float,
    fig_height: float,
    max_plot_points: int,
    save_pdf: bool,
    overwrite: bool,
) -> list[Task]:
    tasks: list[Task] = []
    folders = sorted({p.parent for p in input_root.rglob("*.mat")})
    for folder in folders:
        mat_files = sorted(folder.glob("*.mat"), key=natural_key)
        if limit_per_folder is not None:
            mat_files = mat_files[:limit_per_folder]

        for mat_path in mat_files:
            rel = mat_path.relative_to(input_root)
            output_path = output_root / rel.with_suffix(".png")
            tasks.append(
                Task(
                    mat_path=str(mat_path),
                    output_path=str(output_path),
                    input_root=str(input_root),
                    output_root=str(output_root),
                    dpi=dpi,
                    fig_width=fig_width,
                    fig_height=fig_height,
                    max_plot_points=max_plot_points,
                    save_pdf=save_pdf,
                    overwrite=overwrite,
                )
            )
            if max_files is not None and len(tasks) >= max_files:
                return tasks
    return tasks


def render_one(task: Task) -> dict[str, object]:
    mat_path = Path(task.mat_path)
    output_path = Path(task.output_path)
    input_root = Path(task.input_root)
    output_root = Path(task.output_root)
    rel_mat = mat_path.relative_to(input_root).as_posix()
    rel_png = output_path.relative_to(output_root).as_posix()

    try:
        if output_path.exists() and not task.overwrite:
            return {
                "status": "skipped",
                "mat_rel_path": rel_mat,
                "png_rel_path": rel_png,
                "samples": "",
                "duration_ms": "",
                "error": "",
            }

        voltage, time_s = load_waveform(mat_path)
        visualize_waveform(
            voltage,
            time_s,
            output_path,
            dpi=task.dpi,
            fig_width=task.fig_width,
            fig_height=task.fig_height,
            max_plot_points=task.max_plot_points,
            save_pdf=task.save_pdf,
        )
        return {
            "status": "ok",
            "mat_rel_path": rel_mat,
            "png_rel_path": rel_png,
            "samples": len(voltage),
            "duration_ms": float((time_s[-1] - time_s[0]) * 1000.0),
            "error": "",
        }
    except Exception as exc:
        return {
            "status": "error",
            "mat_rel_path": rel_mat,
            "png_rel_path": rel_png,
            "samples": "",
            "duration_ms": "",
            "error": str(exc),
        }


def write_mapping(output_root: Path, rows: list[dict[str, object]]) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    mapping_path = output_root / "file_mapping.csv"
    fieldnames = ["status", "mat_rel_path", "png_rel_path", "samples", "duration_ms", "error"]
    with mapping_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return mapping_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize PD_data_27_1 MAT waveforms with IEEE TIM-style axes only."
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--limit-per-folder",
        type=int,
        default=None,
        help="Render only the first N MAT files in each folder that directly contains MAT files.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional global cap after applying limit-per-folder.",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--fig-width", type=float, default=7.16)
    parser.add_argument("--fig-height", type=float, default=2.7)
    parser.add_argument("--max-plot-points", type=int, default=200_000)
    parser.add_argument("--save-pdf", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()

    if not input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")

    tasks = collect_tasks(
        input_root=input_root,
        output_root=output_root,
        limit_per_folder=args.limit_per_folder,
        max_files=args.max_files,
        dpi=args.dpi,
        fig_width=args.fig_width,
        fig_height=args.fig_height,
        max_plot_points=args.max_plot_points,
        save_pdf=args.save_pdf,
        overwrite=args.overwrite,
    )
    if not tasks:
        raise FileNotFoundError(f"No .mat files found under: {input_root}")

    print(f"Input : {input_root}")
    print(f"Output: {output_root}")
    print(f"Files : {len(tasks)}")
    print(f"Workers: {args.workers}")

    if args.workers == 1:
        rows = []
        for idx, task in enumerate(tasks, 1):
            row = render_one(task)
            rows.append(row)
            print(f"[{idx}/{len(tasks)}] {row['status']} {row['mat_rel_path']}")
    else:
        with Pool(processes=args.workers) as pool:
            rows = list(pool.imap_unordered(render_one, tasks))

    mapping_path = write_mapping(output_root, rows)
    ok = sum(row["status"] == "ok" for row in rows)
    skipped = sum(row["status"] == "skipped" for row in rows)
    errors = [row for row in rows if row["status"] == "error"]

    print("Done.")
    print(f"  Rendered: {ok}")
    print(f"  Skipped : {skipped}")
    print(f"  Errors  : {len(errors)}")
    print(f"  Mapping : {mapping_path}")
    for row in errors[:10]:
        print(f"  ERROR {row['mat_rel_path']}: {row['error']}")


if __name__ == "__main__":
    main()
