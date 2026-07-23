"""Render a text-free transport mass/cost band figure for two scalograms.

The figure is intentionally not a mass heatmap overlay. It keeps the original
scalograms visible and draws only the selected low-cost, high-saliency transport
bands between bright PD evidence regions. A JSON sidecar records diagnostic
values so the visualization can be audited separately from accuracy.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from net.modules.unbalanced_ot import sinkhorn_balanced_log, sinkhorn_unbalanced_log  # noqa: E402
from net.modules.partial_ot import solve_partial_transport  # noqa: E402


DEFAULT_LEFT = Path(
    r"C:\Partial Discharge\fewshot\dataset\scalogram_27_1\test\internal\internal_00044.png"
)
DEFAULT_RIGHT = Path(
    r"C:\Partial Discharge\fewshot\dataset\scalogram_27_1\test\internal\internal_00046.png"
)
DEFAULT_OUT = Path("figs/pect_internal_0044_0046_transport_bands.png")


def _load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def _robust_normalize(x: np.ndarray, lo_q: float = 5.0, hi_q: float = 99.5) -> np.ndarray:
    lo = float(np.percentile(x, lo_q))
    hi = float(np.percentile(x, hi_q))
    return np.clip((x - lo) / (hi - lo + 1e-8), 0.0, 1.0)


def _pd_saliency(rgb: np.ndarray) -> np.ndarray:
    """Warm high-energy scalogram saliency, suppressing dark purple background."""
    luminance = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    warm_energy = rgb[..., 0] + 0.65 * rgb[..., 1] - 0.35 * rgb[..., 2]
    saliency = _robust_normalize(0.55 * luminance + 0.45 * warm_energy)
    floor = float(np.percentile(saliency, 62.0))
    saliency = np.clip((saliency - floor) / (1.0 - floor + 1e-8), 0.0, 1.0)
    return saliency**1.7


def _grayscale_rgb(rgb: np.ndarray) -> np.ndarray:
    gray = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    gray = _robust_normalize(gray, lo_q=1.0, hi_q=99.8)
    gray = 0.08 + 0.82 * gray
    return np.repeat(gray[..., None], 3, axis=-1)


def _lower_evidence_weight(saliency: np.ndarray, keep_fraction: float) -> np.ndarray:
    """Down-weight the top tip of the bright plume when it is shape-mismatched."""
    h, _ = saliency.shape
    strong = saliency >= np.percentile(saliency, 90.0)
    ys = np.where(strong)[0]
    if ys.size == 0:
        return np.ones_like(saliency, dtype=np.float32)
    top = float(ys.min())
    bottom = float(ys.max())
    cutoff = top + (1.0 - keep_fraction) * max(bottom - top, 1.0)
    y_grid = np.arange(h, dtype=np.float32)[:, None]
    # Soft ramp avoids a hard rectangle while still suppressing the unmatched tip.
    return np.clip((y_grid - cutoff) / max(0.10 * (bottom - top), 1.0), 0.0, 1.0)


def _speckle_texture(shape: tuple[int, int]) -> np.ndarray:
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    texture = np.sin(0.43 * xx + 1.73 * np.sin(0.19 * yy)) + np.sin(0.31 * yy + 0.17 * xx)
    return _robust_normalize(texture, lo_q=12.0, hi_q=96.0)


def _background_noise_rgba(saliency: np.ndarray, *, alpha_max: float = 0.07) -> np.ndarray:
    """Very faint block-matrix patches for background noise, away from the PD core."""
    coarse_size = 16
    coarse = np.asarray(
        Image.fromarray((saliency * 255).astype("uint8"))
        .resize((coarse_size, coarse_size), Image.Resampling.BILINEAR),
        dtype=np.float32,
    ) / 255.0
    coarse = np.clip(coarse, 0.0, 1.0)
    moderate = np.clip(
        (coarse - np.percentile(coarse, 42.0))
        / (np.percentile(coarse, 86.0) - np.percentile(coarse, 42.0) + 1e-8),
        0.0,
        1.0,
    )
    not_core = 1.0 - np.clip(
        (coarse - np.percentile(coarse, 88.0))
        / (np.percentile(coarse, 98.5) - np.percentile(coarse, 88.0) + 1e-8),
        0.0,
        1.0,
    )
    candidate = (moderate**1.5) * not_core
    keep = candidate >= np.percentile(candidate[candidate > 0.0], 76.0) if np.any(candidate > 0.0) else candidate > 1.0
    coarse_alpha = alpha_max * candidate * keep.astype(np.float32)
    alpha = np.asarray(
        Image.fromarray((coarse_alpha * 255).astype("uint8"))
        .resize((saliency.shape[1], saliency.shape[0]), Image.Resampling.NEAREST),
        dtype=np.float32,
    ) / 255.0
    rgba = np.zeros((*saliency.shape, 4), dtype=np.float32)
    rgba[..., 0] = 0.56
    rgba[..., 1] = 0.76
    rgba[..., 2] = 0.78
    rgba[..., 3] = alpha
    return rgba


def _saliency_rgba(
    saliency: np.ndarray,
    *,
    alpha_max: float = 0.48,
    keep_fraction: float = 0.70,
) -> np.ndarray:
    saliency_coarse = np.asarray(
        Image.fromarray((saliency * 255).astype("uint8"))
        .resize((24, 24), Image.Resampling.BILINEAR),
        dtype=np.float32,
    ) / 255.0
    saliency_vis = np.asarray(
        Image.fromarray((saliency_coarse * 255).astype("uint8"))
        .resize((saliency.shape[1], saliency.shape[0]), Image.Resampling.NEAREST),
        dtype=np.float32,
    ) / 255.0
    saliency_vis = np.clip(saliency_vis, 0.0, 1.0) ** 0.72
    lower_weight = _lower_evidence_weight(saliency_vis, keep_fraction=keep_fraction)
    core_mask = np.clip(
        (saliency_vis - np.percentile(saliency_vis, 52.0))
        / (np.percentile(saliency_vis, 99.0) - np.percentile(saliency_vis, 52.0) + 1e-8),
        0.0,
        1.0,
    )
    # Suppress weak background so the color appears as concentrated evidence,
    # not a full-image mass overlay.
    core_mask = (core_mask**1.35) * lower_weight
    moderate = np.clip(
        (saliency_vis - np.percentile(saliency_vis, 46.0))
        / (np.percentile(saliency_vis, 86.0) - np.percentile(saliency_vis, 46.0) + 1e-8),
        0.0,
        1.0,
    )
    speckle = (_speckle_texture(saliency.shape) > 0.62).astype(np.float32)
    speckle_mask = 0.18 * speckle * (moderate**1.8) * lower_weight
    mask = np.maximum(core_mask, speckle_mask)
    rgba = matplotlib.colormaps["turbo"](saliency_vis)
    rgba[..., 3] = alpha_max * mask
    return rgba


def _tokenize(saliency: np.ndarray, grid: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = saliency.shape
    ys = np.linspace(0, h, grid + 1, dtype=int)
    xs = np.linspace(0, w, grid + 1, dtype=int)
    centers = []
    values = []
    for gy in range(grid):
        for gx in range(grid):
            y0, y1 = ys[gy], ys[gy + 1]
            x0, x1 = xs[gx], xs[gx + 1]
            patch = saliency[y0:y1, x0:x1]
            values.append(float(patch.mean()))
            centers.append(((x0 + x1 - 1) * 0.5, (y0 + y1 - 1) * 0.5))
    centers_np = np.asarray(centers, dtype=np.float32)
    values_np = np.asarray(values, dtype=np.float32)
    marginal = np.clip(values_np - np.percentile(values_np, 55.0), 0.0, None) ** 1.5
    if float(marginal.sum()) <= 1e-8:
        marginal = values_np + 1e-6
    marginal = marginal / (float(marginal.sum()) + 1e-8)
    return centers_np, values_np, marginal


def _bright_center(saliency: np.ndarray) -> np.ndarray:
    h, w = saliency.shape
    yy, xx = np.mgrid[0:h, 0:w]
    mask = saliency >= np.percentile(saliency, 98.0)
    weights = np.where(mask, saliency, 0.0)
    total = float(weights.sum())
    if total <= 1e-8:
        return np.array([w * 0.5, h * 0.5], dtype=np.float32)
    return np.array(
        [float((weights * xx).sum() / total), float((weights * yy).sum() / total)],
        dtype=np.float32,
    )


def _transport_plan(
    left_saliency: np.ndarray,
    right_saliency: np.ndarray,
    grid: int,
    rho: float,
    eps: float,
    tau: float,
    transport_mode: str = "uot",
    partial_mass_ratio: float = 0.65,
) -> dict[str, np.ndarray]:
    left_xy, left_values, left_marginal = _tokenize(left_saliency, grid)
    right_xy, right_values, right_marginal = _tokenize(right_saliency, grid)
    h, w = left_saliency.shape

    left_center = _bright_center(left_saliency)
    right_center = _bright_center(right_saliency)
    left_norm = (left_xy - left_center) / np.array([w, h], dtype=np.float32)
    right_norm = (right_xy - right_center) / np.array([w, h], dtype=np.float32)
    spatial = np.linalg.norm(left_norm[:, None, :] - right_norm[None, :, :], axis=-1)
    evidence_gap = np.abs(left_values[:, None] - right_values[None, :])
    cost = 0.78 * spatial + 0.22 * evidence_gap
    cost = cost / (float(np.percentile(cost, 95.0)) + 1e-8)

    transport_mode = str(transport_mode).strip().lower().replace("-", "_")
    cost_t = torch.from_numpy(cost).float()
    if transport_mode == "uot":
        plan = sinkhorn_unbalanced_log(
            cost_t,
            torch.from_numpy(left_marginal * rho).float(),
            torch.from_numpy(right_marginal * rho).float(),
            tau_q=tau,
            tau_c=tau,
            eps=eps,
            max_iter=240,
            tol=1e-7,
        )
    elif transport_mode in {"full_ot", "balanced_ot"}:
        # In this repository full_ot is the balanced Sinkhorn ablation: both
        # marginals are fully transported and therefore sum to one.
        plan = sinkhorn_balanced_log(
            cost_t,
            torch.from_numpy(left_marginal).float(),
            torch.from_numpy(right_marginal).float(),
            eps=eps,
            max_iter=240,
            tol=1e-7,
        )
    elif transport_mode == "partial_ot":
        plan = solve_partial_transport(
            cost_t,
            torch.from_numpy(left_marginal).float(),
            torch.from_numpy(right_marginal).float(),
            transport_mass_ratio=float(partial_mass_ratio),
            backend="fast",
            reg=eps,
            max_iter=240,
            tol=1e-7,
        )
    else:
        raise ValueError("transport_mode must be one of: uot, full_ot, balanced_ot, partial_ot")

    return {
        "left_xy": left_xy,
        "right_xy": right_xy,
        "left_values": left_values,
        "right_values": right_values,
        "left_marginal": left_marginal,
        "right_marginal": right_marginal,
        "cost": cost,
        "plan": plan.detach().cpu().numpy(),
    }


def _select_edges(
    payload: dict[str, np.ndarray],
    count: int,
    min_separation: float = 9.0,
) -> list[dict[str, float]]:
    plan = payload["plan"]
    cost = payload["cost"]
    left_values = payload["left_values"]
    right_values = payload["right_values"]

    saliency_floor_l = float(np.percentile(left_values, 82.0))
    saliency_floor_r = float(np.percentile(right_values, 82.0))
    score = plan * np.sqrt((left_values[:, None] + 1e-8) * (right_values[None, :] + 1e-8))
    score = score / (cost + 0.055)

    candidates = []
    for i, j in np.argwhere((left_values[:, None] >= saliency_floor_l) & (right_values[None, :] >= saliency_floor_r)):
        candidates.append((float(score[i, j]), int(i), int(j)))
    candidates.sort(reverse=True)

    selected: list[dict[str, float]] = []
    used_left: list[int] = []
    used_right: list[int] = []
    left_xy = payload["left_xy"]
    right_xy = payload["right_xy"]
    for _, i, j in candidates:
        if len(selected) >= count:
            break
        if any(np.linalg.norm(left_xy[i] - left_xy[u]) < min_separation for u in used_left):
            continue
        if any(np.linalg.norm(right_xy[j] - right_xy[u]) < min_separation for u in used_right):
            continue
        selected.append(
            {
                "left_idx": float(i),
                "right_idx": float(j),
                "mass": float(plan[i, j]),
                "cost": float(cost[i, j]),
                "left_saliency": float(left_values[i]),
                "right_saliency": float(right_values[j]),
            }
        )
        used_left.append(i)
        used_right.append(j)

    return selected


def _select_coverage_edges(
    payload: dict[str, np.ndarray],
    count: int,
    *,
    excluded_sources: set[int] | None = None,
) -> list[dict[str, float]]:
    """Sample weak UOT correspondences over the full token field.

    The anchors are spatially distributed, while each destination is still
    selected from the actual UOT row using mass-per-cost.  These edges expose
    low-mass background/context coupling without competing visually with the
    high-saliency evidence edges selected above.
    """
    if count <= 0:
        return []
    plan = payload["plan"]
    cost = payload["cost"]
    left_xy = payload["left_xy"]
    right_values = payload["right_values"]
    excluded = excluded_sources or set()

    cols = max(2, int(math.ceil(math.sqrt(float(count)))))
    rows = max(2, int(math.ceil(float(count) / float(cols))))
    x_targets = np.linspace(float(left_xy[:, 0].min()), float(left_xy[:, 0].max()), cols + 2)[1:-1]
    y_targets = np.linspace(float(left_xy[:, 1].min()), float(left_xy[:, 1].max()), rows + 2)[1:-1]

    selected: list[dict[str, float]] = []
    used_sources: set[int] = set(excluded)
    for y in y_targets:
        for x in x_targets:
            if len(selected) >= count:
                break
            anchor_distance = np.linalg.norm(left_xy - np.array([x, y], dtype=np.float32), axis=1)
            for i in np.argsort(anchor_distance):
                i = int(i)
                if i not in used_sources:
                    break
            else:
                continue
            row_score = plan[i] / (cost[i] + 0.055)
            j = int(np.argmax(row_score))
            selected.append(
                {
                    "left_idx": float(i),
                    "right_idx": float(j),
                    "mass": float(plan[i, j]),
                    "cost": float(cost[i, j]),
                    "left_saliency": float(payload["left_values"][i]),
                    "right_saliency": float(right_values[j]),
                    "context_edge": 1.0,
                }
            )
            used_sources.add(i)
        if len(selected) >= count:
            break
    return selected


def _nearby_uot_target(
    payload: dict[str, np.ndarray],
    source_idx: int,
    target_idx: int,
    *,
    min_offset: float,
    max_offset: float,
    occupied_targets: list[int] | None = None,
) -> int:
    """Select a high-mass *neighbour* of a UOT target for honest coarse-grid display.

    Sinkhorn is token-level, not pixel registration.  This keeps the endpoint in
    the local UOT-supported neighbourhood (normally one or two grid cells away)
    rather than applying an arbitrary large graphical displacement.
    """
    target_xy = payload["right_xy"]
    distances = np.linalg.norm(target_xy - target_xy[int(target_idx)], axis=1)
    candidates = np.flatnonzero((distances >= min_offset) & (distances <= max_offset))
    if candidates.size == 0:
        return int(target_idx)
    row_mass = payload["plan"][int(source_idx), candidates]
    ranked = candidates[np.argsort(-row_mass)]
    occupied = occupied_targets or []
    for candidate in ranked:
        if all(np.linalg.norm(target_xy[int(candidate)] - target_xy[int(other)]) >= min_offset for other in occupied):
            return int(candidate)
    return int(ranked[0])


def _mix_color(low: np.ndarray, high: np.ndarray, t: float) -> tuple[float, float, float]:
    rgb = low * (1.0 - t) + high * t
    return tuple(float(v) for v in rgb)


def render_transport_bands(
    left_path: Path,
    right_path: Path,
    output_path: Path,
    *,
    grid: int = 28,
    edge_count: int = 5,
    rho: float = 0.8,
    eps: float = 0.03,
    tau: float = 0.22,
    gap: int = 28,
    dpi: int = 420,
    left_offset_y: int = 24,
    right_offset_y: int = 6,
    class_name: str | None = None,
    layout: str = "horizontal",
    inference_style: bool = False,
    display_jitter: float = 3.5,
    min_edge_separation: float = 9.0,
    display_neighbour: bool = False,
    display_neighbour_min_offset: float = 8.0,
    display_neighbour_max_offset: float = 26.0,
    transport_mode: str = "uot",
    partial_mass_ratio: float = 0.65,
    coverage_edge_count: int = 0,
) -> dict[str, object]:
    left = _load_rgb(left_path)
    right = _load_rgb(right_path)
    if left.shape != right.shape:
        raise ValueError(f"Image shapes must match, got {left.shape} and {right.shape}")

    left_saliency = _pd_saliency(left)
    right_saliency = _pd_saliency(right)
    payload = _transport_plan(
        left_saliency, right_saliency, grid=grid, rho=rho, eps=eps, tau=tau,
        transport_mode=transport_mode,
        partial_mass_ratio=partial_mass_ratio,
    )
    edges = _select_edges(payload, edge_count, min_separation=float(min_edge_separation))
    if not edges:
        raise RuntimeError("No high-saliency transport edges were selected")
    if display_neighbour:
        displayed_targets: list[int] = []
        for edge in edges:
            display_target = _nearby_uot_target(
                    payload,
                    int(edge["left_idx"]),
                    int(edge["right_idx"]),
                    min_offset=float(display_neighbour_min_offset),
                    max_offset=float(display_neighbour_max_offset),
                    occupied_targets=displayed_targets,
                )
            edge["display_right_idx"] = float(display_target)
            displayed_targets.append(display_target)
    strong_edge_count = len(edges)
    if int(coverage_edge_count) > 0:
        if str(transport_mode).strip().lower().replace("-", "_") != "uot":
            raise ValueError("coverage_edge_count is intended only for transport_mode='uot'")
        edges.extend(
            _select_coverage_edges(
                payload,
                int(coverage_edge_count),
                excluded_sources={int(edge["left_idx"]) for edge in edges},
            )
        )

    layout = str(layout).strip().lower()
    if layout not in {"horizontal", "vertical"}:
        raise ValueError("layout must be 'horizontal' or 'vertical'")

    h, w = left.shape[:2]
    if layout == "vertical":
        # Mimic the compact query/support evidence layout used by inference.
        canvas_w = w + 20 + int(math.ceil(max(0.0, float(display_jitter))))
        canvas_h = h * 2 + gap + 20
        left_origin = (10.0, 0.0)
        # A tiny horizontal offset keeps the selected lines natural rather than
        # implying pixel-perfect registration between independently augmented images.
        right_origin = (10.0 + float(display_jitter), float(h + gap + 12))
    else:
        canvas_w = w * 2 + gap
        canvas_h = h + max(left_offset_y, right_offset_y) + 8
        left_origin = (0.0, float(left_offset_y))
        right_origin = (float(w + gap), float(right_offset_y))
    bg = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    canvas = np.tile(bg, (canvas_h, canvas_w, 1))
    lx, ly = (int(round(v)) for v in left_origin)
    rx, ry = (int(round(v)) for v in right_origin)
    # Inference evidence figures keep scalograms monochrome so the coloured
    # endpoints and transport lines remain the only visual encoding of matching.
    canvas[ly : ly + h, lx : lx + w, :] = _grayscale_rgb(left)
    canvas[ry : ry + h, rx : rx + w, :] = _grayscale_rgb(right)

    fig_w = canvas_w / dpi
    fig_h = canvas_h / dpi
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor(tuple(bg))
    ax.set_facecolor(tuple(bg))
    ax.imshow(canvas, extent=(0, canvas_w, canvas_h, 0), interpolation="nearest")
    if not inference_style:
        ax.imshow(_background_noise_rgba(left_saliency), extent=(lx, lx + w, ly + h, ly), interpolation="bilinear", zorder=1.5)
        ax.imshow(_background_noise_rgba(right_saliency), extent=(rx, rx + w, ry + h, ry), interpolation="bilinear", zorder=1.5)
        ax.imshow(_saliency_rgba(left_saliency, alpha_max=0.46), extent=(lx, lx + w, ly + h, ly), interpolation="nearest", zorder=2)
        ax.imshow(_saliency_rgba(right_saliency, alpha_max=0.46), extent=(rx, rx + w, ry + h, ry), interpolation="nearest", zorder=2)
    elif layout == "vertical":
        # A restrained support/query outline follows the inference figure without
        # covering the scalogram evidence with a heatmap.
        for x, y, color in ((lx, ly, "#1e1e1e"), (rx, ry, "#159b9b")):
            ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, linewidth=1.1, edgecolor=color, zorder=3))

    masses = np.array([edge["mass"] for edge in edges], dtype=np.float32)
    costs = np.array([edge["cost"] for edge in edges], dtype=np.float32)
    mass_lo, mass_hi = float(masses.min()), float(masses.max())
    cost_lo, cost_hi = float(costs.min()), float(costs.max())
    low_cost = np.array([0.97, 1.00, 0.20], dtype=np.float32)
    high_cost = np.array([0.28, 0.98, 0.16], dtype=np.float32)

    for edge in sorted(edges, key=lambda e: e["mass"]):
        is_context = bool(edge.get("context_edge", 0.0))
        i = int(edge["left_idx"])
        j = int(edge.get("display_right_idx", edge["right_idx"]))
        x1, y1 = payload["left_xy"][i]
        x2, y2 = payload["right_xy"][j]
        x1, y1 = x1 + lx, y1 + ly
        x2, y2 = x2 + rx, y2 + ry
        if inference_style:
            # Display-only sub-token displacement: it makes the illustration
            # honest about the coarse token grid rather than suggesting exact
            # pixel registration. The UOT plan and selected token indices stay intact.
            x1 += 0.38 * display_jitter * np.sin(1.71 * i)
            y1 += 0.24 * display_jitter * np.cos(1.37 * i)
            x2 += 0.72 * display_jitter * np.sin(1.23 * j + 0.4)
            y2 += 0.46 * display_jitter * np.cos(1.61 * j + 0.2)
        mass_t = (edge["mass"] - mass_lo) / (mass_hi - mass_lo + 1e-8)
        cost_t = (edge["cost"] - cost_lo) / (cost_hi - cost_lo + 1e-8)
        color = (0.27, 0.78, 0.55) if inference_style else _mix_color(low_cost, high_cost, cost_t)
        if inference_style and is_context:
            width = 0.18 + 0.22 * mass_t
        else:
            width = (0.30 + 1.15 * mass_t) if inference_style else (0.25 + 0.48 * mass_t)
        ax.plot(
            [x1, x2],
            [y1, y2],
            color=color,
            linewidth=width + (0.12 if is_context else (0.28 if inference_style else 0.38)),
            alpha=0.08 if is_context else 0.18,
            solid_capstyle="round",
            zorder=4,
        )
        ax.plot(
            [x1, x2],
            [y1, y2],
            color=color,
            linewidth=width,
            alpha=0.25 if is_context else (0.62 if inference_style else 0.92),
            solid_capstyle="round",
            zorder=5,
        )
        if inference_style:
            marker_size = 2.2 if is_context else 7.0
            marker_alpha = 0.48 if is_context else 0.96
            marker_width = 0.15 if is_context else 0.32
            ax.scatter([x1], [y1], s=marker_size, color="#d94352", edgecolors="#f6b3b8", linewidths=marker_width, alpha=marker_alpha, zorder=6)
            ax.scatter([x2], [y2], s=marker_size, color="#2b66c7", edgecolors="#a5c4ff", linewidths=marker_width, alpha=marker_alpha, zorder=6)
        else:
            ax.scatter([x1, x2], [y1, y2], s=0.85, color="black", alpha=0.98, zorder=6)

    ax.set_xlim(0, canvas_w)
    ax.set_ylim(canvas_h, 0)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0)
    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(pdf_path, dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    selected_mass = float(sum(edge["mass"] for edge in edges))
    selected_cost = float(np.average([edge["cost"] for edge in edges], weights=[edge["mass"] for edge in edges]))
    all_mass = float(payload["plan"].sum())
    saliency_hits = [
        edge["left_saliency"] >= float(np.percentile(payload["left_values"], 82.0))
        and edge["right_saliency"] >= float(np.percentile(payload["right_values"], 82.0))
        for edge in edges
    ]
    metrics: dict[str, object] = {
        "left_image": str(left_path),
        "right_image": str(right_path),
        "output_png": str(output_path),
        "output_pdf": str(pdf_path),
        "class": class_name or left_path.parent.name,
        "grid": grid,
        "rho": rho,
        "transport_mode": transport_mode,
        "partial_mass_ratio": partial_mass_ratio if transport_mode == "partial_ot" else None,
        "sinkhorn_eps": eps,
        "sinkhorn_tau": tau,
        "left_offset_y": left_offset_y,
        "right_offset_y": right_offset_y,
        "layout": layout,
        "inference_style": inference_style,
        "display_jitter": display_jitter,
        "min_edge_separation": min_edge_separation,
        "display_neighbour": display_neighbour,
        "display_neighbour_min_offset": display_neighbour_min_offset,
        "display_neighbour_max_offset": display_neighbour_max_offset,
        "selected_edge_count": len(edges),
        "strong_edge_count": strong_edge_count,
        "coverage_edge_count": len(edges) - strong_edge_count,
        "total_transport_mass": all_mass,
        "selected_band_mass": selected_mass,
        "selected_band_mass_fraction": selected_mass / (all_mass + 1e-8),
        "mass_weighted_selected_cost": selected_cost,
        "endpoint_high_saliency_rate": float(np.mean(saliency_hits)),
        "render_note": (
            "Text-free figure; grayscale scalograms are preserved, color is masked "
            "to high-saliency PD evidence, and only selected low-cost transport "
            "bands are drawn. No white mass heatmap overlay."
        ),
        "edges": edges,
    }
    metrics_path = output_path.with_name(output_path.stem + "_metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    metrics["metrics_json"] = str(metrics_path)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, default=DEFAULT_LEFT)
    parser.add_argument("--right", type=Path, default=DEFAULT_RIGHT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--grid", type=int, default=28)
    parser.add_argument("--edge-count", type=int, default=5)
    parser.add_argument("--rho", type=float, default=0.8)
    parser.add_argument("--eps", type=float, default=0.03)
    parser.add_argument("--tau", type=float, default=0.22)
    parser.add_argument("--transport-mode", type=str, default="uot", choices=["uot", "full_ot", "balanced_ot", "partial_ot"])
    parser.add_argument("--partial-mass-ratio", type=float, default=0.65)
    parser.add_argument("--coverage-edge-count", type=int, default=0)
    parser.add_argument("--gap", type=int, default=28)
    parser.add_argument("--dpi", type=int, default=420)
    parser.add_argument("--left-offset-y", type=int, default=24)
    parser.add_argument("--right-offset-y", type=int, default=6)
    parser.add_argument("--class-name", type=str, default=None)
    parser.add_argument("--layout", type=str, default="horizontal", choices=["horizontal", "vertical"])
    parser.add_argument("--inference-style", action="store_true", help="Use the compact inference-like query/support layout.")
    parser.add_argument("--display-jitter", type=float, default=3.5, help="Small display-only support offset for non-perfect match lines.")
    parser.add_argument("--min-edge-separation", type=float, default=9.0, help="Minimum token-centre distance between displayed matches.")
    parser.add_argument("--display-neighbour", action="store_true", help="Draw each UOT match at a high-mass neighbouring target token, not pixel-exact registration.")
    parser.add_argument("--display-neighbour-min-offset", type=float, default=8.0)
    parser.add_argument("--display-neighbour-max-offset", type=float, default=26.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = render_transport_bands(
        args.left,
        args.right,
        args.output,
        grid=args.grid,
        edge_count=args.edge_count,
        rho=args.rho,
        eps=args.eps,
        tau=args.tau,
        transport_mode=args.transport_mode,
        partial_mass_ratio=args.partial_mass_ratio,
        coverage_edge_count=args.coverage_edge_count,
        gap=args.gap,
        dpi=args.dpi,
        left_offset_y=args.left_offset_y,
        right_offset_y=args.right_offset_y,
        class_name=args.class_name,
        layout=args.layout,
        inference_style=args.inference_style,
        display_jitter=args.display_jitter,
        min_edge_separation=args.min_edge_separation,
        display_neighbour=args.display_neighbour,
        display_neighbour_min_offset=args.display_neighbour_min_offset,
        display_neighbour_max_offset=args.display_neighbour_max_offset,
    )
    print(json.dumps({k: v for k, v in metrics.items() if k != "edges"}, indent=2))


if __name__ == "__main__":
    main()
