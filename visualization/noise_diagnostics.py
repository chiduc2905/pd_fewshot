"""Q1-style evidence figures and dataset noise profiling utilities."""

from __future__ import annotations

import csv
import math
import os
from collections import defaultdict
from typing import Any, Iterable, Sequence

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.patches import Rectangle


PROFILE_METRICS = (
    "mean_intensity",
    "std_intensity",
    "energy_entropy",
    "top10_ratio",
    "top20_ratio",
    "edge_density",
    "vertical_center",
    "horizontal_center",
    "noise_proxy",
)

SUPPORT_MODEL_METRIC_KEYS = (
    "support_weights",
    "shot_rho",
    "shot_barycenter_distance",
    "shot_transport_cost",
    "shot_transported_mass",
)


def infer_token_hw(num_tokens: int) -> tuple[int, int]:
    """Infer a near-square token grid from a flat token count."""
    if num_tokens <= 0:
        raise ValueError("num_tokens must be positive")
    root = int(math.sqrt(num_tokens))
    best = (1, num_tokens)
    best_gap = num_tokens - 1
    for height in range(1, root + 1):
        if num_tokens % height != 0:
            continue
        width = num_tokens // height
        gap = abs(width - height)
        if gap < best_gap:
            best = (height, width)
            best_gap = gap
    return best


def _ensure_numpy_image(image: torch.Tensor | np.ndarray) -> np.ndarray:
    if torch.is_tensor(image):
        image = image.detach().float().cpu().numpy()
    image = np.asarray(image, dtype=np.float32)
    if image.ndim == 3 and image.shape[0] in {1, 3}:
        image = np.transpose(image, (1, 2, 0))
    if image.ndim == 3:
        image = image.mean(axis=-1)
    if image.ndim != 2:
        raise ValueError(f"Expected a single image, got shape={image.shape}")
    image = image - float(image.min())
    scale = float(image.max())
    if scale > 0.0:
        image = image / scale
    return image


def _normalize_heatmap(heatmap: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    heatmap = np.asarray(heatmap, dtype=np.float32)
    heatmap = np.nan_to_num(heatmap, nan=0.0, posinf=0.0, neginf=0.0)
    heatmap = np.clip(heatmap, a_min=0.0, a_max=None)
    total = float(heatmap.sum())
    if total <= eps:
        heatmap = np.ones_like(heatmap, dtype=np.float32)
        total = float(heatmap.sum())
    return heatmap / max(total, eps)


def _token_values_to_heatmap(
    token_values: torch.Tensor | np.ndarray,
    target_hw: tuple[int, int],
) -> np.ndarray:
    if torch.is_tensor(token_values):
        token_values = token_values.detach().float().cpu()
    else:
        token_values = torch.as_tensor(token_values, dtype=torch.float32)
    token_values = token_values.flatten()
    token_h, token_w = infer_token_hw(int(token_values.numel()))
    token_map = token_values.view(1, 1, token_h, token_w)
    heatmap = F.interpolate(token_map, size=target_hw, mode="bilinear", align_corners=False)
    return heatmap.squeeze(0).squeeze(0).cpu().numpy()


def _safe_detach_tensor(value: Any) -> torch.Tensor | None:
    if not torch.is_tensor(value):
        return None
    return value.detach()


def _extract_class_token_maps(
    outputs: dict[str, Any],
    way_num: int,
    shot_num: int,
) -> tuple[torch.Tensor | None, str | None]:
    competitive_assignment = _safe_detach_tensor(outputs.get("competitive_assignment"))
    if competitive_assignment is not None and competitive_assignment.dim() == 3 and competitive_assignment.shape[-1] == way_num:
        return competitive_assignment.permute(0, 2, 1), "competitive_assignment"

    for key in ("probe_query_reliability", "query_token_weights", "query_token_mass"):
        tensor = _safe_detach_tensor(outputs.get(key))
        if tensor is None:
            continue
        if tensor.dim() == 4 and tensor.shape[1] == way_num:
            return tensor.mean(dim=2), key
        if tensor.dim() == 3 and tensor.shape[1] == way_num:
            return tensor, key
        if tensor.dim() == 3 and tensor.shape[1] == way_num * shot_num:
            reshaped = tensor.view(tensor.shape[0], way_num, shot_num, tensor.shape[-1])
            return reshaped.mean(dim=2), key

    transport_plan = _safe_detach_tensor(outputs.get("transport_plan"))
    if transport_plan is not None:
        if transport_plan.dim() == 5 and transport_plan.shape[1] == way_num:
            query_mass = transport_plan.sum(dim=-1).mean(dim=2)
            return query_mass, "transport_plan"
        if transport_plan.dim() == 4 and transport_plan.shape[1] == way_num:
            query_mass = transport_plan.sum(dim=-1)
            return query_mass, "transport_plan"
        if transport_plan.dim() == 4 and transport_plan.shape[1] == way_num * shot_num:
            reshaped = transport_plan.view(
                transport_plan.shape[0],
                way_num,
                shot_num,
                transport_plan.shape[-2],
                transport_plan.shape[-1],
            )
            query_mass = reshaped.sum(dim=-1).mean(dim=2)
            return query_mass, "transport_plan"
    return None, None


def _extract_spif_gate_maps(model: Any, query_images: torch.Tensor) -> tuple[torch.Tensor | None, str | None]:
    if not hasattr(model, "encode_tokens"):
        return None, None
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = query_images.device
    with torch.no_grad():
        encoded = model.encode_tokens(query_images.to(device))
    gate = getattr(encoded, "gate", None)
    if gate is None and isinstance(encoded, dict):
        gate = encoded.get("gate")
    if gate is None:
        return None, None
    gate = gate.detach().float().cpu()
    if gate.dim() == 3 and gate.shape[-1] == 1:
        gate = gate.squeeze(-1)
    if gate.dim() != 2:
        return None, None
    return gate, "spif_gate"


def _extract_encoder_norm_maps(model: Any, query_images: torch.Tensor) -> tuple[torch.Tensor | None, str | None]:
    encoder = None
    if hasattr(model, "encode"):
        encoder = model.encode
    elif hasattr(model, "encoder"):
        encoder = model.encoder
    if encoder is None:
        return None, None
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = query_images.device
    with torch.no_grad():
        features = encoder(query_images.to(device))
    if not torch.is_tensor(features) or features.dim() != 4:
        return None, None
    heatmaps = features.detach().pow(2).sum(dim=1).sqrt().cpu()
    return heatmaps, "encoder_norm"


def extract_focus_maps(
    model: Any,
    outputs: dict[str, Any] | torch.Tensor,
    query_images: torch.Tensor,
    support_images: torch.Tensor,
) -> dict[str, Any]:
    """Build spatial focus maps from model diagnostics or encoder internals."""
    if query_images.dim() != 4:
        raise ValueError(f"query_images must have shape (NumQuery, C, H, W), got {tuple(query_images.shape)}")
    if support_images.dim() != 5:
        raise ValueError(f"support_images must have shape (Way, Shot, C, H, W), got {tuple(support_images.shape)}")

    outputs_dict = outputs if isinstance(outputs, dict) else {"logits": outputs}
    way_num, shot_num = support_images.shape[:2]
    image_hw = tuple(int(dim) for dim in query_images.shape[-2:])

    class_token_maps, source = _extract_class_token_maps(outputs_dict, way_num=way_num, shot_num=shot_num)
    if class_token_maps is not None:
        class_heatmaps = []
        for query_idx in range(class_token_maps.shape[0]):
            row = []
            for class_idx in range(class_token_maps.shape[1]):
                row.append(_token_values_to_heatmap(class_token_maps[query_idx, class_idx], image_hw))
            class_heatmaps.append(np.stack(row, axis=0))
        class_heatmaps = np.stack(class_heatmaps, axis=0)
        class_heatmaps = np.stack(
            [
                np.stack([_normalize_heatmap(heatmap) for heatmap in row], axis=0)
                for row in class_heatmaps
            ],
            axis=0,
        )
        shared_maps = np.stack([_normalize_heatmap(row.mean(axis=0)) for row in class_heatmaps], axis=0)
        return {
            "source": source,
            "class_specific": True,
            "class_maps": class_heatmaps,
            "shared_maps": shared_maps,
        }

    gate_maps, source = _extract_spif_gate_maps(model, query_images)
    if gate_maps is not None:
        shared_maps = np.stack(
            [_normalize_heatmap(_token_values_to_heatmap(gate_maps[idx], image_hw)) for idx in range(gate_maps.shape[0])],
            axis=0,
        )
        return {
            "source": source,
            "class_specific": False,
            "class_maps": None,
            "shared_maps": shared_maps,
        }

    encoder_maps, source = _extract_encoder_norm_maps(model, query_images)
    if encoder_maps is not None:
        if tuple(encoder_maps.shape[-2:]) != image_hw:
            encoder_maps = F.interpolate(
                encoder_maps.unsqueeze(1),
                size=image_hw,
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
        shared_maps = np.stack(
            [_normalize_heatmap(encoder_maps[idx].cpu().numpy()) for idx in range(encoder_maps.shape[0])],
            axis=0,
        )
        return {
            "source": source,
            "class_specific": False,
            "class_maps": None,
            "shared_maps": shared_maps,
        }

    uniform = np.full((query_images.shape[0], *image_hw), 1.0 / float(image_hw[0] * image_hw[1]), dtype=np.float32)
    return {
        "source": "uniform",
        "class_specific": False,
        "class_maps": None,
        "shared_maps": uniform,
    }


def compute_focus_metrics(
    image: torch.Tensor | np.ndarray,
    heatmap: torch.Tensor | np.ndarray,
    signal_quantile: float = 0.7,
) -> dict[str, float]:
    image_np = _ensure_numpy_image(image)
    heatmap_np = np.asarray(heatmap, dtype=np.float32)
    if heatmap_np.shape != image_np.shape:
        raise ValueError(f"heatmap shape {heatmap_np.shape} does not match image shape {image_np.shape}")
    focus = _normalize_heatmap(heatmap_np)
    threshold = float(np.quantile(image_np, signal_quantile))
    signal_mask = image_np >= threshold
    background_mask = ~signal_mask

    flat = np.sort(focus.reshape(-1))[::-1]
    topk = max(1, int(round(0.1 * flat.size)))
    entropy = float(-(focus * np.log(np.clip(focus, 1e-8, None))).sum() / math.log(max(2, focus.size)))
    signal_ratio = float(focus[signal_mask].sum())
    background_ratio = float(focus[background_mask].sum())
    return {
        "signal_focus_ratio": signal_ratio,
        "background_focus_ratio": background_ratio,
        "focus_entropy": entropy,
        "focus_top10_ratio": float(flat[:topk].sum()),
    }


def _build_support_montage(support_images: torch.Tensor | np.ndarray, pad: int = 4) -> np.ndarray:
    if torch.is_tensor(support_images):
        support_images = support_images.detach().cpu()
    images = [_ensure_numpy_image(image) for image in support_images]
    height = max(image.shape[0] for image in images)
    widths = [image.shape[1] for image in images]
    canvas = np.ones((height, sum(widths) + pad * max(0, len(images) - 1)), dtype=np.float32)
    cursor = 0
    for idx, image in enumerate(images):
        image_h, image_w = image.shape
        y0 = (height - image_h) // 2
        canvas[y0 : y0 + image_h, cursor : cursor + image_w] = image
        cursor += image_w
        if idx < len(images) - 1:
            cursor += pad
    return canvas


def _plot_overlay(ax: plt.Axes, image: np.ndarray, heatmap: np.ndarray, title: str) -> None:
    signal_mask = image >= float(np.quantile(image, 0.7))
    ax.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
    ax.imshow(_normalize_heatmap(heatmap), cmap="inferno", alpha=0.62, vmin=0.0, vmax=1.0)
    ax.contour(signal_mask.astype(np.float32), levels=[0.5], colors=["#0f766e"], linewidths=1.0)
    ax.set_title(title, fontsize=10, weight="bold")
    ax.axis("off")


def _minmax01(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    values = np.clip(values, a_min=0.0, a_max=None)
    vmax = float(values.max())
    values = values - float(values.min())
    scale = float(values.max())
    if scale <= eps:
        if vmax > eps:
            return np.ones_like(values, dtype=np.float32)
        return np.zeros_like(values, dtype=np.float32)
    return values / scale


def _build_query_support_montage(
    query_image: torch.Tensor | np.ndarray,
    support_images: torch.Tensor | np.ndarray,
    *,
    pad: int = 5,
) -> np.ndarray:
    query_np = _ensure_numpy_image(query_image)
    support_np = _build_support_montage(support_images, pad=pad)
    height = max(query_np.shape[0], support_np.shape[0])
    width = query_np.shape[1] + support_np.shape[1] + pad
    canvas = np.ones((height, width), dtype=np.float32)
    q_y = (height - query_np.shape[0]) // 2
    s_y = (height - support_np.shape[0]) // 2
    canvas[q_y : q_y + query_np.shape[0], : query_np.shape[1]] = query_np
    x0 = query_np.shape[1] + pad
    canvas[s_y : s_y + support_np.shape[0], x0 : x0 + support_np.shape[1]] = support_np
    return canvas


def _build_pair_canvas(
    query_image: np.ndarray,
    support_image: np.ndarray,
    query_heatmap: np.ndarray,
    support_heatmap: np.ndarray,
    *,
    pad: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    height = max(query_image.shape[0], support_image.shape[0])
    width = query_image.shape[1] + support_image.shape[1] + pad
    image_canvas = np.ones((height, width), dtype=np.float32)
    heat_canvas = np.zeros((height, width), dtype=np.float32)

    q_y = (height - query_image.shape[0]) // 2
    s_y = (height - support_image.shape[0]) // 2
    image_canvas[q_y : q_y + query_image.shape[0], : query_image.shape[1]] = query_image
    heat_canvas[q_y : q_y + query_heatmap.shape[0], : query_heatmap.shape[1]] = _minmax01(query_heatmap)

    x0 = query_image.shape[1] + pad
    image_canvas[s_y : s_y + support_image.shape[0], x0 : x0 + support_image.shape[1]] = support_image
    heat_canvas[s_y : s_y + support_heatmap.shape[0], x0 : x0 + support_heatmap.shape[1]] = _minmax01(
        support_heatmap
    )
    return image_canvas, heat_canvas


def _plot_transport_pair(
    ax: plt.Axes,
    query_image: np.ndarray,
    support_image: np.ndarray,
    query_heatmap: np.ndarray,
    support_heatmap: np.ndarray,
    *,
    title: str,
    subtitle: str,
    query_outline: np.ndarray | None = None,
    support_outline: np.ndarray | None = None,
) -> None:
    image_canvas, heat_canvas = _build_pair_canvas(query_image, support_image, query_heatmap, support_heatmap)
    ax.imshow(image_canvas, cmap="gray", vmin=0.0, vmax=1.0)
    ax.imshow(heat_canvas, cmap="inferno", alpha=0.64, vmin=0.0, vmax=1.0)
    if query_outline is not None or support_outline is not None:
        outline_q = np.zeros_like(query_image, dtype=np.float32) if query_outline is None else query_outline
        outline_s = np.zeros_like(support_image, dtype=np.float32) if support_outline is None else support_outline
        _, outline_canvas = _build_pair_canvas(query_image, support_image, outline_q, outline_s)
        if float(np.nanmax(outline_canvas)) > 0.0:
            ax.contour(outline_canvas, levels=[0.5], colors=["#2dd4bf"], linewidths=1.1, alpha=0.95)
    ax.set_title(title, fontsize=11, weight="bold")
    ax.text(
        0.02,
        0.02,
        subtitle,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        color="white",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "black", "alpha": 0.62, "edgecolor": "none"},
    )
    ax.axis("off")


def _token_centers_on_canvas(
    *,
    num_tokens: int,
    image_hw: tuple[int, int],
    x_offset: int,
    y_offset: int,
) -> np.ndarray:
    token_h, token_w = infer_token_hw(int(num_tokens))
    image_h, image_w = image_hw
    y_centers = y_offset + (np.arange(token_h, dtype=np.float32) + 0.5) * (image_h / float(token_h))
    x_centers = x_offset + (np.arange(token_w, dtype=np.float32) + 0.5) * (image_w / float(token_w))
    grid_y, grid_x = np.meshgrid(y_centers, x_centers, indexing="ij")
    return np.stack([grid_x.reshape(-1), grid_y.reshape(-1)], axis=-1)


def _top_transport_matches(
    pair_plan: torch.Tensor | np.ndarray,
    *,
    match_scores: torch.Tensor | np.ndarray | None = None,
    max_matches: int = 14,
    matches_per_query: int = 2,
    min_mass_fraction: float = 0.0,
    eps: float = 1e-8,
) -> list[tuple[int, int, float, float]]:
    if torch.is_tensor(pair_plan):
        plan = pair_plan.detach().float().cpu()
    else:
        plan = torch.as_tensor(pair_plan, dtype=torch.float32)
    if plan.dim() != 2:
        raise ValueError(f"pair_plan must be 2-D, got shape={tuple(plan.shape)}")
    if match_scores is None:
        scores = plan
    elif torch.is_tensor(match_scores):
        scores = match_scores.detach().float().cpu()
    else:
        scores = torch.as_tensor(match_scores, dtype=torch.float32)
    if tuple(scores.shape) != tuple(plan.shape):
        raise ValueError(f"match_scores must have shape={tuple(plan.shape)}, got {tuple(scores.shape)}")
    scores = scores.clamp_min(0.0)
    flat_scores = scores.flatten()
    total_score = float(flat_scores.sum().item())
    if total_score <= eps or flat_scores.numel() == 0:
        return []
    support_tokens = int(plan.shape[1])
    query_score = scores.sum(dim=-1)
    support_score = scores.sum(dim=-2)
    max_matches = min(max(1, int(max_matches)), int(flat_scores.numel()))
    matches_per_query = min(max(1, int(matches_per_query)), support_tokens)

    candidates: dict[tuple[int, int], tuple[float, float]] = {}
    query_count = min(max_matches, int((query_score > eps).sum().item()))
    if query_count > 0:
        _query_values, query_indices = torch.topk(query_score, k=query_count)
        for query_token in query_indices.tolist():
            row = scores[int(query_token)]
            support_count = min(matches_per_query, int((row > eps).sum().item()))
            if support_count <= 0:
                continue
            score_values, support_indices = torch.topk(row, k=support_count)
            for score_value, support_token in zip(score_values.tolist(), support_indices.tolist()):
                q_idx = int(query_token)
                s_idx = int(support_token)
                mass = float(plan[q_idx, s_idx].item())
                score_value = float(score_value)
                anchor_score = float(query_score[q_idx].item() + support_score[s_idx].item() + score_value)
                candidates[(q_idx, s_idx)] = (mass, anchor_score)

    support_count = min(max(1, max_matches // 2), int((support_score > eps).sum().item()))
    if support_count > 0:
        _support_values, support_indices = torch.topk(support_score, k=support_count)
        for support_token in support_indices.tolist():
            column = scores[:, int(support_token)]
            query_token = int(column.argmax().item())
            score_value = float(column[query_token].item())
            if score_value <= eps:
                continue
            mass = float(plan[query_token, int(support_token)].item())
            s_idx = int(support_token)
            anchor_score = float(query_score[query_token].item() + support_score[s_idx].item() + score_value)
            candidates[(query_token, s_idx)] = (mass, anchor_score)

    matches: list[tuple[int, int, float, float]] = []
    sorted_candidates = sorted(candidates.items(), key=lambda item: item[1][1], reverse=True)
    for (query_token, support_token), (mass, _anchor_score) in sorted_candidates[:max_matches]:
        if mass <= eps:
            continue
        fraction = float(scores[query_token, support_token].item()) / max(total_score, eps)
        if fraction < float(min_mass_fraction):
            continue
        matches.append((query_token, support_token, mass, fraction))
    return matches


def _plot_transport_correspondence(
    ax: plt.Axes,
    query_image: np.ndarray,
    support_image: np.ndarray,
    pair_plan: torch.Tensor,
    query_heatmap: np.ndarray,
    support_heatmap: np.ndarray,
    *,
    pair_scores: torch.Tensor | None = None,
    title: str,
    subtitle: str,
    max_matches: int = 14,
) -> tuple[float, float, int]:
    pad = 18
    image_canvas, heat_canvas = _build_pair_canvas(
        query_image,
        support_image,
        query_heatmap,
        support_heatmap,
        pad=pad,
    )
    ax.imshow(image_canvas, cmap="gray", vmin=0.0, vmax=1.0)
    ax.imshow(heat_canvas, cmap="inferno", alpha=0.46, vmin=0.0, vmax=1.0)

    height = max(query_image.shape[0], support_image.shape[0])
    q_y = (height - query_image.shape[0]) // 2
    s_y = (height - support_image.shape[0]) // 2
    s_x = query_image.shape[1] + pad
    query_centers = _token_centers_on_canvas(
        num_tokens=int(pair_plan.shape[0]),
        image_hw=tuple(int(dim) for dim in query_image.shape),
        x_offset=0,
        y_offset=q_y,
    )
    support_centers = _token_centers_on_canvas(
        num_tokens=int(pair_plan.shape[1]),
        image_hw=tuple(int(dim) for dim in support_image.shape),
        x_offset=s_x,
        y_offset=s_y,
    )

    rank_scores = pair_scores.detach().float().cpu().clamp_min(0.0) if pair_scores is not None else None
    matches = _top_transport_matches(pair_plan, match_scores=rank_scores, max_matches=max_matches)
    top_match_score_fraction = float(sum(match[3] for match in matches))
    max_match_score_fraction = float(matches[0][3]) if matches else 0.0
    endpoint_scores = rank_scores if rank_scores is not None else pair_plan.detach().float().cpu()
    query_score = endpoint_scores.sum(dim=-1)
    support_score = endpoint_scores.sum(dim=-2)
    query_score_max = float(query_score.max().item()) if query_score.numel() else 0.0
    support_score_max = float(support_score.max().item()) if support_score.numel() else 0.0
    for rank, (query_token, support_token, _mass, fraction) in enumerate(reversed(matches)):
        qx, qy = query_centers[query_token]
        sx, sy = support_centers[support_token]
        endpoint_strength = max(
            float(query_score[query_token].item()) / max(query_score_max, 1e-8),
            float(support_score[support_token].item()) / max(support_score_max, 1e-8),
        )
        pair_strength = fraction / max(max_match_score_fraction, 1e-8)
        strength = min(1.0, max(pair_strength, 0.45 * endpoint_strength))
        alpha = 0.24 + 0.58 * strength
        linewidth = 0.7 + 2.6 * strength
        color = "#14b8a6" if rank >= len(matches) // 2 else "#f59e0b"
        ax.plot([qx, sx], [qy, sy], color=color, alpha=alpha, linewidth=linewidth, solid_capstyle="round")

    if matches:
        q_points = np.asarray([query_centers[item[0]] for item in matches], dtype=np.float32)
        s_points = np.asarray([support_centers[item[1]] for item in matches], dtype=np.float32)
        sizes = np.asarray(
            [
                36.0
                + 360.0
                * max(
                    float(query_score[item[0]].item()) / max(query_score_max, 1e-8),
                    float(support_score[item[1]].item()) / max(support_score_max, 1e-8),
                )
                for item in matches
            ]
        )
        ax.scatter(q_points[:, 0], q_points[:, 1], s=sizes, c="#0f766e", edgecolors="white", linewidths=0.55, alpha=0.9)
        ax.scatter(s_points[:, 0], s_points[:, 1], s=sizes, c="#b45309", edgecolors="white", linewidths=0.55, alpha=0.9)

    ax.set_title(title, fontsize=11, weight="bold")
    ax.text(
        0.02,
        0.02,
        subtitle,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        color="white",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "black", "alpha": 0.62, "edgecolor": "none"},
    )
    ax.axis("off")
    return top_match_score_fraction, max_match_score_fraction, len(matches)


def _plot_paper_transport_correspondence(
    ax: plt.Axes,
    query_image: np.ndarray,
    support_image: np.ndarray,
    pair_plan: torch.Tensor,
    query_heatmap: np.ndarray,
    support_heatmap: np.ndarray,
    *,
    pair_scores: torch.Tensor | None = None,
    title: str = "",
    max_matches: int = 6,
    accent: str = "#0f766e",
    muted: bool = False,
) -> tuple[float, float, int]:
    """Compact paper-facing UOT panel: only the discriminative correspondences."""
    pad = 20
    image_canvas, heat_canvas = _build_pair_canvas(
        query_image,
        support_image,
        query_heatmap,
        support_heatmap,
        pad=pad,
    )
    ax.imshow(image_canvas, cmap="gray", vmin=0.0, vmax=1.0)
    ax.imshow(
        heat_canvas,
        cmap="inferno",
        alpha=0.30 if muted else 0.48,
        vmin=0.0,
        vmax=1.0,
    )

    height = max(query_image.shape[0], support_image.shape[0])
    q_y = (height - query_image.shape[0]) // 2
    s_y = (height - support_image.shape[0]) // 2
    s_x = query_image.shape[1] + pad
    ax.add_patch(
        Rectangle(
            (-0.5, q_y - 0.5),
            query_image.shape[1],
            query_image.shape[0],
            fill=False,
            edgecolor="#111827",
            linewidth=0.8,
            alpha=0.75,
        )
    )
    ax.add_patch(
        Rectangle(
            (s_x - 0.5, s_y - 0.5),
            support_image.shape[1],
            support_image.shape[0],
            fill=False,
            edgecolor=accent if not muted else "#64748b",
            linewidth=1.4 if not muted else 0.9,
            alpha=0.9,
        )
    )

    query_centers = _token_centers_on_canvas(
        num_tokens=int(pair_plan.shape[0]),
        image_hw=tuple(int(dim) for dim in query_image.shape),
        x_offset=0,
        y_offset=q_y,
    )
    support_centers = _token_centers_on_canvas(
        num_tokens=int(pair_plan.shape[1]),
        image_hw=tuple(int(dim) for dim in support_image.shape),
        x_offset=s_x,
        y_offset=s_y,
    )

    rank_scores = pair_scores.detach().float().cpu().clamp_min(0.0) if pair_scores is not None else None
    if rank_scores is not None and float(rank_scores.sum().item()) <= 1e-8:
        rank_scores = None
    matches = _top_transport_matches(
        pair_plan,
        match_scores=rank_scores,
        max_matches=max_matches,
        matches_per_query=1,
        min_mass_fraction=0.0,
    )
    top_match_score_fraction = float(sum(match[3] for match in matches))
    max_match_score_fraction = float(matches[0][3]) if matches else 0.0

    endpoint_scores = rank_scores if rank_scores is not None else pair_plan.detach().float().cpu()
    query_score = endpoint_scores.sum(dim=-1)
    support_score = endpoint_scores.sum(dim=-2)
    query_score_max = float(query_score.max().item()) if query_score.numel() else 0.0
    support_score_max = float(support_score.max().item()) if support_score.numel() else 0.0
    line_color = "#64748b" if muted else accent
    dot_color = "#94a3b8" if muted else accent

    for query_token, support_token, _mass, fraction in reversed(matches):
        qx, qy = query_centers[query_token]
        sx, sy = support_centers[support_token]
        endpoint_strength = max(
            float(query_score[query_token].item()) / max(query_score_max, 1e-8),
            float(support_score[support_token].item()) / max(support_score_max, 1e-8),
        )
        pair_strength = fraction / max(max_match_score_fraction, 1e-8)
        strength = min(1.0, max(pair_strength, 0.45 * endpoint_strength))
        linewidth = (0.8 if muted else 1.2) + (1.8 if muted else 2.9) * strength
        alpha = (0.25 if muted else 0.34) + (0.42 if muted else 0.56) * strength
        line = ax.plot(
            [qx, sx],
            [qy, sy],
            color=line_color,
            alpha=alpha,
            linewidth=linewidth,
            solid_capstyle="round",
            zorder=4,
        )[0]
        line.set_path_effects(
            [
                path_effects.Stroke(linewidth=linewidth + 1.2, foreground="white", alpha=0.48),
                path_effects.Normal(),
            ]
        )

    if matches:
        q_points = np.asarray([query_centers[item[0]] for item in matches], dtype=np.float32)
        s_points = np.asarray([support_centers[item[1]] for item in matches], dtype=np.float32)
        sizes = np.asarray(
            [
                22.0
                + 150.0
                * max(
                    float(query_score[item[0]].item()) / max(query_score_max, 1e-8),
                    float(support_score[item[1]].item()) / max(support_score_max, 1e-8),
                )
                for item in matches
            ],
            dtype=np.float32,
        )
        ax.scatter(
            q_points[:, 0],
            q_points[:, 1],
            s=sizes,
            c=dot_color,
            edgecolors="white",
            linewidths=0.55,
            alpha=0.92 if not muted else 0.72,
            zorder=5,
        )
        ax.scatter(
            s_points[:, 0],
            s_points[:, 1],
            s=sizes,
            c=dot_color,
            edgecolors="white",
            linewidths=0.55,
            alpha=0.92 if not muted else 0.72,
            zorder=5,
        )

    if title:
        ax.set_title(title, fontsize=10.5, weight="bold", color="#111827", pad=6)
    ax.axis("off")
    return top_match_score_fraction, max_match_score_fraction, len(matches)


def _transport_pair_payload(
    *,
    outputs: dict[str, Any] | torch.Tensor,
    plan: torch.Tensor,
    cost_matrix: torch.Tensor | None,
    query_idx: int,
    class_idx: int,
    query_image: np.ndarray,
    support_images: torch.Tensor,
    way_num: int,
    shot_num: int,
) -> dict[str, Any]:
    shot_idx = _select_transport_shot(
        outputs,
        query_idx=query_idx,
        class_idx=class_idx,
        way_num=way_num,
        shot_num=shot_num,
    )
    pair_plan = plan[query_idx, class_idx, shot_idx]
    query_mass = pair_plan.sum(dim=-1)
    support_mass = pair_plan.sum(dim=-2)
    transported_mass = float(pair_plan.sum().item())
    expected_mass = _shot_expected_mass(
        outputs,
        query_idx=query_idx,
        class_idx=class_idx,
        shot_idx=shot_idx,
        fallback_mass=max(transported_mass, 1e-8),
        way_num=way_num,
        shot_num=shot_num,
    )
    unmatched_fraction = max(0.0, 1.0 - transported_mass / max(expected_mass, 1e-8))
    support_image = _ensure_numpy_image(support_images[class_idx, shot_idx])
    pair_cost = None
    if cost_matrix is not None and query_idx < cost_matrix.shape[0]:
        pair_cost = cost_matrix[query_idx, class_idx, shot_idx]
    transport_threshold = _threshold_for_transport_pair(
        outputs,
        query_idx=query_idx,
        class_idx=class_idx,
        shot_idx=shot_idx,
    )
    positive_evidence = _positive_transport_evidence(
        pair_plan=pair_plan,
        pair_cost=pair_cost,
        threshold=transport_threshold,
    )
    if positive_evidence is not None and float(positive_evidence.sum().item()) > 1e-8:
        pair_scores = positive_evidence
        query_values = positive_evidence.sum(dim=-1)
        support_values = positive_evidence.sum(dim=-2)
        visual_source = "positive_evidence"
        positive_evidence_total = float(positive_evidence.sum().item())
    else:
        pair_scores = None
        query_values = query_mass
        support_values = support_mass
        visual_source = "transport_mass"
        positive_evidence_total = None

    query_bright_mask = _token_bright_mask(query_image, num_tokens=int(query_mass.numel()))
    support_bright_mask = _token_bright_mask(support_image, num_tokens=int(support_mass.numel()))
    pulse_to_pulse_mass_ratio = _pulse_to_pulse_ratio(pair_plan, query_bright_mask, support_bright_mask)
    positive_pulse_to_pulse_ratio = (
        _pulse_to_pulse_ratio(pair_scores, query_bright_mask, support_bright_mask)
        if pair_scores is not None
        else None
    )
    query_bright_mass_ratio = _masked_token_ratio(query_values, query_bright_mask)
    support_bright_mass_ratio = _masked_token_ratio(support_values, support_bright_mask)

    return {
        "shot_idx": shot_idx,
        "pair_plan": pair_plan,
        "pair_scores": pair_scores,
        "query_values": query_values,
        "support_values": support_values,
        "query_heat": _token_values_to_heatmap(query_values, tuple(int(dim) for dim in query_image.shape)),
        "support_heat": _token_values_to_heatmap(support_values, tuple(int(dim) for dim in support_image.shape)),
        "support_image": support_image,
        "transported_mass": transported_mass,
        "expected_mass": expected_mass,
        "unmatched_fraction": unmatched_fraction,
        "visual_source": visual_source,
        "positive_evidence_total": positive_evidence_total,
        "pulse_to_pulse_mass_ratio": pulse_to_pulse_mass_ratio,
        "positive_pulse_to_pulse_ratio": positive_pulse_to_pulse_ratio,
        "query_bright_mass_ratio": query_bright_mass_ratio,
        "support_bright_mass_ratio": support_bright_mass_ratio,
        "transport_threshold": transport_threshold,
        "sink_mass": _noise_sink_value(
            outputs,
            "ecot_noise_sink_query_mass",
            query_idx=query_idx,
            class_idx=class_idx,
            shot_idx=shot_idx,
        )
        + _noise_sink_value(
            outputs,
            "ecot_noise_sink_support_mass",
            query_idx=query_idx,
            class_idx=class_idx,
            shot_idx=shot_idx,
        ),
    }


def _region_transport_pair_payload(
    *,
    outputs: dict[str, Any] | torch.Tensor,
    region_plan: torch.Tensor,
    query_idx: int,
    class_idx: int,
    query_image: np.ndarray,
    support_images: torch.Tensor,
    way_num: int,
    shot_num: int,
    shot_idx: int | None = None,
) -> dict[str, Any]:
    if shot_idx is None:
        shot_idx = _select_transport_shot(
            outputs,
            query_idx=query_idx,
            class_idx=class_idx,
            way_num=way_num,
            shot_num=shot_num,
        )
    shot_idx = int(max(0, min(shot_idx, int(region_plan.shape[2]) - 1)))
    pair_plan = region_plan[query_idx, class_idx, shot_idx].detach().float().cpu()
    query_values = pair_plan.sum(dim=-1)
    support_values = pair_plan.sum(dim=-2)
    support_image = _ensure_numpy_image(support_images[class_idx, shot_idx])
    return {
        "shot_idx": shot_idx,
        "pair_plan": pair_plan,
        "pair_scores": pair_plan,
        "query_heat": _token_values_to_heatmap(query_values, tuple(int(dim) for dim in query_image.shape)),
        "support_heat": _token_values_to_heatmap(support_values, tuple(int(dim) for dim in support_image.shape)),
        "support_image": support_image,
    }


def _transport_plan_by_shot(
    outputs: dict[str, Any] | torch.Tensor,
    *,
    way_num: int,
    shot_num: int,
) -> torch.Tensor | None:
    if not isinstance(outputs, dict):
        return None
    plan = _safe_detach_tensor(outputs.get("transport_plan"))
    if plan is None:
        return None
    plan = plan.float().cpu()
    if plan.dim() == 5 and plan.shape[1] == way_num and plan.shape[2] == shot_num:
        return plan
    if plan.dim() == 4 and plan.shape[1] == way_num * shot_num:
        return plan.view(plan.shape[0], way_num, shot_num, plan.shape[-2], plan.shape[-1])
    if plan.dim() == 4 and plan.shape[1] == way_num:
        return plan.unsqueeze(2)
    return None


def _matrix_by_shot(
    outputs: dict[str, Any] | torch.Tensor,
    key: str,
    *,
    way_num: int,
    shot_num: int,
) -> torch.Tensor | None:
    if not isinstance(outputs, dict):
        return None
    tensor = _safe_detach_tensor(outputs.get(key))
    if tensor is None:
        return None
    tensor = tensor.float().cpu()
    if tensor.dim() == 5 and tensor.shape[1] == way_num and tensor.shape[2] == shot_num:
        return tensor
    if tensor.dim() == 4 and tensor.shape[1] == way_num * shot_num:
        return tensor.view(tensor.shape[0], way_num, shot_num, tensor.shape[-2], tensor.shape[-1])
    if tensor.dim() == 4 and tensor.shape[1] == way_num:
        return tensor.unsqueeze(2)
    return None


def _threshold_for_transport_pair(
    outputs: dict[str, Any] | torch.Tensor,
    *,
    query_idx: int,
    class_idx: int,
    shot_idx: int,
) -> float | None:
    if not isinstance(outputs, dict):
        return None
    for key in ("adaptive_transport_cost_threshold", "transport_cost_threshold"):
        tensor = _safe_detach_tensor(outputs.get(key))
        if tensor is None:
            continue
        tensor = tensor.detach().float().cpu()
        if tensor.numel() == 1:
            return float(tensor.reshape(-1)[0].item())
        if tensor.dim() == 3 and query_idx < tensor.shape[0]:
            shot_idx_clamped = min(shot_idx, int(tensor.shape[2]) - 1)
            return float(tensor[query_idx, class_idx, shot_idx_clamped].item())
        if tensor.dim() == 2 and query_idx < tensor.shape[0]:
            return float(tensor[query_idx, class_idx].item())
        if tensor.dim() == 1 and query_idx < tensor.shape[0]:
            return float(tensor[query_idx].item())
    return None


def _positive_transport_evidence(
    *,
    pair_plan: torch.Tensor,
    pair_cost: torch.Tensor | None,
    threshold: float | None,
) -> torch.Tensor | None:
    if pair_cost is None or threshold is None:
        return None
    pair_cost = pair_cost.detach().float().cpu()
    pair_plan = pair_plan.detach().float().cpu()
    if tuple(pair_cost.shape) != tuple(pair_plan.shape):
        return None
    threshold_tensor = pair_cost.new_tensor(float(threshold))
    return pair_plan * (threshold_tensor - pair_cost).clamp_min(0.0)


def _token_bright_mask(
    image: np.ndarray,
    *,
    num_tokens: int,
    quantile: float = 0.80,
) -> torch.Tensor:
    token_h, token_w = infer_token_hw(int(num_tokens))
    image_t = torch.as_tensor(image, dtype=torch.float32).reshape(1, 1, image.shape[0], image.shape[1])
    token_energy = F.adaptive_avg_pool2d(image_t, output_size=(token_h, token_w)).flatten()
    threshold = torch.quantile(token_energy, float(quantile))
    mask = token_energy >= threshold
    if not bool(mask.any()):
        mask[token_energy.argmax()] = True
    return mask


def _token_mask_to_heatmap(
    mask: torch.Tensor,
    target_hw: tuple[int, int],
) -> np.ndarray:
    mask = mask.detach().float().cpu().flatten()
    token_h, token_w = infer_token_hw(int(mask.numel()))
    token_map = mask.view(1, 1, token_h, token_w)
    heatmap = F.interpolate(token_map, size=target_hw, mode="nearest")
    return heatmap.squeeze(0).squeeze(0).cpu().numpy()


def _masked_token_ratio(values: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> float:
    values = values.detach().float().cpu().flatten().clamp_min(0.0)
    mask = mask.detach().cpu().bool().flatten()
    if values.numel() != mask.numel():
        return 0.0
    return float(values[mask].sum().item() / max(float(values.sum().item()), eps))


def _pulse_to_pulse_ratio(
    pair_scores: torch.Tensor,
    query_mask: torch.Tensor,
    support_mask: torch.Tensor,
    eps: float = 1e-8,
) -> float:
    scores = pair_scores.detach().float().cpu().clamp_min(0.0)
    query_mask = query_mask.detach().cpu().bool().flatten()
    support_mask = support_mask.detach().cpu().bool().flatten()
    if scores.dim() != 2 or scores.shape[0] != query_mask.numel() or scores.shape[1] != support_mask.numel():
        return 0.0
    selected = scores[query_mask][:, support_mask].sum()
    return float(selected.item() / max(float(scores.sum().item()), eps))


def _uot_all_class_match_path(save_path: str) -> str:
    stem, ext = os.path.splitext(save_path)
    return f"{stem}_all_classes{ext or '.png'}"


def _shot_tensor_by_class(
    outputs: dict[str, Any] | torch.Tensor,
    key: str,
    *,
    way_num: int,
    shot_num: int,
) -> torch.Tensor | None:
    if not isinstance(outputs, dict):
        return None
    tensor = _safe_detach_tensor(outputs.get(key))
    if tensor is None:
        return None
    tensor = tensor.float().cpu()
    if tensor.dim() == 3 and tensor.shape[1] == way_num and tensor.shape[2] == shot_num:
        return tensor
    if tensor.dim() == 2 and tensor.shape[1] == way_num * shot_num:
        return tensor.view(tensor.shape[0], way_num, shot_num)
    if tensor.dim() == 2 and tensor.shape[0] == way_num and tensor.shape[1] == shot_num:
        return tensor.unsqueeze(0)
    return None


def _select_transport_shot(
    outputs: dict[str, Any] | torch.Tensor,
    *,
    query_idx: int,
    class_idx: int,
    way_num: int,
    shot_num: int,
) -> int:
    for key in ("shot_pool_weights", "shot_transported_mass"):
        tensor = _shot_tensor_by_class(outputs, key, way_num=way_num, shot_num=shot_num)
        if tensor is not None and query_idx < tensor.shape[0]:
            values = tensor[query_idx, class_idx]
            if values.numel() > 0:
                return int(values.argmax().item())
    return 0


def _query_evidence_tokens(outputs: dict[str, Any] | torch.Tensor) -> torch.Tensor | None:
    if not isinstance(outputs, dict):
        return None
    tokens = _safe_detach_tensor(outputs.get("ecot_ccem_query_reliability"))
    if tokens is None:
        tokens = _safe_detach_tensor(outputs.get("ecot_ccem_query_pi"))
    if tokens is None:
        return None
    tokens = tokens.float().cpu()
    if tokens.dim() == 5 and tokens.shape[0] == 1:
        tokens = tokens.squeeze(0)
    return tokens if tokens.dim() in {2, 4} else None


def _support_evidence_tokens(
    outputs: dict[str, Any] | torch.Tensor,
    *,
    way_num: int,
    shot_num: int,
) -> torch.Tensor | None:
    if not isinstance(outputs, dict):
        return None
    tokens = _safe_detach_tensor(outputs.get("ecot_ccem_support_reliability"))
    if tokens is None:
        tokens = _safe_detach_tensor(outputs.get("ecot_ccem_support_pi"))
    if tokens is None:
        return None
    tokens = tokens.float().cpu()
    if tokens.dim() in {4, 5} and tokens.shape[0] == 1:
        tokens = tokens.squeeze(0)
    if tokens.dim() == 3 and tokens.shape[0] == way_num:
        return tokens
    if tokens.dim() == 4:
        return tokens
    if tokens.dim() == 2 and tokens.shape[0] == way_num * shot_num:
        return tokens.view(way_num, shot_num, tokens.shape[-1])
    return None


def _noise_sink_value(
    outputs: dict[str, Any] | torch.Tensor,
    key: str,
    *,
    query_idx: int,
    class_idx: int,
    shot_idx: int,
) -> float:
    if not isinstance(outputs, dict):
        return 0.0
    tensor = _safe_detach_tensor(outputs.get(key))
    if tensor is None:
        return 0.0
    tensor = tensor.float().cpu()
    if tensor.dim() == 4 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    if tensor.dim() != 3 or query_idx >= tensor.shape[0]:
        return 0.0
    shot_idx = min(shot_idx, int(tensor.shape[2]) - 1)
    return float(tensor[query_idx, class_idx, shot_idx].item())


def _shot_expected_mass(
    outputs: dict[str, Any] | torch.Tensor,
    *,
    query_idx: int,
    class_idx: int,
    shot_idx: int,
    fallback_mass: float,
    way_num: int,
    shot_num: int,
) -> float:
    for key in ("shot_rho", "rho"):
        tensor = _shot_tensor_by_class(outputs, key, way_num=way_num, shot_num=shot_num)
        if tensor is not None and query_idx < tensor.shape[0]:
            return float(tensor[query_idx, class_idx, shot_idx].item())
    return float(fallback_mass)


def _class_label(class_names: Sequence[str], class_idx: int) -> str:
    return class_names[class_idx] if 0 <= class_idx < len(class_names) else str(class_idx)


def _export_uot_paper_evidence_figure(
    *,
    outputs: dict[str, Any] | torch.Tensor,
    query_images: torch.Tensor,
    support_images: torch.Tensor,
    logits: torch.Tensor,
    preds: torch.Tensor,
    targets: torch.Tensor,
    class_names: Sequence[str],
    plan: torch.Tensor,
    cost_matrix: torch.Tensor | None,
    save_path: str,
    episode_index: int,
    query_indices: Sequence[int],
    transport_kind: str,
    variant_text: str,
    max_matches: int = 6,
) -> list[dict[str, Any]]:
    way_num, shot_num = support_images.shape[:2]
    region_plan = _matrix_by_shot(outputs, "region_uot_coarse_plan", way_num=way_num, shot_num=shot_num)
    has_region_plan = region_plan is not None
    show_rival = way_num > 1
    if has_region_plan:
        num_cols = 4 if show_rival else 3
        width = 3.4 + 5.55 * (num_cols - 1)
    else:
        num_cols = 3 if show_rival else 2
        width = 3.4 + (5.9 if show_rival else 6.1) * (num_cols - 1)
    fig, axes = plt.subplots(
        len(query_indices),
        num_cols,
        figsize=(width, max(3.4, 3.35 * len(query_indices))),
        squeeze=False,
    )
    fig.patch.set_facecolor("white")
    rows: list[dict[str, Any]] = []

    for row_idx, query_idx_raw in enumerate(query_indices):
        query_idx = int(query_idx_raw)
        true_class = int(targets[query_idx].item())
        pred_class = int(preds[query_idx].item())
        score_np = logits[query_idx].numpy()
        if len(score_np) > 1:
            wrong_scores = score_np.copy()
            wrong_scores[true_class] = -np.inf
            best_wrong_class = int(np.argmax(wrong_scores))
            best_wrong_score = float(wrong_scores[best_wrong_class])
        else:
            best_wrong_class = true_class
            best_wrong_score = 0.0
        true_score = float(score_np[true_class])
        decision_margin = true_score - best_wrong_score

        query_np = _ensure_numpy_image(query_images[query_idx])
        ax_query = axes[row_idx, 0]
        ax_query.imshow(query_np, cmap="gray", vmin=0.0, vmax=1.0)
        query_edge = "#0f766e" if pred_class == true_class else "#dc2626"
        ax_query.add_patch(
            Rectangle(
                (-0.5, -0.5),
                query_np.shape[1],
                query_np.shape[0],
                fill=False,
                edgecolor=query_edge,
                linewidth=1.4,
                alpha=0.95,
            )
        )
        ax_query.set_title("(a) Query", fontsize=10.5, weight="bold", color="#111827", pad=6)
        ax_query.axis("off")

        primary_class = pred_class if 0 <= pred_class < way_num else true_class
        rival_class = best_wrong_class if primary_class == true_class else true_class
        primary_payload = _transport_pair_payload(
            outputs=outputs,
            plan=plan,
            cost_matrix=cost_matrix,
            query_idx=query_idx,
            class_idx=primary_class,
            query_image=query_np,
            support_images=support_images,
            way_num=way_num,
            shot_num=shot_num,
        )
        primary_axis = 2 if has_region_plan else 1
        if has_region_plan and region_plan is not None:
            region_primary_payload = _region_transport_pair_payload(
                outputs=outputs,
                region_plan=region_plan,
                query_idx=query_idx,
                class_idx=primary_class,
                query_image=query_np,
                support_images=support_images,
                way_num=way_num,
                shot_num=shot_num,
                shot_idx=int(primary_payload["shot_idx"]),
            )
            _plot_paper_transport_correspondence(
                axes[row_idx, 1],
                query_np,
                region_primary_payload["support_image"],
                region_primary_payload["pair_plan"],
                region_primary_payload["query_heat"],
                region_primary_payload["support_heat"],
                pair_scores=region_primary_payload["pair_scores"],
                title=f"(b) Region UOT prior: {_class_label(class_names, primary_class)}",
                max_matches=max(4, min(max_matches, 5)),
                accent="#0f766e" if primary_class == true_class else "#dc2626",
                muted=False,
            )
        primary_top_fraction, primary_max_fraction, primary_match_count = _plot_paper_transport_correspondence(
            axes[row_idx, primary_axis],
            query_np,
            primary_payload["support_image"],
            primary_payload["pair_plan"],
            primary_payload["query_heat"],
            primary_payload["support_heat"],
            pair_scores=primary_payload["pair_scores"],
            title=(
                f"(c) Fine UOT evidence: {_class_label(class_names, primary_class)}"
                if has_region_plan
                else f"(b) UOT evidence: {_class_label(class_names, primary_class)}"
            ),
            max_matches=max_matches,
            accent="#0f766e" if primary_class == true_class else "#dc2626",
            muted=False,
        )

        if show_rival:
            rival_axis = 3 if has_region_plan else 2
            if has_region_plan and region_plan is not None:
                rival_payload = _region_transport_pair_payload(
                    outputs=outputs,
                    region_plan=region_plan,
                    query_idx=query_idx,
                    class_idx=rival_class,
                    query_image=query_np,
                    support_images=support_images,
                    way_num=way_num,
                    shot_num=shot_num,
                )
                rival_title = f"(d) Region rival: {_class_label(class_names, rival_class)}"
            else:
                rival_payload = _transport_pair_payload(
                    outputs=outputs,
                    plan=plan,
                    cost_matrix=cost_matrix,
                    query_idx=query_idx,
                    class_idx=rival_class,
                    query_image=query_np,
                    support_images=support_images,
                    way_num=way_num,
                    shot_num=shot_num,
                )
                rival_title = f"(c) Closest rival: {_class_label(class_names, rival_class)}"
            _plot_paper_transport_correspondence(
                axes[row_idx, rival_axis],
                query_np,
                rival_payload["support_image"],
                rival_payload["pair_plan"],
                rival_payload["query_heat"],
                rival_payload["support_heat"],
                pair_scores=rival_payload["pair_scores"],
                title=rival_title,
                max_matches=max(3, max_matches - 1),
                accent="#64748b",
                muted=True,
            )

        metric_payload = primary_payload
        if primary_class != true_class:
            metric_payload = _transport_pair_payload(
                outputs=outputs,
                plan=plan,
                cost_matrix=cost_matrix,
                query_idx=query_idx,
                class_idx=true_class,
                query_image=query_np,
                support_images=support_images,
                way_num=way_num,
                shot_num=shot_num,
            )
            true_top_matches = _top_transport_matches(
                metric_payload["pair_plan"],
                match_scores=metric_payload["pair_scores"],
                max_matches=max_matches,
                matches_per_query=1,
            )
            top_match_score_fraction = float(sum(match[3] for match in true_top_matches))
            max_match_score_fraction = float(true_top_matches[0][3]) if true_top_matches else 0.0
            top_match_count = len(true_top_matches)
        else:
            top_match_score_fraction = primary_top_fraction
            max_match_score_fraction = primary_max_fraction
            top_match_count = primary_match_count

        rows.append(
            {
                "episode_index": episode_index,
                "query_rank_in_episode": query_idx,
                "variant_label": variant_text,
                "transport_kind": transport_kind,
                "true_class": true_class,
                "true_class_name": _class_label(class_names, true_class),
                "pred_class": pred_class,
                "pred_class_name": _class_label(class_names, pred_class),
                "best_wrong_class": best_wrong_class,
                "best_wrong_class_name": _class_label(class_names, best_wrong_class),
                "selected_shot": int(metric_payload["shot_idx"]),
                "correct": int(true_class == pred_class),
                "true_score": true_score,
                "best_wrong_score": best_wrong_score,
                "decision_margin": decision_margin,
                "expected_mass": float(metric_payload["expected_mass"]),
                "transported_mass": float(metric_payload["transported_mass"]),
                "unmatched_fraction": float(metric_payload["unmatched_fraction"]),
                "evidence_map_source": metric_payload["visual_source"],
                "positive_evidence_total": metric_payload["positive_evidence_total"],
                "pulse_to_pulse_mass_ratio": metric_payload["pulse_to_pulse_mass_ratio"],
                "query_bright_mass_ratio": metric_payload["query_bright_mass_ratio"],
                "support_bright_mass_ratio": metric_payload["support_bright_mass_ratio"],
                "positive_pulse_to_pulse_ratio": metric_payload["positive_pulse_to_pulse_ratio"],
                "transport_cost_threshold": metric_payload["transport_threshold"],
                "top_match_count": top_match_count,
                "top_match_score_fraction": top_match_score_fraction,
                "max_match_score_fraction": max_match_score_fraction,
                "top_match_mass_fraction": top_match_score_fraction,
                "max_match_mass_fraction": max_match_score_fraction,
                "evidence_background_mass_ratio": None,
                "evidence_query_background_mass_ratio": None,
                "evidence_support_background_mass_ratio": None,
                "noise_sink_mass": metric_payload["sink_mass"],
                "save_path": save_path,
                "all_class_match_path": "",
            }
        )

    fig.tight_layout(pad=0.45, w_pad=0.45, h_pad=0.65)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=320, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return rows


def _export_uot_all_class_match_figure(
    *,
    outputs: dict[str, Any] | torch.Tensor,
    query_images: torch.Tensor,
    support_images: torch.Tensor,
    logits: torch.Tensor,
    preds: torch.Tensor,
    targets: torch.Tensor,
    class_names: Sequence[str],
    plan: torch.Tensor,
    cost_matrix: torch.Tensor | None,
    save_path: str,
    episode_index: int,
    query_indices: Sequence[int],
    method_title: str,
    variant_text: str,
) -> str:
    way_num, shot_num = support_images.shape[:2]
    fig, axes = plt.subplots(
        len(query_indices),
        way_num,
        figsize=(5.4 * way_num, max(4.2, 3.9 * len(query_indices))),
        squeeze=False,
    )
    fig.patch.set_facecolor("white")

    for row_idx, query_idx_raw in enumerate(query_indices):
        query_idx = int(query_idx_raw)
        true_class = int(targets[query_idx].item())
        pred_class = int(preds[query_idx].item())
        score_np = logits[query_idx].numpy()
        query_np = _ensure_numpy_image(query_images[query_idx])
        image_hw = tuple(int(dim) for dim in query_np.shape)

        for class_idx in range(way_num):
            shot_idx = _select_transport_shot(
                outputs,
                query_idx=query_idx,
                class_idx=class_idx,
                way_num=way_num,
                shot_num=shot_num,
            )
            pair_plan = plan[query_idx, class_idx, shot_idx]
            pair_cost = None
            if cost_matrix is not None and query_idx < cost_matrix.shape[0]:
                pair_cost = cost_matrix[query_idx, class_idx, shot_idx]
            transport_threshold = _threshold_for_transport_pair(
                outputs,
                query_idx=query_idx,
                class_idx=class_idx,
                shot_idx=shot_idx,
            )
            positive_evidence = _positive_transport_evidence(
                pair_plan=pair_plan,
                pair_cost=pair_cost,
                threshold=transport_threshold,
            )
            if positive_evidence is not None:
                query_values = positive_evidence.sum(dim=-1)
                support_values = positive_evidence.sum(dim=-2)
                evidence_total = float(positive_evidence.sum().item())
                score_label = f"positive evidence={evidence_total:.4f}"
            else:
                query_values = pair_plan.sum(dim=-1)
                support_values = pair_plan.sum(dim=-2)
                score_label = f"mass={float(pair_plan.sum().item()):.3f}"

            support_np = _ensure_numpy_image(support_images[class_idx, shot_idx])
            support_hw = tuple(int(dim) for dim in support_np.shape)
            query_heat = _token_values_to_heatmap(query_values, image_hw)
            support_heat = _token_values_to_heatmap(support_values, support_hw)

            class_name = class_names[class_idx] if class_idx < len(class_names) else str(class_idx)
            roles = []
            if class_idx == true_class:
                roles.append("true")
            if class_idx == pred_class:
                roles.append("pred")
            role_text = f" ({'/'.join(roles)})" if roles else ""
            logit_text = f"logit={float(score_np[class_idx]):+.3f}" if class_idx < len(score_np) else "logit=n/a"
            subtitle = f"{score_label}, {logit_text}, shot={shot_idx + 1}"
            _plot_transport_correspondence(
                axes[row_idx, class_idx],
                query_np,
                support_np,
                pair_plan,
                query_heat,
                support_heat,
                pair_scores=positive_evidence,
                title=f"{class_name}{role_text}",
                subtitle=subtitle,
            )
            if class_idx == 0:
                axes[row_idx, class_idx].text(
                    0.02,
                    0.98,
                    f"query {query_idx}",
                    transform=axes[row_idx, class_idx].transAxes,
                    ha="left",
                    va="top",
                    fontsize=8.5,
                    color="white",
                    bbox={"boxstyle": "round,pad=0.25", "facecolor": "black", "alpha": 0.62, "edgecolor": "none"},
                )

    title_kind = "Partial OT" if "partial" in str(method_title).lower() else ("UOT" if "uot" in str(method_title).lower() else "OT")
    fig.suptitle(
        f"Query-to-all-class {title_kind} evidence | Episode {episode_index} | {variant_text} | {method_title}",
        fontsize=15,
        weight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.955))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=260, bbox_inches="tight")
    plt.close(fig)
    return save_path


def export_uot_evidence_figure(
    *,
    outputs: dict[str, Any] | torch.Tensor,
    query_images: torch.Tensor,
    support_images: torch.Tensor,
    logits: torch.Tensor,
    preds: torch.Tensor,
    targets: torch.Tensor,
    class_names: Sequence[str],
    save_path: str,
    episode_index: int,
    query_indices: Iterable[int] | None = None,
    transport_kind: str = "uot",
    variant_label: str = "",
    visual_style: str = "legacy",
) -> list[dict[str, Any]]:
    """Export UOT/Partial-OT evidence figures with explicit query-support token correspondences."""
    if query_images.dim() != 4:
        raise ValueError(f"query_images must have shape (NumQuery, C, H, W), got {tuple(query_images.shape)}")
    if support_images.dim() != 5:
        raise ValueError(f"support_images must have shape (Way, Shot, C, H, W), got {tuple(support_images.shape)}")

    way_num, shot_num = support_images.shape[:2]
    plan = _transport_plan_by_shot(outputs, way_num=way_num, shot_num=shot_num)
    if plan is None:
        return []
    cost_matrix = _matrix_by_shot(outputs, "cost_matrix", way_num=way_num, shot_num=shot_num)
    query_evidence_tokens = _query_evidence_tokens(outputs)
    support_evidence_tokens = _support_evidence_tokens(outputs, way_num=way_num, shot_num=shot_num)
    has_evidence = query_evidence_tokens is not None and support_evidence_tokens is not None

    selected_indices = list(query_indices) if query_indices is not None else list(range(int(query_images.shape[0])))
    if not selected_indices:
        return []

    logits = logits.detach().float().cpu()
    preds = preds.detach().cpu()
    targets = targets.detach().cpu()
    transport_kind = str(transport_kind or "uot").strip().lower()
    is_partial_ot = transport_kind in {"partial", "partial_ot", "partial_sinkhorn"}
    is_uot = transport_kind not in {"balanced", "balanced_ot", "full_ot", "ot"} and not is_partial_ot
    if is_partial_ot:
        method_title = "Fast Partial OT output"
    elif has_evidence:
        method_title = "CCEM-UOT"
    else:
        method_title = "Proposed UOT" if is_uot else "Balanced OT output"
    variant_text = str(
        variant_label
        or (
            "Ours_final Partial OT"
            if is_partial_ot
            else ("Ours_final UOT" if is_uot else "Ours_final full OT")
        )
    )
    visual_style = str(visual_style or "legacy").strip().lower().replace("-", "_")
    if visual_style in {"paper", "q1", "compact", "publication"}:
        return _export_uot_paper_evidence_figure(
            outputs=outputs,
            query_images=query_images,
            support_images=support_images,
            logits=logits,
            preds=preds,
            targets=targets,
            class_names=class_names,
            plan=plan,
            cost_matrix=cost_matrix,
            save_path=save_path,
            episode_index=episode_index,
            query_indices=selected_indices,
            transport_kind=transport_kind,
            variant_text=variant_text,
        )

    rows: list[dict[str, Any]] = []
    fig, axes = plt.subplots(
        len(selected_indices),
        5,
        figsize=(26, max(4.3, 4.0 * len(selected_indices))),
        squeeze=False,
    )
    fig.patch.set_facecolor("white")
    all_class_match_path = _uot_all_class_match_path(save_path)

    for row_idx, query_idx_raw in enumerate(selected_indices):
        query_idx = int(query_idx_raw)
        true_class = int(targets[query_idx].item())
        pred_class = int(preds[query_idx].item())
        score_np = logits[query_idx].numpy()
        if len(score_np) > 1:
            wrong_scores = score_np.copy()
            wrong_scores[true_class] = -np.inf
            best_wrong_class = int(np.argmax(wrong_scores))
            best_wrong_score = float(wrong_scores[best_wrong_class])
        else:
            best_wrong_class = true_class
            best_wrong_score = 0.0
        true_score = float(score_np[true_class])
        decision_margin = true_score - best_wrong_score

        shot_idx = _select_transport_shot(
            outputs,
            query_idx=query_idx,
            class_idx=true_class,
            way_num=way_num,
            shot_num=shot_num,
        )
        pair_plan = plan[query_idx, true_class, shot_idx]
        query_mass = pair_plan.sum(dim=-1)
        support_mass = pair_plan.sum(dim=-2)
        transported_mass = float(pair_plan.sum().item())
        expected_mass = _shot_expected_mass(
            outputs,
            query_idx=query_idx,
            class_idx=true_class,
            shot_idx=shot_idx,
            fallback_mass=max(transported_mass, 1e-8),
            way_num=way_num,
            shot_num=shot_num,
        )
        unmatched_fraction = max(0.0, 1.0 - transported_mass / max(expected_mass, 1e-8))

        query_np = _ensure_numpy_image(query_images[query_idx])
        support_np = _ensure_numpy_image(support_images[true_class, shot_idx])
        image_hw = tuple(int(dim) for dim in query_np.shape)
        support_hw = tuple(int(dim) for dim in support_np.shape)
        query_bright_mask = _token_bright_mask(query_np, num_tokens=int(query_mass.numel()))
        support_bright_mask = _token_bright_mask(support_np, num_tokens=int(support_mass.numel()))
        pulse_to_pulse_mass_ratio = _pulse_to_pulse_ratio(
            pair_plan,
            query_bright_mask,
            support_bright_mask,
        )
        pair_cost = None
        if cost_matrix is not None and query_idx < cost_matrix.shape[0]:
            pair_cost = cost_matrix[query_idx, true_class, shot_idx]
        transport_threshold = _threshold_for_transport_pair(
            outputs,
            query_idx=query_idx,
            class_idx=true_class,
            shot_idx=shot_idx,
        )
        positive_evidence = _positive_transport_evidence(
            pair_plan=pair_plan,
            pair_cost=pair_cost,
            threshold=transport_threshold,
        )
        if positive_evidence is not None:
            query_visual_values = positive_evidence.sum(dim=-1)
            support_visual_values = positive_evidence.sum(dim=-2)
            visual_source = "positive_evidence"
            positive_evidence_total = float(positive_evidence.sum().item())
            positive_pulse_to_pulse_ratio = _pulse_to_pulse_ratio(
                positive_evidence,
                query_bright_mask,
                support_bright_mask,
            )
        else:
            query_visual_values = query_mass
            support_visual_values = support_mass
            visual_source = "transport_mass"
            positive_evidence_total = None
            positive_pulse_to_pulse_ratio = None
        query_bright_mass_ratio = _masked_token_ratio(query_visual_values, query_bright_mask)
        support_bright_mass_ratio = _masked_token_ratio(support_visual_values, support_bright_mask)
        query_heat = _token_values_to_heatmap(query_visual_values, image_hw)
        support_heat = _token_values_to_heatmap(support_visual_values, support_hw)
        query_bright_heat = _token_mask_to_heatmap(query_bright_mask, image_hw)
        support_bright_heat = _token_mask_to_heatmap(support_bright_mask, support_hw)
        query_ref = _token_values_to_heatmap(torch.ones_like(query_mass), image_hw)
        support_ref = _token_values_to_heatmap(torch.ones_like(support_mass), support_hw)
        query_evidence = None
        support_evidence = None
        query_bg_mass_ratio = None
        support_bg_mass_ratio = None
        evidence_bg_mass_ratio = None
        query_evidence_heat = None
        support_evidence_heat = None
        if has_evidence and query_evidence_tokens is not None and support_evidence_tokens is not None:
            if query_evidence_tokens.dim() == 4:
                support_shot_idx = min(shot_idx, int(query_evidence_tokens.shape[2]) - 1)
                query_evidence = query_evidence_tokens[query_idx, true_class, support_shot_idx].flatten()
            else:
                query_evidence = query_evidence_tokens[query_idx].flatten()
            if support_evidence_tokens.dim() == 4:
                support_shot_idx = min(shot_idx, int(support_evidence_tokens.shape[2]) - 1)
                support_evidence = support_evidence_tokens[query_idx, true_class, support_shot_idx].flatten()
            else:
                support_shot_idx = min(shot_idx, int(support_evidence_tokens.shape[1]) - 1)
                support_evidence = support_evidence_tokens[true_class, support_shot_idx].flatten()
            if query_evidence.numel() == query_mass.numel():
                query_bg = query_evidence < 0.5
                query_bg_mass_ratio = float(
                    query_mass[query_bg].sum().item() / max(float(query_mass.sum().item()), 1e-8)
                )
                query_evidence_heat = _token_values_to_heatmap(query_evidence, image_hw)
            if support_evidence.numel() == support_mass.numel():
                support_bg = support_evidence < 0.5
                support_bg_mass_ratio = float(
                    support_mass[support_bg].sum().item() / max(float(support_mass.sum().item()), 1e-8)
                )
                support_evidence_heat = _token_values_to_heatmap(support_evidence, support_hw)
            bg_parts = [
                value
                for value in (query_bg_mass_ratio, support_bg_mass_ratio)
                if value is not None and np.isfinite(value)
            ]
            if bg_parts:
                evidence_bg_mass_ratio = float(np.mean(bg_parts))
        sink_mass = _noise_sink_value(
            outputs,
            "ecot_noise_sink_query_mass",
            query_idx=query_idx,
            class_idx=true_class,
            shot_idx=shot_idx,
        ) + _noise_sink_value(
            outputs,
            "ecot_noise_sink_support_mass",
            query_idx=query_idx,
            class_idx=true_class,
            shot_idx=shot_idx,
        )

        ax_episode, ax_balanced, ax_uot, ax_match, ax_margin = axes[row_idx]
        episode_canvas = _build_query_support_montage(query_images[query_idx], support_images[true_class])
        ax_episode.imshow(episode_canvas, cmap="gray", vmin=0.0, vmax=1.0)
        ax_episode.set_title("(a) Few-shot episode", fontsize=11, weight="bold")
        ax_episode.text(
            0.02,
            0.02,
            f"Q: {class_names[true_class] if true_class < len(class_names) else true_class} | supports={shot_num}",
            transform=ax_episode.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.5,
            color="white",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "black", "alpha": 0.62, "edgecolor": "none"},
        )
        ax_episode.axis("off")

        _plot_transport_pair(
            ax_balanced,
            query_np,
            support_np,
            query_ref,
            support_ref,
            title="(b) Balanced OT reference",
            subtitle="Full token marginals; background still participates",
        )
        if visual_source == "positive_evidence":
            threshold_text = f", T={transport_threshold:.4f}" if transport_threshold is not None else ""
            uot_subtitle = (
                f"positive evidence={positive_evidence_total:.4f}{threshold_text}, "
                f"pulse-pulse={pulse_to_pulse_mass_ratio:.1%}, shot={shot_idx + 1}"
            )
            uot_title = f"(c) Positive {method_title} evidence"
        elif has_evidence and evidence_bg_mass_ratio is not None:
            uot_subtitle = (
                f"evidence-gated mass={transported_mass:.3f}, bg-mass={evidence_bg_mass_ratio:.1%}, "
                f"pulse-pulse={pulse_to_pulse_mass_ratio:.1%}, sink={sink_mass:.3f}"
            )
            uot_title = f"(c) {method_title} mass"
        else:
            uot_subtitle = (
                f"mass={transported_mass:.3f}, pulse-pulse={pulse_to_pulse_mass_ratio:.1%}, "
                f"unmatched={unmatched_fraction:.1%}, shot={shot_idx + 1}"
            )
            uot_title = f"(c) {method_title} mass"

        _plot_transport_pair(
            ax_uot,
            query_np,
            support_np,
            query_heat,
            support_heat,
            title=uot_title,
            subtitle=uot_subtitle,
            query_outline=query_bright_heat if query_evidence_heat is None else np.maximum(query_bright_heat, query_evidence_heat),
            support_outline=(
                support_bright_heat if support_evidence_heat is None else np.maximum(support_bright_heat, support_evidence_heat)
            ),
        )
        correspondence_subtitle = (
            "points/lines scale with positive evidence"
            if visual_source == "positive_evidence"
            else "points mark high-mass tokens; lines show strongest pairings"
        )
        top_match_score_fraction, max_match_score_fraction, top_match_count = _plot_transport_correspondence(
            ax_match,
            query_np,
            support_np,
            pair_plan,
            query_heat,
            support_heat,
            pair_scores=positive_evidence,
            title="(d) Top UOT correspondences",
            subtitle=correspondence_subtitle,
        )

        rows.append(
            {
                "episode_index": episode_index,
                "query_rank_in_episode": query_idx,
                "variant_label": variant_text,
                "transport_kind": transport_kind,
                "true_class": true_class,
                "true_class_name": class_names[true_class] if true_class < len(class_names) else str(true_class),
                "pred_class": pred_class,
                "pred_class_name": class_names[pred_class] if pred_class < len(class_names) else str(pred_class),
                "best_wrong_class": best_wrong_class,
                "best_wrong_class_name": (
                    class_names[best_wrong_class] if best_wrong_class < len(class_names) else str(best_wrong_class)
                ),
                "selected_shot": shot_idx,
                "correct": int(true_class == pred_class),
                "true_score": true_score,
                "best_wrong_score": best_wrong_score,
                "decision_margin": decision_margin,
                "expected_mass": expected_mass,
                "transported_mass": transported_mass,
                "unmatched_fraction": unmatched_fraction,
                "evidence_map_source": visual_source,
                "positive_evidence_total": positive_evidence_total,
                "pulse_to_pulse_mass_ratio": pulse_to_pulse_mass_ratio,
                "query_bright_mass_ratio": query_bright_mass_ratio,
                "support_bright_mass_ratio": support_bright_mass_ratio,
                "positive_pulse_to_pulse_ratio": positive_pulse_to_pulse_ratio,
                "transport_cost_threshold": transport_threshold,
                "top_match_count": top_match_count,
                "top_match_score_fraction": top_match_score_fraction,
                "max_match_score_fraction": max_match_score_fraction,
                "top_match_mass_fraction": top_match_score_fraction,
                "max_match_mass_fraction": max_match_score_fraction,
                "evidence_background_mass_ratio": evidence_bg_mass_ratio,
                "evidence_query_background_mass_ratio": query_bg_mass_ratio,
                "evidence_support_background_mass_ratio": support_bg_mass_ratio,
                "noise_sink_mass": sink_mass,
                "save_path": save_path,
                "all_class_match_path": all_class_match_path,
            }
        )

        class_positions = np.arange(len(score_np))
        colors = ["#94a3b8"] * len(score_np)
        colors[true_class] = "#0f766e"
        if best_wrong_class != true_class:
            colors[best_wrong_class] = "#b45309"
        if pred_class != true_class and pred_class < len(colors):
            colors[pred_class] = "#dc2626"
        ax_margin.barh(class_positions, score_np, color=colors, alpha=0.92)
        ax_margin.axvline(best_wrong_score, color="#b45309", linestyle="--", linewidth=1.0, alpha=0.75)
        ax_margin.axvline(true_score, color="#0f766e", linestyle="-", linewidth=1.2, alpha=0.9)
        ax_margin.set_yticks(class_positions)
        ax_margin.set_yticklabels(class_names[: len(score_np)], fontsize=8.5)
        ax_margin.invert_yaxis()
        ax_margin.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.35)
        ax_margin.set_title("(e) Decision effect", fontsize=11, weight="bold")
        ax_margin.set_xlabel(f"margin={decision_margin:+.3f}", fontsize=9)

    fig.suptitle(
        f"Evidence-selective transport | Episode {episode_index} | {variant_text}",
        fontsize=15,
        weight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=260, bbox_inches="tight")
    plt.close(fig)
    _export_uot_all_class_match_figure(
        outputs=outputs,
        query_images=query_images,
        support_images=support_images,
        logits=logits,
        preds=preds,
        targets=targets,
        class_names=class_names,
        plan=plan,
        cost_matrix=cost_matrix,
        save_path=all_class_match_path,
        episode_index=episode_index,
        query_indices=selected_indices,
        method_title=method_title,
        variant_text=variant_text,
    )
    return rows


def export_episode_q1_figure(
    *,
    model: Any,
    outputs: dict[str, Any] | torch.Tensor,
    query_images: torch.Tensor,
    support_images: torch.Tensor,
    logits: torch.Tensor,
    preds: torch.Tensor,
    targets: torch.Tensor,
    class_names: Sequence[str],
    save_path: str,
    episode_index: int,
    query_indices: Iterable[int] | None = None,
    query_file_paths: Sequence[str | None] | None = None,
) -> list[dict[str, Any]]:
    """Export a paper-style figure for selected queries in one episode."""
    focus_payload = extract_focus_maps(model, outputs, query_images, support_images)
    selected_indices = list(query_indices) if query_indices is not None else list(range(int(query_images.shape[0])))
    if not selected_indices:
        return []

    rows = []
    num_rows = len(selected_indices)
    fig, axes = plt.subplots(num_rows, 6, figsize=(24, max(4.2, 3.8 * num_rows)), squeeze=False)
    fig.patch.set_facecolor("white")
    column_titles = ["Query", "True Focus", "Pred Focus", "Support(True)", "Support(Pred)", "Scores"]

    for col_idx, title in enumerate(column_titles):
        axes[0, col_idx].set_title(title, fontsize=12, weight="bold")

    for row_idx, query_idx in enumerate(selected_indices):
        query_idx = int(query_idx)
        true_class = int(targets[query_idx].item())
        pred_class = int(preds[query_idx].item())
        image_np = _ensure_numpy_image(query_images[query_idx])
        score_np = logits[query_idx].detach().float().cpu().numpy()

        if focus_payload["class_specific"]:
            true_map = focus_payload["class_maps"][query_idx, true_class]
            pred_map = focus_payload["class_maps"][query_idx, pred_class]
        else:
            true_map = focus_payload["shared_maps"][query_idx]
            pred_map = focus_payload["shared_maps"][query_idx]

        true_metrics = compute_focus_metrics(image_np, true_map)
        pred_metrics = compute_focus_metrics(image_np, pred_map)
        file_path = None
        if query_file_paths is not None and 0 <= query_idx < len(query_file_paths):
            file_path = query_file_paths[query_idx]

        rows.append(
            {
                "episode_index": episode_index,
                "query_rank_in_episode": query_idx,
                "file_path": file_path or "",
                "focus_source": focus_payload["source"],
                "true_class": true_class,
                "true_class_name": class_names[true_class] if true_class < len(class_names) else str(true_class),
                "pred_class": pred_class,
                "pred_class_name": class_names[pred_class] if pred_class < len(class_names) else str(pred_class),
                "correct": int(true_class == pred_class),
                "true_signal_focus_ratio": true_metrics["signal_focus_ratio"],
                "true_background_focus_ratio": true_metrics["background_focus_ratio"],
                "true_focus_entropy": true_metrics["focus_entropy"],
                "pred_signal_focus_ratio": pred_metrics["signal_focus_ratio"],
                "pred_background_focus_ratio": pred_metrics["background_focus_ratio"],
                "pred_focus_entropy": pred_metrics["focus_entropy"],
                "pred_focus_top10_ratio": pred_metrics["focus_top10_ratio"],
                "true_logit": float(score_np[true_class]),
                "pred_logit": float(score_np[pred_class]),
                "score_gap": float(score_np[pred_class] - score_np[true_class]),
            }
        )

        query_ax = axes[row_idx, 0]
        query_ax.imshow(image_np, cmap="gray", vmin=0.0, vmax=1.0)
        label = (
            f"Q{query_idx} | T={class_names[true_class]} | P={class_names[pred_class]}"
            if true_class < len(class_names) and pred_class < len(class_names)
            else f"Q{query_idx} | T={true_class} | P={pred_class}"
        )
        if file_path:
            label = f"{label}\n{os.path.basename(file_path)}"
        query_ax.set_ylabel(label, fontsize=10)
        query_ax.axis("off")

        _plot_overlay(
            axes[row_idx, 1],
            image_np,
            true_map,
            f"Signal={true_metrics['signal_focus_ratio']:.2f}",
        )
        _plot_overlay(
            axes[row_idx, 2],
            image_np,
            pred_map,
            f"Signal={pred_metrics['signal_focus_ratio']:.2f}",
        )

        axes[row_idx, 3].imshow(_build_support_montage(support_images[true_class]), cmap="gray", vmin=0.0, vmax=1.0)
        axes[row_idx, 3].axis("off")
        axes[row_idx, 4].imshow(_build_support_montage(support_images[pred_class]), cmap="gray", vmin=0.0, vmax=1.0)
        axes[row_idx, 4].axis("off")

        score_ax = axes[row_idx, 5]
        class_positions = np.arange(len(class_names))
        colors = ["#94a3b8"] * len(class_positions)
        colors[true_class] = "#0f766e"
        colors[pred_class] = "#b45309"
        score_ax.barh(class_positions, score_np, color=colors, alpha=0.92)
        score_ax.set_yticks(class_positions)
        score_ax.set_yticklabels(class_names, fontsize=9)
        score_ax.invert_yaxis()
        score_ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.35)
        score_ax.set_xlabel("Logit / Score", fontsize=9)

    fig.suptitle(
        f"Q1 Noise-Focus Diagnostic | Episode {episode_index} | Focus source: {focus_payload['source']}",
        fontsize=16,
        weight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return rows


def compute_scalogram_statistics(
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    split_name: str,
    class_names: Sequence[str],
    file_paths: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Compute per-image scalar statistics that expose noisy/dispersed scalograms."""
    if images.shape[0] != labels.shape[0]:
        raise ValueError("images and labels must have the same length")
    if file_paths is not None and len(file_paths) != images.shape[0]:
        raise ValueError("file_paths length must match number of images")

    records: list[dict[str, Any]] = []
    for idx in range(int(images.shape[0])):
        image = _ensure_numpy_image(images[idx])
        label = int(labels[idx].item())
        energy = np.clip(image.astype(np.float32), a_min=0.0, a_max=None)
        total_energy = float(energy.sum())
        if total_energy <= 1e-8:
            energy = np.ones_like(energy, dtype=np.float32)
            total_energy = float(energy.sum())
        mass = energy / total_energy
        flat_mass = mass.reshape(-1)
        sorted_energy = np.sort(energy.reshape(-1))[::-1]
        top10 = max(1, int(round(0.10 * sorted_energy.size)))
        top20 = max(1, int(round(0.20 * sorted_energy.size)))

        y_coords = np.linspace(0.0, 1.0, image.shape[0], dtype=np.float32)[:, None]
        x_coords = np.linspace(0.0, 1.0, image.shape[1], dtype=np.float32)[None, :]
        grad_y = np.abs(np.diff(image, axis=0)).mean() if image.shape[0] > 1 else 0.0
        grad_x = np.abs(np.diff(image, axis=1)).mean() if image.shape[1] > 1 else 0.0
        entropy = float(-(flat_mass * np.log(np.clip(flat_mass, 1e-8, None))).sum() / math.log(max(2, flat_mass.size)))
        top10_ratio = float(sorted_energy[:top10].sum() / total_energy)
        top20_ratio = float(sorted_energy[:top20].sum() / total_energy)
        edge_density = float(0.5 * (grad_x + grad_y))
        vertical_center = float((mass * y_coords).sum())
        horizontal_center = float((mass * x_coords).sum())
        noise_proxy = float(entropy * max(0.0, 1.0 - top20_ratio))

        records.append(
            {
                "split": split_name,
                "index": idx,
                "class_id": label,
                "class_name": class_names[label] if label < len(class_names) else str(label),
                "file_path": "" if file_paths is None else file_paths[idx],
                "mean_intensity": float(image.mean()),
                "std_intensity": float(image.std()),
                "energy_entropy": entropy,
                "top10_ratio": top10_ratio,
                "top20_ratio": top20_ratio,
                "edge_density": edge_density,
                "vertical_center": vertical_center,
                "horizontal_center": horizontal_center,
                "noise_proxy": noise_proxy,
            }
        )
    return records


def _extract_support_metric_matrices(
    outputs: dict[str, Any] | torch.Tensor | None,
    way_num: int,
    shot_num: int,
) -> dict[str, np.ndarray]:
    if not isinstance(outputs, dict):
        return {}

    metric_matrices: dict[str, np.ndarray] = {}
    for key in SUPPORT_MODEL_METRIC_KEYS:
        tensor = _safe_detach_tensor(outputs.get(key))
        if tensor is None:
            continue
        if tensor.dim() == 3 and tensor.shape[1] == way_num and tensor.shape[2] == shot_num:
            metric_matrices[key] = tensor.float().mean(dim=0).cpu().numpy()
            continue
        if tensor.dim() == 2 and tensor.shape[0] == way_num and tensor.shape[1] == shot_num:
            metric_matrices[key] = tensor.float().cpu().numpy()
            continue
        if tensor.dim() == 1 and tensor.shape[0] == way_num * shot_num:
            metric_matrices[key] = tensor.float().view(way_num, shot_num).cpu().numpy()
    return metric_matrices


def compute_support_episode_distribution(
    support_images: torch.Tensor,
    *,
    class_names: Sequence[str],
    episode_index: int,
    outputs: dict[str, Any] | torch.Tensor | None = None,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Quantify within-class shot dispersion to surface unreliable support shots."""
    if support_images.dim() != 5:
        raise ValueError(f"support_images must have shape (Way, Shot, C, H, W), got {tuple(support_images.shape)}")

    way_num, shot_num = support_images.shape[:2]
    metric_matrices = _extract_support_metric_matrices(outputs, way_num=way_num, shot_num=shot_num)
    rows: list[dict[str, Any]] = []
    class_pairwise_means = []
    class_outlier_max = []
    class_entropy_std = []

    for class_idx in range(way_num):
        class_images = support_images[class_idx]
        shot_records = compute_scalogram_statistics(
            class_images,
            torch.full((shot_num,), class_idx, dtype=torch.long),
            split_name=f"episode_{episode_index}",
            class_names=class_names,
        )
        flat_images = np.stack([_ensure_numpy_image(image).reshape(-1) for image in class_images], axis=0)
        centroid = flat_images.mean(axis=0, keepdims=True)
        centroid_distance = np.linalg.norm(flat_images - centroid, axis=1)
        if shot_num > 1:
            diffs = flat_images[:, None, :] - flat_images[None, :, :]
            pairwise_distance = np.linalg.norm(diffs, axis=-1)
            pairwise_mean = float(
                pairwise_distance[np.triu_indices(shot_num, k=1)].mean()
            )
        else:
            pairwise_mean = 0.0
        mean_centroid = float(max(centroid_distance.mean(), 1e-8))
        outlier_score = centroid_distance / mean_centroid
        entropy_values = np.array([float(record["energy_entropy"]) for record in shot_records], dtype=np.float32)
        top20_values = np.array([float(record["top20_ratio"]) for record in shot_records], dtype=np.float32)

        class_pairwise_means.append(pairwise_mean)
        class_outlier_max.append(float(outlier_score.max()))
        class_entropy_std.append(float(entropy_values.std(ddof=0)))

        for shot_idx, record in enumerate(shot_records):
            row = {
                "episode_index": episode_index,
                "class_id": class_idx,
                "class_name": class_names[class_idx] if class_idx < len(class_names) else str(class_idx),
                "shot_index": shot_idx,
                "pairwise_distance_mean": pairwise_mean,
                "centroid_distance": float(centroid_distance[shot_idx]),
                "outlier_score": float(outlier_score[shot_idx]),
                "class_entropy_std": float(entropy_values.std(ddof=0)),
                "class_top20_std": float(top20_values.std(ddof=0)),
            }
            for metric_name in PROFILE_METRICS:
                row[metric_name] = float(record[metric_name])
            for metric_name, matrix in metric_matrices.items():
                row[metric_name] = float(matrix[class_idx, shot_idx])
            rows.append(row)

    summary = {
        "support_pairwise_distance_mean": float(np.mean(class_pairwise_means)) if class_pairwise_means else 0.0,
        "support_outlier_score_max": float(np.max(class_outlier_max)) if class_outlier_max else 0.0,
        "support_entropy_std_mean": float(np.mean(class_entropy_std)) if class_entropy_std else 0.0,
    }
    return rows, summary


def export_support_distribution_figure(
    support_images: torch.Tensor,
    *,
    class_names: Sequence[str],
    episode_index: int,
    outputs: dict[str, Any] | torch.Tensor | None,
    save_path: str,
) -> list[dict[str, Any]]:
    """Render a per-episode support-shot distribution figure when shot > 1."""
    rows, summary = compute_support_episode_distribution(
        support_images,
        class_names=class_names,
        episode_index=episode_index,
        outputs=outputs,
    )
    if support_images.shape[1] <= 1:
        return rows

    metric_name = None
    if rows:
        for candidate in SUPPORT_MODEL_METRIC_KEYS:
            if candidate in rows[0]:
                metric_name = candidate
                break

    way_num, shot_num = support_images.shape[:2]
    fig, axes = plt.subplots(way_num, 2, figsize=(16, max(4.0, 2.9 * way_num)), squeeze=False)
    fig.patch.set_facecolor("white")

    row_lookup: dict[tuple[int, int], dict[str, Any]] = {
        (int(row["class_id"]), int(row["shot_index"])): row for row in rows
    }
    for class_idx in range(way_num):
        montage_ax = axes[class_idx, 0]
        montage_ax.imshow(_build_support_montage(support_images[class_idx]), cmap="gray", vmin=0.0, vmax=1.0)
        montage_ax.set_title(
            f"{class_names[class_idx]} | shots={shot_num}",
            fontsize=11,
            weight="bold",
        )
        montage_ax.axis("off")

        stats_ax = axes[class_idx, 1]
        shot_positions = np.arange(shot_num)
        centroid_values = np.array(
            [row_lookup[(class_idx, shot_idx)]["centroid_distance"] for shot_idx in range(shot_num)],
            dtype=np.float32,
        )
        outlier_values = np.array(
            [row_lookup[(class_idx, shot_idx)]["outlier_score"] for shot_idx in range(shot_num)],
            dtype=np.float32,
        )
        stats_ax.bar(shot_positions, centroid_values, color="#b45309", alpha=0.85, label="centroid_distance")
        stats_ax.plot(
            shot_positions,
            outlier_values,
            color="#0f766e",
            marker="o",
            linewidth=1.8,
            label="outlier_score",
        )
        if metric_name is not None:
            model_values = np.array(
                [row_lookup[(class_idx, shot_idx)][metric_name] for shot_idx in range(shot_num)],
                dtype=np.float32,
            )
            twin_ax = stats_ax.twinx()
            twin_ax.plot(
                shot_positions,
                model_values,
                color="#1d4ed8",
                marker="s",
                linestyle="--",
                linewidth=1.6,
                label=metric_name,
            )
            twin_ax.set_ylabel(metric_name, color="#1d4ed8", fontsize=9)
            twin_ax.tick_params(axis="y", labelsize=8, colors="#1d4ed8")
        stats_ax.set_xticks(shot_positions)
        stats_ax.set_xticklabels([f"s{shot_idx + 1}" for shot_idx in range(shot_num)])
        stats_ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
        stats_ax.set_ylabel("distance / score", fontsize=9)
        stats_ax.set_title(
            f"disp={row_lookup[(class_idx, 0)]['pairwise_distance_mean']:.3f} | "
            f"entropy_std={row_lookup[(class_idx, 0)]['class_entropy_std']:.3f}",
            fontsize=10,
        )
        handles, labels = stats_ax.get_legend_handles_labels()
        if metric_name is not None:
            twin_handles, twin_labels = twin_ax.get_legend_handles_labels()
            handles += twin_handles
            labels += twin_labels
        stats_ax.legend(handles, labels, frameon=False, fontsize=8, loc="upper right")

    fig.suptitle(
        "Support-Shot Distribution Diagnostic | "
        f"Episode {episode_index} | pairwise={summary['support_pairwise_distance_mean']:.3f}",
        fontsize=15,
        weight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return rows


def _aggregate_profile_records(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["split"], int(record["class_id"]), str(record["class_name"]))].append(record)

    summary = []
    for (split_name, class_id, class_name), items in sorted(grouped.items()):
        row: dict[str, Any] = {
            "split": split_name,
            "class_id": class_id,
            "class_name": class_name,
            "count": len(items),
        }
        for metric in PROFILE_METRICS:
            values = np.array([float(item[metric]) for item in items], dtype=np.float32)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=0))
        summary.append(row)
    return summary


def _write_csv(path: str, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _build_non_empty_split_groups(
    records: Sequence[dict[str, Any]],
    splits: Sequence[str],
    metric_name: str,
) -> tuple[list[list[float]], list[str]]:
    groups: list[list[float]] = []
    labels: list[str] = []
    for split_name in splits:
        values = [float(record[metric_name]) for record in records if record["split"] == split_name]
        if not values:
            continue
        groups.append(values)
        labels.append(split_name)
    return groups, labels


def _plot_dataset_profile(
    records: Sequence[dict[str, Any]],
    summary: Sequence[dict[str, Any]],
    *,
    class_names: Sequence[str],
    save_path: str,
    title: str,
) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    splits = list(dict.fromkeys(record["split"] for record in records))
    class_ids = list(range(len(class_names)))

    fig, axes = plt.subplots(2, 2, figsize=(18, 13))
    fig.patch.set_facecolor("white")

    if not records:
        for ax in axes.flat:
            ax.axis("off")
        axes[0, 0].text(
            0.5,
            0.5,
            "No dataset profile records available",
            ha="center",
            va="center",
            fontsize=14,
        )
        fig.suptitle(title, fontsize=16, weight="bold")
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
        fig.savefig(save_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        return

    counts = np.zeros((len(splits), len(class_ids)), dtype=np.float32)
    for summary_row in summary:
        if summary_row["split"] not in splits or int(summary_row["class_id"]) >= len(class_ids):
            continue
        split_idx = splits.index(summary_row["split"])
        class_idx = int(summary_row["class_id"])
        counts[split_idx, class_idx] = float(summary_row["count"])

    width = 0.8 / max(1, len(splits))
    x = np.arange(len(class_ids))
    for split_idx, split_name in enumerate(splits):
        axes[0, 0].bar(
            x + (split_idx - (len(splits) - 1) / 2.0) * width,
            counts[split_idx],
            width=width,
            label=split_name,
            alpha=0.9,
        )
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(class_names, rotation=25, ha="right")
    axes[0, 0].set_title("Per-Class Sample Count", fontsize=12, weight="bold")
    axes[0, 0].set_ylabel("Samples")
    legend_handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    if legend_handles:
        axes[0, 0].legend(frameon=False, fontsize=9)

    noise_proxy_groups, noise_proxy_labels = _build_non_empty_split_groups(records, splits, "noise_proxy")
    if noise_proxy_groups:
        axes[0, 1].boxplot(noise_proxy_groups, labels=noise_proxy_labels, patch_artist=True)
    else:
        axes[0, 1].text(0.5, 0.5, "No noise proxy values", ha="center", va="center", fontsize=12)
    axes[0, 1].set_title("Noise Proxy Distribution", fontsize=12, weight="bold")
    axes[0, 1].set_ylabel("entropy * (1 - top20)")
    axes[0, 1].tick_params(axis="x", rotation=20)

    top20_groups, top20_labels = _build_non_empty_split_groups(records, splits, "top20_ratio")
    if top20_groups:
        axes[1, 0].boxplot(top20_groups, labels=top20_labels, patch_artist=True)
    else:
        axes[1, 0].text(0.5, 0.5, "No energy concentration values", ha="center", va="center", fontsize=12)
    axes[1, 0].set_title("Energy Concentration Distribution", fontsize=12, weight="bold")
    axes[1, 0].set_ylabel("Top-20% energy ratio")
    axes[1, 0].tick_params(axis="x", rotation=20)

    comparison_candidates = [name for name in splits if name != "train"]
    comparison_split = None
    for preferred in ("final_test", "test", "test_clean", "val"):
        if preferred in comparison_candidates:
            comparison_split = preferred
            break
    if comparison_split is None and comparison_candidates:
        comparison_split = comparison_candidates[-1]

    train_lookup = {
        int(item["class_id"]): item
        for item in summary
        if item["split"] == "train"
    }
    compare_lookup = {
        int(item["class_id"]): item
        for item in summary
        if comparison_split is not None and item["split"] == comparison_split
    }

    heatmap_metrics = ("noise_proxy", "energy_entropy", "top20_ratio", "edge_density")
    if train_lookup and compare_lookup and comparison_split is not None:
        delta = np.zeros((len(class_ids), len(heatmap_metrics)), dtype=np.float32)
        for class_idx in class_ids:
            train_row = train_lookup.get(class_idx)
            compare_row = compare_lookup.get(class_idx)
            if train_row is None or compare_row is None:
                continue
            for metric_idx, metric_name in enumerate(heatmap_metrics):
                delta[class_idx, metric_idx] = (
                    float(compare_row[f"{metric_name}_mean"]) - float(train_row[f"{metric_name}_mean"])
                )
        max_abs = float(np.abs(delta).max()) if delta.size > 0 else 1.0
        max_abs = max(max_abs, 1e-6)
        heat_ax = axes[1, 1]
        im = heat_ax.imshow(delta, cmap="coolwarm", vmin=-max_abs, vmax=max_abs, aspect="auto")
        heat_ax.set_xticks(np.arange(len(heatmap_metrics)))
        heat_ax.set_xticklabels(["noise", "entropy", "top20", "edge"], rotation=20)
        heat_ax.set_yticks(np.arange(len(class_names)))
        heat_ax.set_yticklabels(class_names)
        heat_ax.set_title(f"Class Shift: {comparison_split} - train", fontsize=12, weight="bold")
        for row_idx in range(delta.shape[0]):
            for col_idx in range(delta.shape[1]):
                heat_ax.text(
                    col_idx,
                    row_idx,
                    f"{delta[row_idx, col_idx]:+.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black",
                )
        fig.colorbar(im, ax=heat_ax, fraction=0.046, pad=0.04)
    else:
        axes[1, 1].axis("off")
        split_means = []
        for split_name in splits:
            split_records = [record for record in records if record["split"] == split_name]
            if not split_records:
                continue
            noise_mean = np.mean([float(record["noise_proxy"]) for record in split_records])
            entropy_mean = np.mean([float(record["energy_entropy"]) for record in split_records])
            top20_mean = np.mean([float(record["top20_ratio"]) for record in split_records])
            split_means.append(
                f"{split_name}: noise={noise_mean:.3f}, entropy={entropy_mean:.3f}, top20={top20_mean:.3f}"
            )
        axes[1, 1].text(
            0.02,
            0.98,
            "\n".join(split_means) if split_means else "No summary available",
            va="top",
            ha="left",
            fontsize=11,
        )
        axes[1, 1].set_title("Split-Level Summary", fontsize=12, weight="bold")

    fig.suptitle(title, fontsize=16, weight="bold")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def export_dataset_noise_profile(
    split_payloads: dict[str, tuple[torch.Tensor, torch.Tensor, Sequence[str] | None]],
    *,
    class_names: Sequence[str],
    save_dir: str,
    run_stem: str,
) -> dict[str, Any]:
    """Export CSV/figure artifacts that summarize dataset noise and distribution shift."""
    all_records: list[dict[str, Any]] = []
    for split_name, (images, labels, file_paths) in split_payloads.items():
        if images is None or labels is None or len(images) == 0:
            continue
        all_records.extend(
            compute_scalogram_statistics(
                images,
                labels,
                split_name=split_name,
                class_names=class_names,
                file_paths=file_paths,
            )
        )

    summary = _aggregate_profile_records(all_records)
    raw_csv_path = os.path.join(save_dir, f"{run_stem}_dataset_profile_raw.csv")
    summary_csv_path = os.path.join(save_dir, f"{run_stem}_dataset_profile_summary.csv")
    figure_path = os.path.join(save_dir, f"{run_stem}_dataset_profile_q1.png")
    _write_csv(raw_csv_path, all_records)
    _write_csv(summary_csv_path, summary)
    _plot_dataset_profile(
        all_records,
        summary,
        class_names=class_names,
        save_path=figure_path,
        title=f"Dataset Noise Profile | {run_stem}",
    )

    log_metrics: dict[str, float] = {}
    for split_name in split_payloads:
        split_records = [record for record in all_records if record["split"] == split_name]
        if not split_records:
            continue
        for metric in ("noise_proxy", "energy_entropy", "top20_ratio", "edge_density"):
            values = [float(record[metric]) for record in split_records]
            log_metrics[f"{split_name}_{metric}_mean"] = float(np.mean(values))
        log_metrics[f"{split_name}_count"] = float(len(split_records))

    if "train" in split_payloads:
        for split_name in split_payloads:
            if split_name == "train":
                continue
            train_value = log_metrics.get("train_noise_proxy_mean")
            other_value = log_metrics.get(f"{split_name}_noise_proxy_mean")
            if train_value is not None and other_value is not None:
                log_metrics[f"{split_name}_minus_train_noise_proxy_mean"] = other_value - train_value

    return {
        "raw_csv_path": raw_csv_path,
        "summary_csv_path": summary_csv_path,
        "figure_path": figure_path,
        "metrics": log_metrics,
        "records": all_records,
        "summary": summary,
    }
