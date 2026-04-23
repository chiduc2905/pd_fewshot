"""Q1-style evidence figures and dataset noise profiling utilities."""

from __future__ import annotations

import csv
import math
import os
from collections import defaultdict
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


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
