"""Text-free, publication-grade NR-OT mechanism figure.

Renders, per selected query, the two NR-OT mechanisms side by side:

  col 1  query scalogram
  col 2  background-novelty marginal a_q        (mechanism B: where the budget goes)
  col 3  class transport mass  q -> S_c         (evidence collected for class c)
  col 4  reference transport mass  q -> beta_-c (the background / null hypothesis)
  col 5  debiased map  (class - reference)       (mechanism C: class-specific evidence)

All panels are images: no titles, axis labels, ticks, colorbars, or legends, so the
figure can be captioned freely in the paper.  Inputs are the per-query NR-OT
evidence payload tensors produced by ``NuisanceReferencedOT`` with
``return_evidence_payload=True`` (surfaced into the model outputs dict during
evidence export).
"""

from __future__ import annotations

import math
import os
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

# Perceptually-uniform colormaps; sequential for mass, diverging for the debiased map.
_SEQ_CMAP = "magma"
_DIV_CMAP = "RdBu_r"


def _to_numpy(value: Any) -> np.ndarray | None:
    if not torch.is_tensor(value):
        return None
    return value.detach().float().cpu().numpy()


def _infer_hw(length: int) -> tuple[int, int]:
    if length <= 0:
        raise ValueError("token length must be positive")
    root = int(math.sqrt(length))
    best = (1, int(length))
    best_gap = int(length) - 1
    for h in range(1, root + 1):
        if length % h != 0:
            continue
        w = length // h
        gap = abs(w - h)
        if gap < best_gap:
            best = (h, w)
            best_gap = gap
    return best


def _display_image(img: np.ndarray) -> np.ndarray:
    """(C, H, W) -> (H, W) or (H, W, 3), per-image min-max normalized for display."""
    if img.ndim == 3 and img.shape[0] in (1, 3):
        img = np.transpose(img, (1, 2, 0))
        if img.shape[-1] == 1:
            img = img[..., 0]
    lo, hi = float(np.min(img)), float(np.max(img))
    if hi - lo < 1e-8:
        return np.zeros_like(img)
    return (img - lo) / (hi - lo)


def _grid(vec: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32).reshape(-1)
    h, w = hw
    if h * w != int(vec.size):
        h, w = _infer_hw(int(vec.size))
    return vec.reshape(h, w)


def _class_name(class_names: Sequence[str] | None, class_idx: int) -> str:
    if class_names is not None and 0 <= class_idx < len(class_names):
        return str(class_names[class_idx])
    return str(class_idx)


def _pair_scalar(outputs: dict[str, Any], key: str, query_idx: int, class_idx: int) -> float | None:
    value = _to_numpy(outputs.get(key))
    if value is None or value.ndim != 2:
        return None
    if query_idx >= value.shape[0] or class_idx >= value.shape[1]:
        return None
    return float(value[query_idx, class_idx])


def export_nr_ot_debiasing_figure(
    *,
    outputs: dict[str, Any] | Any,
    query_images: torch.Tensor,
    preds: torch.Tensor,
    targets: torch.Tensor,
    save_path: str,
    query_indices: Iterable[int],
    class_names: Sequence[str] | None = None,
    spatial_hw: tuple[int, int] | None = None,
    file_format: str = "pdf",
    panel_size: float = 2.1,
    dpi: int = 320,
) -> list[dict[str, Any]]:
    """Export the NR-OT novelty + debiasing figure.  Returns row metadata (or [])."""
    if not isinstance(outputs, dict):
        return []
    query_marginal = _to_numpy(outputs.get("nr_ot_query_marginal"))      # (Q, W, L)
    class_plan = _to_numpy(outputs.get("nr_ot_class_transport_plan"))    # (Q, W, K, Lq, Ls)
    ref_plan = _to_numpy(outputs.get("nr_ot_ref_transport_plan"))        # (Q, W, Lq, Mb)
    if query_marginal is None or class_plan is None or ref_plan is None:
        return []
    if query_marginal.ndim != 3 or class_plan.ndim != 5 or ref_plan.ndim != 4:
        return []
    if (
        query_marginal.shape[:2] != class_plan.shape[:2]
        or query_marginal.shape[:2] != ref_plan.shape[:2]
        or query_marginal.shape[-1] != class_plan.shape[-2]
        or query_marginal.shape[-1] != ref_plan.shape[-2]
    ):
        return []

    images = query_images.detach().float().cpu().numpy()
    preds_np = preds.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()
    max_queries = min(
        int(query_marginal.shape[0]),
        int(images.shape[0]),
        int(preds_np.shape[0]),
        int(targets_np.shape[0]),
    )
    way_num = int(query_marginal.shape[1])
    indices = []
    for raw_idx in query_indices:
        q = int(raw_idx)
        if q < 0 or q >= max_queries:
            continue
        c = int(preds_np[q])
        if c < 0 or c >= way_num:
            continue
        indices.append(q)
    if not indices:
        return []

    L = query_marginal.shape[-1]
    hw = spatial_hw if spatial_hw is not None else _infer_hw(L)

    n_rows = len(indices)
    n_cols = 5
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * panel_size, n_rows * panel_size),
        squeeze=False,
    )
    fig.patch.set_facecolor("white")

    rows_meta: list[dict[str, Any]] = []
    for r, q in enumerate(indices):
        c = int(preds_np[q])  # the class whose evidence produced the decision
        nov = _grid(query_marginal[q, c], hw)
        cls_mass = _grid(class_plan[q, c].sum(axis=(0, -1)), hw)   # sum over (K, Ls)
        ref_mass = _grid(ref_plan[q, c].sum(axis=-1), hw)         # sum over Mb
        debiased = cls_mass - ref_mass

        panels = [
            (_display_image(images[q]), None, None, None),
            (nov, _SEQ_CMAP, 0.0, None),
            (cls_mass, _SEQ_CMAP, 0.0, None),
            (ref_mass, _SEQ_CMAP, 0.0, None),
            (debiased, _DIV_CMAP, None, None),
        ]
        for col, (data, cmap, vmin, vmax) in enumerate(panels):
            ax = axes[r][col]
            if col == 0:
                ax.imshow(data, cmap=None if data.ndim == 3 else "gray", aspect="equal")
            elif col == 4:
                lim = float(np.max(np.abs(data))) or 1.0
                ax.imshow(
                    data, cmap=cmap, vmin=-lim, vmax=lim,
                    interpolation="bilinear", aspect="equal",
                )
            else:
                ax.imshow(
                    data, cmap=cmap, vmin=vmin,
                    vmax=(float(np.max(data)) or 1.0),
                    interpolation="bilinear", aspect="equal",
                )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.6)
                spine.set_color("black")

        rows_meta.append(
            {
                "query_idx": int(q),
                "pred_class": int(c),
                "pred_class_name": _class_name(class_names, int(c)),
                "true_class": int(targets_np[q]),
                "true_class_name": _class_name(class_names, int(targets_np[q])),
                "nr_ot_class_evidence": _pair_scalar(outputs, "nr_ot_class_evidence", q, c),
                "nr_ot_ref_evidence": _pair_scalar(outputs, "nr_ot_ref_evidence", q, c),
                "nr_ot_debias_gap": _pair_scalar(outputs, "nr_ot_debias_gap", q, c),
            }
        )

    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005, wspace=0.04, hspace=0.04)

    stem, ext = os.path.splitext(save_path)
    if not stem.endswith("_nr_ot_mechanism"):
        stem = f"{stem}_nr_ot_mechanism"
    fmt = str(file_format or "pdf").lower().lstrip(".")
    out_paths = [f"{stem}.{fmt}"]
    if fmt != "png":
        out_paths.append(f"{stem}.png")
    for path in out_paths:
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    for meta in rows_meta:
        meta["paths"] = list(out_paths)
    return rows_meta
