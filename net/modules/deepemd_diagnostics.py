from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _infer_token_hw(num_tokens: int) -> tuple[int, int]:
    root = int(math.sqrt(num_tokens))
    for height in range(root, 0, -1):
        if num_tokens % height == 0:
            return height, num_tokens // height
    return 1, num_tokens


def _normalized_entropy(values: torch.Tensor, eps: float) -> torch.Tensor:
    probabilities = values.clamp_min(0.0)
    probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(eps)
    entropy = -(probabilities * probabilities.clamp_min(eps).log()).sum(dim=-1)
    if values.shape[-1] > 1:
        entropy = entropy / math.log(float(values.shape[-1]))
    else:
        entropy = torch.zeros_like(entropy)
    return entropy


def _signal_mask(
    images: torch.Tensor,
    *,
    token_hw: tuple[int, int],
    quantile: float,
    eps: float,
) -> torch.Tensor:
    if images.dim() != 4:
        raise ValueError(f"images must have shape (N, C, H, W), got {tuple(images.shape)}")
    intensity = images.detach().float().mean(dim=1, keepdim=True)
    flat = intensity.flatten(1)
    minimum = flat.amin(dim=1, keepdim=True)
    scale = (flat.amax(dim=1, keepdim=True) - minimum).clamp_min(eps)
    intensity = ((flat - minimum) / scale).reshape_as(intensity)
    token_intensity = F.adaptive_avg_pool2d(intensity, token_hw).flatten(1)
    threshold = torch.quantile(token_intensity, quantile, dim=1, keepdim=True)
    mask = token_intensity >= threshold
    empty = ~mask.any(dim=1)
    if empty.any():
        maxima = token_intensity.argmax(dim=1)
        mask[empty, maxima[empty]] = True
    return mask


def compute_deepemd_diagnostics(
    *,
    similarity: torch.Tensor,
    flow: torch.Tensor,
    query_weights: torch.Tensor,
    support_weights: torch.Tensor,
    query_images: torch.Tensor,
    support_images: torch.Tensor,
    logits: torch.Tensor,
    uniform_weight_logits: torch.Tensor | None,
    temperature: float,
    normalize_score_by_mass: bool,
    signal_quantile: float = 0.70,
    common_margin: float = 0.05,
    eps: float = 1e-8,
) -> dict[str, torch.Tensor]:
    """Observational DeepEMD diagnostics; returned values never alter logits."""
    if similarity.dim() != 4 or flow.shape != similarity.shape:
        raise ValueError("similarity and flow must share shape (Nq, Way, Lq, Ls)")
    num_query, way_num, query_len, support_len = similarity.shape
    if query_weights.shape != (num_query, way_num, query_len):
        raise ValueError("query_weights shape does not match similarity")
    if support_weights.shape != (way_num, num_query, support_len):
        raise ValueError("support_weights shape does not match similarity")
    if support_images.dim() != 5 or support_images.shape[0] != way_num:
        raise ValueError("support_images must have shape (Way, Shot, C, H, W)")
    if not 0.0 < signal_quantile < 1.0:
        raise ValueError("signal_quantile must be in (0, 1)")

    query_mask = _signal_mask(
        query_images,
        token_hw=_infer_token_hw(query_len),
        quantile=signal_quantile,
        eps=eps,
    )
    support_proto_images = support_images.mean(dim=1)
    support_mask = _signal_mask(
        support_proto_images,
        token_hw=_infer_token_hw(support_len),
        quantile=signal_quantile,
        eps=eps,
    )
    if query_mask.shape[-1] != query_len or support_mask.shape[-1] != support_len:
        raise ValueError("DeepEMD diagnostics require rectangular token grids")

    mass = flow.detach().clamp_min(0.0)
    similarity_detached = similarity.detach()
    total_mass = mass.sum(dim=(-1, -2)).clamp_min(eps)
    if normalize_score_by_mass:
        score_scale = float(temperature) / total_mass
    else:
        score_scale = mass.new_full(total_mass.shape, float(temperature) / float(query_len))

    query_fg = query_mask[:, None, :, None]
    support_fg = support_mask[None, :, None, :]
    fg_fg = query_fg & support_fg
    bg_bg = (~query_fg) & (~support_fg)
    background_involved = ~fg_fg
    query_background = ~query_fg

    edge_score = mass * similarity_detached

    def _mass_ratio(mask: torch.Tensor) -> torch.Tensor:
        return (mass * mask.to(mass.dtype)).sum(dim=(-1, -2)) / total_mass

    def _masked_score(mask: torch.Tensor) -> torch.Tensor:
        return (edge_score * mask.to(edge_score.dtype)).sum(dim=(-1, -2)) * score_scale

    query_weight_mass = query_weights.detach().clamp_min(0.0)
    support_weight_mass = support_weights.detach().permute(1, 0, 2).clamp_min(0.0)
    query_weight_signal_ratio = (
        query_weight_mass * query_mask[:, None, :].to(query_weight_mass.dtype)
    ).sum(dim=-1) / query_weight_mass.sum(dim=-1).clamp_min(eps)
    support_weight_signal_ratio = (
        support_weight_mass * support_mask[None, :, :].to(support_weight_mass.dtype)
    ).sum(dim=-1) / support_weight_mass.sum(dim=-1).clamp_min(eps)

    query_row_mass = mass.sum(dim=-1)
    support_column_mass = mass.sum(dim=-2)
    query_transport_signal_ratio = (
        query_row_mass * query_mask[:, None, :].to(query_row_mass.dtype)
    ).sum(dim=-1) / total_mass
    support_transport_signal_ratio = (
        support_column_mass * support_mask[None, :, :].to(support_column_mass.dtype)
    ).sum(dim=-1) / total_mass

    best_similarity_by_class = similarity_detached.amax(dim=-1)
    if way_num > 1:
        top_two = best_similarity_by_class.topk(k=2, dim=1).values
        common_query_mask = (top_two[:, 0] - top_two[:, 1]) <= float(common_margin)
    else:
        common_query_mask = torch.zeros(
            num_query,
            query_len,
            dtype=torch.bool,
            device=similarity.device,
        )
    common_query_mass_ratio = (
        query_row_mass * common_query_mask[:, None, :].to(query_row_mass.dtype)
    ).sum(dim=-1) / total_mass

    query_entropy = _normalized_entropy(query_row_mass, eps)
    support_entropy = _normalized_entropy(support_column_mass, eps)
    query_weight_entropy = _normalized_entropy(query_weight_mass, eps)
    support_weight_entropy = _normalized_entropy(support_weight_mass, eps)

    signal_only_score = _masked_score(fg_fg)
    background_only_score = _masked_score(bg_bg)
    background_involved_score = _masked_score(background_involved)
    query_background_score = _masked_score(query_background)

    fg_fg_mass_ratio = _mass_ratio(fg_fg)
    bg_bg_mass_ratio = _mass_ratio(bg_bg)
    background_involved_mass_ratio = _mass_ratio(background_involved)

    return {
        "transport_plan": mass,
        "query_token_weights": query_weight_mass,
        "support_token_weights": support_weight_mass,
        "deepemd_signal_only_score": signal_only_score,
        "deepemd_uniform_weight_score": (
            uniform_weight_logits.detach() if uniform_weight_logits is not None else logits.detach()
        ),
        "deepemd_background_only_score": background_only_score,
        "deepemd_background_involved_score": background_involved_score,
        "deepemd_query_background_score": query_background_score,
        "deepemd_fg_fg_mass_ratio": fg_fg_mass_ratio,
        "deepemd_bg_bg_mass_ratio": bg_bg_mass_ratio,
        "deepemd_background_involved_mass_ratio": background_involved_mass_ratio,
        "deepemd_common_query_mass_ratio": common_query_mass_ratio,
        "deepemd_query_weight_signal_ratio": query_weight_signal_ratio,
        "deepemd_support_weight_signal_ratio": support_weight_signal_ratio,
        "deepemd_query_transport_signal_ratio": query_transport_signal_ratio,
        "deepemd_support_transport_signal_ratio": support_transport_signal_ratio,
        "deepemd_query_transport_entropy": query_entropy,
        "deepemd_support_transport_entropy": support_entropy,
        "deepemd_query_weight_entropy": query_weight_entropy,
        "deepemd_support_weight_entropy": support_weight_entropy,
        "deepemd/debug_enabled": logits.new_tensor(1.0),
        "deepemd/signal_quantile": logits.new_tensor(float(signal_quantile)),
        "deepemd/signal_token_fraction": query_mask.float().mean(),
        "deepemd/fg_fg_mass_ratio": fg_fg_mass_ratio.mean(),
        "deepemd/bg_bg_mass_ratio": bg_bg_mass_ratio.mean(),
        "deepemd/background_involved_mass_ratio": background_involved_mass_ratio.mean(),
        "deepemd/common_query_mass_ratio": common_query_mass_ratio.mean(),
        "deepemd/query_weight_signal_ratio": query_weight_signal_ratio.mean(),
        "deepemd/support_weight_signal_ratio": support_weight_signal_ratio.mean(),
        "deepemd/query_transport_signal_ratio": query_transport_signal_ratio.mean(),
        "deepemd/support_transport_signal_ratio": support_transport_signal_ratio.mean(),
        "deepemd/query_transport_entropy": query_entropy.mean(),
        "deepemd/support_transport_entropy": support_entropy.mean(),
        "deepemd/query_weight_entropy": query_weight_entropy.mean(),
        "deepemd/support_weight_entropy": support_weight_entropy.mean(),
        "deepemd/score_reconstruction_error": (logits.detach() - edge_score.sum(dim=(-1, -2)) * score_scale)
        .abs()
        .mean(),
    }


__all__ = ["compute_deepemd_diagnostics"]
