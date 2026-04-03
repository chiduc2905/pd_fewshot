"""Explicit loss functions for SC-LFI v2."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_flow_matching_loss_v2(
    flow_model,
    support_latents: torch.Tensor,
    class_summary: torch.Tensor,
    support_memory: torch.Tensor,
    support_masses: torch.Tensor | None = None,
    *,
    time_schedule: str = "uniform",
) -> torch.Tensor:
    """Few-shot flow matching with support-memory-conditioned velocity regression.

    Formula:
    - path states: `y_t = (1 - t) epsilon + t e`
    - target velocity: `u_t = e - epsilon`
    - weighted regression:
      `L_FM = sum_i a_i^sup ||v_theta(y_t^i, t; h_c, M_c) - u_t^i||_2^2`

    Our adaptation:
    - support reliability masses weight the per-token flow regression so noisy
      support tokens do not dominate the transport objective.
    """

    if support_latents.dim() != 3:
        raise ValueError(
            f"support_latents must have shape (Way, SupportTokens, LatentDim), got {tuple(support_latents.shape)}"
        )
    if class_summary.dim() != 2:
        raise ValueError(
            f"class_summary must have shape (Way, ContextDim), got {tuple(class_summary.shape)}"
        )
    if support_memory.dim() != 3:
        raise ValueError(
            f"support_memory must have shape (Way, MemoryTokens, LatentDim), got {tuple(support_memory.shape)}"
        )
    if support_latents.shape[0] != class_summary.shape[0] or support_latents.shape[0] != support_memory.shape[0]:
        raise ValueError("support_latents, class_summary, and support_memory must share class dimension")
    if support_masses is not None and support_masses.shape != support_latents.shape[:-1]:
        raise ValueError(
            "support_masses must match support_latents without the latent dim: "
            f"latents={tuple(support_latents.shape)} masses={tuple(support_masses.shape)}"
        )

    way_num, num_support_tokens, latent_dim = support_latents.shape
    flat_evidence = support_latents.reshape(way_num * num_support_tokens, latent_dim)
    flat_summary = class_summary.unsqueeze(1).expand(-1, num_support_tokens, -1).reshape(
        way_num * num_support_tokens,
        class_summary.shape[-1],
    )
    flat_memory = support_memory.unsqueeze(1).expand(-1, num_support_tokens, -1, -1).reshape(
        way_num * num_support_tokens,
        support_memory.shape[1],
        support_memory.shape[2],
    )

    fm_batch = flow_model.sample_flow_matching_inputs(flat_evidence, schedule=time_schedule)
    predicted_velocity = flow_model(
        fm_batch["path_states"],
        fm_batch["time_values"],
        flat_summary,
        flat_memory,
    )
    per_token_loss = (predicted_velocity - fm_batch["target_velocity"]).pow(2).mean(dim=-1).reshape(
        way_num,
        num_support_tokens,
    )
    if support_masses is None:
        return per_token_loss.mean()

    normalized_support_masses = support_masses / support_masses.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return (normalized_support_masses * per_token_loss).sum(dim=-1).mean()


def compute_support_anchoring_loss_v2(
    generated_particles: torch.Tensor,
    support_latents: torch.Tensor,
    *,
    generated_masses: torch.Tensor | None,
    support_masses: torch.Tensor | None,
    alignment_distance,
) -> torch.Tensor:
    """Anchor the generated class measure to the weighted support evidence measure."""

    if generated_particles.dim() != 3 or support_latents.dim() != 3:
        raise ValueError(
            "generated_particles and support_latents must have shapes "
            "(Way, Particles, LatentDim) and (Way, SupportTokens, LatentDim)"
        )
    return alignment_distance(
        generated_particles,
        support_latents,
        source_masses=generated_masses,
        target_masses=support_masses,
        reduction="mean",
    )


def compute_distribution_margin_loss_v2(
    pairwise_distances: torch.Tensor,
    targets: torch.Tensor,
    *,
    margin: float = 0.1,
) -> torch.Tensor:
    """Hard-negative margin on distribution distances.

    Formula:
    - `d_true = D(q, class_true)`
    - `d_neg = min_{c != y} D(q, class_c)`
    - `L_margin = mean relu(margin + d_true - d_neg)`
    """

    if pairwise_distances.dim() != 2:
        raise ValueError(
            f"pairwise_distances must have shape (NumQuery, Way), got {tuple(pairwise_distances.shape)}"
        )
    if targets.dim() != 1 or targets.shape[0] != pairwise_distances.shape[0]:
        raise ValueError(
            "targets must have shape (NumQuery,) matching pairwise_distances. "
            f"targets={tuple(targets.shape)} distances={tuple(pairwise_distances.shape)}"
        )
    if pairwise_distances.shape[1] < 2:
        return pairwise_distances.new_zeros(())

    row_index = torch.arange(pairwise_distances.shape[0], device=pairwise_distances.device)
    true_distances = pairwise_distances[row_index, targets]
    negative_mask = torch.ones_like(pairwise_distances, dtype=torch.bool)
    negative_mask[row_index, targets] = False
    negative_distances = pairwise_distances.masked_fill(~negative_mask, float("inf"))
    hard_negative = negative_distances.min(dim=1).values
    return F.relu(float(margin) + true_distances - hard_negative).mean()


def compute_support_distribution_fit_loss_v2(
    query_latents: torch.Tensor,
    query_masses: torch.Tensor,
    anchor_particles: torch.Tensor,
    anchor_masses: torch.Tensor,
    targets: torch.Tensor,
    *,
    scoring_distance,
    score_temperature: float,
) -> torch.Tensor:
    """Direct CE supervision against the support-anchored class measure.

    Formula:
    - anchor score: `s_c^anchor(q) = -tau D(nu_q, mu_c^anchor)`
    - loss: `L_support-fit = CE(s^anchor(q), y_q)`

    This is not a prototype cosine shortcut. The anchor branch still compares
    weighted latent token measures under the transport scoring distance.
    """

    if query_latents.dim() != 3 or anchor_particles.dim() != 3:
        raise ValueError(
            "query_latents and anchor_particles must have shapes "
            "(NumQuery, QueryTokens, LatentDim) and (Way, AnchorTokens, LatentDim)"
        )
    if query_masses.shape != query_latents.shape[:-1]:
        raise ValueError(
            "query_masses must match query_latents without the latent dim: "
            f"latents={tuple(query_latents.shape)} masses={tuple(query_masses.shape)}"
        )
    if anchor_masses.shape != anchor_particles.shape[:-1]:
        raise ValueError(
            "anchor_masses must match anchor_particles without the latent dim: "
            f"particles={tuple(anchor_particles.shape)} masses={tuple(anchor_masses.shape)}"
        )
    if targets.dim() != 1 or targets.shape[0] != query_latents.shape[0]:
        raise ValueError(
            "targets must have shape (NumQuery,) matching query_latents. "
            f"targets={tuple(targets.shape)} latents={tuple(query_latents.shape)}"
        )

    pairwise_distances = scoring_distance.pairwise_distance(
        query_latents,
        anchor_particles,
        query_masses=query_masses,
        support_masses=anchor_masses,
        reduction="none",
    )
    logits = -float(score_temperature) * pairwise_distances
    return F.cross_entropy(logits, targets)


def compute_context_smoothness_loss_v2(
    generated_particles: torch.Tensor,
    class_summary: torch.Tensor,
    *,
    generated_masses: torch.Tensor | None,
    alignment_distance,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Optional smoothness regularizer over nearby class contexts."""

    if generated_particles.dim() != 3 or class_summary.dim() != 2:
        raise ValueError(
            "generated_particles and class_summary must have shapes "
            "(Way, Particles, LatentDim) and (Way, ContextDim)"
        )
    if generated_particles.shape[0] < 2:
        return generated_particles.new_zeros(())

    context_distances = torch.cdist(class_summary, class_summary, p=2)
    diagonal_mask = torch.eye(context_distances.shape[0], device=context_distances.device, dtype=torch.bool)
    context_distances = context_distances.masked_fill(diagonal_mask, float("inf"))
    nearest_indices = torch.argmin(context_distances, dim=-1)

    unique_pairs: set[tuple[int, int]] = set()
    for class_idx, neighbor_idx in enumerate(nearest_indices.tolist()):
        if class_idx != neighbor_idx:
            unique_pairs.add(tuple(sorted((class_idx, neighbor_idx))))
    if not unique_pairs:
        return generated_particles.new_zeros(())

    pair_left = torch.tensor([pair[0] for pair in sorted(unique_pairs)], device=generated_particles.device)
    pair_right = torch.tensor([pair[1] for pair in sorted(unique_pairs)], device=generated_particles.device)
    left_particles = generated_particles.index_select(0, pair_left)
    right_particles = generated_particles.index_select(0, pair_right)
    if generated_masses is None:
        left_masses = right_masses = None
    else:
        left_masses = generated_masses.index_select(0, pair_left)
        right_masses = generated_masses.index_select(0, pair_right)

    pair_distances = alignment_distance(
        left_particles,
        right_particles,
        source_masses=left_masses,
        target_masses=right_masses,
        reduction="none",
    )
    weights = torch.exp(-context_distances[pair_left, pair_right]).to(dtype=pair_distances.dtype)
    return (weights * pair_distances).sum() / weights.sum().clamp_min(eps)
