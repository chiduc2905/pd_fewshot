"""Loss helpers for SC-LFI."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_flow_matching_loss(
    flow_model,
    support_latents: torch.Tensor,
    class_contexts: torch.Tensor,
    *,
    time_schedule: str = "uniform",
) -> torch.Tensor:
    """Compute the few-shot flow-matching objective on support latent evidence."""

    if support_latents.dim() != 3:
        raise ValueError(
            f"support_latents must have shape (Way, SupportTokens, LatentDim), got {tuple(support_latents.shape)}"
        )
    if class_contexts.dim() != 2:
        raise ValueError(
            f"class_contexts must have shape (Way, ContextDim), got {tuple(class_contexts.shape)}"
        )
    if support_latents.shape[0] != class_contexts.shape[0]:
        raise ValueError(
            "support_latents and class_contexts must share the class dimension: "
            f"support={tuple(support_latents.shape)} context={tuple(class_contexts.shape)}"
        )

    way_num, num_support_tokens, latent_dim = support_latents.shape
    flat_evidence = support_latents.reshape(way_num * num_support_tokens, latent_dim)
    flat_contexts = class_contexts.unsqueeze(1).expand(-1, num_support_tokens, -1).reshape(
        way_num * num_support_tokens,
        class_contexts.shape[-1],
    )
    fm_batch = flow_model.sample_flow_matching_inputs(flat_evidence, schedule=time_schedule)
    predicted_velocity = flow_model(
        fm_batch["path_states"],
        fm_batch["time_values"],
        flat_contexts,
    )
    return F.mse_loss(predicted_velocity, fm_batch["target_velocity"])


def compute_support_anchoring_loss(
    generated_particles: torch.Tensor,
    support_latents: torch.Tensor,
    distance_module,
) -> torch.Tensor:
    """Anchor the sampled class distribution to observed support evidence."""

    if generated_particles.dim() != 3 or support_latents.dim() != 3:
        raise ValueError(
            "generated_particles and support_latents must have shapes "
            "(Way, Particles, LatentDim) and (Way, SupportTokens, LatentDim)"
        )
    return distance_module(generated_particles, support_latents, reduction="mean")


def compute_context_smoothness_loss(
    generated_particles: torch.Tensor,
    class_contexts: torch.Tensor,
    distance_module,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Penalize large distribution shifts between nearest support contexts."""

    if generated_particles.dim() != 3 or class_contexts.dim() != 2:
        raise ValueError(
            "generated_particles and class_contexts must have shapes "
            "(Way, Particles, LatentDim) and (Way, ContextDim)"
        )
    if generated_particles.shape[0] < 2:
        return generated_particles.new_zeros(())

    context_distances = torch.cdist(class_contexts, class_contexts, p=2)
    diagonal_mask = torch.eye(context_distances.shape[0], device=context_distances.device, dtype=torch.bool)
    context_distances = context_distances.masked_fill(diagonal_mask, float("inf"))
    nearest_indices = torch.argmin(context_distances, dim=-1)

    unique_pairs: set[tuple[int, int]] = set()
    for class_idx, neighbor_idx in enumerate(nearest_indices.tolist()):
        if class_idx == neighbor_idx:
            continue
        unique_pairs.add(tuple(sorted((class_idx, neighbor_idx))))

    if not unique_pairs:
        return generated_particles.new_zeros(())

    pair_left = torch.tensor([pair[0] for pair in sorted(unique_pairs)], device=generated_particles.device)
    pair_right = torch.tensor([pair[1] for pair in sorted(unique_pairs)], device=generated_particles.device)

    left_particles = generated_particles.index_select(0, pair_left)
    right_particles = generated_particles.index_select(0, pair_right)
    pair_distances = distance_module(left_particles, right_particles, reduction="none")
    weights = torch.exp(-context_distances[pair_left, pair_right]).to(dtype=pair_distances.dtype)
    return (weights * pair_distances).sum() / weights.sum().clamp_min(eps)
