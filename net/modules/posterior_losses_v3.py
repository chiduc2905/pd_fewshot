"""Posterior-consistent losses for SC-LFI v3."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from net.modules.transport_distance_v2 import normalize_measure_masses


def compute_distribution_margin_loss_v3(
    pairwise_distances: torch.Tensor,
    targets: torch.Tensor,
    *,
    margin: float = 0.1,
) -> torch.Tensor:
    """Hard-negative margin on final posterior transport distances."""
    if pairwise_distances.dim() != 2:
        raise ValueError(
            f"pairwise_distances must have shape (NumQuery, Way), got {tuple(pairwise_distances.shape)}"
        )
    if targets.dim() != 1 or targets.shape[0] != pairwise_distances.shape[0]:
        raise ValueError("targets must have shape (NumQuery,) matching pairwise_distances")
    if pairwise_distances.shape[1] < 2:
        return pairwise_distances.new_zeros(())

    row_index = torch.arange(pairwise_distances.shape[0], device=pairwise_distances.device)
    true_distances = pairwise_distances[row_index, targets]
    negative_mask = torch.ones_like(pairwise_distances, dtype=torch.bool)
    negative_mask[row_index, targets] = False
    hard_negative = pairwise_distances.masked_fill(~negative_mask, float("inf")).min(dim=1).values
    return F.relu(float(margin) + true_distances - hard_negative).mean()


def _prepare_transport_inputs(
    source: torch.Tensor,
    target: torch.Tensor,
    source_masses: torch.Tensor,
    target_masses: torch.Tensor,
    *,
    normalize_inputs: bool,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if source.dim() != 3 or target.dim() != 3:
        raise ValueError("source and target must have shape (Way, Tokens, Dim)")
    if source.shape[0] != target.shape[0]:
        raise ValueError("source and target must share class dimension")
    if source.shape[-1] != target.shape[-1]:
        raise ValueError("source and target latent dims must match")
    if normalize_inputs:
        source = F.normalize(source, p=2, dim=-1)
        target = F.normalize(target, p=2, dim=-1)
    source_masses = normalize_measure_masses(
        source_masses,
        target_shape=source.shape[:-1],
        device=source.device,
        dtype=source.dtype,
        eps=eps,
    )
    target_masses = normalize_measure_masses(
        target_masses,
        target_shape=target.shape[:-1],
        device=target.device,
        dtype=target.dtype,
        eps=eps,
    )
    return source, target, source_masses, target_masses


def compute_sinkhorn_plan_v3(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    source_masses: torch.Tensor,
    target_masses: torch.Tensor,
    sinkhorn_epsilon: float = 0.05,
    sinkhorn_iterations: int = 80,
    cost_power: float = 2.0,
    normalize_inputs: bool = True,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return log-domain entropic OT plan and transport cost.

    Formula:
    - cost matrix: `C_ij = ||x_i - y_j||_2^p`
    - plan: `P = argmin <P, C> + eps * KL(P || a ⊗ b)`
    """

    if sinkhorn_epsilon <= 0.0 or sinkhorn_iterations <= 0 or cost_power <= 0.0:
        raise ValueError("invalid Sinkhorn hyperparameters")
    source, target, source_masses, target_masses = _prepare_transport_inputs(
        source,
        target,
        source_masses,
        target_masses,
        normalize_inputs=normalize_inputs,
        eps=eps,
    )
    cost_matrix = torch.cdist(source, target, p=2).pow(cost_power)
    log_a = torch.log(source_masses.clamp_min(eps))
    log_b = torch.log(target_masses.clamp_min(eps))
    log_kernel = -cost_matrix / sinkhorn_epsilon

    log_u = torch.zeros_like(log_a)
    log_v = torch.zeros_like(log_b)
    for _ in range(int(sinkhorn_iterations)):
        log_u = log_a - torch.logsumexp(log_kernel + log_v.unsqueeze(-2), dim=-1)
        log_v = log_b - torch.logsumexp(log_kernel + log_u.unsqueeze(-1), dim=-2)

    log_plan = log_kernel + log_u.unsqueeze(-1) + log_v.unsqueeze(-2)
    plan = torch.exp(log_plan)
    cost = (plan * cost_matrix).sum(dim=(-2, -1))
    return plan, cost


def compute_posterior_flow_matching_loss_v3(
    flow_model,
    base_atoms: torch.Tensor,
    base_masses: torch.Tensor,
    support_atoms: torch.Tensor,
    support_masses: torch.Tensor,
    class_summary: torch.Tensor,
    support_memory: torch.Tensor,
    episode_context: torch.Tensor,
    *,
    sinkhorn_epsilon: float = 0.05,
    sinkhorn_iterations: int = 80,
    sinkhorn_cost_power: float = 2.0,
    normalize_inputs: bool = True,
    time_schedule: str = "uniform",
    eps: float = 1e-8,
) -> torch.Tensor:
    """Posterior flow matching from base atoms to support-consistent barycentric targets.

    We first compute a soft coupling between posterior base atoms and support
    empirical atoms, then derive barycentric support targets for each base atom.
    The residual transport field is regressed on linear conditional paths
    between these paired atoms.
    """

    plan, _ = compute_sinkhorn_plan_v3(
        base_atoms,
        support_atoms,
        source_masses=base_masses,
        target_masses=support_masses,
        sinkhorn_epsilon=sinkhorn_epsilon,
        sinkhorn_iterations=sinkhorn_iterations,
        cost_power=sinkhorn_cost_power,
        normalize_inputs=normalize_inputs,
        eps=eps,
    )
    source_mass = plan.sum(dim=-1, keepdim=True).clamp_min(eps)
    barycentric_targets = torch.matmul(plan, support_atoms) / source_mass

    way_num, num_atoms, latent_dim = base_atoms.shape
    flat_base = base_atoms.reshape(way_num * num_atoms, latent_dim)
    flat_target = barycentric_targets.reshape(way_num * num_atoms, latent_dim)
    flat_summary = class_summary.unsqueeze(1).expand(-1, num_atoms, -1).reshape(way_num * num_atoms, class_summary.shape[-1])
    flat_episode = episode_context.unsqueeze(1).expand(-1, num_atoms, -1).reshape(
        way_num * num_atoms,
        episode_context.shape[-1],
    )
    flat_memory = support_memory.unsqueeze(1).expand(-1, num_atoms, -1, -1).reshape(
        way_num * num_atoms,
        support_memory.shape[1],
        support_memory.shape[2],
    )

    fm_batch = flow_model.sample_flow_matching_inputs(flat_base, flat_target, schedule=time_schedule)
    encoded_memory = flow_model.velocity_field.memory_attention.encode_memory(flat_memory)
    predicted_velocity = flow_model(
        fm_batch["path_states"],
        fm_batch["time_values"],
        flat_summary,
        flat_memory,
        flat_episode,
        encoded_memory=encoded_memory,
    )
    per_atom_loss = (predicted_velocity - fm_batch["target_velocity"]).pow(2).mean(dim=-1).reshape(way_num, num_atoms)
    normalized_base_masses = normalize_measure_masses(
        base_masses,
        target_shape=base_masses.shape,
        device=base_atoms.device,
        dtype=base_atoms.dtype,
        eps=eps,
    )
    return (normalized_base_masses * per_atom_loss).sum(dim=-1).mean()


def compute_posterior_alignment_loss_v3(
    posterior_atoms: torch.Tensor,
    posterior_masses: torch.Tensor,
    support_atoms: torch.Tensor,
    support_masses: torch.Tensor,
    *,
    alignment_distance,
) -> torch.Tensor:
    """Align posterior predictive class measures to support empirical measures."""
    return alignment_distance(
        posterior_atoms,
        support_atoms,
        source_masses=posterior_masses,
        target_masses=support_masses,
        reduction="mean",
    )


def compute_entropy_floor_loss_v3(
    masses: torch.Tensor,
    *,
    target_ratio: float = 0.5,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Encourage distributions not to collapse onto one token.

    Formula:
    - entropy: `H(a) = -sum_i a_i log a_i`
    - target floor: `H_target = ratio * log(N)`
    - loss: `relu(H_target - H(a))`
    """
    if not 0.0 <= target_ratio <= 1.0:
        raise ValueError("target_ratio must be in [0, 1]")
    if masses.dim() < 2:
        raise ValueError(f"masses must have at least 2 dims, got {tuple(masses.shape)}")
    normalized = normalize_measure_masses(
        masses,
        target_shape=masses.shape,
        device=masses.device,
        dtype=masses.dtype,
        eps=eps,
    )
    entropy = -(normalized * torch.log(normalized.clamp_min(eps))).sum(dim=-1)
    target_entropy = float(target_ratio) * math.log(float(normalized.shape[-1]))
    return F.relu(normalized.new_tensor(target_entropy) - entropy).mean()


def compute_posterior_regularization_loss_v3(
    support_masses: torch.Tensor,
    query_masses: torch.Tensor,
    query_conditioned_class_masses: torch.Tensor,
    *,
    support_entropy_target_ratio: float = 0.5,
    query_entropy_target_ratio: float = 0.4,
    relevance_entropy_target_ratio: float = 0.35,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Minimal anti-collapse regularization for few-shot evidence masses."""
    support_loss = compute_entropy_floor_loss_v3(
        support_masses,
        target_ratio=support_entropy_target_ratio,
        eps=eps,
    )
    query_loss = compute_entropy_floor_loss_v3(
        query_masses,
        target_ratio=query_entropy_target_ratio,
        eps=eps,
    )
    relevance_loss = compute_entropy_floor_loss_v3(
        query_conditioned_class_masses,
        target_ratio=relevance_entropy_target_ratio,
        eps=eps,
    )
    return support_loss + query_loss + relevance_loss
