"""Posterior residual transport flow for SC-LFI v3.

Core formulas:
- posterior predictive measure:
  `muhat_c = (T_theta,c)_# mu_c^0`
- conditional velocity:
  `v_theta(y, t; h_c, M_c, t_ctx)`

Paper grounding:
- fixed-step Euler/Heun integration follows practical latent flow solvers
- linear conditional flow-matching paths remain valid for supervised transport

Our adaptation / novelty:
- the transported object is the posterior base measure, not Gaussian noise
- masses are preserved under deterministic pushforward
- the residual field is initialized near identity to reduce low-shot overfitting
"""

from __future__ import annotations

import torch
import torch.nn as nn

from net.modules.conditional_flow_v2 import (
    FixedStepFlowSolverV2,
    SinusoidalTimeEmbeddingV2,
    SupportMemoryAttentionSummaryV2,
    sample_flow_times_v2,
)


def sample_posterior_conditional_path_v3(
    base_atoms: torch.Tensor,
    target_atoms: torch.Tensor,
    time_values: torch.Tensor,
) -> torch.Tensor:
    """Return `y_t = (1 - t) x_0 + t x_1` for posterior transport pairs."""
    if base_atoms.shape != target_atoms.shape:
        raise ValueError(
            "base_atoms and target_atoms must match exactly: "
            f"base={tuple(base_atoms.shape)} target={tuple(target_atoms.shape)}"
        )
    if time_values.shape != base_atoms.shape[:-1]:
        raise ValueError(
            "time_values must match leading atom dims: "
            f"time={tuple(time_values.shape)} atoms={tuple(base_atoms.shape)}"
        )
    return (1.0 - time_values.unsqueeze(-1)) * base_atoms + time_values.unsqueeze(-1) * target_atoms


def target_posterior_transport_velocity_v3(
    base_atoms: torch.Tensor,
    target_atoms: torch.Tensor,
) -> torch.Tensor:
    """Return exact target velocity `u_t = x_1 - x_0`."""
    if base_atoms.shape != target_atoms.shape:
        raise ValueError(
            "base_atoms and target_atoms must match exactly: "
            f"base={tuple(base_atoms.shape)} target={tuple(target_atoms.shape)}"
        )
    return target_atoms - base_atoms


class PosteriorTransportVelocityFieldV3(nn.Module):
    """Residual velocity field over posterior base atoms.

    Shapes:
    - states: `[Batch, LatentDim]`
    - class_summary: `[Batch, ContextDim]`
    - support_memory: `[Batch, MemoryTokens, LatentDim]`
    - episode_context: `[Batch, ContextDim]`
    """

    def __init__(
        self,
        latent_dim: int,
        context_dim: int,
        hidden_dim: int,
        time_embedding_dim: int,
        memory_num_heads: int = 4,
        conditioning_type: str = "film",
    ) -> None:
        super().__init__()
        if latent_dim <= 0 or context_dim <= 0 or hidden_dim <= 0 or time_embedding_dim <= 0:
            raise ValueError("latent_dim, context_dim, hidden_dim, and time_embedding_dim must be positive")
        conditioning_type = str(conditioning_type).lower()
        if conditioning_type not in {"concat", "film"}:
            raise ValueError(f"Unsupported conditioning_type: {conditioning_type}")

        self.conditioning_type = conditioning_type
        self.time_embedding = SinusoidalTimeEmbeddingV2(int(time_embedding_dim))
        self.state_trunk = nn.Sequential(
            nn.LayerNorm(latent_dim + time_embedding_dim),
            nn.Linear(latent_dim + time_embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.memory_attention = SupportMemoryAttentionSummaryV2(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_heads=memory_num_heads,
        )
        self.context_proj = nn.Sequential(
            nn.LayerNorm(context_dim * 2),
            nn.Linear(context_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.pre_condition_norm = nn.LayerNorm(hidden_dim)

        if self.conditioning_type == "concat":
            self.conditioner = nn.Sequential(
                nn.LayerNorm(hidden_dim * 3),
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        else:
            self.film = nn.Linear(hidden_dim, hidden_dim * 2)
            self.conditioner = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        # Start near identity transport so low-shot episodes do not immediately
        # drift into an overfit transport field.
        nn.init.zeros_(self.output_head[-1].weight)
        nn.init.zeros_(self.output_head[-1].bias)

    def forward(
        self,
        states: torch.Tensor,
        time_values: torch.Tensor,
        class_summary: torch.Tensor,
        support_memory: torch.Tensor,
        episode_context: torch.Tensor,
        *,
        encoded_memory: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if states.dim() != 2:
            raise ValueError(f"states must have shape (Batch, LatentDim), got {tuple(states.shape)}")
        if time_values.dim() != 1:
            raise ValueError(f"time_values must have shape (Batch,), got {tuple(time_values.shape)}")
        if class_summary.dim() != 2 or episode_context.dim() != 2:
            raise ValueError("class_summary and episode_context must have shape (Batch, ContextDim)")
        if support_memory.dim() != 3:
            raise ValueError(
                f"support_memory must have shape (Batch, MemoryTokens, LatentDim), got {tuple(support_memory.shape)}"
            )
        batch_size = states.shape[0]
        if (
            time_values.shape[0] != batch_size
            or class_summary.shape[0] != batch_size
            or episode_context.shape[0] != batch_size
            or support_memory.shape[0] != batch_size
        ):
            raise ValueError("all inputs must share batch size")

        time_emb = self.time_embedding(time_values)
        state_hidden = self.state_trunk(torch.cat([states, time_emb], dim=-1))
        memory_summary = self.memory_attention(state_hidden, support_memory, encoded_memory=encoded_memory)
        context_hidden = self.context_proj(torch.cat([class_summary, episode_context], dim=-1))

        if self.conditioning_type == "concat":
            conditioned = self.conditioner(torch.cat([state_hidden, memory_summary, context_hidden], dim=-1))
        else:
            combined = self.pre_condition_norm(state_hidden + memory_summary)
            scale, shift = self.film(context_hidden).chunk(2, dim=-1)
            combined = (1.0 + scale) * combined + shift
            conditioned = self.conditioner(combined) + combined
        return self.output_head(conditioned)


class PosteriorTransportFlowModelV3(nn.Module):
    """Transport posterior base atoms with a support-conditioned residual flow."""

    def __init__(
        self,
        latent_dim: int,
        context_dim: int,
        hidden_dim: int = 128,
        time_embedding_dim: int = 32,
        memory_num_heads: int = 4,
        conditioning_type: str = "film",
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.velocity_field = PosteriorTransportVelocityFieldV3(
            latent_dim=latent_dim,
            context_dim=context_dim,
            hidden_dim=hidden_dim,
            time_embedding_dim=time_embedding_dim,
            memory_num_heads=memory_num_heads,
            conditioning_type=conditioning_type,
        )

    def forward(
        self,
        states: torch.Tensor,
        time_values: torch.Tensor,
        class_summary: torch.Tensor,
        support_memory: torch.Tensor,
        episode_context: torch.Tensor,
        *,
        encoded_memory: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.velocity_field(
            states,
            time_values,
            class_summary,
            support_memory,
            episode_context,
            encoded_memory=encoded_memory,
        )

    def sample_flow_matching_inputs(
        self,
        base_atoms: torch.Tensor,
        target_atoms: torch.Tensor,
        *,
        schedule: str = "uniform",
    ) -> dict[str, torch.Tensor]:
        if base_atoms.dim() != 2 or target_atoms.dim() != 2:
            raise ValueError("base_atoms and target_atoms must have shape (Batch, LatentDim)")
        time_values = sample_flow_times_v2(
            (base_atoms.shape[0],),
            device=base_atoms.device,
            dtype=base_atoms.dtype,
            schedule=schedule,
        )
        path_states = sample_posterior_conditional_path_v3(base_atoms, target_atoms, time_values)
        target_velocity = target_posterior_transport_velocity_v3(base_atoms, target_atoms)
        return {
            "time_values": time_values,
            "path_states": path_states,
            "target_velocity": target_velocity,
        }

    def transport(
        self,
        base_atoms: torch.Tensor,
        class_summary: torch.Tensor,
        support_memory: torch.Tensor,
        episode_context: torch.Tensor,
        *,
        num_steps: int,
        solver_type: str = "heun",
    ) -> torch.Tensor:
        if base_atoms.dim() != 3:
            raise ValueError(
                f"base_atoms must have shape (Way, BaseAtoms, LatentDim), got {tuple(base_atoms.shape)}"
            )
        if class_summary.dim() != 2 or episode_context.dim() != 2:
            raise ValueError("class_summary and episode_context must have shape (Way, ContextDim)")
        if support_memory.dim() != 3:
            raise ValueError(
                f"support_memory must have shape (Way, MemoryTokens, LatentDim), got {tuple(support_memory.shape)}"
            )
        if base_atoms.shape[0] != class_summary.shape[0] or base_atoms.shape[0] != support_memory.shape[0]:
            raise ValueError("base_atoms, class_summary, and support_memory must share class dimension")
        if episode_context.shape[0] != class_summary.shape[0]:
            raise ValueError("episode_context must share class dimension")
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")

        way_num, num_atoms, _ = base_atoms.shape
        flat_states = base_atoms.reshape(way_num * num_atoms, self.latent_dim)
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
        encoded_memory = self.velocity_field.memory_attention.encode_memory(flat_memory)
        solver = FixedStepFlowSolverV2(solver_type=solver_type)

        def velocity_fn(states: torch.Tensor, time_values: torch.Tensor) -> torch.Tensor:
            return self(
                states,
                time_values,
                flat_summary,
                flat_memory,
                flat_episode,
                encoded_memory=encoded_memory,
            )

        transported = solver.integrate(
            velocity_fn,
            flat_states,
            num_steps=num_steps,
            t_start=0.0,
            t_end=1.0,
        )
        return transported.reshape(way_num, num_atoms, self.latent_dim)
