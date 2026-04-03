"""Memory-conditioned latent flow and fixed-step solvers for SC-LFI v2."""

from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


def sample_flow_times_v2(
    shape: tuple[int, ...],
    *,
    device: torch.device,
    dtype: torch.dtype,
    schedule: str = "uniform",
) -> torch.Tensor:
    """Sample flow-matching times.

    Paper grounding:
    - follows the flow-matching practice of sampling scalar times on `[0, 1]`.

    Our few-shot adaptation:
    - we expose schedule control, but keep the standard uniform schedule as the
      default.
    """

    schedule = str(schedule).lower()
    if schedule == "uniform":
        return torch.rand(shape, device=device, dtype=dtype)
    if schedule == "beta":
        beta = torch.distributions.Beta(0.5, 0.5)
        return beta.sample(shape).to(device=device, dtype=dtype)
    raise ValueError(f"Unsupported flow time schedule: {schedule}")


def sample_linear_conditional_path_v2(
    noise: torch.Tensor,
    evidence: torch.Tensor,
    time_values: torch.Tensor,
) -> torch.Tensor:
    """Return `y_t = (1 - t) epsilon + t e`.

    This is our few-shot adaptation of an affine conditional probability path.
    """

    if noise.shape != evidence.shape:
        raise ValueError(
            "noise and evidence must match exactly: "
            f"noise={tuple(noise.shape)} evidence={tuple(evidence.shape)}"
        )
    if time_values.shape != evidence.shape[:-1]:
        raise ValueError(
            "time_values must match the leading evidence shape: "
            f"time={tuple(time_values.shape)} evidence={tuple(evidence.shape)}"
        )
    return (1.0 - time_values.unsqueeze(-1)) * noise + time_values.unsqueeze(-1) * evidence


def target_linear_path_velocity_v2(
    noise: torch.Tensor,
    evidence: torch.Tensor,
) -> torch.Tensor:
    """Return the exact target velocity `u_t = e - epsilon`."""

    if noise.shape != evidence.shape:
        raise ValueError(
            "noise and evidence must match exactly: "
            f"noise={tuple(noise.shape)} evidence={tuple(evidence.shape)}"
        )
    return evidence - noise


class SinusoidalTimeEmbeddingV2(nn.Module):
    """Fixed sinusoidal embedding for scalar time values."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        self.embedding_dim = int(embedding_dim)

    def forward(self, time_values: torch.Tensor) -> torch.Tensor:
        if time_values.dim() != 1:
            raise ValueError(f"time_values must have shape (Batch,), got {tuple(time_values.shape)}")
        half_dim = self.embedding_dim // 2
        if half_dim == 0:
            return time_values.unsqueeze(-1)
        exponent = torch.linspace(
            0.0,
            1.0,
            steps=half_dim,
            device=time_values.device,
            dtype=time_values.dtype,
        )
        frequencies = torch.exp(math.log(1000.0) * exponent)
        angles = (2.0 * math.pi) * time_values.unsqueeze(-1) * frequencies.unsqueeze(0)
        embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        if embedding.shape[-1] < self.embedding_dim:
            embedding = F.pad(embedding, (0, self.embedding_dim - embedding.shape[-1]), value=0.0)
        return embedding


class FixedStepFlowSolverV2:
    """Fixed-step ODE solver for latent particle transport.

    Paper grounding:
    - solver separation follows the official `flow_matching` ODE solver design;
    - fixed-step Euler/Heun are consistent with practical latent flow samplers
      used in LFM.

    Engineering approximation:
    - this is still a numerical ODE approximation, not an exact continuous
      solution.
    """

    def __init__(self, solver_type: str = "heun") -> None:
        solver_type = str(solver_type).lower()
        if solver_type not in {"euler", "heun"}:
            raise ValueError(f"Unsupported solver_type: {solver_type}")
        self.solver_type = solver_type

    def integrate(
        self,
        velocity_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        initial_states: torch.Tensor,
        *,
        num_steps: int,
        t_start: float = 0.0,
        t_end: float = 1.0,
    ) -> torch.Tensor:
        if initial_states.dim() != 2:
            raise ValueError(
                f"initial_states must have shape (Batch, LatentDim), got {tuple(initial_states.shape)}"
            )
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")

        states = initial_states
        dt = float(t_end - t_start) / float(num_steps)
        for step_idx in range(num_steps):
            t0 = t_start + float(step_idx) * dt
            t1 = t0 + dt
            t0_batch = states.new_full((states.shape[0],), t0)

            if self.solver_type == "euler":
                k1 = velocity_fn(states, t0_batch)
                states = states + dt * k1
                continue

            k1 = velocity_fn(states, t0_batch)
            predicted = states + dt * k1
            t1_batch = states.new_full((states.shape[0],), t1)
            k2 = velocity_fn(predicted, t1_batch)
            states = states + 0.5 * dt * (k1 + k2)
        return states


class SupportMemoryAttentionSummaryV2(nn.Module):
    """Lightweight cross-attention summary from a latent state to support memory.

    Formula:
    - `Q = W_q g`
    - `K = W_k M_c`
    - `V = W_v M_c`
    - `m = softmax(Q K^T / sqrt(d_h)) V`

    Engineering note:
    - this replaces generic `nn.MultiheadAttention` with a lean single-query
      attention path, which is cheaper inside repeated ODE integration while
      preserving the intended support-memory conditioning.
    """

    def __init__(self, latent_dim: int, hidden_dim: int, num_heads: int = 4) -> None:
        super().__init__()
        if latent_dim <= 0 or hidden_dim <= 0:
            raise ValueError("latent_dim and hidden_dim must be positive")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads) if hidden_dim % num_heads == 0 else 1
        self.head_dim = self.hidden_dim // self.num_heads
        self.scale = self.head_dim ** -0.5

        self.memory_adapter = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, hidden_dim),
        )
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def encode_memory(self, support_memory: torch.Tensor) -> torch.Tensor:
        if support_memory.dim() != 3:
            raise ValueError(
                f"support_memory must have shape (Batch, MemoryTokens, LatentDim), got {tuple(support_memory.shape)}"
            )
        return self.memory_adapter(support_memory)

    def forward(
        self,
        query_hidden: torch.Tensor,
        support_memory: torch.Tensor,
        *,
        encoded_memory: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if query_hidden.dim() != 2:
            raise ValueError(
                f"query_hidden must have shape (Batch, HiddenDim), got {tuple(query_hidden.shape)}"
            )
        memory_hidden = encoded_memory if encoded_memory is not None else self.encode_memory(support_memory)
        if memory_hidden.dim() != 3:
            raise ValueError(
                f"memory_hidden must have shape (Batch, MemoryTokens, HiddenDim), got {tuple(memory_hidden.shape)}"
            )
        if memory_hidden.shape[0] != query_hidden.shape[0]:
            raise ValueError("query_hidden and memory_hidden must share batch size")

        batch_size, num_memory_tokens, _ = memory_hidden.shape
        query = self.query_proj(query_hidden).reshape(batch_size, self.num_heads, self.head_dim)
        keys = self.key_proj(memory_hidden).reshape(batch_size, num_memory_tokens, self.num_heads, self.head_dim)
        values = self.value_proj(memory_hidden).reshape(batch_size, num_memory_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        attention_logits = (query.unsqueeze(-2) * keys).sum(dim=-1) * self.scale
        attention_weights = torch.softmax(attention_logits, dim=-1)
        summary = (attention_weights.unsqueeze(-1) * values).sum(dim=-2)
        return self.out_proj(summary.reshape(batch_size, self.hidden_dim))


class SupportMemoryConditionedVelocityFieldV2(nn.Module):
    """Velocity field `v_theta(y, t; h_c, M_c)` for SC-LFI v2.

    Formula:
    - trunk state:
      `g = Trunk([y, gamma(t)])`
    - memory summary:
      `m = Attn(q = Q(g), K = K(M_c), V = V(M_c))`
    - conditioned hidden:
      `g' = Condition(g, m, h_c)`
    - velocity:
      `v = Head(g')`

    Conditioning modes:
    - `concat`:
      - concatenate `g`, attention summary `m`, and `h_c`
    - `film`:
      - add attention summary to `g`, then apply FiLM from `h_c`
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
        self.latent_dim = int(latent_dim)
        self.context_dim = int(context_dim)
        self.hidden_dim = int(hidden_dim)
        self.conditioning_type = str(conditioning_type).lower()
        if self.conditioning_type not in {"concat", "film"}:
            raise ValueError(f"Unsupported conditioning_type: {conditioning_type}")

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
        self.pre_condition_norm = nn.LayerNorm(hidden_dim)

        if self.conditioning_type == "concat":
            self.conditioner = nn.Sequential(
                nn.LayerNorm(hidden_dim * 2 + context_dim),
                nn.Linear(hidden_dim * 2 + context_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        else:
            self.film = nn.Linear(context_dim, hidden_dim * 2)
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

    def forward(
        self,
        states: torch.Tensor,
        time_values: torch.Tensor,
        class_summary: torch.Tensor,
        support_memory: torch.Tensor,
        *,
        encoded_memory: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if states.dim() != 2:
            raise ValueError(f"states must have shape (Batch, LatentDim), got {tuple(states.shape)}")
        if time_values.dim() != 1:
            raise ValueError(f"time_values must have shape (Batch,), got {tuple(time_values.shape)}")
        if class_summary.dim() != 2:
            raise ValueError(
                f"class_summary must have shape (Batch, ContextDim), got {tuple(class_summary.shape)}"
            )
        if support_memory.dim() != 3:
            raise ValueError(
                f"support_memory must have shape (Batch, MemoryTokens, LatentDim), got {tuple(support_memory.shape)}"
            )
        batch_size = states.shape[0]
        if time_values.shape[0] != batch_size or class_summary.shape[0] != batch_size or support_memory.shape[0] != batch_size:
            raise ValueError("states, time_values, class_summary, and support_memory must share batch size")

        time_embedding = self.time_embedding(time_values)
        state_hidden = self.state_trunk(torch.cat([states, time_embedding], dim=-1))
        attn_summary = self.memory_attention(
            state_hidden,
            support_memory,
            encoded_memory=encoded_memory,
        )

        if self.conditioning_type == "concat":
            conditioned = self.conditioner(torch.cat([state_hidden, attn_summary, class_summary], dim=-1))
        else:
            combined = self.pre_condition_norm(state_hidden + attn_summary)
            scale, shift = self.film(class_summary).chunk(2, dim=-1)
            combined = (1.0 + scale) * combined + shift
            conditioned = self.conditioner(combined) + combined

        return self.output_head(conditioned)


class SupportConditionedParticleMassPredictorV2(nn.Module):
    """Predict non-uniform flow particle masses for the generated class measure.

    Formula:
    - particle hidden: `g = Phi_particle(y)`
    - memory summary: `m = Attn(g, M_c)`
    - context bias: `c = W_h h_c`
    - particle logit: `r = W_mass([g, m, c])`
    - normalized masses: `pi = softmax(r)`
    """

    def __init__(
        self,
        latent_dim: int,
        context_dim: int,
        hidden_dim: int,
        memory_num_heads: int = 4,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if latent_dim <= 0 or context_dim <= 0 or hidden_dim <= 0:
            raise ValueError("latent_dim, context_dim, and hidden_dim must be positive")
        if eps <= 0.0:
            raise ValueError("eps must be positive")
        self.eps = float(eps)
        self.particle_proj = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.memory_attention = SupportMemoryAttentionSummaryV2(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_heads=memory_num_heads,
        )
        self.context_proj = nn.Linear(context_dim, hidden_dim)
        self.mass_head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def _normalize_masses(self, mass_logits: torch.Tensor) -> torch.Tensor:
        masses = torch.softmax(mass_logits, dim=-1).clamp_min(0.0)
        normalizer = masses.sum(dim=-1, keepdim=True)
        zero_rows = normalizer <= self.eps
        if zero_rows.any():
            uniform = torch.full_like(masses, 1.0 / float(masses.shape[-1]))
            masses = torch.where(zero_rows, uniform, masses)
            normalizer = masses.sum(dim=-1, keepdim=True)
        return masses / normalizer.clamp_min(self.eps)

    def forward(
        self,
        particles: torch.Tensor,
        class_summary: torch.Tensor,
        support_memory: torch.Tensor,
    ) -> torch.Tensor:
        if particles.dim() != 3:
            raise ValueError(
                f"particles must have shape (Way, Particles, LatentDim), got {tuple(particles.shape)}"
            )
        if class_summary.dim() != 2:
            raise ValueError(
                f"class_summary must have shape (Way, ContextDim), got {tuple(class_summary.shape)}"
            )
        if support_memory.dim() != 3:
            raise ValueError(
                f"support_memory must have shape (Way, MemoryTokens, LatentDim), got {tuple(support_memory.shape)}"
            )
        if particles.shape[0] != class_summary.shape[0] or particles.shape[0] != support_memory.shape[0]:
            raise ValueError("particles, class_summary, and support_memory must share class dimension")

        way_num, num_particles, _ = particles.shape
        flat_particles = particles.reshape(way_num * num_particles, particles.shape[-1])
        flat_summary = class_summary.unsqueeze(1).expand(-1, num_particles, -1).reshape(
            way_num * num_particles,
            class_summary.shape[-1],
        )
        flat_memory = support_memory.unsqueeze(1).expand(-1, num_particles, -1, -1).reshape(
            way_num * num_particles,
            support_memory.shape[1],
            support_memory.shape[2],
        )

        particle_hidden = self.particle_proj(flat_particles)
        memory_summary = self.memory_attention(particle_hidden, flat_memory)
        context_hidden = self.context_proj(flat_summary)
        logits = self.mass_head(
            torch.cat([particle_hidden, memory_summary, context_hidden], dim=-1)
        ).reshape(way_num, num_particles)
        return self._normalize_masses(logits)


class ConditionalLatentFlowModelV2(nn.Module):
    """Support-memory-conditioned latent flow for SC-LFI v2."""

    def __init__(
        self,
        latent_dim: int,
        context_dim: int,
        hidden_dim: int = 128,
        time_embedding_dim: int = 32,
        memory_num_heads: int = 4,
        conditioning_type: str = "film",
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.velocity_field = SupportMemoryConditionedVelocityFieldV2(
            latent_dim=latent_dim,
            context_dim=context_dim,
            hidden_dim=hidden_dim,
            time_embedding_dim=time_embedding_dim,
            memory_num_heads=memory_num_heads,
            conditioning_type=conditioning_type,
        )
        self.particle_mass_predictor = SupportConditionedParticleMassPredictorV2(
            latent_dim=latent_dim,
            context_dim=context_dim,
            hidden_dim=hidden_dim,
            memory_num_heads=memory_num_heads,
            eps=eps,
        )

    def forward(
        self,
        states: torch.Tensor,
        time_values: torch.Tensor,
        class_summary: torch.Tensor,
        support_memory: torch.Tensor,
        *,
        encoded_memory: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.velocity_field(
            states,
            time_values,
            class_summary,
            support_memory,
            encoded_memory=encoded_memory,
        )

    def sample_flow_matching_inputs(
        self,
        evidence: torch.Tensor,
        *,
        schedule: str = "uniform",
    ) -> dict[str, torch.Tensor]:
        if evidence.dim() != 2:
            raise ValueError(f"evidence must have shape (Batch, LatentDim), got {tuple(evidence.shape)}")
        noise = torch.randn_like(evidence)
        time_values = sample_flow_times_v2(
            (evidence.shape[0],),
            device=evidence.device,
            dtype=evidence.dtype,
            schedule=schedule,
        )
        path_states = sample_linear_conditional_path_v2(noise, evidence, time_values)
        target_velocity = target_linear_path_velocity_v2(noise, evidence)
        return {
            "noise": noise,
            "time_values": time_values,
            "path_states": path_states,
            "target_velocity": target_velocity,
        }

    def sample_particles(
        self,
        class_summary: torch.Tensor,
        support_memory: torch.Tensor,
        *,
        num_particles: int,
        num_steps: int,
        solver_type: str = "heun",
        base_noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if class_summary.dim() != 2:
            raise ValueError(
                f"class_summary must have shape (Way, ContextDim), got {tuple(class_summary.shape)}"
            )
        if support_memory.dim() != 3:
            raise ValueError(
                f"support_memory must have shape (Way, MemoryTokens, LatentDim), got {tuple(support_memory.shape)}"
            )
        if class_summary.shape[0] != support_memory.shape[0]:
            raise ValueError("class_summary and support_memory must share the class dimension")
        if num_particles <= 0 or num_steps <= 0:
            raise ValueError("num_particles and num_steps must be positive")

        way_num = class_summary.shape[0]
        if base_noise is None:
            particles = torch.randn(
                way_num,
                num_particles,
                self.latent_dim,
                device=class_summary.device,
                dtype=class_summary.dtype,
            )
        else:
            expected_shape = (way_num, num_particles, self.latent_dim)
            if tuple(base_noise.shape) != expected_shape:
                raise ValueError(f"base_noise must have shape {expected_shape}, got {tuple(base_noise.shape)}")
            particles = base_noise

        flat_states = particles.reshape(way_num * num_particles, self.latent_dim)
        flat_summary = class_summary.unsqueeze(1).expand(-1, num_particles, -1).reshape(
            way_num * num_particles,
            class_summary.shape[-1],
        )
        flat_memory = support_memory.unsqueeze(1).expand(-1, num_particles, -1, -1).reshape(
            way_num * num_particles,
            support_memory.shape[1],
            support_memory.shape[2],
        )
        solver = FixedStepFlowSolverV2(solver_type=solver_type)
        encoded_memory = self.velocity_field.memory_attention.encode_memory(flat_memory)

        def velocity_fn(states: torch.Tensor, time_values: torch.Tensor) -> torch.Tensor:
            return self(
                states,
                time_values,
                flat_summary,
                flat_memory,
                encoded_memory=encoded_memory,
            )

        integrated = solver.integrate(
            velocity_fn,
            flat_states,
            num_steps=num_steps,
            t_start=0.0,
            t_end=1.0,
        )
        return integrated.reshape(way_num, num_particles, self.latent_dim)

    def estimate_particle_masses(
        self,
        particles: torch.Tensor,
        class_summary: torch.Tensor,
        support_memory: torch.Tensor,
    ) -> torch.Tensor:
        return self.particle_mass_predictor(particles, class_summary, support_memory)
