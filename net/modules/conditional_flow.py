"""Class-conditioned latent flow utilities for SC-LFI."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sample_flow_times(
    shape: tuple[int, ...],
    *,
    device: torch.device,
    dtype: torch.dtype,
    schedule: str = "uniform",
) -> torch.Tensor:
    """Sample flow-matching times with an explicit schedule hook."""

    schedule = str(schedule).lower()
    if schedule == "uniform":
        return torch.rand(shape, device=device, dtype=dtype)
    if schedule == "beta":
        beta = torch.distributions.Beta(0.5, 0.5)
        return beta.sample(shape).to(device=device, dtype=dtype)
    raise ValueError(f"Unsupported flow-matching time schedule: {schedule}")


def sample_linear_conditional_path(
    noise: torch.Tensor,
    evidence: torch.Tensor,
    time_values: torch.Tensor,
) -> torch.Tensor:
    """Build the linear conditional path `y_t = (1 - t) epsilon + t e`."""

    if noise.shape != evidence.shape:
        raise ValueError(
            "noise and evidence must match exactly: "
            f"noise={tuple(noise.shape)} evidence={tuple(evidence.shape)}"
        )
    if time_values.shape != evidence.shape[:-1]:
        raise ValueError(
            "time_values must match the leading dimensions of evidence: "
            f"time={tuple(time_values.shape)} evidence={tuple(evidence.shape)}"
        )
    time_values = time_values.unsqueeze(-1)
    return (1.0 - time_values) * noise + time_values * evidence


def target_linear_path_velocity(
    noise: torch.Tensor,
    evidence: torch.Tensor,
) -> torch.Tensor:
    """Return the exact target velocity for the linear FM path."""

    if noise.shape != evidence.shape:
        raise ValueError(
            "noise and evidence must match exactly: "
            f"noise={tuple(noise.shape)} evidence={tuple(evidence.shape)}"
        )
    return evidence - noise


class SinusoidalTimeEmbedding(nn.Module):
    """Fixed sinusoidal embedding for scalar flow time values."""

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


class _ConcatVelocityField(nn.Module):
    """Small residual-free velocity MLP conditioned by concatenation."""

    def __init__(
        self,
        latent_dim: int,
        context_dim: int,
        time_embedding_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        input_dim = int(latent_dim + context_dim + time_embedding_dim)
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, y: torch.Tensor, time_embedding: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat([y, time_embedding, context], dim=-1)
        return self.network(inputs)


class _FiLMVelocityField(nn.Module):
    """FiLM-conditioned velocity MLP for support-conditioned flow inference."""

    def __init__(
        self,
        latent_dim: int,
        context_dim: int,
        time_embedding_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.LayerNorm(latent_dim + time_embedding_dim),
            nn.Linear(latent_dim + time_embedding_dim, hidden_dim),
        )
        self.hidden_proj = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_norm_1 = nn.LayerNorm(hidden_dim)
        self.hidden_norm_2 = nn.LayerNorm(hidden_dim)
        self.context_film_1 = nn.Linear(context_dim, hidden_dim * 2)
        self.context_film_2 = nn.Linear(context_dim, hidden_dim * 2)
        self.output = nn.Linear(hidden_dim, latent_dim)

    @staticmethod
    def _apply_film(hidden: torch.Tensor, film_params: torch.Tensor) -> torch.Tensor:
        scale, shift = film_params.chunk(2, dim=-1)
        return (1.0 + scale) * hidden + shift

    def forward(self, y: torch.Tensor, time_embedding: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        hidden = self.input_proj(torch.cat([y, time_embedding], dim=-1))
        hidden = self.hidden_norm_1(hidden)
        hidden = self._apply_film(hidden, self.context_film_1(context))
        hidden = F.gelu(hidden)

        residual = hidden
        hidden = self.hidden_proj(hidden)
        hidden = self.hidden_norm_2(hidden)
        hidden = self._apply_film(hidden, self.context_film_2(context))
        hidden = F.gelu(hidden)
        hidden = hidden + residual
        return self.output(hidden)


class ConditionalLatentFlowModel(nn.Module):
    """Class-conditioned latent flow used for support-conditioned evidence inference.

    The training principle follows flow matching: regress a vector field on a
    fixed conditional path. The specific path over support latent evidence is
    our few-shot adaptation from `latent.md`.
    """

    def __init__(
        self,
        latent_dim: int,
        context_dim: int,
        hidden_dim: int = 128,
        time_embedding_dim: int = 32,
        conditioning_type: str = "concat",
    ) -> None:
        super().__init__()
        if latent_dim <= 0 or context_dim <= 0 or hidden_dim <= 0 or time_embedding_dim <= 0:
            raise ValueError("latent_dim, context_dim, hidden_dim, and time_embedding_dim must be positive")
        self.latent_dim = int(latent_dim)
        self.context_dim = int(context_dim)
        self.time_embedding = SinusoidalTimeEmbedding(int(time_embedding_dim))
        conditioning_type = str(conditioning_type).lower()
        self.conditioning_type = conditioning_type
        if conditioning_type == "concat":
            self.velocity_field = _ConcatVelocityField(
                latent_dim=latent_dim,
                context_dim=context_dim,
                time_embedding_dim=time_embedding_dim,
                hidden_dim=hidden_dim,
            )
        elif conditioning_type == "film":
            self.velocity_field = _FiLMVelocityField(
                latent_dim=latent_dim,
                context_dim=context_dim,
                time_embedding_dim=time_embedding_dim,
                hidden_dim=hidden_dim,
            )
        else:
            raise ValueError(f"Unsupported conditioning_type: {conditioning_type}")

    def forward(
        self,
        states: torch.Tensor,
        time_values: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        if states.dim() != 2:
            raise ValueError(f"states must have shape (Batch, LatentDim), got {tuple(states.shape)}")
        if time_values.dim() != 1:
            raise ValueError(f"time_values must have shape (Batch,), got {tuple(time_values.shape)}")
        if context.dim() != 2:
            raise ValueError(f"context must have shape (Batch, ContextDim), got {tuple(context.shape)}")
        if states.shape[0] != time_values.shape[0] or states.shape[0] != context.shape[0]:
            raise ValueError(
                "states, time_values, and context must share the same batch size: "
                f"states={tuple(states.shape)} time={tuple(time_values.shape)} context={tuple(context.shape)}"
            )
        time_embedding = self.time_embedding(time_values)
        return self.velocity_field(states, time_embedding, context)

    def sample_flow_matching_inputs(
        self,
        evidence: torch.Tensor,
        *,
        schedule: str = "uniform",
    ) -> dict[str, torch.Tensor]:
        if evidence.dim() != 2:
            raise ValueError(f"evidence must have shape (Batch, LatentDim), got {tuple(evidence.shape)}")
        noise = torch.randn_like(evidence)
        time_values = sample_flow_times(
            (evidence.shape[0],),
            device=evidence.device,
            dtype=evidence.dtype,
            schedule=schedule,
        )
        path_states = sample_linear_conditional_path(noise, evidence, time_values)
        target_velocity = target_linear_path_velocity(noise, evidence)
        return {
            "noise": noise,
            "time_values": time_values,
            "path_states": path_states,
            "target_velocity": target_velocity,
        }

    def sample_particles(
        self,
        class_contexts: torch.Tensor,
        *,
        num_particles: int,
        num_steps: int,
        base_noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Euler sample the time-1 pushforward distribution from `N(0, I)`."""

        if class_contexts.dim() != 2:
            raise ValueError(
                f"class_contexts must have shape (Way, ContextDim), got {tuple(class_contexts.shape)}"
            )
        if num_particles <= 0 or num_steps <= 0:
            raise ValueError("num_particles and num_steps must be positive")

        way_num = class_contexts.shape[0]
        if base_noise is None:
            particles = torch.randn(
                way_num,
                num_particles,
                self.latent_dim,
                device=class_contexts.device,
                dtype=class_contexts.dtype,
            )
        else:
            expected_shape = (way_num, num_particles, self.latent_dim)
            if tuple(base_noise.shape) != expected_shape:
                raise ValueError(f"base_noise must have shape {expected_shape}, got {tuple(base_noise.shape)}")
            particles = base_noise

        flat_particles = particles.reshape(way_num * num_particles, self.latent_dim)
        flat_contexts = class_contexts.unsqueeze(1).expand(-1, num_particles, -1).reshape(
            way_num * num_particles,
            class_contexts.shape[-1],
        )
        step_size = 1.0 / float(num_steps)

        for step_idx in range(num_steps):
            time_value = (step_idx + 0.5) * step_size
            time_values = flat_particles.new_full((flat_particles.shape[0],), time_value)
            velocity = self(flat_particles, time_values, flat_contexts)
            flat_particles = flat_particles + step_size * velocity

        return flat_particles.reshape(way_num, num_particles, self.latent_dim)
