"""Evidence-Budgeted Optimal Transport for noisy scalogram few-shot learning."""

from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from net.fewshot_common import BaseConv64FewShotModel, feature_map_to_tokens


class EBOTOutput(dict):
    """Dict-like output that exposes `.shape` through logits."""

    @property
    def shape(self):
        logits = self.get("logits")
        return None if logits is None else logits.shape


def _inverse_softplus(value: float) -> float:
    value = max(float(value), 1e-8)
    return math.log(math.expm1(value))


def _autocast_disabled(device: torch.device):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device.type, enabled=False)
    if device.type == "cuda":
        return torch.cuda.amp.autocast(enabled=False)
    return nullcontext()


class ScalogramPriorExtractor(nn.Module):
    """Build local energy and contrast priors on the backbone token grid."""

    def __init__(
        self,
        use_scalogram_priors: bool = True,
        use_energy_prior: bool = True,
        use_gradient_prior: bool = True,
        use_tf_coords: bool = True,
        log_power_alpha: float = 10.0,
        prior_norm: str = "mean",
        gray_mode: str = "mean",
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        prior_norm = str(prior_norm).lower()
        gray_mode = str(gray_mode).lower()
        if prior_norm not in {"mean", "zscore", "none"}:
            raise ValueError("prior_norm must be one of {'mean', 'zscore', 'none'}")
        if gray_mode not in {"mean", "luminance"}:
            raise ValueError("gray_mode must be 'mean' or 'luminance'")
        if log_power_alpha <= 0.0:
            raise ValueError("log_power_alpha must be positive")
        if eps <= 0.0:
            raise ValueError("eps must be positive")

        self.use_scalogram_priors = bool(use_scalogram_priors)
        self.use_energy_prior = bool(use_energy_prior)
        self.use_gradient_prior = bool(use_gradient_prior)
        self.use_tf_coords = bool(use_tf_coords)
        self.log_power_alpha = float(log_power_alpha)
        self.prior_norm = prior_norm
        self.gray_mode = gray_mode
        self.eps = float(eps)

        sobel_t = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3) / 8.0
        sobel_f = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3) / 8.0
        self.register_buffer("sobel_t", sobel_t, persistent=False)
        self.register_buffer("sobel_f", sobel_f, persistent=False)

    @property
    def prior_dim(self) -> int:
        dim = 0
        if self.use_scalogram_priors:
            if self.use_energy_prior:
                dim += 1
            if self.use_gradient_prior:
                dim += 3
        if self.use_tf_coords:
            dim += 2
        return dim

    def _to_gray(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"x must have shape (B, C, H, W), got {tuple(x.shape)}")
        if x.shape[1] == 1:
            return x[:, :1]
        if x.shape[1] == 3:
            if self.gray_mode == "luminance":
                weights = x.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
                return (x * weights).sum(dim=1, keepdim=True)
            return x.mean(dim=1, keepdim=True)
        raise ValueError(f"EBOT supports 1-channel or 3-channel scalograms, got C={x.shape[1]}")

    def _normalize_signal_priors(self, priors: torch.Tensor) -> torch.Tensor:
        if priors.numel() == 0 or self.prior_norm == "none":
            return priors
        if self.prior_norm == "mean":
            denom = priors.mean(dim=1, keepdim=True).abs().clamp_min(self.eps)
            return priors / denom
        mean = priors.mean(dim=1, keepdim=True)
        std = priors.std(dim=1, keepdim=True, unbiased=False).clamp_min(self.eps)
        return (priors - mean) / std

    @staticmethod
    def _coords(
        out_hw: tuple[int, int],
        *,
        device: torch.device,
        dtype: torch.dtype,
        batch_size: int,
    ) -> torch.Tensor:
        height, width = out_hw
        freq = torch.linspace(-1.0, 1.0, steps=height, device=device, dtype=dtype)
        time = torch.linspace(-1.0, 1.0, steps=width, device=device, dtype=dtype)
        ff, tt = torch.meshgrid(freq, time, indexing="ij")
        coords = torch.stack([tt, ff], dim=-1).reshape(1, height * width, 2)
        return coords.expand(batch_size, -1, -1)

    def forward(self, x: torch.Tensor, out_hw: tuple[int, int]) -> torch.Tensor:
        if len(out_hw) != 2 or out_hw[0] <= 0 or out_hw[1] <= 0:
            raise ValueError(f"out_hw must contain positive (Hf, Wf), got {out_hw}")
        batch_size = x.shape[0]
        out_h, out_w = int(out_hw[0]), int(out_hw[1])
        dtype = x.dtype

        signal_priors: list[torch.Tensor] = []
        if self.use_scalogram_priors and (self.use_energy_prior or self.use_gradient_prior):
            # Scalogram-aware reliability prior.
            gray = self._to_gray(x).float()
            gray = gray - gray.amin(dim=(-2, -1), keepdim=True)
            gray = gray / gray.amax(dim=(-2, -1), keepdim=True).clamp_min(self.eps)
            alpha = float(self.log_power_alpha)
            gray_log = torch.log1p(alpha * gray) / math.log1p(alpha)

            if self.use_energy_prior:
                energy = gray_log.square()
                energy = F.adaptive_avg_pool2d(energy, (out_h, out_w))
                signal_priors.append(energy)

            if self.use_gradient_prior:
                sobel_t = self.sobel_t.to(device=gray_log.device, dtype=gray_log.dtype)
                sobel_f = self.sobel_f.to(device=gray_log.device, dtype=gray_log.dtype)
                grad_t = F.conv2d(gray_log, sobel_t, padding=1).abs()
                grad_f = F.conv2d(gray_log, sobel_f, padding=1).abs()
                grad_mag = torch.sqrt(grad_t.square() + grad_f.square() + self.eps)
                signal_priors.extend(
                    [
                        F.adaptive_avg_pool2d(grad_t, (out_h, out_w)),
                        F.adaptive_avg_pool2d(grad_f, (out_h, out_w)),
                        F.adaptive_avg_pool2d(grad_mag, (out_h, out_w)),
                    ]
                )

        parts: list[torch.Tensor] = []
        if signal_priors:
            signal = torch.cat(signal_priors, dim=1)
            signal = signal.flatten(2).transpose(1, 2).contiguous()
            parts.append(self._normalize_signal_priors(signal).to(dtype=dtype))

        if self.use_tf_coords:
            parts.append(self._coords((out_h, out_w), device=x.device, dtype=dtype, batch_size=batch_size))

        if not parts:
            return x.new_empty(batch_size, out_h * out_w, 0)
        return torch.cat(parts, dim=-1)


class ReliabilityMassEstimator(nn.Module):
    """Predict token reliability, evidence budget, and transport masses."""

    def __init__(
        self,
        token_dim: int,
        prior_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        min_budget: float = 0.15,
        max_budget: float = 0.95,
        gate_temperature: float = 1.0,
        use_cross_reference: bool = True,
        use_uncertainty_prior: bool = False,
        use_dustbin: bool = True,
        use_evidence_budget: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if token_dim <= 0:
            raise ValueError("token_dim must be positive")
        if prior_dim < 0:
            raise ValueError("prior_dim must be non-negative")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if min_budget <= 0.0 or max_budget > 1.0 or min_budget > max_budget:
            raise ValueError("min_budget and max_budget must satisfy 0 < min <= max <= 1")
        if gate_temperature <= 0.0:
            raise ValueError("gate_temperature must be positive")
        if eps <= 0.0:
            raise ValueError("eps must be positive")

        self.token_dim = int(token_dim)
        self.prior_dim = int(prior_dim)
        self.use_cross_reference = bool(use_cross_reference)
        self.use_uncertainty_prior = bool(use_uncertainty_prior)
        self.use_dustbin = bool(use_dustbin)
        self.use_evidence_budget = bool(use_evidence_budget)
        self.min_budget = float(min_budget)
        self.max_budget = float(max_budget)
        self.gate_temperature = float(gate_temperature)
        self.eps = float(eps)

        input_dim = self.token_dim + self.prior_dim
        if self.use_cross_reference:
            input_dim += 1
        if self.use_uncertainty_prior:
            input_dim += 1

        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        token_features: torch.Tensor,
        priors: torch.Tensor | None,
        cross_ref: torch.Tensor | None = None,
        uncertainty: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if token_features.dim() != 3:
            raise ValueError(
                f"token_features must have shape (B, N, D), got {tuple(token_features.shape)}"
            )
        batch_size, token_count, _ = token_features.shape

        parts = [token_features]
        if self.prior_dim > 0:
            if priors is None:
                raise ValueError("priors are required when prior_dim > 0")
            if priors.shape[:2] != (batch_size, token_count) or priors.shape[-1] != self.prior_dim:
                raise ValueError(
                    f"priors must have shape {(batch_size, token_count, self.prior_dim)}, "
                    f"got {tuple(priors.shape)}"
                )
            parts.append(priors.to(device=token_features.device, dtype=token_features.dtype))

        if self.use_cross_reference:
            if cross_ref is None:
                cross_ref = token_features.new_zeros(batch_size, token_count, 1)
            parts.append(cross_ref.to(device=token_features.device, dtype=token_features.dtype))

        if self.use_uncertainty_prior:
            if uncertainty is None:
                uncertainty = token_features.new_zeros(batch_size, token_count, 1)
            parts.append(uncertainty.to(device=token_features.device, dtype=token_features.dtype))

        reliability_logits = self.net(torch.cat(parts, dim=-1)).squeeze(-1)
        gate = torch.sigmoid(reliability_logits / self.gate_temperature)

        # Evidence budget.
        if self.use_evidence_budget and self.use_dustbin:
            budget = self.min_budget + (self.max_budget - self.min_budget) * gate.mean(dim=-1)
        else:
            budget = torch.ones(batch_size, device=token_features.device, dtype=token_features.dtype)
        budget = budget.clamp(self.eps, 1.0)

        mass_real = budget.unsqueeze(-1) * torch.softmax(reliability_logits, dim=-1)
        if self.use_evidence_budget and self.use_dustbin:
            dust = (1.0 - budget).clamp_min(self.eps).unsqueeze(-1)
            mass_aug = torch.cat([mass_real, dust], dim=-1)
            mass_aug = mass_aug / mass_aug.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        else:
            mass_aug = mass_real / mass_real.sum(dim=-1, keepdim=True).clamp_min(self.eps)
            mass_real = mass_aug

        return {
            "logits": reliability_logits,
            "gate": gate,
            "budget": budget,
            "mass_real": mass_real,
            "mass_aug": mass_aug,
        }


def log_sinkhorn(
    cost: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    epsilon: float = 0.05,
    n_iters: int = 50,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Differentiable batched balanced Sinkhorn in log-space."""

    if cost.dim() != 3:
        raise ValueError(f"cost must have shape (B, N, M), got {tuple(cost.shape)}")
    if a.shape != cost.shape[:2]:
        raise ValueError(f"a must have shape {tuple(cost.shape[:2])}, got {tuple(a.shape)}")
    if b.shape != (cost.shape[0], cost.shape[2]):
        raise ValueError(f"b must have shape {(cost.shape[0], cost.shape[2])}, got {tuple(b.shape)}")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    if n_iters <= 0:
        raise ValueError("n_iters must be positive")

    cost = torch.nan_to_num(cost, nan=0.0, posinf=1e4, neginf=-1e4)
    a = a.clamp_min(eps)
    b = b.clamp_min(eps)
    a = a / a.sum(dim=-1, keepdim=True).clamp_min(eps)
    b = b / b.sum(dim=-1, keepdim=True).clamp_min(eps)

    log_a = a.log()
    log_b = b.log()
    log_k = -cost / float(epsilon)
    u = torch.zeros_like(log_a)
    v = torch.zeros_like(log_b)

    for _ in range(int(n_iters)):
        u = log_a - torch.logsumexp(log_k + v.unsqueeze(-2), dim=-1)
        v = log_b - torch.logsumexp(log_k + u.unsqueeze(-1), dim=-2)

    log_t = log_k + u.unsqueeze(-1) + v.unsqueeze(-2)
    transport = torch.exp(log_t)
    return torch.nan_to_num(transport, nan=0.0, posinf=0.0, neginf=0.0)


class EvidenceBudgetedOTMatcher(nn.Module):
    """Reliability-aware partial/dustbin OT matcher for local evidence tokens."""

    def __init__(
        self,
        input_dim: int,
        proj_dim: int = 256,
        prior_dim: int = 6,
        reliability_hidden_dim: int = 128,
        reliability_dropout: float = 0.1,
        sinkhorn_epsilon: float = 0.05,
        sinkhorn_iters: int = 50,
        dustbin_cost: float = 0.7,
        learnable_dustbin_cost: bool = True,
        alpha_unmatched: float = 0.10,
        min_budget: float = 0.15,
        max_budget: float = 0.95,
        gate_temperature: float = 1.0,
        use_cross_reference: bool = True,
        use_dustbin: bool = True,
        use_evidence_budget: bool = True,
        symmetric_matching: bool = False,
        use_uncertainty_prior: bool = False,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or proj_dim <= 0:
            raise ValueError("input_dim and proj_dim must be positive")
        if sinkhorn_epsilon <= 0.0:
            raise ValueError("sinkhorn_epsilon must be positive")
        if sinkhorn_iters <= 0:
            raise ValueError("sinkhorn_iters must be positive")
        if dustbin_cost < 0.0 or alpha_unmatched < 0.0:
            raise ValueError("dustbin_cost and alpha_unmatched must be non-negative")

        self.input_dim = int(input_dim)
        self.proj_dim = int(proj_dim)
        self.prior_dim = int(prior_dim)
        self.sinkhorn_epsilon = float(sinkhorn_epsilon)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.alpha_unmatched = float(alpha_unmatched)
        self.use_dustbin = bool(use_dustbin)
        self.use_evidence_budget = bool(use_evidence_budget)
        self.symmetric_matching = bool(symmetric_matching)
        self.eps = float(eps)

        self.projector = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, self.proj_dim, bias=False),
        )
        self.reliability = ReliabilityMassEstimator(
            token_dim=self.proj_dim,
            prior_dim=self.prior_dim,
            hidden_dim=reliability_hidden_dim,
            dropout=reliability_dropout,
            min_budget=min_budget,
            max_budget=max_budget,
            gate_temperature=gate_temperature,
            use_cross_reference=use_cross_reference,
            use_uncertainty_prior=use_uncertainty_prior,
            use_dustbin=self.use_dustbin,
            use_evidence_budget=self.use_evidence_budget,
            eps=eps,
        )

        self.learnable_dustbin_cost = bool(learnable_dustbin_cost)
        if self.learnable_dustbin_cost:
            self.raw_dustbin_cost = nn.Parameter(torch.tensor(_inverse_softplus(dustbin_cost)))
        else:
            self.register_buffer(
                "fixed_dustbin_cost",
                torch.tensor(float(dustbin_cost), dtype=torch.float32),
                persistent=False,
            )

    @property
    def dustbin_cost(self) -> torch.Tensor:
        if self.learnable_dustbin_cost:
            return F.softplus(self.raw_dustbin_cost)
        return self.fixed_dustbin_cost

    def _build_augmented_cost(self, cost: torch.Tensor) -> torch.Tensor:
        batch_size, nq, ns = cost.shape
        dust_cost = self.dustbin_cost.to(device=cost.device, dtype=cost.dtype)
        aug = cost.new_full((batch_size, nq + 1, ns + 1), float(dust_cost.detach().item()))
        aug[:, :nq, :ns] = cost
        aug[:, nq, ns] = 0.0
        if self.learnable_dustbin_cost:
            aug[:, :nq, ns] = dust_cost
            aug[:, nq, :ns] = dust_cost
        return aug

    def _solve_transport(
        self,
        cost: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        with _autocast_disabled(cost.device):
            return log_sinkhorn(
                cost.float(),
                a.float(),
                b.float(),
                epsilon=self.sinkhorn_epsilon,
                n_iters=self.sinkhorn_iters,
                eps=self.eps,
            ).to(dtype=cost.dtype)

    def _match_one_direction(
        self,
        q_tokens: torch.Tensor,
        s_tokens: torch.Tensor,
        q_priors: torch.Tensor,
        s_priors: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if q_tokens.dim() != 3 or s_tokens.dim() != 3:
            raise ValueError("q_tokens and s_tokens must both have shape (B, N, D)")
        if q_tokens.shape[0] != s_tokens.shape[0]:
            raise ValueError("q_tokens and s_tokens must have the same batch size")
        if q_tokens.shape[-1] != self.input_dim or s_tokens.shape[-1] != self.input_dim:
            raise ValueError(
                f"token dim mismatch: expected {self.input_dim}, got "
                f"{q_tokens.shape[-1]} and {s_tokens.shape[-1]}"
            )

        q = F.normalize(self.projector(q_tokens), dim=-1, eps=self.eps)
        s = F.normalize(self.projector(s_tokens), dim=-1, eps=self.eps)
        sim = torch.bmm(q, s.transpose(1, 2))
        cost = 1.0 - sim

        cross_q = sim.max(dim=-1).values.unsqueeze(-1)
        cross_s = sim.max(dim=-2).values.unsqueeze(-1)
        uncert_q = torch.zeros_like(cross_q)
        uncert_s = torch.zeros_like(cross_s)

        q_mass = self.reliability(q, q_priors, cross_q, uncert_q)
        s_mass = self.reliability(s, s_priors, cross_s, uncert_s)

        # Dustbin-augmented partial transport.
        use_dustbin_transport = self.use_dustbin and self.use_evidence_budget
        if use_dustbin_transport:
            transport_cost = self._build_augmented_cost(cost)
            a = q_mass["mass_aug"]
            b = s_mass["mass_aug"]
            plan = self._solve_transport(transport_cost, a, b)
            nq, ns = cost.shape[-2], cost.shape[-1]
            real_plan = plan[:, :nq, :ns]
            unmatched_q = plan[:, :nq, ns].sum(dim=-1)
            unmatched_s = plan[:, nq, :ns].sum(dim=-1)
            unmatched_mass = unmatched_q + unmatched_s
        else:
            transport_cost = cost
            a = q_mass["mass_real"]
            b = s_mass["mass_real"]
            plan = self._solve_transport(transport_cost, a, b)
            real_plan = plan
            unmatched_mass = cost.new_zeros(cost.shape[0])

        matched_mass = real_plan.sum(dim=(-1, -2))
        real_cost = (real_plan * cost).sum(dim=(-1, -2)) / matched_mass.clamp_min(self.eps)
        pair_score = -real_cost - float(self.alpha_unmatched) * unmatched_mass
        entropy = -(plan * plan.clamp_min(self.eps).log()).sum(dim=(-1, -2))
        err_a = (plan.sum(dim=-1) - a).abs().mean(dim=-1)
        err_b = (plan.sum(dim=-2) - b).abs().mean(dim=-1)
        sinkhorn_error = 0.5 * (err_a + err_b)

        return {
            "score": pair_score,
            "real_cost": real_cost,
            "matched_mass": matched_mass,
            "unmatched_mass": unmatched_mass,
            "q_budget": q_mass["budget"],
            "s_budget": s_mass["budget"],
            "q_gate": q_mass["gate"],
            "s_gate": s_mass["gate"],
            "q_mass_real": q_mass["mass_real"],
            "s_mass_real": s_mass["mass_real"],
            "transport_entropy": entropy,
            "sinkhorn_error": sinkhorn_error,
            "transport_plan": plan.detach(),
            "cost_matrix": transport_cost.detach(),
        }

    def forward(
        self,
        q_tokens: torch.Tensor,
        s_tokens: torch.Tensor,
        q_priors: torch.Tensor,
        s_priors: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        primary = self._match_one_direction(q_tokens, s_tokens, q_priors, s_priors)
        if not self.symmetric_matching:
            score = primary["score"]
            return score, {key: value for key, value in primary.items() if key != "score"}

        swapped = self._match_one_direction(s_tokens, q_tokens, s_priors, q_priors)
        score = 0.5 * (primary["score"] + swapped["score"])
        aux: dict[str, torch.Tensor] = {}
        for key, value in primary.items():
            if key == "score":
                continue
            other = swapped.get(key)
            if torch.is_tensor(value) and torch.is_tensor(other) and value.shape == other.shape:
                aux[key] = 0.5 * (value + other)
            else:
                aux[key] = value
        return score, aux


class EvidenceBudgetedOTFewShot(BaseConv64FewShotModel):
    """Few-shot classifier using reliability-budgeted dustbin OT evidence."""

    def __init__(
        self,
        in_channels: int = 3,
        hidden_dim: int = 64,
        proj_dim: int = 256,
        reliability_hidden_dim: int = 128,
        reliability_dropout: float = 0.1,
        backbone_name: str = "resnet12",
        image_size: int = 84,
        resnet12_drop_rate: float = 0.0,
        resnet12_dropblock_size: int = 5,
        sinkhorn_epsilon: float = 0.05,
        sinkhorn_iters: int = 50,
        dustbin_cost: float = 0.7,
        learnable_dustbin_cost: bool = True,
        alpha_unmatched: float = 0.10,
        min_budget: float = 0.15,
        max_budget: float = 0.95,
        gate_temperature: float = 1.0,
        use_scalogram_priors: bool = True,
        use_energy_prior: bool = True,
        use_gradient_prior: bool = True,
        use_tf_coords: bool = True,
        log_power_alpha: float = 10.0,
        prior_norm: str = "mean",
        use_cross_reference: bool = True,
        use_dustbin: bool = True,
        use_evidence_budget: bool = True,
        use_kshot_reweighting: bool = True,
        lambda_score: float = 1.0,
        lambda_mass: float = 0.5,
        lambda_unmatched: float = 0.5,
        score_scale: float = 12.5,
        learnable_score_scale: bool = False,
        symmetric_matching: bool = False,
        use_uncertainty_prior: bool = False,
        use_aux_loss: bool = False,
        budget_floor: float = 0.15,
        budget_ceiling: float = 0.95,
        weight_budget_low: float = 0.01,
        weight_budget_high: float = 0.001,
        eps: float = 1e-6,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            backbone_name=backbone_name,
            image_size=image_size,
            resnet12_drop_rate=resnet12_drop_rate,
            resnet12_dropblock_size=resnet12_dropblock_size,
        )
        if proj_dim <= 0:
            raise ValueError("proj_dim must be positive")
        if score_scale <= 0.0:
            raise ValueError("score_scale must be positive")
        if eps <= 0.0:
            raise ValueError("eps must be positive")

        self.proj_dim = int(proj_dim)
        self.use_kshot_reweighting = bool(use_kshot_reweighting)
        self.lambda_score = float(lambda_score)
        self.lambda_mass = float(lambda_mass)
        self.lambda_unmatched = float(lambda_unmatched)
        self.learnable_score_scale = bool(learnable_score_scale)
        self.use_aux_loss = bool(use_aux_loss)
        self.budget_floor = float(budget_floor)
        self.budget_ceiling = float(budget_ceiling)
        self.weight_budget_low = float(weight_budget_low)
        self.weight_budget_high = float(weight_budget_high)
        self.eps = float(eps)
        self.latest_aux: dict[str, Any] = {}
        self._last_reliability_maps: dict[str, torch.Tensor] = {}
        if self.learnable_score_scale:
            self.raw_score_scale = nn.Parameter(torch.tensor(_inverse_softplus(score_scale)))
        else:
            self.register_buffer(
                "fixed_score_scale",
                torch.tensor(float(score_scale), dtype=torch.float32),
                persistent=False,
            )

        self.prior_extractor = ScalogramPriorExtractor(
            use_scalogram_priors=use_scalogram_priors,
            use_energy_prior=use_energy_prior,
            use_gradient_prior=use_gradient_prior,
            use_tf_coords=use_tf_coords,
            log_power_alpha=log_power_alpha,
            prior_norm=prior_norm,
            eps=eps,
        )
        self.matcher = EvidenceBudgetedOTMatcher(
            input_dim=hidden_dim,
            proj_dim=proj_dim,
            prior_dim=self.prior_extractor.prior_dim,
            reliability_hidden_dim=reliability_hidden_dim,
            reliability_dropout=reliability_dropout,
            sinkhorn_epsilon=sinkhorn_epsilon,
            sinkhorn_iters=sinkhorn_iters,
            dustbin_cost=dustbin_cost,
            learnable_dustbin_cost=learnable_dustbin_cost,
            alpha_unmatched=alpha_unmatched,
            min_budget=min_budget,
            max_budget=max_budget,
            gate_temperature=gate_temperature,
            use_cross_reference=use_cross_reference,
            use_dustbin=use_dustbin,
            use_evidence_budget=use_evidence_budget,
            symmetric_matching=symmetric_matching,
            use_uncertainty_prior=use_uncertainty_prior,
            eps=eps,
        )

    @property
    def score_scale(self) -> torch.Tensor:
        if self.learnable_score_scale:
            return F.softplus(self.raw_score_scale)
        return self.fixed_score_scale

    def _prepare_backbone_images(self, images: torch.Tensor) -> torch.Tensor:
        if images.dim() != 4:
            raise ValueError(f"images must have shape (N, C, H, W), got {tuple(images.shape)}")
        if images.shape[1] == 3:
            return images
        if images.shape[1] == 1:
            return images.repeat(1, 3, 1, 1)
        raise ValueError(f"EBOT supports 1-channel or 3-channel images, got C={images.shape[1]}")

    def _encode_images(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
        feature_map = self.encode(self._prepare_backbone_images(images))
        if feature_map.dim() != 4:
            raise ValueError(f"Backbone must return feature maps, got {tuple(feature_map.shape)}")
        height, width = int(feature_map.shape[-2]), int(feature_map.shape[-1])
        tokens = feature_map_to_tokens(feature_map)
        priors = self.prior_extractor(images, (height, width)).to(device=tokens.device, dtype=tokens.dtype)
        return tokens, priors, (height, width)

    def _aggregate_shots(
        self,
        shot_scores: torch.Tensor,
        matched_mass: torch.Tensor,
        unmatched_mass: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shot_num = int(shot_scores.shape[-1])
        if shot_num == 1:
            weights = torch.ones_like(shot_scores)
            return shot_scores.squeeze(-1), weights
        if not self.use_kshot_reweighting:
            weights = torch.full_like(shot_scores, 1.0 / float(shot_num))
            return (weights * shot_scores).sum(dim=-1), weights

        # Reliability-weighted k-shot aggregation.
        beta_logits = (
            self.lambda_score * shot_scores
            + self.lambda_mass * matched_mass.clamp_min(self.eps).log()
            - self.lambda_unmatched * unmatched_mass
        )
        weights = torch.softmax(beta_logits, dim=-1)
        return (weights * shot_scores).sum(dim=-1), weights

    def _budget_aux_loss(self, q_budget: torch.Tensor, s_budget: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        low = (
            F.relu(self.budget_floor - q_budget).square()
            + F.relu(self.budget_floor - s_budget).square()
        ).mean()
        high = (
            F.relu(q_budget - self.budget_ceiling).square()
            + F.relu(s_budget - self.budget_ceiling).square()
        ).mean()
        aux = q_budget.new_zeros(())
        if self.use_aux_loss:
            aux = aux + self.weight_budget_low * low + self.weight_budget_high * high
        return aux, low.detach(), high.detach()

    def _forward_episode(
        self,
        query: torch.Tensor,
        support: torch.Tensor,
        *,
        needs_payload: bool = False,
        return_aux: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        del return_aux
        if query.dim() != 4:
            raise ValueError(f"query must have shape (NumQuery, C, H, W), got {tuple(query.shape)}")
        if support.dim() != 5:
            raise ValueError(f"support must have shape (Way, Shot, C, H, W), got {tuple(support.shape)}")

        way_num, shot_num = int(support.shape[0]), int(support.shape[1])
        num_query = int(query.shape[0])
        query_tokens, query_priors, (height, width) = self._encode_images(query)
        support_tokens, support_priors, support_hw = self._encode_images(
            support.reshape(way_num * shot_num, *support.shape[-3:])
        )
        if support_hw != (height, width):
            raise ValueError(f"Query/support token grids must match, got {(height, width)} vs {support_hw}")

        token_count = query_tokens.shape[1]
        support_tokens = support_tokens.reshape(way_num, shot_num, token_count, self.hidden_dim)
        support_priors = support_priors.reshape(way_num, shot_num, token_count, self.prior_extractor.prior_dim)

        pair_count = num_query * way_num * shot_num
        q_pair = (
            query_tokens.unsqueeze(1)
            .unsqueeze(2)
            .expand(-1, way_num, shot_num, -1, -1)
            .reshape(pair_count, token_count, self.hidden_dim)
        )
        s_pair = (
            support_tokens.unsqueeze(0)
            .expand(num_query, -1, -1, -1, -1)
            .reshape(pair_count, token_count, self.hidden_dim)
        )
        q_prior_pair = (
            query_priors.unsqueeze(1)
            .unsqueeze(2)
            .expand(-1, way_num, shot_num, -1, -1)
            .reshape(pair_count, token_count, self.prior_extractor.prior_dim)
        )
        s_prior_pair = (
            support_priors.unsqueeze(0)
            .expand(num_query, -1, -1, -1, -1)
            .reshape(pair_count, token_count, self.prior_extractor.prior_dim)
        )

        pair_scores, pair_aux = self.matcher(q_pair, s_pair, q_prior_pair, s_prior_pair)
        shot_scores = pair_scores.reshape(num_query, way_num, shot_num)
        matched_mass = pair_aux["matched_mass"].reshape(num_query, way_num, shot_num)
        unmatched_mass = pair_aux["unmatched_mass"].reshape(num_query, way_num, shot_num)
        class_scores, shot_weights = self._aggregate_shots(shot_scores, matched_mass, unmatched_mass)
        score_scale = self.score_scale.to(device=class_scores.device, dtype=class_scores.dtype)
        logits = score_scale * class_scores

        q_budget = pair_aux["q_budget"].reshape(num_query, way_num, shot_num)
        s_budget = pair_aux["s_budget"].reshape(num_query, way_num, shot_num)
        aux_loss, budget_low, budget_high = self._budget_aux_loss(q_budget, s_budget)

        q_gate = pair_aux["q_gate"].reshape(num_query, way_num, shot_num, token_count)
        s_gate = pair_aux["s_gate"].reshape(num_query, way_num, shot_num, token_count)
        query_reliability_maps = q_gate.mean(dim=(1, 2)).reshape(num_query, height, width)
        support_reliability_maps = s_gate.mean(dim=0).reshape(way_num, shot_num, height, width)
        weighted_matched = (shot_weights * matched_mass).sum(dim=-1)
        weighted_unmatched = (shot_weights * unmatched_mass).sum(dim=-1)

        self._last_reliability_maps = {
            "query": query_reliability_maps.detach(),
            "support": support_reliability_maps.detach(),
        }
        self.latest_aux = {
            "matched_mass_mean": float(matched_mass.detach().mean().item()),
            "unmatched_mass_mean": float(unmatched_mass.detach().mean().item()),
            "budget_query_mean": float(q_budget.detach().mean().item()),
            "budget_support_mean": float(s_budget.detach().mean().item()),
            "gate_query_mean": float(q_gate.detach().mean().item()),
            "gate_support_mean": float(s_gate.detach().mean().item()),
            "dustbin_cost": float(self.matcher.dustbin_cost.detach().item()),
            "score_scale": float(score_scale.detach().item()),
            "sinkhorn_error": float(pair_aux["sinkhorn_error"].detach().mean().item()),
            "feature_hw": (height, width),
        }

        if not needs_payload:
            return logits

        transport_entropy = pair_aux["transport_entropy"].reshape(num_query, way_num, shot_num)
        real_cost = pair_aux["real_cost"].reshape(num_query, way_num, shot_num)
        total_distance = (shot_weights * real_cost).sum(dim=-1)
        outputs: dict[str, torch.Tensor] = {
            "logits": logits,
            "aux_loss": aux_loss,
            "class_scores": logits,
            "raw_logits": class_scores,
            "raw_class_scores": class_scores,
            "shot_scores": shot_scores,
            "shot_aggregation_weights": shot_weights,
            "matched_mass": matched_mass,
            "unmatched_mass": unmatched_mass,
            "transported_mass": weighted_matched,
            "weighted_unmatched_mass": weighted_unmatched,
            "total_distance": total_distance,
            "query_class_distance": total_distance,
            "transport_cost": real_cost,
            "plan_entropy": transport_entropy,
            "q_budget": q_budget,
            "s_budget": s_budget,
            "q_gate_mean": q_gate.mean(dim=-1),
            "s_gate_mean": s_gate.mean(dim=-1),
            "query_reliability_maps": query_reliability_maps,
            "support_reliability_maps": support_reliability_maps,
            "budget_low_loss": budget_low,
            "budget_high_loss": budget_high,
            "sinkhorn_error": pair_aux["sinkhorn_error"].reshape(num_query, way_num, shot_num),
            "mean_shot_distance": real_cost.mean(),
            "mean_budget": 0.5 * (q_budget.mean() + s_budget.mean()),
            "mean_gate": 0.5 * (q_gate.mean() + s_gate.mean()),
            "mean_query_mass_entropy": (
                -(pair_aux["q_mass_real"].clamp_min(self.eps) * pair_aux["q_mass_real"].clamp_min(self.eps).log())
                .sum(dim=-1)
                .mean()
            ),
            "mean_support_mass_entropy": (
                -(pair_aux["s_mass_real"].clamp_min(self.eps) * pair_aux["s_mass_real"].clamp_min(self.eps).log())
                .sum(dim=-1)
                .mean()
            ),
            "dustbin_cost": self.matcher.dustbin_cost.detach(),
            "score_scale": score_scale.detach(),
        }
        return outputs

    @staticmethod
    def _reshape_query_targets(
        query_targets: torch.Tensor | None,
        *,
        batch_size: int,
        num_query: int,
    ) -> torch.Tensor | None:
        if query_targets is None:
            return None
        if query_targets.dim() == 1:
            if query_targets.numel() != batch_size * num_query:
                raise ValueError(
                    f"query_targets must have {batch_size * num_query} elements, got {query_targets.numel()}"
                )
            return query_targets.view(batch_size, num_query)
        if query_targets.dim() == 2 and tuple(query_targets.shape) == (batch_size, num_query):
            return query_targets
        raise ValueError(
            "query_targets must have shape (Batch * NumQuery,) or (Batch, NumQuery), "
            f"got {tuple(query_targets.shape)}"
        )

    @staticmethod
    def _stack_outputs(batch_outputs: list[dict[str, torch.Tensor]], logits: torch.Tensor) -> EBOTOutput:
        stacked: dict[str, Any] = {
            "logits": logits,
            "aux_loss": torch.stack([item["aux_loss"] for item in batch_outputs]).mean(),
        }
        query_first_keys = {
            "class_scores",
            "logits",
            "matched_mass",
            "plan_entropy",
            "q_budget",
            "q_gate_mean",
            "query_reliability_maps",
            "s_budget",
            "s_gate_mean",
            "shot_aggregation_weights",
            "shot_scores",
            "sinkhorn_error",
            "transport_cost",
            "transported_mass",
            "unmatched_mass",
            "weighted_unmatched_mass",
        }
        tensor_keys = set().union(*(item.keys() for item in batch_outputs))
        tensor_keys.discard("logits")
        tensor_keys.discard("aux_loss")
        for key in sorted(tensor_keys):
            values = [item[key] for item in batch_outputs if key in item]
            if not values or not torch.is_tensor(values[0]):
                continue
            if values[0].dim() == 0:
                stacked[key] = torch.stack(values).mean()
            elif key in query_first_keys:
                stacked[key] = torch.cat(values, dim=0)
            else:
                stacked[key] = torch.stack(values, dim=0)
        return EBOTOutput(stacked)

    def forward(
        self,
        query: torch.Tensor,
        support: torch.Tensor,
        *,
        query_targets: torch.Tensor | None = None,
        support_targets: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        del support_targets
        needs_payload = bool(return_aux or self.training)

        if support.dim() == 6:
            batch_size = support.shape[0]
            way_num = support.shape[1]
            if query.dim() == 5:
                if query.shape[0] != batch_size:
                    raise ValueError(
                        f"query/support batch mismatch: query={tuple(query.shape)} support={tuple(support.shape)}"
                    )
                num_query = query.shape[1]

                def get_query(batch_idx: int) -> torch.Tensor:
                    return query[batch_idx]

            elif query.dim() == 4:
                if query.shape[0] % batch_size != 0:
                    raise ValueError(
                        f"flattened query count {query.shape[0]} must be divisible by batch size {batch_size}"
                    )
                num_query = query.shape[0] // batch_size

                def get_query(batch_idx: int) -> torch.Tensor:
                    start = batch_idx * num_query
                    return query[start : start + num_query]

            else:
                raise ValueError(f"query must be 4D or 5D for batched support, got {tuple(query.shape)}")

            self._reshape_query_targets(query_targets, batch_size=batch_size, num_query=num_query)
            batch_outputs: list[dict[str, torch.Tensor]] = []
            batch_logits: list[torch.Tensor] = []
            for batch_idx in range(batch_size):
                out = self._forward_episode(
                    query=get_query(batch_idx),
                    support=support[batch_idx],
                    needs_payload=needs_payload,
                    return_aux=return_aux,
                )
                if needs_payload:
                    batch_outputs.append(out)
                    batch_logits.append(out["logits"])
                else:
                    batch_logits.append(out)

            logits = torch.cat(batch_logits, dim=0).reshape(-1, way_num)
            if not needs_payload:
                return logits
            merged = self._stack_outputs(batch_outputs, logits)
            if return_aux:
                return merged
            return EBOTOutput({"logits": logits, "aux_loss": merged["aux_loss"]})

        if support.dim() != 5 or query.dim() != 4:
            raise ValueError(
                "EBOT expects either query/support as (B,NQ,C,H,W)/(B,Way,Shot,C,H,W) "
                "or (NQ,C,H,W)/(Way,Shot,C,H,W); "
                f"got query={tuple(query.shape)} support={tuple(support.shape)}"
            )

        out = self._forward_episode(query=query, support=support, needs_payload=needs_payload, return_aux=return_aux)
        if not needs_payload:
            return out
        logits = out["logits"].reshape(-1, support.shape[0])
        if return_aux:
            out["logits"] = logits
            return EBOTOutput(out)
        return EBOTOutput({"logits": logits, "aux_loss": out["aux_loss"]})

    def get_last_reliability_maps(self) -> dict[str, torch.Tensor]:
        """Return the latest query/support reliability maps for visualization."""

        return dict(self._last_reliability_maps)


EvidenceBudgetedOT = EvidenceBudgetedOTFewShot
EBOT = EvidenceBudgetedOTFewShot
