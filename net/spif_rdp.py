"""SPIF-RDP: Reliability-calibrated distributional prototype few-shot model.

This model keeps the SPIF encoder and upgrades the global score to a
distributional distance while adding a complementary local OT branch.
"""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from net.fewshot_common import BaseConv64FewShotModel, feature_map_to_tokens
from net.spif import SPIFEncoder


def _inverse_softplus(value: float, floor: float = 1e-6) -> float:
    value = max(float(value), float(floor))
    return math.log(math.expm1(value))


class SPIFRDPOutput(dict):
    """Dict-like output that still exposes `.shape` via logits."""

    @property
    def shape(self):
        logits = self.get("logits")
        return None if logits is None else logits.shape


@torch.jit.script
def _log_sinkhorn_transport(
    log_K: torch.Tensor,
    log_mu: torch.Tensor,
    log_nu: torch.Tensor,
    n_iters: int,
) -> torch.Tensor:
    u = torch.zeros_like(log_mu)
    v = torch.zeros_like(log_nu)
    for _ in range(n_iters):
        v = log_nu - torch.logsumexp(log_K + u.unsqueeze(-1), dim=-2)
        u = log_mu - torch.logsumexp(log_K + v.unsqueeze(-2), dim=-1)
    return log_K + u.unsqueeze(-1) + v.unsqueeze(-2)


def diagnose_local_contribution(
    global_scores: torch.Tensor,
    local_scores: torch.Tensor,
    logits: torch.Tensor,
) -> dict:
    """Verify whether local scores carry non-redundant information."""

    del logits
    g = global_scores.detach().flatten().float()
    l = local_scores.detach().flatten().float()
    g_z = (g - g.mean()) / (g.std() + 1e-8)
    l_z = (l - l.mean()) / (l.std() + 1e-8)
    corr = (g_z * l_z).mean().item()

    return {
        "corr_global_local": corr,
        "local_score_std": l.std().item(),
        "global_score_std": g.std().item(),
        "local_score_range": (l.min().item(), l.max().item()),
        "is_local_redundant": corr > 0.95,
        "is_local_collapsed": l.std().item() < 0.01,
    }


class SinkhornOTLocalBranch(nn.Module):
    """Local residual branch using Sinkhorn OT over stable token sets."""

    def __init__(
        self,
        stable_dim: int,
        n_sinkhorn_iters: int = 5,
        eps_init: float = 0.1,
        tau_local_init: float = 1.0,
        variance_floor: float = 1e-2,
        chunk_size: int = 16,
        chunk_threshold: int = 10_000_000,
        use_vhat_coupling: bool = True,
    ) -> None:
        super().__init__()
        if int(stable_dim) <= 0:
            raise ValueError("stable_dim must be positive")
        if int(n_sinkhorn_iters) <= 0:
            raise ValueError("n_sinkhorn_iters must be positive")
        if float(eps_init) <= 0.0:
            raise ValueError("eps_init must be positive")
        if float(tau_local_init) <= 0.01:
            raise ValueError("tau_local_init must be greater than 0.01")
        if float(variance_floor) <= 0.0:
            raise ValueError("variance_floor must be positive")
        if int(chunk_size) <= 0:
            raise ValueError("chunk_size must be positive")
        if int(chunk_threshold) <= 0:
            raise ValueError("chunk_threshold must be positive")

        self.stable_dim = int(stable_dim)
        self.n_sinkhorn_iters = int(n_sinkhorn_iters)
        self.variance_floor = float(variance_floor)
        self.chunk_size = int(chunk_size)
        self.chunk_threshold = int(chunk_threshold)
        self.use_vhat_coupling = bool(use_vhat_coupling)

        self.eps_raw = nn.Parameter(torch.tensor(_inverse_softplus(eps_init), dtype=torch.float32))
        self.tau_local_raw = nn.Parameter(
            torch.tensor(_inverse_softplus(max(float(tau_local_init) - 0.01, 1e-6)), dtype=torch.float32)
        )

    def eps_value(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return (F.softplus(self.eps_raw) + 1e-4).to(device=device, dtype=dtype)

    def tau_local_value(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return (F.softplus(self.tau_local_raw) + 0.01).to(device=device, dtype=dtype)

    def _dimension_weights(self, variance: torch.Tensor) -> torch.Tensor:
        if self.use_vhat_coupling:
            w_dim = 1.0 / (variance + self.variance_floor)
        else:
            w_dim = torch.ones_like(variance)
        w_dim = w_dim / w_dim.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return w_dim * float(variance.shape[-1])

    def _score_chunk(
        self,
        query_chunk: torch.Tensor,
        support_weighted: torch.Tensor,
        eps: torch.Tensor,
        tau_local: torch.Tensor,
    ) -> torch.Tensor:
        way_num, km_tokens, _ = support_weighted.shape
        num_query, num_tokens, _ = query_chunk.shape

        sim = torch.einsum("qmd,wkd->qwmk", query_chunk, support_weighted)
        cost = 1.0 - sim

        log_mu = cost.new_full((num_query, way_num, num_tokens), -math.log(float(num_tokens)))
        log_nu = cost.new_full((num_query, way_num, km_tokens), -math.log(float(km_tokens)))
        log_K = -cost / eps
        log_gamma = _log_sinkhorn_transport(log_K, log_mu, log_nu, self.n_sinkhorn_iters)
        gamma = log_gamma.exp()
        ot_cost = (gamma * cost).sum(dim=(-1, -2))
        return -ot_cost / tau_local

    def forward(
        self,
        query_tokens: torch.Tensor,
        support_tokens: torch.Tensor,
        variance: torch.Tensor,
    ) -> torch.Tensor:
        if query_tokens.dim() != 3:
            raise ValueError(f"query_tokens must have shape [NumQuery, M, D], got {tuple(query_tokens.shape)}")
        if support_tokens.dim() != 4:
            raise ValueError(f"support_tokens must have shape [Way, Shot, M, D], got {tuple(support_tokens.shape)}")
        if variance.dim() != 2:
            raise ValueError(f"variance must have shape [Way, D], got {tuple(variance.shape)}")
        if query_tokens.shape[-1] != self.stable_dim:
            raise ValueError(f"Expected stable_dim={self.stable_dim}, got query_tokens={query_tokens.shape[-1]}")
        if support_tokens.shape[-1] != self.stable_dim or variance.shape[-1] != self.stable_dim:
            raise ValueError(
                f"Expected stable_dim={self.stable_dim}, got support_tokens={support_tokens.shape[-1]} "
                f"variance={variance.shape[-1]}"
            )
        if support_tokens.shape[0] != variance.shape[0]:
            raise ValueError("support_tokens and variance must agree on way dimension")

        way_num, shot_num, num_tokens, feature_dim = support_tokens.shape
        num_query = query_tokens.shape[0]
        km_tokens = shot_num * num_tokens

        w_dim = self._dimension_weights(variance)
        support_flat = support_tokens.reshape(way_num, km_tokens, feature_dim)
        support_weighted = support_flat * w_dim.unsqueeze(1)

        eps = self.eps_value(query_tokens.device, query_tokens.dtype)
        tau_local = self.tau_local_value(query_tokens.device, query_tokens.dtype)

        total_floats = num_query * way_num * num_tokens * km_tokens
        chunk_size = self.chunk_size if total_floats > self.chunk_threshold else max(1, num_query)

        local_scores = []
        for start_idx in range(0, num_query, chunk_size):
            end_idx = min(start_idx + chunk_size, num_query)
            local_scores.append(self._score_chunk(query_tokens[start_idx:end_idx], support_weighted, eps, tau_local))

        return torch.cat(local_scores, dim=0) if local_scores else query_tokens.new_empty((0, way_num))


class ReliabilityCalibratedDistributionalHead(nn.Module):
    """Construct a support-defined class object and score queries against it."""

    def __init__(
        self,
        stable_dim: int,
        lambda_init: float = 1e-3,
        alpha_init: float = 1.0,
        tau_init: float = 10.0,
        gamma_init: float = 1.0,
        variance_floor: float = 1e-2,
        eps: float = 1e-6,
        use_bures_global: bool = True,
    ) -> None:
        super().__init__()
        if int(stable_dim) <= 0:
            raise ValueError("stable_dim must be positive")
        if float(lambda_init) <= 0.0:
            raise ValueError("lambda_init must be positive")
        if float(alpha_init) <= 0.0:
            raise ValueError("alpha_init must be positive")
        if float(tau_init) <= 0.1:
            raise ValueError("tau_init must be greater than 0.1")
        if float(variance_floor) <= 0.0:
            raise ValueError("variance_floor must be positive")
        if float(eps) <= 0.0:
            raise ValueError("eps must be positive")

        self.stable_dim = int(stable_dim)
        self.variance_floor = float(variance_floor)
        self.eps = float(eps)
        self.use_bures_global = bool(use_bures_global)

        self.lambda_raw = nn.Parameter(torch.tensor(_inverse_softplus(lambda_init), dtype=torch.float32))
        self.alpha_raw = nn.Parameter(torch.tensor(_inverse_softplus(alpha_init), dtype=torch.float32))
        self.tau_raw = nn.Parameter(torch.tensor(_inverse_softplus(tau_init - 0.1), dtype=torch.float32))
        # Log-det correction weight: γ controls how much the log-det term
        # penalizes high-variance classes. At γ=0 → original scoring.
        # At K→∞, γ converges to the true value → full Gaussian QDA.
        self.gamma_raw = nn.Parameter(torch.tensor(_inverse_softplus(max(float(gamma_init), 1e-4)), dtype=torch.float32))

    def lambda_value(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return F.softplus(self.lambda_raw).to(device=device, dtype=dtype)

    def alpha_value(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return F.softplus(self.alpha_raw).to(device=device, dtype=dtype)

    def tau_value(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return (F.softplus(self.tau_raw) + 0.1).to(device=device, dtype=dtype)

    def gamma_value(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return F.softplus(self.gamma_raw).to(device=device, dtype=dtype)

    def forward(
        self,
        query_global: torch.Tensor,
        support_global: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if query_global.dim() != 2:
            raise ValueError(f"query_global must have shape [NumQuery, D], got {tuple(query_global.shape)}")
        if support_global.dim() != 3:
            raise ValueError(f"support_global must have shape [Way, Shot, D], got {tuple(support_global.shape)}")
        if query_global.shape[-1] != self.stable_dim or support_global.shape[-1] != self.stable_dim:
            raise ValueError(
                f"Expected stable_dim={self.stable_dim}, got query={query_global.shape[-1]} "
                f"support={support_global.shape[-1]}"
            )

        query_global = F.normalize(query_global, p=2, dim=-1)
        support_global = F.normalize(support_global, p=2, dim=-1)

        mu_bar = support_global.mean(dim=1)
        sq_dist_to_center = (support_global - mu_bar.unsqueeze(1)).square().sum(dim=-1)
        lambda_value = self.lambda_value(query_global.device, query_global.dtype)
        support_relevance = torch.exp(-lambda_value * sq_dist_to_center)
        support_weights = support_relevance / support_relevance.sum(dim=1, keepdim=True).clamp_min(self.eps)

        prototype = torch.sum(support_weights.unsqueeze(-1) * support_global, dim=1)
        sq_dev = (support_global - prototype.unsqueeze(1)).square()
        diagonal_variance = torch.sum(support_weights.unsqueeze(-1) * sq_dev, dim=1)
        variance = diagonal_variance.clamp_min(self.variance_floor)

        compactness = diagonal_variance.mean(dim=-1)
        alpha_value = self.alpha_value(query_global.device, query_global.dtype)
        reliability = torch.exp(-alpha_value * compactness).clamp(min=self.eps, max=1.0)

        diff = query_global.unsqueeze(1) - prototype.unsqueeze(0)
        euclidean_distance = diff.square().sum(dim=-1)
        mahalanobis_distance = (diff.square() / variance.unsqueeze(0).clamp_min(self.eps)).sum(dim=-1)

        # ===== Architectural contribution: log-determinant correction =====
        # Full Gaussian log-likelihood: log p(q|c) = -½ d_M(q,c) - ½ Σ_j log v_{c,j} - const
        # The log-det term penalizes high-variance classes, which is missing
        # in pure Mahalanobis scoring.
        # d_corrected = ρ_c · [d_M(q,c) + γ · Σ_j log v_{c,j}] + (1-ρ_c) · d_E(q,c)
        gamma_value = self.gamma_value(query_global.device, query_global.dtype)
        log_det_per_class = variance.clamp_min(self.eps).log().sum(dim=-1)  # (Way,)
        mahalanobis_with_logdet = mahalanobis_distance + gamma_value * log_det_per_class.unsqueeze(0)

        # ===== Architectural contribution: reliability-modulated distance blend =====
        total_distance = reliability.unsqueeze(0) * mahalanobis_with_logdet + (
            1.0 - reliability.unsqueeze(0)
        ) * euclidean_distance
        total_distance = bw_distance if self.use_bures_global else original_distance

        tau_value = self.tau_value(query_global.device, query_global.dtype)
        global_scores = -total_distance / tau_value

        return {
            "global_scores": global_scores,
            "prototype": prototype,
            "variance": variance,
            "compactness": compactness,
            "reliability": reliability,
            "support_weights": support_weights,
            "mahalanobis_distance": mahalanobis_distance,
            "euclidean_distance": euclidean_distance,
            "log_det_per_class": log_det_per_class,
            "total_distance": total_distance,
            "lambda_value": lambda_value.detach(),
            "alpha_value": alpha_value.detach(),
            "tau_value": tau_value.detach(),
            "gamma_value": gamma_value.detach(),
        }


class SPIFRDP(BaseConv64FewShotModel):
    """SPIF-RDP with global BW scoring and Sinkhorn-OT local residuals."""

    def __init__(
        self,
        in_channels: int = 3,
        hidden_dim: int = 64,
        stable_dim: int = 64,
        variant_dim: int = 64,
        gate_hidden: int = 16,
        top_r: int = 5,
        gate_on: bool = True,
        factorization_on: bool = True,
        global_only: bool = False,
        local_only: bool = False,
        token_l2norm: bool = True,
        consistency_weight: float = 0.0,
        decorr_weight: float = 0.0,
        sparse_weight: float = 0.0,
        consistency_dropout: float = 0.1,
        rdp_lambda_init: float = 1e-3,
        rdp_alpha_init: float = 1.0,
        rdp_tau_init: float = 10.0,
        rdp_gamma_init: float = 1.0,
        rdp_variance_floor: float = 1e-2,
        rdp_compact_loss_weight: float = 0.1,
        rdp_sep_loss_weight: float = 0.05,
        rdp_sep_margin: float = 0.5,
        rdp_eps: float = 1e-6,
        rdp_beta_init: float = 0.5,
        rdp_rho_var_weight: float = 0.01,
        local_n_sinkhorn_iters: int = 5,
        local_eps_init: float = 0.1,
        local_tau_init: float = 1.0,
        local_chunk_size: int = 16,
        local_chunk_threshold: int = 10_000_000,
        local_use_vhat_coupling: bool = True,
        use_bures_global: bool = True,
        backbone_name: str = "resnet12",
        image_size: int = 64,
        resnet12_drop_rate: float = 0.0,
        resnet12_dropblock_size: int = 5,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            backbone_name=backbone_name,
            image_size=image_size,
            resnet12_drop_rate=resnet12_drop_rate,
            resnet12_dropblock_size=resnet12_dropblock_size,
        )
        if global_only and local_only:
            raise ValueError("global_only and local_only cannot both be true")
        if int(top_r) <= 0:
            raise ValueError("top_r must be positive")
        if float(rdp_compact_loss_weight) < 0.0 or float(rdp_sep_loss_weight) < 0.0:
            raise ValueError("rdp loss weights must be non-negative")
        if float(rdp_sep_margin) < 0.0:
            raise ValueError("rdp_sep_margin must be non-negative")
        if float(rdp_eps) <= 0.0:
            raise ValueError("rdp_eps must be positive")
        if float(rdp_beta_init) <= 0.0:
            raise ValueError("rdp_beta_init must be positive")
        if float(rdp_rho_var_weight) < 0.0:
            raise ValueError("rdp_rho_var_weight must be non-negative")

        self.stable_dim = int(stable_dim)
        self.variant_dim = int(variant_dim)
        self.top_r = int(top_r)
        self.gate_on = bool(gate_on)
        self.factorization_on = bool(factorization_on)
        self.global_only = bool(global_only)
        self.local_only = bool(local_only)
        self.token_l2norm = bool(token_l2norm)
        self.consistency_weight = float(consistency_weight)
        self.decorr_weight = float(decorr_weight)
        self.sparse_weight = float(sparse_weight)
        self.consistency_dropout = float(consistency_dropout)
        self.rdp_compact_loss_weight = float(rdp_compact_loss_weight)
        self.rdp_sep_loss_weight = float(rdp_sep_loss_weight)
        self.rdp_sep_margin = float(rdp_sep_margin)
        self.rdp_eps = float(rdp_eps)
        self.rdp_rho_var_weight = float(rdp_rho_var_weight)
        self.use_bures_global = bool(use_bures_global)

        self.encoder_head = SPIFEncoder(
            input_dim=hidden_dim,
            stable_dim=self.stable_dim,
            variant_dim=self.variant_dim,
            gate_hidden=int(gate_hidden),
            token_l2norm=self.token_l2norm,
        )
        self.variant_align = (
            nn.Identity()
            if self.variant_dim == self.stable_dim
            else nn.Linear(self.variant_dim, self.stable_dim, bias=False)
        )
        self.distributional_head = ReliabilityCalibratedDistributionalHead(
            stable_dim=self.stable_dim,
            lambda_init=rdp_lambda_init,
            alpha_init=rdp_alpha_init,
            tau_init=rdp_tau_init,
            gamma_init=rdp_gamma_init,
            variance_floor=rdp_variance_floor,
            eps=self.rdp_eps,
            use_bures_global=self.use_bures_global,
        )
        self.local_branch = SinkhornOTLocalBranch(
            stable_dim=self.stable_dim,
            n_sinkhorn_iters=int(local_n_sinkhorn_iters),
            eps_init=local_eps_init,
            tau_local_init=local_tau_init,
            variance_floor=rdp_variance_floor,
            chunk_size=int(local_chunk_size),
            chunk_threshold=int(local_chunk_threshold),
            use_vhat_coupling=bool(local_use_vhat_coupling),
        )
        self.beta_raw = nn.Parameter(torch.tensor(_inverse_softplus(rdp_beta_init), dtype=torch.float32))

    def encode_tokens(self, images: torch.Tensor):
        tokens = feature_map_to_tokens(self.encode(images))
        return self.encoder_head(tokens, factorization_on=self.factorization_on, gate_on=self.gate_on)

    def _encode_episode(self, query: torch.Tensor, support: torch.Tensor) -> dict[str, torch.Tensor]:
        way_num, shot_num = support.shape[:2]
        flat_support = support.reshape(way_num * shot_num, *support.shape[-3:])
        query_outputs = self.encode_tokens(query)
        support_outputs = self.encode_tokens(flat_support)
        return {
            "query_global": query_outputs.stable_global,
            "query_tokens": query_outputs.stable_tokens,
            "query_variant_global": query_outputs.variant_global,
            "query_gate": query_outputs.gate,
            "support_global": support_outputs.stable_global.reshape(way_num, shot_num, -1),
            "support_tokens": support_outputs.stable_tokens.reshape(
                way_num,
                shot_num,
                -1,
                support_outputs.stable_tokens.shape[-1],
            ),
            "support_variant_global": support_outputs.variant_global.reshape(way_num, shot_num, -1),
            "support_gate": support_outputs.gate.reshape(way_num, shot_num, -1, 1),
        }

    def _consistency_loss(self, stable_tokens: torch.Tensor) -> torch.Tensor:
        if self.consistency_dropout <= 0.0:
            return stable_tokens.new_zeros(())
        view1 = F.dropout(stable_tokens, p=self.consistency_dropout, training=True).mean(dim=1)
        view2 = F.dropout(stable_tokens, p=self.consistency_dropout, training=True).mean(dim=1)
        view1 = F.normalize(view1, p=2, dim=-1)
        view2 = F.normalize(view2, p=2, dim=-1)
        return (1.0 - F.cosine_similarity(view1, view2, dim=-1)).mean()

    @staticmethod
    def _decorrelation_loss(stable_global: torch.Tensor, variant_global: torch.Tensor) -> torch.Tensor:
        return F.cosine_similarity(
            F.normalize(stable_global, p=2, dim=-1),
            F.normalize(variant_global, p=2, dim=-1),
            dim=-1,
        ).pow(2).mean()

    @staticmethod
    def _sparse_gate_loss(gate: torch.Tensor) -> torch.Tensor:
        return gate.mean()

    def _factorization_aux_loss(self, episode: dict[str, torch.Tensor]) -> torch.Tensor:
        if not self.training:
            return episode["query_tokens"].new_zeros(())

        loss = episode["query_tokens"].new_zeros(())
        if self.consistency_weight > 0.0:
            all_stable_tokens = torch.cat(
                [
                    episode["query_tokens"],
                    episode["support_tokens"].reshape(
                        -1,
                        episode["support_tokens"].shape[-2],
                        episode["support_tokens"].shape[-1],
                    ),
                ],
                dim=0,
            )
            loss = loss + self.consistency_weight * self._consistency_loss(all_stable_tokens)

        if self.factorization_on and self.decorr_weight > 0.0:
            stable_global = torch.cat(
                [episode["query_global"], episode["support_global"].reshape(-1, episode["support_global"].shape[-1])],
                dim=0,
            )
            variant_global = torch.cat(
                [
                    episode["query_variant_global"],
                    episode["support_variant_global"].reshape(-1, episode["support_variant_global"].shape[-1]),
                ],
                dim=0,
            )
            variant_global = self.variant_align(variant_global)
            loss = loss + self.decorr_weight * self._decorrelation_loss(stable_global, variant_global)

        if self.gate_on and self.sparse_weight > 0.0:
            all_gates = torch.cat(
                [episode["query_gate"].reshape(-1, 1), episode["support_gate"].reshape(-1, 1)],
                dim=0,
            )
            loss = loss + self.sparse_weight * self._sparse_gate_loss(all_gates)
        return loss

    def compute_separation_loss(self, prototype: torch.Tensor) -> torch.Tensor:
        if prototype.shape[0] < 2 or self.rdp_sep_loss_weight <= 0.0:
            return prototype.new_zeros(())
        pairwise_sqdist = torch.cdist(prototype, prototype, p=2).square() / float(prototype.shape[-1])
        off_diag_mask = ~torch.eye(prototype.shape[0], dtype=torch.bool, device=prototype.device)
        violations = F.relu(self.rdp_sep_margin - pairwise_sqdist[off_diag_mask])
        return violations.mean() if violations.numel() > 0 else prototype.new_zeros(())

    def _forward_episode(self, query: torch.Tensor, support: torch.Tensor) -> dict[str, torch.Tensor | None]:
        episode = self._encode_episode(query, support)
        global_outputs = self.distributional_head(
            query_global=episode["query_global"],
            support_global=episode["support_global"],
        )

        global_scores = global_outputs["global_scores"]
        local_scores = None
        beta_eff = None
        logits = global_scores

        if not self.global_only:
            local_scores = self.local_branch(
                query_tokens=episode["query_tokens"],
                support_tokens=episode["support_tokens"],
                variance=global_outputs["variance"],
            )
            rho_bar = global_outputs["reliability"].mean()
            beta_value = F.softplus(self.beta_raw).to(device=global_scores.device, dtype=global_scores.dtype)
            beta_eff = beta_value * (1.0 - rho_bar)
            fused_logits = global_scores + beta_eff * local_scores
            logits = local_scores if self.local_only else fused_logits

        compact_loss = global_outputs["compactness"].mean()
        separation_loss = self.compute_separation_loss(global_outputs["prototype"])
        factorization_aux = self._factorization_aux_loss(episode)
        rho_c = global_outputs["reliability"]
        rho_var_loss = rho_c.var() if rho_c.numel() > 1 else rho_c.new_zeros(())

        if self.training:
            aux_loss = (
                self.rdp_compact_loss_weight * compact_loss
                + self.rdp_sep_loss_weight * separation_loss
                + factorization_aux
                + self.rdp_rho_var_weight * rho_var_loss
            )
        else:
            aux_loss = logits.new_zeros(())

        mean_gate = torch.cat(
            [episode["query_gate"].reshape(-1, 1), episode["support_gate"].reshape(-1, 1)],
            dim=0,
        ).mean()

        reliability_expanded = global_outputs["reliability"].unsqueeze(0).expand(query.shape[0], -1)
        compactness_expanded = global_outputs["compactness"].unsqueeze(0).expand(query.shape[0], -1)

        return {
            "logits": logits,
            "aux_loss": aux_loss,
            "global_scores": global_scores.detach(),
            "total_distance": global_outputs["total_distance"].detach(),
            "mahalanobis_distance": global_outputs["mahalanobis_distance"].detach(),
            "euclidean_distance": global_outputs["euclidean_distance"].detach(),
            "class_reliability": reliability_expanded.detach(),
            "class_compactness": compactness_expanded.detach(),
            "prototype": global_outputs["prototype"].detach(),
            "variance": global_outputs["variance"].detach(),
            "support_weights": global_outputs["support_weights"].detach(),
            "mean_reliability": global_outputs["reliability"].mean().detach(),
            "compact_loss": compact_loss.detach(),
            "separation_loss": separation_loss.detach(),
            "factorization_aux_loss": factorization_aux.detach(),
            "lambda_value": global_outputs["lambda_value"],
            "alpha_value": global_outputs["alpha_value"],
            "tau_value": global_outputs["tau_value"],
            "gamma_value": global_outputs["gamma_value"],
            "log_det_per_class": global_outputs["log_det_per_class"].detach(),
            "mean_gate": mean_gate.detach(),
            "stable_global_embeddings": torch.cat(
                [
                    episode["query_global"],
                    episode["support_global"].reshape(-1, episode["support_global"].shape[-1]),
                ],
                dim=0,
            ).detach(),
            "variant_global_embeddings": torch.cat(
                [
                    episode["query_variant_global"],
                    episode["support_variant_global"].reshape(-1, episode["support_variant_global"].shape[-1]),
                ],
                dim=0,
            ).detach(),
            "local_scores": None if local_scores is None else local_scores.detach(),
            "bw_distance": global_outputs["bw_distance"].detach(),
            "rho_var_loss": rho_var_loss.detach(),
            "beta_eff": None if beta_eff is None else beta_eff.detach(),
        }

    def forward(
        self,
        query: torch.Tensor,
        support: torch.Tensor,
        return_aux: bool = False,
    ) -> torch.Tensor | Dict[str, torch.Tensor]:
        bsz, _, _, _, _, _ = self.validate_episode_inputs(query, support)
        batch_outputs = []
        diagnostics = []
        aux_losses = []
        for batch_idx in range(bsz):
            episode = self._forward_episode(query[batch_idx], support[batch_idx])
            batch_outputs.append(episode["logits"])
            diagnostics.append(episode)
            aux_losses.append(episode["aux_loss"])

        logits = torch.cat(batch_outputs, dim=0)
        aux_loss = torch.stack(aux_losses).mean() if aux_losses else logits.new_zeros(())

        if not return_aux:
            if self.training:
                return SPIFRDPOutput(
                    {
                        "logits": logits,
                        "aux_loss": aux_loss,
                    }
                )
            return logits

        return {
            "logits": logits,
            "aux_loss": aux_loss,
            "global_scores": torch.cat([item["global_scores"] for item in diagnostics], dim=0),
            "total_distance": torch.cat([item["total_distance"] for item in diagnostics], dim=0),
            "mahalanobis_distance": torch.cat([item["mahalanobis_distance"] for item in diagnostics], dim=0),
            "euclidean_distance": torch.cat([item["euclidean_distance"] for item in diagnostics], dim=0),
            "class_reliability": torch.cat([item["class_reliability"] for item in diagnostics], dim=0),
            "class_compactness": torch.cat([item["class_compactness"] for item in diagnostics], dim=0),
            "prototype": torch.stack([item["prototype"] for item in diagnostics], dim=0),
            "variance": torch.stack([item["variance"] for item in diagnostics], dim=0),
            "support_weights": torch.stack([item["support_weights"] for item in diagnostics], dim=0),
            "mean_reliability": torch.stack([item["mean_reliability"] for item in diagnostics]).mean(),
            "compact_loss": torch.stack([item["compact_loss"] for item in diagnostics]).mean(),
            "separation_loss": torch.stack([item["separation_loss"] for item in diagnostics]).mean(),
            "factorization_aux_loss": torch.stack([item["factorization_aux_loss"] for item in diagnostics]).mean(),
            "lambda_value": torch.stack([item["lambda_value"] for item in diagnostics]).mean(),
            "alpha_value": torch.stack([item["alpha_value"] for item in diagnostics]).mean(),
            "tau_value": torch.stack([item["tau_value"] for item in diagnostics]).mean(),
            "gamma_value": torch.stack([item["gamma_value"] for item in diagnostics]).mean(),
            "log_det_per_class": torch.stack([item["log_det_per_class"] for item in diagnostics], dim=0),
            "mean_gate": torch.stack([item["mean_gate"] for item in diagnostics]).mean(),
            "stable_global_embeddings": torch.cat(
                [item["stable_global_embeddings"] for item in diagnostics],
                dim=0,
            ),
            "variant_global_embeddings": torch.cat(
                [item["variant_global_embeddings"] for item in diagnostics],
                dim=0,
            ),
        }
