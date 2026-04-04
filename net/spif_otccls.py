"""SPIF-OTCCLS: SPIF factorization with OT global scoring and confusability local scoring.

This file implements the architecture described in ``neurocomputing_v2.md``:

- Backbone: ResNet12-family encoder with 3-channel RGB scalogram input
- Stable / variant projection heads with feature-wise gate
- Global branch: energy-weighted sliced Wasserstein distance
- Local branch: confusability-conditioned local cosine scoring
- Fusion: reliability-adaptive residual local contribution
"""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import ot
except ImportError:  # pragma: no cover - dependency availability is checked at runtime.
    ot = None

from net.fewshot_common import BaseConv64FewShotModel, feature_map_to_tokens


def _inverse_softplus(value: float, floor: float = 1e-6) -> float:
    value = max(float(value), float(floor))
    return math.log(math.expm1(value))


class SPIFOTCCLSOutput(dict):
    """Dict-like model output that still exposes `.shape` via logits."""

    @property
    def shape(self):
        logits = self.get("logits")
        return None if logits is None else logits.shape


class SPIFOTCCLS(BaseConv64FewShotModel):
    """SPIF-OTCCLS episode model."""

    def __init__(
        self,
        in_channels: int = 3,
        hidden_dim: int = 640,
        feature_dim: int = 256,
        gate_hidden: int = 128,
        num_projections: int = 64,
        num_quantiles: int = 100,
        tau_g_init: float = 10.0,
        tau_l_init: float = 1.0,
        beta_init: float = 0.5,
        alpha_init: float = 1.0,
        compact_loss_weight: float = 0.1,
        decorr_loss_weight: float = 0.05,
        entropy_loss_weight: float = 0.01,
        global_only: bool = False,
        local_only: bool = False,
        swd_backend: str = "pot",
        eps: float = 1e-6,
        backbone_name: str = "resnet12",
        image_size: int = 84,
        resnet12_drop_rate: float = 0.05,
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
        if int(feature_dim) <= 0:
            raise ValueError("feature_dim must be positive")
        if int(gate_hidden) <= 0:
            raise ValueError("gate_hidden must be positive")
        if int(num_projections) <= 0:
            raise ValueError("num_projections must be positive")
        if int(num_quantiles) <= 1:
            raise ValueError("num_quantiles must be greater than 1")
        if float(tau_g_init) <= 0.1:
            raise ValueError("tau_g_init must be greater than 0.1")
        if float(tau_l_init) <= 0.1:
            raise ValueError("tau_l_init must be greater than 0.1")
        if float(beta_init) <= 0.0:
            raise ValueError("beta_init must be positive")
        if float(alpha_init) <= 0.0:
            raise ValueError("alpha_init must be positive")
        if float(compact_loss_weight) < 0.0:
            raise ValueError("compact_loss_weight must be non-negative")
        if float(decorr_loss_weight) < 0.0:
            raise ValueError("decorr_loss_weight must be non-negative")
        if float(entropy_loss_weight) < 0.0:
            raise ValueError("entropy_loss_weight must be non-negative")
        if float(eps) <= 0.0:
            raise ValueError("eps must be positive")

        self.feature_dim = int(feature_dim)
        self.gate_hidden = int(gate_hidden)
        self.num_projections = int(num_projections)
        self.num_quantiles = int(num_quantiles)
        self.global_only = bool(global_only)
        self.local_only = bool(local_only)
        self.swd_backend = str(swd_backend).lower()
        self.eps = float(eps)
        self.compact_loss_weight = float(compact_loss_weight)
        self.decorr_loss_weight = float(decorr_loss_weight)
        self.entropy_loss_weight = float(entropy_loss_weight)
        if self.swd_backend not in {"pot", "quantile", "auto"}:
            raise ValueError("swd_backend must be one of {'pot', 'quantile', 'auto'}")
        if self.swd_backend == "pot" and ot is None:
            raise ImportError("SPIF-OTCCLS with swd_backend='pot' requires the POT package (`import ot`).")

        self.W_s = nn.Linear(hidden_dim, self.feature_dim, bias=False)
        self.W_v = nn.Linear(hidden_dim, self.feature_dim, bias=False)
        self.gate_mlp = nn.Sequential(
            nn.Linear(hidden_dim, self.gate_hidden, bias=True),
            nn.ReLU(inplace=False),
            nn.Linear(self.gate_hidden, self.feature_dim, bias=True),
            nn.Sigmoid(),
        )

        generator = torch.Generator(device="cpu")
        generator.manual_seed(42)
        projections = torch.randn(self.num_projections, self.feature_dim, generator=generator)
        projections = F.normalize(projections, p=2, dim=1)
        self.register_buffer("projections", projections)
        quantile_levels = torch.linspace(0.01, 0.99, self.num_quantiles, dtype=torch.float32)
        self.register_buffer("quantile_levels", quantile_levels)

        self.tau_g_raw = nn.Parameter(
            torch.tensor(_inverse_softplus(max(float(tau_g_init) - 0.1, 1e-6)), dtype=torch.float32)
        )
        self.tau_l_raw = nn.Parameter(
            torch.tensor(_inverse_softplus(max(float(tau_l_init) - 0.1, 1e-6)), dtype=torch.float32)
        )
        self.beta_raw = nn.Parameter(torch.tensor(_inverse_softplus(beta_init), dtype=torch.float32))
        self.alpha_raw = nn.Parameter(torch.tensor(_inverse_softplus(alpha_init), dtype=torch.float32))

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.W_s.weight)
        nn.init.xavier_uniform_(self.W_v.weight)
        for layer in self.gate_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def tau_g_value(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return (F.softplus(self.tau_g_raw) + 0.1).to(device=device, dtype=dtype)

    def tau_l_value(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return (F.softplus(self.tau_l_raw) + 0.1).to(device=device, dtype=dtype)

    def beta0_value(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return F.softplus(self.beta_raw).to(device=device, dtype=dtype)

    def alpha_value(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return F.softplus(self.alpha_raw).to(device=device, dtype=dtype)

    def _encode_images(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        feature_map = self.encode(images)
        tokens = feature_map_to_tokens(feature_map)
        stable_features = self.W_s(tokens)
        variant_features = self.W_v(tokens)
        gate = self.gate_mlp(tokens)

        stable_tokens = gate * stable_features
        variant_tokens = gate * variant_features
        gate_sum = gate.sum(dim=1).clamp_min(self.eps)

        stable_global = (gate * stable_tokens).sum(dim=1) / gate_sum
        variant_global = (gate * variant_tokens).sum(dim=1) / gate_sum

        return {
            "stable_tokens": stable_tokens,
            "variant_tokens": variant_tokens,
            "gate": gate,
            "stable_global": stable_global,
            "variant_global": variant_global,
        }

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self._encode_images(x)["stable_global"]

    def _weighted_quantile(self, values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        if values.dim() != 2 or weights.dim() != 2:
            raise ValueError(
                "weighted quantile inputs must both be 2D, "
                f"got values={tuple(values.shape)} weights={tuple(weights.shape)}"
            )
        if values.shape != weights.shape:
            raise ValueError(
                f"weighted quantile inputs must share shape, got values={tuple(values.shape)} "
                f"weights={tuple(weights.shape)}"
            )

        batch_size, num_atoms = values.shape
        if batch_size == 0:
            return values.new_empty((0, self.num_quantiles))

        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(self.eps)
        sorted_vals, sort_idx = values.sort(dim=1)
        sorted_weights = weights.gather(1, sort_idx)

        cdf = sorted_weights.cumsum(dim=1)
        cdf_mid = (cdf - 0.5 * sorted_weights).clamp(min=0.0, max=1.0)
        quantiles = self.quantile_levels.to(device=values.device, dtype=values.dtype)
        quantiles = quantiles.unsqueeze(0).expand(batch_size, -1).contiguous()
        cdf_mid = cdf_mid.contiguous()

        idx_hi = torch.searchsorted(cdf_mid, quantiles, right=False).clamp(max=num_atoms - 1)
        idx_lo = (idx_hi - 1).clamp(min=0)

        cdf_lo = cdf_mid.gather(1, idx_lo)
        cdf_hi = cdf_mid.gather(1, idx_hi)
        val_lo = sorted_vals.gather(1, idx_lo)
        val_hi = sorted_vals.gather(1, idx_hi)

        denom = (cdf_hi - cdf_lo).clamp_min(self.eps)
        interp = ((quantiles - cdf_lo) / denom).clamp(0.0, 1.0)
        quantile_values = val_lo + interp * (val_hi - val_lo)
        same_idx = idx_hi.eq(idx_lo)
        return torch.where(same_idx, val_hi, quantile_values)

    def _compute_sliced_wasserstein(
        self,
        query_tokens: torch.Tensor,
        query_weights: torch.Tensor,
        prototype_tokens: torch.Tensor,
        prototype_weights: torch.Tensor,
    ) -> torch.Tensor:
        num_query = query_tokens.shape[0]
        way_num = prototype_tokens.shape[0]
        if num_query == 0:
            return prototype_tokens.new_empty((0, way_num))

        projections = self.projections.to(device=query_tokens.device, dtype=query_tokens.dtype)
        query_proj = torch.matmul(query_tokens, projections.t())
        proto_proj = torch.matmul(prototype_tokens, projections.t())

        backend = self.swd_backend
        if backend == "auto":
            backend = "pot" if ot is not None else "quantile"
        if backend == "pot":
            return self._compute_sliced_wasserstein_pot(query_proj, query_weights, proto_proj, prototype_weights)

        swd = query_tokens.new_zeros((num_query, way_num))
        for proj_idx in range(self.num_projections):
            query_quantiles = self._weighted_quantile(query_proj[:, :, proj_idx], query_weights)
            proto_quantiles = self._weighted_quantile(proto_proj[:, :, proj_idx], prototype_weights)
            swd = swd + (query_quantiles.unsqueeze(1) - proto_quantiles.unsqueeze(0)).abs().mean(dim=-1)
        return swd / float(self.num_projections)

    def _compute_sliced_wasserstein_pot(
        self,
        query_proj: torch.Tensor,
        query_weights: torch.Tensor,
        proto_proj: torch.Tensor,
        proto_weights: torch.Tensor,
    ) -> torch.Tensor:
        if ot is None:
            raise ImportError("POT is required for exact SWD backend, but `import ot` failed.")

        num_query, num_tokens, _ = query_proj.shape
        way_num, km_tokens, _ = proto_proj.shape
        swd = query_proj.new_zeros((num_query, way_num))

        for proj_idx in range(self.num_projections):
            q_1d = query_proj[:, :, proj_idx]
            p_1d = proto_proj[:, :, proj_idx]

            q_values = q_1d.unsqueeze(1).expand(num_query, way_num, num_tokens).reshape(num_query * way_num, num_tokens)
            p_values = p_1d.unsqueeze(0).expand(num_query, way_num, km_tokens).reshape(num_query * way_num, km_tokens)
            q_mass = query_weights.unsqueeze(1).expand(num_query, way_num, num_tokens).reshape(num_query * way_num, num_tokens)
            p_mass = proto_weights.unsqueeze(0).expand(num_query, way_num, km_tokens).reshape(num_query * way_num, km_tokens)

            distances = ot.wasserstein_1d(
                q_values.transpose(0, 1).contiguous(),
                p_values.transpose(0, 1).contiguous(),
                q_mass.transpose(0, 1).contiguous(),
                p_mass.transpose(0, 1).contiguous(),
                p=1,
                require_sort=True,
            )
            swd = swd + distances.view(num_query, way_num)

        return swd / float(self.num_projections)

    def _encode_episode(self, query: torch.Tensor, support: torch.Tensor) -> dict[str, torch.Tensor]:
        way_num, shot_num = support.shape[:2]
        flat_support = support.reshape(way_num * shot_num, *support.shape[-3:])

        query_outputs = self._encode_images(query)
        support_outputs = self._encode_images(flat_support)

        return {
            "query_stable_tokens": query_outputs["stable_tokens"],
            "query_variant_tokens": query_outputs["variant_tokens"],
            "query_gate": query_outputs["gate"],
            "query_stable_global": query_outputs["stable_global"],
            "query_variant_global": query_outputs["variant_global"],
            "support_stable_tokens": support_outputs["stable_tokens"].reshape(
                way_num,
                shot_num,
                -1,
                self.feature_dim,
            ),
            "support_variant_tokens": support_outputs["variant_tokens"].reshape(
                way_num,
                shot_num,
                -1,
                self.feature_dim,
            ),
            "support_gate": support_outputs["gate"].reshape(way_num, shot_num, -1, self.feature_dim),
            "support_stable_global": support_outputs["stable_global"].reshape(way_num, shot_num, self.feature_dim),
            "support_variant_global": support_outputs["variant_global"].reshape(way_num, shot_num, self.feature_dim),
        }

    def _compute_local_scores(
        self,
        query_tokens: torch.Tensor,
        support_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        way_num = support_tokens.shape[0]
        mu_ct = support_tokens.mean(dim=1)

        mu_i = mu_ct.unsqueeze(1)
        mu_j = mu_ct.unsqueeze(0)
        pair_dist = (mu_i - mu_j).norm(dim=-1)
        diag_mask = torch.eye(way_num, device=pair_dist.device, dtype=torch.bool).unsqueeze(-1)
        pair_dist = pair_dist.masked_fill(diag_mask, float("inf"))
        delta_ct = pair_dist.min(dim=1).values

        tau_l = self.tau_l_value(query_tokens.device, query_tokens.dtype)
        inv_delta = 1.0 / (delta_ct + self.eps)
        a_ct = F.softmax(inv_delta / tau_l, dim=1)

        cosine_scores = F.cosine_similarity(
            query_tokens.unsqueeze(1),
            mu_ct.unsqueeze(0),
            dim=-1,
        )
        local_scores = (a_ct.unsqueeze(0) * cosine_scores).sum(dim=-1)
        return local_scores, a_ct, tau_l

    @staticmethod
    def _decorrelation_loss(stable_global: torch.Tensor, variant_global: torch.Tensor) -> torch.Tensor:
        stable_norm = F.normalize(stable_global, p=2, dim=-1)
        variant_norm = F.normalize(variant_global, p=2, dim=-1)
        return (stable_norm * variant_norm).sum(dim=-1).pow(2).mean()

    def _top2_margin(self, scores: torch.Tensor) -> torch.Tensor:
        if scores.shape[1] < 2:
            return scores.new_zeros(())
        top2 = scores.topk(k=2, dim=1).values
        return (top2[:, 0] - top2[:, 1]).mean()

    def _forward_episode(self, query: torch.Tensor, support: torch.Tensor) -> dict[str, torch.Tensor | None]:
        episode = self._encode_episode(query, support)
        support_tokens = episode["support_stable_tokens"]
        query_tokens = episode["query_stable_tokens"]
        support_global = episode["support_stable_global"]

        way_num, shot_num, num_tokens, _ = support_tokens.shape
        num_query = query_tokens.shape[0]

        energy = support_tokens.square().sum(dim=-1).reshape(way_num, shot_num * num_tokens)
        support_weights = energy / energy.sum(dim=1, keepdim=True).clamp_min(self.eps)
        prototype_tokens = support_tokens.reshape(way_num, shot_num * num_tokens, self.feature_dim)
        query_weights = query_tokens.new_full((num_query, num_tokens), 1.0 / float(num_tokens))

        swd = self._compute_sliced_wasserstein(
            query_tokens=query_tokens,
            query_weights=query_weights,
            prototype_tokens=prototype_tokens,
            prototype_weights=support_weights,
        )
        tau_g = self.tau_g_value(query_tokens.device, query_tokens.dtype)
        global_scores = -swd / tau_g

        local_scores, conf_attention, tau_l = self._compute_local_scores(query_tokens, support_tokens)

        support_proto = support_global.mean(dim=1)
        scatter = ((support_global - support_proto.unsqueeze(1)).square().sum(dim=-1)).mean()
        alpha = self.alpha_value(query_tokens.device, query_tokens.dtype)
        rho_bar = torch.exp(-alpha * scatter / float(self.feature_dim)).clamp(min=self.eps, max=1.0)
        beta0 = self.beta0_value(query_tokens.device, query_tokens.dtype)
        beta_eff = beta0 * (1.0 - rho_bar)

        if self.global_only:
            logits = global_scores
        elif self.local_only:
            logits = local_scores
        else:
            logits = global_scores + beta_eff * local_scores

        compact_loss = ((support_global - support_proto.unsqueeze(1)).square().sum(dim=-1)).mean()
        entropy = -(conf_attention * (conf_attention + self.eps).log()).sum(dim=1)
        entropy_loss = -entropy.mean()

        stable_global_all = torch.cat(
            [episode["query_stable_global"], support_global.reshape(-1, self.feature_dim)],
            dim=0,
        )
        variant_global_all = torch.cat(
            [episode["query_variant_global"], episode["support_variant_global"].reshape(-1, self.feature_dim)],
            dim=0,
        )
        decorr_loss = self._decorrelation_loss(stable_global_all, variant_global_all)

        if self.training:
            aux_loss = (
                self.compact_loss_weight * compact_loss
                + self.decorr_loss_weight * decorr_loss
                + self.entropy_loss_weight * entropy_loss
            )
        else:
            aux_loss = logits.new_zeros(())

        mean_gate = torch.cat(
            [episode["query_gate"].reshape(-1, 1), episode["support_gate"].reshape(-1, 1)],
            dim=0,
        ).mean()
        global_margin = self._top2_margin(global_scores)
        local_margin = self._top2_margin(local_scores)
        support_energy_entropy = -(support_weights * support_weights.clamp_min(self.eps).log()).sum(dim=1).mean()
        attention_entropy = entropy.mean()

        return {
            "logits": logits,
            "aux_loss": aux_loss,
            "global_scores": global_scores.detach(),
            "local_scores": None if self.global_only else local_scores.detach(),
            "total_distance": swd.detach(),
            "support_weights": support_weights.detach(),
            "confusability_attention": conf_attention.detach(),
            "prototype_tokens": prototype_tokens.detach(),
            "stable_global_embeddings": stable_global_all.detach(),
            "variant_global_embeddings": variant_global_all.detach(),
            "mean_gate": mean_gate.detach(),
            "mean_reliability": rho_bar.detach(),
            "rho_bar": rho_bar.detach(),
            "beta_eff": beta_eff.detach(),
            "beta0": beta0.detach(),
            "alpha_value": alpha.detach(),
            "tau_value": tau_g.detach(),
            "tau_local_value": tau_l.detach(),
            "compact_loss": compact_loss.detach(),
            "decorr_loss": decorr_loss.detach(),
            "entropy_loss": entropy_loss.detach(),
            "global_margin": global_margin.detach(),
            "local_margin": local_margin.detach(),
            "support_energy_entropy": support_energy_entropy.detach(),
            "mean_attention_entropy": attention_entropy.detach(),
        }

    def forward(
        self,
        query: torch.Tensor,
        support: torch.Tensor,
        return_aux: bool = False,
    ) -> torch.Tensor | Dict[str, torch.Tensor]:
        bsz, _, _, _, _, _ = self.validate_episode_inputs(query, support)
        batch_logits = []
        batch_aux = []
        diagnostics = []

        for batch_idx in range(bsz):
            episode = self._forward_episode(query[batch_idx], support[batch_idx])
            batch_logits.append(episode["logits"])
            batch_aux.append(episode["aux_loss"])
            diagnostics.append(episode)

        logits = torch.cat(batch_logits, dim=0)
        aux_loss = torch.stack(batch_aux).mean() if batch_aux else logits.new_zeros(())

        if not return_aux:
            if self.training:
                return SPIFOTCCLSOutput({"logits": logits, "aux_loss": aux_loss})
            return logits

        local_scores = None
        if diagnostics and diagnostics[0]["local_scores"] is not None:
            local_scores = torch.cat([item["local_scores"] for item in diagnostics], dim=0)

        return SPIFOTCCLSOutput(
            {
                "logits": logits,
                "aux_loss": aux_loss,
                "global_scores": torch.cat([item["global_scores"] for item in diagnostics], dim=0),
                "local_scores": local_scores,
                "total_distance": torch.cat([item["total_distance"] for item in diagnostics], dim=0),
                "support_weights": torch.stack([item["support_weights"] for item in diagnostics], dim=0),
                "confusability_attention": torch.stack(
                    [item["confusability_attention"] for item in diagnostics],
                    dim=0,
                ),
                "prototype_tokens": torch.stack([item["prototype_tokens"] for item in diagnostics], dim=0),
                "stable_global_embeddings": torch.cat(
                    [item["stable_global_embeddings"] for item in diagnostics],
                    dim=0,
                ),
                "variant_global_embeddings": torch.cat(
                    [item["variant_global_embeddings"] for item in diagnostics],
                    dim=0,
                ),
                "mean_gate": torch.stack([item["mean_gate"] for item in diagnostics]).mean(),
                "mean_reliability": torch.stack([item["mean_reliability"] for item in diagnostics]).mean(),
                "rho_bar": torch.stack([item["rho_bar"] for item in diagnostics]).mean(),
                "beta_eff": torch.stack([item["beta_eff"] for item in diagnostics]).mean(),
                "beta0": torch.stack([item["beta0"] for item in diagnostics]).mean(),
                "alpha_value": torch.stack([item["alpha_value"] for item in diagnostics]).mean(),
                "tau_value": torch.stack([item["tau_value"] for item in diagnostics]).mean(),
                "tau_local_value": torch.stack([item["tau_local_value"] for item in diagnostics]).mean(),
                "compact_loss": torch.stack([item["compact_loss"] for item in diagnostics]).mean(),
                "decorr_loss": torch.stack([item["decorr_loss"] for item in diagnostics]).mean(),
                "entropy_loss": torch.stack([item["entropy_loss"] for item in diagnostics]).mean(),
                "global_margin": torch.stack([item["global_margin"] for item in diagnostics]).mean(),
                "local_margin": torch.stack([item["local_margin"] for item in diagnostics]).mean(),
                "support_energy_entropy": torch.stack(
                    [item["support_energy_entropy"] for item in diagnostics]
                ).mean(),
                "mean_attention_entropy": torch.stack(
                    [item["mean_attention_entropy"] for item in diagnostics]
                ).mean(),
            }
        )
