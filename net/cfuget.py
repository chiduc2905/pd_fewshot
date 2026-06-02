"""Class-contrastive rival-calibrated unbalanced evidence transport."""

import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from net.fewshot_common import BaseConv64FewShotModel
from net.modules.fgw_uot_solver import (
    fgw_uot_solve,
    normalize_intra_dist,
    pairwise_sq_l2,
)


SHOT_AGGREGATIONS = {"mean", "j_logmeanexp"}
ABLATION_MODES = {
    "full",
    "no_structure",
    "no_rival",
    "no_coherent_score",
    "cost_only",
    "feature_only",
    "with_structure",
}


class CFUGETResult(dict):
    """Dict-like output that exposes `.shape` through logits."""

    @property
    def shape(self):
        logits = self.get("logits")
        return None if logits is None else logits.shape


class CFUGETFewShot(BaseConv64FewShotModel):
    """Few-shot classifier with rival-calibrated coherent UOT evidence."""

    def __init__(
        self,
        in_channels: int = 3,
        hidden_dim: int = 64,
        token_dim: int = 128,
        backbone_name: str = "conv64f",
        image_size: int = 84,
        rho: float = 0.8,
        tau: float = 0.5,
        eps_sinkhorn: float = 0.08,
        fgw_iters: int = 1,
        sinkhorn_iters: int = 50,
        sinkhorn_tol: float = 1e-5,
        alpha_init: float = 1e-6,
        score_scale_init: float = 16.0,
        threshold_init: float = 0.5,
        rival_temperature: float = 0.07,
        rival_margin: float = 0.02,
        coherent_temperature: float = 0.10,
        spatial_structure_weight: float = 0.0,
        mass_weight: float = 1.0,
        cost_weight: float = 1.0,
        normalize_tokens: bool = True,
        structure_detach: bool = False,
        rival_detach: bool = True,
        ablation_mode: str = "full",
        shot_aggregation: str = "j_logmeanexp",
        eps: float = 1e-8,
    ):
        super().__init__(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            backbone_name=backbone_name,
            image_size=image_size,
        )
        if token_dim <= 0:
            raise ValueError("token_dim must be positive")
        if not 0.0 < rho <= 1.0:
            raise ValueError("rho must satisfy 0 < rho <= 1")
        if tau <= 0.0:
            raise ValueError("tau must be positive")
        if eps_sinkhorn <= 0.0:
            raise ValueError("eps_sinkhorn must be positive")
        if fgw_iters < 0 or sinkhorn_iters <= 0:
            raise ValueError("fgw_iters must be non-negative and sinkhorn_iters must be positive")
        if rival_temperature <= 0.0 or coherent_temperature <= 0.0:
            raise ValueError("rival_temperature and coherent_temperature must be positive")
        if not 0.0 <= spatial_structure_weight <= 1.0:
            raise ValueError("spatial_structure_weight must be in [0, 1]")
        if mass_weight < 0.0 or cost_weight <= 0.0:
            raise ValueError("mass_weight must be non-negative and cost_weight must be positive")

        ablation_mode = str(ablation_mode).lower()
        if ablation_mode not in ABLATION_MODES:
            raise ValueError(f"ablation_mode must be one of {sorted(ABLATION_MODES)}, got {ablation_mode}")
        shot_aggregation = str(shot_aggregation).lower()
        if shot_aggregation not in SHOT_AGGREGATIONS:
            raise ValueError(
                f"shot_aggregation must be one of {sorted(SHOT_AGGREGATIONS)}, got {shot_aggregation}"
            )

        self.token_dim = int(token_dim)
        self.rho = float(rho)
        self.tau = float(tau)
        self.eps_sinkhorn = float(eps_sinkhorn)
        self.fgw_iters = int(fgw_iters)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.sinkhorn_tol = float(sinkhorn_tol)
        self.rival_temperature = float(rival_temperature)
        self.rival_margin = float(rival_margin)
        self.coherent_temperature = float(coherent_temperature)
        self.spatial_structure_weight = float(spatial_structure_weight)
        self.mass_weight = float(mass_weight)
        self.cost_weight = float(cost_weight)
        self.normalize_tokens = bool(normalize_tokens)
        self.structure_detach = bool(structure_detach)
        self.rival_detach = bool(rival_detach)
        self.ablation_mode = ablation_mode
        self.shot_aggregation = shot_aggregation
        self.eps = float(eps)

        if self.token_dim != hidden_dim:
            self.token_projector: nn.Module = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, self.token_dim, bias=False),
            )
        else:
            self.token_projector = nn.Identity()

        alpha_init = min(max(float(alpha_init), self.eps), 1.0 - self.eps)
        self.raw_alpha = nn.Parameter(torch.tensor(math.log(alpha_init / (1.0 - alpha_init))))
        self.raw_score_scale = nn.Parameter(
            torch.tensor(math.log(math.expm1(max(float(score_scale_init), self.eps))))
        )
        self.raw_threshold = nn.Parameter(
            torch.tensor(math.log(math.expm1(max(float(threshold_init), self.eps))))
        )

    @property
    def alpha(self) -> torch.Tensor:
        return torch.sigmoid(self.raw_alpha)

    @property
    def score_scale(self) -> torch.Tensor:
        return F.softplus(self.raw_score_scale)

    @property
    def threshold(self) -> torch.Tensor:
        return F.softplus(self.raw_threshold)

    @property
    def uses_structure(self) -> bool:
        return self.ablation_mode == "with_structure"

    @property
    def uses_evidence_gate(self) -> bool:
        return self.ablation_mode not in {"no_coherent_score", "feature_only"}

    @property
    def uses_coherent_score(self) -> bool:
        return self.ablation_mode not in {"no_coherent_score", "feature_only"}

    def _encode_tokens(self, images: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        feat = self.encode(images)
        if feat.dim() != 4:
            raise ValueError(f"encoded feature map must be 4D, got {tuple(feat.shape)}")
        spatial_hw = (int(feat.shape[-2]), int(feat.shape[-1]))
        tokens = feat.flatten(2).transpose(1, 2).contiguous()
        tokens = self.token_projector(tokens)
        if self.normalize_tokens:
            tokens = F.normalize(tokens, dim=-1)
        return tokens, spatial_hw

    def _spatial_structure(
        self,
        token_count: int,
        spatial_hw: tuple[int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        height, width = spatial_hw
        if height * width != token_count:
            raise ValueError(
                f"spatial grid {spatial_hw} does not match token count {token_count}"
            )
        ys = torch.linspace(0.0, 1.0, height, device=device, dtype=dtype)
        xs = torch.linspace(0.0, 1.0, width, device=device, dtype=dtype)
        yy = ys.view(height, 1).expand(height, width)
        xx = xs.view(1, width).expand(height, width)
        coords = torch.stack((yy, xx), dim=-1).reshape(token_count, 2)
        return normalize_intra_dist(pairwise_sq_l2(coords.unsqueeze(0), coords.unsqueeze(0)), self.eps).squeeze(0)

    def _structure_matrix(
        self,
        tokens: torch.Tensor,
        spatial_hw: tuple[int, int],
    ) -> torch.Tensor:
        batch_size, token_count, _ = tokens.shape
        if not self.uses_structure:
            return tokens.new_zeros(batch_size, token_count, token_count)

        D_feat = normalize_intra_dist(pairwise_sq_l2(tokens, tokens), self.eps)
        if self.spatial_structure_weight > 0.0:
            D_spatial = self._spatial_structure(
                token_count,
                spatial_hw,
                tokens.device,
                tokens.dtype,
            ).unsqueeze(0)
            D = (1.0 - self.spatial_structure_weight) * D_feat + self.spatial_structure_weight * D_spatial
            D = normalize_intra_dist(D, self.eps)
        else:
            D = D_feat
        if self.structure_detach:
            D = D.detach()
        return D

    def _solve_pair_batch(
        self,
        query_tokens: torch.Tensor,
        support_tokens: torch.Tensor,
        query_spatial_hw: tuple[int, int],
        support_spatial_hw: tuple[int, int],
    ) -> Dict[str, torch.Tensor]:
        pair_count = query_tokens.shape[0]
        token_q = query_tokens.shape[1]
        token_s = support_tokens.shape[1]

        C_feat = pairwise_sq_l2(query_tokens, support_tokens)
        D_q = self._structure_matrix(query_tokens, query_spatial_hw)
        D_s = self._structure_matrix(support_tokens, support_spatial_hw)

        rho = query_tokens.new_full((pair_count,), self.rho).clamp(self.eps, 1.0)
        log_rho = rho.log()
        log_a = (log_rho - math.log(token_q)).unsqueeze(-1).expand(-1, token_q)
        log_b = (log_rho - math.log(token_s)).unsqueeze(-1).expand(-1, token_s)
        alpha = self.alpha if self.uses_structure else C_feat.new_zeros(())
        fgw_iters = max(1, self.fgw_iters) if self.uses_structure else 1

        plan, C_final = fgw_uot_solve(
            C_feat,
            D_q,
            D_s,
            log_a,
            log_b,
            alpha=alpha,
            tau=self.tau,
            eps=self.eps_sinkhorn,
            fgw_iters=fgw_iters,
            sinkhorn_iters=self.sinkhorn_iters,
            tol=self.sinkhorn_tol,
        )

        transport_cost = (plan * C_final).sum(dim=(-1, -2))
        transported_mass = plan.sum(dim=(-1, -2))
        return {
            "C_feat": C_feat,
            "D_q": D_q,
            "D_s": D_s,
            "rho": rho,
            "transport_plan": plan,
            "C_final": C_final,
            "transport_cost": transport_cost,
            "transported_mass": transported_mass,
        }

    def _class_contrastive_gate(
        self,
        plan: torch.Tensor,
        C_final: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_query, way_num, shot_num, token_q, _ = plan.shape
        del shot_num

        affinity = torch.exp((-C_final / self.coherent_temperature).clamp_min(-50.0))
        token_evidence = (plan * affinity).sum(dim=-1)
        class_evidence = token_evidence.mean(dim=2)
        gate_source = class_evidence.detach() if self.rival_detach else class_evidence

        if way_num <= 1 or self.ablation_mode == "no_rival":
            rival_evidence = torch.zeros_like(gate_source)
        else:
            rivals = gate_source.unsqueeze(1).expand(num_query, way_num, way_num, token_q)
            eye = torch.eye(way_num, device=plan.device, dtype=torch.bool).view(1, way_num, way_num, 1)
            rival_evidence = rivals.masked_fill(eye, float("-inf")).amax(dim=2)
            rival_evidence = rival_evidence.clamp_min(0.0)

        if self.uses_evidence_gate:
            gate_logits = (gate_source - rival_evidence - self.rival_margin) / self.rival_temperature
            query_token_gate = torch.sigmoid(gate_logits)
        else:
            query_token_gate = torch.ones_like(gate_source)

        gate = query_token_gate.unsqueeze(2).unsqueeze(-1)
        return gate, class_evidence, rival_evidence

    def _score_pairs(
        self,
        pair_out: Dict[str, torch.Tensor],
        pair_shape: tuple[int, int, int],
        token_q: int,
        token_s: int,
    ) -> Dict[str, torch.Tensor]:
        plan = pair_out["transport_plan"].reshape(*pair_shape, token_q, token_s)
        C_final = pair_out["C_final"].reshape(*pair_shape, token_q, token_s)
        raw_cost = (plan * C_final).sum(dim=(-1, -2))
        raw_mass = plan.sum(dim=(-1, -2))

        gate, class_evidence, rival_evidence = self._class_contrastive_gate(plan, C_final)
        coherent_plan = plan * gate
        coherent_cost = (coherent_plan * C_final).sum(dim=(-1, -2))
        coherent_mass = coherent_plan.sum(dim=(-1, -2))
        mean_gate = gate.squeeze(-1).expand(-1, -1, pair_shape[2], -1).mean(dim=-1)

        if self.ablation_mode == "cost_only":
            score_cost = coherent_cost if self.uses_coherent_score else raw_cost
            score_mass = torch.zeros_like(score_cost)
        elif self.uses_coherent_score:
            score_cost = coherent_cost
            score_mass = coherent_mass
        else:
            score_cost = raw_cost
            score_mass = raw_mass

        score_energy = self.cost_weight * score_cost - self.mass_weight * self.threshold * score_mass
        shot_logits = self.score_scale * (-score_energy)
        return {
            "shot_logits": shot_logits,
            "score_energy": score_energy,
            "raw_cost": raw_cost,
            "raw_mass": raw_mass,
            "coherent_cost": coherent_cost,
            "coherent_mass": coherent_mass,
            "query_token_gate": gate.squeeze(-1).squeeze(2),
            "class_evidence": class_evidence,
            "rival_evidence": rival_evidence,
            "mean_gate": mean_gate,
        }

    def _pool_shot_scores(
        self,
        shot_logits: torch.Tensor,
        metrics: Dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        if shot_logits.shape[-1] == 1:
            weights = torch.ones_like(shot_logits)
            pooled = {
                key: value.squeeze(-1)
                for key, value in metrics.items()
                if value.dim() == shot_logits.dim()
            }
            return shot_logits.squeeze(-1), pooled, weights

        if self.shot_aggregation == "j_logmeanexp":
            weights = torch.softmax(shot_logits, dim=-1)
            logits = torch.logsumexp(shot_logits, dim=-1) - math.log(float(shot_logits.shape[-1]))
        else:
            weights = torch.full_like(shot_logits, 1.0 / float(shot_logits.shape[-1]))
            logits = (weights * shot_logits).sum(dim=-1)

        pooled = {
            key: (weights * value).sum(dim=-1)
            for key, value in metrics.items()
            if value.dim() == shot_logits.dim()
        }
        return logits, pooled, weights

    def _forward_episode(
        self,
        query: torch.Tensor,
        support: torch.Tensor,
        query_targets: Optional[torch.Tensor] = None,
        support_targets: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ) -> Dict[str, Any]:
        del query_targets, support_targets
        if query.dim() != 4:
            raise ValueError(f"query must have shape (NumQuery, C, H, W), got {tuple(query.shape)}")
        if support.dim() != 5:
            raise ValueError(f"support must have shape (Way, Shot, C, H, W), got {tuple(support.shape)}")

        way_num = support.shape[0]
        shot_num = support.shape[1]
        num_query = query.shape[0]

        query_tokens, query_spatial_hw = self._encode_tokens(query)
        support_tokens, support_spatial_hw = self._encode_tokens(
            support.reshape(way_num * shot_num, *support.shape[2:])
        )

        token_q = query_tokens.shape[1]
        token_s = support_tokens.shape[1]
        token_dim = query_tokens.shape[2]

        support_shots = support_tokens.reshape(way_num, shot_num, token_s, token_dim)
        pair_count = num_query * way_num * shot_num
        query_batch = (
            query_tokens.unsqueeze(1)
            .unsqueeze(2)
            .expand(-1, way_num, shot_num, -1, -1)
            .reshape(pair_count, token_q, token_dim)
        )
        support_batch = (
            support_shots.unsqueeze(0)
            .expand(num_query, -1, -1, -1, -1)
            .reshape(pair_count, token_s, token_dim)
        )

        pair_out = self._solve_pair_batch(query_batch, support_batch, query_spatial_hw, support_spatial_hw)
        pair_shape = (num_query, way_num, shot_num)
        score_out = self._score_pairs(pair_out, pair_shape, token_q, token_s)

        shot_transport_cost = pair_out["transport_cost"].reshape(pair_shape)
        shot_transported_mass = pair_out["transported_mass"].reshape(pair_shape)
        shot_rho = pair_out["rho"].reshape(pair_shape)
        shot_logits = score_out["shot_logits"]
        pool_metrics = {
            "score_energy": score_out["score_energy"],
            "transport_cost": shot_transport_cost,
            "transported_mass": shot_transported_mass,
            "rho": shot_rho,
            "coherent_cost": score_out["coherent_cost"],
            "coherent_mass": score_out["coherent_mass"],
            "raw_cost": score_out["raw_cost"],
            "raw_mass": score_out["raw_mass"],
            "mean_gate": score_out["mean_gate"],
        }
        logits, pooled, shot_weights = self._pool_shot_scores(shot_logits, pool_metrics)
        logits = logits.reshape(num_query, way_num)
        aux_loss = logits.sum() * 0.0

        result: Dict[str, Any] = {
            "logits": logits,
            "aux_loss": aux_loss,
            "rho": pooled["rho"].detach().reshape(num_query, way_num),
            "transported_mass": pooled["transported_mass"].detach().reshape(num_query, way_num),
            "transport_cost": pooled["transport_cost"].detach().reshape(num_query, way_num),
            "query_class_distance": pooled["score_energy"].detach().reshape(num_query, way_num),
            "coherent_mass": pooled["coherent_mass"].detach().reshape(num_query, way_num),
            "coherent_cost": pooled["coherent_cost"].detach().reshape(num_query, way_num),
            "raw_mass": pooled["raw_mass"].detach().reshape(num_query, way_num),
            "raw_cost": pooled["raw_cost"].detach().reshape(num_query, way_num),
            "mean_gate": pooled["mean_gate"].detach().reshape(num_query, way_num),
            "alpha": self.alpha.detach() if self.uses_structure else logits.new_zeros(()),
            "score_scale": self.score_scale.detach(),
            "threshold": self.threshold.detach(),
            "shot_distance": score_out["score_energy"].detach(),
            "shot_aggregation_weights": shot_weights.detach(),
            "shot_logits": shot_logits.detach(),
            "shot_pool_weights": shot_weights.detach(),
            "shot_transport_cost": shot_transport_cost.detach(),
            "shot_transported_mass": shot_transported_mass.detach(),
            "shot_coherent_cost": score_out["coherent_cost"].detach(),
            "shot_coherent_mass": score_out["coherent_mass"].detach(),
            "mean_shot_distance": score_out["score_energy"].detach().mean(),
        }
        if return_aux:
            result["rho_shot"] = shot_rho.detach()
            result["transport_plan"] = pair_out["transport_plan"].detach().reshape(
                num_query,
                way_num,
                shot_num,
                token_q,
                token_s,
            )
            result["C_feat"] = pair_out["C_feat"].detach().reshape(num_query, way_num, shot_num, token_q, token_s)
            result["C_final"] = pair_out["C_final"].detach().reshape(num_query, way_num, shot_num, token_q, token_s)
            result["D_q"] = pair_out["D_q"].detach().reshape(num_query, way_num, shot_num, token_q, token_q)
            result["D_s"] = pair_out["D_s"].detach().reshape(num_query, way_num, shot_num, token_s, token_s)
            result["query_token_gate"] = score_out["query_token_gate"].detach()
            result["class_evidence"] = score_out["class_evidence"].detach()
            result["rival_evidence"] = score_out["rival_evidence"].detach()
        return result

    @staticmethod
    def _slice_query_targets(
        query_targets: Optional[torch.Tensor],
        batch_idx: int,
        num_query_per_ep: int,
        batch_size: int,
    ) -> Optional[torch.Tensor]:
        if query_targets is None:
            return None
        if query_targets.dim() == 2:
            if query_targets.shape[0] != batch_size:
                raise ValueError(
                    f"query_targets batch mismatch: expected {batch_size}, got {query_targets.shape[0]}"
                )
            return query_targets[batch_idx]
        if query_targets.dim() == 1:
            start = batch_idx * num_query_per_ep
            return query_targets[start : start + num_query_per_ep]
        raise ValueError(f"query_targets must be 1D or 2D, got {tuple(query_targets.shape)}")

    @staticmethod
    def _slice_support_targets(
        support_targets: Optional[torch.Tensor],
        batch_idx: int,
        batch_size: int,
    ) -> Optional[torch.Tensor]:
        if support_targets is None:
            return None
        if support_targets.dim() == 3:
            if support_targets.shape[0] != batch_size:
                raise ValueError(
                    f"support_targets batch mismatch: expected {batch_size}, got {support_targets.shape[0]}"
                )
            return support_targets[batch_idx]
        if batch_size == 1 and support_targets.dim() == 2:
            return support_targets
        raise ValueError(f"support_targets must be 2D or 3D, got {tuple(support_targets.shape)}")

    @staticmethod
    def _merge_episode_outputs(
        outputs: list[Dict[str, Any]],
        logits: torch.Tensor,
        aux_loss: torch.Tensor,
        return_aux: bool,
    ) -> CFUGETResult:
        merged: Dict[str, Any] = {"logits": logits, "aux_loss": aux_loss}
        if return_aux:
            for key in (
                "rho",
                "transported_mass",
                "transport_cost",
                "query_class_distance",
                "coherent_mass",
                "coherent_cost",
                "raw_mass",
                "raw_cost",
                "mean_gate",
                "shot_distance",
                "shot_aggregation_weights",
                "shot_logits",
                "shot_pool_weights",
                "shot_transport_cost",
                "shot_transported_mass",
                "shot_coherent_cost",
                "shot_coherent_mass",
                "rho_shot",
                "transport_plan",
                "C_feat",
                "C_final",
                "D_q",
                "D_s",
                "query_token_gate",
                "class_evidence",
                "rival_evidence",
            ):
                if key in outputs[0]:
                    merged[key] = torch.cat([item[key] for item in outputs], dim=0)
            merged["alpha"] = outputs[-1]["alpha"]
            merged["score_scale"] = outputs[-1]["score_scale"]
            merged["threshold"] = outputs[-1]["threshold"]
            if "mean_shot_distance" in outputs[0]:
                merged["mean_shot_distance"] = torch.stack(
                    [item["mean_shot_distance"] for item in outputs]
                ).mean()
        return CFUGETResult(merged)

    def forward(
        self,
        query: torch.Tensor,
        support: torch.Tensor,
        query_targets: Optional[torch.Tensor] = None,
        support_targets: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ):
        needs_payload = bool(return_aux or self.training)

        if support.dim() == 6:
            batch_size = support.shape[0]
            way_num = support.shape[1]

            if query.dim() == 5:
                if query.shape[0] != batch_size:
                    raise ValueError(
                        f"query/support batch mismatch: query={tuple(query.shape)} support={tuple(support.shape)}"
                    )
                num_query_per_ep = query.shape[1]

                def get_query(batch_idx: int) -> torch.Tensor:
                    return query[batch_idx]

            elif query.dim() == 4:
                if query.shape[0] % batch_size != 0:
                    raise ValueError(
                        f"flattened query count {query.shape[0]} must be divisible by batch size {batch_size}"
                    )
                num_query_per_ep = query.shape[0] // batch_size

                def get_query(batch_idx: int) -> torch.Tensor:
                    start = batch_idx * num_query_per_ep
                    return query[start : start + num_query_per_ep]

            else:
                raise ValueError(f"query must be 4D or 5D for batched support, got {tuple(query.shape)}")

            outputs: list[Dict[str, Any]] = []
            logits_per_episode: list[torch.Tensor] = []
            aux_losses: list[torch.Tensor] = []
            for batch_idx in range(batch_size):
                out_b = self._forward_episode(
                    get_query(batch_idx),
                    support[batch_idx],
                    query_targets=self._slice_query_targets(
                        query_targets,
                        batch_idx,
                        num_query_per_ep,
                        batch_size,
                    ),
                    support_targets=self._slice_support_targets(support_targets, batch_idx, batch_size),
                    return_aux=return_aux,
                )
                outputs.append(out_b)
                logits_per_episode.append(out_b["logits"])
                aux_losses.append(out_b["aux_loss"])

            logits = torch.cat(logits_per_episode, dim=0).reshape(-1, way_num)
            aux_loss = torch.stack(aux_losses).mean()
            if not needs_payload:
                return logits
            return self._merge_episode_outputs(outputs, logits, aux_loss, return_aux=return_aux)

        if support.dim() != 5:
            raise ValueError(f"support must be 5D or 6D, got {tuple(support.shape)}")
        if query.dim() != 4:
            raise ValueError(f"query must be 4D for single-episode support, got {tuple(query.shape)}")

        out = self._forward_episode(
            query,
            support,
            query_targets=query_targets,
            support_targets=support_targets,
            return_aux=return_aux,
        )
        logits = out["logits"].reshape(-1, support.shape[0])
        if not needs_payload:
            return logits
        if return_aux:
            out["logits"] = logits
            return CFUGETResult(out)
        return CFUGETResult({"logits": logits, "aux_loss": out["aux_loss"]})
