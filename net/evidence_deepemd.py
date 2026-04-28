"""Evidence-DeepEMD: DeepEMD plus local evidence and shot reliability."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linprog

from net.encoders.smnet_conv64f_encoder import build_resnet12_family_encoder
try:
    import cv2
except ImportError:  # pragma: no cover - optional dependency
    cv2 = None

try:
    from qpth.qp import QPFunction
except ImportError:  # pragma: no cover - optional dependency
    QPFunction = None


def _normalize_transport_weight(weight: torch.Tensor) -> torch.Tensor:
    weight = F.relu(weight) + 1e-5
    return (weight * weight.shape[-1]) / weight.sum(dim=-1, keepdim=True)


def sinkhorn_distance(
    cost: torch.Tensor,
    weight1: torch.Tensor,
    weight2: torch.Tensor,
    n_iters: int = 20,
    reg: float = 0.05,
):
    """Batched Sinkhorn fallback when neither OpenCV nor QPTH is usable."""

    batch_size, num_q, _ = cost.size()
    kernel = torch.exp(-cost / reg)

    weight1 = _normalize_transport_weight(weight1)
    weight2 = _normalize_transport_weight(weight2)

    u = torch.ones_like(weight1)
    v = torch.ones_like(weight2)
    for _ in range(n_iters):
        v = weight2 / (torch.bmm(kernel.transpose(1, 2), u.unsqueeze(-1)).squeeze(-1) + 1e-8)
        u = weight1 / (torch.bmm(kernel, v.unsqueeze(-1)).squeeze(-1) + 1e-8)

    return u.unsqueeze(-1) * kernel * v.unsqueeze(1)


BALANCED_SOLVERS = {"opencv", "qpth", "linprog", "sinkhorn"}


def emd_inference_qpth(distance_matrix: torch.Tensor, weight1: torch.Tensor, weight2: torch.Tensor, form: str = "L2", l2_strength: float = 1e-6):
    """Device-safe port of the official qpth EMD solver."""

    if QPFunction is None:
        raise RuntimeError("DeepEMD solver='qpth' requires qpth to be installed.")

    weight1 = _normalize_transport_weight(weight1)
    weight2 = _normalize_transport_weight(weight2)

    nbatch = distance_matrix.shape[0]
    num_src = weight1.shape[1]
    num_dst = weight2.shape[1]
    num_var = distance_matrix.shape[1] * distance_matrix.shape[2]
    device = distance_matrix.device
    dtype = torch.double

    q_1 = distance_matrix.reshape(-1, 1, num_var).to(dtype=dtype)
    if form == "QP":
        eye = torch.eye(num_var, device=device, dtype=dtype).unsqueeze(0).expand(nbatch, -1, -1)
        q = torch.bmm(q_1.transpose(2, 1), q_1) + 1e-4 * eye
        p = torch.zeros(nbatch, num_var, device=device, dtype=dtype)
    elif form == "L2":
        q = (l2_strength * torch.eye(num_var, device=device, dtype=dtype)).unsqueeze(0).expand(nbatch, -1, -1)
        p = distance_matrix.reshape(nbatch, num_var).to(dtype=dtype)
    else:
        raise ValueError(f"Unknown qpth form: {form}")

    h_1 = torch.zeros(nbatch, num_var, device=device, dtype=dtype)
    h_2 = torch.cat([weight1, weight2], dim=1).to(dtype=dtype)
    h = torch.cat((h_1, h_2), dim=1)

    g_1 = -torch.eye(num_var, device=device, dtype=dtype).unsqueeze(0).expand(nbatch, -1, -1)
    g_2 = torch.zeros(nbatch, num_src + num_dst, num_var, device=device, dtype=dtype)
    for src_idx in range(num_src):
        g_2[:, src_idx, num_dst * src_idx : num_dst * (src_idx + 1)] = 1
    for dst_idx in range(num_dst):
        g_2[:, num_src + dst_idx, dst_idx::num_dst] = 1
    g = torch.cat((g_1, g_2), dim=1)

    a = torch.ones(nbatch, 1, num_var, device=device, dtype=dtype)
    b = torch.min(weight1.sum(dim=1), weight2.sum(dim=1)).unsqueeze(1).to(dtype=dtype)
    flow = QPFunction(verbose=-1)(q, p, g, h, a, b)
    emd_score = torch.sum((1 - q_1).squeeze(1) * flow, dim=1)
    return emd_score, flow.view(-1, num_src, num_dst)


@lru_cache(maxsize=16)
def _linprog_ub_constraints(num_src: int, num_dst: int):
    num_var = num_src * num_dst
    num_ub = num_src + num_dst
    a_ub = np.zeros((num_ub, num_var), dtype=np.float64)

    for src_idx in range(num_src):
        start = src_idx * num_dst
        a_ub[src_idx, start : start + num_dst] = 1.0

    for dst_idx in range(num_dst):
        a_ub[num_src + dst_idx, dst_idx::num_dst] = 1.0

    a_eq = np.ones((1, num_var), dtype=np.float64)
    return a_ub, a_eq


def emd_inference_linprog(cost_matrix: torch.Tensor, weight1: torch.Tensor, weight2: torch.Tensor) -> torch.Tensor:
    """Exact transportation plan via SciPy LP, used only as a fallback."""

    cost_np = cost_matrix.detach().cpu().double().numpy()
    weight1_np = _normalize_transport_weight(weight1.unsqueeze(0)).squeeze(0).detach().cpu().double().numpy()
    weight2_np = _normalize_transport_weight(weight2.unsqueeze(0)).squeeze(0).detach().cpu().double().numpy()

    num_src, num_dst = cost_np.shape
    a_ub, a_eq = _linprog_ub_constraints(num_src, num_dst)
    b_ub = np.concatenate([weight1_np, weight2_np], axis=0)
    b_eq = np.array([min(weight1_np.sum(), weight2_np.sum())], dtype=np.float64)
    result = linprog(
        c=cost_np.reshape(-1),
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=(0, None),
        method="highs",
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"linprog failed: {result.message}")

    return torch.from_numpy(result.x.reshape(num_src, num_dst))


def emd_inference_opencv(cost_matrix: torch.Tensor, weight1: torch.Tensor, weight2: torch.Tensor):
    """Official OpenCV solver path used by the reference DeepEMD repo."""

    if cv2 is None:
        raise RuntimeError("DeepEMD solver='opencv' requires opencv-python to be installed.")

    cost_np = cost_matrix.detach().cpu().numpy()
    weight1_np = _normalize_transport_weight(weight1.unsqueeze(0)).squeeze(0).view(-1, 1).detach().cpu().numpy()
    weight2_np = _normalize_transport_weight(weight2.unsqueeze(0)).squeeze(0).view(-1, 1).detach().cpu().numpy()
    _, _, flow = cv2.EMD(weight1_np, weight2_np, cv2.DIST_USER, cost_np)
    return flow


class EvidenceMLP(nn.Module):
    """Lightweight class-conditioned token reliability head."""

    def __init__(self, feat_dim: int):
        super().__init__()
        hidden_dim = max(feat_dim // 4, 64)
        self.net = nn.Sequential(
            nn.Linear(feat_dim * 4, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, tokens: torch.Tensor, class_proto: torch.Tensor) -> torch.Tensor:
        proto = class_proto.unsqueeze(-2)
        tokens, proto = torch.broadcast_tensors(tokens, proto)
        evidence_input = torch.cat(
            [tokens, proto, tokens * proto, torch.abs(tokens - proto)],
            dim=-1,
        )
        return torch.sigmoid(self.net(evidence_input).squeeze(-1))


class EvidenceDeepEMD(nn.Module):
    """DeepEMD with optional evidence-weighted masses and K-shot reliability."""

    def __init__(
        self,
        image_size: int = 64,
        temperature: float = 12.5,
        metric: str = "cosine",
        norm: str = "center",
        solver: str = "sinkhorn",
        qpth_form: str = "L2",
        qpth_l2_strength: float = 1e-6,
        sinkhorn_reg: float = 0.05,
        sinkhorn_iterations: int = 20,
        sinkhorn_tolerance: float = 1e-6,
        uot_tau_q: float = 0.5,
        uot_tau_c: float = 0.5,
        uot_score_normalize: bool = False,
        partial_mass_fraction: float = 0.5,
        partial_transport_mass: float | None = None,
        partial_score_normalize: bool = True,
        partial_backend: str = "native",
        partial_exact: bool = False,
        eps: float = 1e-8,
        sfc_lr: float = 0.1,
        sfc_update_step: int = 15,
        sfc_bs: int = 4,
        fewshot_backbone: str = "resnet12",
        use_evidence_weight: bool = True,
        use_shot_reliability: bool = True,
        evidence_eps: float = 1e-3,
        consensus_scale: float = 5.0,
        shot_temperature: float = 1.0,
        debug_evidence: bool = False,
        device: str = "cuda",
    ):
        super().__init__()
        self.encoder = build_resnet12_family_encoder(
            image_size=image_size,
            backbone_name=fewshot_backbone,
            pool_output=False,
            variant="fewshot",
            drop_rate=0.0,
        )
        self.feat_dim = int(self.encoder.out_channels)
        self.temperature = float(temperature)
        self.metric = str(metric)
        self.norm = str(norm)
        self.solver = str(solver).lower()
        if self.solver not in BALANCED_SOLVERS:
            raise ValueError("EvidenceDeepEMD supports only balanced DeepEMD solvers: opencv, qpth, linprog, sinkhorn")
        self.qpth_form = str(qpth_form)
        self.qpth_l2_strength = float(qpth_l2_strength)
        if sinkhorn_reg <= 0.0:
            raise ValueError("sinkhorn_reg must be positive")
        if sinkhorn_iterations <= 0:
            raise ValueError("sinkhorn_iterations must be positive")
        if sinkhorn_tolerance < 0.0:
            raise ValueError("sinkhorn_tolerance must be non-negative")
        if uot_tau_q <= 0.0 or uot_tau_c <= 0.0:
            raise ValueError("uot_tau_q and uot_tau_c must be positive")
        if partial_mass_fraction <= 0.0 or partial_mass_fraction > 1.0:
            raise ValueError("partial_mass_fraction must be in (0, 1]")
        if partial_transport_mass is not None and partial_transport_mass <= 0.0:
            raise ValueError("partial_transport_mass must be positive when provided")
        partial_backend = str(partial_backend).lower()
        if partial_backend not in {"native", "pot"}:
            raise ValueError("partial_backend must be 'native' or 'pot'")

        self.sinkhorn_reg = float(sinkhorn_reg)
        self.sinkhorn_iterations = int(sinkhorn_iterations)
        self.sinkhorn_tolerance = float(sinkhorn_tolerance)
        self.uot_tau_q = float(uot_tau_q)
        self.uot_tau_c = float(uot_tau_c)
        self.uot_score_normalize = bool(uot_score_normalize)
        self.partial_mass_fraction = float(partial_mass_fraction)
        self.partial_transport_mass = None if partial_transport_mass is None else float(partial_transport_mass)
        self.partial_score_normalize = bool(partial_score_normalize)
        self.partial_backend = partial_backend
        self.partial_exact = bool(partial_exact)
        self.eps = float(eps)
        self.sfc_lr = float(sfc_lr)
        self.sfc_update_step = int(sfc_update_step)
        self.sfc_bs = int(sfc_bs)
        if evidence_eps <= 0.0:
            raise ValueError("evidence_eps must be positive")
        if consensus_scale <= 0.0:
            raise ValueError("consensus_scale must be positive")
        if shot_temperature <= 0.0:
            raise ValueError("shot_temperature must be positive")
        self.use_evidence_weight = bool(use_evidence_weight)
        self.use_shot_reliability = bool(use_shot_reliability)
        self.evidence_eps = float(evidence_eps)
        self.consensus_scale = float(consensus_scale)
        self.shot_temperature = float(shot_temperature)
        self.debug_evidence = bool(debug_evidence)
        self.evidence_mlp = EvidenceMLP(self.feat_dim)
        self.last_evidence_debug: dict[str, torch.Tensor | bool] | None = None
        self.to(device)

    def _debug_check(self, name: str, tensor: torch.Tensor) -> None:
        if self.debug_evidence and not torch.isfinite(tensor).all():
            raise FloatingPointError(f"EvidenceDeepEMD {name} contains non-finite values")

    def _debug_stats(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        detached = tensor.detach()
        return detached.mean(), detached.min(), detached.max()

    def _update_debug(self, **items) -> None:
        debug = {key: value.detach() if torch.is_tensor(value) else value for key, value in items.items()}
        self.last_evidence_debug = debug
        if not self.debug_evidence:
            return
        parts = []
        if "token_reliability" in debug:
            mean_v, min_v, max_v = self._debug_stats(debug["token_reliability"])
            parts.append(
                f"token_rel(mean/min/max)={mean_v.item():.4f}/{min_v.item():.4f}/{max_v.item():.4f}"
            )
        if "query_mass_sum" in debug:
            parts.append(f"query_mass_sum={debug['query_mass_sum'].mean().item():.4f}")
        if "support_mass_sum" in debug:
            parts.append(f"support_mass_sum={debug['support_mass_sum'].mean().item():.4f}")
        if "shot_alpha_sum" in debug:
            parts.append(f"shot_alpha_sum={debug['shot_alpha_sum'].mean().item():.4f}")
        if "logits" in debug:
            parts.append(f"logits_has_nan={torch.isnan(debug['logits']).any().item()}")
        if parts:
            print("[EvidenceDeepEMD] " + ", ".join(parts))

    def normalize_feature(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm == "center":
            return x - x.mean(dim=1, keepdim=True)
        return x

    def get_weight_vector(self, feat_a: torch.Tensor, feat_b: torch.Tensor) -> torch.Tensor:
        pooled_b = F.adaptive_avg_pool2d(feat_b, [1, 1])
        combination = (feat_a.unsqueeze(1) * pooled_b.unsqueeze(0)).sum(dim=2)
        return F.relu(combination.reshape(feat_a.shape[0], feat_b.shape[0], -1)) + 1e-3

    def get_similarity_map(self, proto: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        num_proto = proto.shape[0]
        num_query = query.shape[0]
        query = query.reshape(num_query, query.shape[1], -1).transpose(1, 2)
        proto = proto.reshape(num_proto, proto.shape[1], -1).transpose(1, 2)

        proto = proto.unsqueeze(0)
        query = query.unsqueeze(1)
        if self.metric == "cosine":
            return F.cosine_similarity(proto.unsqueeze(2), query.unsqueeze(3), dim=-1)
        if self.metric == "l2":
            return 1 - (proto.unsqueeze(2) - query.unsqueeze(3)).pow(2).sum(dim=-1)
        raise ValueError(f"Unsupported metric: {self.metric}")

    def _local_tokens(self, feature_map: torch.Tensor) -> torch.Tensor:
        return feature_map.flatten(-2).transpose(-1, -2).contiguous()

    def _class_prototypes_from_support(self, support: torch.Tensor) -> torch.Tensor:
        return support.mean(dim=(1, 3, 4))

    def _class_prototypes_from_proto(self, proto: torch.Tensor) -> torch.Tensor:
        return self._local_tokens(proto).mean(dim=1)

    def _token_reliability(self, tokens: torch.Tensor, class_proto: torch.Tensor) -> torch.Tensor:
        proto = class_proto.unsqueeze(-2)
        tokens, proto = torch.broadcast_tensors(tokens, proto)
        mlp_rel = self.evidence_mlp(tokens, class_proto)
        consensus = torch.sigmoid(
            self.consensus_scale * F.cosine_similarity(tokens, proto, dim=-1, eps=self.eps)
        )
        reliability = (mlp_rel * consensus).clamp_min(self.eps)
        self._debug_check("token reliability", reliability)
        return reliability

    def _normalize_probability_mass(self, mass: torch.Tensor) -> torch.Tensor:
        mass = mass.clamp_min(self.evidence_eps)
        return mass / mass.sum(dim=-1, keepdim=True).clamp_min(self.eps)

    def _evidence_mass(self, weight: torch.Tensor, reliability: torch.Tensor) -> torch.Tensor:
        mass = weight.clamp_min(self.eps) * (self.evidence_eps + reliability)
        mass = self._normalize_probability_mass(mass)
        self._debug_check("EMD mass", mass)
        return mass

    def _query_reliability(self, query: torch.Tensor, class_proto: torch.Tensor) -> torch.Tensor:
        query_tokens = self._local_tokens(query)
        return self._token_reliability(query_tokens.unsqueeze(1), class_proto.unsqueeze(0))

    def _proto_reliability(self, proto: torch.Tensor, class_proto: torch.Tensor) -> torch.Tensor:
        proto_tokens = self._local_tokens(proto)
        return self._token_reliability(proto_tokens, class_proto)

    def _support_shot_reliability(
        self,
        support: torch.Tensor,
        class_proto: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        support_tokens = self._local_tokens(support)
        reliability = self._token_reliability(support_tokens, class_proto.unsqueeze(1))
        return reliability, support_tokens

    def _resolve_solver(self, exact: bool | None = None) -> str:
        if exact is True:
            solver = "opencv" if cv2 is not None else "linprog"
        elif exact is False:
            solver = self.solver
        elif self.training:
            solver = self.solver
        else:
            solver = "opencv" if cv2 is not None else "linprog"
        if solver not in BALANCED_SOLVERS:
            raise ValueError(f"EvidenceDeepEMD supports only balanced solvers, got: {solver}")
        return solver

    def _aggregate_similarity_score(
        self,
        similarity: torch.Tensor,
        flow: torch.Tensor,
        num_node: int,
        *,
        normalize_by_mass: bool,
    ) -> torch.Tensor:
        raw_score = (flow * similarity).sum(dim=(-1, -2))
        if normalize_by_mass:
            mass = flow.sum(dim=(-1, -2)).clamp_min(self.eps)
            return raw_score * (self.temperature / mass)
        return raw_score * (self.temperature / float(num_node))

    def get_emd_distance(self, similarity_map: torch.Tensor, weight1: torch.Tensor, weight2: torch.Tensor, solver: str) -> torch.Tensor:
        if solver not in BALANCED_SOLVERS:
            raise ValueError(f"EvidenceDeepEMD supports only balanced solvers, got: {solver}")

        num_query = similarity_map.shape[0]
        num_proto = similarity_map.shape[1]
        num_node = weight1.shape[-1]

        if solver == "opencv":
            weighted_similarity = similarity_map.clone()
            for query_idx in range(num_query):
                for class_idx in range(num_proto):
                    flow = emd_inference_opencv(
                        1 - similarity_map[query_idx, class_idx],
                        weight1[query_idx, class_idx],
                        weight2[class_idx, query_idx],
                    )
                    flow_t = torch.from_numpy(flow).to(device=similarity_map.device, dtype=similarity_map.dtype)
                    weighted_similarity[query_idx, class_idx] = weighted_similarity[query_idx, class_idx] * flow_t
            return weighted_similarity.sum(dim=-1).sum(dim=-1) * (self.temperature / num_node)

        if solver == "qpth":
            weight2 = weight2.permute(1, 0, 2)
            flat_similarity = similarity_map.reshape(num_query * num_proto, similarity_map.shape[-2], similarity_map.shape[-1])
            flat_weight1 = weight1.reshape(num_query * num_proto, weight1.shape[-1])
            flat_weight2 = weight2.reshape(num_query * num_proto, weight2.shape[-1])
            _, flows = emd_inference_qpth(
                1 - flat_similarity,
                flat_weight1,
                flat_weight2,
                form=self.qpth_form,
                l2_strength=self.qpth_l2_strength,
            )
            logits = (flows * flat_similarity).reshape(num_query, num_proto, flows.shape[-2], flows.shape[-1])
            return logits.sum(dim=-1).sum(dim=-1) * (self.temperature / num_node)

        if solver == "linprog":
            logits = []
            for query_idx in range(num_query):
                class_scores = []
                for class_idx in range(num_proto):
                    flow = emd_inference_linprog(
                        1 - similarity_map[query_idx, class_idx],
                        weight1[query_idx, class_idx],
                        weight2[class_idx, query_idx],
                    ).to(device=similarity_map.device, dtype=similarity_map.dtype)
                    class_scores.append((flow * similarity_map[query_idx, class_idx]).sum() * (self.temperature / num_node))
                logits.append(torch.stack(class_scores))
            return torch.stack(logits, dim=0)

        if solver == "sinkhorn":
            weight2 = weight2.permute(1, 0, 2)
            flat_similarity = similarity_map.reshape(num_query * num_proto, similarity_map.shape[-2], similarity_map.shape[-1])
            flat_weight1 = weight1.reshape(num_query * num_proto, weight1.shape[-1])
            flat_weight2 = weight2.reshape(num_query * num_proto, weight2.shape[-1])
            flow = sinkhorn_distance(
                1 - flat_similarity,
                flat_weight1,
                flat_weight2,
                n_iters=self.sinkhorn_iterations,
                reg=self.sinkhorn_reg,
            )
            scores = self._aggregate_similarity_score(
                flat_similarity,
                flow,
                num_node,
                normalize_by_mass=False,
            )
            return scores.reshape(num_query, num_proto)

        raise ValueError(f"Unsupported EvidenceDeepEMD solver: {solver}")

    def emd_forward(
        self,
        proto: torch.Tensor,
        query: torch.Tensor,
        solver: str,
        class_proto: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weight1 = self.get_weight_vector(query, proto)
        weight2 = self.get_weight_vector(proto, query)

        token_reliability = None
        query_mass_sum = None
        support_mass_sum = None
        if self.use_evidence_weight:
            if class_proto is None:
                class_proto = self._class_prototypes_from_proto(proto)
            query_rel = self._query_reliability(query, class_proto)
            proto_rel = self._proto_reliability(proto, class_proto)
            weight1 = self._evidence_mass(weight1, query_rel)
            weight2 = self._evidence_mass(
                weight2,
                proto_rel.unsqueeze(1).expand(-1, query.shape[0], -1),
            )
            token_reliability = torch.cat(
                [query_rel.reshape(-1), proto_rel.reshape(-1)],
                dim=0,
            )
            query_mass_sum = weight1.sum(dim=-1)
            support_mass_sum = weight2.sum(dim=-1)

        proto = self.normalize_feature(proto)
        query = self.normalize_feature(query)
        similarity_map = self.get_similarity_map(proto, query)
        logits = self.get_emd_distance(similarity_map, weight1, weight2, solver=solver)
        self._debug_check("logits", logits)
        if self.use_evidence_weight:
            self._update_debug(
                token_reliability=token_reliability,
                query_mass_sum=query_mass_sum,
                support_mass_sum=support_mass_sum,
                logits=logits,
            )
        else:
            self._update_debug(logits=logits)
        return logits

    def shot_reliability_forward(
        self,
        support: torch.Tensor,
        query: torch.Tensor,
        solver: str,
    ) -> torch.Tensor:
        way_num, shot_num = support.shape[:2]
        class_proto = self._class_prototypes_from_support(support)
        support_rel, support_tokens = self._support_shot_reliability(support, class_proto)
        proto = support.reshape(way_num * shot_num, self.feat_dim, support.shape[-2], support.shape[-1])

        weight1 = self.get_weight_vector(query, proto)
        weight2 = self.get_weight_vector(proto, query)
        query_rel = None
        query_mass_sum = None
        support_mass_sum = None
        if self.use_evidence_weight:
            query_rel = self._query_reliability(query, class_proto)
            query_rel_by_shot = (
                query_rel.unsqueeze(2)
                .expand(-1, -1, shot_num, -1)
                .reshape(query.shape[0], way_num * shot_num, -1)
            )
            support_rel_by_shot = (
                support_rel.reshape(way_num * shot_num, -1)
                .unsqueeze(1)
                .expand(-1, query.shape[0], -1)
            )
            weight1 = self._evidence_mass(weight1, query_rel_by_shot)
            weight2 = self._evidence_mass(weight2, support_rel_by_shot)
            query_mass_sum = weight1.sum(dim=-1)
            support_mass_sum = weight2.sum(dim=-1)

        proto_norm = self.normalize_feature(proto)
        query_norm = self.normalize_feature(query)
        similarity_map = self.get_similarity_map(proto_norm, query_norm)
        shot_scores = self.get_emd_distance(similarity_map, weight1, weight2, solver=solver)
        shot_scores = shot_scores.reshape(query.shape[0], way_num, shot_num)

        if shot_num == 1:
            alpha = torch.ones_like(shot_scores)
            logits = shot_scores.squeeze(-1)
        else:
            shot_rel = support_rel.mean(dim=-1)
            shot_proto = support_tokens.mean(dim=-2)
            shot_cons = F.cosine_similarity(
                shot_proto,
                class_proto.unsqueeze(1),
                dim=-1,
                eps=self.eps,
            )
            raw_alpha = shot_scores + 0.5 * shot_rel.unsqueeze(0) + 0.5 * shot_cons.unsqueeze(0)
            alpha = F.softmax(raw_alpha / self.shot_temperature, dim=-1)
            self._debug_check("shot alpha", alpha)
            logits = (alpha * shot_scores).sum(dim=-1)

        self._debug_check("logits", logits)
        token_rel_parts = [support_rel.reshape(-1)]
        if query_rel is not None:
            token_rel_parts.append(query_rel.reshape(-1))
        self._update_debug(
            token_reliability=torch.cat(token_rel_parts, dim=0),
            query_mass_sum=query_mass_sum if query_mass_sum is not None else weight1.sum(dim=-1),
            support_mass_sum=support_mass_sum if support_mass_sum is not None else weight2.sum(dim=-1),
            shot_alpha=alpha,
            shot_alpha_sum=alpha.sum(dim=-1),
            logits=logits,
        )
        return logits

    def get_sfc(
        self,
        support: torch.Tensor,
        solver: str,
        sfc_update_step: int | None = None,
        sfc_bs: int | None = None,
    ) -> torch.Tensor:
        way_num, shot_num = support.shape[:2]
        sfc_update_step = max(1, int(self.sfc_update_step if sfc_update_step is None else sfc_update_step))
        sfc_bs = max(1, int(self.sfc_bs if sfc_bs is None else sfc_bs))
        support_flat = support.reshape(way_num * shot_num, self.feat_dim, support.shape[-2], support.shape[-1])
        sfc = support.mean(dim=1).clone().detach()
        sfc = nn.Parameter(sfc.detach(), requires_grad=True)

        optimizer = torch.optim.SGD([sfc], lr=self.sfc_lr, momentum=0.9, dampening=0.9, weight_decay=0.0)
        label_shot = torch.arange(way_num, device=support.device).repeat_interleave(shot_num)

        with torch.enable_grad():
            for _ in range(sfc_update_step):
                rand_id = torch.randperm(way_num * shot_num, device=support.device)
                for start in range(0, way_num * shot_num, sfc_bs):
                    selected_id = rand_id[start : min(start + sfc_bs, way_num * shot_num)]
                    batch_shot = support_flat[selected_id]
                    batch_label = label_shot[selected_id]
                    optimizer.zero_grad(set_to_none=True)
                    logits = self.emd_forward(sfc, batch_shot.detach(), solver=solver)
                    loss = F.cross_entropy(logits, batch_label)
                    loss.backward()
                    optimizer.step()
        return sfc.detach()

    def forward(
        self,
        query: torch.Tensor,
        support: torch.Tensor,
        refine_proto: bool | None = None,
        exact: bool | None = None,
        sfc_update_step_override: int | None = None,
        sfc_bs_override: int | None = None,
    ) -> torch.Tensor:
        batch_size, num_query, channels, height, width = query.size()
        _, way_num, shot_num, _, _, _ = support.size()
        solver = self._resolve_solver(exact=exact)
        if refine_proto is None:
            refine_proto = shot_num > 1

        query_feat = self.encoder.forward_features(query.reshape(-1, channels, height, width))
        support_feat = self.encoder.forward_features(support.reshape(-1, channels, height, width))
        feat_h, feat_w = support_feat.size(-2), support_feat.size(-1)
        query_feat = query_feat.reshape(batch_size, num_query, self.feat_dim, feat_h, feat_w)
        support_feat = support_feat.reshape(batch_size, way_num, shot_num, self.feat_dim, feat_h, feat_w)

        outputs = []
        for batch_idx in range(batch_size):
            class_proto = self._class_prototypes_from_support(support_feat[batch_idx])
            if self.use_shot_reliability:
                logits = self.shot_reliability_forward(
                    support_feat[batch_idx],
                    query_feat[batch_idx],
                    solver=solver,
                )
            else:
                proto = support_feat[batch_idx].mean(dim=1)
                if refine_proto and shot_num > 1:
                    proto = self.get_sfc(
                        support_feat[batch_idx],
                        solver=solver,
                        sfc_update_step=sfc_update_step_override,
                        sfc_bs=sfc_bs_override,
                    )
                logits = self.emd_forward(
                    proto,
                    query_feat[batch_idx],
                    solver=solver,
                    class_proto=class_proto,
                )
            outputs.append(logits)

        return torch.cat(outputs, dim=0)


EvidenceDeepEMDSimple = EvidenceDeepEMD
