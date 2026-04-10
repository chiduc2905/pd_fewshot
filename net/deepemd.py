"""DeepEMD aligned with the official repo's EMD/SFC workflow."""

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


def sinkhorn_distance(cost: torch.Tensor, weight1: torch.Tensor, weight2: torch.Tensor, n_iters: int = 20, reg: float = 0.05):
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


class DeepEMD(nn.Module):
    """DeepEMD with official-style solver selection and 5-shot SFC refinement."""

    def __init__(
        self,
        image_size: int = 64,
        temperature: float = 12.5,
        metric: str = "cosine",
        norm: str = "center",
        solver: str = "opencv",
        qpth_form: str = "L2",
        qpth_l2_strength: float = 1e-6,
        sfc_lr: float = 0.1,
        sfc_update_step: int = 100,
        sfc_bs: int = 4,
        fewshot_backbone: str = "resnet12",
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
        self.qpth_form = str(qpth_form)
        self.qpth_l2_strength = float(qpth_l2_strength)
        self.sfc_lr = float(sfc_lr)
        self.sfc_update_step = int(sfc_update_step)
        self.sfc_bs = int(sfc_bs)
        self.to(device)

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

    def _resolve_solver(self, exact: bool | None = None) -> str:
        if exact is True:
            return "opencv" if cv2 is not None else "linprog"
        if exact is False:
            if self.training:
                return self.solver
            return "opencv" if cv2 is not None else "linprog"
        if self.training:
            return self.solver
        return "opencv" if cv2 is not None else "linprog"

    def get_emd_distance(self, similarity_map: torch.Tensor, weight1: torch.Tensor, weight2: torch.Tensor, solver: str) -> torch.Tensor:
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
            flow = sinkhorn_distance(1 - flat_similarity, flat_weight1, flat_weight2)
            logits = (flow * flat_similarity).reshape(num_query, num_proto, flow.shape[-2], flow.shape[-1])
            return logits.sum(dim=-1).sum(dim=-1) * (self.temperature / num_node)

        raise ValueError(f"Unsupported DeepEMD solver: {solver}")

    def emd_forward(self, proto: torch.Tensor, query: torch.Tensor, solver: str) -> torch.Tensor:
        weight1 = self.get_weight_vector(query, proto)
        weight2 = self.get_weight_vector(proto, query)

        proto = self.normalize_feature(proto)
        query = self.normalize_feature(query)
        similarity_map = self.get_similarity_map(proto, query)
        return self.get_emd_distance(similarity_map, weight1, weight2, solver=solver)

    def get_sfc(self, support: torch.Tensor, solver: str) -> torch.Tensor:
        way_num, shot_num = support.shape[:2]
        support_flat = support.reshape(way_num * shot_num, self.feat_dim, support.shape[-2], support.shape[-1])
        sfc = support.mean(dim=1).clone().detach()
        sfc = nn.Parameter(sfc.detach(), requires_grad=True)

        optimizer = torch.optim.SGD([sfc], lr=self.sfc_lr, momentum=0.9, dampening=0.9, weight_decay=0.0)
        label_shot = torch.arange(way_num, device=support.device).repeat_interleave(shot_num)

        with torch.enable_grad():
            for _ in range(self.sfc_update_step):
                rand_id = torch.randperm(way_num * shot_num, device=support.device)
                for start in range(0, way_num * shot_num, self.sfc_bs):
                    selected_id = rand_id[start : min(start + self.sfc_bs, way_num * shot_num)]
                    batch_shot = support_flat[selected_id]
                    batch_label = label_shot[selected_id]
                    optimizer.zero_grad()
                    logits = self.emd_forward(sfc, batch_shot.detach(), solver=solver)
                    loss = F.cross_entropy(logits, batch_label)
                    loss.backward()
                    optimizer.step()
        return sfc.detach()

    def forward(self, query: torch.Tensor, support: torch.Tensor, refine_proto: bool | None = None, exact: bool | None = None) -> torch.Tensor:
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
            proto = support_feat[batch_idx].mean(dim=1)
            if refine_proto and shot_num > 1:
                proto = self.get_sfc(support_feat[batch_idx], solver=solver)
            logits = self.emd_forward(proto, query_feat[batch_idx], solver=solver)
            outputs.append(logits)

        return torch.cat(outputs, dim=0)


DeepEMDSimple = DeepEMD
