"""DICE-EMD: class-competitive evidence-aware DeepEMD transport."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from net.deepemd import (
    DeepEMD,
    emd_inference_linprog,
    emd_inference_opencv,
    emd_inference_qpth,
    sinkhorn_distance,
)


BALANCED_SOLVERS = {"opencv", "qpth", "linprog", "sinkhorn"}


class DICEEMD(DeepEMD):
    """DeepEMD variant with discriminative class-competitive transport costs.

    The encoder, feature map dimensionality, k-shot prototype handling, and
    balanced EMD/Sinkhorn interface are inherited from DeepEMD. Only the cost
    used to compute the transport plan is changed; final logits are still
    scored with the original cosine similarity map.
    """

    def __init__(
        self,
        *args: Any,
        lambda_disc: float = 0.2,
        tau_comp: float = 0.1,
        use_softmin_distance: bool = True,
        tau_softmin: float = 0.1,
        debug_transport: bool = False,
        lambda_flow: float = 0.0,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        if lambda_disc < 0.0:
            raise ValueError("lambda_disc must be non-negative")
        if tau_comp <= 0.0:
            raise ValueError("tau_comp must be positive")
        if tau_softmin <= 0.0:
            raise ValueError("tau_softmin must be positive")
        if lambda_flow < 0.0:
            raise ValueError("lambda_flow must be non-negative")
        if self.solver not in BALANCED_SOLVERS:
            raise ValueError("DICE-EMD supports only balanced DeepEMD solvers: opencv, qpth, linprog, sinkhorn")

        self.lambda_disc = float(lambda_disc)
        self.tau_comp = float(tau_comp)
        self.use_softmin_distance = bool(use_softmin_distance)
        self.tau_softmin = float(tau_softmin)
        self.debug_transport = bool(debug_transport)
        self.lambda_flow = float(lambda_flow)
        self.last_transport_debug: dict[str, torch.Tensor] | None = None

    def _resolve_solver(self, exact: bool | None = None) -> str:
        solver = super()._resolve_solver(exact=exact)
        if solver not in BALANCED_SOLVERS:
            raise ValueError(f"DICE-EMD supports only balanced solvers, got: {solver}")
        return solver

    def _debug_check(self, name: str, tensor: torch.Tensor) -> None:
        if self.debug_transport and not torch.isfinite(tensor).all():
            raise FloatingPointError(f"DICE-EMD {name} contains non-finite values")

    def _compute_class_competitive_evidence(
        self,
        cost: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if cost.ndim != 4:
            raise ValueError(f"cost must have shape [num_query, way, num_query_tokens, support_tokens], got {cost.shape}")

        num_classes = cost.shape[1]
        if self.use_softmin_distance:
            distance = -self.tau_softmin * torch.logsumexp(-cost / self.tau_softmin, dim=-1)
        else:
            distance = cost.min(dim=-1).values

        if num_classes <= 1:
            assignment = torch.ones_like(distance)
            disc = torch.ones_like(distance[:, 0])
            evidence = assignment
            return distance, assignment, disc, evidence

        assignment = F.softmax(-distance / self.tau_comp, dim=1)
        entropy = -(assignment * torch.log(assignment.clamp_min(self.eps))).sum(dim=1)
        disc = (1.0 - entropy / math.log(num_classes)).clamp(min=0.0, max=1.0)
        evidence = assignment * disc.unsqueeze(1)

        self._debug_check("class distance", distance)
        self._debug_check("class assignment", assignment)
        self._debug_check("discriminative evidence", evidence)
        return distance, assignment, disc, evidence

    def _score_with_balanced_plan(
        self,
        similarity_map: torch.Tensor,
        transport_cost: torch.Tensor,
        weight1: torch.Tensor,
        weight2: torch.Tensor,
        solver: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_query = similarity_map.shape[0]
        num_proto = similarity_map.shape[1]
        num_node = weight1.shape[-1]

        if solver == "opencv":
            logits = similarity_map.new_empty(num_query, num_proto)
            plans = torch.empty_like(similarity_map)
            for query_idx in range(num_query):
                for class_idx in range(num_proto):
                    flow = emd_inference_opencv(
                        transport_cost[query_idx, class_idx],
                        weight1[query_idx, class_idx],
                        weight2[class_idx, query_idx],
                    )
                    flow_t = torch.from_numpy(flow).to(device=similarity_map.device, dtype=similarity_map.dtype)
                    plans[query_idx, class_idx] = flow_t
                    logits[query_idx, class_idx] = (
                        flow_t * similarity_map[query_idx, class_idx]
                    ).sum() * (self.temperature / float(num_node))
            return logits, plans

        if solver == "linprog":
            logits = similarity_map.new_empty(num_query, num_proto)
            plans = torch.empty_like(similarity_map)
            for query_idx in range(num_query):
                for class_idx in range(num_proto):
                    flow = emd_inference_linprog(
                        transport_cost[query_idx, class_idx],
                        weight1[query_idx, class_idx],
                        weight2[class_idx, query_idx],
                    ).to(device=similarity_map.device, dtype=similarity_map.dtype)
                    plans[query_idx, class_idx] = flow
                    logits[query_idx, class_idx] = (
                        flow * similarity_map[query_idx, class_idx]
                    ).sum() * (self.temperature / float(num_node))
            return logits, plans

        flat_similarity = similarity_map.reshape(
            num_query * num_proto,
            similarity_map.shape[-2],
            similarity_map.shape[-1],
        )
        flat_cost = transport_cost.reshape(
            num_query * num_proto,
            transport_cost.shape[-2],
            transport_cost.shape[-1],
        )
        flat_weight1 = weight1.reshape(num_query * num_proto, weight1.shape[-1])
        flat_weight2 = weight2.permute(1, 0, 2).reshape(num_query * num_proto, weight2.shape[-1])

        if solver == "qpth":
            _, flat_plans = emd_inference_qpth(
                flat_cost,
                flat_weight1,
                flat_weight2,
                form=self.qpth_form,
                l2_strength=self.qpth_l2_strength,
            )
        elif solver == "sinkhorn":
            flat_plans = sinkhorn_distance(
                flat_cost,
                flat_weight1,
                flat_weight2,
                n_iters=self.sinkhorn_iterations,
                reg=self.sinkhorn_reg,
            )
        else:
            raise ValueError(f"Unsupported DICE-EMD solver: {solver}")

        flat_scores = self._aggregate_similarity_score(
            flat_similarity,
            flat_plans,
            num_node,
            normalize_by_mass=False,
        )
        plans = flat_plans.reshape(num_query, num_proto, flat_plans.shape[-2], flat_plans.shape[-1])
        return flat_scores.reshape(num_query, num_proto), plans

    def get_emd_distance(
        self,
        similarity_map: torch.Tensor,
        weight1: torch.Tensor,
        weight2: torch.Tensor,
        solver: str,
    ) -> torch.Tensor:
        if solver not in BALANCED_SOLVERS:
            raise ValueError(f"DICE-EMD supports only balanced solvers, got: {solver}")

        cost = 1.0 - similarity_map
        distance, assignment, disc, evidence = self._compute_class_competitive_evidence(cost)
        transport_cost = cost + self.lambda_disc * (1.0 - evidence.unsqueeze(-1))

        self._debug_check("transport cost", transport_cost)
        logits, plans = self._score_with_balanced_plan(
            similarity_map=similarity_map,
            transport_cost=transport_cost,
            weight1=weight1,
            weight2=weight2,
            solver=solver,
        )
        self._debug_check("transport plan", plans)

        if self.debug_transport:
            expected_evidence_shape = similarity_map.shape[:3]
            if evidence.shape != expected_evidence_shape:
                raise RuntimeError(
                    f"DICE-EMD evidence shape {evidence.shape} does not match expected {expected_evidence_shape}"
                )
            if logits.shape != similarity_map.shape[:2]:
                raise RuntimeError(f"DICE-EMD logits shape {logits.shape} does not match {similarity_map.shape[:2]}")
            self.last_transport_debug = {
                "cost": cost,
                "transport_cost": transport_cost,
                "distance": distance,
                "assignment_u": assignment,
                "disc": disc,
                "evidence_g": evidence,
                "plan": plans,
            }
        else:
            self.last_transport_debug = None

        return logits

    def emd_forward(
        self,
        proto: torch.Tensor,
        query: torch.Tensor,
        solver: str,
        return_aux: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        weight1 = self.get_weight_vector(query, proto)
        weight2 = self.get_weight_vector(proto, query)

        proto = self.normalize_feature(proto)
        query = self.normalize_feature(query)
        similarity_map = self.get_similarity_map(proto, query)
        logits = self.get_emd_distance(similarity_map, weight1, weight2, solver=solver)
        if not return_aux:
            return logits

        outputs: dict[str, torch.Tensor] = {
            "logits": logits,
            "aux_loss": logits.new_zeros(()),
        }
        if self.debug_transport and self.last_transport_debug is not None:
            outputs.update(self.last_transport_debug)
        return outputs

    @staticmethod
    def _merge_debug_outputs(debug_outputs: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        if not debug_outputs:
            return {}
        merged: dict[str, torch.Tensor] = {}
        for key in debug_outputs[0]:
            values = [item[key] for item in debug_outputs if key in item]
            if values and torch.is_tensor(values[0]):
                merged[key] = torch.cat(values, dim=0)
        return merged

    def forward(
        self,
        query: torch.Tensor,
        support: torch.Tensor,
        refine_proto: bool | None = None,
        exact: bool | None = None,
        sfc_update_step_override: int | None = None,
        sfc_bs_override: int | None = None,
        return_aux: bool = False,
        query_targets: torch.Tensor | None = None,
        support_targets: torch.Tensor | None = None,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        del query_targets, support_targets

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
        debug_outputs = []
        for batch_idx in range(batch_size):
            proto = support_feat[batch_idx].mean(dim=1)
            if refine_proto and shot_num > 1:
                proto = self.get_sfc(
                    support_feat[batch_idx],
                    solver=solver,
                    sfc_update_step=sfc_update_step_override,
                    sfc_bs=sfc_bs_override,
                )
            logits = self.emd_forward(proto, query_feat[batch_idx], solver=solver)
            outputs.append(logits)
            if self.debug_transport and self.last_transport_debug is not None:
                debug_outputs.append(self.last_transport_debug)

        logits = torch.cat(outputs, dim=0)
        if not return_aux:
            return logits

        payload: dict[str, torch.Tensor] = {
            "logits": logits,
            "aux_loss": logits.new_zeros(()),
        }
        if self.debug_transport:
            payload.update(self._merge_debug_outputs(debug_outputs))
        return payload


DICEEMDSimple = DICEEMD
