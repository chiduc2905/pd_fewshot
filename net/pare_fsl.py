"""PARE-FSL: Partial Adaptive Relational Evidence for few-shot scalogram matching.

Designed around entropic Partial OT (native batched solver only):
  - Token marginals sum to 1 (no fixed rho budget).
  - Transported mass is a *fraction* s in (0, 1] of available mass (mass-ratio bank).
  - Shot-decomposed matching (per support shot, then pool).
  - Class logits from a partial mass-response curve, not UOT KL relaxation.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from net.fewshot_common import BaseConv64FewShotModel, feature_map_to_tokens
from net.modules.partial_ot import (
    compute_partial_transport_cost,
    compute_partial_transported_mass,
    solve_partial_transport,
)


def parse_mass_ratio_bank(value: str | list[float] | tuple[float, ...]) -> tuple[float, ...]:
  if isinstance(value, str):
    ratios = [float(item.strip()) for item in value.split(",") if item.strip()]
  else:
    ratios = [float(item) for item in value]
  if not ratios:
    raise ValueError("pare_mass_ratio_bank must contain at least one ratio")
  unique: list[float] = []
  for ratio in ratios:
    if not 0.0 < ratio <= 1.0:
      raise ValueError("all mass ratios must be in (0, 1]")
    if not any(abs(ratio - kept) <= 1e-8 for kept in unique):
      unique.append(ratio)
  return tuple(sorted(unique))


class PAREFSLResult(dict):
  @property
  def shape(self):
    logits = self.get("logits")
    return None if logits is None else logits.shape


class PAREFSL(BaseConv64FewShotModel):
  """Shot-wise partial OT with discriminative marginals and mass-ratio response."""

  VALID_MARGINAL_MODES = frozenset({"uniform", "discriminative"})
  VALID_SHOT_POOLING = frozenset({"mean", "logsumexp"})
  VALID_MASS_AGGREGATION = frozenset({"softmax", "mean", "logmeanexp", "max"})

  def __init__(
    self,
    in_channels: int = 3,
    hidden_dim: int = 640,
    token_dim: int = 128,
    backbone_name: str = "resnet12",
    image_size: int = 84,
    resnet12_drop_rate: float = 0.0,
    resnet12_dropblock_size: int = 5,
    pare_mass_ratio_bank: str | list[float] | tuple[float, ...] = "0.35,0.5,0.65",
    sinkhorn_epsilon: float = 0.04,
    sinkhorn_iterations: int = 80,
    sinkhorn_tolerance: float = 1e-6,
    score_scale: float = 16.0,
    marginal_mode: str = "discriminative",
    marginal_temperature: float = 0.15,
    mass_aggregation: str = "softmax",
    mass_temperature: float = 1.0,
    shot_pooling: str = "logsumexp",
    shot_temperature_init: float = 1.0,
    eps: float = 1e-8,
  ) -> None:
    super().__init__(
      in_channels=in_channels,
      hidden_dim=hidden_dim,
      backbone_name=backbone_name,
      image_size=image_size,
      resnet12_drop_rate=resnet12_drop_rate,
      resnet12_dropblock_size=resnet12_dropblock_size,
    )
    if int(token_dim) <= 0:
      raise ValueError("token_dim must be positive")
    if float(sinkhorn_epsilon) <= 0.0:
      raise ValueError("sinkhorn_epsilon must be positive")
    if int(sinkhorn_iterations) <= 0:
      raise ValueError("sinkhorn_iterations must be positive")
    if float(sinkhorn_tolerance) < 0.0:
      raise ValueError("sinkhorn_tolerance must be non-negative")
    if float(score_scale) <= 0.0:
      raise ValueError("score_scale must be positive")
    if float(marginal_temperature) <= 0.0:
      raise ValueError("marginal_temperature must be positive")
    if float(mass_temperature) <= 0.0:
      raise ValueError("mass_temperature must be positive")
    if float(shot_temperature_init) <= 0.0:
      raise ValueError("shot_temperature_init must be positive")
    if float(eps) <= 0.0:
      raise ValueError("eps must be positive")

    marginal_mode = str(marginal_mode).lower()
    if marginal_mode not in self.VALID_MARGINAL_MODES:
      raise ValueError(f"marginal_mode must be one of {sorted(self.VALID_MARGINAL_MODES)}")
    shot_pooling = str(shot_pooling).lower()
    if shot_pooling not in self.VALID_SHOT_POOLING:
      raise ValueError(f"shot_pooling must be one of {sorted(self.VALID_SHOT_POOLING)}")
    mass_aggregation = str(mass_aggregation).lower()
    if mass_aggregation not in self.VALID_MASS_AGGREGATION:
      raise ValueError(f"mass_aggregation must be one of {sorted(self.VALID_MASS_AGGREGATION)}")

    self.token_dim = int(token_dim)
    self.pare_mass_ratio_bank = parse_mass_ratio_bank(pare_mass_ratio_bank)
    self.sinkhorn_epsilon = float(sinkhorn_epsilon)
    self.sinkhorn_iterations = int(sinkhorn_iterations)
    self.sinkhorn_tolerance = float(sinkhorn_tolerance)
    self.score_scale = float(score_scale)
    self.marginal_mode = marginal_mode
    self.marginal_temperature = float(marginal_temperature)
    self.mass_aggregation = mass_aggregation
    self.mass_temperature = float(mass_temperature)
    self.shot_pooling = shot_pooling
    self.eps = float(eps)

    self.register_buffer(
      "mass_ratio_bank_tensor",
      torch.tensor(self.pare_mass_ratio_bank, dtype=torch.float32),
      persistent=True,
    )
    self.token_projector = nn.Sequential(
      nn.LayerNorm(hidden_dim),
      nn.Linear(hidden_dim, self.token_dim, bias=False),
    )
    self.raw_shot_temperature = nn.Parameter(
      torch.tensor(math.log(math.expm1(float(shot_temperature_init))), dtype=torch.float32)
    )

  @property
  def num_mass_ratios(self) -> int:
    return int(self.mass_ratio_bank_tensor.numel())

  @property
  def shot_temperature(self) -> torch.Tensor:
    return F.softplus(self.raw_shot_temperature).clamp_min(self.eps)

  def _encode_images(self, images: torch.Tensor) -> torch.Tensor:
    tokens = feature_map_to_tokens(self.encode(images))
    tokens = self.token_projector(tokens)
    return F.normalize(tokens, p=2, dim=-1, eps=self.eps)

  def _encode_episode(
    self,
    query: torch.Tensor,
    support: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    if query.dim() != 4:
      raise ValueError(f"query must be (NumQuery, C, H, W), got {tuple(query.shape)}")
    if support.dim() != 5:
      raise ValueError(f"support must be (Way, Shot, C, H, W), got {tuple(support.shape)}")
    way_num, shot_num = support.shape[:2]
    images = torch.cat([query, support.reshape(way_num * shot_num, *support.shape[2:])], dim=0)
    tokens = self._encode_images(images)
    query_tokens = tokens[: query.shape[0]]
    support_tokens = tokens[query.shape[0] :].reshape(way_num, shot_num, tokens.shape[-2], tokens.shape[-1])
    return query_tokens, support_tokens

  @staticmethod
  def _pairwise_squared_distance(
    query_tokens: torch.Tensor,
    support_tokens: torch.Tensor,
  ) -> torch.Tensor:
    if query_tokens.dim() != 3:
      raise ValueError(
        "query_tokens must be (NumQuery, QueryTokens, Dim), "
        f"got {tuple(query_tokens.shape)}"
      )
    if support_tokens.dim() != 4:
      raise ValueError(
        "support_tokens must be (Way, Shot, SupportTokens, Dim), "
        f"got {tuple(support_tokens.shape)}"
      )
    diff = query_tokens[:, None, None, :, None, :] - support_tokens[None, :, :, None, :, :]
    return diff.square().sum(dim=-1).clamp_min(0.0)

  def _build_marginals(self, cost: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    leading = cost.shape[:-2]
    nq, ns = cost.shape[-2:]
    if self.marginal_mode == "uniform":
      a = cost.new_full((*leading, nq), 1.0 / float(nq))
      b = cost.new_full((*leading, ns), 1.0 / float(ns))
      return a, b

    tau = self.marginal_temperature
    q_logits = -cost.amin(dim=-1)
    s_logits = -cost.amin(dim=-2)
    a = torch.softmax(q_logits / tau, dim=-1)
    b = torch.softmax(s_logits / tau, dim=-1)
    return a, b

  def _aggregate_mass_scores(self, mass_scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if self.mass_aggregation == "mean":
      weights = torch.full_like(mass_scores, 1.0 / float(mass_scores.shape[-1]))
      return mass_scores.mean(dim=-1), weights
    if self.mass_aggregation == "max":
      indices = mass_scores.argmax(dim=-1, keepdim=True)
      weights = torch.zeros_like(mass_scores).scatter_(-1, indices, 1.0)
      return mass_scores.gather(-1, indices).squeeze(-1), weights
    if self.mass_aggregation == "logmeanexp":
      temperature = float(self.mass_temperature)
      logits = temperature * (
        torch.logsumexp(mass_scores / temperature, dim=-1) - math.log(float(mass_scores.shape[-1]))
      )
      weights = torch.softmax(mass_scores / temperature, dim=-1)
      return logits, weights
    weights = torch.softmax(mass_scores / float(self.mass_temperature), dim=-1)
    return (weights * mass_scores).sum(dim=-1), weights

  def _pool_shot_scores(self, shot_scores: torch.Tensor) -> torch.Tensor:
    if self.shot_pooling == "mean":
      return shot_scores.mean(dim=-1)
    tau = self.shot_temperature.to(device=shot_scores.device, dtype=shot_scores.dtype)
    scaled = shot_scores / tau.clamp_min(self.eps)
    return tau * (torch.logsumexp(scaled, dim=-1) - math.log(float(shot_scores.shape[-1])))

  def compute_partial_match(
    self,
    query_tokens: torch.Tensor,
    support_tokens: torch.Tensor,
  ) -> PAREFSLResult:
    """POT matching with shape (NumQuery, Way, Shot, Q, S) cost tensor."""
    cost = self._pairwise_squared_distance(query_tokens, support_tokens)
    num_query, way_num, shot_num, query_len, support_len = cost.shape
    num_pairs = num_query * way_num * shot_num
    num_ratios = self.num_mass_ratios

    a, b = self._build_marginals(cost)
    flat_cost = cost.reshape(num_pairs, query_len, support_len)
    flat_a = a.reshape(num_pairs, query_len)
    flat_b = b.reshape(num_pairs, support_len)

    flat_cost_exp = (
      flat_cost[:, None, :, :]
      .expand(num_pairs, num_ratios, query_len, support_len)
      .reshape(num_pairs * num_ratios, query_len, support_len)
    )
    flat_a_exp = (
      flat_a[:, None, :]
      .expand(num_pairs, num_ratios, query_len)
      .reshape(num_pairs * num_ratios, query_len)
    )
    flat_b_exp = (
      flat_b[:, None, :]
      .expand(num_pairs, num_ratios, support_len)
      .reshape(num_pairs * num_ratios, support_len)
    )
    ratio_bank = self.mass_ratio_bank_tensor.to(device=flat_cost.device, dtype=flat_cost.dtype)
    flat_ratio = ratio_bank.reshape(1, num_ratios).expand(num_pairs, num_ratios).reshape(-1)

    flat_plan = solve_partial_transport(
      flat_cost_exp,
      flat_a_exp,
      flat_b_exp,
      transport_mass_ratio=flat_ratio,
      backend="native",
      reg=self.sinkhorn_epsilon,
      max_iter=self.sinkhorn_iterations,
      tol=self.sinkhorn_tolerance,
      eps=self.eps,
    )
    flat_transport_cost = compute_partial_transport_cost(flat_plan, flat_cost_exp)
    flat_transport_mass = compute_partial_transported_mass(flat_plan)

    transport_cost_bank = flat_transport_cost.reshape(num_query, way_num, shot_num, num_ratios)
    transported_mass_bank = flat_transport_mass.reshape(num_query, way_num, shot_num, num_ratios)
    plan_bank = flat_plan.reshape(num_query, way_num, shot_num, num_ratios, query_len, support_len)

    partial_discrepancy = transport_cost_bank / transported_mass_bank.clamp_min(self.eps)
    mass_scores = -self.score_scale * partial_discrepancy

    shot_mass_scores, mass_weights = self._aggregate_mass_scores(mass_scores)
    logits = self._pool_shot_scores(shot_mass_scores)

    transport_cost = (mass_weights * transport_cost_bank).sum(dim=-1)
    transported_mass = (mass_weights * transported_mass_bank).sum(dim=-1)
    class_discrepancy = (mass_weights * partial_discrepancy).sum(dim=-1)
    class_discrepancy = self._pool_shot_scores(class_discrepancy)

    return PAREFSLResult(
      {
        "logits": logits,
        "class_scores": logits,
        "aux_loss": logits.new_zeros(()),
        "mass_scores": mass_scores,
        "mass_weights": mass_weights,
        "mass_ratio_bank": ratio_bank,
        "partial_discrepancy_bank": partial_discrepancy,
        "transport_cost_bank": transport_cost_bank,
        "transported_mass_bank": transported_mass_bank,
        "transport_cost": self._pool_shot_scores(transport_cost),
        "transported_mass": self._pool_shot_scores(transported_mass),
        "total_distance": class_discrepancy,
        "shot_temperature": self.shot_temperature.detach().to(device=logits.device, dtype=logits.dtype),
        "transport_plan_bank": plan_bank.detach(),
        "cost_matrix": cost.detach(),
      }
    )

  def _forward_episode(self, query: torch.Tensor, support: torch.Tensor) -> PAREFSLResult:
    query_tokens, support_tokens = self._encode_episode(query, support)
    outputs = self.compute_partial_match(query_tokens, support_tokens)
    outputs["query_tokens"] = query_tokens.detach()
    outputs["support_tokens"] = support_tokens.detach()
    return outputs

  @staticmethod
  def _stack_outputs(batch_outputs: list[PAREFSLResult]) -> PAREFSLResult:
    stacked: dict[str, Any] = {
      "logits": torch.cat([item["logits"] for item in batch_outputs], dim=0),
      "aux_loss": torch.stack([item["aux_loss"] for item in batch_outputs]).mean(),
      "class_scores": torch.cat([item["class_scores"] for item in batch_outputs], dim=0),
      "mass_scores": torch.cat([item["mass_scores"] for item in batch_outputs], dim=0),
      "mass_weights": torch.cat([item["mass_weights"] for item in batch_outputs], dim=0),
      "mass_ratio_bank": batch_outputs[0]["mass_ratio_bank"],
      "partial_discrepancy_bank": torch.cat(
        [item["partial_discrepancy_bank"] for item in batch_outputs], dim=0
      ),
      "transport_cost_bank": torch.cat([item["transport_cost_bank"] for item in batch_outputs], dim=0),
      "transported_mass_bank": torch.cat([item["transported_mass_bank"] for item in batch_outputs], dim=0),
      "transport_cost": torch.cat([item["transport_cost"] for item in batch_outputs], dim=0),
      "transported_mass": torch.cat([item["transported_mass"] for item in batch_outputs], dim=0),
      "total_distance": torch.cat([item["total_distance"] for item in batch_outputs], dim=0),
      "shot_temperature": batch_outputs[0]["shot_temperature"],
    }
    for key in ("transport_plan_bank", "cost_matrix", "query_tokens", "support_tokens"):
      if key in batch_outputs[0]:
        stacked[key] = torch.cat([item[key] for item in batch_outputs], dim=0)
    return PAREFSLResult(stacked)

  def forward(
    self,
    query: torch.Tensor,
    support: torch.Tensor,
    return_aux: bool = False,
  ) -> PAREFSLResult:
    del return_aux
    bsz, _, _, _, _, _ = self.validate_episode_inputs(query, support)
    batch_outputs = [self._forward_episode(query[idx], support[idx]) for idx in range(bsz)]
    return self._stack_outputs(batch_outputs)
