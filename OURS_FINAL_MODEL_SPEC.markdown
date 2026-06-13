# Ours-Final: Standalone Model Specification

This document describes the original standalone `--model ours_final` path in
this repository. Ours-Final is treated here as a complete model with its own
architecture and scoring rule.

## 1. Model Identity

Ours-Final is a few-shot scalogram classifier built around local evidence
transport. Each image is represented as a set of spatial backbone descriptors.
Each support shot remains separate until after transport. For every
query/support-shot pair, the model solves one fixed-budget KL-relaxed
unbalanced optimal transport problem and scores the resulting plan with a
learned threshold-mass evidence criterion.

Compact identity:

```text
Ours-Final =
  local image token measures
  + shot-decomposed support matching
  + single-budget rho=0.8 unbalanced OT
  + threshold-calibrated mass-cost evidence score
  + episode-calibrated support-shot pooling
```

Repository entrypoint:

```bash
python main.py --model ours_final
```

Implementation path:

```text
net/model_factory.py
  -> model name: ours_final
  -> class: net.ours.OursM2
  -> base class: net.jecot_m2.JECOTM2
  -> backend: net.hrot_fsl.HROTFSL
  -> backend mode: J_ECOT_M2
```

## 2. Default Architecture Contract

The original Ours-Final path has the following fixed architectural contract:

| Item | Default | Meaning |
|---|---:|---|
| `model` | `ours_final` | Standalone paper-facing entrypoint. |
| backend mode | `J_ECOT_M2` | Single-budget ECOT/M2 implementation. |
| `hrot_ecot_rho_bank` | `0.8` | One transport budget. |
| `hrot_ecot_base_rho` | `0.8` | Active rho value. |
| `hrot_fixed_mass` | `0.8` | Nominal fixed transport mass. |
| `hrot_ecot_transport_mode` | `unbalanced` | KL-relaxed unbalanced OT. |
| `hrot_ecot_m2_ablate_threshold_mass` | `false` | Keep the `T * mass` term. |
| `hrot_ecot_m2_cost_per_mass_score` | `false` | Use threshold-mass score, not cost-per-mass. |
| `hrot_ecot_m2_mass_score_mode` | `standard` | Score each support shot with its own transported mass. |
| token marginals | uniform | Fixed by the architecture; EGSM is not available on `ours_final`. |
| `hrot_ecot_enable_tau_shot` | `true` | Episode-calibrated support-shot pooling. |
| `hrot_ecot_tau_shot_min/max` | `0.5 / 2.0` | Bounds for shot-pooling temperature. |
| `hrot_score_scale` | `16.0` | Multiplier applied to shot evidence. |
| `hrot_mass_bonus_init` | `1.0` | Gives default threshold init `T = 1/16`. |
| `hrot_sinkhorn_epsilon` | `0.1` | Entropic regularization. |
| `hrot_tau_q`, `hrot_tau_c` | `0.5`, `0.5` | UOT marginal relaxation strengths. |
| `hrot_sinkhorn_iterations` | `60` | Sinkhorn iteration cap. |
| `hrot_ot_backend` | `native` | Native log-domain OT solver. |
| `hrot_ground_cost` | `auto` | Euclidean cost over normalized projected tokens. |

Passing `--hrot_ecot_enable_egsm true` or enabling EGSM adaptive rho with an
Ours-Final model is rejected. EGSM remains available only to the separate
`ours` and `ours_cpm` model families. The redundant
`--ours_ablation no_egsm` selector is also rejected for Ours-Final.

The learned evidence threshold is a positive scalar:

```text
T = softplus(raw_transport_cost_threshold)
```

With default initialization:

```text
T_init = hrot_mass_bonus_init / hrot_score_scale = 1.0 / 16.0 = 0.0625
```

## 3. Episodic Interface

Ours-Final follows the repository few-shot episode shape:

```text
query:   [Batch, NumQuery, 3, H, W]
support: [Batch, Way, Shot, 3, H, W]
logits:  [Batch * NumQuery, Way]
```

For one episode, the model evaluates each query image against every support
shot separately:

```text
query q_i vs support shot x_{c,k}
```

The transport evidence is measured before class-level aggregation. A class logit
is produced only after all shot-level scores for that class have been computed.

## 4. Local Token Measures

Each image is encoded by the configured few-shot backbone:

```text
image
  -> backbone feature map F in R^{C x Hf x Wf}
  -> spatial tokens Z in R^{L x C}, L = Hf * Wf
  -> transport projector
  -> L2-normalized Euclidean tokens
```

The matching object is the local token grid. For query token `r` and support
token `l`, the default ground cost is:

```text
D_{i,c,k}[r,l] = || q_i[r] - s_{c,k}[l] ||_2^2
```

This gives Ours-Final a local evidence geometry: the model compares spatial
descriptor sets instead of reducing each image to one global prototype before
matching.

## 5. Fixed-Budget Uniform Marginals

Ours-Final uses uniform token marginals with one nominal transport budget:

```text
a_r = rho / Lq
b_l = rho / Ls
rho = 0.8
```

The total nominal query mass and support mass are both `0.8`. The unbalanced OT
solver may softly deviate from these marginals through KL relaxation, but the
intended inductive bias is fixed partial evidence transport rather than forced
full-image matching.

## 6. Unbalanced Transport

For each query/support-shot pair, Ours-Final solves:

```text
P_{i,c,k} = UOT(D_{i,c,k}, a, b)
```

The native solver uses entropy-regularized unbalanced Sinkhorn scaling with:

```text
epsilon = 0.1
tau_q   = 0.5
tau_c   = 0.5
iters   = 60
```

The classifier consumes two plan statistics:

```text
C_{i,c,k} = sum_{r,l} P_{i,c,k}[r,l] * D_{i,c,k}[r,l]
M_{i,c,k} = sum_{r,l} P_{i,c,k}[r,l]
```

`C` is transported cost. `M` is transported real-token mass.

## 7. Threshold-Mass Evidence Score

Ours-Final converts each transport plan into a shot-level evidence score:

```text
E_{i,c,k} = score_scale * (T * M_{i,c,k} - C_{i,c,k})
```

where:

```text
T = learned positive cost threshold
M = transported mass
C = transported cost
```

Equivalently:

```text
T * M - C = sum_{r,l} P[r,l] * (T - D[r,l])
```

A transported token correspondence contributes positive evidence when its cost
is below the learned threshold `T`, and negative evidence when its cost is above
`T`. The score therefore rewards transported mass only when that mass moves
through sufficiently low-cost local correspondences.

The model's central evidence question is:

```text
Can this class transport enough local scalogram evidence below the learned
cost threshold?
```

## 8. Episode-Calibrated Shot Pooling

After transport, each class has one score per support shot:

```text
E_{i,c,1}, ..., E_{i,c,K}
```

Ours-Final pools these scores with an episode-conditioned log-mean-exp
temperature:

```text
S_{i,c} = tau_shot * (logsumexp_k(E_{i,c,k} / tau_shot) - log K)
```

For `K=1`, this equals the single support-shot score. For `K>1`, smaller
`tau_shot` behaves closer to best-shot pooling, while larger `tau_shot` smooths
the aggregation across support shots.

`tau_shot` is produced from episode diagnostics such as score statistics, cost
statistics, transported-mass statistics, class margins, shot entropy, and
budget-slope statistics. In the standalone Ours-Final path there is only one
transport budget, so the useful adaptive output is the shot-pooling
temperature.

The final class logit is:

```text
logit(q_i, class c) = S_{i,c}
```

## 9. Forward Computation

The standalone forward path can be summarized as:

```text
1. Encode query and support images with the few-shot backbone.
2. Convert feature maps to spatial token sets.
3. Project and L2-normalize local tokens.
4. Build one token cost matrix for every query/support-shot pair.
5. Assign uniform rho=0.8 token marginals.
6. Solve KL-relaxed unbalanced OT for each pair.
7. Compute transported cost C and transported mass M.
8. Score each shot with score_scale * (T*M - C).
9. Pool support-shot scores with episode-calibrated log-mean-exp.
10. Return class logits.
```

## 10. Expected Diagnostics

When auxiliary outputs are requested, the core path exposes diagnostics such as:

```text
transport_cost
transported_mass
rho
shot_rho
shot_transport_cost
shot_transported_mass
shot_logits
shot_pool_weights
transport_plan
ecot_tau_shot
ecot_rho_bank
ecot_base_score
ecot_budget_scores
ecot_shot_transport_cost_bank
ecot_shot_transported_mass_bank
ecot_diagnostics
transport_cost_threshold
```

Useful behavior to inspect:

```text
true-class transport_cost is lower than strong wrong classes
true-class transported_mass does not collapse to near zero
true-class shot_logits separate from wrong-class shot_logits
shot_pool_weights emphasize plausible support shots in multi-shot episodes
learned threshold T stays finite and nontrivial
```

## 11. Standalone Claim

Ours-Final formulates few-shot partial-discharge scalogram classification as
threshold-calibrated partial evidence transport. Each image is represented by
local backbone descriptors, each support shot is matched independently, and
each query/support-shot pair is compared through a fixed rho=0.8 KL-relaxed
unbalanced OT problem.

The resulting transport plan is scored by the learned threshold-mass criterion
`T*M-C`, which rewards transported mass only when it moves through low-cost
local correspondences. Class logits are formed by episode-conditioned
log-mean-exp pooling over support-shot evidence.

In short:

```text
Ours-Final is a standalone, threshold-calibrated, single-budget,
shot-decomposed UOT matcher for few-shot partial-discharge scalograms.
```
