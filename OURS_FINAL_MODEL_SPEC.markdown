# Ours-Final: Threshold-Calibrated Single-Budget UOT

This document describes the current paper-facing `--model ours_final` path. It
is meant as the source document for an LLM that must judge the design,
theoretical novelty, and empirical value of Ours-Final without confusing it
with legacy `--model ours`, EGSM, or the cost-only M2 baseline.

## 1. One-Sentence Summary

**Ours-Final is a few-shot scalogram classifier that represents each image by
local backbone descriptors, keeps support shots separate, solves one
rho=0.8 KL-relaxed unbalanced optimal transport problem per query-support-shot
pair, scores each shot with a learned threshold-mass evidence score
`score_scale * (T * transported_mass - transported_cost)`, and pools shots with
an episode-calibrated log-mean-exp temperature.**

Compact identity:

```text
Ours-Final =
  local shot-decomposed token measures
  + single-budget rho=0.8 UOT
  + threshold-calibrated mass-cost evidence score
  + episode-calibrated support-shot pooling
```

## 2. Current Repository Identity

Use this note for:

```bash
python main.py --model ours_final
```

Implementation path:

```text
net/model_factory.py
  -> model name: ours_final
  -> class: net.ours.OursM2
  -> base class: net.jecot_m2.JECOTM2
  -> backend: net.hrot_fsl.HROTFSL with variant="J_ECOT_M2"
```

The registry metadata describes the same core path:

```text
Backbone local descriptors
-> single-budget rho=0.8 UOT
-> threshold-mass score T*M-C
-> episode-calibrated shot pooling
EGSM is disabled by default.
```

Important naming distinction:

```text
--model ours
  legacy Ours path documented in OURS_NOTE.markdown
  EGSM is on by default for full/full_ot/gap
  threshold-mass score is ablated by default

--model ours_final
  final paper path described here
  EGSM is off by default
  threshold-mass score is on by default
```

## 3. Default Contract

The default `ours_final` configuration is intentionally conservative. It keeps
the single-budget M2 transport backbone and turns on the threshold-mass evidence
score, while leaving learned token-marginal extensions off.

| Item | Current default | Meaning |
|---|---:|---|
| model | `ours_final` | Paper-facing final entrypoint. |
| backend variant | `J_ECOT_M2` | Single-budget ECOT/M2 implementation path. |
| `ours_ablation` | `full` | Local descriptor cost path. |
| `hrot_ecot_rho_bank` | `0.8` | One transport budget only. |
| `hrot_ecot_base_rho` | `0.8` | The active budget is rho=0.8. |
| `hrot_fixed_mass` | `0.8` | Compatibility/default mass value. |
| `hrot_ecot_transport_mode` | `unbalanced` | KL-relaxed UOT, not balanced full OT. |
| `hrot_ecot_m2_ablate_threshold_mass` | `false` | Keep `T * mass` reward in the score. |
| `hrot_ecot_m2_cost_per_mass_score` | `false` | Do not use cost-per-mass scoring. |
| `hrot_ecot_enable_egsm` | `false` | EGSM is not part of Ours-Final. |
| MEA / CRS / CCDM / NNCS | off | No learned/cross-reference token marginals. |
| AQM / SWTS / MNCR / noise sink | off | Not part of the default final method. |
| `hrot_ecot_enable_tau_shot` | `true` | Use episode-calibrated shot pooling. |
| `hrot_ecot_tau_shot_min/max` | `0.5 / 2.0` | Bounds for the shot-pooling temperature. |
| `hrot_score_scale` | `16.0` | Multiplier for shot logits. |
| `hrot_mass_bonus_init` | `1.0` | Gives default threshold init `T = 1/16`. |
| `hrot_sinkhorn_epsilon` | `0.1` | Entropic regularization. |
| `hrot_tau_q`, `hrot_tau_c` | `0.5`, `0.5` | UOT marginal relaxation strengths. |
| `hrot_sinkhorn_iterations` | `60` | Sinkhorn iteration cap. |
| `hrot_ot_backend` | `native` | Native log-domain solver unless overridden. |
| `hrot_ground_cost` | `auto` | For J_ECOT_M2, this uses Euclidean cost over normalized projected tokens. |

The threshold is a learned positive scalar:

```text
T = softplus(raw_transport_cost_threshold)
```

If `--hrot_transport_cost_threshold_init` is omitted, the initialization is:

```text
T_init = hrot_mass_bonus_init / hrot_score_scale = 1.0 / 16.0 = 0.0625
```

## 4. Episode Interface

The model follows the repository's episodic few-shot interface:

```text
query:   [Batch, NumQuery, 3, H, W]
support: [Batch, Way, Shot, 3, H, W]
logits:  [Batch * NumQuery, Way]
```

For one episode, Ours-Final does not collapse all support shots in a class into
one prototype before transport. It computes a separate local transport score
for every query image and every `(class, shot)` pair:

```text
query q_i vs support shot x_{c,k}
```

This shot decomposition is central to the design. The model can later pool
shots, but the transport evidence is measured before class-level aggregation.

## 5. Tokenization And Cost Geometry

Each image is encoded by the configured few-shot backbone, commonly `resnet12`,
`conv64f`, `fsl_mamba`, or `slim_mamba`:

```text
image -> backbone feature map F in R^{C x Hf x Wf}
      -> spatial tokens Z in R^{L x C}, L = Hf * Wf
      -> LayerNorm/Linear transport projector
      -> L2-normalized Euclidean tokens
```

For `ours_final` with default `ours_ablation=full`, the active matching object
is the local token grid. The default `hrot_ground_cost=auto` is not hyperbolic
on `J_ECOT_M2`; it falls through to Euclidean cost over the normalized
projected tokens.

Conceptually:

```text
D_{i,c,k}[r,l] = || q_i[r] - s_{c,k}[l] ||_2^2
```

where `r` indexes query tokens and `l` indexes support-shot tokens. If the user
overrides `--hrot_ground_cost cosine`, the cost becomes cosine distance. That
override is not the default Ours-Final claim.

The `gap` ablation changes this object:

```text
full: local spatial tokens
gap:  one global-average-pooled token per image
```

The `gap` variant is a control for whether local descriptor transport is
actually contributing beyond global prototype-like matching.

## 6. Uniform Fixed-Budget Marginals

Default Ours-Final uses uniform token marginals with one nominal transport
budget:

```text
a_r = rho / Lq
b_l = rho / Ls
rho = 0.8
```

The total nominal query and support mass is therefore `rho` on both sides. The
unbalanced solver is KL-relaxed, so the optimized plan may softly deviate from
these marginals, but the intended inductive bias is a fixed partial-evidence
budget rather than full-token matching.

This is a deliberate choice. Ours-Final does not use:

```text
EGSM episode-gated marginals
MEA attention marginals
CRS cross-reference support marginals
CCDM cost-derived marginals
NNCS consistency marginals
AQM query marginals
amplitude-weighted marginals
token saliency gates
noise sinks
```

These are separate extensions or ablations in the codebase, not the final
method described here.

## 7. UOT Solver

For each query-support-shot pair, Ours-Final solves an entropic KL-relaxed
unbalanced OT problem using the repository's Sinkhorn solver:

```text
P_{i,c,k} = UOT(D_{i,c,k}, a, b)
```

The native solver follows the standard entropy-regularized unbalanced Sinkhorn
scaling:

```text
K = exp(-D / epsilon)
rho_q = tau_q / (tau_q + epsilon)
rho_c = tau_c / (tau_c + epsilon)
```

with defaults:

```text
epsilon = 0.1
tau_q   = 0.5
tau_c   = 0.5
iters   = 60
```

The quantities used by the classifier are:

```text
C_{i,c,k} = sum_{r,l} P_{i,c,k}[r,l] * D_{i,c,k}[r,l]
M_{i,c,k} = sum_{r,l} P_{i,c,k}[r,l]
```

`C` is transported cost. `M` is transported real-token mass. In the default
path there is no noise sink, so all transported mass is real query/support token
mass.

## 8. Threshold-Mass Evidence Score

The key difference from the cost-only M2 baseline is the shot score. Ours-Final
keeps the threshold-mass term:

```text
E_{i,c,k} = score_scale * (T * M_{i,c,k} - C_{i,c,k})
```

where:

```text
T = learned positive cost threshold
M = transported mass
C = transported cost
```

Interpretation:

```text
T * M - C = sum_{r,l} P[r,l] * (T - D[r,l])
```

So a transported token pair contributes positive evidence only when its cost is
below the learned threshold `T`, and contributes negative evidence when its cost
is above `T`. This is not just "reward more mass"; it rewards mass only when the
transport plan can move it through low-cost token correspondences.

Why this matters:

```text
Cost-only score:        -C
Threshold-mass score:   T*M - C
```

Cost-only scoring can make very small transported mass look attractive if the
remaining transport is cheap. Threshold-mass scoring asks a stronger question:

```text
Can this class transport enough local evidence below the learned cost threshold?
```

This is the main theoretical change that separates Ours-Final from the
standalone M2 cost-only note.

## 9. Episode-Calibrated Shot Pooling

After transport, each class has one shot score per support example:

```text
E_{i,c,1}, ..., E_{i,c,K}
```

Ours-Final pools these shot scores with an episode-conditioned temperature
`tau_shot`:

```text
S_{i,c} = tau_shot * (logsumexp_k(E_{i,c,k} / tau_shot) - log K)
```

For `K=1`, this is just the single shot score. For `K>1`, the temperature
controls how aggressively the model trusts the best support shots:

```text
smaller tau_shot -> closer to max-shot pooling
larger tau_shot  -> smoother averaging over shots
```

`tau_shot` is produced by the ECOT episode controller from summary diagnostics
of the current episode, including score statistics, cost statistics, transported
mass statistics, class margin statistics, shot entropy, and budget-slope
statistics. In Ours-Final there is only one budget, so the budget-policy part of
the controller degenerates to the single rho=0.8 expert; the useful controller
output is the shot-pooling temperature.

This makes Ours-Final shot-aware without learning per-shot transport masses.
The support examples remain decomposed until after their transport evidence has
been measured.

## 10. What Is Novel

Ours-Final is not claiming a new OT solver. It is a method-level composition of
low-variance decisions for noisy, local, few-shot scalogram evidence.

The defensible novelty is:

1. **Threshold-calibrated partial transport evidence.**
   The score treats the learned threshold as a per-unit-mass evidence boundary:
   each transported correspondence contributes `T - cost`, so the model favors
   both low cost and sufficient transported mass.

2. **Single-budget UOT as an inductive bias.**
   A fixed rho=0.8 budget avoids full-token overmatching while avoiding the
   variance of a learned mass policy or a multi-budget selector.

3. **Shot-decomposed local evidence.**
   Each support shot is matched independently at the local-token level before
   class aggregation, preserving information about shot quality and avoiding
   premature class prototype averaging.

4. **Episode-calibrated shot aggregation.**
   The model adapts only how support-shot evidence is pooled, using episode
   diagnostics, while the transport problem itself remains stable and
   interpretable.

The expected paper framing should be "a threshold-calibrated, single-budget,
shot-decomposed UOT matcher", not "a new optimal transport algorithm".

## 11. Why It Can Be Effective

Ours-Final targets partial-discharge scalograms where discriminative evidence
is local, incomplete, and sometimes corrupted by background/noise. Its design
answers four practical failure modes:

| Failure mode | Ours-Final response |
|---|---|
| Full-image prototypes wash out local signatures. | Match local token measures. |
| Full OT forces irrelevant regions to match. | Use rho=0.8 UOT partial evidence. |
| Tiny cheap matches can dominate cost-only scoring. | Use `T*M-C` to require enough low-cost transported mass. |
| Some support shots are worse than others. | Keep shots separate and pool with episode-conditioned temperature. |

The central effectiveness hypothesis is:

```text
The correct class should transport a larger amount of local scalogram evidence
below the learned cost threshold, and the shot-pooling controller should rely
more on the support shots that exhibit this pattern.
```

## 12. Required Ablations For Novelty And Effectiveness

The code already defines the main Ours-Final ablation suite in
`run_all_experiments.py`.

Contribution ablations:

| Variant | Meaning | Question answered |
|---|---|---|
| `ours_final_full` | Default Ours-Final. | Full method. |
| `ours_final_full_ot` | Balanced full OT, rho=1.0. | Is partial UOT better than forcing full transport? |
| `ours_final_gap` | GAP token cost instead of local descriptors. | Are local token measures necessary? |
| `ours_final_mass_off` | Cost-only score; removes `T*M`. | Does threshold-mass evidence help beyond cost-only M2? |
| `ours_final_tau_shot_off` | Disables ECOT tau-shot pooling. | Does episode-calibrated shot pooling help? |
| `ours_final_rho_*` | Rho grid, e.g. 0.6/0.7/0.8/0.9. | Is rho=0.8 a stable budget or tuned accident? |

Useful commands:

```bash
python main.py --model ours_final
```

```bash
python run_all_experiments.py \
  --models ours_final \
  --ours_final_ablation_suite contrib
```

```bash
python run_all_experiments.py \
  --models ours_final \
  --ours_final_ablation_suite complete
```

Minimum evidence needed for a credible claim:

```text
full > mass_off      supports threshold-mass scoring
full > full_ot       supports partial UOT / rho<1
full > gap           supports local token transport
full > tau_shot_off  supports episode-calibrated shot pooling
rho=0.8 competitive across rho grid  supports fixed-budget robustness
```

If only `full` beats external baselines but these internal ablations are weak,
the novelty claim should be softened to an engineering combination rather than
a clearly validated methodological contribution.

## 13. External Baselines To Compare

To judge novelty and effectiveness, compare against both prototype-style and
transport-style baselines:

```text
ProtoNet / cosine:
  tests whether local transport is needed beyond global embedding geometry.

DN4 / DeepEMD:
  tests whether Ours-Final beats established local-descriptor matching.

J-ECOT-M2 / m2:
  cost-only single-budget UOT baseline.

legacy --model ours:
  EGSM marginal path; separates final threshold-mass design from EGSM.

full OT ablations:
  tests whether partial unbalanced transport matters.
```

For noise or SNR-shifted splits, report clean and noise results separately. A
strong Ours-Final story is not only higher mean accuracy, but stable gains under
low-shot and noisy conditions where local partial evidence should matter.

## 14. Diagnostics To Inspect

When `return_aux=True`, the active path exposes the useful diagnostics:

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

Good diagnostic signs:

```text
true-class transport_cost lower than best wrong class
true-class transported_mass not collapsed to tiny values
true-class shot_logits separated from wrong classes
shot_pool_weights concentrate on plausible support shots in 5-shot episodes
learned T is finite and not driven to a trivial extreme
rho-grid does not show severe instability around rho=0.8
```

Bad diagnostic signs:

```text
transported_mass collapses for all classes
T grows so large that mass reward dominates cost discrimination
T shrinks so low that score becomes nearly cost-only
shot_pool_weights always uniform despite mixed support quality
performance depends sharply on one rho value
```

## 15. Claim Boundary

Safe claim:

```text
Ours-Final is a threshold-calibrated, single-budget, shot-decomposed UOT
few-shot matcher for PD scalograms. It scores local transport evidence by
learned per-unit-mass threshold surplus, T*M-C, and aggregates support shots
with an episode-conditioned log-mean-exp temperature.
```

Do not claim:

```text
new optimal transport solver
hyperbolic transport geometry
learned token-mass policy
multi-budget ECOT policy
EGSM / MEA / CRS / CCDM marginal learning
noise-sink rejection
DeepEMD-style cross-reference weighting
global prototype-only matching
```

The novelty should be evaluated as a calibrated few-shot matching design, not
as a fundamental OT theory contribution.

## 16. Relation To Nearby Notes

Use the following boundaries when reading the other documents:

```text
OURS_NOTE.markdown:
  legacy --model ours with EGSM on and threshold-mass ablated by default.
  Not the Ours-Final claim.

J_ECOT_M2_STANDALONE_MODEL_SPEC.md:
  standalone M2 / SB-ECOT cost-only baseline.
  Useful foundation, but it disables T*M and therefore is not Ours-Final.

MEA_M2_NOTE.markdown, CRS_M2_NOTE.markdown, NS_M2_NOTE.markdown:
  alternative ECOT marginal/noise extensions.
  They are not enabled in Ours-Final.
```

## 17. Short Abstract-Style Description

Ours-Final formulates few-shot partial-discharge scalogram classification as
threshold-calibrated partial evidence transport. Each image is represented by
local backbone descriptors, and each support shot remains a separate empirical
measure. For every query-support-shot pair, the model solves a fixed rho=0.8
KL-relaxed unbalanced OT problem. The resulting plan is scored by a learned
threshold-mass criterion, `T*M-C`, which rewards transported mass only when it
moves through sufficiently low-cost local correspondences. Class logits are
formed by an episode-conditioned log-mean-exp pooling over support shots. The
method therefore combines local matching, partial evidence selection,
threshold-calibrated transport scoring, and adaptive shot aggregation without
introducing learned token marginals or a new OT solver.
