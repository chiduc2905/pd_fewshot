# RV-UOT: Reciprocal Verified UOT for Ours-Final

## Problem

The current `ours_final + global residual w=0.1` can classify well while the
visualized UOT plan still transports mass through unrelated regions.  The reason
is that the global residual changes only the final class score.  It does not
change the coupling that `mass overlay` and `matching` figures draw.

## Proposed Change

RV-UOT keeps the original Ours-Final UOT solver, then projects its plan onto a
reciprocal, locally coherent sub-plan:

```text
P = UOT(C, a, b)
V_ij = reciprocal_low_cost(C_ij) * neighborhood_coherence(C_ij)
P_verified = P * [(1 - beta) + beta * V]
score = score_scale * (T * mass(P_verified) - cost(P_verified))
```

The final `transport_plan` exposed to visualization is `P_verified`; the
original plan is stored as `rvuot_unverified_transport_plan`.

## Theory

Plain UOT answers whether a pair can move mass cheaply.  A single accidental
low-cost token pair can therefore attract visible mass even when it is not part
of the shared discharge pattern.

RV-UOT adds two necessary conditions for evidence:

1. Reciprocal distinctiveness: a match must be low-cost from both row-wise
   query-to-support and column-wise support-to-query views.
2. Local structural support: neighboring query tokens and neighboring support
   tokens must also support the same correspondence.

This turns the evidence object from an isolated low-cost edge into a locally
verified transport subgraph.

## Novelty Claim

The change is not another saliency prior and not a learned attention mask.  It is
a post-Sinkhorn verification operator over the transport graph itself.  It
aligns the scoring object with the visualization object: the model scores and
plots the same verified plan.

## Anti-Bias Argument

RV-UOT does not use absolute time indices, fixed time windows, brightness
thresholds, connected components, or PD-specific coordinates.  All gates are
episode-relative functions of the cost matrix and local token-grid adjacency.

If PD scalograms in one dataset concentrate at a certain time, RV-UOT can still
use that area when it is reciprocally and locally supported, but the method does
not assume that the area must be there.  A translated or differently positioned
evidence cluster receives the same verification response, up to boundary
effects.

## Few-Shot Fit

RV-UOT is parameter-free by default and uses only query/support tokens in the
current episode.  It adds no class-level memory, no extra labels, and no
dataset-level foreground detector, so it preserves the few-shot setting.

Recommended starting command fragment:

```bash
--model ours_final \
--enable_reciprocal_verified_uot \
--enable_global_residual_score \
--global_residual_weight 0.1
```

Useful diagnostics:

```text
rvuot/retained_mass_ratio
rvuot/removed_mass_mean
rvuot/gate_mean
rvuot/coherence_gate_mean
rvuot/shot_logit_delta_abs
```
