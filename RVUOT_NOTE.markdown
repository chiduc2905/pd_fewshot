# RV-UOT: Reciprocal Verified UOT for Ours-Final

## Problem

The current `ours_final + global residual w=0.1` can classify well while the
visualized UOT plan still transports mass through unrelated regions.  The reason
is that the global residual changes only the final class score.  It does not
change the coupling that `mass overlay` and `matching` figures draw.

## Proposed Change

RV-UOT keeps the original Ours-Final UOT solver, then builds two episode-local
sub-plans from the same cost matrix and transport plan:

1. A stable scoring plan, verified by reciprocal low-cost and local coherence.
2. A stricter evidence certificate, additionally filtered by one-vs-rival
   episode contrast.

```text
P = UOT(C, a, b)
S_ij = reciprocal_low_cost(C_ij)
     * neighborhood_coherence(C_ij)
R_ij = rival_contrast(C_ij, P)

P_score = P * [(1 - beta) + beta * S]
P_evidence = P * S * R
score = score_scale * (T * mass(P_score) - cost(P_score))
```

By default, `rvuot_rival_score_mix=0.0`, so the rival contrast does not perturb
the prediction score.  It only defines `rvuot_evidence_transport_plan`, which is
the plan used by the paper-style mass overlay and transport matrix.  If desired,
`rvuot_rival_score_mix>0` can be used as an ablation to mix contrastive evidence
back into the scoring plan.

The final `transport_plan` remains the stable scoring plan; the original plan is
stored as `rvuot_unverified_transport_plan`, and the stricter visualization plan
is stored as `rvuot_evidence_transport_plan`.

The interpolation floor `(1 - beta)` is intentionally kept out of
`P_evidence`.  The floor is useful for stable logits, but it is not valid visual
evidence because a large noisy transport mass would remain visible even after
being rejected by the verifier.

## Theory

Plain UOT answers whether a pair can move mass cheaply.  A single accidental
low-cost token pair can therefore attract visible mass even when it is not part
of the shared discharge pattern.

RV-UOT adds three necessary conditions for evidence:

1. Reciprocal distinctiveness: a match must be low-cost from both row-wise
   query-to-support and column-wise support-to-query views.
2. Local structural support: neighboring query tokens and neighboring support
   tokens must also support the same correspondence.
3. Episode-contrastive specificity: a query token must support the candidate
   class more than rival classes in the same episode.  Common-mode texture or
   noise that matches several classes is down-weighted even if it is low-cost.

This turns the visual evidence object from an isolated low-cost edge into a
locally verified and class-specific transport subgraph, while keeping the
classifier on the less brittle structural RV-UOT measure unless explicitly
ablated.

## Novelty Claim

The change is not another saliency prior and not a learned attention mask.  It is
a post-Sinkhorn verification operator over the episode transport graph itself.
The contrast term is specific to few-shot classification: support labels define
the rival classes inside the episode, so the verifier can reject class-common
transport mass without external foreground labels or dataset-level noise
assumptions.

The novelty is the separation between a decision measure and an evidence
certificate derived from the same transport graph.  The decision measure stays
stable for few-shot accuracy; the certificate is stricter and asks whether a
transported region is not only cheap and locally coherent, but also specific to
the predicted/true class against episode rivals.

## Anti-Bias Argument

RV-UOT does not use absolute time indices, fixed time windows, brightness
thresholds, connected components, or PD-specific coordinates.  All gates are
episode-relative functions of the cost matrix, local token-grid adjacency, and
one-vs-rival evidence inside the current support set.

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
rvuot/evidence_retained_mass_ratio
rvuot/evidence_removed_mass_mean
rvuot/gate_mean
rvuot/evidence_gate_mean
rvuot/coherence_gate_mean
rvuot/rival_gate_mean
rvuot/rival_score_mix
rvuot/rival_cost_gate_mean
rvuot/rival_mass_gate_mean
rvuot/shot_logit_delta_abs
```
