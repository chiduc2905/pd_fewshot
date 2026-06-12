# Residual-Aligned UOT from Ours-Final + Global Residual

## Starting Point

The accepted main model is:

```text
local UOT logits + 0.1 * global prototype residual logits
```

The logs show that the global residual branch often carries the class decision,
while local UOT can still spend mass on cheap common texture.  Residual-Aligned
UOT keeps the same global residual branch and uses it as an episode-local anchor
before solving UOT.

## Core Idea

For each class, build the same support prototype used by global residual scoring.
Then remove the episode mean prototype to form a class-contrastive residual
anchor:

```text
p_c = normalize(mean support tokens of class c)
r_c = normalize(p_c - mean_k p_k)
```

Each query/support token receives a local residual contribution:

```text
q_evidence(i,c) = sigmoid(standardize(<q_i, r_c>))
s_evidence(j,c) = sigmoid(standardize(<s_j^c, r_c>))
pair_evidence(i,j,c) = sqrt(q_evidence(i,c) * s_evidence(j,c))
```

The UOT ground problem is then modified before transport:

```text
C_ra = C - lambda * std(C) * center(pair_evidence)
a_ra, b_ra = blend(uniform, residual-token probabilities)
P_ra = UOT(C_ra, a_ra, b_ra)
```

This is not a post-hoc verifier.  It changes the cost/marginals that UOT sees.

## Why This Is Not Brightness Or Time Bias

The method does not use pixel intensity, fixed time windows, frequency bands,
dark/bright thresholds, connected components, or manually selected PD regions.
It uses only support/query features in the current few-shot episode and the
existing global residual prototypes.  A token is preferred when it explains the
same class residual direction that is already used by the global residual score.

## Relation To RV-UOT And ECT-UOT

- RV-UOT verifies a solved plan.  With `rvuot_rival_score_mix=0`, rival
  verification is evidence-only and does not fix score-path matching.
- ECT-UOT compares local costs against rival costs.  In the inspected log it
  collapsed: almost every pair was treated as common.
- Residual-Aligned UOT starts from the stronger branch, global residual, and
  decomposes it into token-level evidence before UOT.

## What To Check

Useful diagnostics:

```text
residual_aligned/pair_evidence_mean
residual_aligned/pair_evidence_std
residual_aligned/cost_delta_ratio
residual_aligned/query_weight_entropy
transport_audit/common_mass_ratio
transport_audit/common_score_term
local_score_gap
global_score_gap
```

The method is promising only if local score gap improves, common mass decreases
or becomes score-justified, and clean/noise accuracy does not drop.
