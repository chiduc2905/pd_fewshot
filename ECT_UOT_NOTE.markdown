# ECT-UOT: Episode-Contrastive Transport for Ours-Final

## Problem

Ours-Final scores a class with local UOT terms:

```text
score = scale * (T * transported_mass - transport_cost)
```

If weak texture/noise regions have low feature cost and exist across several
classes, UOT can transport mass through them.  Global residual scoring can fix a
class logit, but it does not change the local transport plan.  Post-hoc RV-UOT
can filter the displayed plan, but it is still downstream of a cost matrix that
may already make common-mode regions cheap.

## Core Idea

ECT-UOT changes the transport problem before solving UOT.  For each query token
and candidate class, it compares the candidate pair cost with the best rival
class cost for the same query token:

```text
rival_i^c = softmin_{c' != c, s, j} C(i, c', s, j)
advantage(i,c,s,j) = rival_i^c - C(i,c,s,j)
gate(i,c,s,j) = sigmoid((advantage - margin) / tau)
```

Then it uses this gate in two places:

```text
C_ect = C + lambda * std(C) * (1 - gate)
a_ect = blend(uniform, softmax(log max_j gate / tau_q))
P_ect = UOT(C_ect, a_ect, b)
```

So common-mode tokens that match rival classes equally well become expensive and
receive less query-side mass before UOT creates the coupling.

## Why This Is Not Brightness Or Time Bias

The method never uses pixel intensity, dark/bright thresholds, fixed time
windows, frequency bands, connected components, or scalogram coordinates.  A
token is suppressed only when it is not class-specific inside the current
episode.  If a real PD region moves to another time location, it is still kept
when its local cost is better for the candidate class than for the rival
classes.

## Relation To Existing Components

- Global residual: changes the final image-level logit, but not the transport
  plan.
- RV-UOT: verifies a solved plan after UOT; useful for evidence, but it cannot
  prevent UOT from initially spending mass on common-mode low-cost regions.
- Old cost-evidence marginals: up-weight tokens with low soft-min cost.  That is
  insufficient here because common/noise regions can also have low cost.
- ECT-UOT: uses one-vs-rival cost advantage, not absolute low cost.

## Novelty Claim

ECT-UOT is an episode-level contrastive modification of the UOT ground problem:
it changes both the ground cost and query marginal using the same rival-aware
transport evidence.  It is parameter-free apart from scalar temperatures and
weights, uses only query/support tokens in the current few-shot episode, and does
not require foreground labels or a learned attention branch.

## Audit Metrics

The implementation logs these for comparing Ours-Final, global residual, RV-UOT,
and ECT-UOT:

```text
transport_audit/common_mass_ratio
transport_audit/specific_mass_ratio
transport_audit/cost_per_mass
transport_audit/query_mass_entropy
episode_contrast/gate_mean
episode_contrast/cost_delta_ratio
episode_contrast/query_weight_entropy
```

The key sanity check is that `common_mass_ratio` should drop while accuracy does
not fall.  If it drops but accuracy falls, the contrast gate is too aggressive
and should be relaxed through `ect_uot_cost_weight` or `ect_uot_query_mass_mix`.
