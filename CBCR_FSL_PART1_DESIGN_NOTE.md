# UBT-FSL Formulation and Architecture Design Note

This file keeps its legacy CBCR filename so older references do not break. The method should be written and discussed as:

```text
UBT-FSL: Uncertain Barycentric Transport for Few-Shot Learning
```

## 1. Purpose

The goal is to present the model as one compact few-shot transport formulation:

```text
support set -> uncertain barycentric class object -> robust class transport scoring
```

The model should not read as "barycenter + epsilon + competition + sink". The contribution is the class-level uncertain transport formulation.

## 2. Motivation

DeepEMD is strong because it preserves local descriptors and solves structured transport between query and support evidence. Its inference object is still pairwise query-support matching. UBT-FSL changes the inference object: each class becomes an uncertain barycentric transport object, and each query is scored by robust transport to that object.

Main claim:

```text
From pairwise support matching to uncertainty-aware class transport.
```

## 3. Class Object

For support shot measures `mu_{c,k}`, construct:

```text
P_c = (nu_c_hat, epsilon_c)
```

where:

- `nu_c_hat` is the barycentric token measure estimated from support shots;
- `epsilon_c` is the class transport radius estimated from support dispersion around the barycenter.

The implementation exposes this as `UncertainBarycentricClass`.

## 4. Radius

The radius is tied to the barycenter:

```text
d_c = (1 / K) sum_k W_epsilon(mu_{c,k}, nu_c_hat)
epsilon_c = max(alpha d_c, beta / sqrt(K))
```

This is an episode-specific and class-specific robust transport envelope. It should not be described as a threshold module.

Implementation names:

- `compute_transport_radius(...)`
- `transport_radius`
- `support_dispersion`
- `class_radius`

## 5. Scoring

The default score is:

```text
D_c(q) = OT(nu_q^c, nu_c_hat)
s_c(q) = -[D_c(q) - epsilon_c]_+
```

Implementation name:

```text
scoring = "robust_transport_envelope"
```

Raw OT scoring is an ablation only.

## 6. Query Ambiguity Correction

Query competition is optional and secondary:

```text
use_query_competition = true
```

It is a query-side correction that prevents ambiguous tokens from independently supporting multiple classes before transport. It should not be presented as the central class representation.

Implementation names:

- `compute_query_class_affinities(...)`
- `apply_query_ambiguity_correction(...)`
- `query_competition_temperature`

## 7. Transport Backend

Unbalanced OT is an internal backend for partial matching:

```text
transport_backend = "unbalanced_ot"
```

Balanced OT is exposed for ablation:

```text
transport_backend = "balanced_ot"
```

The sink or unbalanced solver should not be framed as a novelty.

## 8. Implementation API

Primary names:

- `UBTFSL`
- `UBTFSLResult`
- `UncertainBarycentricClass`
- `TokenMeasure`
- `build_uncertain_class_measure(...)`
- `compute_barycentric_measure(...)`
- `compute_transport_radius(...)`
- `build_query_measure(...)`
- `apply_query_ambiguity_correction(...)`
- `robust_transport_score(...)`

Backward-compatible aliases:

- `CBCRFSL = UBTFSL`
- `CBCRFSLResult = UBTFSLResult`
- `compute_barycenter(...)`
- `estimate_epsilon(...)`
- `compute_competitive_allocation(...)`

## 9. Ablations

Main:

1. `deepemd_pairwise`
2. `barycentric_only`
3. `uncertain_barycentric`
4. `ubt_full`

Appendix:

- `fixed_radius`
- `dynamic_radius`
- `balanced_ot`
- `unbalanced_ot`
- `no_beta_floor`
- `tau_sensitivity`
- `alpha_sensitivity`

The main ablation table should support:

```text
DeepEMD pairwise
+ Barycentric class measure
+ Transport radius
+ Query ambiguity correction
```

## 10. Paper-Facing Summary

UBT-FSL reformulates DeepEMD-style pairwise local matching as robust transport to an uncertain barycentric class measure. Each few-shot class is represented by a Wasserstein barycentric token measure equipped with a support-dispersion transport radius, and query classification penalizes only evidence outside that class transport envelope. Optional query ambiguity correction reallocates query token mass across classes before transport.
