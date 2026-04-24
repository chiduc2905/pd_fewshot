# UBT-FSL Compact Research Note

This note supersedes the earlier CBCR-FSL wording. The implementation keeps the old filename and `CBCRFSL` alias for compatibility, but the paper-facing method name is:

```text
UBT-FSL: Uncertain Barycentric Transport for Few-Shot Learning
```

## 1. Diagnosis

The old CBCR framing made the method look like a stack of modules:

- class Wasserstein barycenter;
- dynamic uncertainty radius;
- competitive query allocation;
- unbalanced OT as a partial matching mechanism.

That story is too fragmented. Reviewers can read it as "DeepEMD plus several add-ons" rather than a compact class-level transport formulation.

## 2. Core Claim

The defensible contribution is:

```text
From pairwise support matching to uncertainty-aware class transport.
```

DeepEMD performs pairwise support matching. UBT-FSL performs robust transport to an uncertain barycentric class object.

## 3. Unified Class Object

Each support shot is a token measure `mu_{c,k}`. UBT-FSL represents class `c` as:

```text
P_c = (nu_c_hat, epsilon_c)
```

where `nu_c_hat` is a barycentric token measure estimated from the support shots, and `epsilon_c` is a transport radius estimated from support dispersion around that barycenter.

The radius is tied to the barycenter:

```text
d_c = (1 / K) sum_k W_epsilon(mu_{c,k}, nu_c_hat)
epsilon_c = max(alpha d_c, beta / sqrt(K))
```

This makes the radius an episode-specific and class-specific robust transport envelope, not an arbitrary threshold.

## 4. Robust Class Transport Score

Classification uses:

```text
D_c(q) = OT(nu_q^c, nu_c_hat)
s_c(q) = -[D_c(q) - epsilon_c]_+
```

The default scoring mode is `robust_transport_envelope`. Raw OT scores should appear only as ablations.

## 5. Optional Query Ambiguity Correction

Query competition should be described as a query-side correction, not as a main contribution. It prevents ambiguous query tokens from independently supporting multiple classes before transport.

Implementation names:

- `use_query_competition`
- `query_competition_temperature`
- `compute_query_class_affinities(...)`
- `apply_query_ambiguity_correction(...)`

Avoid presenting this as the central novelty.

## 6. Transport Backend

Unbalanced OT is an implementation backend for partial matching:

```text
transport_backend = "unbalanced_ot"
```

Balanced OT remains available for ablation:

```text
transport_backend = "balanced_ot"
```

Do not present the sink or unbalanced solver as a separate contribution. The contribution is the uncertain barycentric class object and robust class transport scoring.

## 7. Recommended Paper Text

DeepEMD-style few-shot classifiers preserve local structure by computing optimal transport between query and support images, but they still perform inference through independent query-support comparisons. We instead represent each few-shot class as an uncertain barycentric transport object: a Wasserstein barycenter of support token measures equipped with a transport dispersion radius. Classification is performed by robust transport scoring, which penalizes only the query evidence that lies outside the class transport envelope. An optional query ambiguity correction allocates query token mass across classes before transport to reduce ambiguous matches. This yields a compact class-level OT formulation for uncertainty-aware few-shot recognition.

Contributions:

1. We reformulate local OT-based few-shot recognition from pairwise support matching to class-level transport inference by representing each class as a Wasserstein barycentric token measure.

2. We introduce an uncertain barycentric class object that augments the barycenter with a class-specific transport radius estimated from support dispersion, yielding an episode-adaptive robust transport envelope.

3. We instantiate query classification with a robust transport score and an optional query-side ambiguity correction, improving robustness to noisy or ambiguous local evidence.

## 8. Ablation Plan

Main ablations:

1. `deepemd_pairwise`: pairwise query-support transport baseline.
2. `barycentric_only`: class barycenter without radius.
3. `uncertain_barycentric`: barycenter plus transport radius, no query competition.
4. `ubt_full`: uncertain barycentric class object plus robust score and optional query ambiguity correction.

Appendix ablations:

- `fixed_radius`
- `dynamic_radius`
- `balanced_ot`
- `unbalanced_ot`
- `no_beta_floor`
- `tau_sensitivity`
- `alpha_sensitivity`

The main table should read:

```text
DeepEMD pairwise
+ Barycentric class measure
+ Transport radius
+ Query ambiguity correction
```

## 9. Default Implementation Recipe

Use:

- `--model ubt_fsl`
- `--ubt_fsl_barycenter_method mixture`
- `--ubt_fsl_transport_backend unbalanced_ot`
- `--ubt_fsl_scoring robust_transport_envelope`
- `--ubt_fsl_use_query_competition true`
- `--ubt_fsl_score_scale 8.0`
- `--ubt_fsl_solver_backend native`
- moderate `--ubt_fsl_sinkhorn_epsilon`, for example `0.08` or `0.1` if costs are normalized.

Legacy scripts can continue to use `--model cbcr_fsl` and `cbcr_fsl_*` flags.
