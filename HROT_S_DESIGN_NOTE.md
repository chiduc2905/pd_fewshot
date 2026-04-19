# HROT-FSL Variant S: Contribution-Weighted Geodesic UOT

## Claim

HROT-FSL-S extends HROT-FSL-H with two contribution mechanisms that are native to unbalanced optimal transport:

1. token contribution priors for UOT marginals;
2. geodesic support-shot contribution pooling.

The goal is to fix the two main limitations of H:

- H uses uniform token marginals, so every spatial token starts with equal mass.
- H averages support-shot logits, so every support shot contributes equally to the class logit.

Variant S keeps H's clean threshold-calibrated UOT score and adds contribution weights without adding Q/R-specific noise sinks, adaptive thresholds, or structure costs.

## Token Contribution Prior

For every query-class-shot triple `(q,c,k)`, H predicts a shot transport budget:

```text
rho_qck = EAM(g_qck)
```

H then uses uniform marginals:

```text
a_i = rho_qck / L
b_j = rho_qck / L
```

S replaces these with geodesic contribution priors:

```text
p_q(i | c,k) = softmax_i( - d_H(z_qi, mu_ck) / tau_token )
p_s(j | q)   = softmax_j( - d_H(z_ckj, mu_q) / tau_token )
```

and:

```text
a_i = rho_qck * p_q(i | c,k)
b_j = rho_qck * p_s(j | q)
```

This is not a hard token selector. It is a prior over where reliable transport mass should start. The UOT solver can still move less mass than the supplied marginals because the marginal constraints are KL-relaxed:

```text
min_P <P,C>
    + tau_q KL(P 1 || a)
    + tau_s KL(P^T 1 || b)
    - eps H(P)
```

So the theoretical role is:

- token contribution prior says which local evidence should be trusted first;
- UOT decides how much of that prior should actually be transported.

This is a better fit for partial PD scalograms than balanced OT, because discriminative discharge evidence may occupy only a subset of time-frequency tokens.

## Shot Contribution Pooling

H computes shot logits:

```text
l_qck = -s D_qck + s T m_qck
```

then averages:

```text
l_qc = mean_k l_qck
```

S uses the same geodesic EAM feature vector:

```text
g_qck = [
  d_H(mu_q, mu_ck),
  d_H(mu_ck, mu_c),
  var_H(q),
  var_H(c,k)
]
```

to compute shot contribution weights:

```text
u_qck = MLP_w( standardize_k(g_qck) )
alpha_qck = softmax_k( u_qck / tau_shot )
omega_qck = (1 - lambda_shot) / K + lambda_shot * alpha_qck
```

The class score becomes:

```text
l_qc = sum_k omega_qck * l_qck
```

The initial scorer weights are interpretable:

```text
[-query_shot_distance, -shot_class_spread, -query_variance, -support_variance]
```

This biases the model toward shots that are close to the query, consistent with their class, and less dispersed. The uniform mixture keeps S close to H at initialization and lets training learn how much to trust shot selection.

## Why This Is UOT-Specific

In balanced OT, changing token weights only changes which mass must be transported. The solver still has to move all supplied mass.

In S, token and shot contribution are coupled with UOT:

- `rho_qck` controls how much evidence the query-shot pair is allowed to explain;
- token priors distribute that budget across local evidence;
- UOT can reject unreliable or unmatched mass through relaxed marginals;
- threshold scoring rewards transported mass only when its average cost is below `T`.

This gives a clean decision principle:

```text
reliable local evidence = high contribution prior + low UOT cost + transported mass
```

## Difference From Nearby Variants

```text
H: geodesic rho + threshold score, but uniform tokens and mean shots.
P: token contribution UOT, but mean shots.
Q: probe-UOT reliability, adaptive threshold, noise sink, robust shot pooling.
R: Q plus posterior structure cost.
S: H plus token contribution priors and geodesic shot contribution pooling only.
```

S is therefore the clean ablation for the two contribution findings. Q/R remain stronger exploratory variants, while S isolates the contribution-weighted UOT idea.

## Difference From DeepEMD

DeepEMD computes cross-reference token weights and then solves a mostly balanced EMD-style transport problem.

S differs in three ways:

1. token weights are geodesic contribution priors inside UOT marginals;
2. total transported mass is adaptive and can be reduced by UOT;
3. support-shot contribution is learned per query-class-shot before class aggregation.

So S should be described as contribution-weighted unbalanced transport, not as a copy of DeepEMD token weighting.

## Implementation Hooks

- `net/hrot_fsl.py`
  - `variant == "S"` enables `uses_hyperbolic_token_attention` and `uses_geodesic_shot_pooling`.
  - `_compute_hyperbolic_token_marginals(...)` builds non-uniform token masses.
  - `_pool_shot_scores(...)` builds shot contribution weights from geodesic features.
  - `shot_logits = -score_scale * cost + score_scale * threshold * transported_mass`.
  - `logits = sum_k shot_pool_weights * shot_logits`.

- `tests/test_hrot_fsl.py`
  - `test_hrot_fsl_variant_s_adds_token_and_shot_contribution_uot` checks the score formula, token mass sums, and non-uniform token/shot weights.
  - `test_hrot_fsl_variant_s_backpropagates_contribution_parameters` checks gradient flow through token temperature and shot contribution parameters.
