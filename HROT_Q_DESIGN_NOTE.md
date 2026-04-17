# HROT-Q Design Note

## One-Sentence Summary

**HROT-Q** is a noise-calibrated hierarchical unbalanced optimal transport few-shot classifier that upgrades HROT-H by replacing uniform token marginals with transport-calibrated token reliability, same-class support consensus, an explicit noise sink, and robust shot-level evidence pooling.

The implementation lives in:

- `net/hrot_fsl.py`
- variant flag: `--hrot_variant Q`

HROT-Q is intended for partial-discharge scalogram classification where discriminative evidence is local, incomplete, and often surrounded by background noise.

## Why Q Exists

HROT-H already improves over plain balanced OT by using:

- shot-decomposed matching;
- unbalanced OT;
- geodesic episode-adaptive transported mass `rho`;
- threshold-calibrated transported-mass scoring.

However, HROT-H still uses uniform token marginals:

```text
a_i = rho / T
b_j = rho / T
```

This means every spatial token in a scalogram contributes equally before UOT relaxes the transport. For noisy scalograms, this is too weak: UOT can reduce total transported mass, but it is not directly told which tokens are likely foreground evidence and which are background.

DeepEMD partially solves this by using cross-reference token weights. HROT-Q deliberately does **not** copy that mechanism. Instead, it estimates token reliability from the behavior of a preliminary UOT plan and from support-set consistency.

## Core Difference From DeepEMD

DeepEMD token weighting asks:

```text
Does this token look similar to the global prototype/class summary?
```

HROT-Q token weighting asks:

```text
Was this token actually used by a low-cost, concentrated transport explanation,
and is the support token consistent across same-class support shots?
```

So HROT-Q is transport-posterior calibrated, not prototype cross-reference calibrated.

## High-Level Pipeline

For every query `q`, class `c`, and support shot `k`, HROT-Q solves a shot-level matching problem.

```text
query tokens:        X_q       in R^{Tq x D}
support-shot tokens: X_{c,k}   in R^{Ts x D}
cost matrix:         C_{q,c,k} in R^{Tq x Ts}
```

The full HROT-Q path is:

1. Encode query and support images into Euclidean and hyperbolic tokens.
2. Predict shot-level transported mass `rho_{q,c,k}` using geodesic EAM.
3. Run a uniform-marginal UOT probe.
4. Extract token evidence features from the probe plan.
5. Compute query/support token reliability distributions.
6. Add support cross-shot consensus to support-token reliability.
7. Scale token reliability by `rho` to form adaptive marginals.
8. Append a learnable noise sink token.
9. Solve final UOT with adaptive marginals and noise sink.
10. Score only real-token transport.
11. Pool shot logits with robust evidence pooling.

## Base H Components Kept

HROT-Q keeps the strongest parts of HROT-H.

### Geodesic EAM

For each query-class-shot triple, HROT-Q builds:

```text
g_{q,c,k} = [
  d_H(mu_q, mu_{c,k}),
  d_H(mu_{c,k}, mu_c),
  var_H(q),
  var_H(c,k)
]
```

Then:

```text
rho_{q,c,k} = EAM(g_{q,c,k})
```

This remains the shot-level transported-mass budget.

### Threshold-Calibrated Score

For each shot:

```text
logit_{q,c,k}
= -s * D_{q,c,k} + s * T * m_{q,c,k}
```

where:

- `s` is `score_scale`;
- `T` is the learnable transport-cost threshold;
- `D` is real-token transported cost;
- `m` is real-token transported mass.

Equivalently:

```text
logit = s * m * (T - D / m)
```

So transported mass is rewarded only when average transported cost is below the learned threshold.

## New Component 1: Probe-UOT Token Evidence

HROT-Q first runs a no-gradient UOT probe with uniform token marginals:

```text
Pi_probe = UOT(C, uniform rho)
```

From this probe plan, it computes per-query-token features:

```text
row_mass_i      = sum_j Pi_probe[i,j]
row_cost_i      = sum_j Pi_probe[i,j] C[i,j] / row_mass_i
row_entropy_i   = entropy(Pi_probe[i,:] / row_mass_i)
row_min_cost_i  = min_j C[i,j]
```

and per-support-token features:

```text
col_mass_j      = sum_i Pi_probe[i,j]
col_cost_j      = sum_i Pi_probe[i,j] C[i,j] / col_mass_j
col_entropy_j   = entropy(Pi_probe[:,j] / col_mass_j)
col_min_cost_j  = min_i C[i,j]
```

Interpretation:

- high mass means the token participates in an explanation;
- low cost means the token matches real evidence cheaply;
- low entropy means the token has a concentrated correspondence;
- low min cost means at least one plausible local match exists.

These features are standardized over tokens before scoring.

Code:

- `HROTFSL._build_probe_token_features(...)`

## New Component 2: Learnable Token Reliability

The probe features are converted into token reliability logits:

```text
r_q(i) = w_q^T phi_q(i)
r_s(j) = w_s^T phi_s(j)
```

The initial weights are intentionally interpretable:

```text
[+mass, -cost, -entropy, -min_cost]
```

Then:

```text
attn_q = softmax(r_q / tau_token)
attn_s = softmax(r_s / tau_token)
```

To avoid destabilizing early training, HROT-Q mixes reliability weights with uniform weights:

```text
w_q = (1 - lambda_token) * uniform + lambda_token * attn_q
w_s = (1 - lambda_token) * uniform + lambda_token * attn_s
```

`lambda_token` is learnable and initialized small, so the model starts close to HROT-H and gradually learns token selectivity.

The final adaptive real-token marginals are:

```text
a_i = rho_{q,c,k} * w_q(i)
b_j = rho_{q,c,k} * w_s(j)
```

Code:

- `HROTFSL._compute_noise_calibrated_token_marginals(...)`

## New Component 3: Support Cross-Shot Consensus

For `K > 1`, support tokens that are consistent across same-class support shots should be more reliable than one-shot artifacts.

For each support token in shot `k`, HROT-Q computes its nearest-token cost to every other same-class shot, averages the nearest costs, and turns this into a standardized consensus score:

```text
consensus(c,k,j)
= standardize_j(
    - mean_{k' != k} min_{j'} ||x_{c,k,j} - x_{c,k',j'}||^2 / tau_consensus
  )
```

This consensus score is added to support-token reliability logits:

```text
r_s(j) = r_s(j) + lambda_consensus * consensus(c,k,j)
```

For `K = 1`, consensus is zero. Therefore, the method naturally degenerates to probe-UOT token reliability without cross-shot support consensus.

Code:

- `HROTFSL._build_support_consensus_scores(...)`

## New Component 4: Explicit Noise Sink

HROT-Q appends a learnable dustbin/noise-sink token to both query and support sides.

The augmented cost matrix is:

```text
C_aug =
[
  C_real       C_to_sink
  C_from_sink 0
]
```

where `C_to_sink` and `C_from_sink` are controlled by a learnable positive scalar:

```text
noise_sink_cost > 0
```

The sink mass is:

```text
sink_mass = 1 - rho
```

This gives the transport solver a structured way to absorb unmatched/noisy tokens rather than forcing all real tokens to match real tokens.

After solving the final UOT on the augmented matrix:

```text
Pi_aug = UOT(C_aug, a_aug, b_aug)
```

HROT-Q keeps only the real-token block for scoring:

```text
Pi_real = Pi_aug[:-1, :-1]
D = <Pi_real, C_real>
m = sum(Pi_real)
```

The sink allocations are exposed as diagnostics:

- `noise_sink_query_mass`
- `noise_sink_support_mass`
- `noise_sink_self_mass`

Code:

- `HROTFSL._append_noise_sink(...)`

## New Component 5: Robust Shot Pooling

HROT-H averages shot logits:

```text
logit_{q,c} = mean_k logit_{q,c,k}
```

This is vulnerable when one or more support shots are noisy or outliers.

HROT-Q uses learnable robust shot pooling:

```text
alpha = softmax(normalize(logit_{q,c,k}) / tau_shot)
```

with a uniform mixture:

```text
shot_weight
= (1 - lambda_shot) * uniform + lambda_shot * alpha
```

Then:

```text
logit_{q,c} = sum_k shot_weight_{q,c,k} * logit_{q,c,k}
```

Like token reliability, `lambda_shot` is initialized small so the model starts close to H's mean pooling.

Code:

- `HROTFSL._pool_shot_scores(...)`

## Forward Branch

The HROT-Q branch is selected by:

```python
self.uses_noise_calibrated_transport = variant == "Q"
```

The main branch is in `_forward_episode(...)`:

```text
if self.uses_noise_calibrated_transport:
    ...
```

The branch performs:

```text
flat_cost
geodesic EAM rho
probe-UOT token reliability
support consensus
adaptive token marginals
noise sink augmentation
final UOT
real-token score
robust shot pooling
diagnostics
```

## Diagnostics Returned

When `return_aux=True`, HROT-Q returns the usual HROT diagnostics plus Q-specific tensors:

```text
query_token_mass
support_token_mass
probe_query_reliability
probe_support_reliability
support_consensus
shot_logits
shot_pool_weights
noise_sink_query_mass
noise_sink_support_mass
noise_sink_self_mass
token_temperature
token_reliability_mix
support_consensus_mix
noise_sink_cost
shot_pool_temperature
shot_pool_mix
```

These are useful for visualizing:

- which query regions were treated as evidence;
- which support regions were treated as reliable;
- which regions were sent to the noise sink;
- which support shots were trusted.

## Learnable Parameters Added by Q

HROT-Q adds:

```text
query_reliability_weights
support_reliability_weights
raw_token_temperature
raw_token_reliability_mix
raw_support_consensus_mix
raw_consensus_temperature
raw_noise_sink_cost
raw_shot_pool_temperature
raw_shot_pool_mix
```

The mix parameters are sigmoid-gated and initialized low. This makes Q start near HROT-H rather than changing the inference rule abruptly at initialization.

## Expected Benefits

HROT-Q should help especially when:

- background dominates the scalogram;
- discriminative discharge evidence is local;
- support shots contain outlier regions;
- some support shots are lower quality than others;
- 5-shot performance is limited by noisy support aggregation.

Compared with HROT-H:

```text
HROT-H:
  shot-level reliability only
  uniform token mass
  mean shot pooling

HROT-Q:
  shot-level reliability
  transport-calibrated token reliability
  support-set consensus
  explicit noise sink
  robust shot pooling
```

Compared with DeepEMD:

```text
DeepEMD:
  token importance from cross-reference dot product
  balanced/full EMD
  prototype/SFC support refinement

HROT-Q:
  token reliability from UOT posterior evidence
  unbalanced OT
  support cross-shot consensus
  explicit noise sink
  robust shot aggregation
```

## Suggested Ablations

Use HROT-H as the clean baseline and Q as the full model.

Recommended ablation table:

```text
H
H + robust shot pooling
H + probe-UOT token reliability
H + support consensus
H + noise sink
Q without support consensus
Q without noise sink
Q without robust shot pooling
Q full
P hyperbolic token attention
DeepEMD
```

The most important comparison is:

```text
P vs Q
```

because P is closer to DeepEMD-style token attention, while Q uses transport-calibrated token reliability.

## Main Claims To Validate

1. Q should outperform H when background noise is meaningful.
2. Q should outperform or match P if transport-posterior token reliability is better than geodesic/prototype-like token attention.
3. Q should improve 5-shot robustness due to support consensus and robust shot pooling.
4. Noise-sink mass should correlate with visually noisy/background regions.
5. Shot-pool weights should downweight outlier support shots.

## Risks And Weak Points

Potential issues an LLM reviewer should inspect:

1. The UOT probe is currently no-gradient, so token reliability features do not backpropagate through the probe solver. This is intentional for stability but may limit end-to-end adaptivity.
2. The final token reliability weights are simple linear weights over probe features, not a larger MLP. This improves interpretability but may underfit.
3. The noise sink uses a single global learnable cost, not class- or episode-conditioned sink cost.
4. Robust shot pooling uses shot logits as evidence. If early logits are poorly calibrated, it may need slow warmup or stronger uniform mixing.
5. Support consensus uses nearest-token agreement across shots, which is useful for K > 1 but inactive at 1-shot.
6. The final score ignores sink transport cost and scores only real-token transport. This is deliberate, but one could argue for an explicit sink penalty in the logit.

## Suggested Reviewer Questions

Ask an LLM or human reviewer:

1. Is HROT-Q sufficiently distinct from DeepEMD cross-reference weighting?
2. Is the probe-UOT token reliability mechanism theoretically coherent?
3. Should the probe plan be detached, partially detached, or differentiable?
4. Should the noise sink contribute an explicit penalty to the final logit?
5. Are the diagnostics enough to prove that Q handles background noise?
6. Which ablation is essential to support the novelty claim?
7. Could robust shot pooling collapse to selecting one lucky shot, and how should that be regularized?

## Minimal Run Command

```bash
python run_all_experiments.py \
  --project variant_best \
  --dataset_path /workspace/pd_fewshot/scalogram_27_1 \
  --dataset_name scalogram_27_1 \
  --models hrot_fsl \
  --hrot_variant Q
```

For a single 5-shot all-sample run:

```bash
python run_all_experiments.py \
  --project variant_best \
  --dataset_path /workspace/pd_fewshot/scalogram_27_1 \
  --dataset_name scalogram_27_1 \
  --models hrot_fsl \
  --hrot_variant Q \
  --shot_num 5 \
  --mode_id 4
```

