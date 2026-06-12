# Ours-Final Failure Probe and Utility-Contrastive Adaptive Marginals

## Scope

This work starts from the original `ours_final` path:

- local descriptor cost matrix `D`;
- unbalanced OT with the original `rho=0.8`;
- original Sinkhorn solver;
- original threshold-mass score;
- no RAE, HC-UOT, pulse, residual-aligned, or other replacement path.

The probe is observational only. Enabling it does not modify cost, marginals,
transport plan, logits, loss, or model parameters.

## Original score and the uniform-marginal problem

For each query-class-shot pair, Ours-Final scores the solved plan `P` by

`S = alpha * (T * sum(P) - sum(P * D))`

or equivalently

`S = alpha * sum(P * (T - D))`.

Therefore each transported edge has utility

`u_ij = T - D_ij`.

An edge with `D_ij < T` contributes positive utility. An edge with
`D_ij > T` contributes negative utility.

Uniform token marginals ignore this utility structure. They assign the same
capacity to:

- discriminative pulse tokens;
- background or common tokens;
- query tokens that have no positive-utility match;
- support tokens that are close to several competing classes.

The solver minimizes transport cost under the supplied marginal constraints.
It does not directly maximize the final class margin. Consequently, a
cost-minimizing plan can still spend mass on edges that are weak,
class-common, or negative under the final `T*M-C` score.

## Failure probe

Enable with:

`--enable_ours_final_failure_probe true`

The probe reads the solved plan and reports:

1. `ours_probe/negative_utility_mass_ratio`

   Fraction of transported mass placed on edges with `D > T`.
   A high value is direct evidence that the marginal constraints force mass
   into edges that reduce the original Ours-Final score.

2. `ours_probe/common_mass_ratio`

   Fraction of mass assigned to query tokens whose best and second-best class
   costs are too similar. A high value means the plan spends capacity on
   evidence shared by competing classes.

3. `ours_probe/dead_query_ratio`

   Fraction of query tokens for which a class has no edge with `D < T`.
   If this remains high after changing marginals, the main bottleneck is
   feature quality, cost calibration, or threshold calibration.

4. `ours_probe/harm_share`

   Negative utility magnitude divided by total absolute transported utility.
   This measures how much of the plan's score contribution is actively
   harmful, rather than merely uninformative.

5. `ours_probe/query_mass_entropy` and
   `ours_probe/effective_query_fraction`

   These detect an almost uniform plan and an over-collapsed plan. High
   entropy together with high harm/common mass indicates uniform allocation
   waste. Very low entropy with poor accuracy indicates collapse onto a small
   number of unreliable tokens.

6. Winner agreements

   `full_mass_winner_agreement`, `full_cost_winner_agreement`, and
   `mass_cost_winner_agreement` show whether the predicted class is mainly
   selected by transported mass or by low transported cost.

The per-class tensors are also summarized against the true label. This makes
it possible to compare the true class with the best negative class instead of
only observing an episode-wide mean.

## Novelty: UCAM

The novelty module is `UtilityContrastiveMarginals`, abbreviated UCAM.

UCAM changes only the token marginals. It preserves the original cost,
transport solver, rho, and `T*M-C` score.

### Query evidence

For each query token and class, UCAM computes a temperature-normalized soft
minimum over all shots and support tokens:

`d_c(i) = softmin_{k,j} D_{c,k,i,j}`.

It then measures class-specific advantage against the best rival:

`a_c(i) = min_{r != c} d_r(i) - d_c(i)`.

Only positive advantage is retained. This suppresses tokens that are close to
several classes.

Positive-utility acceptance is:

`q_pos(i) = sigmoid((T - min_{c,k,j} D_{c,k,i,j}) / tau)`.

The shared query evidence is:

`e_q(i) = max_c specificity(a_c(i)) * q_pos(i)`.

The query marginal is shared across classes. This is important because
class-specific query budgets would make class logits incomparable.

### Support evidence

For each class and shot, support evidence combines:

- query-token class specificity;
- positive utility under `T-D`;
- local affinity relative to the row minimum.

It is integrated under the shared query marginal to produce a class-specific
support marginal.

### Confidence-adaptive shrinkage

UCAM does not force a selective marginal when the episode is ambiguous.

`lambda_eff = lambda * confidence^p`

`m = (1 - lambda_eff) * uniform + lambda_eff * selective`.

When no rival-specific positive evidence exists, `lambda_eff` approaches zero
and the method falls back to the original uniform marginal. This limits damage
on episodes where the cost matrix cannot justify a non-uniform allocation.

## Experiment suite

Run:

```powershell
python run_all_experiments.py `
  --models ours_final `
  --ours_final_ablation_suite failure_probe
```

The suite produces three matched variants:

1. `ours_final_probe_uniform`

   Original Ours-Final uniform marginals plus the observational probe.

2. `ours_final_probe_utility_fixed`

   UCAM evidence with a fixed marginal mixing strength.

3. `ours_final_probe_utility_adaptive`

   UCAM with confidence-adaptive shrinkage to uniform.

All three variants use the same Ours-Final transport and score settings.

## Acceptance criteria

UCAM is supported by the experiment only if accuracy improves together with
mechanism-consistent changes:

- lower true-class negative-utility mass;
- lower true-class common mass;
- lower true-class harm share;
- a larger true-vs-negative utility-score gap;
- no severe entropy collapse;
- stable behavior in both 1-shot and 5-shot.

If accuracy does not improve and `dead_query_ratio` stays high, the next
research target should be the descriptor/cost/threshold subsystem, not a more
aggressive marginal.
