# Compact CBCR-FSL Research Note

## 1. Diagnosis

The current exact CBCR version is too solver-heavy for training:

- POT unbalanced Sinkhorn can produce `Numerical errors at iteration 0` when the cost scale, entropy, or marginal mass becomes ill-conditioned.
- The exact class barycenter solves an additional Sinkhorn barycenter for every class in every episode.
- The uncertainty radius then solves more transport problems from each shot to the barycenter.

This makes the model look ambitious, but it weakens the paper story: reviewers may see a stack of OT machinery rather than a clean few-shot contribution.

## 2. Compact Core Claim

The defensible contribution should be:

> Few-shot local matching should compare a query against a class-level support measure with uncertainty-calibrated and class-competitive evidence, instead of independently matching every query to every support shot.

Everything else is implementation detail.

## 3. Compact Model

Use the name **CBCR-Lite** or keep **CBCR-FSL** and describe it as the compact implementation.

### Module A: Consensus Class Measure

Each support shot remains a token distribution. Token masses are estimated by cross-shot consensus inside the same class.

Instead of solving a Wasserstein barycenter, build a fixed-support class measure by a weighted mixture over the union of support tokens:

`nu_c = sum_k pi_ck sum_j a_ckj delta(v_ckj)`

This preserves the key idea, "class as distribution", but removes the expensive barycenter solver.

### Module B: Dispersion Radius

Estimate class uncertainty from shot disagreement around the class measure:

`epsilon_c = max(alpha * dispersion_c, beta / sqrt(K))`

For speed, dispersion can be computed with weighted nearest-neighbor cost from each shot to the other shots in the same class. This keeps the uncertainty story without adding K extra Sinkhorn solves, and it avoids the trivial zero distance caused by matching a shot to its own tokens.

### Module C: Competitive Query Allocation

For each query token, compute its relative evidence for all class measures and assign a soft class budget:

`r_ic = softmax_c(-d(u_i, nu_c) / tau)`

Then each class only receives the fraction of query evidence it wins. This is the most reviewer-visible novelty because it prevents ambiguous scalogram background from helping all classes equally.

### Module D: Robust Partial Transport Score

Score each class with native log-domain unbalanced Sinkhorn:

`score(q,c) = -[D_uot(q_c, nu_c) - epsilon_c]_+`

Use the native PyTorch solver by default, not POT. POT remains an optional diagnostic baseline.

## 4. What To Remove From The Main Claim

Do not sell these as separate major contributions:

- exact Wasserstein barycenter;
- unbalanced barycenter;
- sink/noise mechanism as a standalone novelty;
- many learned scalar hyperparameters;
- multiple OT variants in the main model.

They can appear as optional ablations or appendix variants. The paper should not look parameter-greedy.

## 5. Paper Story For Neurocomputing/Q1

Recommended contribution paragraph:

> We propose a compact class-barycentric robust transport classifier for few-shot partial-discharge scalogram recognition. The method first converts the support set of each class into a consensus-weighted class measure, then estimates an episode-adaptive uncertainty radius from support dispersion, and finally performs class-competitive query evidence allocation before robust partial transport scoring. Unlike pairwise local matching methods, the proposed formulation treats the few-shot class itself as the inference object and prevents ambiguous query regions from being over-explained by multiple classes.

## 6. Ablation Plan

Minimal ablations:

1. Pairwise query-support OT vs class measure.
2. Uniform support mass vs cross-shot consensus mass.
3. No radius vs dispersion radius.
4. Independent class scoring vs competitive query allocation.
5. POT exact barycenter vs compact mixture barycenter.

The most important runtime table should report:

- ms/episode;
- GPU memory;
- number of Sinkhorn solves per episode;
- accuracy and confidence interval.

## 7. Default Implementation Recipe

Use:

- `--cbcr_fsl_barycenter_method mixture`
- `--cbcr_fsl_ot_backend native`
- moderate `--cbcr_fsl_sinkhorn_epsilon`, e.g. `0.08` or `0.1` if costs are normalized;
- fewer Sinkhorn iterations, e.g. `30-50`, then only increase if validation improves.

This default removes the POT warning path and reduces the number of episode-level Sinkhorn solves.
