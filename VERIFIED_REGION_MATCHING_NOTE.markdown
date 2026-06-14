# Verified Region Matching UOT Note

Date checked: 2026-06-14

## Problem

Local descriptor OT can spend mass on accidental low-cost matches:

```text
unrelated region A looks like unrelated region B
=> token cost is low
=> UOT rewards the match through T * mass - cost
```

Pulse/region saliency asks whether a token region looks important. VRM instead
asks whether a candidate pair is trustworthy before UOT solves the transport
plan.

## Implemented VRM

The implementation is parameter-free and modifies only the ground cost:

```text
C_vrm = C * (1 + lambda * (1 - gate))
```

The v1.1 gate is a geometric soft-AND over the enabled checks:

```text
gate = geometric_mean(enabled_gate_components)
```

The first v1 implementation used a direct product. Diagnostics from
`vrm_full` showed `gate_mean ~= 0.005`, `accepted_mass_ratio = 0`, and nearly
uniform cost inflation. The geometric mean keeps the logical "all checks should
agree" behavior while avoiding gate collapse when one component has a naturally
small absolute scale.

Optional rival evidence can be enabled with `--vrm_use_rival true`. It reuses the
shared `compute_rival_discriminative_query_evidence()` helper from
`cost_evidence_marginals.py`, instead of reimplementing rival aggregation.

### Concentration Gate

For each pair `(i, j)`, compute row and column softmax confidence from the cost
matrix:

```text
row_conf(i,j) = softmax_j(-C(i,:) / tau_c)[j]
col_conf(i,j) = softmax_i(-C(:,j) / tau_c)[i]
```

Uniform confidence is subtracted before combining, so flat rows/columns produce
near-zero gates. This replaces reciprocal nearest-neighbor checks, which are too
weak when the token grid is small.

### Region Patch Gate

Each token is compared using average-pooled neighborhood features rather than
raw token-pair cost:

```text
region_q_i = avgpool(neighborhood(q_i))
region_s_j = avgpool(neighborhood(s_j))
patch_gate(i,j) = exp(-D(region_q_i, region_s_j) / tau_region)
```

This breaks the circular dependency where raw token cost is used to validate the
same raw token cost.

## Diagnostics

VRM logs the following non-accuracy signals:

- `verified/gate_mean`, `gate_min`, `gate_max`
- `verified/gate_q50`, `gate_q90`, `gate_q95`
- `verified/concentration_score_mean`
- `verified/patch_consistency_mean`
- `verified/rival_specificity_mean`
- `verified/region_vs_token_consistency_gap`
- `verified/gate_class_gap`
- `verified/cost_delta_ratio`
- `verified/accepted_mass_ratio`
- `verified/rejected_mass_ratio`
- `verified/plan_gate_mean`
- `verified/plan_gate_correlation`
- `verified/top10_gate_mass_ratio`, `top20_gate_mass_ratio`
- `verified/top10_gate_mean`, `top20_gate_mean`

Interpretation:

| Observation | Meaning |
|---|---|
| `region_vs_token_consistency_gap` high | token-level match looks good but region context rejects it |
| `plan_gate_correlation` positive | UOT mass follows verified gates |
| `rejected_mass_ratio` high | solver still spends mass on low-gate matches |
| `gate_mean` near zero | VRM may over-penalize and collapse local evidence |
| `cost_delta_ratio` very high | `vrm_lambda` is too strong |
| `top10_gate_mass_ratio` low | transport is not concentrating on the best relative VRM matches |

## Novelty Boundary

Do not claim novelty for local descriptor OT, attention/contextual matching,
neighborhood consensus, or hubness reduction individually.

Closest prior directions:

- DeepEMD: differentiable EMD over dense local descriptors for few-shot
  classification.
- SuperGlue: learned local feature matching with differentiable OT and
  non-match rejection.
- NCNet/PMNC: neighborhood consensus for dense semantic correspondence.
- Local scaling/CSLS-style ideas: reduce hubness by considering local
  neighborhood concentration.

The defensible claim is narrower:

```text
parameter-free concentration + region-context verification applied before
threshold-mass UOT scoring for few-shot PD scalogram recognition, without masks,
fixed time/frequency priors, or learned verification parameters.
```

That claim should be retained only if VRM improves clean/noise accuracy and the
diagnostics show that transported mass shifts toward high-gate correspondences.

## Experiment Suite

Use:

```bash
python run_all_experiments.py \
  --models ours_final \
  --ours_final_ablation_suite verified_region \
  --ours_final_ablation_variants all
```

Variants:

| Tag | Gates |
|---|---|
| `vrm_baseline` | Ours-Final + global residual w=0.1 |
| `vrm_concentration` | concentration only |
| `vrm_patch_region` | region-context patch only |
| `vrm_conc_patch` | concentration + region-context patch |
| `vrm_full` | concentration + region-context patch + reused rival evidence |

## Primary References

- [DeepEMD, CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhang_DeepEMD_Few-Shot_Image_Classification_With_Differentiable_Earth_Movers_Distance_and_CVPR_2020_paper.pdf)
- [SuperGlue, CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Sarlin_SuperGlue_Learning_Feature_Matching_With_Graph_Neural_Networks_CVPR_2020_paper.pdf)
- [NCNet, TPAMI](https://inria.hal.science/hal-03086922/file/ncnet_pami.pdf)
- [PatchMatch-Based Neighborhood Consensus, CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Lee_PatchMatch-Based_Neighborhood_Consensus_for_Semantic_Correspondence_CVPR_2021_paper.pdf)
- [Local and Global Scaling Reduce Hubs, JMLR 2012](https://jmlr.org/papers/v13/schnitzer12a.html)
