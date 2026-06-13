# EAR-UOT for Ours-Final

## Failure Being Addressed

The current 60-sample run can obtain a positive global-score gap while the
local UOT score gap remains negative. Validation transport also places roughly
0.92-0.98 of its mass on query tokens classified as common across candidate
classes. Cost gates and post-hoc plan filtering have therefore not made the
local plan reliably PD-specific.

Standard Ours-Final uses one query relaxation and one support relaxation:

```text
min_P <P,C> + eps H(P)
      + tau_q KL(P1 | a)
      + tau_s KL(P^T1 | b)
```

The same `tau` is applied to PD evidence, background, and noise. UOT can destroy
mass, but it has no token-level instruction about where mass destruction is
safe.

## Proposed Mechanism

EAR-UOT means Evidence-Adaptive Relaxation UOT. It keeps:

- the original local descriptors;
- the original ground cost;
- uniform target marginals;
- `rho=0.8`;
- the original `T*M-C` score.

It changes the marginal penalties to token-wise fields:

```text
min_P <P,C> + eps H(P)
      + sum_i tau_q(i,c) KL((P1)_i | a_i)
      + sum_j tau_s(j,c) KL((P^T1)_j | b_j)
```

High `tau` makes destruction of a reliable token expensive. Low `tau` allows
common or non-matchable tokens to lose mass.

Reliability is analytic and episode-local:

1. Soft-min candidate cost is compared with rival-class costs.
2. Candidate specificity is multiplied by positive `T-C` matchability.
3. A local spatial average requires neighboring scalogram tokens to support the
   same decision.
4. Support reliability is induced by reliable query-to-support edges.
5. Reliability is mapped to `[tau_min, tau_max]`.

The adaptive Sinkhorn updates are:

```text
rho_q(i) = tau_q(i) / (tau_q(i) + eps)
rho_s(j) = tau_s(j) / (tau_s(j) + eps)

u_i = (a_i / (Kv)_i) ^ rho_q(i)
v_j = (b_j / (K^T u)_j) ^ rho_s(j)
```

## Research and Novelty Boundary

Related foundations:

- Chizat et al., *Scaling Algorithms for Unbalanced Transport Problems*:
  entropic UOT and pointwise scaling algorithms
  (<https://arxiv.org/abs/1607.05816>).
- Sejourne et al., *Sinkhorn Divergences for Unbalanced Optimal Transport*:
  UOT robustness and pointwise contractive Sinkhorn updates
  (<https://arxiv.org/abs/1910.12958>).
- DeepEMD: dense few-shot matching with cross-reference descriptor weights,
  but balanced EMD rather than class-conditional token-wise mass relaxation
  (<https://openaccess.thecvf.com/content_CVPR_2020/html/Zhang_DeepEMD_Few-Shot_Image_Classification_With_Differentiable_Earth_Movers_Distance_and_CVPR_2020_paper.html>).
- Chapel et al., *Unbalanced Optimal Transport through Non-negative Penalized
  Linear Regression*: weighted marginal penalties in UOT are established
  (<https://arxiv.org/abs/2106.04145>).
- Lacombe, *Homogeneous Unbalanced Regularized Optimal Transport*: spatially
  varying divergence in an UOT setting is established
  (<https://arxiv.org/abs/2201.02082>).
- Pardini and Papagiannouli, *Cost-Regularized Unbalanced Optimal Transport*
  (2025/2026): adapts the ground cost while allowing mass variation
  (<https://arxiv.org/abs/2511.19075>).

Not novel alone:

- UOT;
- Sinkhorn scaling;
- weighted or spatially varying marginal penalties;
- rival-aware descriptor weighting;
- spatial smoothing.

Candidate standalone novelty:

- episode-conditioned and class-conditional token-wise UOT KL relaxation for
  few-shot PD scalogram matching;
- the joint reliability construction from candidate-vs-rival specificity,
  positive threshold utility, and scalogram-neighborhood consensus;
- query evidence inducing the support-side relaxation field through reliable
  cross-image edges;
- unchanged uniform targets and unchanged `T*M-C` score, making the effect
  attributable to the mass-destruction field;
- plan-level diagnostics and reliability heatmaps for PD evidence auditing.

This is a candidate novelty claim, not an exhaustive patent or literature
clearance. Keep the claim only if it improves clean/noise results and transport
evidence over Ours-Final and DeepEMD.

## Debug Outputs

Important scalar metrics:

```text
ear_uot/query_tau_mean
ear_uot/support_tau_mean
ear_uot/query_tau_std
ear_uot/support_tau_std
ear_uot/mass_reliability_correlation
ear_uot/reliable_mass_fraction
ear_uot/high_reliability_mass_fraction
ear_uot/low_reliability_destroyed_fraction
```

True-vs-negative metrics:

```text
ear_uot_reliable_mass_gap
ear_uot_high_reliability_mass_fraction_gap
ear_uot_low_reliability_mass_distance_gap
ear_uot_low_reliability_destroyed_distance_gap
```

The UOT evidence exporter also consumes `ear_uot_query_reliability` and
`ear_uot_support_reliability`, so the evidence heatmap can be compared directly
with the solved mass overlay.

## Ablation

```powershell
python run_all_experiments.py `
  --models ours_final `
  --ours_final_ablation_suite adaptive_relaxation `
  --ours_final_ablation_variants all `
  --mode_id 1 `
  --shot_num 1 `
  --seeds 42,43,44 `
  --test_protocol clean `
  --extra_test_protocols noise `
  --noise_test_splits auto `
  --export_uot_evidence_figure true `
  --uot_evidence_artifacts aux_only
```

The suite contains:

1. Scalar UOT baseline.
2. EAR solver with constant `tau=0.5`, which must reproduce the baseline.
3. Token-wise EAR without spatial consensus.
4. Full EAR-UOT.

## Acceptance Criteria

Do not accept EAR-UOT from accuracy alone. Require:

- clean and all three noise tests improve or remain statistically stable;
- `local_score_gap` becomes less negative or positive;
- positive `ear_uot_reliable_mass_gap`;
- positive mass-reliability correlation;
- low-reliability destruction rises without collapsing transported mass;
- mass overlay and reliability heatmap overlap PD-relevant regions;
- gains persist across seeds and both 1-shot and 5-shot;
- scalar-control matches the original UOT within numerical tolerance.

Reject the mechanism if only the global residual improves accuracy, if the
local plan remains common-region dominated, or if the noise gains trade away
clean performance.
