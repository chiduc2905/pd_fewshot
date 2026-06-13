# Dustbin-Contrastive Residual Score for Ours-Final

## Research Summary

The full dustbin OT variant borrowed the useful idea from SuperGlue/SALAD: give
local tokens an unmatched option instead of forcing every token to match a real
support token. That idea is relevant for PD scalograms because a query window
can contain background, narrowband noise, or non-diagnostic time-frequency
patches.

The weak point is that full dustbin OT also changes the solved transport plan
and the matched mass. In this few-shot setting, a wrong class can keep a small
set of easy/common background matches, reject the rest, and still improve its
relative score. That behavior is useful for retrieval and feature matching, but
it is risky for class-discriminative PD diagnosis when the positive evidence is
often sparse.

The first real 60-sample/1-shot dustbin run supports this failure analysis:

```text
accuracy = 0.745
real_mass_gap = 0.00728
real_mass_fraction_gap = 0.00910
dustbin_score = -1.01897
accepted_edge_score_mean = -0.26040
row_residual = 0.00080
```

The score identity was exact, but the augmented solver had not fully converged.
More importantly, accepted mass was nearly non-discriminative between the true
and best-negative classes. This means increasing Sinkhorn iterations may remove
the `INVALID` label, but is unlikely to solve the classification weakness.

Related external work checked:

- SuperGlue, CVPR 2020: differentiable feature matching with a dustbin row and
  column for unmatched keypoints.
- SALAD, CVPR 2024: OT aggregation with a dustbin cluster that discards
  uninformative local descriptors.
- Outlier-robust and partial OT: supports the broader idea that not every mass
  should be forcibly matched, but usually optimizes a partial/rejected transport
  objective rather than a class residual score.
- Noisy few-shot learning: confirms that few-shot classifiers need defenses
  against distracting/noisy support or query evidence, but does not give this
  ECOT threshold-mass residual design.

## New Design

The new module is DCR: Dustbin-Contrastive Residual Score.

It keeps the original Ours-Final UOT solver untouched:

```text
P = original UOT plan
score_raw = sum(P * (T - C))
```

Then it computes a dustbin-style utility gate and an episode-local rival
specificity gate:

```text
utility = T - C
dustbin_gate = sigmoid((utility - alpha) / tau)
specificity = sigmoid((class_best_utility - rival_best_utility - margin) / tau)
```

Only non-specific positive evidence is subtracted:

```text
score_dcr = score_raw - beta * sum(P * (1 - gate) * max(T - C, 0))
```

Negative evidence is preserved. This is the key change from the full dustbin
variant: DCR cannot make a wrong class look better by hiding bad matches. It
only removes part of the positive evidence that looks both rejectable and not
class-specific.

## Fit With Ours-Final

DCR is intentionally a score residual, not a new OT solver:

- It preserves the original `transport_plan`, `transport_cost`, and
  `transported_mass`.
- It is compatible only with `ours_final_score_mode=threshold_mass`.
- It uses the dustbin idea as a diagnostic rejection signal, not as the full
  classification objective.

This is a better fit for the current PD problem than full dustbin OT when the
observed failure is poor accuracy despite plausible token rejection.

## Novelty Check

Not novel by itself:

- dustbin matching,
- partial/rejected OT,
- noisy few-shot defenses,
- rival-aware class comparisons.

Potential standalone novelty for this repo:

- an ECOT/Ours-Final few-shot PD scalogram classifier that keeps the original
  UOT transport plan, then applies a dustbin-style and rival-contrastive
  residual only to positive `T-C` evidence;
- explicit diagnostics that test whether the residual removes more positive
  background evidence from best-negative classes than from true classes.

The novelty should be claimed narrowly as a PD few-shot score calibration module,
not as a new OT solver.

## Debug Metrics

Debug report:

```text
debug_dcr_<dataset>_ours_final_<samples>_<shot>shot*.txt
```

Important metrics:

- `dcr/shot_logit_delta_abs`: verifies the module is active.
- `dcr_unadjusted_pred_acc`: original UOT accuracy on the same forward pass.
- `dcr_accuracy_delta`: DCR accuracy minus unadjusted UOT accuracy.
- `dcr_fix_rate`: originally wrong predictions corrected by DCR.
- `dcr_harm_rate`: originally correct predictions broken by DCR.
- `dcr_prediction_change_rate`: fraction of predictions changed by DCR.
- `dcr/removed_positive_share`: average fraction of positive evidence removed.
- `dcr_removed_positive_true_distance`
- `dcr_removed_positive_best_negative_distance`
- `dcr_removed_positive_distance_gap`
- `dcr_removed_positive_share_distance_gap`
- `dcr_retained_positive_gap`

Acceptance signal beyond accuracy:

```text
dcr_removed_positive_distance_gap > 0
```

This means the best negative class loses more non-specific positive evidence
than the true class. If accuracy improves without this gap, the change is likely
not doing what it was designed to do.

## Command

Use this isolated runner variant first:

```bash
python run_all_experiments.py \
  --project dustbin \
  --seeds 42,43,44 \
  --gpu_id 0 \
  --test_protocol clean \
  --extra_test_protocols noise \
  --noise_test_root /workspace/pd_fewshot/scalogram_27_1_pd_noise_benchmark_test_moderate \
  --noise_test_splits test_snr15db_rf_1_15mhz,test_snr10db_rf_1_15mhz,test_snr5db_rf_1_15mhz \
  --dataset_path /workspace/pd_fewshot/scalogram_27_1 \
  --dataset_name knee_aug_split \
  --fewshot_backbone resnet12 \
  --models ours_final \
  --ours_final_ablation_suite objective_score \
  --ours_final_ablation_variants score_dcr
```

Small grid after the first result:

```text
dcr_beta: 0.25, 0.50, 0.75
dcr_tau: 0.15, 0.25, 0.40
dcr_margin: 0.00, 0.05
```

Keep `dcr_min_gate=0.05` unless the debug report shows excessive suppression.
