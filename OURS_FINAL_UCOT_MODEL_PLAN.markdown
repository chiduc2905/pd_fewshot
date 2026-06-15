# Ours-Final UCOT Model Implementation Plan

Date: 2026-06-15

## Goal

Implement a new standalone model, not a patched ablation flag:

`ours_final_ucot`

Working name:

`Utility-Calibrated Ours-Final UOT` or `UCOT-Ours`

This model starts from the accepted Ours-Final decision recipe:

- Ours-Final local descriptor cost.
- Fixed-budget rho=0.8 KL-UOT.
- Support-strict relaxation: `tau_q=0.5`, `tau_c=0.8`.
- Global prototype residual head with `weight=0.1`.

Then it redesigns the local transport branch so threshold calibration,
token quotas, transport, and scoring all use one shared utility:

`U_ij = T_calibrated - D_ij`

Do not implement this by merely enabling the old
`--ours_final_marginal_mode score_aligned` path. That path is useful as
prior code context, but the new model must own a coherent flow.

## Why A New Model Is Needed

Recent audit file:

`audit_ours_final_knee_aug_split_ours_final_60samples_1shot_seed42_ours_final_global_res_w0p1.txt`

Key facts:

- `pred_acc = 0.805000`
- `local_pred_acc = 0.803333`
- `global_pred_acc = 0.785000`
- `global_vs_local_accuracy_delta = 0.001667`
- `audit_threshold = 0.082146`
- `audit_true_avg_cost = 0.364781`
- `audit_true_avg_cost_below_threshold = 0.001667`
- `ours_probe/negative_utility_mass_ratio = 0.971481`
- `ours_probe/dead_query_ratio = 0.948100`
- `ours_probe/harm_share = 0.996485`

Interpretation:

- Global residual is not the primary problem; it only slightly improves local.
- Support strictness alone is not enough.
- The local cost matrix has signal: low-cost/cost-per-mass winner accuracy is
  around 0.80.
- The bottleneck is that `T` is far below useful transported costs, so
  `T*M-C` wins mostly by being less negative, not by positive evidence.

Therefore the next model must fix the local utility calibration and the mass
allocation that follows from it.

## Novelty Position

Do not claim novelty for these pieces alone:

- OT/EMD over local descriptors for few-shot classification. DeepEMD already
  formalizes few-shot classification as dense-region EMD and uses a
  cross-reference weighting mechanism.
- KL-relaxed UOT. This is established optimal transport methodology.
- Context-aware or learned marginals in fine-grained alignment. Related work
  already dynamically assigns matching quotas.

Defensible novelty claim:

> A utility-calibrated fixed-budget UOT model for few-shot partial-discharge
> scalograms, where a single episode-calibrated `T-D` utility controls
> threshold calibration, query/support transport quotas, KL-UOT evidence, and
> final `T*M-C` scoring, with asymmetric query/support KL relaxation and a
> small global residual head.

This must remain a measured architecture-level claim, not a claim that
"weighted marginals" are new.

References to cite in notes/paper:

- DeepEMD, CVPR 2020:
  https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhang_DeepEMD_Few-Shot_Image_Classification_With_Differentiable_Earth_Movers_Distance_and_CVPR_2020_paper.pdf
- On Unbalanced Optimal Transport, JMLR 2023:
  https://www.jmlr.org/papers/v24/22-1158.html
- QC-Align context-aware marginals, CVPR 2026:
  https://openaccess.thecvf.com/content/CVPR2026/html/Li_Quota-Calibrated_Fine-Grained_Alignment_with_Context-Aware_Marginals_for_Text-based_Person_Retrieval_CVPR_2026_paper.html

## Model Contract

New CLI model name:

`--model ours_final_ucot`

Default behavior must be equivalent to:

```text
ours_final
+ enable_global_residual_score=true
+ global_residual_mode=residual
+ global_residual_weight=0.1
+ hrot_tau_q=0.5
+ hrot_tau_c=0.8
+ hrot_ecot_rho_bank=0.8
+ hrot_ecot_base_rho=0.8
+ hrot_fixed_mass=0.8
+ hrot_ecot_transport_mode=unbalanced
+ ours_final_score_mode=threshold_mass
+ UCOT local utility calibration enabled
```

The model should not require users to pass global residual or tau profile flags.
Those are part of the standalone architecture.

## Core Flow

Current Ours-Final flow:

```text
tokens -> cost D -> uniform query/support quotas -> UOT plan P
       -> score = scale * (T * mass(P) - cost(P))
       -> local logits + 0.1 * global residual
```

New UCOT flow:

```text
tokens -> cost D
       -> episode-calibrated threshold T_cal
       -> utility U = T_cal - D
       -> utility-specific query/support quotas
       -> support-strict UOT plan P
       -> score = scale * sum(P * U)
       -> local logits + 0.1 * global residual
```

Important constraint:

The same `T_cal` must be used for:

- positive-utility evidence,
- query quota,
- support quota,
- final threshold-mass score,
- audit/probe diagnostics.

This avoids a stitched-together design where marginals optimize one objective
and logits score another objective.

## Threshold Calibration

Problem:

Current `transport_cost_threshold` is an absolute learned scalar. In the audit
it is `0.082`, while true transported average cost is around `0.365`.

Implement a scalar episode threshold first. Do not start with per-class or
per-query thresholds; those make logits harder to compare and are unnecessary
for the first model.

Recommended formula:

```python
cost_view = flat_cost.detach().reshape(Nq, Way, Shot, Lq, Ls)
best_edge_per_query_token = cost_view.amin(dim=(1, 2, 4))  # (Nq, Lq)
episode_threshold = quantile(best_edge_per_query_token.flatten(), q)
base_threshold = self.transport_cost_threshold.detach()
T_cal = (1 - mix) * base_threshold + mix * episode_threshold
T_cal = T_cal.clamp_min(self.eps)
```

Initial defaults:

```text
ucot_threshold_quantile = 0.70
ucot_threshold_mix = 1.0
ucot_threshold_detach = true
```

Rationale:

- It is label-free.
- It calibrates `T` to the current episode's feature/cost scale.
- It avoids directly using true class labels.
- It fixes the audit failure mode before marginal selection.

Guardrail:

If `ucot/positive_edge_rate` becomes too high and mass winner accuracy
dominates low-cost winner accuracy, lower the quantile to `0.55`.

## Utility-Calibrated Quotas

Compute quotas from the calibrated utility, not from a separate MLP.

Use normalized temperature:

```python
cost_scale = flat_cost.detach().std(dim=(1, 2, 3), keepdim=True).clamp_min(eps)
temp = ucot_temperature * cost_scale
utility = (T_cal - cost_view) / temp_5d
positive = sigmoid(utility)
```

Class-specific query evidence:

```python
class_token_cost = softmin over shot/support for each class and query token
rival_cost = min class_token_cost over other classes
rival_advantage = rival_cost - class_token_cost
specificity = sigmoid((rival_advantage - ucot_rival_margin) / temp_query)
```

Shared query quota:

```python
query_raw_i = max_c specificity[c, i] * max_{c,k,j} positive[c,k,i,j]
query_selective = normalize(query_raw, uniform fallback)
query_prob = (1 - effective_mix) * uniform + effective_mix * query_selective
```

Support quota:

```python
row_affinity = exp(-(D - row_min(D)) / temp)
pair_evidence = specificity[:, :, None, :, None] * positive * row_affinity
support_raw[c,k,j] = sum_i query_prob[i] * pair_evidence[c,k,i,j]
support_selective = normalize(support_raw, uniform fallback)
support_prob = (1 - effective_mix) * uniform + effective_mix * support_selective
```

Initial defaults:

```text
ucot_temperature = 0.25
ucot_marginal_mix = 0.65
ucot_adaptive_mix = true
ucot_confidence_power = 1.0
ucot_uniform_floor = 0.05
ucot_rival_margin = 0.0
```

Effective mix:

```python
confidence = topk_mean(query_raw, k=max(1, Lq // 8))
effective_mix = ucot_marginal_mix * confidence.pow(ucot_confidence_power)
```

Do not allow complete collapse:

```python
query_prob = (1 - floor) * query_prob + floor * uniform
support_prob = (1 - floor) * support_prob + floor * uniform
```

## Transport And Score

Pass `query_prob` and `support_prob` into the existing UOT path as target
probabilities. Let the existing code scale them by rho.

Keep:

```text
rho = 0.8
tau_q = 0.5
tau_c = 0.8
```

Score:

```python
shot_score = score_scale * (T_cal * transported_mass - transported_cost)
```

This is still the Ours-Final `T*M-C` score, but now `T` is calibrated to the
episode and the plan is encouraged to use tokens with positive, specific
utility.

## Files To Edit

### `net/model_factory.py`

Add a new model set:

```python
OURS_FINAL_UCOT_MODEL_NAMES = frozenset({"ours_final_ucot"})
```

Include it in:

```python
OURS_FINAL_MODEL_NAMES
OURS_ENTRYPOINT_MODEL_NAMES
M2_LIKE_MODEL_NAMES
HROT_FSL_MODEL_NAMES
```

Add registry metadata near `ours_final`:

```python
"ours_final_ucot": {
    "display_name": "Ours-Final-UCOT",
    "paper_name": "Utility-Calibrated Ours-Final UOT",
    "architecture": "...",
    "metric": "Utility-Calibrated Threshold-Mass UOT",
}
```

Load class:

```python
_load_symbol("net.ours", "OursFinalUCOT")
```

when `args.model in OURS_FINAL_UCOT_MODEL_NAMES`.

Set defaults in factory for UCOT:

```python
tau_q = 0.5
tau_c = 0.8
enable_global_residual_score = True
global_residual_mode = "residual"
global_residual_weight = 0.1
enable_ours_final_failure_probe = True for evaluation configs if feasible
```

Update all current hard checks that say `args.model != "ours_final"` for
Ours-Final-only flags. They should use:

```python
args.model in OURS_FINAL_MODEL_NAMES
```

unless the flag truly only applies to canonical `ours_final`.

### `net/ours.py`

Add class:

```python
class OursFinalUCOT(OursFinalM2):
    """Standalone utility-calibrated Ours-Final with global residual and support-strict UOT."""
```

This class should inject defaults before `super().__init__`:

```python
kwargs.setdefault("enable_global_residual_score", True)
kwargs.setdefault("global_residual_mode", "residual")
kwargs.setdefault("global_residual_weight", 0.1)
kwargs.setdefault("tau_q", 0.5)
kwargs.setdefault("tau_c", 0.8)
kwargs.setdefault("ecot_enable_egsm", False)
kwargs.setdefault("ecot_m2_ablate_threshold_mass", False)
kwargs.setdefault("ecot_m2_cost_per_mass_score", False)
kwargs.setdefault("ours_final_score_mode", "threshold_mass")
kwargs.setdefault("enable_ucot_calibration", True)
```

Add config fields to `OursM2.__init__` or the subclass:

```text
enable_ucot_calibration
ucot_threshold_quantile
ucot_threshold_mix
ucot_threshold_detach
ucot_temperature
ucot_marginal_mix
ucot_adaptive_mix
ucot_confidence_power
ucot_uniform_floor
ucot_rival_margin
ucot_ablation
```

Recommended `ucot_ablation` choices:

```text
full
threshold_only
marginal_only
off
```

`threshold_only` uses calibrated threshold with uniform marginals.
`marginal_only` uses the original learned threshold but UCOT quotas.
`full` uses both.

Add methods:

```python
def _compute_ucot_threshold(self, flat_cost) -> tuple[torch.Tensor, dict]:
    ...

def _compute_ucot_marginals(self, flat_cost, threshold, way_num, shot_num) -> tuple[torch.Tensor, torch.Tensor, dict]:
    ...

def _apply_ucot_local_flow(self, flat_cost, way_num, shot_num) -> tuple[query_weight, support_weight, threshold_override, payload]:
    ...
```

Integrate these in the existing Ours-Final ECOT budget-bank flow before calling
`super()._forward_ecot_budget_bank(...)`.

Do not call the old `UtilityContrastiveMarginals` module from the new model.
It can remain for ablations, but UCOT must own its threshold calibration and
quota construction.

### `net/hrot_fsl.py`

Modify `_forward_ecot_budget_bank` to accept a threshold override:

```python
def _forward_ecot_budget_bank(..., threshold_override: torch.Tensor | None = None):
```

Replace:

```python
threshold = self.transport_cost_threshold.to(...)
```

with:

```python
if threshold_override is None:
    threshold = self.transport_cost_threshold.to(...)
else:
    threshold = threshold_override.to(...).clamp_min(self.eps)
```

Keep threshold scalar for v1. If a tensor is passed, support only scalar or
shape broadcastable to budget scores. Do not implement class-specific
thresholds in v1.

Always put the active threshold in payload when the override is used:

```python
payload["ecot_threshold"] = threshold.detach()
payload["transport_cost_threshold"] = threshold.detach()
```

The audit must read the active calibrated threshold, not the stale learned
base threshold.

### `main.py`

Add parser flags for UCOT:

```text
--ucot_threshold_quantile float default 0.70
--ucot_threshold_mix float default 1.0
--ucot_threshold_detach true/false default true
--ucot_temperature float default 0.25
--ucot_marginal_mix float default 0.65
--ucot_adaptive_mix true/false default true
--ucot_confidence_power float default 1.0
--ucot_uniform_floor float default 0.05
--ucot_rival_margin float default 0.0
--ucot_ablation {full,threshold_only,marginal_only,off}
```

Update model description text so `ours_final_ucot` is displayed as a separate
architecture, not an Ours-Final ablation.

Update audit report header to include:

```text
Tau q/c
UCOT enabled
UCOT ablation
UCOT threshold quantile/mix
UCOT marginal mix
```

Add `ucot/` metrics to `important_keys` and `debug_keys`.

### `run_all_experiments.py`

Add `ours_final_ucot` to model choices where needed.

Add a dedicated suite or simple model support:

```text
--models ours_final_ucot
```

Do not require:

```text
--ours_final_ablation_suite global_residual
--ours_final_tau_profile support_strict
```

Those are defaults of the new model.

Add optional UCOT ablation suite:

```text
--ours_final_ucot_ablation_suite core
```

Variants:

```text
ucot_full
ucot_threshold_only
ucot_marginal_only
ucot_off_equiv
```

The `ucot_off_equiv` control should match:

`ours_final + global residual w=0.1 + tau_q=0.5 + tau_c=0.8`

## Diagnostics Required

The model must log effectiveness diagnostics beyond accuracy.

Add these scalar diagnostics:

```text
ucot/enabled
ucot/ablation_id
ucot/base_threshold
ucot/calibrated_threshold
ucot/threshold_ratio
ucot/threshold_quantile
ucot/positive_edge_rate
ucot/query_positive_acceptance_mean
ucot/edge_positive_acceptance_mean
ucot/query_specificity_mean
ucot/query_specificity_peak
ucot/evidence_confidence
ucot/effective_mix
ucot/query_entropy
ucot/support_entropy
ucot/query_peak
ucot/support_peak
ucot/query_l1_from_uniform
ucot/support_l1_from_uniform
ucot/uniform_fallback_query_share
ucot/uniform_fallback_support_share
ucot/transported_mass_fraction
ucot/query_l1_drift
ucot/support_l1_drift
```

Also keep these existing probe/audit metrics:

```text
ours_probe/negative_utility_mass_ratio
ours_probe/dead_query_ratio
ours_probe/harm_share
ours_probe/common_mass_ratio
audit_true_avg_cost_below_threshold
audit_true_threshold_margin
audit_utility_gap
local_pred_acc
global_vs_local_accuracy_delta
global_vs_local_fix_rate
global_vs_local_harm_rate
```

Diagnostic interpretation:

- If `ucot/calibrated_threshold` is still near `0.08`, calibration failed.
- If `ucot/effective_mix` is near zero, quotas are falling back to uniform.
- If `ucot/query_l1_from_uniform` is near zero, the new quota path is inert.
- If `ucot/query_peak` is too high and entropy collapses, lower
  `ucot_marginal_mix` or raise `ucot_uniform_floor`.
- If `negative_utility_mass_ratio` remains high but positive edge rate rises,
  the UOT relaxation is ignoring quota targets; inspect `query_l1_drift` and
  `support_l1_drift`.

## Tests To Add

### Unit tests

Add to `tests/test_ours_dmuot.py` or a new `tests/test_ours_final_ucot.py`.

1. Factory builds the new model:

```python
args.model = "ours_final_ucot"
model = build_model_from_args(args)
assert model.enable_global_residual_score
assert model.global_residual_weight == approx(0.1)
assert model.tau_q == approx(0.5)
assert model.tau_c == approx(0.8)
assert model.enable_ucot_calibration
```

2. Threshold calibration is label-free and raises low threshold:

Create a synthetic cost tensor where base `T=0.08` and best-edge quantile is
around `0.30`. Assert `T_cal > base_T`.

3. Quotas conserve probability:

```python
query_prob.sum(-1) == 1
support_prob.sum(-1) == 1
query_prob >= 0
support_prob >= 0
```

4. Uniform fallback works:

With all costs equal, query/support quotas should be close to uniform and no
NaNs should appear.

5. Threshold override reaches score:

Run a tiny forward where `threshold_override > base_threshold`; assert
`outputs["ecot_threshold"]` equals the override and score changes relative to
the original threshold.

6. Diagnostics surface:

Forward with `return_aux=True` and assert all required `ucot/` keys are present
and finite.

### Regression tests

Add one test that `ours_final` outputs are unchanged when `model="ours_final"`
and UCOT flags are default/off.

Add one test that `UtilityContrastiveMarginals` old path still works for
existing ablations, unless intentionally deprecated.

## Experiment Commands

Baseline control already used:

```bash
python run_all_experiments.py \
  --project test_tau_strict_and_debug_ours_global \
  --seeds 42,43,44 \
  --gpu_id 0 \
  --mode_id 1 \
  --shot_num 1 \
  --test_protocol clean \
  --extra_test_protocols noise \
  --noise_test_root /workspace/pd_fewshot/scalogram_27_1_pd_noise_benchmark_test_moderate \
  --noise_test_splits test_snr15db_rf_1_15mhz,test_snr10db_rf_1_15mhz,test_snr5db_rf_1_15mhz \
  --dataset_path /workspace/pd_fewshot/scalogram_27_1 \
  --dataset_name knee_aug_split \
  --fewshot_backbone resnet12 \
  --models ours_final \
  --ours_final_ablation_suite global_residual \
  --ours_final_ablation_variants global_res_w0p1 \
  --ours_final_tau_profile support_strict \
  --enable_ours_final_failure_probe true
```

New model 1-shot:

```bash
python run_all_experiments.py \
  --project test_ucot_ours_global \
  --seeds 42,43,44 \
  --gpu_id 0 \
  --mode_id 1 \
  --shot_num 1 \
  --test_protocol clean \
  --extra_test_protocols noise \
  --noise_test_root /workspace/pd_fewshot/scalogram_27_1_pd_noise_benchmark_test_moderate \
  --noise_test_splits test_snr15db_rf_1_15mhz,test_snr10db_rf_1_15mhz,test_snr5db_rf_1_15mhz \
  --dataset_path /workspace/pd_fewshot/scalogram_27_1 \
  --dataset_name knee_aug_split \
  --fewshot_backbone resnet12 \
  --models ours_final_ucot \
  --enable_ours_final_failure_probe true
```

New model 5-shot:

```bash
python run_all_experiments.py \
  --project test_ucot_ours_global \
  --seeds 42,43,44 \
  --gpu_id 1 \
  --mode_id 1 \
  --shot_num 5 \
  --test_protocol clean \
  --extra_test_protocols noise \
  --noise_test_root /workspace/pd_fewshot/scalogram_27_1_pd_noise_benchmark_test_moderate \
  --noise_test_splits test_snr15db_rf_1_15mhz,test_snr10db_rf_1_15mhz,test_snr5db_rf_1_15mhz \
  --dataset_path /workspace/pd_fewshot/scalogram_27_1 \
  --dataset_name knee_aug_split \
  --fewshot_backbone resnet12 \
  --models ours_final_ucot \
  --enable_ours_final_failure_probe true
```

Minimal ablation after implementation:

```bash
--models ours_final_ucot --ucot_ablation threshold_only
--models ours_final_ucot --ucot_ablation marginal_only
--models ours_final_ucot --ucot_ablation full
```

## Acceptance Criteria

The new model is worth keeping only if it improves mechanism and accuracy.

For the 60-sample 1-shot seed42 audit, compared with
`ours_final_global_res_w0p1 + support_strict`:

- `audit_true_avg_cost_below_threshold` increases materially.
- `ours_probe/negative_utility_mass_ratio` drops by at least 25 percent relative.
- `ours_probe/dead_query_ratio` drops by at least 20 percent relative.
- `ours_probe/harm_share` drops by at least 25 percent relative.
- `local_pred_acc` does not drop.
- `global_vs_local_accuracy_delta` does not increase; global residual must not
  become the main decision carrier.

Across seeds 42,43,44:

- Mean clean accuracy improves or remains equal with clearly better audit.
- Noise extra tests do not regress by more than 0.5 percentage points unless
  clean accuracy improves substantially and diagnostics justify the tradeoff.
- 5-shot does not show entropy collapse or support quota collapse.

Fail conditions:

- Accuracy improves but audit shows higher common/harm mass.
- `ucot/effective_mix` is near zero in most episodes.
- `ucot/query_l1_from_uniform` and `support_l1_from_uniform` are near zero.
- `global_vs_local_fix_rate` rises while local metrics stay poor.
- The model cannot be distinguished from Ours-Final except by extra flags.

## Implementation Order

1. Add registry/model name and factory defaults for `ours_final_ucot`.
2. Add threshold override support in `HROTFSL._forward_ecot_budget_bank`.
3. Add `OursFinalUCOT` subclass and UCOT config parsing.
4. Implement threshold calibration and diagnostics.
5. Implement query/support quotas and pass them to existing UOT.
6. Wire active threshold into score, payload, audit, and probe.
7. Add unit tests.
8. Add run_all_experiments support and commands.
9. Run the baseline and new model on seed42 1-shot first.
10. Only then run seeds 42,43,44 and 5-shot.

## Notes For The Next Agent

- Keep the first implementation scalar-threshold only.
- Do not add a neural MLP or another attention branch in v1.
- Do not change the global residual weight unless audit proves local is fixed
  and residual is the remaining bottleneck.
- Prefer analytic, episode-local calibration over learned complexity.
- Preserve `ours_final` byte path unless the user explicitly asks to change it.
- Every new component must expose diagnostics that show whether it actually
  changed transport behavior.
