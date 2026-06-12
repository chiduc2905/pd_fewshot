# Hubness-Calibrated UOT for Ours-Final

## Failure diagnosis

The current Ours-Final score is:

```text
score = scale * (T * transported_mass - transport_cost)
```

The inspected 60-sample log shows a repeated validation pattern:

```text
global_score_gap > 0
local_score_gap <= 0
transport_audit/common_mass_ratio ~= 0.92 ... 0.98
```

The global residual can rescue the final class decision, but it does not change
the local transport plan. Common low-cost descriptors therefore remain visible
in the mass overlay and continue to receive the `T * mass` reward.

## Research basis

DeepEMD uses query/support cross-reference weights and balanced EMD over dense
descriptors:

- https://openaccess.thecvf.com/content_CVPR_2020/html/Zhang_DeepEMD_Few-Shot_Image_Classification_With_Differentiable_Earth_Movers_Distance_and_CVPR_2020_paper.html

Cross-domain similarity local scaling (CSLS) corrects high-dimensional hubs by
subtracting the local neighborhood density on both sides of a match:

- https://arxiv.org/abs/1710.04087

Hubness is also a documented failure mode in few-shot representations:

- https://openaccess.thecvf.com/content/CVPR2023/html/Trosten_Hubs_and_Hyperspheres_Reducing_Hubness_and_Improving_Transductive_Few-Shot_Learning_CVPR_2023_paper.html

The literature check did not find the complete mechanism below in few-shot
image classification. This is not an exhaustive novelty proof.

## HC-UOT

For normalized query token `q_i` and support token `s_c,k,j`, define cosine
similarity:

```text
sim(i,c,k,j) = cosine(q_i, s_c,k,j)
```

For each candidate class `c`, estimate two class-excluded null densities:

```text
r_q(i,c)     = mean over rival class-shots of top-K sim(q_i, rival tokens)
r_s(c,k,j)   = mean over rival class-shots of top-K sim(s_c,k,j, rival tokens)
```

The two-sided adjusted similarity is:

```text
A(i,c,k,j) = 2 * sim(i,c,k,j) - r_q(i,c) - r_s(c,k,j)
```

A descriptor that matches every class has `A` near zero. A correspondence that
is strong relative to both rival neighborhoods has positive `A`.

HC-UOT uses the same signal once to calibrate the UOT problem:

```text
specificity = tanh(A / temperature)
D_hc        = clamp(D - lambda * std(D) * specificity, min=0)
```

Positive specificity lowers the transport cost; negative specificity raises
it. Positive specificity also defines query and support marginals, mixed with
a uniform exploration floor. If no positive specificity exists, the marginal
falls back exactly to uniform.

The original Ours-Final solver and score remain:

```text
P       = UOT(D_hc, a_hc, b_hc, rho=0.8)
S_local = scale * (T * mass(P) - <P, D_hc>)
S       = S_local + 0.1 * S_global
```

## Why this is not data bias

HC-UOT receives no pixel brightness, time coordinate, frequency band, manually
selected PD region, noise-test label, or dataset-specific mask. It uses only
the current episode's learned descriptors and support labels, which are already
available in the standard few-shot task.

The implementation is equivariant to token permutations and invariant when
identical support shots are duplicated. Hyperparameters must be fixed before
opening noise-test results.

## Novelty boundary

Do not claim novelty for CSLS, UOT, cosine local matching, or global residuals
individually. The candidate contribution is the combination of:

- class-excluded, two-sided local-density calibration inside an FSL episode;
- shot-count-normalized rival density;
- joint calibration of UOT cost and both token marginals;
- exact uniform fallback for episodes without positive specificity;
- threshold-mass UOT scoring plus a separate global residual.

The claim should be retained only if clean/noise results and cross-seed
stability improve over DeepEMD and the global-residual baseline.

## Locked experiment

Use the built-in suite with the same data split, episode seeds, backbone,
training budget, and global residual:

```powershell
python run_all_experiments.py `
  --dataset_path "C:\Partial Discharge\fewshot\dataset\scalogram_27_1" `
  --dataset_name scalogram_27_1 `
  --models ours_final `
  --mode_id 1 `
  --shot_num 1 `
  --seeds 42,43,44 `
  --ours_final_ablation_suite hubness_uot `
  --test_protocol clean `
  --extra_test_protocols noise `
  --noise_test_root "C:\Partial Discharge\fewshot\dataset\scalogram_27_1_pd_noise_benchmark_test_moderate" `
  --export_uot_evidence_figure true `
  --uot_evidence_artifacts aux_only
```

Repeat with `--shot_num 5`. The suite contains:

```text
global residual baseline
HC-UOT full
HC-UOT cost-only
HC-UOT marginal-only
```

Reject HC-UOT if it only improves the overlay, if `local_score_gap` remains
negative, if clean accuracy falls, or if gains disappear across seeds.
