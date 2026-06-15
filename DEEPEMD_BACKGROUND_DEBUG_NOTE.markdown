# DeepEMD Foreground/Background Debug Probe

## Goal

This probe tests why DeepEMD performs well on the PD scalogram benchmark and
whether low-intensity background regions behave as nuisance noise or as
structured class evidence.

The probe is observational. It does not change DeepEMD features, transport
weights, transport plans, logits, losses, or parameters.

## Command

```bash
python run_all_experiments.py \
  --project deepemd \
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
  --models deepemd \
  --deepemd_debug true \
  --deepemd_debug_signal_quantile 0.70 \
  --deepemd_debug_common_margin 0.05 \
  --result_artifacts all \
  --export_dataset_profile true \
  --export_q1_report true \
  --q1_num_episodes 8 \
  --q1_queries_per_episode 2 \
  --q1_misclassified_only false \
  --experiment_tag deepemd_bg_probe \
  --log_cli_command true
```

`run_all_experiments.py` forwards the DeepEMD debug flags to `main.py`.

## Outputs

Each seed and clean/noise split produces:

- `debug_deepemd_*.txt`: mechanism verdict and all aggregate metrics;
- `results_*.txt`: normal accuracy plus DeepEMD diagnostic metrics;
- `q1_focus_*.png`: transport-mass focus over the query scalogram;
- `dataset_noise_profile_*.png` and CSV files: image-level noise proxies.

## Main tests

### Foreground-only counterfactual

`deepemd_signal_only_pred_acc` classifies with only transport edges connecting
high-intensity query and support tokens.

If it remains within about two accuracy points of `pred_acc`, foreground
matching is sufficient for DeepEMD.

### Background-only counterfactual

`deepemd_background_only_pred_acc` uses only background-to-background transport
contributions.

For four-way classification, chance is 0.25. A value clearly above chance
means the background is structured or correlated with class identity; it
should not be treated as pure removable noise.

### Cross-reference focus

`deepemd_uniform_weight_pred_acc` re-solves the same similarity matrices with
uniform node weights. The gap from `pred_acc` directly measures the value of
DeepEMD cross-reference weighting.

`deepemd/query_weight_signal_ratio` and
`deepemd/support_weight_signal_ratio` measure whether DeepEMD cross-reference
weights favor high-intensity regions. With quantile 0.70, the nominal signal
token fraction is about 0.30.

Values substantially above 0.30 indicate that DeepEMD already suppresses much
of the low-intensity background through its pair-conditioned node weights.

### Transport allocation

- `deepemd/fg_fg_mass_ratio`: foreground-to-foreground transport mass;
- `deepemd/background_involved_mass_ratio`: mass touching at least one
  background token;
- `deepemd/common_query_mass_ratio`: mass on query tokens whose top two class
  matches are too similar;
- transport and weight entropy: whether matching is uniform or concentrated.

## Decision rule

The background-noise hypothesis is supported only when:

1. foreground-only accuracy is close to full DeepEMD accuracy;
2. background-only accuracy is near chance;
3. noise severity increases background/common-token mass or reduces the
   true-class foreground score gap;
4. the same trend is stable across seeds 42, 43, and 44.

If background-only accuracy is high or foreground-only accuracy drops
materially, the data background contains useful structure. In that case,
aggressive background removal is a weak explanation for why Ours-Final should
beat DeepEMD.

## Novelty check

This diagnostic adds no architectural novelty to DeepEMD.

DeepEMD already uses dense local descriptors, pair-conditioned cross-reference
node weights, and balanced full-mass transport. Therefore, a standalone
Ours-Final novelty claim cannot be only "remove background matches."

The Ours-Final claim remains distinct only if its fixed-budget UOT,
threshold-mass scoring, utility-calibrated marginals, or global residual
mechanism produces matched improvements over DeepEMD and the diagnostic
metrics verify the claimed mechanism rather than only accuracy.
