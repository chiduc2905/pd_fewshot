# Dustbin OT Ours-Final Implementation Note

## Research and Novelty Check

External precedent:

- SuperGlue uses differentiable optimal transport with a dustbin row/column to
  match local features while rejecting non-matchable points:
  https://openaccess.thecvf.com/content_CVPR_2020/papers/Sarlin_SuperGlue_Learning_Feature_Matching_With_Graph_Neural_Networks_CVPR_2020_paper.pdf
- SALAD uses optimal transport aggregation with a dustbin cluster to discard
  uninformative local descriptors:
  https://openaccess.thecvf.com/content/CVPR2024/papers/Izquierdo_Optimal_Transport_Aggregation_for_Visual_Place_Recognition_CVPR_2024_paper.pdf
- PD few-shot/scalogram literature commonly focuses on Siamese/few-shot
  diagnosis, CWT scalogram classifiers, or attention/fusion classifiers, not an
  ECOT budget-bank classifier with a learned dustbin rejection layer.
  Examples checked:
  https://www.mdpi.com/1424-8220/20/19/5562 and
  https://www.semanticscholar.org/paper/Effectiveness-of-Wavelet-Scalogram-on-Partial-of-Sahoo-Karmakar/053db78c390c67d97fb618db5f1d0396575c40cd

Novelty claim for this repo should therefore be scoped carefully:

- Not novel: dustbin OT itself.
- Potentially novel as a standalone Ours-Final architecture: applying a learned
  SuperGlue/SALAD-style dustbin OT rejection layer inside the Ours-Final ECOT
  fixed-budget bank for PD scalogram few-shot classification, while preserving
  the `T * accepted_mass - accepted_cost` class-score identity and logging
  class-wise accepted-mass gaps.

## Fit With Ours-Final

Dustbin OT fits the current Ours-Final failure analysis better than only
changing marginals:

- Original fixed-budget UOT still sends most mass to real support tokens.
- Score-aligned marginals can move mass toward salient regions, but wrong
  classes can still spend their budget on easy background matches.
- Dustbin OT keeps uniform real-token capacity as the stable prior, but gives
  each token an explicit alternative: match real evidence or go to dustbin.

Implemented score:

```text
score = sum(P_rr * (T - C)) = T * real_mass - real_cost
```

The learned scalar `dustbin_score` is the rejection boundary. Real-real edges
compete against dustbin edges during Sinkhorn, but dustbin mass is excluded from
class evidence.

## Diagnostics Added

Debug report:

```text
debug_dustbin_ot_<dataset>_ours_final_<samples>_<shot>shot*.txt
```

Main metrics:

- `dustbin_ot/real_mass_fraction`
- `dustbin_ot/query_dustbin_fraction`
- `dustbin_ot/support_dustbin_fraction`
- `dustbin_ot/dustbin_score`
- `dustbin_ot/accepted_edge_score_mean`
- `dustbin_ot/rejected_query_score_mean`
- `dustbin_ot/score_identity_error`
- `dustbin_real_mass_true`
- `dustbin_real_mass_best_negative`
- `dustbin_real_mass_gap`
- `dustbin_score_true_score`
- `dustbin_score_best_negative_score`
- `dustbin_score_score_gap`

Acceptance is not based on accuracy alone. A useful run should show a positive
`dustbin_real_mass_gap`, non-collapsed real mass, and query/support dustbin
fractions that rise on noisy/background-heavy cases.

## Validation Run Locally

Executed checks:

```bash
python -m pytest tests/test_dustbin_ot.py tests/test_elastic_ot.py -q
python -m py_compile main.py run_all_experiments.py net/hrot_fsl.py net/ours.py net/model_factory.py net/modules/dustbin_ot.py
PYTHONIOENCODING=utf-8 python main.py --help
python run_all_experiments.py --help
```

Also ran a direct smoke test of `HROTFSL(... ecot_m2_score_mode="dustbin_ot")`
on a synthetic ECOT cost tensor and verified `logits`, `transport_plan`, and
`dustbin_ot/score_identity_error`.
