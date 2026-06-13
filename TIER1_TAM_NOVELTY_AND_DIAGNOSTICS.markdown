# Tier-1 TAM: Novelty and Diagnostic Note

Date checked: 2026-06-13

## Architecture under test

Baseline:

`Ours-Final + uniform token marginals + KL-UOT(tau_q=tau_c=0.5) + T*M-C + global residual w=0.1`

Tier-1 candidate:

`Ours-Final + learned budget-preserving TokenAttentionMarginal + optional asymmetric KL-UOT + T*M-C + global residual w=0.1`

The TAM distribution is:

`rho * ((1-floor) * softmax(MLP(token) / temperature) + floor / L)`

The query and support scorer is shared by default. The total marginal mass
remains exactly `rho`.

## Novelty check

### Claims that are not novel alone

- Local-descriptor optimal transport for few-shot classification is established
  by DeepEMD.
- Non-uniform region weights in an OT/EMD formulation are also established.
  DeepEMD generates region weights with a cross-reference mechanism to reduce
  background influence.
- Attention over local descriptors is common in few-shot representation
  learning.
- UOT with KL-relaxed marginal constraints is established optimal-transport
  methodology.
- Learned or context-aware OT marginals now appear in other fine-grained
  alignment tasks.

Therefore, the paper should not claim that "learned attention marginals for
few-shot OT" are new by themselves.

### Defensible architecture-level novelty direction

The potentially novel unit is the complete standalone combination:

1. A fixed total transport budget with learned token allocation on both sides.
2. Asymmetric query/support KL relaxation in token-level UOT.
3. A learned positive threshold using the `T*M-C` classification utility.
4. A small global prototype residual retained only at the decision head.
5. Application to few-shot partial-discharge scalogram recognition.

A targeted literature search did not find this exact combination. This is an
architecture-level novelty claim, not proof of universal novelty. It should be
phrased as "to the best of our knowledge" only after a broader bibliography and
patent search.

### Difference from the closest baseline

DeepEMD's node weight is pair-conditioned through cross-reference with the
other image. TAM is a shared unary scorer learned from each token itself, then
used as a budget-preserving prior inside KL-UOT. The strongest distinction is
not the MLP, but how its prior interacts with mass destruction, asymmetric KL
penalties, and the threshold-mass score.

## Effectiveness diagnostics

The implementation logs these metrics in addition to accuracy:

- `token_marginal/normalized_entropy`: collapse indicator.
- `token_marginal/max_weight`, `min_weight`: concentration range.
- `token_marginal/l1_from_uniform`: whether TAM actually leaves the baseline.
- `token_marginal/logit_std`: whether the scorer separates tokens.
- `token_marginal/temperature`: learned temperature trajectory.
- `token_marginal/scorer_norm`: parameter-learning trajectory.
- `token_marginal/mass_error`: exact budget-conservation check.
- `token_marginal/query_l1_drift`, `support_l1_drift`: UOT marginal relaxation.
- `token_marginal/transported_mass_fraction`: mass-collapse check.
- `transport_cost_threshold`: existing threshold trajectory.

Interpretation:

| Observation | Meaning |
|---|---|
| `l1_from_uniform` near zero | TAM is functionally the uniform baseline |
| normalized entropy near zero | attention collapse |
| max weight near `1/L` | scorer is not discriminating |
| high marginal drift | UOT ignores the learned prior |
| low transported-mass fraction | loose relaxation destroys too much mass |
| scorer norm and logit std stay flat | weak or absent learning signal |

## Adoption rule

Adopt TAM only if it improves accuracy while all of these hold:

- Mass error remains numerically negligible.
- Entropy does not collapse.
- Marginals measurably differ from uniform.
- Transported mass remains stable relative to the shared baseline.
- Improvement repeats across seeds and clean/noise protocols.

## Primary references

- [DeepEMD, CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhang_DeepEMD_Few-Shot_Image_Classification_With_Differentiable_Earth_Movers_Distance_and_CVPR_2020_paper.pdf)
- [Adaptive Distribution Calibration with Hierarchical OT, 2022](https://arxiv.org/abs/2210.04144)
- [On Unbalanced Optimal Transport, JMLR 2023](https://www.jmlr.org/papers/volume24/22-1158/22-1158.pdf)
- [Context-Aware Marginals for Fine-Grained Alignment, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/papers/Li_Quota-Calibrated_Fine-Grained_Alignment_with_Context-Aware_Marginals_for_Text-based_Person_Retrieval_CVPR_2026_paper.pdf)
