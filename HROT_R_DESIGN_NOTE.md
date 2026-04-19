# HROT-R Design Note

## One-Sentence Summary

**HROT-R** is a structure-consistent posterior-UOT few-shot classifier. It keeps the noise-calibrated HROT-Q pipeline, then adds a learnable fused structural cost derived from a detached UOT posterior plan before the final noise-sink UOT solve.

Use it with:

```bash
python run_all_experiments.py --models hrot_fsl --hrot_variant R
```

Optional strength:

```bash
--hrot_structure_cost_init 0.05
```

## Why R Exists

HROT-Q already improves H by adding:

- probe-UOT token reliability;
- support cross-shot consensus;
- explicit noise sink;
- robust shot pooling.

The remaining gap against DeepEMD is likely not only token weighting. DeepEMD's local EMD can still win when a query and support share coherent local evidence patterns, while HROT-Q can overtrust isolated low-cost token matches. R targets this by asking a stronger UOT question:

```text
Does a token match preserve the local relational geometry implied by the UOT posterior?
```

This is closer to fused/unbalanced Gromov-Wasserstein reasoning: match feature-similar tokens, but penalize correspondences whose surrounding token-distance structure is inconsistent.

## Literature Motivation

- UOT theory relaxes hard marginal conservation and can create/destroy mass, making it suitable for outliers, missing regions, and unequal support.
- Scaling algorithms for UOT extend Sinkhorn-style matrix scaling to GPU-friendly unbalanced matching.
- Robust OT for point-cloud registration shows that unbalanced/robust OT helps when shapes have partial overlap and outliers.
- UOT for word alignment uses null/unmatched alignment as a first-class outcome rather than forcing all words to match.
- UOT and semi-UOT have been used in deep generative modeling to improve robustness under corrupted datasets.
- OT/UOT co-attention papers show that transport plans can act as global attention rather than purely local dot-product attention.
- Unbalanced GW and unbalanced COOT motivate adding structure/feature alignment beyond first-order feature costs.

## Mechanism

For each query `q`, class `c`, and shot `k`, R first follows Q:

1. compute Euclidean token cost `C`;
2. predict shot mass `rho`;
3. compute Q token marginals from probe-UOT reliability, support consensus, and hyperbolic token prior.

Then R computes a detached posterior UOT plan:

```text
P_probe = UOT(C, a_Q, b_Q)
```

From query-token relational distances `Dq(i,i')` and support-token relational distances `Ds(j,j')`, R forms a structural mismatch cost:

```text
S(i,j) = E_(i',j')~P_probe [(Dq(i,i') - Ds(j,j'))^2]
```

This is implemented efficiently without materializing the full six-dimensional tensor:

```text
S = Dq^2 row(P) + Ds^2 col(P) - 2 Dq P Ds
```

The final real-token cost is:

```text
C_R = C + lambda_struct * normalize(S)
```

where `lambda_struct = softplus(raw_structure_cost_weight)` is learnable and initialized by `--hrot_structure_cost_init`.

Finally, R uses Q's noise sink and robust shot pooling on `C_R`.

## Difference From Q

Q learns which tokens are reliable.

R learns which token-to-token matches are locally coherent under the UOT posterior.

This matters for scalograms because random background/noise can produce one or two cheap feature matches, but it is much harder for those matches to preserve the local relational pattern of nearby time-frequency evidence.

## Difference From DeepEMD

DeepEMD uses cross-reference token weights and a balanced/full EMD-style local alignment.

R uses:

- UOT mass relaxation;
- explicit sink for unmatched/noisy evidence;
- posterior-derived structure consistency;
- shot-level robust pooling.

So the novelty is not "attention over tokens" alone. It is posterior-UOT structural denoising before the final UOT solve.

## Diagnostics

`return_aux=True` adds:

- `base_cost_matrix`: original Euclidean token cost;
- `structure_cost`: added structure-consistency cost;
- `structure_probe_mass`: transported mass of the posterior structure probe;
- `structure_cost_weight`: learned structural-cost strength.

The key visualization is to compare `base_cost_matrix`, `structure_cost`, `transport_plan`, and `noise_sink_*` for cases where DeepEMD wins.

## Suggested Ablation

Use this table:

```text
H
Q
R with structure_cost_init=0.01
R with structure_cost_init=0.03
R with structure_cost_init=0.05
R with structure_cost_init=0.10
DeepEMD
```

If R beats Q but not DeepEMD, inspect whether `structure_cost_weight` grows or collapses. Collapse suggests the structure term is too strong/noisy early and may need warmup. Growth with better validation suggests the structural posterior is useful.

