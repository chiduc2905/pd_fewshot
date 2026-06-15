# ECOT-FSL: Full Mathematical and Implementation Design

Last updated: 2026-06-15

## 1. Scope

ECOT-FSL stands for **Episode-Competitive Optimal Transport for Few-Shot
Learning**.

The method is designed as a standalone extension of DeepEMD for general
few-shot image classification. It does not assume:

- scalogram-specific axes;
- foreground/background labels;
- a noise model;
- partial-discharge-specific morphology;
- any previous model implemented in this repository.

Only benchmark infrastructure is shared:

- ResNet12 family encoder;
- episodic query/support tensor format;
- cross-entropy training;
- optimizer, scheduler, checkpointing, validation, and test protocol from
  `run_all_experiments.py`.

The ECOT model and solver are implemented from scratch in `net/ecot_fsl.py`.

## 2. Research problem

DeepEMD solves one independent transport problem for every query-class pair:

\[
P_c^*
=
\arg\min_{P_c\in\Pi(a,b_c)}
\langle P_c,C_c\rangle.
\]

Every class receives a separate copy of the complete query mass. Consequently,
the same query token can be claimed strongly by several candidate classes.
Class competition happens only after all pairwise transport scores have been
computed.

The ECOT research hypothesis is:

> Local correspondence should be solved jointly with episode-level class
> discrimination. Labelled support descriptors from every candidate class
> should compete for one shared query-token mass budget inside the transport
> problem.

## 3. Reference formulation

### 3.1 Generalized unbalanced entropic OT

PythonOT/POT documents the generalized unbalanced Sinkhorn problem as:

\[
\min_{P\ge0}
\langle P,C\rangle
+\varepsilon\,\mathrm{KL}(P\Vert ab^\top)
+\rho_1\,\mathrm{KL}(P\mathbf1\Vert a)
+\rho_2\,\mathrm{KL}(P^\top\mathbf1\Vert b).
\]

Its implementation explicitly supports a semi-relaxed case using:

```python
reg_m = (float("inf"), rho)
```

The first marginal is then exact and the second marginal is KL-relaxed.

Reference code:

- PythonOT/POT unbalanced Sinkhorn:
  <https://github.com/PythonOT/POT/blob/master/ot/unbalanced/_sinkhorn.py>
- POT documentation:
  <https://pythonot.github.io/gen_modules/ot.unbalanced.html>

Primary algorithm reference:

- Chizat, Peyre, Schmitzer, and Vialard, **Scaling Algorithms for Unbalanced
  Transport Problems**, Mathematics of Computation, 2018:
  <https://arxiv.org/abs/1607.05816>

Semi-relaxed OT references:

- Fukunaga and Kasai, **Fast Block-Coordinate Frank-Wolfe Algorithm for
  Semi-Relaxed Optimal Transport**, 2021:
  <https://arxiv.org/abs/2103.05857>
- Fukunaga and Kasai, SROT repository:
  <https://github.com/hiroyuki-kasai/SROT>

The public SROT repository currently contains the project description but not
an executable solver. Therefore, the implementation follows the maintained POT
generalized Sinkhorn update rather than inventing a new update rule.

### 3.2 ECOT objective

For one query image, collect every support token from all episode classes into
one labelled target measure.

Let:

- \(M\): number of query tokens;
- \(N\): number of episode classes;
- \(K\): number of support shots;
- \(L\): number of support tokens per image;
- \(T=NKL\): total number of labelled support tokens.

Define:

\[
a_i = \frac{1}{M},
\qquad
b_{c,k,j} = \frac{1}{NKL}.
\]

ECOT solves:

\[
\boxed{
\min_{P\ge0,\;P\mathbf1=a}
\langle P,C\rangle
+\varepsilon\,\mathrm{KL}(P\Vert ab^\top)
+\rho\,\mathrm{KL}(P^\top\mathbf1\Vert b)
}
\]

where:

- the query marginal is exact;
- the support marginal is relaxed;
- all classes compete for the same query mass;
- \(b\) is a class-balanced capacity prior, not a required target marginal.

This is the POT problem with:

\[
\rho_1=\infty,
\qquad
\rho_2=\rho.
\]

## 4. Solver derivation

Define the KL-reference Gibbs kernel:

\[
K
=
(ab^\top)\odot\exp(-C/\varepsilon).
\]

The generalized Sinkhorn scaling has:

\[
P=\operatorname{diag}(u)K\operatorname{diag}(v).
\]

For an exact source marginal:

\[
\phi_1=1.
\]

For a target KL penalty:

\[
\phi_2=\frac{\rho}{\rho+\varepsilon}.
\]

The updates are:

\[
u
=
\frac{a}{Kv},
\]

\[
v
=
\left(
\frac{b}{K^\top u}
\right)^{\frac{\rho}{\rho+\varepsilon}}.
\]

The implementation uses log-domain updates:

\[
\log u
=
\log a
-\operatorname{LSE}_j(\log K_{ij}+\log v_j),
\]

\[
\log v
=
\frac{\rho}{\rho+\varepsilon}
\left[
\log b
-\operatorname{LSE}_i(\log K_{ij}+\log u_i)
\right].
\]

After the final target update, `u` is recomputed once to enforce:

\[
P\mathbf1=a
\]

to numerical precision.

## 5. Why balanced OT is not used

If all support tokens are concatenated and balanced OT enforces

\[
P^\top\mathbf1=b,
\]

each class receives exactly its prior mass:

\[
\sum_{i,k,j}P_{i,c,k,j}=\frac1N.
\]

Class mass then cannot be used for prediction. The support marginal must be
relaxed for classes to compete.

## 6. Why a completely free target marginal is not used

If the target marginal has no regularization, every query token can collapse
onto its single cheapest support token. This produces:

- unstable hard assignment early in training;
- poor gradient coverage;
- arbitrary sensitivity to individual support descriptors;
- class collapse.

The target KL term provides a soft capacity prior while still permitting
unequal class mass.

## 7. Architecture

### 7.1 Inputs

```text
query   : [B, N*Q, 3, H, W]
support : [B, N, K, 3, H, W]
```

The benchmark currently uses:

- \(N=4\);
- \(Q=1\);
- \(K\in\{1,5\}\);
- \(H=W=84\).

### 7.2 Backbone

ECOT uses the same ResNet12 family encoder selected by the benchmark:

```text
image -> ResNet12 -> feature map [640, h, w]
```

No new attention, background branch, projector, or domain-specific encoder is
added.

### 7.3 Local descriptors

The feature map is flattened:

\[
Z\in\mathbb R^{L\times640},
\qquad
L=hw.
\]

Each descriptor is L2-normalized:

\[
\bar z_i=\frac{z_i}{\|z_i\|_2}.
\]

### 7.4 Joint cost

All support descriptors are kept:

\[
S
\in
\mathbb R^{N\times K\times L\times640}.
\]

They are concatenated only as the target axis of one OT problem. Their class,
shot, and token indices remain recoverable.

The ground cost is cosine distance:

\[
C_{i,c,k,j}
=
1-\langle\bar q_i,\bar s_{c,k,j}\rangle.
\]

No pairwise DeepEMD plan is solved.

### 7.5 Joint transport

For every query image:

```text
query tokens [M]
        |
        | one shared source marginal
        v
joint semi-relaxed OT plan [M, N*K*L]
        |
        +-- class 1 support tokens
        +-- class 2 support tokens
        +-- ...
        +-- class N support tokens
```

### 7.6 Class mass

The mass assigned to class \(c\) is:

\[
m_c
=
\sum_{i=1}^{M}
\sum_{k=1}^{K}
\sum_{j=1}^{L}
P_{i,c,k,j}.
\]

Because the query mass sums to one:

\[
\sum_c m_c=1.
\]

The class masses are therefore a categorical distribution generated directly
by the joint coupling.

### 7.7 Logits and loss

The default logits are:

\[
\ell_c=\log(m_c+\delta).
\]

The repository's standard cross-entropy gives:

\[
\mathcal L_{\mathrm{CE}}
=
-\log m_y.
\]

No auxiliary classification head is required.

An exposed `ecot_logit_scale` exists for controlled temperature experiments,
but the default is `1.0` so the logits retain the direct probabilistic
interpretation.

## 8. K-shot behavior

ECOT does not average support shots before transport.

For K-shot classification:

- every shot remains an independent set of support atoms;
- each class has prior mass \(1/N\);
- each shot has prior class mass \(1/(NK)\);
- each support token has prior \(1/(NKL)\).

This prevents a class from receiving more prior mass merely because it has more
support tokens.

There is no SFC inner-loop optimization. The 1-shot and 5-shot models use the
same architecture and solver.

## 9. Parameter interpretation

### 9.1 Entropic regularization \(\varepsilon\)

- Small \(\varepsilon\): sharper local assignments.
- Large \(\varepsilon\): smoother, more uniform assignments.
- Default: `0.05`.

### 9.2 Target relaxation \(\rho\)

The target scaling exponent is:

\[
\phi=\frac{\rho}{\rho+\varepsilon}.
\]

- \(\rho\rightarrow0\): weak target capacity, near row-wise Gibbs assignment.
- \(\rho\rightarrow\infty\): target marginal approaches \(b\).
- If \(\rho\) is too large, every class approaches mass \(1/N\), destroying
  classification.
- Default: `0.10`.

The ratio \(\rho/\varepsilon\), rather than either value alone, controls much of
the competition strength.

### 9.3 Iterations and tolerance

- Default maximum iterations: `60`.
- Default fixed-point tolerance: `1e-6`.
- Final source re-projection remains active even if the loop stops early.

## 10. Mathematical properties

### 10.1 Exact query conservation

\[
P\mathbf1=a.
\]

Each query token owns one finite mass budget shared across all classes.

### 10.2 Class permutation equivariance

Permuting support classes only permutes class blocks in the plan and logits.
It cannot change the semantic prediction.

### 10.3 Token permutation invariance

With no positional term, permuting query or support token order only permutes
the corresponding plan axes.

### 10.4 Episode-level competition

Increasing mass assigned to one class necessarily reduces mass available to
other classes because:

\[
\sum_c m_c=1.
\]

This property does not hold for independent DeepEMD plans.

### 10.5 Inductive inference

Each query is solved independently against the support episode. ECOT does not
use other unlabeled query images and is not a transductive OT classifier.

## 11. Diagnostics beyond accuracy

All metrics are returned under the `ecot/` namespace and are written into the
normal result text file and W&B diagnostics.

### 11.1 Numerical validity

- `ecot/source_marginal_l1`
- `ecot/fixed_point_residual`
- `ecot/target_kl`
- `ecot/linear_cost`
- `ecot/plan_kl`
- `ecot/objective`

Required expectations:

```text
source_marginal_l1 <= 1e-5
fixed_point_residual small and stable across training
all values finite
```

### 11.2 Competition behavior

- `ecot/class_mass_entropy`
- `ecot/class_mass_peak`
- `ecot/effective_class_count`
- `ecot/token_class_entropy`
- `ecot/token_claim_collision`
- `ecot/token_claim_count`

`token_claim_collision` is a pre-transport diagnostic. It measures the
fraction of query tokens for which at least two classes have a best local
similarity within `ecot_claim_margin` of the strongest class.

### 11.3 Label-aware mechanism checks

- `ecot/true_mass`
- `ecot/best_negative_mass`
- `ecot/true_rival_mass_gap`
- `ecot/true_rival_log_mass_margin`
- `ecot/mass_prediction_accuracy`

### 11.4 Inspectable tensors

When diagnostics are enabled:

- `ecot_class_mass`: `[num_queries, way]`
- `ecot_token_class_assignment`: `[num_queries, query_tokens, way]`

The second tensor is the per-query-token class allocation:

\[
r_{i,c}
=
\frac{\sum_{k,j}P_{i,c,k,j}}{a_i}.
\]

## 12. Falsification criteria

The research mechanism is not supported if any of the following holds:

1. `token_claim_collision` does not correlate with DeepEMD errors.
2. ECOT does not improve true-rival mass margins.
3. The shared query constraint can be removed without affecting results.
4. Joint class mass behaves like post-hoc softmax over independent scores.
5. Class mass collapses to one class in most episodes before meaningful
   feature learning.
6. Class mass remains nearly uniform because \(\rho\) is too high.
7. Accuracy gains appear while source marginal or fixed-point errors are
   numerically invalid.
8. Gains disappear under class-order permutation.

## 13. Required ablations

| Ablation | Question |
|---|---|
| DeepEMD | Main published baseline |
| Pairwise entropic OT | Is the gain only the solver? |
| Joint OT with fixed target marginal | Shows why balanced concatenation cannot classify by mass |
| Joint semi-relaxed OT | Core ECOT |
| Post-hoc softmax over independent scores | Is joint coupling necessary? |
| \(\rho/\varepsilon\) grid | Is competition controlled as predicted? |
| Class-order permutation | Does the model preserve equivariance? |
| Shared query constraint removed | Is the proposed mechanism active? |

Recommended initial grid:

```text
epsilon:          0.03, 0.05, 0.08
target_relaxation: 0.03, 0.05, 0.10, 0.20
```

Do not run a broad grid before confirming numerical validity on the defaults.

## 14. Verification implemented in code

`tests/test_ecot_fsl.py` contains:

1. Direct comparison against SciPy constrained optimization for the exact
   ECOT objective.
2. Exact source marginal verification.
3. Finite backpropagation through the solver.
4. A toy two-class episode with known matching classes.
5. Class permutation equivariance.
6. Model factory registration and configuration.

The SciPy comparison is important because it checks the objective itself,
rather than comparing against another implementation copied from this
repository.

## 15. Training and evaluation protocol

ECOT uses the common `run_all_experiments.py` protocol used by Ours-Final:

- ResNet12;
- 4-way episodic classification;
- 1-shot or 5-shot;
- one query per class for train/validation/test;
- 130 training episodes per epoch;
- 150 validation episodes;
- 150 test episodes;
- 100 epochs;
- AdamW;
- learning rate `5e-4`;
- weight decay `5e-4`;
- cosine scheduler;
- five warmup epochs;
- no label smoothing;
- no train augmentation;
- validation model selection;
- deterministic CuDNN settings;
- standard cross-entropy.

## 16. Commands

### 16.1 One-shot, GPU 0

```bash
python run_all_experiments.py \
  --project ecot_fsl \
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
  --models ecot_fsl
```

### 16.2 Five-shot, GPU 1

```bash
python run_all_experiments.py \
  --project ecot_fsl \
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
  --models ecot_fsl
```

### 16.3 Explicit solver override

```bash
python run_all_experiments.py \
  ... \
  --models ecot_fsl \
  --ecot_epsilon 0.05 \
  --ecot_target_relaxation 0.10 \
  --ecot_sinkhorn_iterations 60 \
  --ecot_sinkhorn_tolerance 1e-6
```

Unknown launcher arguments are forwarded to `main.py`, so ECOT solver flags can
be supplied directly to `run_all_experiments.py`.

## 17. Novelty boundary

ECOT must not claim:

- the first semi-relaxed OT method;
- the first generalized Sinkhorn method;
- the first OT classifier;
- the first multi-class OT method;
- the first local-descriptor few-shot method.

The narrow candidate contribution is:

> A DeepEMD-style inductive few-shot classifier that replaces independent
> query-class local transport plans with one N-way semi-relaxed coupling, in
> which labelled support descriptors from all episode classes compete for a
> shared query-token mass budget and class probabilities are induced directly
> by transported class mass.

This claim still requires a broader literature and patent search before paper
submission.

## 18. Implementation files

- `net/ecot_fsl.py`: new model and new batched log-domain solver.
- `tests/test_ecot_fsl.py`: mathematical and integration verification.
- `net/model_factory.py`: registry and constructor.
- `main.py`: CLI, model reporting, forwarding, and diagnostics collection.
- `run_all_experiments.py`: uses the existing common protocol without a
  model-specific training branch.
