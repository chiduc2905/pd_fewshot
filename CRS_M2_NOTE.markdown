# CRS-M2: Cross-Referenced Selective-Marginal M2

CRS-M2 is a support-marginal variant of the existing J-ECOT-M2 path. It keeps
the ECOT rho bank, base rho, budget policy, shot-decomposed UOT solver,
transported cost/mass computation, tau-shot pooling, and score

```text
E = score_scale * (threshold * transported_mass - transported_cost)
```

unchanged. The only default change when enabled is the support-side UOT
marginal.

Original M2 uses uniform fixed-budget marginals:

```text
a_r = rho / Lq
b_l = rho / Ls
```

CRS-M2 keeps `a` uniform by default and replaces only `b`:

```text
b_{i,c,k,l} = rho * pi_{i,c,k,l}
sum_l pi_{i,c,k,l} = 1
sum_l b_{i,c,k,l} = rho
```

The cross-reference branch follows the DeepEMD-style local cosine reference:

```text
q_bar = mean_l normalize(q_l)
r_cr = relu(cosine(normalize(s_l), normalize(q_bar))) + eps
p_cr = r_cr / sum_l r_cr
```

The selective branch contextually scores support tokens with a lightweight 2D
H+/H-/W+/W- scan, or `mamba_ssm.Mamba` when that backend is available and
selected:

```text
h = SSM_2D(LN(support_tokens))
p_ssm = softmax(Linear(LN(h)) / tau_ssm)
```

The final budget-preserving mixture is:

```text
p_rel = (1 - lambda_cr) * p_ssm + lambda_cr * p_cr
pi = (1 - eta) * uniform + eta * p_rel
pi = pi / sum_l pi_l
b = rho * pi
```

Ablation aliases in `run_jecot_ablation.py`:

```text
m2_original: CRS disabled
m2_cr_only: cross-reference branch only
m2_ssm_only: selective-SSM branch only
m2_crs: cross-reference + selective-SSM
```
