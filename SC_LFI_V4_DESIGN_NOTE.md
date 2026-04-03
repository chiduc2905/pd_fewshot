# SC-LFI-v4 Design Note

Working title:

- **SC-LFI-v4**
- Expanded name: **Support-Conditioned Latent Flow Inference via Hierarchical Predictive Posterior Transport**

## 1. Why `v3` Is Still Not Few-Shot-Native

`sc_lfi_v3` fixed several weaknesses of `v1/v2`, but its failure mode under the fair shared protocol shows that the remaining problem is architectural, not merely optimization.

Observed failure pattern:

- `1-shot` reaches high train accuracy extremely early, while validation peaks early and then degrades.
- `5-shot` is not reliably better than `1-shot`, and can even collapse despite having more support evidence.

The core reason is that `v3` still learns a **support-fitting posterior** more than a **posterior predictive class object**.

The main theoretical weaknesses are:

1. support shots are flattened too early into one token bag;
2. the prior is still generated from the same support summary, so it is not a strong shrinkage target;
3. query-conditioned scoring rewrites class masses instead of keeping the class posterior fixed and modulating the transport operator;
4. auxiliary objectives align the posterior too directly to the current support sample cloud.

For few-shot learning, especially `1-shot`, this is the wrong bias.

## 2. Core Claim Preserved

The original SC-LFI claim remains unchanged:

> each class is represented as a support-conditioned latent evidence distribution, and query classification is a distribution-fit problem rather than prototype cosine matching.

`v4` preserves this exactly.

## 3. Main Shift in `v4`

`v4` changes the central object from:

- a support-conditioned cloud that is later compared to the query,

to:

- a **hierarchical posterior predictive evidence measure** that is explicitly built from:
  - shot-level support evidence,
  - a learned meta-prior dictionary,
  - and a conservative residual transport operator.

The design target is:

`score_c(q) = - D_{q,c}( nu_q, muhat_c )`

where:

- `nu_q` is the query latent evidence measure;
- `muhat_c` is a class posterior predictive evidence measure;
- `D_{q,c}` is a query-conditioned transport discrepancy that keeps `muhat_c` fixed and modulates the transport metric, not the class distribution itself.

## 4. Exact Mathematical Object

### 4.1 Query evidence measure

Given query tokens `{u_j}` and learned token masses `{rho_j^q}`:

`nu_q = sum_j rho_j^q delta_{u_j}`

with:

- `rho_j^q >= 0`
- `sum_j rho_j^q = 1`

### 4.2 Shot-level support measures

For class `c` and support shot `k`, latent support tokens are `{e_{c,k,j}}`.

We define the shot evidence measure:

`nu_{c,k}^{shot} = sum_j rho_{c,k,j}^{sup} delta_{e_{c,k,j}}`

To keep local structure while remaining efficient, each support image is compressed into a small invariant shot basis:

`B_{c,k} = { b_{c,k,r} }_{r=1}^{R_s}`

with internal masses:

`omega_{c,k,r} >= 0`, `sum_r omega_{c,k,r} = 1`

This gives the shot basis measure:

`nu_{c,k}^{basis} = sum_r omega_{c,k,r} delta_{b_{c,k,r}}`

### 4.3 Shot aggregation

Each support image should contribute as an observation, not disappear into one token bag.

We therefore define shot weights:

`pi_{c,k} >= 0`, `sum_k pi_{c,k} = 1`

and the empirical class support measure:

`nu_c^{emp} = sum_k pi_{c,k} nu_{c,k}^{basis}`

This is the first major few-shot correction relative to `v3`.

### 4.4 Meta-prior measure

`v4` uses a learned global evidence dictionary:

`G = { g_m }_{m=1}^{M_g}`

that is shared across episodes.

For class `c`, support-conditioned attention over this dictionary produces prior atoms:

`p_{c,r} = sum_m eta_{c,r,m} g_m`

with:

- `eta_{c,r,m} >= 0`
- `sum_m eta_{c,r,m} = 1`

and prior masses:

`beta_{c,r}^{prior} >= 0`, `sum_r beta_{c,r}^{prior} = 1`

This yields:

`pi_c = sum_r beta_{c,r}^{prior} delta_{p_{c,r}}`

Unlike `v3`, this prior is not merely another support transform. It is a true meta-learned shrinkage anchor.

### 4.5 Shot-aware shrinkage

The empirical-support trust must be conservative in low-shot regimes.

`v4` defines:

`tau(K) = K / (K + kappa)`

`gamma_c = sigma( g_phi(h_c, t_ctx, uncert_c) )`

`alpha_c = tau(K) * gamma_c`

where:

- `K` is shot number;
- `h_c` is class context;
- `t_ctx` is episode context;
- `uncert_c` is support uncertainty.

This guarantees:

- `alpha_c` is small in `1-shot`;
- `alpha_c` grows with shot count only when uncertainty allows it.

### 4.6 Posterior base measure

The support-prior base measure is:

`mu_c^0 = alpha_c * nu_c^{emp} + (1 - alpha_c) * pi_c`

### 4.7 Residual transport refinement

The transported class posterior is:

`muhat_c = (T_{theta,c})_# mu_c^0`

where `T_{theta,c}` is a support-conditioned residual transport map.

Important design choice:

- the transported atoms are **few shot-basis atoms plus prior atoms**, not the full support token cloud;
- the flow is therefore a conservative posterior refiner rather than a token memorizer.

### 4.8 Query-conditioned transport metric

`v4` does not rewrite the class posterior measure per query.

Instead it modulates the transport geometry:

`C_{q,c}(x, y) = || diag(s_{q,c})^{1/2} x - diag(s_{q,c})^{1/2} y ||_2^2`

where:

`s_{q,c} = softplus( g_psi( u_q^{pool}, h_c, t_ctx ) ) + eps`

This defines a query-conditioned diagonal metric.

The class measure `muhat_c` stays fixed; only the transport operator changes.

This absorbs the useful lesson from DeepEMD/CAN while preserving a clean class object.

## 5. Final Score

The final class score is:

`score_c(q) = - lambda_score * OT_eps( nu_q, muhat_c ; C_{q,c} )`

Optional prototype rescue branches are not part of the default design.

## 6. Loss Design

The total loss is:

`L_total = L_cls + lambda_fm L_fm + lambda_align L_align + lambda_margin L_margin + lambda_loo L_loo + lambda_reg L_reg`

### 6.1 `L_cls`

Cross-entropy on final transport logits.

### 6.2 `L_fm`

Flow matching on the posterior base atoms against shot-level support-consistent barycentric targets.

This is still flow matching, but the targets are defined over the few-shot support-shot measure rather than a flattened token bag.

### 6.3 `L_align`

Posterior alignment between `muhat_c` and the empirical class support measure `nu_c^{emp}`.

### 6.4 `L_margin`

Hard-negative margin on final query-class transport distances.

### 6.5 `L_loo`

Leave-one-shot-out predictive loss.

For `K >= 2`, one support shot per class is held out, the class posterior is rebuilt from the remaining `K-1` shots, and the held-out shot basis acts as a pseudo-query.

This is the key few-shot-predictive objective missing from `v3`.

### 6.6 `L_reg`

Anti-collapse regularization on:

- query token masses;
- support token masses;
- shot masses;
- prior masses.

## 7. What Is Kept from Earlier Versions

Kept unchanged or nearly unchanged:

- episodic backbone pipeline;
- latent evidence projector with token reliability masses;
- fixed-step Euler/Heun transport solver abstraction;
- weighted entropic OT utility.

## 8. What Is Rewritten

Rewritten completely:

- posterior context construction;
- query-conditioned transport scoring;
- auxiliary losses;
- model-level episode logic.

## 9. Why `v4` Is More Few-Shot-Appropriate

This design matches strong few-shot principles more closely:

1. support is preserved as a set of support-shot observations;
2. more shots increase evidence through explicit shot aggregation;
3. `1-shot` remains conservative because shrinkage is structurally bounded;
4. the prior is a true global meta-prior instead of a support self-reconstruction;
5. predictive leave-one-shot-out training favors generalization over support memorization.

## 10. Degenerate Special Cases

`v4` still contains clean limits:

- if `R_s = 1`, `M_p = 0`, and flow is off, the method approaches a support-barycenter classifier;
- if query-conditioned metric modulation is disabled, scoring reduces to a fixed transport distance to the class posterior;
- if `alpha_c -> 1` and each class has one support atom, the model approaches a prototype-like limit.

These limits are important because they show the model is a strict generalization of simpler few-shot classifiers rather than an unrelated head.
