# HROT-FSL Variant H: Reviewer-Facing Technical Note

## 1. One-Sentence Claim

**HROT-FSL-H** is a shot-decomposed unbalanced optimal transport head with geodesic episode-adaptive mass and threshold-calibrated transported-mass scoring for few-shot partial-discharge scalogram classification.

Reviewer-safe wording:

- Variant H is **not** a full hyperbolic-cost OT variant in the current implementation.
- Like Variant G, its transport cost is Euclidean token distance.
- Unlike Variant G, its mass predictor uses a **geodesic 4-feature EAM** rather than the full tangent-statistics EAM.
- Unlike Variant G, its mass reward is not a free `mass_bonus`; it is parameterized by a learnable **transport-cost threshold**.

A precise paper name is:

**Threshold-Calibrated Geodesic-Mass Shot-Decomposed UOT (HROT-FSL-H)**.

This matches the implementation in `net/hrot_fsl.py`: variant `H` enables `uses_unbalanced_transport`, `uses_learned_mass`, `uses_shot_decomposed_transport`, `uses_geodesic_eam`, and `uses_cost_threshold_score`, while `uses_hyperbolic_geometry=False` for the transport cost.

## 2. What H Adds Over G

Variant G already performs shot-decomposed Euclidean unbalanced OT with learned hyperbolic mass. Variant H adds two reviewer-relevant corrections:

1. **Geodesic EAM**

   H predicts transported mass from interpretable geodesic features:

   - query-to-shot mean distance;
   - shot-to-class mean spread;
   - query hyperbolic variance;
   - support-shot hyperbolic variance.

   This makes the mass network easier to explain than a high-dimensional concatenation of tangent means.

2. **Cost-threshold scoring**

   H replaces the unconstrained mass reward `beta * m` with:

   ```math
   s \, T \, m
   ```

   where `T > 0` is a learnable transport-cost threshold and `s` is the score scale.

   The shot logit becomes:

   ```math
   \ell_{q,c,k}
   =
   -s D_{q,c,k}
   + sT m_{q,c,k}
   =
   s m_{q,c,k}
   \left(
     T - \frac{D_{q,c,k}}{m_{q,c,k}}
   \right)
   ```

   Therefore, transported mass is rewarded only when the average transported cost `D/m` is below the learned threshold `T`. This is a stronger and cleaner story for reviewers than a free mass bonus.

## 3. Problem Addressed

Few-shot PD scalograms have partial and uncertain evidence:

1. discriminative traces may occupy only a subset of local time-frequency regions;
2. support shots can vary in quality, noise, or phase-frequency coverage;
3. balanced transport overmatches irrelevant query regions;
4. a free mass bonus can reward mass even when the transported evidence is not sufficiently cheap.

Variant H addresses these issues by:

- preserving support shots during inference;
- predicting query-class-shot mass from geodesic compatibility and shot dispersion;
- using unbalanced OT to avoid forced matching;
- rewarding transported mass only relative to a learned cost threshold.

## 4. Episode Setup

Let:

- `N` be the number of classes;
- `K` be the number of support shots per class;
- `L` be the number of spatial tokens per image;
- `q` index query samples;
- `c` index classes;
- `k` index support shots;
- `d` be the projected token dimension.

The backbone produces feature tokens:

```math
X_q = f_\theta(x_q) \in \mathbb{R}^{L \times C}
```

```math
X_{c,k} = f_\theta(x_{c,k}) \in \mathbb{R}^{L \times C}
```

After layer normalization and linear projection:

```math
Y = W \, \mathrm{LN}(X)
```

The Euclidean transport tokens are:

```math
Z^E = \frac{Y}{\|Y\|_2 + \epsilon}
```

The hyperbolic tokens for EAM are:

```math
Z^H = \exp_0^c(\alpha Y)
```

where `c > 0` is Poincare-ball curvature and `\alpha` is the projection scale.

## 5. Hyperbolic Statistics

The Poincare distance is:

```math
d_{\mathbb{D}_c}(u,v)
=
\frac{2}{\sqrt{c}}
\tanh^{-1}
\left(
  \sqrt{c}\|(-u)\oplus_c v\|_2
\right)
```

For a token set `Z`, the implementation uses:

```math
\mu(Z)
=
\exp_0^c
\left(
  \frac{1}{|Z|}
  \sum_{z_i \in Z}
  \log_0^c(z_i)
\right)
```

and:

```math
\sigma^2(Z)
=
\frac{1}{|Z|}
\sum_{z_i \in Z}
d_{\mathbb{D}_c}(z_i, \mu(Z))^2
```

For each class `c`, define the merged class support token set:

```math
Z_c^H
=
\bigcup_{k=1}^{K} Z_{c,k}^H
```

with class hyperbolic mean:

```math
\mu_c = \mu(Z_c^H)
```

For each shot:

```math
\mu_{c,k} = \mu(Z_{c,k}^H)
```

For each query:

```math
\mu_q = \mu(Z_q^H)
```

## 6. Geodesic Episode-Adaptive Mass

Variant H builds a 4-dimensional feature vector for each query-class-shot triple:

```math
g_{q,c,k}
=
\left[
  d_{\mathbb{D}_c}(\mu_q, \mu_{c,k}),
  d_{\mathbb{D}_c}(\mu_{c,k}, \mu_c),
  \sigma^2(Z_q^H),
  \sigma^2(Z_{c,k}^H)
\right]
```

Interpretation:

- `d(mu_q, mu_{c,k})`: direct query-shot compatibility;
- `d(mu_{c,k}, mu_c)`: how atypical a support shot is relative to its class evidence;
- `sigma^2(Z_q^H)`: query evidence dispersion;
- `sigma^2(Z_{c,k}^H)`: support-shot evidence dispersion.

The transported mass is:

```math
\rho_{q,c,k}
=
\mathrm{clip}
\left(
  \mathrm{MLP}_{\psi}(g_{q,c,k}),
  \rho_{\min},
  1
\right)
```

In the code, the EAM input dimension is exactly `4` for Variant H.

## 7. Shot-Decomposed Euclidean Unbalanced OT

Variant H solves one unbalanced OT problem for each `(q,c,k)`.

The Euclidean token cost is:

```math
C_{q,c,k}(i,j)
=
\|z^E_{q,i} - z^E_{c,k,j}\|_2^2
```

The mass-scaled marginals are:

```math
a_{q,c,k}(i)
=
\frac{\rho_{q,c,k}}{L}
```

```math
b_{q,c,k}(j)
=
\frac{\rho_{q,c,k}}{L}
```

The unbalanced entropic OT plan is:

```math
\Pi^*_{q,c,k}
=
\arg\min_{\Pi \ge 0}
\left[
  \langle \Pi, C_{q,c,k} \rangle
  + \tau_q \mathrm{KL}(\Pi\mathbf{1}\|a_{q,c,k})
  + \tau_s \mathrm{KL}(\Pi^\top\mathbf{1}\|b_{q,c,k})
  - \varepsilon H(\Pi)
\right]
```

The shot transport cost is:

```math
D_{q,c,k}
=
\langle \Pi^*_{q,c,k}, C_{q,c,k} \rangle
```

The transported mass is:

```math
m_{q,c,k}
=
\mathbf{1}^{\top}\Pi^*_{q,c,k}\mathbf{1}
```

## 8. Threshold-Calibrated Scoring

Variant H learns a positive transport-cost threshold:

```math
T = \mathrm{softplus}(\eta)
```

where `\eta` is `raw_transport_cost_threshold` in the implementation.

The mass reward weight is:

```math
w_m = sT
```

where `s` is `hrot_score_scale`.

The shot-level logit is:

```math
\ell_{q,c,k}
=
-sD_{q,c,k}
+ sT m_{q,c,k}
```

Equivalently:

```math
\ell_{q,c,k}
=
s m_{q,c,k}
\left(
  T - \bar{D}_{q,c,k}
\right)
```

where:

```math
\bar{D}_{q,c,k}
=
\frac{D_{q,c,k}}{m_{q,c,k}}
```

is the average transported cost per unit transported mass.

This gives H a clean decision principle:

- if `\bar{D}_{q,c,k} < T`, transporting mass is beneficial;
- if `\bar{D}_{q,c,k} > T`, transporting mass is penalized;
- the model can learn the threshold that separates reliable evidence transport from noisy overmatching.

The class logit is:

```math
\ell_{q,c}
=
\frac{1}{K}
\sum_{k=1}^{K}
\ell_{q,c,k}
```

Prediction is:

```math
\hat{y}_q
=
\arg\max_c \ell_{q,c}
```

## 9. Training Objective

The main episodic classification loss is:

```math
\mathcal{L}_{cls}
=
-\frac{1}{|\mathcal{Q}|}
\sum_{q \in \mathcal{Q}}
\log
\frac{
  \exp(\ell_{q,y_q})
}{
  \sum_{c=1}^{N}
  \exp(\ell_{q,c})
}
```

The mass regularizer is:

```math
\mathcal{L}_{\rho}
=
\frac{1}{|\mathcal{Q}|NK}
\sum_{q,c,k}
\left(
  \rho_{q,c,k} - \rho_0
\right)^2
```

The optional curvature floor penalty is:

```math
\mathcal{L}_{c}
=
\left[
  \max(0, c_{\min}-c)
\right]^2
```

The total objective is:

```math
\mathcal{L}
=
\mathcal{L}_{cls}
+ \lambda_\rho \mathcal{L}_\rho
+ \lambda_c \mathcal{L}_c
```

The threshold parameter `T` is trained through the classification loss because it directly affects the logits.

## 10. Reviewer-Facing Novelty

The strongest novelty story for Variant H is:

1. **Geodesic compatibility controls transported mass**

   The model does not use a fixed partial-transport budget. It predicts `rho` from direct query-shot geodesic distance, shot-to-class geodesic spread, and query/support dispersion.

2. **Shot-level support reliability is explicit**

   The feature `d(mu_{c,k}, mu_c)` penalizes or modulates atypical support shots. This is useful in few-shot PD because a single support shot can be noisy or have incomplete discharge evidence.

3. **Mass reward is threshold-calibrated**

   Instead of rewarding transported mass unconditionally, H learns a threshold `T` in the same unit as transport cost. This makes the score interpretable as a reward for transporting low-cost evidence and a penalty for transporting high-cost evidence.

4. **The model remains inductive and shot-preserving**

   It does not use query-batch transduction. Each query is scored against support shots and then aggregated.

## 11. What H Solves Better Than G

Variant G:

```math
\ell^G_{q,c,k}
=
-sD_{q,c,k}
+ \beta m_{q,c,k}
```

Variant H:

```math
\ell^H_{q,c,k}
=
-sD_{q,c,k}
+ sT m_{q,c,k}
```

The difference is not only a reparameterization. In H, the mass reward `sT` is explicitly tied to the transport cost scale. This yields:

```math
\ell^H_{q,c,k}
=
s m_{q,c,k}
\left(
  T - \frac{D_{q,c,k}}{m_{q,c,k}}
\right)
```

That gives a reviewer-ready explanation:

> Variant H learns the maximum average transport cost that should still count as reliable matched evidence.

This is more defensible than saying "we add a learned mass bonus".

## 12. Ablation Story

The important comparisons are:

| Comparison | What It Tests |
|---|---|
| `G -> H` | effect of geodesic EAM and threshold-calibrated scoring |
| `F -> H` | effect of shot decomposition plus geodesic threshold scoring |
| `E -> H` | class-level hyperbolic-cost EAM UOT vs shot-level Euclidean UOT with geodesic mass |
| `B -> H` | fixed-mass Euclidean UOT vs learned geodesic mass with threshold scoring |
| `A -> H` | balanced Euclidean OT baseline vs full shot-decomposed adaptive UOT |

A clean ablation table can show:

| Variant | Cost | UOT | Learned mass | Shot axis | EAM features | Mass reward |
|---|---|---|---|---|---|---|
| G | Euclidean | Yes | Yes | Preserved | tangent/statistical EAM | free `beta m` |
| H | Euclidean | Yes | Yes | Preserved | geodesic 4-feature EAM | threshold `sT m` |

The post-H ablations isolate the H-specific claims:

| Variant | Change relative to H | Question answered |
|---|---|---|
| I | replace geodesic EAM with Euclidean/L2 4-feature EAM | Are hyperbolic geodesic summaries better than ordinary L2 summaries? |
| J | use fixed `rho` with the same shot-decomposed threshold score | Is learned mass necessary beyond threshold scoring? |
| K | keep geodesic EAM but replace threshold reward with free `beta m` | Is the learned cost threshold better than an unconstrained mass bonus? |
| L | keep geodesic EAM + threshold but remove shot-to-class spread | Is `d(mu_ck, mu_c)` useful as shot outlierness / reliability evidence? |
| M | average H/I/L/fixed-rho mass and use threshold reward plus free bonus | Can the post-H ablation improvements be combined into a stronger exploratory hybrid? |
| N | keep H-style geodesic EAM + threshold and add label-free geodesic order consistency for `rho` during training | Can structural geometry-consistency make learned mass more useful without changing inference? |

The cleanest reviewer claim is supported if:

```text
H > I, H > J, H > K, and H > L under the same training protocol.
```

Then the conclusion is not just "H has more components"; it is:

- `H > I`: the improvement comes from geodesic reliability summaries, not merely from using four scalar EAM features;
- `H > J`: adaptive mass matters beyond fixed partial transport;
- `H > K`: threshold-calibrated mass reward is better than free mass reward;
- `H > L`: shot-to-class geodesic spread contributes useful support-shot reliability information.

Variant M should be treated differently from I/J/K/L. It is not a clean ablation of H; it is a hybrid candidate:

```math
\rho_M
=
\frac{1}{4}
\left(
  \rho_H
  + \rho_I
  + \rho_L
  + \rho_0
\right)
```

and:

```math
\ell^M_{q,c,k}
=
-sD_{q,c,k}
+ (sT + \beta)m_{q,c,k}
```

If M outperforms H consistently, the correct conclusion is that an ensemble-style reliability controller may be beneficial. If M is similar to H, H remains the cleaner main contribution.

Variant N is a training-objective variant for EAM. It preserves the H inference path and adds a label-free geodesic order-consistency regularizer:

```math
d_i < d_j
\Rightarrow
\rho_i \gtrsim \rho_j
```

```math
\mathcal{L}_{rank}
=
\frac{1}{Z}
\sum_{i,j}
\left[
\tilde d_j - \tilde d_i
\right]_+
\tau
\log
\left(
1 + \exp
\frac{\gamma - (\rho_i - \rho_j)}{\tau}
\right)
```

where `i` and `j` index support shots for the same query across the episode, `d` is the query-shot geodesic mean distance feature used by EAM, and `d` is normalized per query before weighting.

```math
Z = \sum_{i,j} [\tilde d_j - \tilde d_i]_+
```

The auxiliary objective becomes:

```math
\mathcal{L}_{aux}
=
\lambda_\rho\mathcal{L}_\rho
+ \lambda_{rank}\mathcal{L}_{rank}
+ \lambda_c\mathcal{L}_c
```

This is still episodic/few-shot compatible and benchmark-friendly because it uses no query label, no extra annotation, and no inference-time information; it only enforces the architectural claim that geodesic compatibility should order the learned transport budget.

## 13. Reviewer-Safe Paragraph

Suggested paper/rebuttal text:

> Variant H introduces a threshold-calibrated geodesic mass mechanism for shot-decomposed unbalanced OT. For each query-class-shot pair, the transported mass is predicted from four interpretable hyperbolic statistics: query-to-shot geodesic distance, shot-to-class geodesic spread, query variance, and support-shot variance. The final score uses a learnable transport-cost threshold, so mass is rewarded only when the average transported cost is below the learned reliability threshold. This is well suited to few-shot PD scalograms, where only part of the local time-frequency evidence may be comparable and individual support shots can vary in reliability.

## 14. Claims to Avoid

Avoid these claims unless the implementation changes:

- Do not say "Variant H uses hyperbolic distance as the OT cost."
- Do not say "Variant H performs full hyperbolic OT."
- Do not say "Variant H learns support-shot softmax weights." It averages shot logits uniformly after per-shot OT.
- Do not say "Variant H is transductive." It does not refine predictions using the query batch.

Safe claims:

- "H uses Euclidean token-cost UOT with geodesic hyperbolic statistics for adaptive mass."
- "H preserves support shots through the transport step."
- "H uses a learned transport-cost threshold to calibrate the mass reward."
- "H rewards transported evidence only when its average transport cost is below the threshold."

## 15. Implementation Hooks

Relevant implementation points:

- `net/hrot_fsl.py`
  - `variant == "H"` enables `uses_geodesic_eam` and `uses_cost_threshold_score`.
  - `EpisodeAdaptiveMass(..., input_dim=4)` is used for H.
  - `_build_geodesic_rho_per_shot(...)` constructs the four geodesic features.
  - `flat_cost = self._euclidean_cost(...)` confirms the transport cost is Euclidean.
  - `transport_cost_threshold = softplus(raw_transport_cost_threshold)`.
  - `_mass_reward_weight(...) = score_scale * transport_cost_threshold`.
  - `shot_logits = -score_scale * shot_transport_cost + mass_reward * shot_transported_mass`.
  - `logits = shot_logits.mean(dim=-1)`.

- `tests/test_hrot_fsl.py`
  - `test_hrot_fsl_variant_h_uses_geodesic_eam_with_shot_decomposed_score` checks the score formula and feature shape `(query, way, shot, 4)`.
  - `test_hrot_fsl_variant_h_backpropagates_cost_threshold_and_geodesic_eam` checks gradient flow through threshold and EAM.

## 16. Short Abstract Version

HROT-FSL-H targets few-shot PD scalogram classification where evidence is local, partial, and support-shot dependent. It preserves each support shot through unbalanced OT scoring, predicts transported mass from interpretable hyperbolic geodesic statistics, and replaces a free mass bonus with a learnable transport-cost threshold. This makes the score reward low-cost transported evidence while discouraging forced matching of noisy or irrelevant local tokens.
