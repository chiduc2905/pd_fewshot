# SPIFAEB-v2 Detailed Specification

This document explains the current `spifaeb_v2` implementation in
`pulse_fewshot` after the latest redesign.

It is written for another LLM or engineer who may need to extend the model.
The emphasis is:

- exact behavior
- exact tensor shapes
- exact controller semantics
- what belongs to the architecture contribution
- what is only a training safeguard

Primary source files:

- [`net/spif_aeb_v2.py`](/mnt/disk2/nhatnc/res/scalogram_fewshot/pulse_fewshot/net/spif_aeb_v2.py)
- [`net/spif.py`](/mnt/disk2/nhatnc/res/scalogram_fewshot/pulse_fewshot/net/spif.py)
- [`main.py`](/mnt/disk2/nhatnc/res/scalogram_fewshot/pulse_fewshot/main.py)
- [`net/model_factory.py`](/mnt/disk2/nhatnc/res/scalogram_fewshot/pulse_fewshot/net/model_factory.py)

Related base model:

- [`SPIFAEB_DETAILED_SPEC.md`](/mnt/disk2/nhatnc/res/scalogram_fewshot/pulse_fewshot/SPIFAEB_DETAILED_SPEC.md)

---

## 1. Why `spifaeb_v2` Exists

Base `spifaeb` is strong and stable, but its local controller has two problems:

1. the budget predictor only sees global geometry
2. the local operator relies on `threshold + fallback softmax`

This makes the local branch hard to interpret. In logs, base `spifaeb` often
shows:

- `budget_true < budget_best_negative`
- `active_matches_true < active_matches_best_negative`
- `fallback_fraction_true > fallback_fraction_best_negative`

That behavior is not necessarily bad, but it means the old "budget" semantics
are not clean.

`spifaeb_v2` is the attempt to make the local branch more principled while
preserving the main strength of the family:

- stable global branch remains the anchor
- local branch becomes an adaptive evidence retention module
- inference stays class-level and pooled, not shot-routed

---

## 2. Clean Contribution Claim of `spifaeb_v2`

The intended contribution of `spifaeb_v2` is:

1. keep the SPIF stable/variant global branch
2. build a local branch as a **stable-anchor residual refinement**, not a free
   detached alternative representation
3. predict a class-wise **retention controller** from the **local evidence
   profile itself**
4. decompose retention into:
   - a monotonic prior from evidence sharpness
   - a learned residual correction
5. replace threshold+fallback with a **monotonic rank-gated soft top-k**
   operator
6. supervise not only fused classification, but also:
   - local branch classification
   - controller ranking semantics

This is the architectural story.

What is **not** the core claim:

- `detach_local_backbone`
- `detach_budget_context`
- fixed `alpha = 0.7`
- exact choice of controller hidden size

Those are engineering choices that protect the stable base model.

---

## 3. Model Identity in One Sentence

`spifaeb_v2` is a global-prototype few-shot model whose local branch is a
stable-token residual head with evidence-profile-conditioned retention and
monotonic rank-gated matching.

---

## 4. Inheritance Structure

`spifaeb_v2` does **not** subclass `_SPIFBase`.

Instead it directly subclasses `BaseConv64FewShotModel` and reuses:

- the shared CNN backbone
- `SPIFEncoder`
- `_make_projection_head`

Code:

- encoder reuse: [spif_aeb_v2.py](/mnt/disk2/nhatnc/res/scalogram_fewshot/pulse_fewshot/net/spif_aeb_v2.py#L165)

This means `v2` is not just "base SPIFAEB plus a small patch". It is a new
head architecture built on SPIF components.

---

## 5. End-to-End Pipeline

For each image:

1. extract a backbone feature map
2. flatten it into tokens
3. run the SPIF global encoder to obtain:
   - stable tokens
   - variant tokens
   - stable gate
   - stable global vector
   - variant global vector
4. build local tokens from:
   - a stable anchor
   - plus a small residual branch

At episode level:

1. build class prototypes from support stable globals
2. build pooled support local token banks per class
3. compute global prototype scores
4. compute local evidence profile from query local tokens and support local token
   pools
5. predict class-wise retention ratios from that evidence profile
6. compute a strong SPIF-style local anchor score by partial top-r matching
7. compute an adaptive local refinement score by monotonic rank-gated soft top-k
8. blend anchor + adaptive local scores into the final local branch
9. fuse global and local scores
10. optimize:
   - fused CE
   - optional branch CE
   - controller ranking loss
   - residual shrinkage on the controller correction

---

## 6. Tensor Shapes

Notation:

- `B`: episodic batch size
- `Way`: number of classes per episode
- `Shot`: support shots per class
- `NQ`: number of queries per episode
- `M`: number of local tokens per image
- `D_in`: backbone token dimension, usually `64`
- `D_s`: stable dimension
- `D_v`: variant dimension
- `D_l`: local branch dimension

Input shapes:

- query: `[B, NQ, C, H, W]`
- support: `[B, Way, Shot, C, H, W]`

After backbone flattening:

- query raw tokens: `[NQ, M, D_in]`
- support raw tokens: `[Way*Shot, M, D_in]`

After SPIF global encoder:

- query stable tokens: `[NQ, M, D_s]`
- support stable tokens: `[Way*Shot, M, D_s]`
- query stable global: `[NQ, D_s]`
- support stable global: `[Way*Shot, D_s]`
- query gate: `[NQ, M, 1]`
- support gate: `[Way*Shot, M, 1]`

Reshaped per episode:

- support global: `[Way, Shot, D_s]`
- support local tokens: `[Way, Shot, M, D_l]`
- support gate: `[Way, Shot, M, 1]`

Class-level pooled support local bank:

- `[Way, Shot*M, D_l]`

Similarity tensor:

- `similarity = einsum("qld,wtd->qwlt", query_tokens, support_pool)`
- shape: `[NQ, Way, M_query, M_support_total]`

Scores:

- global scores: `[NQ, Way]`
- local scores: `[NQ, Way]`
- retention `rho`: `[NQ, Way]`
- active match counts: `[NQ, Way]`
- retained fraction: `[NQ, Way]`
- logits: `[NQ, Way]`

---

## 7. The Global Branch

The global branch is intentionally close to base SPIF/SPIFAEB.

### 7.1 Stable/Variant Encoder

The model reuses `SPIFEncoder`.

For raw token `x_i`:

- stable token `s_i`
- variant token `v_i`
- gate `g_i`

are computed exactly as in SPIF.

### 7.2 Stable Global Vector

`z_stable = normalize( sum_i g_i * s_i / sum_i g_i )`

### 7.3 Class Prototype

For class `c`:

`p_c = normalize( mean_k z_stable(x_c^k) )`

### 7.4 Global Score

`score_global(q,c) = cosine(z_q^stable, p_c)`

This branch is still the main stabilizer of the whole family.

Code:

- prototype and global score: [spif_aeb_v2.py](/mnt/disk2/nhatnc/res/scalogram_fewshot/pulse_fewshot/net/spif_aeb_v2.py#L271)

---

## 8. The Local Branch

This is where `v2` differs from base `spifaeb`.

### 8.1 Stable-Anchor Residual Local Head

The local branch does **not** directly use a free MLP over raw tokens as the
main descriptor.

Instead:

1. stable tokens from the global branch are aligned into local space:

   `a_i = local_anchor_align(s_i)`

2. raw tokens are passed through a learned local residual head:

   `r_i = local_head(x_i)`

3. local token is:

   `l_i = normalize( a_i + lambda_res * r_i )`

where:

- `lambda_res = aeb_v2_local_residual_scale`
- default `0.25`

Meaning:

- the stable SPIF tokens are the anchor
- the local head is only allowed to make a moderate correction

This is one of the most important architectural fixes in `v2`.

Code:

- [spif_aeb_v2.py](/mnt/disk2/nhatnc/res/scalogram_fewshot/pulse_fewshot/net/spif_aeb_v2.py#L209)

### 8.2 Optional Local Isolation

By default:

- `aeb_v2_detach_local_backbone = true`

This means the local path sees detached raw/stable tokens, so local branch
training does not directly distort the backbone/global trunk.

This is not the main contribution. It is an engineering safeguard that protects
the strongest part of the model while the local branch is being improved.

Code:

- [spif_aeb_v2.py](/mnt/disk2/nhatnc/res/scalogram_fewshot/pulse_fewshot/net/spif_aeb_v2.py#L240)

### 8.3 Class-Pooled Local Bank

As in base SPIFAEB, all local support tokens of a class are concatenated:

`pool_c = concat_k l_{c,k,*}`

So `v2` remains:

- class-level
- pooled
- support-order invariant

It is **not** shot-preserving at inference.

### 8.4 Prior-Anchored Local Matching

The final local branch is **not** only the adaptive controller score.

`v2` now keeps a strong SPIF-style local anchor:

`score_anchor(q,c) = mean_{rows} top_r( S_{q,c,row,*} )`

and blends it with the adaptive retention score:

`score_local(q,c) = lambda_anchor * score_anchor(q,c) + (1 - lambda_anchor) * score_adaptive(q,c)`

where:

- `lambda_anchor = aeb_v2_anchor_mix`
- current default `0.35`
- `top_r = aeb_v2_anchor_top_r`
- current default `4`

Why this exists:

- base SPIF/SPIFAEB already had a strong low-variance local matcher
- fully replacing it made `v2` more fragile on validation/test
- the anchor keeps the strong baseline geometry
- the adaptive controller becomes a refinement, not a full replacement

---

## 9. Local Evidence Profile

The central redesign in `v2` is that the controller now sees the local
similarity structure itself.

### 9.1 Similarity Tensor

For query local tokens `u_{q,l}` and class pooled support local tokens
`t_{c,t}`:

`S_{q,c,l,t} = <u_{q,l}, t_{c,t}>`

All tokens are L2-normalized before this.

### 9.2 Statistics Extracted from `S`

The model computes several summary statistics:

- `top1`: mean over query rows of the best support similarity
- `row_top1`: best support similarity per query row
- `topk_mean`: mean of top-`k` values per row, then mean over rows
- `sim_mean`: mean of all similarities
- `sim_std`: mean row-wise std over support positions
- `top1_std`: std of `row_top1` across query rows
- `concentration_gap = top1 - topk_mean`
- `contrast_gap = top1 - sim_mean`

The final evidence-profile vector is:

`stats = [top1, top1_std, topk_mean, sim_mean, sim_std, contrast_gap + concentration_gap]`

Shape:

- `[NQ, Way, 6]`

Code:

- [spif_aeb_v2.py](/mnt/disk2/nhatnc/res/scalogram_fewshot/pulse_fewshot/net/spif_aeb_v2.py#L327)

### 9.3 Interpretation

Roughly:

- high `top1` and high contrast mean some support tokens strongly match query
- high `top1_std` means row quality is uneven
- high `contrast_gap + concentration_gap` means evidence is sharp and peaked

This matters because `v2` uses the evidence profile to decide how many support
correspondences to retain.

---

## 10. Retention Controller

This is the main conceptual change.

`v2` does not use a free sigmoid budget predictor like base `spifaeb`.
It uses:

- a monotonic prior
- plus a learned residual

### 10.1 Evidence Sharpness Scalar

The code defines:

`sharpness = stats[5] - stats[1]`

that is:

`sharpness = (contrast_gap + concentration_gap) - top1_std`

Interpretation:

- larger sharpness means evidence is more concentrated and more reliable

Code:

- [spif_aeb_v2.py](/mnt/disk2/nhatnc/res/scalogram_fewshot/pulse_fewshot/net/spif_aeb_v2.py#L300)

### 10.2 Monotonic Prior

The prior retention ratio is:

`prior_unit = sigmoid( -k_prior * sharpness )`

`rho_prior = rho_min + (rho_max - rho_min) * prior_unit`

with:

- `k_prior = aeb_v2_budget_prior_scale`
- defaults:
  - `rho_min = 0.2`
  - `rho_max = 0.85`
  - `k_prior = 6.0`

Important meaning:

- if evidence is sharper, `sharpness` increases
- `sigmoid(-k * sharpness)` decreases
- `rho_prior` decreases

So `v2` is explicitly an **adaptive evidence retention** model:

- sharp evidence -> retain fewer correspondences
- diffuse evidence -> retain more correspondences

This is a crucial semantic point.

Base `spifaeb` did not encode this monotonic prior explicitly.

### 10.3 Learned Residual

The model also predicts a residual:

`delta = tanh(MLP(stats))`

Then:

`rho = clamp( rho_prior + (rho_max - rho_min) * lambda_residual * delta )`

with:

- `lambda_residual = aeb_v2_budget_residual_scale`
- default `0.15`

So the final controller is:

`rho = rho_prior + small learned correction`

not:

`rho = arbitrary MLP(stats)`

This is another major architectural fix. The residual is deliberately bounded.

Code:

- predictor: [spif_aeb_v2.py](/mnt/disk2/nhatnc/res/scalogram_fewshot/pulse_fewshot/net/spif_aeb_v2.py#L190)
- final controller: [spif_aeb_v2.py](/mnt/disk2/nhatnc/res/scalogram_fewshot/pulse_fewshot/net/spif_aeb_v2.py#L310)

### 10.4 Optional Context Detach

By default:

- `aeb_v2_detach_budget_context = true`

This means the stats used by the controller are detached before the residual MLP
sees them.

Reason:

- protect the main representation from controller instability

Again, this is a training safeguard, not the conceptual novelty.

---

## 11. Monotonic Rank-Gated Local Matching

Base `spifaeb` used thresholding and fallback.
`v2` replaces that with a differentiable rank-based operator.

### 11.1 Sort Similarities

For each query row and class:

- sort support similarities descending

`S_sorted(q,c,l,1) >= S_sorted(q,c,l,2) >= ...`

### 11.2 Effective Soft Top-k

Let:

- `T = number of support tokens in the class pool`
- `k_eff(q,c) = rho(q,c) * T`

For rank position `r = 1..T`, define:

`gate_r = sigmoid( (k_eff + 0.5 - r) / tau_rank )`

with:

- `tau_rank = aeb_v2_rank_temperature`
- default `0.35`

This produces a soft mask over ranks:

- top ranks get weights near `1`
- low ranks get weights near `0`
- transition is smooth

### 11.3 Rank Weights

Normalize along rank:

`w_r = gate_r / sum_r gate_r`

### 11.4 Row Score

For each query row:

`row_score(q,c,l) = sum_r w_r * S_sorted(q,c,l,r)`

### 11.5 Aggregate Over Rows

Default mode:

- `query_to_support`
- mean over query rows

Optional mode:

- `bidirectional`
- average query-to-support and support-to-query reductions

### 11.6 Retained Fraction

The model also logs:

- `row_budget_counts = sum_r gate_r`
- `retained_fraction = row_budget_counts / T`

This is the effective fraction of retained support tokens per query-class pair.

Code:

- rank-gated operator: [spif_aeb_v2.py](/mnt/disk2/nhatnc/res/scalogram_fewshot/pulse_fewshot/net/spif_aeb_v2.py#L356)
- local score path: [spif_aeb_v2.py](/mnt/disk2/nhatnc/res/scalogram_fewshot/pulse_fewshot/net/spif_aeb_v2.py#L395)

---

## 12. Score Fusion

The fused score is:

`logit(q,c) = alpha(q) * score_global(q,c) + (1-alpha(q)) * score_local(q,c)`

Two modes exist:

### 12.1 Fixed Fusion

Default current mode:

- `aeb_v2_fusion_mode = fixed`

Then:

`alpha(q) = alpha_base`

with default:

- `alpha_base = 0.7`

This keeps the model fair and conservative.

### 12.2 Margin-Adaptive Fusion

Optional mode:

- `margin_adaptive`

Then alpha is increased toward the global branch when global margin is better.
In the latest stable configuration, fixed fusion is preferred because adaptive
fusion introduced unnecessary drift in earlier `v2` runs.

Code:

- [spif_aeb_v2.py](/mnt/disk2/nhatnc/res/scalogram_fewshot/pulse_fewshot/net/spif_aeb_v2.py#L435)

---

## 13. Losses

This is where `v2` becomes a real architectural model rather than a renamed
inference trick.

### 13.1 Main Classification Loss

`L_fuse = CE(logits_fused, y)`

This is always present.

### 13.2 Branch CE Losses

Optional auxiliary losses:

- `L_global = CE(global_scores, y)`
- `L_local = CE(local_scores, y)`

Current defaults:

- `aeb_v2_global_ce_weight = 0.0`
- `aeb_v2_local_ce_weight = 0.05`

So in the current stable setting:

- global branch is trusted and left alone
- local branch gets a mild direct CE supervision

### 13.3 Controller Ranking Loss

This is new and important.

For each query:

1. find the true-class retention:

   `rho_true = rho(q, y)`

2. find the hardest negative class according to current local scores:

   `c_hard = argmax_{c != y} score_local(q,c)`

3. define:

   `rho_hard = rho(q, c_hard)`

4. budget gap:

   `gap = rho_hard - rho_true`

5. loss:

   `L_rank = softplus( margin - gap )`

with:

- `margin = aeb_v2_budget_rank_margin`
- default `0.05`

Interpretation:

- the model is encouraged to make the hard negative retain **more** evidence
  than the true class
- this matches the new semantics:
  - true class should rely on fewer, sharper matches
  - hard negatives need broader evidence to justify themselves

This is not a hack. It is part of the controller semantics.

Code:

- [spif_aeb_v2.py](/mnt/disk2/nhatnc/res/scalogram_fewshot/pulse_fewshot/net/spif_aeb_v2.py#L461)

### 13.4 Total Auxiliary Loss

During training:

`L_aux = lambda_g * L_global + lambda_l * L_local + lambda_r * L_rank + lambda_res * L_residual`

Current stable defaults:

- `lambda_g = 0.0`
- `lambda_l = 0.05`
- `lambda_r = 0.1`
- `lambda_res = 0.01`

where:

`L_residual = mean( rho_residual^2 )`

This term is important in the current codebase because the controller residual
can otherwise saturate to `-1/+1`, which tends to improve train fit faster than
validation/test generalization.

Total training loss:

`L_total = L_fuse + L_aux`

---

## 14. Diagnostics and How to Read Them

The trainer now logs many `v2`-specific signals.

### 14.1 Branch Strength

- `global_score_gap`
- `local_score_gap`
- `anchor_local_score_gap`
- `adaptive_local_score_gap`
- `global_local_agreement`
- `global_margin`
- `local_margin`

Interpretation:

- if `global_score_gap` collapses, the redesign broke the strongest branch
- if `local_score_gap` is positive and meaningful, local is helping
- `anchor_local_score_gap` tells you whether the baseline partial matcher is
  still strong
- `adaptive_local_score_gap` tells you whether the controller-driven refinement
  adds real signal beyond the anchor
- if `global_local_agreement` is high but local gap is still positive, local is
  consistent and slightly complementary

### 14.2 Controller Semantics

- `budget_true`
- `budget_best_negative`
- `budget_gap`
- `budget_prior_true`
- `budget_prior_best_negative`
- `budget_prior_gap`
- `budget_residual_true`
- `budget_residual_best_negative`
- `budget_residual_gap`
- `evidence_sharpness_true`
- `evidence_sharpness_best_negative`
- `evidence_sharpness_gap`
- `budget_rank_loss`
- `budget_residual_reg_loss`

Correct interpretation for current `v2`:

- `budget_true < budget_best_negative` is expected, not a bug
- `evidence_sharpness_true > evidence_sharpness_best_negative` is desirable
- `budget_prior` should already move in the correct direction
- `budget_residual` should refine, not saturate
- `budget_rank_loss` should go down as controller semantics become cleaner
- `budget_residual_reg_loss` should stay small; if it grows while val stalls,
  the residual is doing too much work

### 14.3 Local Retention

- `active_matches_true`
- `active_matches_best_negative`
- `retained_fraction_true`
- `retained_fraction_best_negative`

With current semantics:

- true class often should keep fewer matches than hard negatives

### 14.4 Health Checks

Good signs:

- `global_score_gap` near or above base `spifaeb`
- `local_score_gap` clearly positive
- `alpha` stable
- `budget_prior_gap < 0`
- `evidence_sharpness_gap > 0`
- `budget_residual` not saturated at `-1/+1`

Warning signs:

- `global_score_gap` much lower than base
- `local_score_gap` near zero or negative
- `budget_residual` saturating to extremes
- `budget_rank_loss` staying high while local CE is low

---

## 15. What Is Part of the Contribution vs What Is a Stabilizer

### 15.1 Part of the Architectural Contribution

- stable-anchor residual local head
- evidence-profile-driven controller
- monotonic prior on retention
- residual correction on top of prior
- monotonic rank-gated local operator
- prior-anchored local score blending
- controller ranking loss
- controller residual shrinkage

### 15.2 Primarily Engineering Safeguards

- `detach_local_backbone=true`
- `detach_budget_context=true`
- fixed `alpha=0.7`
- small local CE weight
- zero global CE weight

These choices are there to stop the local redesign from destroying the already
good SPIF global geometry.

---

## 16. Why Current `v2` Semantics Are Better Than Base AEB

Base `spifaeb` had ambiguous controller semantics:

- the code implied "positive class may need larger budget"
- logs often showed the opposite
- fallback softmax made interpretation even murkier

Current `v2` is cleaner because:

1. the model explicitly says it is doing **retention**, not naive "more budget
   is better"
2. the prior makes that semantics monotonic by construction
3. the residual only makes a bounded correction
4. the ranking loss enforces the same semantics during training
5. there is no fallback branch that silently converts sparse matching back to
   dense matching

This is the main reason the current `v2` is more defensable.

---

## 17. Current Default Configuration

Current important defaults:

- `spif_stable_dim = 64`
- `spif_variant_dim = 64`
- `spif_gate_hidden = 16`
- `spif_alpha_init = 0.7`
- `spif_gate_on = true`
- `spif_factorization_on = true`
- `spif_token_l2norm = true`

`v2` defaults:

- `aeb_v2_hidden = max(16, stable_dim / 2)`
- `aeb_v2_min_budget = 0.2`
- `aeb_v2_max_budget = 0.85`
- `aeb_v2_rank_temperature = 0.35`
- `aeb_v2_global_scale = 1.0`
- `aeb_v2_local_scale = 1.0`
- `aeb_v2_anchor_top_r = 4`
- `aeb_v2_anchor_mix = 0.35`
- `aeb_v2_local_residual_scale = 0.25`
- `aeb_v2_budget_prior_scale = 6.0`
- `aeb_v2_budget_residual_scale = 0.15`
- `aeb_v2_budget_rank_weight = 0.1`
- `aeb_v2_budget_rank_margin = 0.05`
- `aeb_v2_budget_residual_reg_weight = 0.01`
- `aeb_v2_fusion_mode = fixed`
- `aeb_v2_local_score_mode = query_to_support`
- `aeb_v2_share_local_head = false`
- `aeb_v2_query_gate_weighting = false`
- `aeb_v2_detach_budget_context = true`
- `aeb_v2_detach_local_backbone = true`
- `aeb_v2_global_ce_weight = 0.0`
- `aeb_v2_local_ce_weight = 0.05`
- `aeb_v2_eps = 1e-6`

Parser:

- [main.py](/mnt/disk2/nhatnc/res/scalogram_fewshot/pulse_fewshot/main.py#L237)

Builder:

- [model_factory.py](/mnt/disk2/nhatnc/res/scalogram_fewshot/pulse_fewshot/net/model_factory.py#L723)

---

## 18. Safe Modification Rules for Future Work

If you edit `spifaeb_v2`, keep these invariants unless you intentionally want a
new model:

1. do not let the local branch freely replace stable SPIF tokens
2. do not remove the controller prior and go back to a fully unconstrained MLP
3. do not reintroduce threshold+fallback unless you explicitly want the old
   semantics
4. do not let adaptive fusion dominate unless you have evidence it helps
5. do not judge the controller by `budget_true > budget_neg`; judge it by
   whether the semantics are internally consistent and improve fused accuracy

Recommended future upgrades:

- better evidence-profile statistics
- better regularization on `budget_residual` to avoid saturation
- mild support-structure statistics, but still class-level pooled inference
- stronger local head only if global branch remains protected

High-risk changes:

- shot-preserving inference in the main `v2` path
- removing detach safeguards before local branch is clearly stronger
- aggressive adaptive fusion
- turning the local residual branch into a full replacement branch

---

## 19. Minimal Pseudocode

```text
for each image x:
    F = backbone(x)
    T = flatten_spatial(F)

    # SPIF global branch
    S, V, g = SPIFEncoder(T)
    z_stable = normalize(weighted_mean(S, g))
    z_variant = normalize(mean(V))

    # v2 local branch
    anchor = align(S)
    residual = local_head(T)
    L = normalize(anchor + lambda_local * residual)

for each class c:
    prototype_c = normalize(mean_k z_stable(x_c^k))
    pool_c = concat_k L(x_c^k)

for each query q and class c:
    global_score = cosine(z_stable(q), prototype_c)

    similarity = cosine_matrix(L(q), pool_c)
    stats = evidence_profile(similarity)

    sharpness = stats[5] - stats[1]
    rho_prior = budget_min + (budget_max - budget_min) * sigmoid(-k_prior * sharpness)
    delta = tanh(MLP(stats))
    rho = clamp(rho_prior + (budget_max - budget_min) * lambda_residual * delta)

    sorted_similarity = sort_desc(similarity)
    rank_gate = sigmoid((rho * T_support + 0.5 - rank) / tau_rank)
    rank_weights = normalize(rank_gate)
    local_score = mean_query_rows(sum(rank_weights * sorted_similarity))

    logit = alpha * global_score + (1 - alpha) * local_score

L_fuse = CE(logit, y)
L_local = CE(local_score, y)
L_rank = softplus(margin - (rho_hard_negative - rho_true))
L_total = L_fuse + lambda_l * L_local + lambda_r * L_rank
```

---

## 20. One-Line Summary

`spifaeb_v2` is the version of SPIFAEB where the local branch is no longer a
heuristic sparse matcher but a stable-anchor residual matcher with a
local-evidence-conditioned retention controller and an explicit controller
semantics loss.
