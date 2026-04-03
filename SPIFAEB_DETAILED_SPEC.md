# SPIFAEB Detailed Specification

This document explains the `spifaeb` model in `pulse_fewshot` at a level meant
for another LLM or engineer who needs to reason about the architecture without
reconstructing it from scattered code.

The goal is not to sell the model. The goal is to state exactly what the code
does, what the intended contribution is, what assumptions are safe, and what
assumptions are dangerous.

Primary source files:

- [`net/spif.py`](/mnt/disk2/nhatnc/res/scalogram_fewshot/pulse_fewshot/net/spif.py)
- [`net/spif_aeb.py`](/mnt/disk2/nhatnc/res/scalogram_fewshot/pulse_fewshot/net/spif_aeb.py)
- [`net/model_factory.py`](/mnt/disk2/nhatnc/res/scalogram_fewshot/pulse_fewshot/net/model_factory.py)
- [`main.py`](/mnt/disk2/nhatnc/res/scalogram_fewshot/pulse_fewshot/main.py)

## 1. What SPIFAEB Is

`SPIFAEB` stands for:

- `SPIF`: Stable Partial Invariance Few-shot Networks
- `AEB`: Adaptive Evidence Budget

`spifaeb` is the fair, CE-only SPIF variant where:

- the backbone and token factorization are inherited from SPIF
- the global branch is unchanged from SPIF
- the local branch is changed from fixed top-`r` partial matching to an
  adaptive thresholded local matcher controlled by a learned class-conditioned
  budget

In code, `SPIFAEB` subclasses `_SPIFBase` and only overrides the local scoring
rule and the budget predictor:

- base SPIF core: [spif.py](/mnt/disk2/nhatnc/res/scalogram_fewshot/pulse_fewshot/net/spif.py#L142)
- AEB extension: [spif_aeb.py](/mnt/disk2/nhatnc/res/scalogram_fewshot/pulse_fewshot/net/spif_aeb.py#L14)

## 2. Contribution Claim of SPIFAEB

The clean contribution claim for `spifaeb` is:

1. Factorize local descriptors into a stable branch and a variant branch.
2. Use only the stable branch for few-shot matching.
3. Keep the low-variance global prototype branch from SPIF.
4. Replace the fixed local top-`r` rule with a query-class adaptive evidence
   budget that changes how much local evidence is retained.

This is not:

- support-query cross-attention
- episodic transformer reasoning
- shot-preserving inference
- a reconstruction classifier
- a Wasserstein/transport model

`spifaeb` is intentionally conservative. It is best understood as:

`stable-token SPIF` + `adaptive sparse local evidence selection`

## 3. High-Level Architecture

Given an episodic few-shot task:

- support: `S` with `Way` classes and `Shot` support images per class
- query: `Q` query images

each image is processed as follows:

1. A shared CNN backbone produces a feature map.
2. The spatial feature map is flattened into local tokens.
3. Tokens are projected into:
   - a stable branch
   - a variant branch
4. A stable evidence gate predicts one scalar per token.
5. The gated stable tokens produce:
   - token-level stable descriptors
   - a pooled stable global vector
6. The variant tokens produce a pooled variant global vector.

At episode level:

- global branch:
  - per-class prototype from support stable globals
  - cosine similarity between query stable global and class prototypes
- local branch:
  - all support stable tokens inside a class are pooled together
  - query stable tokens are compared against the class token pool
  - an adaptive evidence budget controls which similarities survive
- final score:
  - fixed weighted fusion of global and local scores

## 4. Tensor Shapes

Use the following notation:

- `B`: episodic batch size
- `Way`: number of classes in an episode
- `Shot`: support shots per class
- `NQ`: number of queries per episode
- `Hf, Wf`: backbone spatial size
- `M = Hf * Wf`: number of local tokens per image
- `D_in`: backbone channel dimension, usually `64`
- `D_s`: stable token dimension
- `D_v`: variant token dimension

Episode input shapes:

- query images: `[B, NQ, C, H, W]`
- support images: `[B, Way, Shot, C, H, W]`

Per-image tokenization:

- feature map to tokens: `[N, M, D_in]`

SPIF encoder outputs:

- stable tokens: `[N, M, D_s]`
- variant tokens: `[N, M, D_v]`
- gate: `[N, M, 1]`
- stable global: `[N, D_s]`
- variant global: `[N, D_v]`

Per-episode shapes inside `_SPIFBase._encode_episode(...)`:

- query global: `[NQ, D_s]`
- query tokens: `[NQ, M, D_s]`
- support global: `[Way, Shot, D_s]`
- support tokens: `[Way, Shot, M, D_s]`
- support pooled token bank for one class: `[Way, Shot * M, D_s]`

Scoring shapes:

- global scores: `[NQ, Way]`
- local scores: `[NQ, Way]`
- fused logits: `[NQ, Way]`

Local similarity tensor in `spifaeb`:

- `similarity = einsum("qld,wtd->qwlt", query_tokens, support_token_pool)`
- shape: `[NQ, Way, M_query, M_support_total]`

## 5. SPIF Core Before AEB

### 5.1 Token Factorization

The SPIF encoder factorizes each local token `x_i` into:

- stable token `s_i`
- variant token `v_i`

If factorization is enabled:

- `s_i = stable_head(x_i)`
- `v_i = variant_head(x_i)`

If factorization is disabled:

- `s_i = shared_head(x_i)`
- `v_i = shared_variant_adapter(s_i)`

Code:

- factorization: [spif.py](/mnt/disk2/nhatnc/res/scalogram_fewshot/pulse_fewshot/net/spif.py#L80)

### 5.2 Stable Gate

For each stable token:

- `g_i = gate_head(s_i)` in `(0, 1)`

If gating is disabled:

- `g_i = 1`

Code:

- gate computation: [spif.py](/mnt/disk2/nhatnc/res/scalogram_fewshot/pulse_fewshot/net/spif.py#L90)

### 5.3 Stable Global Vector

Stable global vector is a gate-weighted average:

`z_stable = sum_i g_i * s_i / sum_i g_i`

In code, stable tokens are multiplied by the gate before pooling:

- `gated_stable = stable_tokens * gate`
- `stable_global = pool_global(gated_stable, gate)`

Code:

- pooling: [spif.py](/mnt/disk2/nhatnc/res/scalogram_fewshot/pulse_fewshot/net/spif.py#L102)
- encoder forward: [spif.py](/mnt/disk2/nhatnc/res/scalogram_fewshot/pulse_fewshot/net/spif.py#L116)

### 5.4 Variant Global Vector

Variant global vector is the simple mean of variant tokens:

`z_variant = mean_i v_i`

### 5.5 Normalization

The encoder normalizes:

- stable global
- variant global
- gated stable tokens
- variant tokens

with L2 normalization if `token_l2norm=true`.

## 6. SPIF Few-Shot Head

### 6.1 Support Prototype

For class `c`, the support stable-global prototype is:

`p_c = normalize( mean_k z_stable(x_c^k) )`

Code:

- [spif.py](/mnt/disk2/nhatnc/res/scalogram_fewshot/pulse_fewshot/net/spif.py#L263)

### 6.2 Global Score

For query `q`:

`score_global(q, c) = cosine(z_q^stable, p_c)`

Code:

- [spif.py](/mnt/disk2/nhatnc/res/scalogram_fewshot/pulse_fewshot/net/spif.py#L274)

### 6.3 Original SPIF Local Score

Before AEB, SPIF local score uses fixed top-`r` partial matching:

1. Pool all support stable tokens of a class into one bank.
2. Compute query-token-to-support-token cosine similarities.
3. Keep top-`r` support matches for each query token.
4. Average over the retained scores and over query tokens.

For similarity tensor:

`S_{q,c,l,t} = <u_{q,l}, t_{c,t}>`

SPIF local score is:

`score_local_SPIF(q,c) = mean_l mean_top_r_t S_{q,c,l,t}`

Code:

- [spif.py](/mnt/disk2/nhatnc/res/scalogram_fewshot/pulse_fewshot/net/spif.py#L282)

### 6.4 Final SPIF Logit

Let `alpha in [0,1]` be the fusion weight.

`logit(q,c) = alpha * score_global(q,c) + (1-alpha) * score_local(q,c)`

In fair SPIF variants:

- `alpha` is typically fixed
- default is `0.7`

Code:

- alpha and fusion: [spif.py](/mnt/disk2/nhatnc/res/scalogram_fewshot/pulse_fewshot/net/spif.py#L211)
- fusion: [spif.py](/mnt/disk2/nhatnc/res/scalogram_fewshot/pulse_fewshot/net/spif.py#L351)

## 7. What SPIFAEB Changes

`spifaeb` changes only the local score.

Everything below remains from SPIF:

- backbone
- token factorization
- stable gate
- stable global prototype branch
- fusion rule

The changed part is:

- instead of fixed top-`r`, local evidence is filtered by a class-conditioned
  adaptive threshold

Code:

- [spif_aeb.py](/mnt/disk2/nhatnc/res/scalogram_fewshot/pulse_fewshot/net/spif_aeb.py#L68)

## 8. SPIFAEB Adaptive Evidence Budget

### 8.1 Budget Predictor Input

For query `q` and class prototype `p_c`, the code forms:

- `q`
- `p_c`
- `|q - p_c|`
- `q * p_c`

and concatenates them.

This means the budget predictor is driven by global stable geometry, not by
local similarity statistics.

If `D_s` is the stable dimension, predictor input shape is:

- `[NQ, Way, 4 * D_s]`

Code:

- [spif_aeb.py](/mnt/disk2/nhatnc/res/scalogram_fewshot/pulse_fewshot/net/spif_aeb.py#L52)

### 8.2 Budget Value

The predictor outputs:

`rho(q,c) in (0,1)`

through a sigmoid MLP.

Intended interpretation in the original code:

- larger `rho` means lower threshold
- lower threshold means keep more evidence

Code:

- [spif_aeb.py](/mnt/disk2/nhatnc/res/scalogram_fewshot/pulse_fewshot/net/spif_aeb.py#L44)

### 8.3 Local Similarity Tensor

Let:

- query stable tokens be `u_{q,l}`
- pooled support stable tokens for class `c` be `t_{c,t}`

Then:

`S_{q,c,l,t} = <u_{q,l}, t_{c,t}>`

where all tokens are L2-normalized.

Code:

- [spif_aeb.py](/mnt/disk2/nhatnc/res/scalogram_fewshot/pulse_fewshot/net/spif_aeb.py#L74)

### 8.4 Row Statistics

For each query row `l`, the code computes:

- row mean:
  `mu_{q,c,l} = mean_t S_{q,c,l,t}`
- row std:
  `sigma_{q,c,l} = std_t S_{q,c,l,t}`

### 8.5 Threshold

The adaptive threshold is:

`T_{q,c,l} = mu_{q,c,l} + beta * (1 - rho(q,c)) * sigma_{q,c,l}`

Interpretation:

- if `rho` is large, `(1-rho)` is small, threshold goes down
- lower threshold keeps more entries
- if `rho` is small, threshold goes up

Code:

- [spif_aeb.py](/mnt/disk2/nhatnc/res/scalogram_fewshot/pulse_fewshot/net/spif_aeb.py#L81)

### 8.6 Sparse Evidence Weights

The preliminary sparse weights are:

`P_{q,c,l,t} = relu( S_{q,c,l,t} - T_{q,c,l} )`

These are then normalized row-wise:

`W_sparse = P / sum_t P`

If an entire row is zero after thresholding, the code uses a fallback:

`W_fallback = softmax_t S_{q,c,l,t}`

Final row weights:

- use `W_sparse` if the row retained something
- otherwise use `W_fallback`

Code:

- [spif_aeb.py](/mnt/disk2/nhatnc/res/scalogram_fewshot/pulse_fewshot/net/spif_aeb.py#L88)

### 8.7 Local Score

The local AEB score is:

`score_local_AEB(q,c) = mean_l sum_t W_{q,c,l,t} * S_{q,c,l,t}`

Code:

- [spif_aeb.py](/mnt/disk2/nhatnc/res/scalogram_fewshot/pulse_fewshot/net/spif_aeb.py#L94)

### 8.8 Diagnostics Produced by SPIFAEB

The local branch also returns:

- `rho`
- `active_match_counts`
- `fallback_row_fraction`

Interpretation:

- `active_match_counts`: average number of support tokens surviving the raw
  threshold before fallback
- `fallback_row_fraction`: fraction of query-token rows that had no surviving
  entries and therefore used dense softmax fallback

Code:

- [spif_aeb.py](/mnt/disk2/nhatnc/res/scalogram_fewshot/pulse_fewshot/net/spif_aeb.py#L95)

## 9. Final SPIFAEB Inference Rule

The final logit is:

`logit(q,c) = alpha * score_global(q,c) + (1-alpha) * score_local_AEB(q,c)`

with:

- `alpha = 0.7` by default
- `alpha` fixed in fair `spifaeb`

Code:

- [spif.py](/mnt/disk2/nhatnc/res/scalogram_fewshot/pulse_fewshot/net/spif.py#L357)
- [spif_aeb.py](/mnt/disk2/nhatnc/res/scalogram_fewshot/pulse_fewshot/net/spif_aeb.py#L118)

## 10. Training Objective

### 10.1 Base Loss

`spifaeb` is intended as a fair CE-only model.

The training objective is standard episodic cross-entropy on final logits:

`L_cls = CE(logits, y)`

### 10.2 Auxiliary Losses

Unlike `SPIFMAX`, `spifaeb` disables the SPIF auxiliary regularizers by
default:

- `consistency_weight = 0`
- `decorr_weight = 0`
- `sparse_weight = 0`

Even though `_aux_loss(...)` is still called through the shared base, its
default contribution in `spifaeb` is zero.

Code:

- defaults set in constructor: [spif_aeb.py](/mnt/disk2/nhatnc/res/scalogram_fewshot/pulse_fewshot/net/spif_aeb.py#L22)

Therefore the clean training objective for `spifaeb` is:

`L_total = L_cls`

## 11. Important Implementation Semantics

These points are critical. A future LLM must not guess them incorrectly.

### 11.1 SPIFAEB Is Class-Pooled, Not Shot-Preserving

The local branch pools all support tokens of a class together:

`support_token_pool[c] = concat_k support_tokens[c,k]`

This means:

- support shot order does not matter for local inference
- shot identity is not preserved
- any shot-level routing or shot-wise weighting is not part of base `spifaeb`

Code:

- [spif.py](/mnt/disk2/nhatnc/res/scalogram_fewshot/pulse_fewshot/net/spif.py#L268)

### 11.2 The Gate Affects Both Global and Local Branches

The stable tokens stored in the episode are already gate-multiplied and
normalized by the encoder.

So:

- local matching is performed on gated stable tokens
- global pooling is also derived from gated stable tokens

This is part of why `spifaeb` is usually stable: the same gate influences both
branches, making them more aligned.

### 11.3 The Budget Predictor Is Not Local-Aware

The original `spifaeb` budget predictor uses only:

- query stable global
- class prototype
- simple global interactions

It does not directly inspect the local similarity matrix.

This is one of the main reasons it is easy for the model to learn semantically
ambiguous budget behavior.

### 11.4 Fallback Is Not a Minor Detail

The fallback softmax branch is structurally important.

If thresholding removes everything in a row, the local branch becomes dense
again for that row. This means the model can silently avoid the intended sparse
behavior if the threshold is too aggressive.

This detail often explains why the raw budget values do not behave as a naive
reader might expect.

## 12. Empirical Caveat: Do Not Assume `rho_true > rho_neg`

The code comments suggest:

- larger `rho` keeps more evidence
- therefore the true class might be expected to have larger `rho`

In actual training logs, this often does not happen.

Observed behavior in real runs:

- `budget_true < budget_best_negative`
- `active_matches_true < active_matches_best_negative`
- `fallback_fraction_true > fallback_fraction_best_negative`

This means the trained base model often behaves more like:

- true class: evidence is sharp, only a small amount is needed, fallback may
  still rescue rows
- negative class: evidence is diffuse, more raw matches may remain

Important consequence:

- do not hard-code the assumption that the true class must have the largest
  budget
- if you redesign the local branch, first decide whether you want the semantic
  of the controller to be:
  - "keep more evidence for positives"
  - or "retain fewer but sharper matches for positives"

These are different model hypotheses.

## 13. Why SPIFAEB Often Works Well in Practice

The base `spifaeb` tends to be strong and stable because:

1. The global branch is low-variance and class-level.
2. The local branch is class-pooled, not shot-fragmented.
3. The final fusion is conservative and fixed.
4. The gate couples token selection and global pooling.
5. There is no aggressive episode-level reasoning module to destabilize
   training.

In short:

- global branch provides robustness
- local branch provides extra evidence
- fusion is simple enough not to overfit

## 14. Why Some Upgrades Failed

This section is included because an LLM modifying the family can easily repeat
old mistakes.

### 14.1 `spifaeb_shot` Failed Because It Broke the Good Bias

The shot-preserving variant was worse because:

- local evidence calibration became shot-specific
- class-pooled evidence was broken into noisier shot-level pieces
- the original class-level AEB semantics did not transfer cleanly

### 14.2 Naive `spifaeb_v2` Iterations Failed Because They Hurt the Global Branch

Several `v2` attempts were worse than base `spifaeb` because:

- the local branch became too strong or too free
- adaptive fusion drifted away from the reliable global branch
- controller semantics were changed without enough inductive bias

The main lesson is:

- protect the global branch first
- then let local branch add value
- never let a noisy local redesign destroy the stable base geometry

## 15. Config Parameters That Matter for Base SPIFAEB

Base SPIF config:

- `--spif_stable_dim`
- `--spif_variant_dim`
- `--spif_gate_hidden`
- `--spif_alpha_init`
- `--spif_gate_on`
- `--spif_factorization_on`
- `--spif_global_only`
- `--spif_local_only`
- `--spif_token_l2norm`

SPIFAEB-specific config:

- `--aeb_hidden`
- `--aeb_beta`
- `--aeb_eps`

Relevant defaults in `pulse_fewshot`:

- stable dim: `64`
- variant dim: `64`
- gate hidden: `16`
- alpha: `0.7`
- gate on: `true`
- factorization on: `true`
- token L2 norm: `true`
- `aeb_hidden`: `max(16, stable_dim / 2)`
- `aeb_beta`: `1.0`
- `aeb_eps`: `1e-6`

Parser:

- [main.py](/mnt/disk2/nhatnc/res/scalogram_fewshot/pulse_fewshot/main.py#L171)
- [main.py](/mnt/disk2/nhatnc/res/scalogram_fewshot/pulse_fewshot/main.py#L219)

Builder:

- [model_factory.py](/mnt/disk2/nhatnc/res/scalogram_fewshot/pulse_fewshot/net/model_factory.py#L666)

## 16. Diagnostics Available in Trainer

The trainer can log the following SPIFAEB diagnostics:

- `global_true_score`
- `global_best_negative_score`
- `global_score_gap`
- `local_true_score`
- `local_best_negative_score`
- `local_score_gap`
- `global_local_agreement`
- `budget_true`
- `budget_best_negative`
- `budget_gap`
- `active_matches_true`
- `active_matches_best_negative`
- `active_matches_gap`
- `fallback_fraction_true`
- `fallback_fraction_best_negative`
- `fallback_fraction_gap`
- `mean_budget`
- `mean_gate`
- `alpha`

Diagnostic extraction lives in:

- [main.py](/mnt/disk2/nhatnc/res/scalogram_fewshot/pulse_fewshot/main.py#L1077)

When reading logs:

- if `global_score_gap` is high and stable, the base representation is healthy
- if `local_score_gap` is positive but smaller, local branch is additive
- if `global_local_agreement` is near `1.0`, local branch is mostly redundant
- if `fallback_fraction_true` becomes too high, sparse matching is collapsing
  into dense fallback

## 17. Safe Claims for a Paper or Report

Defensible claims for `spifaeb`:

- it is a conservative extension of SPIF
- it preserves the stable global prototype branch
- it introduces adaptive local evidence selection
- it tends to be stable across many experiments
- it is often stronger than more aggressive redesigns because it does not break
  class-level calibration

Claims that are too strong unless separately proven:

- "the true class always receives larger evidence budget"
- "AEB learns semantically interpretable budgets by itself"
- "the local branch is the main performance driver"

## 18. If You Need To Modify SPIFAEB

Recommended order of operations:

1. Protect the global branch first.
2. Do not break class-pooled local matching unless you have a very strong
   reason.
3. If redesigning the budget controller, decide explicitly what the controller
   should mean.
4. Add diagnostics before changing inference.
5. Avoid introducing shot-level complexity into base `spifaeb`.

Low-risk modifications:

- better logging
- better local controller inputs
- mild branch-specific calibration
- better local token head while keeping class pooling

High-risk modifications:

- shot-preserving routing in the main inference path
- fully adaptive fusion without safeguards
- letting a new local head jointly destabilize the backbone/global branch

## 19. Minimal Pseudocode

```text
for each image x:
    F = backbone(x)
    T = flatten_spatial(F)
    S = stable_head(T)
    V = variant_head(T)
    g = gate_head(S)
    S_gated = normalize(S * g)
    z_stable = normalize(sum(S * g) / sum(g))
    z_variant = normalize(mean(V))

for each class c:
    p_c = normalize(mean_k z_stable(x_c^k))
    token_pool_c = concat_k S_gated(x_c^k)

for each query q and class c:
    global_score(q,c) = cosine(z_stable(q), p_c)

    rho(q,c) = MLP([z_stable(q), p_c, |z_stable(q)-p_c|, z_stable(q)*p_c])
    similarity(q,c,l,t) = cosine(S_gated(q)_l, token_pool_c_t)
    threshold(q,c,l) = mean_t(similarity) + beta * (1-rho) * std_t(similarity)
    pre_weight = relu(similarity - threshold)
    if row is empty:
        weight = softmax(similarity)
    else:
        weight = normalize(pre_weight)
    local_score(q,c) = mean_l sum_t weight * similarity

    logit(q,c) = alpha * global_score(q,c) + (1-alpha) * local_score(q,c)

loss = cross_entropy(logits, labels)
```

## 20. One-Line Summary

`spifaeb` is a stable-global-prototype few-shot model whose only real novelty is
an adaptive sparse local matcher on gated stable tokens; its strength comes more
from staying conservative and well-calibrated than from complex episode-level
reasoning.
