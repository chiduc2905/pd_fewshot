# SPIFCE: Stable Partial Invariance Few-shot Network with Episodic Cross-Entropy Only

## Abstract

SPIFCE is a few-shot classification architecture designed around a simple but
strong hypothesis: a query image should be classified from the subset of its
spatial evidence that is stable with respect to class identity, while nuisance
variation should be isolated rather than fully suppressed at the backbone
level. Instead of using heavy support-query interaction, transductive episode
reasoning, or optimal transport over dense local correspondences, SPIFCE
factorizes tokenized backbone features into a stable branch and a variant
branch, learns a lightweight per-token stable evidence gate, and combines two
non-heavy classification paths: a low-variance global prototype path and a
lightweight top-r local partial matching path. The implementation in this
repository is intentionally conservative: SPIFCE uses only episodic
cross-entropy and disables all auxiliary regularization terms that are enabled
in SPIFMAX. This note explains SPIFCE as if it were a methods section of a
paper, but every claim is grounded in the current implementation in
`net/spif.py`.

## 1. Problem Setting

We consider the standard episodic few-shot classification setup. In each
episode, we observe a support set

$$
\mathcal{S} = \{(x_{c,k}, c)\}_{c=1,\dots,N;\;k=1,\dots,K}
$$

with `N` classes and `K` shots per class, and a query set

$$
\mathcal{Q} = \{(x_q, y_q)\}_{q=1}^{NQ}.
$$

The goal is to predict the label of each query sample from only a few support
examples. The design target of SPIFCE is especially strict in the 1-shot
regime: the model should still make full architectural sense when `K = 1`.
This design philosophy is explicitly stated in `new_model.md`.

## 2. Design Motivation

Many few-shot methods trade off between two extremes:

1. Global prototype methods are stable and low-variance, but may miss subtle
   local evidence.
2. Fine-grained matching methods preserve local information, but often become
   computationally expensive, query-conditioned, or high-variance.

SPIFCE is built to occupy the middle ground:

- preserve local evidence,
- avoid heavy learned matching heads,
- avoid support-query cross-attention,
- avoid transductive inference,
- avoid dependence on `K > 1`,
- keep the decision rule mostly geometric and non-parametric.

The central idea is not "learn a stronger relation module", but rather "learn
which spatial evidence is stable enough to trust, then classify from it with a
light head".

## 3. High-Level Architecture

For an input image, SPIFCE performs the following steps:

1. Encode the image with a shared few-shot backbone.
2. Flatten the resulting feature map into a set of spatial tokens.
3. Split each token into a stable branch and a variant branch.
4. Predict a scalar gate from the stable branch only.
5. Use the gated stable tokens to form:
   - a global stable embedding,
   - a token-level local matching representation.
6. Classify each query with a fusion of:
   - a global prototype score,
   - a local top-r partial matching score.

In code, the core components are:

- `SPIFEncoder` in `net/spif.py`
- `_SPIFBase` in `net/spif.py`
- `SPIFCE` in `net/spif.py`

## 4. Backbone and Tokenization

SPIFCE inherits from `BaseConv64FewShotModel` in `net/fewshot_common.py`. The
shared backbone returns a spatial feature map

$$
F \in \mathbb{R}^{B \times C \times H \times W},
$$

which is flattened into a token sequence by

$$
X = \text{Tokenize}(F) \in \mathbb{R}^{B \times L \times C},
\quad L = H \cdot W.
$$

This tokenization is implemented by `feature_map_to_tokens`, which reshapes
`(N, C, H, W)` into `(N, H*W, C)`.

### Practical builder defaults in this repo

When `SPIFCE` is instantiated through `net/model_factory.py`, the common pulse
few-shot benchmark path currently uses:

- `hidden_dim = 64`
- `stable_dim = 64` by default
- `variant_dim = 64` by default
- `gate_hidden = 16` by default
- `alpha_init = 0.7`
- `top_r = 3` from the CLI/config path unless overridden
- `fewshot_backbone = resnet12` when `--fewshot_backbone default`

These builder-level values are important because the runtime config can
override constructor defaults in `net/spif.py`.

## 5. Stable-Variant Token Factorization

The first architectural ingredient is a token-wise decomposition into stable
and variant subspaces.

Given token sequence

$$
X \in \mathbb{R}^{B \times L \times C},
$$

SPIFCE computes

$$
X^s = f_s(X), \qquad X^v = f_v(X),
$$

where `f_s` and `f_v` are two lightweight projection heads of the form

$$
\text{LayerNorm} \rightarrow \text{Linear} \rightarrow \text{GELU} \rightarrow \text{Linear}.
$$

In implementation, `_make_projection_head` constructs this block, and
`SPIFEncoder` owns:

- `stable_head`
- `variant_head`

The motivation is geometric rather than reconstructive: the network is asked
to expose a subspace for class-stable evidence and a second subspace for
nuisance or non-essential variation, without requiring a heavy auxiliary task.

### Optional factorization ablation

The encoder also contains a `shared_head` and `shared_variant_adapter`, used
when `factorization_on = false`. In that case, both branches come from the same
shared projection. This keeps ablation fair by removing factorization while
preserving overall architectural shape.

## 6. Stable Evidence Gate

Not every stable token should contribute equally. SPIFCE therefore predicts a
scalar gate from stable tokens only:

$$
g = \sigma(f_g(X^s)), \qquad
g \in \mathbb{R}^{B \times L \times 1},
$$

where `f_g` is a lightweight MLP:

$$
\text{LayerNorm} \rightarrow \text{Linear} \rightarrow \text{GELU} \rightarrow \text{Linear} \rightarrow \text{Sigmoid}.
$$

The gate is intentionally:

- per-token,
- per-sample,
- computed only from the sample itself,
- not conditioned on support classes,
- not query-adaptive across samples.

This is a crucial design choice: SPIFCE does not rely on cross-attention to
decide which local evidence matters. Instead, it learns a weak intrinsic prior
over which tokens are likely to be stable evidence.

The gated stable tokens are

$$
\hat{X}^s = g \odot X^s.
$$

If `gate_on = false`, the implementation returns an all-ones gate, turning
this component into an identity mapping.

## 7. Global Stable Embedding

From the gated stable token map, SPIFCE computes a weighted pooled global
stable embedding:

$$
z^s = \frac{\sum_{\ell=1}^{L} \hat{X}^s_\ell}{\sum_{\ell=1}^{L} g_\ell + \varepsilon}.
$$

In code, this is implemented by `pool_global` inside `SPIFEncoder`. The output
is then L2-normalized:

$$
\tilde{z}^s = \frac{z^s}{\lVert z^s \rVert_2}.
$$

The variant branch also produces

$$
z^v = \text{mean}(X^v),
$$

followed by L2 normalization. In SPIFCE this variant global embedding is not
used directly for classification, but remains part of the architecture for
symmetry, diagnostics, and compatibility with the SPIFMAX regularized variant.

## 8. Token-Level Representation for Local Matching

SPIFCE does not stop at global pooling. It also preserves a local token
representation by normalizing the gated stable tokens:

$$
T^s = \text{Norm}(\hat{X}^s),
$$

where the implementation uses:

1. `LayerNorm` over the feature dimension, then
2. optional L2 normalization along the token feature axis.

These normalized stable tokens are the basis for local matching between query
and support.

An important implementation nuance is that the local path uses the gated stable
tokens after normalization, not the raw stable tokens. This means local
matching is already filtered by the stable evidence selector.

## 9. Episodic Tensor Organization

Within each episode, support and query images are encoded independently by the
same network. The support tensor of shape

$$
(Way, Shot, C, H, W)
$$

is flattened over the `Way x Shot` dimension for shared encoding, then reshaped
back into episodic form.

The resulting tensors are:

- `support_global`: `(Way, Shot, Ds)`
- `support_tokens`: `(Way, Shot, L, Ds)`
- `query_global`: `(NumQuery, Ds)`
- `query_tokens`: `(NumQuery, L, Ds)`

This separation is handled by `_encode_episode` in `_SPIFBase`.

## 10. Global Prototype Branch

The first classification branch is the standard low-variance prototype path.
For each class `c`, SPIFCE computes a stable prototype by averaging support
global embeddings:

$$
p_c = \frac{1}{K} \sum_{k=1}^{K} \tilde{z}^s_{c,k}.
$$

Each prototype is then L2-normalized. Query-to-class global scores are computed
by cosine similarity:

$$
s_{\text{global}}(q,c) =
\left\langle \tilde{z}^{s}_{q}, \tilde{p}_{c} \right\rangle.
$$

This branch is robust and low-variance, especially in low-shot settings, but
it compresses all local evidence into a single vector and can therefore miss
fine-grained spatial cues.

## 11. Local Top-r Partial Matching Branch

To preserve local evidence, SPIFCE builds a support token pool for each class
by concatenating the stable tokens across shots:

$$
\mathcal{T}_c \in \mathbb{R}^{(K \cdot L) \times D_s}.
$$

For a query token set

$$
T_q \in \mathbb{R}^{L \times D_s},
$$

the model computes pairwise cosine similarities:

$$
A_{q,c,\ell,t} = \left\langle T_{q,\ell}, \mathcal{T}_{c,t} \right\rangle.
$$

Instead of aggregating all correspondences, SPIFCE keeps only the top-r
matches for each query token:

$$
s_{\text{local}}(q,c) =
\frac{1}{L}
\sum_{\ell=1}^{L}
\left(
\frac{1}{r}
\sum_{t \in \text{Top-}r(A_{q,c,\ell,:})}
A_{q,c,\ell,t}
\right).
$$

This branch is intentionally lightweight:

- no OT solver,
- no transport plan,
- no dense transformer matching,
- no query-conditioned adaptation module.

The branch keeps only the strongest few local correspondences, which is why
the method is called "partial matching" rather than dense matching.

## 12. Score Fusion

The final class logit is a convex combination of the global and local scores:

$$
s(q,c) = \alpha \, s_{\text{global}}(q,c) + (1-\alpha)\, s_{\text{local}}(q,c).
$$

For SPIFCE, `alpha` is fixed and non-learnable. In the current implementation:

- `alpha_init = 0.7`
- `learnable_alpha = False`

This choice encodes a prior preference for the global prototype path while
still allowing local evidence to refine the final decision.

The implementation also supports two ablations:

- `global_only = true`
- `local_only = true`

These are useful for measuring how much of the final performance comes from
stable global geometry versus local stable evidence.

## 13. Training Objective

SPIFCE is the "fair" variant of the SPIF family. It is trained with episodic
cross-entropy only:

$$
\mathcal{L}_{\text{SPIFCE}} = \mathcal{L}_{\text{CE}}.
$$

There are no active auxiliary losses in SPIFCE. In particular, the class
constructor explicitly disables:

- consistency regularization,
- stable/variant decorrelation regularization,
- gate sparsity regularization.

This matters scientifically: if SPIFCE improves over a baseline, the gain can
be attributed primarily to the core architecture rather than to auxiliary loss
engineering.

## 14. Relationship to SPIFMAX

SPIFMAX uses the same forward architecture, but activates three lightweight
regularizers in training:

- stable embedding consistency,
- stable/variant decorrelation,
- gate sparsity penalty.

SPIFCE therefore isolates the architectural effect of:

- stable-variant factorization,
- stable evidence gating,
- global prototype plus local top-r fusion.

This is why SPIFCE is the correct variant to analyze when one wants to
understand the method itself before studying regularization.

## 15. Why the Method Should Work

SPIFCE can be understood as a structured bias over local geometry:

1. The stable branch is encouraged by task pressure to carry class-relevant
   information.
2. The gate suppresses low-value stable tokens before both global pooling and
   local matching.
3. The global branch gives a low-variance class anchor.
4. The local branch restores discriminative detail that would be lost under
   global pooling.

This combination is especially suitable for defect and fault patterns in
scalogram-like imagery, where the presence of a class often depends on a small
number of informative regions rather than on uniform global texture.

## 16. Strengths

### 16.1 1-shot compatibility

SPIFCE does not rely on cross-shot reasoning modules. Every major component
still behaves sensibly when `Shot = 1`.

### 16.2 Mostly non-parametric head

The decision head is geometric:

- cosine prototypes globally,
- top-r cosine matching locally,
- scalar fusion.

This keeps the head lower-variance than a learned relation network.

### 16.3 Local evidence preservation

Unlike pure prototype methods, SPIFCE retains token-level evidence all the way
to the final classification score.

### 16.4 Interpretability

The gate values, global scores, local scores, and stable embeddings are all
available in diagnostics when `return_aux=True`, which makes the model easier
to inspect than heavily entangled matching architectures.

## 17. Limitations and Caveats

### 17.1 The variant branch is weakly constrained in SPIFCE

Because SPIFCE disables decorrelation and consistency penalties, the stable and
variant branches are not explicitly forced apart during training. The model may
still use the factorized structure beneficially, but this separation is more
strongly enforced in SPIFMAX than in SPIFCE.

### 17.2 Local matching cost still scales with token count

Although much lighter than OT or transformer reasoning, local top-r matching
still computes a dense similarity tensor between query tokens and pooled
support tokens.

### 17.3 Fusion scalar is globally fixed

For SPIFCE, `alpha` is a fixed scalar. This preserves simplicity and fairness,
but it may underfit cases where some episodes need more local evidence and
others need more global anchoring.

### 17.4 Implementation-path caveat

In the pulse_fewshot benchmark, the runtime builder in `net/model_factory.py`
can override some constructor defaults in `net/spif.py`. For example, the
effective `top_r` used in training is determined by the config path
(`main.py` -> `model_factory.py`) unless explicitly overridden from the
command line.

## 18. Complexity Discussion

Let `M = K * L` be the total number of support tokens per class after pooling
over shots. Then:

- global branch complexity is approximately `O(NumQuery * Way * Ds)`,
- local branch similarity cost is approximately
  `O(NumQuery * Way * L * M * Ds)`,
- top-r reduction itself is cheap relative to similarity formation.

This is more expensive than a pure prototype head, but substantially simpler
than transport-based local matching or transformer-style support-query fusion.

## 19. Implementation Map

The most important implementation anchors are:

- `net/spif.py`
  - `_make_projection_head`
  - `SPIFEncoder`
  - `_SPIFBase`
  - `SPIFCE`
- `net/fewshot_common.py`
  - `BaseConv64FewShotModel`
  - `feature_map_to_tokens`
- `net/model_factory.py`
  - builder path for `"spifce"`
- `main.py`
  - CLI arguments controlling SPIF configuration

## 20. Summary

SPIFCE is best viewed as a clean architectural baseline for stable-evidence
few-shot recognition. Its contribution is not a sophisticated optimizer, not a
heavy relation head, and not an auxiliary-loss stack. Its contribution is a
particular inductive bias:

- decompose local evidence into stable and variant parts,
- softly select stable evidence,
- classify with both global anchors and local partial correspondences,
- keep the entire head light enough to remain meaningful in 1-shot.

That is why SPIFCE is a useful research object on its own: it isolates whether
"stable partial invariance" is already a good few-shot principle before adding
the extra regularization terms of SPIFMAX or the more advanced SPIF variants
that appear elsewhere in this repository.
