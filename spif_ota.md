# SPIF-OTA: Single-Branch Physics-Aware Optimal Transport Inference for Few-Shot Scalogram Classification

## 1. Overview

### 1.1 Core idea

SPIF-OTA is a **single-branch few-shot inference architecture** that replaces the conventional **global branch + local branch + score fusion** paradigm with a **unified optimal-transport matching head**.

Instead of separately computing:

* a global class score from pooled features, and
* a local score from token-level matching,

SPIF-OTA performs **one structured matching operation** between the query token distribution and the support/class token distribution.

The key object is the **optimal transport plan**:
[
\gamma^* \in \mathbb{R}^{M_q \times M_s},
]
which encodes how much mass from each query token is matched to each support token. The final class score is derived directly from the OT objective. Therefore:

* **local correspondence** is encoded in the coupling (\gamma^*),
* **global decision** is encoded in the transport cost,
* and no explicit two-branch decomposition is needed.

This is the central paradigm shift of SPIF-OTA.

---

## 2. Motivation

### 2.1 Why not global + local?

In many dual-branch few-shot architectures:

* the **global branch** compresses spatial evidence too early through pooling or prototype formation,
* the **local branch** computes token-level matching separately,
* and the final prediction depends on a fusion rule.

This design has two weaknesses:

1. **Premature collapse of information**
   Global pooling removes token-level geometry before class-conditional comparison is fully resolved.

2. **Branch mismatch**
   The global and local branches often optimize different notions of similarity, making fusion unstable or redundant.

In particular, for **1-shot**, a separate local branch may contribute less than expected because there is only one support example per class, and the fusion logic may not exploit fine-grained correspondence effectively.

### 2.2 Why optimal transport?

Optimal transport is suitable because it does not only provide a scalar similarity value. It provides:

* a **cost matrix** (C), describing token-to-token dissimilarity,
* a **transport plan** (\gamma^*), describing correspondence structure,
* and a **transport objective**, yielding a principled class score.

Therefore OT can serve as a **unified matching operator** that simultaneously captures:

* geometric similarity,
* sparse/dense token correspondence,
* and class-level evidence.

### 2.3 Why physics-aware weighting for scalograms?

Scalograms are not generic images. Their spatial axes carry semantic meaning:

* one axis is related to time,
* the other is related to frequency.

Thus, not all spatial positions should necessarily contribute equally to matching. SPIF-OTA incorporates a **physics-aware mass construction mechanism** that biases the transport marginals toward frequency bands that are more likely to contain discriminative information.

This is not a hard-coded rule that low-frequency is always important. Instead, it is a **weak learnable inductive bias** encoded in the OT marginals.

---

## 3. Problem formulation

We consider an (N)-way (K)-shot episodic classification problem.

* Query sample: (x_q)
* Support set for class (c):
  [
  \mathcal{S}*c = {x*{c,1}, x_{c,2}, \dots, x_{c,K}}
  ]
* Number of classes per episode: (N)
* Number of support examples per class: (K)

The goal is to assign the query (x_q) to one of the (N) classes.

---

## 4. Architecture

SPIF-OTA contains five conceptual modules:

1. **Backbone encoder**
2. **Token projector**
3. **Physics-aware mass generator**
4. **Optimal transport matching block**
5. **Class evidence aggregator**

Importantly, these are **not separate prediction branches**. They form a **single inference pipeline**.

---

## 5. Module 1: Backbone encoder

A shared backbone (f_\theta) maps an input scalogram to a feature map:
[
F = f_\theta(x) \in \mathbb{R}^{D \times H \times W}
]

Typical choices:

* Conv64F
* ResNet12

To keep comparison fair with standard few-shot baselines, the backbone should remain conventional. The main contribution of SPIF-OTA lies in the inference head rather than in modifying the backbone.

---

## 6. Module 2: Token projector

The feature map is converted into a set of spatial tokens. Let:
[
M = H \cdot W
]

After flattening spatial dimensions:
[
T = [t_1, t_2, \dots, t_M] \in \mathbb{R}^{M \times D}
]

A token projector (g_\psi) maps these tokens into the matching space:
[
Z = g_\psi(T) = [z_1, z_2, \dots, z_M] \in \mathbb{R}^{M \times d}
]

Then each token is normalized:
[
\hat z_i = \frac{z_i}{|z_i|_2}
]

The projector serves two purposes:

* align backbone outputs with the metric space used by OT,
* decouple the transport geometry from the raw backbone representation.

---

## 7. Module 3: Physics-aware mass generator

### 7.1 Why masses matter

Optimal transport operates on measures, not on unordered token lists. Therefore each token needs an associated mass.

For query tokens, define the measure:
[
\mu_q = \sum_{i=1}^{M} a_i^{(q)} \delta_{\hat z_i^{(q)}}
]
where:
[
a_i^{(q)} \ge 0, \quad \sum_{i=1}^M a_i^{(q)} = 1
]

For a support example ((c,k)):
[
\nu_{c,k} = \sum_{j=1}^{M} b_j^{(c,k)} \delta_{\hat z_j^{(c,k)}}
]
with:
[
b_j^{(c,k)} \ge 0, \quad \sum_{j=1}^M b_j^{(c,k)} = 1
]

Thus, SPIF-OTA performs OT between two **discrete token measures**.

### 7.2 Mass parameterization

Each token receives a score from a content-position mass network:
[
u_i = \phi(\hat z_i, r_i, c_i)
]
where:

* (\hat z_i) is the token embedding,
* (r_i) is its row index,
* (c_i) is its column index.

A practical parameterization is:
[
\nu_i = w^\top \sigma(W_z \hat z_i + e_{r_i}^{(row)} + e_{c_i}^{(col)})
]

The normalized mass is then:
[
a_i = \frac{\exp(\nu_i)}{\sum_{\ell=1}^{M} \exp(\nu_\ell)}
]
(and similarly for (b_j)).

### 7.3 Frequency-position prior

For scalograms, rows correspond to frequency bands. To encode a weak physics-aware prior, we add a learnable row-bias vector:
[
\rho \in \mathbb{R}^{H}
]

Then:
[
\nu_i = \phi_{content}(\hat z_i) + \rho_{r_i} + \eta_{c_i}
]
where:

* (\rho_{r_i}) is the learnable frequency-row bias,
* (\eta_{c_i}) is an optional column bias.

This yields a **physics-aware token mass distribution** rather than a uniform or purely content-based weighting.

This module is important because the OT plan depends not only on pairwise token costs but also on the marginals. Therefore the model can express prior beliefs about where discriminative signal structure tends to lie.

---

## 8. Module 4: OT matching block

### 8.1 Token cost matrix

For query (q) and support sample ((c,k)), define the token-to-token cost matrix:
[
C^{(q,c,k)} \in \mathbb{R}^{M \times M}
]
with entries:
[
C_{ij}^{(q,c,k)} = 1 - \cos(\hat z_i^{(q)}, \hat z_j^{(c,k)})
]

Optionally, spatial/frequency structure can be added:
[
C_{ij}^{(q,c,k)} = 1 - \cos(\hat z_i^{(q)}, \hat z_j^{(c,k)}) + \tau \cdot \Delta_{ij}^{pos}
]
where (\Delta_{ij}^{pos}) measures row/column mismatch.

### 8.2 Entropic OT objective

The transport plan (\gamma^{*(q,c,k)}) is computed as:
[
\gamma^{*(q,c,k)}
=================

\arg\min_{\gamma \in \Pi(a^{(q)}, b^{(c,k)})}
\langle \gamma, C^{(q,c,k)} \rangle - \varepsilon H(\gamma)
]
where:

* (\Pi(a,b)) is the set of couplings with marginals (a) and (b),
* (H(\gamma) = -\sum_{ij} \gamma_{ij} \log \gamma_{ij}) is entropy,
* (\varepsilon > 0) controls regularization.

This can be solved efficiently with the Sinkhorn algorithm.

### 8.3 OT score

Once the optimal plan is obtained, define the shot-level score:
[
s_{q,c,k} = - \langle \gamma^{*(q,c,k)}, C^{(q,c,k)} \rangle
]

This score is class-conditional and correspondence-aware.

Interpretation:

* if a large amount of mass is transported along low-cost token pairs, then the score is high,
* if matching requires transporting mass across high-cost token pairs, then the score is low.

Thus, a single scalar score emerges from a structured token-alignment problem.

---

## 9. Module 5: Class evidence aggregation

The architecture remains single-branch, but in the (K>1) case there must be a mechanism to aggregate evidence across shots.

There are two principled options.

### 9.1 Option A: class-level concatenated measure

Concatenate all support tokens of class (c) into one class measure:
[
\nu_c = \sum_{k=1}^{K} \sum_{j=1}^{M} \beta_{c,k} b_j^{(c,k)} \delta_{\hat z_j^{(c,k)}}
]

Then solve OT directly between (\mu_q) and (\nu_c).

This is simple, but it may blur shot-specific reliability.

### 9.2 Option B: hierarchical shot-wise aggregation

Compute a transport score per support shot:
[
s_{q,c,k} = -\langle \gamma^{*(q,c,k)}, C^{(q,c,k)} \rangle
]

Then aggregate these scores with learned shot weights:
[
\beta_{q,c,k} = \frac{\exp(u_{q,c,k})}{\sum_{k'=1}^{K} \exp(u_{q,c,k'})}
]
where (u_{q,c,k}) is produced by a shot-confidence network.

Final class score:
[
S_{q,c} = \sum_{k=1}^{K} \beta_{q,c,k} s_{q,c,k}
]

Optional consistency regularization:
[
S_{q,c} = \sum_{k=1}^{K} \beta_{q,c,k} s_{q,c,k} - \lambda \cdot \mathrm{Var}*k(s*{q,c,k})
]

This version is usually stronger conceptually because it preserves shot-level structure while remaining within a single branch.

---

## 10. Final decision rule

The final logits for an episode are:
[
\ell_{q,c} = S_{q,c}
]

Prediction:
[
\hat y_q = \arg\max_{c \in {1,\dots,N}} \ell_{q,c}
]

This is the only class prediction pathway. There is no separate local or global head.

---

## 11. Why this is still a single-branch model

A possible confusion is that SPIF-OTA has multiple stages, including shot aggregation. However this does **not** make it a two-branch architecture.

A two-branch architecture would mean:

* branch 1 independently computes one type of score,
* branch 2 independently computes another type of score,
* then the two are fused.

SPIF-OTA does not do that.

Instead it has:

* one token encoder,
* one mass generator,
* one OT matching operator,
* one class scoring pathway.

The transport plan already plays the role that local matching would otherwise play, and the transport objective already yields the class-level scalar that a global branch would otherwise produce.

So SPIF-OTA is best described as:

> a single-branch structured matching framework

not as a local/global hybrid.

---

## 12. Why this should help 1-shot

In 1-shot, each class provides only one support example, so class evidence is extremely sparse.

Prototype-style methods or scalar OT distances may under-utilize the internal spatial structure of the support feature map. In contrast, SPIF-OTA solves an (M \times M) transport problem between query and support tokens, preserving a much richer correspondence structure.

For example, with a (5 \times 5) feature map:

* each query has (M=25) tokens,
* each support image has (M=25) tokens,
* the transport plan contains (25 \times 25 = 625) pairwise alignment variables.

The value of SPIF-OTA in 1-shot does not come merely from having more numbers. It comes from preserving **structured class-conditional token alignment** instead of collapsing support evidence prematurely.

---

## 13. Why this should remain stable in 5-shot

In 5-shot, simple averaging can cause two problems:

* strong support examples may be diluted by weak ones,
* noisy support tokens may dominate pooled class prototypes.

The hierarchical shot-wise version of SPIF-OTA avoids this by:

* aligning query to each shot separately,
* estimating shot-specific transport evidence,
* then aggregating the evidence with learned shot reliability.

This creates a cleaner class decision pathway than simply pooling all tokens at the start.

---

## 14. Training objective

The main training loss is episodic cross-entropy:
[
\mathcal{L}*{CE} = -\log \frac{\exp(\ell*{q,y_q})}{\sum_{c=1}^{N} \exp(\ell_{q,c})}
]

Optional regularizers include:

### 14.1 Entropy control on masses

To avoid degenerate token masses:
[
\mathcal{L}_{mass} = -\sum_i a_i \log a_i - \sum_j b_j \log b_j
]

### 14.2 Shot consistency regularization

For (K>1):
[
\mathcal{L}_{cons} = \mathrm{Var}*k(s*{q,y_q,k})
]

### 14.3 Plan sparsity / concentration control

To regulate excessively diffuse or overly sharp couplings.

Final objective:
[
\mathcal{L} = \mathcal{L}*{CE} + \lambda_1 \mathcal{L}*{mass} + \lambda_2 \mathcal{L}_{cons}
]

The default recommendation is to begin with pure cross-entropy and only add regularization if training becomes unstable.

---

## 15. Tensor shapes (practical implementation view)

Assume:

* query batch size: (N_Q)
* number of classes: (N)
* number of shots: (K)
* tokens per image: (M)
* token dimension: (d)

Then:

* Query tokens:
  [
  Z_q \in \mathbb{R}^{N_Q \times M \times d}
  ]

* Support tokens:
  [
  Z_s \in \mathbb{R}^{N \times K \times M \times d}
  ]

* Query masses:
  [
  a_q \in \mathbb{R}^{N_Q \times M}
  ]

* Support masses:
  [
  b_s \in \mathbb{R}^{N \times K \times M}
  ]

* Cost matrix:
  [
  C \in \mathbb{R}^{N_Q \times N \times K \times M \times M}
  ]

* Transport plan:
  [
  \Gamma \in \mathbb{R}^{N_Q \times N \times K \times M \times M}
  ]

* Shot scores:
  [
  s \in \mathbb{R}^{N_Q \times N \times K}
  ]

* Final class scores:
  [
  S \in \mathbb{R}^{N_Q \times N}
  ]

---

## 16. Minimal forward pass blueprint

```python
# Backbone + projector
z_q = projector(backbone(query))        # (NQ, M, d)
z_s = projector(backbone(support))      # (N, K, M, d)

# Normalize tokens
z_q = l2_normalize(z_q, dim=-1)
z_s = l2_normalize(z_s, dim=-1)

# Mass generation
a_q = mass_net_query(z_q, pos_q)        # (NQ, M)
b_s = mass_net_support(z_s, pos_s)      # (N, K, M)

# Pairwise cost
cost = 1 - cosine_pairwise(z_q, z_s)    # (NQ, N, K, M, M)
# optional: cost += tau * positional_cost

# Sinkhorn transport
gamma = sinkhorn(cost, a_q, b_s, eps)   # (NQ, N, K, M, M)

# Shot-level OT score
shot_score = -(gamma * cost).sum(dim=(-1, -2))   # (NQ, N, K)

# Shot aggregation
beta = softmax(shot_gate(shot_score), dim=-1)    # (NQ, N, K)
class_score = (beta * shot_score).sum(dim=-1)    # (NQ, N)

# Logits
logits = class_score
```

---

## 17. Main novelty statement

A clean contribution statement is:

> We propose SPIF-OTA, a single-branch few-shot inference framework that formulates query-class comparison as a physics-aware optimal transport problem between token measures. Unlike dual-branch global-local architectures, SPIF-OTA uses the transport plan itself to capture fine-grained correspondence while the resulting transport objective directly produces the class-level decision. To adapt OT to scalogram data, we introduce frequency-position-aware mass construction as a weak physics-informed inductive bias.

This statement is stronger than saying:

* “we replace SWD with OT”, or
* “we add a frequency bias”.

Because the true contribution is the **unified inference formulation**.

---

## 18. What should not be claimed

To keep the theory defensible, avoid these claims:

1. **Do not claim that low frequency is always more important.**
   Instead, say that discriminative information is often non-uniformly distributed across frequency bands, and the model learns a weak prior.

2. **Do not claim that OT is new by itself.**
   The novelty is in the way OT is used as a unified single-branch inference mechanism for few-shot scalogram classification.

3. **Do not claim that the transport plan is literally identical to transformer attention.**
   It is better described as a coupling induced by constrained transport.

---

## 19. Reviewer-facing justification

### 19.1 Why this is not just another metric head

Because SPIF-OTA does not merely substitute one scalar distance for another. It changes the decision mechanism from:

* representation compression + scalar comparison,

to:

* class-conditional structured alignment inference over token measures.

### 19.2 Why this is not a two-branch design in disguise

Because there is only one score generation pathway. The apparent “local” and “global” effects are both induced by the same transport optimization problem.

### 19.3 Why this is suitable for scalogram few-shot classification

Because scalograms have structured spatial semantics, and the proposed mass construction explicitly exploits this without requiring handcrafted rules.

---

## 20. Recommended final positioning

If the method is written carefully, the most accurate positioning is:

> SPIF-OTA is a single-branch, transport-based, physics-aware few-shot inference framework.

Its identity should be built around:

* **single-branch unified matching**,
* **optimal transport on token measures**,
* **physics-aware mass construction for scalogram structure**,
* **shot-aware evidence aggregation**.

That is the strongest and cleanest way to present the model.
