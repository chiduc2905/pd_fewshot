# SC-LFI v2 Design Review

This note is the required pre-implementation design review for the next SC-LFI revision.

Target model name:

- `sc_lfi_v2`

Scientific objective kept unchanged:

- each class is represented as a support-conditioned latent evidence distribution;
- query-class scoring is a distribution-fit score, not prototype cosine.

The redesign is driven by the internal audit documents:

- `SC_LFI_DESIGN_NOTE.md`
- `SC_LFI_IMPLEMENTATION_AUDIT.md`

and by external reference implementations / papers:

- Meta `facebookresearch/flow_matching`
- VinAI `VinAIResearch/LFM`
- `nguyenngocbaocmt02/GENTLE`

## 1. Current SC-LFI: Exact Parts Vs Weak Parts

### 1.1 Exact or theory-faithful parts to preserve

These parts are already aligned with the intended theory and should be preserved conceptually:

1. Episodic backbone/token pipeline.
   - query/support token semantics are correct.
   - support tokens are concatenated class-wise exactly as intended.

2. Support-order invariance.
   - class context is permutation invariant.
   - support shot order does not affect inference.

3. Latent evidence space.
   - downstream flow and scoring happen in latent token space, not image space.

4. Linear flow-matching path adaptation.
   - current path:
     - `y_t = (1 - t) epsilon + t e`
     - `u_t = e - epsilon`
   - this is a legitimate few-shot adaptation of affine conditional probability paths.

5. Query-class distribution-fit scoring principle.
   - the main score is a transport/distribution discrepancy, not a prototype cosine.

6. Prototype as a degenerate special case.
   - current `use_flow_branch=False` mode already preserves this idea conceptually.

### 1.2 Acceptable approximations in v1, but not strong enough for v2

1. Finite particle approximation.
   - unavoidable, but must be better controlled.

2. Deterministic eval via fixed noise seed.
   - acceptable and should stay.

3. Compact class summary conditioning.
   - acceptable as one part of conditioning, but not as the only conditioner.

### 1.3 Theoretically weak or numerically weak parts that must be redesigned

1. Weak distance core.
   - `sc_lfi` v1 defaults to the repo's legacy `SlicedWassersteinDistance`.
   - the internal audit already identified this as too weak.
   - for a model whose classifier core is `D(nu_q, mu_c)`, weak `D` is a structural bottleneck.

2. Flow sampling is too crude.
   - current sampler is fixed-step Euler only.
   - no Heun / RK2 correction.
   - training and evaluation particle/step budgets are tied together.

3. Conditioning is over-compressed.
   - current flow sees only `h_c`.
   - multimodal support structure can be discarded before the flow branch even conditions on it.

4. Uniform token masses.
   - current support/query distributions are uniform empirical measures.
   - this is inconsistent with the evidence-weighting intuition needed for time-frequency tokens.

5. Direct discriminative pressure on the distribution branch is missing.
   - current flow branch gets support-side FM loss and indirect CE pressure only.
   - there is no direct margin pressure on query-to-class distances.

6. Alignment and scoring distances are forced into one shared abstraction.
   - this is convenient, but not theory-optimal.
   - classification and anchoring do not necessarily need the same transport geometry.

## 2. Which Current Modules Are Kept Unchanged

The following modules are mathematically neutral or already strong enough and can be reused unchanged.

### 2.1 Keep unchanged

1. `net/fewshot_common.py`
   - `BaseConv64FewShotModel`
   - `feature_map_to_tokens`
   - `merge_support_tokens`
   - reason:
     - these are neutral episodic/backbone utilities, not weak metric code.

2. `net/ssm/set_invariant_pool.py`
   - `SetInvariantMemoryPool`
   - reason:
     - already provides a permutation-invariant fixed-size memory set and summary;
     - this is directly useful for support memory construction in v2.

3. `net/metrics/sliced_wasserstein_paper.py`
   - `PaperSlicedWassersteinDistance`
   - reason:
     - it implements paper-style SW, exact 1D uniform OT under projection, and deterministic/fixed-vs-resample projection handling.

4. `net/metrics/sliced_wasserstein_weighted.py`
   - `WeightedPaperSlicedWassersteinDistance`
   - reason:
     - it is not the weak legacy metric;
     - it already computes exact projected weighted 1D OT under sorted supports;
     - it already supports nonuniform masses and unequal support sizes.

### 2.2 Keep conceptually, but not reuse implementation directly

1. v1 linear FM path idea.
   - keep the formula.
   - rewrite the surrounding flow code and conditioning.

2. v1 support smoothness idea.
   - keep only as optional.
   - rewrite the loss interface so it is not entangled with the old distance wrapper.

## 3. Which Current Modules Are Rewritten Completely

The following modules will be rewritten as v2-specific modules.

### 3.1 Rewrite completely

1. `net/modules/latent_projector_v2.py`
   - new projector with:
     - latent evidence head;
     - token reliability head.

2. `net/modules/set_context_v2.py`
   - new support conditioner with:
     - global weighted class summary `h_c`;
     - fixed-size support memory tokens `M_c`.

3. `net/modules/conditional_flow_v2.py`
   - new class-conditioned flow with:
     - path helpers;
     - memory-conditioned velocity field;
     - fixed-step solver abstraction;
     - Euler and Heun.

4. `net/modules/transport_distance_v2.py`
   - new transport layer with:
     - weighted paper-style SW wrapper for scoring;
     - weighted log-domain Sinkhorn OT for alignment;
     - batch pairwise query-class computation.

5. `net/modules/flow_losses_v2.py`
   - new loss helpers with:
     - FM loss;
     - weighted support anchoring;
     - direct distribution margin loss;
     - optional smoothness.

6. `net/sc_lfi_v2.py`
   - new main model.
   - do not mutate `sc_lfi.py`.

### 3.2 Partially reuse mathematical subroutines, not old architecture

1. Weighted paper-style SW.
   - reuse as a strong scoring primitive.
   - not as a legacy convenience wrapper.

2. SetInvariantMemoryPool.
   - reuse as support memory constructor inside a new conditioner.

## 4. Exact Formulas Of The New Model

This section defines the new model mathematically.

### 4.1 Backbone and tokenization

For any image `x`:

- `Z(x) = [z_1, ..., z_M]`, `z_m in R^d`

Support class token pool:

- `Z_c = concat_k Z(x_{c,k}) in R^{(K*M) x d}`

### 4.2 Latent evidence projector with token masses

For each token `z`:

- latent evidence:
  - `e = Psi(z) in R^{d_l}`
- raw token reliability logit:
  - `r = W_mass(e) in R`

For query image `q` with latent tokens `[u_j]`:

- `a_j^qry = softmax_j(r_j^qry)`

For support class `c` with latent tokens `[e_i^(c)]`:

- `a_i^sup = softmax_i(r_i^sup)`

This means both query and support are weighted empirical measures:

- `nu_q = {(u_j, a_j^qry)}`
- `nu_c^sup = {(e_i^(c), a_i^sup)}`

### 4.3 Support-conditioned class summary and memory

We build two conditioning objects from support latent evidence:

1. weighted global class summary:

   `h_c = Phi_summary(E_c, a_c^sup) in R^{d_h}`

2. fixed-size support memory set:

   `M_c = Phi_mem(E_c) in R^{R x d_l}`

where:

- `E_c = [e_1^(c), ..., e_{K*M}^(c)]`
- `Phi_mem` is permutation invariant over support token order.

The summary is weighted by support masses.
The memory set is permutation invariant but not collapsed to one vector.

### 4.4 Memory-conditioned velocity field

The new velocity field is:

`v_theta(y, t; h_c, M_c)`

Implemented as:

1. trunk state:

   `g = Trunk([y, gamma(t)])`

2. memory attention summary:

   `m = Attn(q = Q(g), K = K(M_c), V = V(M_c))`

3. global summary FiLM:

   `g' = FiLM(LN(g + m); h_c)`

4. residual velocity head:

   `v = Head(g')`

This keeps conditioning lightweight but makes the flow explicitly see support memory structure.

### 4.5 Flow-matching path

We keep the linear path:

- `y_t = (1 - t) epsilon + t e`
- `u_t = e - epsilon`

Reason:

- the Flow Matching repo formalizes training around conditional probability paths;
- our few-shot adaptation remains valid;
- the weakness in v1 was not the path itself, but the conditioner and the solver.

### 4.6 Solver

Sampling starts from:

- `y_0 ~ N(0, I)`

We expose fixed-step solvers:

1. Euler:

   `y_{k+1} = y_k + dt * v_theta(y_k, t_k; h_c, M_c)`

2. Heun / RK2:

   `k1 = v_theta(y_k, t_k; h_c, M_c)`

   `y_tilde = y_k + dt * k1`

   `k2 = v_theta(y_tilde, t_{k+1}; h_c, M_c)`

   `y_{k+1} = y_k + 0.5 * dt * (k1 + k2)`

Class particle measure:

- `muhat_c = {(particle_l^(c), b_l^(c))}_{l=1}^L`

In v2 initial implementation:

- particle masses `b_l^(c)` are uniform.

### 4.7 Query-class score

Core score stays:

`s_c(q) = - D_score(nu_q, muhat_c)`

where:

- `nu_q` is weighted by learned query masses;
- `muhat_c` is the generated class particle measure;
- `D_score` is a strong weighted transport distance.

### 4.8 Transport distances

We separate two roles.

1. Classification distance:

   `D_score = weighted paper-style sliced Wasserstein`

2. Alignment distance:

   `D_align = weighted entropic OT`

Reason:

- scoring needs a sharp, discriminative transport geometry;
- anchoring benefits from smooth gradients and stable mass-aware matching.

### 4.9 Losses

Classification CE:

- `L_cls = CE(logits, y_true)`

Flow matching:

- `L_FM = E || v_theta(y_t, t; h_c, M_c) - (e - epsilon) ||_2^2`

Support anchoring:

- `L_align = D_align(muhat_c, nu_c^sup)`

Direct distributional margin:

- let:
  - `d_true = D_score(nu_q, muhat_{y})`
  - `d_neg = min_{c != y} D_score(nu_q, muhat_c)`

- define:
  - `L_margin = mean max(0, margin + d_true - d_neg)`

Optional smoothness:

- `L_smooth = sum_{(c,c')} w_{cc'} D_align(muhat_c, muhat_c')`

Total:

- `L_total = L_cls + lambda_fm L_FM + lambda_align L_align + lambda_margin L_margin + lambda_smooth L_smooth`

## 5. Which Formulas Are Paper-Grounded

### 5.1 Paper-grounded formulas

1. Flow matching on a fixed conditional probability path.
   - grounded in `facebookresearch/flow_matching`.

2. Affine / linear path view.
   - grounded in the `AffineProbPath` abstraction:
     - `X_t = alpha_t X_1 + sigma_t X_0`
   - our linear path is a special case of this philosophy.

3. Solver separation from path/model.
   - grounded in the official `ODESolver` abstraction from `flow_matching`.

4. Heun/Euler fixed-step sampling for flow models.
   - grounded in LFM practice and explicit Heun/Euler samplers in `VinAIResearch/LFM`.

5. Conditional distribution learning under low-sample regimes requiring regularized matching.
   - grounded in GENTLE's conditional-distribution perspective.

6. Entropic OT as a smooth discrepancy.
   - grounded in GENTLE's entropic transport viewpoint.

7. Exact projected 1D OT inside sliced Wasserstein.
   - grounded in paper-style SW practice and in the existing exact projected OT code already present in the repo.

## 6. Which Formulas Are Our Few-Shot Adaptation

These are not copied equations from those references. They are our few-shot design choices.

1. Treating each few-shot class as a support-conditioned latent evidence distribution.

2. Conditioning the flow on:
   - weighted global summary `h_c`
   - plus compact support memory `M_c`

3. Using latent token reliability masses:
   - `a_i^sup`
   - `a_j^qry`

4. Using weighted query empirical measures against generated class particle measures for classification.

5. Direct query-to-hard-negative distribution margin loss on top of CE.

6. Using weighted support barycenter as the degenerate prototype-like mode when the flow branch is disabled.

## 7. Why Each New Piece Is Necessary

### 7.1 Learned query/support masses

Necessary because:

- the current model treats all evidence tokens uniformly;
- time-frequency evidence is not uniformly informative;
- transport over uniform masses dilutes useful discriminative evidence.

### 7.2 Support memory `M_c`

Necessary because:

- conditioning on only `h_c` can erase multimodal support structure;
- the flow should see a compact but structured class memory, not only a summary scalar/vector.

### 7.3 Strong transport core

Necessary because:

- the classifier itself is a transport discrepancy;
- weak `D` weakens the whole method, not just an auxiliary regularizer.

### 7.4 Separate scoring distance and alignment distance

Necessary because:

- scoring and anchoring serve different roles;
- sharp scoring geometry and smooth alignment geometry do not always coincide.

### 7.5 Heun / RK2 solver

Necessary because:

- Euler-only integration is too crude for a transport-defined class distribution;
- improved solver fidelity is a direct numerical upgrade to the core generative measure.

### 7.6 Direct margin loss

Necessary because:

- the current flow branch receives too little explicit discriminative signal;
- CE alone is not the cleanest distribution-level supervision.

## 8. Which Weaknesses From The Audit Are Fixed By Each Change

| Weakness from audit | v2 change | Why it fixes the weakness |
| --- | --- | --- |
| Weak legacy SW core | use weighted paper-style SW for scoring | makes classifier use exact projected weighted OT instead of heuristic legacy SW |
| Crude Euler-only flow sampling | add solver abstraction with Euler and Heun, separate train/eval budgets | improves particle transport fidelity and evaluation stability |
| Flow conditioned only on `h_c` | add support memory `M_c` and memory-attentive velocity field | preserves multimodal support structure in the conditioner |
| Uniform token masses | add learned token mass head for support/query | lets evidence measures emphasize informative tokens |
| Flow gets only indirect discriminative pressure | add distributional hard-negative margin loss | directly optimizes class separation in transport space |
| Same distance used for all roles | separate `D_score` and `D_align` | sharper scoring, smoother anchoring |
| Weak degenerate special case | define weighted support barycenter degenerate mode | preserves prototype-like reduction in a cleaner weighted-measure form |

## 9. Implementation Decision Summary

The new model will therefore:

1. keep the few-shot episodic backbone path;
2. keep the linear FM path;
3. replace the weak scoring core with a weighted strong transport layer;
4. add learned evidence masses;
5. add support memory conditioning;
6. replace Euler-only sampling with configurable Euler/Heun;
7. add a direct distribution margin loss;
8. preserve support-order invariance and prototype-like degeneracy.

This is the design that will be implemented next under:

- `sc_lfi_v2`
