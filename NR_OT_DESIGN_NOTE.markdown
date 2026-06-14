# NR-OT: Nuisance-Referenced Optimal Transport for Few-Shot PD Scalograms

**Status:** implemented (`--model ours_final --enable_nr_ot`), theory + sanity
checks only; no benchmark results yet.

**One-line identity.** Instead of asking *"does the query match class c cheaply?"*,
NR-OT asks *"does the query match class c cheaply **relative to** matching the
shared episode background?"* — and spends its transport budget on the tokens
that **depart from that background**.

---

## 1. The data, and why it breaks uniform-marginal UOT

Each wavelet scalogram (`dataset/scalogram_27_1`, classes `corona / internal /
notpd / surface`) is, latently,

```
mu  ≈  (1 - s) · B   +   s · S_c ,        s ≪ 1
```

* `B` — a dense **vertical-streak background** (the AC carrier / phase comb).
  Direct inspection of the images shows it occupies > 90 % of the token grid
  and is **near-identical across all four classes**.
* `S_c` — a **sparse discriminative signature**: the bright PD "flame". The
  class is defined by *where in frequency* it sits, *how many* blobs, and *how
  spread* they are. It is a small minority of tokens.

Ours-Final scores a query/support pair with a fixed-budget KL-relaxed UOT over
L2-normalized local tokens, uniform marginals `a_i = b_j = rho / L`, squared
Euclidean cost, and the threshold-mass evidence `E = T·M − C`
(`OURS_FINAL_MODEL_SPEC.markdown`). On this data that regime has five coupled
failure modes:

| | Failure | Mechanism |
|---|---|---|
| **P1** | **Common-mode domination** | `B`-tokens are near-collinear after L2-norm ⇒ `B→B` cost ≈ 0. Most mass flows `B→B`, so `E ≈ T·M_background` — a large term that is **invariant across classes**. The signal `S_q→S_c` is a tiny perturbation drowned by background mass ⇒ class logits barely separate. |
| **P2** | **Relaxation removes the wrong mass** | KL relaxation (`τ=0.5`) drops mass to cut cost. The *expensive* mass is exactly the signal tokens lacking a cheap partner; the *cheap* mass is the background. UOT therefore **suppresses signal and keeps nuisance** — backwards. |
| **P3** | **Entropic blur** | `ε=0.1` spreads each token's mass over many partners; the sparse flame leaks onto background neighbours. |
| **P4** | **Absolute-feature cost ignores structure** | Class identity is a *relative* geometric property (flame position/count/spread vs. background), not absolute appearance. Cosine cost cannot express "match by relative structure". |
| **P5** | **Why SCI / global-residual did not help** | SCI (depthwise conv) edits the **cost** of individual pairs but never touches the **marginal/score** layer where P1/P2 live; on the GAP global-residual its local perturbations are averaged away. The GAP global residual is itself a mean over **all** tokens ⇒ it has the *same* P1 disease (every class's prototype ≈ the background mean). Both are correct as written but address the wrong layer. |

A concrete demonstration of P1 (random-init tiny model, 3-way): base
Ours-Final logits for one query are `[-5.81, -5.87, -5.76]` — essentially
flat. NR-OT turns the same tokens into `[-0.07, -0.39, +0.46]`.

---

## 2. The fix: an explicit, episode-level nuisance reference

The literature on robust / partial / foreground-aware transport (§6) converges
on one principle: **model the nuisance explicitly and remove it**, rather than
hoping the solver discards it. NR-OT instantiates this with a reference that is
**free, class-agnostic, and already present in every episode**.

### (A) The leave-class-out background measure `β_{-c}`

For class `c`, pool the support tokens of **every other class**:

```
β_{-c} = uniform empirical measure over  { s_{c',k,j} : c' ≠ c }      (M_b = (W−1)·K·L atoms)
```

Because class signatures `S_{c'}` differ across ways, pooling them **averages
the signatures out** and leaves the **shared background** `B`. So `β_{-c} ≈ B`
with no segmentation network and no PD prior — it is a learned-free *null /
background hypothesis* read directly off the episode. (For PD specifically,
this null hypothesis is also a natural model of the `notpd` class.)

### (B) Background-novelty marginals

A token's transport mass is proportional to its **departure from `β_{-c}`**:

```
novelty(i) = 1 − max_b ⟨ q_i , β_{-c,b} ⟩            (= ½ · nearest-neighbour squared distance, unit tokens)
z(i)       = (novelty(i) − mean_i) / std_i           (per-image standardization)
a_q(i)     = rho · softmax_i( z(i) / τ_nov )
```

and identically for the class-`c` support tokens → `b_c(j)`. The
**standardization** turns novelty into a *relative anomaly score* ("how many
std above the typical token is this token's departure from the background"), so
the temperature `τ_nov` has a consistent meaning independent of feature scale /
dimensionality. Background tokens cluster at `z≈0` and receive ≈0 mass; the
flame sits in the positive tail and captures the budget. This **inverts P2**:
the solver now spends `rho` on signal, not nuisance. (Sanity check: top-20 %
mass fraction rises from the uniform `0.20` to `≈0.47`.)

> **Novelty is leave-class-out on purpose.** When scoring class `c`, a query
> token is "novel" only if it is unexplained by the *other* classes. A corona
> flame is novel w.r.t. `{internal, notpd, surface}` (→ gets mass when scoring
> corona) but **not** novel w.r.t. `{corona, …}` (→ suppressed when scoring
> surface). This makes the marginal itself a one-vs-rest contrastive object.

### (C) Common-mode debiased score

Run **two** UOT problems with the **same** query measure `(q, a_q)` and switch
only the target, then subtract:

```
E(μ, ν) = T·M(μ,ν) − C(μ,ν)                         (the Ours-Final evidence functional)

score_c = score_scale · [ E(μ_q, μ_c) − E(μ_q, β_{-c}) ]
```

with `T = softplus(raw_threshold)` learned (init = Ours-Final's `1/score_scale`).
Logits are row-centered before fusion, exactly like the global prototype head.

**Common-mode cancellation (informal theorem).** Suppose the query and every
class share the same background sub-measure `B`, and `β_{-c} ≈ B`. Under the
novelty marginal `a_q`, the query's transported mass is dominated by signal
tokens. Then:

* **True class.** Signal → class-`c` signal is cheap (`E(μ_q,μ_c)` large), while
  signal → generic background is expensive (`E(μ_q,β_{-c})` small) ⇒ **large
  positive `score_c`**.
* **Wrong class.** Signal → class-`c` is expensive *and* signal → background is
  expensive ⇒ both terms small ⇒ **`score_c ≈ 0`**.
* **Background-only query (`notpd`).** No token is novel ⇒ both transports
  reflect background and **cancel** ⇒ `score_c ≈ 0` for every PD class, so the
  episode resolves toward the background/`notpd` hypothesis by default.

The subtraction removes precisely the class-invariant `T·M_background` term that
caused **P1**, leaving the evidence that is *specific to class `c` above the
background hypothesis*. This is a Sinkhorn-divergence-style debiasing
(Feydy et al. 2019) where the reference is the **episode background** instead of
the usual self-transport term.

### (D) Optional structural cost — Fused-Partial Gromov–Wasserstein *(not enabled)*

To attack **P4**, the feature cost can be fused with an intra-image structural
(Gromov–Wasserstein) term and a partial-mass constraint
(Fused-Partial GW, arXiv 2502.09934, 2025):

```
C_fused[i,j] = (1−γ)·‖q_i − s_j‖²  +  γ · Σ_{i',j'} |D_q(i,i') − D_s(j,j')|² · π_{i',j'}
```

so that "bright region sitting low, isolated" matches across intensity changes.
Left as a flag for future work because GW solvers are heavier and the
novelty-conditioning is the higher-confidence contribution.

---

## 3. Why this is not a data bias

* `β_{-c}`, the novelty marginal, and the debiasing are **entirely
  class-agnostic**: they never encode what a PD flame looks like (frequency,
  phase, shape). They only encode the *episode-relative* statement "this token
  departs from the shared background." Everything is learned end-to-end through
  the OT evidence loss; the single learnable parameter is the threshold `T`.
* The only structural assumption is the **general physical prior** that genuine
  signals depart from a shared nuisance floor — the same assumption underlying
  all foreground-aware / robust transport.

---

## 4. Relationship to prior art and what is novel here

| Approach (in-repo / literature) | What it does | NR-OT's difference |
|---|---|---|
| **Ours-Final global residual** (`_global_prototype_logits`) | add `0.1 · cos(GAP_q, GAP_c)` | GAP is a mean over *all* tokens ⇒ background-dominated (P1). NR-OT replaces this naive residual with a **principled debiasing**: *subtract* the background hypothesis instead of *adding* a background-dominated prototype. |
| **ECT-UOT** (`_apply_episode_contrastive_uot`) | gates the **cost matrix** when *both* ways match a query token cheaply | implicit, pairwise common-mode detection; **no explicit background measure, no debiasing reference transport**. NR-OT builds `β_{-c}` as a first-class measure and debiases at the **score** layer. |
| **TAM** (token attention marginal) | `a_i ∝ softmax(v·z_i)` | feature-only saliency, **background-unaware**. NR-OT's mass is departure-from-`β`. |
| **SCI** | depthwise-conv token context | edits cost, not marginal/score (P5). |
| **DeepEMD** cross-reference | bilinear query↔support patch weighting | feature-compatibility weighting; no nuisance reference, no debiasing. |
| **FOCT** (ACM MM 2024) | foreground-aware conditional transport, needs a segmentation model | NR-OT gets the foreground/background split *for free* from the leave-class-out episode pool. |
| **MUOT-CLIP / NA-MVP** (2025) | UOT to suppress unreliable image regions | suppression via UOT relaxation only; NR-OT adds explicit `β` + debiasing. |
| **Robust / partial OT** (α-OPT, λ-ROT, outlier-robust OT) | discard a *fixed* fraction of outlier mass | NR-OT does not fix a discard fraction; the novelty marginal *learns* which mass is nuisance per episode, and the reference is a *positive* comparison, not a discard. |
| **Sinkhorn divergence** (Feydy 2019) | debias by subtracting *self*-transport `OT(α,α)` | NR-OT subtracts an **episode-background** transport — a task-meaningful null hypothesis, not the statistical self-term. |

**Novelty statement.** *NR-OT is the first formulation to use a leave-class-out
episode background as an explicit reference measure that simultaneously (i)
defines novelty-weighted transport marginals and (ii) debiases the transport
evidence score, turning few-shot PD recognition into nuisance-referenced
partial transport.*

---

## 5. Implementation

* New module: `net/modules/nuisance_referenced_ot.py` (`NuisanceReferencedOT`).
  Self-contained; reuses the **verified** `sinkhorn_unbalanced_log`,
  `compute_transport_cost`, `compute_transported_mass` from
  `net/modules/unbalanced_ot.py` (theory-checked: it is the standard entropic
  KL-UOT scaling with damping `ρ=τ/(τ+ε)`, matching POT's
  `sinkhorn_unbalanced(reg_type="entropy")`).
* Wiring (`net/ours.py`): instantiated in `OursM2.__init__`; fused in
  `_apply_nr_ot_score` at the same payload-assembly point as the global
  residual. Shares Ours-Final's transport regime exactly (`ε=0.1`, `τ_q=τ_c=0.5`,
  `rho=0.8`, `score_scale=16`, solver iters/tol). Diagnostics surface via
  `_last_nr_ot_diagnostics`.
* CLI (`main.py`, validated *Ours-Final only* in `model_factory.py`):

```bash
python main.py --model ours_final --enable_nr_ot \
  --nr_ot_mode standalone \      # or: residual (adds onto base local UOT logits)
  --nr_ot_weight 1.0 \           # used only in residual mode
  --nr_ot_novelty_temp 0.5 \     # softmax temp on standardized novelty
  --nr_ot_uniform_reference true # uniform background marginal in the reference transport
```

`--enable_nr_ot` is **mutually exclusive** with `--enable_global_residual_score`
(NR-OT is the principled replacement). Requires `way_num ≥ 2` (to form `β_{-c}`).

### Sanity checks (`tests/test_nuisance_referenced_ot.py`, all passing)

* shapes / finite logits / row-centering / finite threshold gradient;
* novelty marginals concentrate (top-20 % mass `> 0.35` vs uniform `0.20`);
* classifies a synthetic *background + sparse-flame* episode (acc `> 0.8`);
* `way_num=1` raises; standalone changes logits vs base and surfaces
  diagnostics; mutual exclusion with the global residual is enforced.

---

## 6. References

1. Chizat, Peyré, Schmitzer, Vialard. *Scaling algorithms for unbalanced optimal
   transport problems.* Math. Comp., 2018. — the reused KL-UOT Sinkhorn scaling.
2. Séjourné, Vialard, Peyré. *Unbalanced Optimal Transport, from theory to
   numerics.* 2023.
3. Feydy et al. *Interpolating between Optimal Transport and MMD using Sinkhorn
   divergences.* AISTATS 2019. — debiasing by subtracting a reference transport.
4. FOCT: *Few-shot Industrial Anomaly Detection with Foreground-aware Online
   Conditional Transport.* ACM MM 2024.
   <https://dl.acm.org/doi/10.1145/3664647.3680771>
5. MUOT-CLIP: *Enhancing Few-Shot Adaptation of CLIP via Inter- and
   Intra-Modality Unbalanced Optimal Transport.* 2025.
   <https://openreview.net/forum?id=BEOq3YB5WM>
6. NA-MVP: *Noise-Aware Few-Shot Learning through Bi-directional Multi-View
   Prompt Alignment.* 2026. <https://arxiv.org/abs/2603.11617>
7. *Fused Partial Gromov–Wasserstein for Structured Objects.* 2025.
   <https://arxiv.org/abs/2502.09934>
8. Mukherjee et al. *On making optimal transport robust to all outliers.*
   <https://arxiv.org/abs/2206.11988>; Nietert et al. *Outlier-Robust Optimal
   Transport.* — partial/robust transport that discards outlier mass.
9. Zhang et al. *DeepEMD: Few-Shot Image Classification with Differentiable
   Earth Mover's Distance and Structured Classifiers.* CVPR 2020. — cross-
   reference baseline contrasted in §4.
10. Zheng, Zhou. *Comparing Probability Distributions with Conditional
    Transport.* 2021. — conditional transport used by FOCT.
