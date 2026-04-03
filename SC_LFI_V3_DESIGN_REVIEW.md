# SC-LFI-v3 Design Review

This document is the required pre-implementation design review for the next SC-LFI redesign.

It should be read together with:

- `FEWSHOT_DISTRIBUTIONAL_SURVEY.md`
- `SC_LFI_V3_THEORY_NOTE.md`
- `SC_LFI_IMPLEMENTATION_AUDIT.md`
- `SC_LFI_V2_DESIGN_REVIEW.md`

The purpose of this note is to answer eight questions before coding:

1. Which parts of current SC-LFI are exact vs weak?
2. Which modules are kept unchanged?
3. Which modules are rewritten completely?
4. What exact formulas define the new model?
5. Which formulas are paper-grounded?
6. Which formulas are our few-shot adaptation?
7. Why is each new piece necessary?
8. Which weaknesses are fixed by each change?

## 1. Current SC-LFI-v2: Exact Parts vs Weak Parts

### 1.1 Exact or acceptable parts

These parts are worth preserving conceptually.

#### A. Core classifier claim

Current SC-LFI still preserves the intended central claim:

- each class is represented by a support-conditioned latent evidence distribution;
- query scoring is a distribution-fit score.

This remains the strongest part of the model and must stay.

#### B. Weighted latent evidence tokens

`v2` introduced learned support/query token masses.
That is correct in spirit.
Uniform evidence masses are too weak for a distributional classifier.

#### C. Stronger transport layer

`v2` replaced the weakest sliced Wasserstein path with stronger weighted transport layers.
This direction is correct and should remain.

#### D. Better numerical flow solver

Configurable Euler/Heun with separate train/eval budgets is the right numerical abstraction.

#### E. Direct distributional margin supervision

The hard-negative distribution margin loss is conceptually correct and should remain in some form.

### 1.2 Weak parts

These are the places where `v2` is still mathematically misaligned with strong few-shot design.

#### A. The class object is still too generator-centric

Even after improvements, the generated class measure still plays too central a role.
The model still feels like:

- support -> summary/memory -> generated class cloud

instead of:

- support -> posterior base measure -> posterior predictive refinement

This is the main theoretical weakness.

#### B. Support mixing is heuristic rather than posterior

The current class measure uses a learned support/flow mixture with bounded weights:

- `support_mix_min`
- `support_mix_max`

This is practical but weak theoretically.
It is a heuristic interpolation, not a principled posterior shrinkage mechanism.

#### C. The support anchor is still too support-image literal in `1-shot`

`v2` improved trainability, but the anchor path can still over-trust a single support image.
This hurts the exact regime that matters most.

#### D. Query conditioning is still too late and too weak

Current `v2` learns support/query masses independently and scores with transport afterward.
This is not enough.
Few-shot evidence selection should be query-conditioned during class inference or scoring.

#### E. The flow is still not few-shot-native enough

The flow sees support memory, but it still operates like a conditional particle generator.
That is better than `v1`, but not yet a truly few-shot posterior transport model.

#### F. Auxiliary support-fit CE is not theory-clean

The support-fit branch helped unstick training, but it is not the cleanest few-shot objective.
It behaves more like a stabilizer than a core principled term.

## 2. Which Current Modules Are Kept Unchanged

Only mathematically neutral components should be kept.

### 2.1 Keep unchanged or almost unchanged

#### A. Episodic backbone / tokenizer pipeline

Backbone feature extraction and episode tensor plumbing are neutral infrastructure as long as they expose local tokens.

#### B. Fixed-step ODE solver utility

The generic Euler/Heun fixed-step integration utility is numerically useful and theory-neutral.

#### C. Low-level weighted transport kernels if stable

The weighted Sinkhorn / weighted SW kernels can be retained as low-level primitives if:

- they support unequal support sizes,
- they support nonuniform masses,
- gradients are stable.

But they should be wrapped in a new scoring module.

### 2.2 Keep only as optional baselines or degenerate branches

#### A. Global prototype branch

Keep only as an optional stabilizer or degenerate mode.
It must not dominate the model.

#### B. Current support-fit CE branch

Keep temporarily only if needed during ablation.
It should not define the new theory.

## 3. Which Current Modules Are Rewritten Completely

The following pieces should be rewritten, not patched.

### 3.1 `sc_lfi_v2.py` -> full rewrite

Reason:

- current model is built around mixed anchor + generated particles;
- the new theory requires posterior base measure construction and posterior predictive transport.

### 3.2 `set_context_v2.py` -> rewrite as posterior context builder

Reason:

- context should output more than `h_c` and `M_c`;
- it must produce:
  - support evidence atoms,
  - compact support memory,
  - uncertainty statistics,
  - prior atoms / prior masses,
  - shot-aware shrinkage inputs.

### 3.3 `conditional_flow_v2.py` -> rewrite as residual posterior transport

Reason:

- the flow should operate on a support-prior posterior base measure;
- not on generic class particle generation from noise.

### 3.4 `flow_losses_v2.py` -> rewrite

Reason:

- current loss stack still contains support-fit stabilization terms not central to the new theory;
- the new losses must reflect posterior base measure refinement and query-conditioned transport discrimination.

### 3.5 `transport_distance_v2.py` -> partial rewrite

Reason:

- low-level OT/SW kernels may stay,
- but the new score must support query-conditioned reliability transport.

### 3.6 `latent_projector_v2.py` -> partial rewrite

Reason:

- token evidence projection is valid,
- but masses should produce both:
  - base evidence reliability,
  - uncertainty diagnostics,
  - and inputs for query-conditioned reweighting.

## 4. Exact Formulas of the New Model

This section defines the new target model mathematically.

## 4.1 Support evidence measure

For class `c`, let support tokens map to latent evidence atoms:

`E_c = { e_{c,i} }_{i=1}^{N_c}`, `e_{c,i} in R^{d_l}`

with masses:

`a_{c,i} = softmax_i( w_{c,i}^{sup} )`

The support empirical evidence measure is:

`nusup_c = sum_i a_{c,i} delta_{e_{c,i}}`

## 4.2 Support-conditioned prior measure

From support/task context, infer prior atoms:

`P_c = { p_{c,r} }_{r=1}^{R}`

with masses:

`b_{c,r} = softmax_r( v_{c,r} )`

The prior measure is:

`pi_c = sum_r b_{c,r} delta_{p_{c,r}}`

Important:

- `pi_c` is support-conditioned;
- it is not a base-class memory lookup;
- it is a meta-learned prior over plausible class evidence geometry.

## 4.3 Shot-aware shrinkage

Let:

- `K` be shot number,
- `u_c` be a support uncertainty statistic,
- `h_c` be a class summary,
- `t_ctx` be an episode context summary.

Define:

`alpha_c = sigma( g_phi( K, u_c, h_c, t_ctx ) )`

Then:

`mu_c^0 = alpha_c * nusup_c + (1 - alpha_c) * pi_c`

This is the posterior base measure.

## 4.4 Posterior evidence transport

Define a support-basis-conditioned residual velocity field:

`v_theta(y, t ; h_c, M_c, t_ctx)`

The posterior predictive class measure is:

`muhat_c = (T_theta,c)_# mu_c^0`

where `T_theta,c` is approximated by fixed-step integration.

## 4.5 Query evidence measure

For query `q`, latent evidence atoms are:

`U_q = { u_j }_{j=1}^{M_q}`

with masses:

`rho_j = softmax_j( w_j^{qry} )`

Then:

`nu_q = sum_j rho_j delta_{u_j}`

## 4.6 Query-conditioned reliability transport

Let `x_{c,l}` denote atoms of `muhat_c`.
Define query-conditioned class relevance logits:

`r_{c,l}(q) = psi_eta( u_q^pool, x_{c,l}, h_c )`

and normalized relevance:

`omega_{c,l}(q) = softmax_l( r_{c,l}(q) )`

The class-side measure used for scoring is:

`muhat_c^q = Reweight( muhat_c ; omega(q,c) )`

Finally:

`score_c(q) = - tau * D_score( nu_q, muhat_c^q )`

where `D_score` is weighted OT or weighted sliced Wasserstein.

## 4.7 Loss

The intended total loss is:

`L_total = L_cls + lambda_fm * L_fm + lambda_align * L_align + lambda_margin * L_margin + lambda_reg * L_reg`

with:

- `L_cls`: CE on distribution-fit logits
- `L_fm`: residual transport flow matching loss
- `L_align`: posterior predictive measure should remain compatible with support evidence
- `L_margin`: direct hard-negative distributional margin
- `L_reg`: uncertainty / entropy / shrinkage regularization

## 5. Which Formulas Are Paper-Grounded

These are grounded in prior literature, though not copied mechanically.

### 5.1 Task-level adaptation

Grounded in FEAT:

- a few-shot classifier should adapt class representations at the episode level.

### 5.2 Query-conditioned structural matching

Grounded in DeepEMD and CAN:

- support relevance should depend on the query;
- local structure matters.

### 5.3 Support-basis semantics

Grounded in DN4 and FRN:

- classes should preserve local evidence pools / bases rather than collapse immediately.

### 5.4 Probabilistic shrinkage and uncertainty control

Grounded in MetaQDA and Distribution Calibration:

- few-shot classification needs explicit uncertainty-aware prior structure;
- `1-shot` cannot trust support empirical distribution blindly.

### 5.5 Optional transductive refinement

Grounded in LaplacianShot, BECLR/OpTA, and Transductive CLIP:

- low-shot episodes may benefit from query-batch geometry or OT refinement.

## 6. Which Formulas Are Our Few-Shot Adaptation

These are the genuinely novel parts and should be presented as such.

### 6.1 Posterior base measure over latent evidence

The shrinkage measure:

`mu_c^0 = alpha_c * nusup_c + (1 - alpha_c) * pi_c`

is our explicit posterior-evidence formulation for SC-LFI.

### 6.2 Residual flow as posterior predictive transport

Using flow matching not as class generation from noise, but as:

`muhat_c = (T_theta,c)_# mu_c^0`

is our central generative adaptation.

### 6.3 Query-conditioned reweighted posterior predictive scoring

`muhat_c^q = Reweight( muhat_c ; omega(q,c) )`

keeps the class representation support-conditioned while making inference query-aware.
That is a new hybrid of support-posterior class modeling and local matching.

### 6.4 Unified special-case view

The fact that the model reduces conceptually toward:

- ProtoNet
- DeepEMD / DN4
- FRN
- MetaQDA-like probabilistic scoring

under explicit constraints is part of the theory strength.

## 7. Why Each New Piece Is Necessary

### 7.1 Support-conditioned prior measure

Necessary because:

- `1-shot` empirical support measure is too weak by itself;
- a prior is needed, but it must still be class-conditioned through support.

### 7.2 Shot-aware shrinkage coefficient

Necessary because:

- the model should not treat `1-shot` and `5-shot` identically;
- this distinction must be architectural, not incidental.

### 7.3 Residual posterior transport

Necessary because:

- we still want expressive latent class distributions;
- but they must remain tied to support evidence.

### 7.4 Query-conditioned reweighting

Necessary because:

- background/noisy atoms should not dominate scoring;
- the relevance of support evidence depends on the query.

### 7.5 Uncertainty regularization

Necessary because:

- learned masses can collapse onto brittle support patches;
- few-shot evidence selection needs calibrated uncertainty.

## 8. Which Weaknesses Are Fixed by Each Change

### Weakness 1: generator-centric class construction

Fix:

- posterior base measure + residual pushforward

### Weakness 2: heuristic support/flow interpolation

Fix:

- shot-aware posterior shrinkage coefficient `alpha_c`

### Weakness 3: over-trust of one support image in `1-shot`

Fix:

- support-prior shrinkage
- uncertainty-aware control
- query-conditioned relevance

### Weakness 4: weak query-conditioned evidence selection

Fix:

- query-conditioned class reweighting inside scoring

### Weakness 5: flow not few-shot-specific enough

Fix:

- support-basis residual transport instead of generic particle generation

### Weakness 6: support-fit CE branch is not theory-clean

Fix:

- replace it with a posterior compatibility objective and uncertainty-aware regularization

## 9. Implementation Consequences

The next implementation should likely create new modules such as:

- `net/sc_lfi_v3.py`
- `net/modules/posterior_context_v3.py`
- `net/modules/posterior_transport_flow_v3.py`
- `net/modules/query_conditioned_transport_v3.py`
- `net/modules/posterior_losses_v3.py`

Recommended semantics:

- `posterior_context_v3.py`
  builds support empirical measure, prior measure, uncertainty stats, and shrinkage coefficient inputs

- `posterior_transport_flow_v3.py`
  transports the posterior base measure into a posterior predictive measure

- `query_conditioned_transport_v3.py`
  applies query-conditioned relevance reweighting and final transport scoring

- `posterior_losses_v3.py`
  defines classifier loss, transport alignment, margin, and uncertainty regularizers

## 10. Final Decision

`SC-LFI-v3` should be implemented as a new model whose novelty is:

> few-shot classification as posterior evidence transport in latent space.

That is the correct way to:

- preserve the original SC-LFI theory,
- become genuinely few-shot-specific,
- and keep a contribution sharp enough to stand beside the strongest A-grade few-shot ideas instead of looking like a patched hybrid.
