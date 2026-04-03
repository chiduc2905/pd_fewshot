# SC-LFI-v3 Implementation Design

This note is the implementation-level design for `sc_lfi_v3`.

It refines:

- `SC_LFI_V3_THEORY_NOTE.md`
- `SC_LFI_V3_DESIGN_REVIEW.md`
- `SC_LFI_V2_DIRECT_AUDIT.md`

into a concrete architecture that can be implemented in the current codebase.

## 1. Design Objective

`sc_lfi_v3` should preserve the original SC-LFI claim:

- each class is represented as a support-conditioned latent evidence distribution
- query-class scoring is a distribution-fit score

but replace the weak `v2` class construction:

- `anchor + generated particles`

with:

- a **posterior base measure**
- followed by a **support-basis residual transport pushforward**
- followed by **query-conditioned reliability reweighting during scoring**

This is the implementation form of:

- posterior evidence transport for few-shot classification

## 2. What Survives from v2

Only low-level or theory-neutral pieces survive:

- the episodic backbone/tokenizer pipeline from `fewshot_common.py`
- weighted latent evidence projection as a general idea
- the fixed-step Euler/Heun solver abstraction
- weighted transport kernels
- direct hard-negative distributional margin loss as a concept

These survive as utilities, not as the central architecture.

## 3. What Is Removed from v2

The following `v2` ideas are removed as central design choices:

- heuristic support/flow mixture weights
- class generation from pure Gaussian noise as the default path
- support anchor as the main support-preserving mechanism
- support-fit CE on the anchor branch
- support-only class measure with no query-conditioned class-side reweighting

## 4. Exact v3 Object

For each class `c` in an episode:

### 4.1 Support empirical evidence measure

Support tokens produce latent atoms and masses:

- `E_c = {e_{c,i}}`
- `a_{c,i} = softmax(w_{c,i}^{sup})`

Support empirical measure:

- `nu_c^sup = sum_i a_{c,i} delta_{e_{c,i}}`

### 4.2 Support-conditioned prior measure

From support/task context, predict a compact prior measure:

- `pi_c = sum_r b_{c,r} delta_{p_{c,r}}`

This prior is support-conditioned, not unconditional.

### 4.3 Shot-aware posterior base measure

Compute:

- `alpha_c = sigma(g_phi(h_c, t_ctx, uncert_c, log(1 + K)))`

where:

- `h_c` is class context
- `t_ctx` is episode context
- `uncert_c` summarizes support uncertainty
- `K` is shot count

Then define the posterior base measure:

- `mu_c^0 = alpha_c * nu_c^sup + (1 - alpha_c) * pi_c`

### 4.4 Posterior transport

Transport atoms of `mu_c^0` through a residual conditional flow:

- `muhat_c = (T_theta,c)_# mu_c^0`

The flow conditions on:

- class summary
- support memory
- episode context

This is a pushforward of the base measure, not Gaussian-noise generation.

### 4.5 Query evidence measure

For query `q`:

- `nu_q = sum_j rho_j delta_{u_j}`

with learned query masses `rho_j`.

### 4.6 Query-conditioned reliability reweighting

For each query-class pair:

- reweight class-side posterior masses using query-class compatibility

Result:

- `muhat_c^q = Reweight(muhat_c ; omega(q,c))`

Final score:

- `score_c(q) = -tau * D_score(nu_q, muhat_c^q)`

## 5. Concrete Module Layout

## 5.1 `posterior_context_v3.py`

Responsibilities:

- support latent evidence projection helpers
- support empirical measure bookkeeping
- support memory construction
- class summary construction
- uncertainty statistics
- support-conditioned prior atom generation
- shot-aware shrinkage coefficient
- posterior base measure assembly
- optional episode adaptation over class summaries

Outputs for each class:

- `support_atoms`
- `support_masses`
- `support_memory`
- `class_summary`
- `episode_context`
- `uncertainty_stats`
- `prior_atoms`
- `prior_masses`
- `alpha`
- `base_atoms`
- `base_masses`

Implementation choice:

- keep **full support latent atoms** in the empirical measure
- use memory tokens only for conditioning, not as a replacement for support atoms

Reason:

- this preserves the support basis explicitly and is much more few-shot-faithful than `v2`

## 5.2 `posterior_transport_flow_v3.py`

Responsibilities:

- time embedding
- support-memory and episode-context conditioned velocity field
- fixed-step Euler/Heun integration
- transport of base atoms
- optional base-atom jitter at training time
- conditional flow matching utilities

Key design:

- the flow transports `base_atoms`
- it does not generate a fresh class particle cloud from Gaussian noise

Conditioner inputs:

- atom state `y`
- time `t`
- class summary `h_c`
- support memory `M_c`
- episode context `t_ctx`

## 5.3 `query_conditioned_transport_v3.py`

Responsibilities:

- compute query pooled summary
- compute query-conditioned relevance logits over class atoms
- fuse base class masses with query-conditioned relevance
- produce reweighted class masses per query-class pair
- compute weighted transport distances
- optionally expose relevance entropy diagnostics

Key design:

- class atoms remain class-defined
- only masses are query-conditioned during scoring

This preserves the class-posterior semantics while making inference few-shot-specific.

## 5.4 `posterior_losses_v3.py`

Responsibilities:

- posterior flow matching loss
- posterior compatibility alignment loss
- hard-negative distributional margin loss
- mass entropy / anti-collapse regularization
- optional shrinkage regularization

## 5.5 `sc_lfi_v3.py`

Responsibilities:

- backbone/token pipeline
- call projector
- build posterior context
- transport posterior base measure
- compute query-conditioned class scores
- compute auxiliary losses
- return training/eval payloads with diagnostics

## 6. Exact Loss Design

The total loss will be:

- `L_total = L_cls + lambda_margin * L_margin + lambda_fm * L_fm + lambda_align * L_align + lambda_reg * L_reg`

### 6.1 `L_cls`

- standard CE on logits from distribution-fit scoring

### 6.2 `L_margin`

- hard-negative distributional margin on final pairwise distances

### 6.3 `L_fm`

Flow-matching loss should align the posterior base measure with support empirical evidence.

Implementation choice:

1. compute a weighted coupling from base atoms to support atoms using the alignment transport distance
2. obtain a barycentric target support atom for each base atom
3. define FM pairs:
   - source `x0 = base_atom`
   - target `x1 = barycentric_support_target`
4. use linear conditional path:
   - `y_t = (1 - t) x0 + t x1`
   - `u_t = x1 - x0`
5. regress the residual velocity field

Why this is better than `v2`:

- `v2` regressed from Gaussian noise to support tokens
- `v3` regresses from posterior base atoms toward support-consistent evidence targets

This is much closer to the posterior-evidence theory.

### 6.4 `L_align`

Alignment between posterior predictive measure and support empirical measure:

- `L_align = D_align(muhat_c, nu_c^sup)`

Use weighted entropic OT by default.

### 6.5 `L_reg`

Use only theory-motivated regularization:

- support/query mass entropy floor
- query-conditioned relevance entropy floor
- optional mild shrinkage regularization on `alpha` if it saturates too aggressively in `1-shot`

No decorative regularizers.

## 7. Overfitting Countermeasures Built into the Architecture

The model must fight the current `1-shot` failure mode structurally.

### 7.1 Support-prior shrinkage

In `1-shot`, the posterior base measure should not be dominated by the support empirical measure.

### 7.2 Explicit support basis preservation

The full support token measure remains part of the posterior object.

### 7.3 Query-conditioned class-side relevance

Support atoms irrelevant to the current query are downweighted during scoring.

### 7.4 Anti-collapse mass regularization

Mass heads must not collapse onto a single support patch.

### 7.5 Residual transport instead of free generation

The flow cannot wander far from a support-conditioned posterior base measure.

## 8. Fairness and Evaluation Defaults

Default mode must remain inductive and fair:

- no query-batch transductive refinement by default
- no extra data
- same episodic setup

Potential optional switches may be exposed later, but defaults should be fair.

## 9. Ablations Enabled by This Design

The implementation should make the following ablations easy:

- prior on/off
- flow on/off
- query-conditioned relevance on/off
- episode adapter on/off
- entropy regularization on/off
- solver type
- train/eval jitter amount
- alignment distance type

## 10. Final Implementation Rule

If any local design choice conflicts with the following statement, the design choice is wrong:

> the class should be inferred as a support-conditioned posterior predictive evidence measure whose scoring remains a transport fit problem, while being explicitly robust to low-shot uncertainty.
