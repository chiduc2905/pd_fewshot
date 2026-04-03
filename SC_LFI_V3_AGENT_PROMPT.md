# SC-LFI-v3 Agent Prompt

Use this prompt as the implementation brief for the coding agent that will redesign and implement the next model.

The target is **not** to patch `sc_lfi_v2`.
The target is to build a new model whose theory is clearer, whose few-shot inductive bias is much stronger, and whose low-shot behavior is materially better than `v2`, especially in `1-shot`.

---

## ROLE

You are a research engineer, theory-first model designer, and few-shot specialist.

Your task is to redesign and implement a new model:

- `sc_lfi_v3`

You must prioritize:

1. **scientific correctness**
2. **few-shot suitability**
3. **low-shot generalization**
4. **clear novelty**
5. **clean runnable code**

This is not a "make it compile" task.
This is a "make the model genuinely stronger and more publishable" task.

---

## 0. HARD REQUIREMENT

You must **not** blindly inherit the architecture of `sc_lfi_v2`.

You must start from theory and few-shot inductive bias, then implement the correct architecture.

You may reuse only low-level utilities that are mathematically neutral, such as:

- backbone tokenization
- stable weighted transport kernels
- fixed-step Euler/Heun solver utility

You must **not** preserve any central `v2` design choice just because it already exists in code.

If a `v2` component is only a stabilizer, heuristic, or transitional approximation, rewrite or remove it.

---

## 1. DOCUMENTS YOU MUST READ FIRST

Before coding, read these local documents carefully:

- `SC_LFI_DESIGN_NOTE.md`
- `SC_LFI_IMPLEMENTATION_AUDIT.md`
- `SC_LFI_V2_DESIGN_REVIEW.md`
- `SC_LFI_V2_DIRECT_AUDIT.md`
- `FEWSHOT_DISTRIBUTIONAL_SURVEY.md`
- `SC_LFI_V3_THEORY_NOTE.md`
- `SC_LFI_V3_DESIGN_REVIEW.md`

You must extract:

- the original SC-LFI core claim
- the exact `v2` formulas
- the reasons `v2` still overfits and fails in `1-shot`
- the new `v3` theoretical object
- the intended contribution / novelty statement

---

## 2. NON-NEGOTIABLE SCIENTIFIC GOAL

The model must preserve the original SC-LFI core:

> each class is represented as a support-conditioned latent evidence distribution, and query-class scoring is a distribution-fit score.

But the model must upgrade this into the following stronger formulation:

> few-shot classification is posterior evidence transport in latent space.

That means the class object is **not** merely:

- a prototype,
- a support summary vector,
- or a generated cloud from Gaussian noise.

The class object must be:

- a **support-conditioned posterior predictive latent evidence measure**

and the flow must serve as:

- a **posterior transport operator**

not as the main novelty by itself.

---

## 3. THE REAL FAILURE TO FIX

`sc_lfi_v2` is no longer weak in the same way as `v1`, but it still fails where it matters:

- it overfits in `1-shot`
- it does not scale with shot strongly enough
- it still defines the class too much as `anchor + generated particles`
- it has no proper shot-aware posterior shrinkage
- it lacks strong query-conditioned evidence selection during scoring

You must treat this as a **structural design failure**, not a tuning failure.

Do not answer this with:

- minor regularization
- early stopping hacks
- small loss-weight changes
- or a different default hyperparameter set

You must fix the **class inference mechanism itself**.

---

## 4. REDESIGN PRINCIPLE

You must redesign the model around this object:

### 4.1 Support empirical latent evidence measure

For support class `c`, extract latent evidence atoms:

`E_c = {e_{c,i}}`

with learned masses:

`a_{c,i} >= 0, sum_i a_{c,i} = 1`

giving:

`nusup_c = sum_i a_{c,i} delta_{e_{c,i}}`

### 4.2 Support-conditioned prior measure

Infer a small support-conditioned prior measure:

`pi_c = sum_r b_{c,r} delta_{p_{c,r}}`

This prior is:

- class-specific to the current support set
- meta-learned
- compact
- not a lookup table over base classes

### 4.3 Shot-aware posterior base measure

Define:

`mu_c^0 = alpha_c * nusup_c + (1 - alpha_c) * pi_c`

where:

- `alpha_c` is shot-aware and uncertainty-aware
- `alpha_c` must behave differently in `1-shot` vs `5-shot`

This is the core fix that makes the method genuinely few-shot-specific.

### 4.4 Posterior residual transport

Define a support-basis-conditioned transport operator:

`muhat_c = (T_theta,c)_# mu_c^0`

The flow parameterization should be used to implement `T_theta,c`, but the measure-theoretic view must come first.

### 4.5 Query-conditioned transport scoring

For query `q`, build:

`nu_q = sum_j rho_j delta_{u_j}`

Then classify by:

`score_c(q) = - tau * D( nu_q, muhat_c^q )`

where `muhat_c^q` is a query-conditioned reweighting of the posterior predictive class measure.

This is essential.
Without query-conditioned evidence selection, the model remains too weak for few-shot local structure.

---

## 5. WHAT THE NEW NOVELTY MUST BE

The novelty must not be:

- "flow matching + OT scoring + support memory"

That is not sharp enough.

The novelty must be articulated as:

1. **Posterior Evidence Transport for Few-Shot Classification**
2. **Shot-Aware Support-Prior Shrinkage Measure**
3. **Support-Basis Residual Flow Pushforward**
4. **Query-Conditioned Reliability Transport Scoring**

The model should be clearly distinguishable from:

- FEAT
- DeepEMD
- FRN
- MetaQDA
- Distribution Calibration
- LaplacianShot

while still learning from them.

Your implementation must preserve this novelty story.

---

## 6. FEW-SHOT DESIGN CONSTRAINTS

The architecture must obey the following few-shot constraints.

### 6.1 Support must remain a basis, not just a summary

You must preserve local support evidence atoms or a compact basis derived from them.

Do not collapse support into one vector early.

### 6.2 `1-shot` and `5-shot` must be treated differently by design

This must happen through architecture, not only loss weights.

At minimum:

- `alpha_c` must depend on shot number and uncertainty
- prior reliance must be stronger in `1-shot`
- support empirical dominance must increase with shot

### 6.3 Query-conditioned support relevance is required

Scoring must include a query-conditioned reweighting step on the class-side evidence.

This reweighting must be:

- permutation-invariant over support order
- local-structure aware
- lightweight enough for episodic evaluation

### 6.4 The model must not secretly collapse back to prototype cosine

Prototype mode may exist as a degenerate branch for testing and ablation.
It must be disabled by default and must not dominate the architecture.

### 6.5 The model must remain fair

Fairness requirements:

- same episodic backbone pipeline unless mathematically necessary
- no extra external data
- no language supervision
- no transductive query-batch tricks enabled by default if the benchmark is inductive

Optional transductive refinement may be exposed as a separate switch, but default evaluation must remain fair and inductive.

---

## 7. REQUIRED ARCHITECTURE

Implement the new model under a new name:

- `sc_lfi_v3`

### 7.1 Backbone and tokenization

Reuse only the neutral episodic backbone/tokenizer path.

Output:

- image tokens `Z(x) in R^{M x d}`

### 7.2 Latent evidence projector

Rewrite or extend the current projector to output:

- latent evidence atoms `e`
- base reliability masses
- uncertainty diagnostics if useful

Output should include:

- `latent_tokens`
- `mass_logits`
- `masses`
- optionally `mass_entropy` or uncertainty summary hooks

### 7.3 Episode adapter

Add a lightweight FEAT-style task adapter before final class posterior construction.

This adapter must:

- be permutation-invariant over class support ordering
- adapt compact class context using episode-level information
- remain lightweight

Do not add a giant transformer.

### 7.4 Posterior context builder

Create a new module that produces for each class:

- support empirical measure atoms and masses
- compact support memory / basis
- class summary
- uncertainty statistics
- support-conditioned prior atoms and prior masses
- shot-aware shrinkage coefficient inputs and final `alpha_c`

This module is central.

### 7.5 Posterior transport flow

The flow must operate on the posterior base measure:

- not on pure Gaussian noise by default

The initial atoms for transport should be sampled or taken from:

- the posterior base measure `mu_c^0`

Optional small noise jitter around base atoms is allowed during training.

The flow should be a residual transport operator, not an unconstrained generator.

Expose:

- `solver_type in {euler, heun}`
- separate train/eval step counts
- separate train/eval particle budgets if you resample

### 7.6 Query-conditioned reliability scoring

Implement a module that, for each query-class pair:

- computes query-conditioned relevance over class-side atoms
- reweights the posterior predictive class measure
- computes weighted transport distance

This should absorb the strongest lesson from DeepEMD/CAN/FRN, but remain consistent with the SC-LFI class-posterior formulation.

### 7.7 Optional transductive refinement

You may expose an optional refinement mode for future use, but:

- it must be off by default
- it must not contaminate the inductive core result

---

## 8. REQUIRED LOSS DESIGN

The loss stack must be theory-consistent and few-shot-consistent.

Do **not** keep `v2` losses mechanically.

The loss should be:

`L_total = L_cls + lambda_margin * L_margin + lambda_fm * L_fm + lambda_align * L_align + lambda_reg * L_reg`

You must define each term clearly.

### 8.1 `L_cls`: distribution-fit classification CE

Main inductive loss:

- logits from query-to-class posterior transport fit
- CE over classes

This remains the primary classification loss.

### 8.2 `L_margin`: direct hard-negative distributional margin

Required.

Use:

`L_margin = mean relu(m + D_true - D_hardneg)`

This is the clean discriminative pressure that `v1` lacked.

### 8.3 `L_fm`: posterior transport flow-matching loss

This must be redesigned carefully.

Do **not** just reuse support-token FM from `v2`.

Preferred design:

- construct source atoms from the posterior base measure `mu_c^0`
- construct target atoms from support empirical evidence `nusup_c`
- build a soft coupling or matching between them
- train the residual transport field with conditional flow matching on these pairs

This keeps the FM objective aligned with the posterior-evidence theory.

The path can remain linear:

`y_t = (1 - t) x_0 + t x_1`

`u_t = x_1 - x_0`

if justified cleanly.

### 8.4 `L_align`: posterior compatibility loss

This loss should ensure the posterior predictive class measure does not drift away from support evidence.

But do **not** simply keep the old support-fit CE anchor branch as the main solution.

Preferred form:

- alignment between posterior predictive measure and support empirical measure under an alignment transport distance

This should be weaker and cleaner than forcing CE directly on the support anchor branch.

### 8.5 `L_reg`: uncertainty / anti-overfit regularization

This must be few-shot-aware and minimal, not decorative.

At least include:

- entropy floor or anti-collapse regularization on token/relevance masses
- optional weak-support consistency regularization under light augmentation or dropout perturbation

The purpose is to stop `1-shot` support atoms from collapsing to a brittle patch explanation.

### 8.6 Shot-aware weighting

You may make some regularization terms or shrinkage behavior depend on shot.

But this must be principled and clearly documented.

Do not hardcode arbitrary behavior without explanation.

---

## 9. OVERFITTING IS A FIRST-CLASS DESIGN TARGET

You must explicitly design against the current `1-shot` overfitting.

Required anti-overfit mechanisms should come from the theory, not hacks:

1. shot-aware support-prior shrinkage
2. query-conditioned evidence selection
3. uncertainty-aware masses
4. posterior compatibility instead of anchor memorization
5. support-basis residual transport, not free generation

You may also expose optional practical controls such as:

- partial backbone freezing for very low-data regimes
- dropout on support memory
- weak support perturbation consistency

But these must be secondary.
The main fix must be architectural.

---

## 10. WHAT NOT TO DO

Do not do any of the following:

- do not patch `sc_lfi_v2` in place
- do not keep `support_mix_min / support_mix_max` as the central idea
- do not generate class particles from pure Gaussian noise as the default class inference path
- do not make the model secretly reduce to prototype cosine by default
- do not rely on heavy transductive tricks to rescue weak inductive behavior
- do not add large fashionable modules with unclear role
- do not add regularizers just because they might help a little

Every major component must have a theory reason.

---

## 11. REQUIRED MODULES

Create or update modules such as:

- `net/sc_lfi_v3.py`
- `net/modules/posterior_context_v3.py`
- `net/modules/posterior_transport_flow_v3.py`
- `net/modules/query_conditioned_transport_v3.py`
- `net/modules/posterior_losses_v3.py`

You may also extend:

- `model_factory.py`
- `main.py`

All modules must include:

- formulas in docstrings
- tensor shapes
- whether the formula is:
  - paper-grounded
  - our few-shot adaptation
  - or engineering approximation

---

## 12. TESTING REQUIREMENTS

You must add tests for at least:

1. support permutation invariance
2. shot-aware shrinkage behavior shape/sanity
3. posterior base measure normalization
4. query-conditioned reweighting normalization and gradient flow
5. flow solver sanity
6. weighted transport distance correctness with nonuniform masses
7. distributional margin behavior
8. degenerate prototype-like reduction
9. gradient flow reachability through posterior builder + scoring path

Add tests under `tests/` with clear names.

---

## 13. REQUIRED DESIGN REVIEW BEFORE CODING

Before implementation, write a design review note that includes:

1. what survives from `v2`
2. what is removed from `v2`
3. exact formulas of `v3`
4. which formulas are paper-grounded
5. which formulas are our novelty
6. how the architecture specifically fixes `1-shot` overfitting
7. why the loss design is more flow-matching-faithful than `v2`
8. what ablations are enabled

Do not start coding before this note is complete.

---

## 14. REQUIRED VERIFICATION AFTER CODING

After implementation:

1. run the unit tests
2. run at least one short smoke training run on the low-shot regime already used for `v2`
3. report whether the early-epoch behavior is less brittle than `v2`
4. produce a final implementation report

The final report must contain:

1. module-by-module changes
2. exact implemented formulas
3. what was removed from `v2`
4. why each new piece exists
5. which `1-shot` failure modes are explicitly addressed
6. which approximations still remain
7. what ablations are now possible
8. what the next likely bottlenecks are

---

## 15. FINAL STANDARD

The success criterion is not:

- "the new model compiles"
- "the new model has more modules"
- "the new model is more complex"

The success criterion is:

> `sc_lfi_v3` is a clearer, more original, more few-shot-faithful model than `v2`, with a sharper theory story and a class inference mechanism that is explicitly designed to generalize in `1-shot`.

Implement to that standard.
