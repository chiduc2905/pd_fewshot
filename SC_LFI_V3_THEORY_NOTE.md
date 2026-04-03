# SC-LFI-v3 Theory Note

Working title:

- **SC-LFI-v3**
- Expanded name: **Support-Conditioned Latent Flow Inference via Posterior Evidence Transport**

This note defines the next redesign target after the survey in
`FEWSHOT_DISTRIBUTIONAL_SURVEY.md`.

The purpose is to preserve the original SC-LFI scientific core while making the method:

- genuinely few-shot-specific;
- theoretically sharper;
- and novelty-strong enough to stand on its own rather than looking like "flow matching + OT metric + some few-shot tricks".

## 1. Core Claim That Must Be Preserved

The original SC-LFI family has one strong core claim worth preserving:

> Each class in an episode should be represented as a support-conditioned latent evidence distribution, and query classification should be a distribution-fit problem rather than prototype cosine matching.

This claim is stronger than ProtoNet-style point estimation.
It is also different from purely discriminative local matching, because it says the classifier should infer a class-level evidence distribution in latent space.

That claim remains the center of `v3`.

## 2. What Must Change

The weak version of that claim is:

> compress support into a class summary, generate a class cloud with a conditional flow, then compare query to that cloud.

That is not enough for strong few-shot learning.
It is too generic, and it is not obviously better than DeepEMD/FRN/MetaQDA in `1-shot`.

The stronger version is:

> infer a posterior distribution over class evidence from a tiny support set, using a shot-aware prior and a support-basis preserving transport model, then score query evidence by transport fit to that posterior evidence distribution.

That is the actual redesign direction.

## 3. Main Theoretical Shift

The main shift is this:

- `v1/v2` implicitly treated the class distribution mostly as a generated object;
- `v3` will treat the class distribution explicitly as a **posterior object**.

This is the theory upgrade that keeps the original idea but makes it sharper.

### 3.1 Latent evidence posterior

Let `S_c` be the support set for class `c`.
Let `E(S_c) = {e_i}` be the latent evidence atoms extracted from support images.

We define the class object not as a point estimate, but as a support-conditioned posterior latent measure:

`P_c = P( G_c | S_c, T )`

where:

- `G_c` is the latent class evidence distribution;
- `S_c` is the class support set;
- `T` is optional task-level context from the full episode.

At inference time we do not need the full posterior family explicitly.
We need a tractable approximation of its posterior predictive evidence measure:

`muhat_c approx PostPred( G_c | S_c, T )`

The query is then classified by:

`score_c(q) = - D( nu_q, muhat_c )`

where `nu_q` is the weighted query empirical evidence measure.

This preserves the original SC-LFI claim exactly.

## 4. The New Novelty: Posterior Evidence Transport

The novelty of `v3` is not "we use flow matching".
That would be too weak.

The novelty should be:

> We formulate few-shot classification as posterior evidence transport: a support-conditioned posterior predictive evidence measure is constructed by applying a residual transport operator to a shot-aware support-prior base measure, and queries are classified by weighted transport fit to this posterior predictive measure.

This gives a much sharper contribution statement.

## 5. Three Core Contributions

### Contribution 1: Shot-Aware Support-Prior Posterior Base Measure

The base measure for class `c` should not be pure support empirical measure and should not be pure noise.
It should be a shot-aware shrinkage measure:

`mu_c^0 = alpha_c * nusup_c + (1 - alpha_c) * pi_c`

where:

- `nusup_c = sum_i a_i^sup delta_{e_i}` is the weighted empirical support evidence measure;
- `pi_c` is a learned support-conditioned prior measure;
- `alpha_c in [0,1]` is shot-aware and uncertainty-aware.

Interpretation:

- if support is trustworthy and plentiful, `alpha_c` increases;
- in `1-shot`, `alpha_c` should usually be smaller, because the empirical support measure is a poor estimator of the class distribution.

This is the right fix for few-shot semantics.

It also gives a much cleaner theory than forcing a fixed support anchor weight.

### Contribution 2: Support-Basis Residual Flow Pushforward

The flow should not generate class particles from generic Gaussian noise alone.
Instead, define a residual transport map around the base measure:

`muhat_c = (T_theta,c)_# mu_c^0`

where:

- `T_theta,c` is a support-conditioned transport operator;
- `(_#)` denotes pushforward of measures.

In practice `T_theta,c` can be implemented by flow matching over latent evidence atoms, but the theory should be stated at the measure level first.

This is stronger than "conditional flow generator" because:

- the support basis remains present in the object being transported;
- the flow is now a **posterior refinement operator**, not an unconditional generator;
- this makes the model genuinely few-shot-specific.

### Contribution 3: Query-Conditioned Reliability Transport Scoring

The class posterior measure remains support-conditioned.
But the scoring operator should be query-conditioned through reliability reweighting:

`D_qc( nu_q, muhat_c ; q, c )`

where support-side masses or transport costs are modulated by query-class compatibility.

Important:

- this does **not** mean the class distribution itself becomes query-generated;
- it means the inference operator recognizes that not all support atoms are equally relevant for a given query.

This is the right way to absorb the lessons of DeepEMD/CAN/FRN without destroying the original SC-LFI identity.

## 6. Why This Is Real Novelty, Not Just a Hybrid

The novelty is sharp because the method is not equivalent to existing families.

### 6.1 Not FEAT

FEAT adapts prototypes or class embeddings by task-level set interaction.
It does not formulate a posterior predictive class evidence distribution with transport scoring.

### 6.2 Not DeepEMD

DeepEMD performs dense support-query matching with EMD and cross-reference weights.
It does not explicitly infer a support-conditioned posterior class evidence measure and then classify by distribution fit to that posterior predictive measure.

### 6.3 Not FRN

FRN uses reconstruction from support descriptors as a class basis.
It does not define a flow-based posterior refinement over class evidence measures.

### 6.4 Not MetaQDA

MetaQDA is probabilistic and elegant, but its uncertainty is over classifier parameters.
Our posterior object is instead a **latent evidence distribution**.

### 6.5 Not distribution calibration

Distribution Calibration transfers statistics to calibrate feature distributions for novel classes.
`v3` instead learns a support-conditioned posterior transport operator over latent evidence measures inside an episodic classifier.

So the novelty is not borrowed wholesale from any one model.
It is a new synthesis centered on a sharper posterior-evidence formulation.

## 7. Theoretical Object of v3

We now define the intended mathematical object more precisely.

### 7.1 Support empirical evidence measure

For support class `c`, let support images yield latent evidence atoms:

`E_c = {e_{c,i}}_{i=1}^{N_c}`, `e_{c,i} in R^{d_l}`

with learned masses:

`a_{c,i} >= 0`, `sum_i a_{c,i} = 1`

Then:

`nusup_c = sum_i a_{c,i} delta_{e_{c,i}}`

### 7.2 Support-conditioned prior measure

We learn a compact prior measure:

`pi_c = sum_r b_{c,r} delta_{p_{c,r}}`

where:

- `p_{c,r}` are prior atoms predicted from support/task context;
- `b_{c,r}` are prior masses.

This is not a base-class lookup table.
It is a meta-learned prior generator conditioned on the support set of the current novel class.

### 7.3 Shot-aware shrinkage coefficient

Let:

- `K` be shot number;
- `u_c` be a support uncertainty statistic, for example support dispersion or entropy of masses.

Then:

`alpha_c = sigma( g_phi( K, u_c, h_c ) )`

with the intended behavior:

- smaller `K` and higher uncertainty imply smaller `alpha_c`;
- larger `K` and lower uncertainty imply larger `alpha_c`.

This turns the model into a few-shot classifier in a mathematically explicit way.

### 7.4 Posterior base measure

`mu_c^0 = alpha_c * nusup_c + (1 - alpha_c) * pi_c`

This is the class posterior base measure.

### 7.5 Residual transport refinement

Define a support-conditioned transport field:

`v_theta(y, t ; h_c, M_c, T)`

where:

- `h_c` is a global class summary;
- `M_c` is a compact support memory set;
- `T` is optional episode-level context.

The refined class predictive measure is:

`muhat_c = (T_theta,c)_# mu_c^0`

This can be approximated by transporting atoms sampled from `mu_c^0` with a fixed-step solver.

### 7.6 Query evidence measure

For query `q`, obtain latent evidence atoms:

`U_q = {u_j}_{j=1}^{M_q}`

with masses:

`rho_j >= 0`, `sum_j rho_j = 1`

Then:

`nu_q = sum_j rho_j delta_{u_j}`

### 7.7 Query-conditioned scoring transport

Instead of static uniform or purely image-local masses, define query-conditioned support relevance:

`omega_{c,l}(q) = Softmax_l( psi( u_q^pool, x_{c,l}, h_c ) )`

where `x_{c,l}` are atoms of `muhat_c` or support memory atoms.

Then the class-side measure used in scoring becomes:

`muhat_c^q = Reweight( muhat_c ; omega(q,c) )`

and the score is:

`score_c(q) = - D( nu_q, muhat_c^q )`

This keeps the class object support-conditioned, while making the inference operator query-aware.

## 8. Why This Fixes the 1-Shot Failure Mode

The current failure mode is essentially:

- the model trusts one support image too much;
- it then either memorizes that image or builds a brittle class anchor around it.

The new theory attacks that directly.

### 8.1 In `1-shot`, the support empirical distribution is weak

So `alpha_c` should not be high by default.
That gives the model room to use the learned prior measure.

### 8.2 One support image still contains many local evidence atoms

So even in `1-shot`, the class is not reduced to one point.
The support basis remains rich.

### 8.3 Query-conditioned relevance suppresses irrelevant support atoms

This is the correct way to keep local descriptor richness without letting background patches dominate.

### 8.4 Residual flow enriches class evidence locally

The flow is useful again because it models **local class variability around the support basis**, not an unconstrained class generator from a summary vector.

## 9. Special-Case Reductions

This is an important theoretical advantage.
`v3` should reduce to several known families under explicit constraints.

### 9.1 ProtoNet-like degenerate case

If:

- `N_c = 1` atom per class,
- `T_theta,c = Identity`,
- `D` reduces to squared Euclidean distance,

then `v3` reduces to prototype-like scoring.

### 9.2 DeepEMD / DN4-like local matching case

If:

- `mu_c^0 = nusup_c`,
- `T_theta,c = Identity`,
- `D` is weighted OT over support/query local atoms,

then `v3` becomes a local descriptor transport matcher.

### 9.3 FRN-like basis case

If:

- class measure is interpreted as a support basis,
- the score uses reconstruction error instead of OT,

then the same support-basis semantics recover FRN-style reasoning.

### 9.4 Probabilistic classifier case

If:

- `mu_c^0` is summarized by Gaussian sufficient statistics,
- `T_theta,c` is affine or identity,
- `D` corresponds to a Gaussian negative log-likelihood or quadratic discriminant score,

then the method approaches a MetaQDA-like probabilistic regime.

This gives `v3` a strong unifying theory story.

## 10. Why Flow Matching Is Still Worth Keeping

Flow matching should remain, but in a more disciplined role.

The right statement is:

> Flow matching is the numerical parameterization of the posterior evidence transport operator, not the main conceptual contribution.

That matters because:

- it keeps the paper/model grounded in modern generative transport tools;
- but the novelty remains few-shot-specific and classifier-centered.

This also prevents the design from looking like a generic conditional CNF applied to few-shot learning.

## 11. Required Architectural Consequences

If we follow the theory above, then the implementation must change accordingly.

### 11.1 Support basis must remain explicit

The model must keep:

- support evidence atoms,
- support masses,
- support memory slots,
- and uncertainty statistics.

No early collapse to a single summary vector.

### 11.2 Base measure must be constructed before flow

The pipeline should be:

`support atoms -> posterior base measure -> residual transport -> class predictive measure`

not:

`support summary -> noise-conditioned generator -> class particles`

### 11.3 Query-conditioned reweighting must happen inside scoring

This is where DeepEMD/CAN-style few-shot specificity should enter.

### 11.4 Shot-aware shrinkage must be explicit

The coefficient `alpha_c` must be an actual architectural object, not an accidental side effect of loss weights.

## 12. Expected Contribution Statement in a Paper

If written cleanly, the contributions of `v3` would be:

1. **Posterior Evidence Transport for Few-Shot Classification**:
   we formulate few-shot class inference as posterior predictive transport in latent evidence space, rather than point prototype estimation or unconditional class generation.

2. **Shot-Aware Support-Prior Shrinkage Measure**:
   we introduce a support-conditioned posterior base measure that interpolates between empirical support evidence and a learned prior, with explicitly shot-aware uncertainty control.

3. **Support-Basis Residual Flow**:
   we parameterize posterior predictive refinement as a residual transport operator over support-basis evidence atoms, preserving few-shot semantics while retaining expressive latent distribution modeling.

4. **Query-Conditioned Reliability Transport Scoring**:
   we classify queries by weighted transport fit to the posterior predictive class measure using query-conditioned evidence reliability, unifying prototype, local matching, and probabilistic few-shot views as special cases.

This is substantially sharper than `v2`.

## 13. What Would Make the Novelty Weak Again

The novelty becomes weak again if we do any of the following:

- make the class distribution mostly a generated particle cloud from Gaussian noise;
- make the support basis disappear before scoring;
- turn the score back into prototype cosine plus auxiliary OT regularization;
- use query-conditioned modules that completely reconstruct a fresh class object for each query, because that blurs the central class-posterior claim;
- treat `1-shot` and `5-shot` with the same mixing logic.

Those should be avoided.

## 14. Final Position

The right redesign is not:

- "improve the flow"
- "improve the OT metric"
- "add more regularization"

The right redesign is:

- **redefine SC-LFI as posterior evidence transport for few-shot learning**.

That keeps the original theory.
It strengthens the few-shot semantics.
And it gives a cleaner, more original, more publishable novelty statement.
