You are a research engineer implementing a NEW few-shot learning model.
Your job is NOT to improvise a generic “flow + classifier” system.
Your job is to implement exactly the research design below, with mathematically careful code and clean modular structure.

====================================
0. PROJECT GOAL
====================================

We want to design a few-shot classifier where each class is represented NOT by a single prototype,
but by a support-conditioned latent evidence distribution.

The central hypothesis is:

    In few-shot learning, support-derived point prototypes are often biased and under-expressive.
    A class should instead be represented by a support-conditioned latent distribution / transport process.

Therefore, we will build:

    Support-Conditioned Latent Flow Inference (working name)

Core idea:
- support set of class c -> permutation-invariant class context h_c
- h_c conditions a latent flow model
- the flow defines a class-conditional latent evidence distribution mu_c(. | h_c)
- query tokens are mapped into the same latent evidence space
- query-to-class score is NOT prototype cosine
- score = negative distribution-fit cost between query latent evidence and class-conditional latent distribution

This is a FEW-SHOT INFERENCE paper, not a generative image synthesis paper.

DO NOT:
- add pixel-space diffusion/generation
- do data augmentation as the main idea
- turn this into a reconstruction paper
- turn this into a generic conditional normalizing flow without episodic few-shot semantics

====================================
1. PAPERS YOU MUST READ CAREFULLY
====================================

You must inspect these papers and repos before coding:

[A] Flow Matching for Generative Modeling
- Read the paper carefully.
- Extract the exact training principle: regressing vector fields of fixed conditional probability paths.
- Understand conditional path parameterization and the role of OT/displacement interpolation paths.
- Repo to inspect:
    facebookresearch/flow_matching

[B] Flow Matching in Latent Space
- Read the paper carefully.
- Extract what is specific to latent-space flow training, not pixel-space.
- Understand how latent variables are used as the state space for the velocity field.
- Repo to inspect:
    VinAIResearch/LFM

[C] Generative Conditional Distributions by Neural (Entropic) Optimal Transport (GENTLE)
- Read the paper carefully.
- Extract the exact role of conditional distribution learning and why low-sample conditional learning matters.
- Extract what is conceptually transferable:
    * modeling a FAMILY of conditional distributions
    * regularization for limited-sample conditional learning
    * entropic OT as a conditional distribution discrepancy
- Repo to inspect:
    nguyenngocbaocmt02/GENTLE

[D] A recent few-shot prototype-bias paper
- Read at least one prototype-bias / prototype-optimization paper
- Purpose: understand the problem we are replacing, not copying.
- We need to clearly separate our method from query-guided prototype correction.

IMPORTANT:
- Do not copy code architecture from those papers blindly.
- Use them to understand exact mathematical objects and stable implementation practices.
- Our final model is NEW and FEW-SHOT SPECIFIC.

====================================
2. RESEARCH POSITIONING
====================================

We are NOT proposing:
- better local matching only
- better prototype correction only
- another query-guided prototype refinement
- diffusion-based support augmentation

We ARE proposing:
- few-shot classification as support-conditioned latent distribution inference
- prototype-based inference as a degenerate special case
- support set acts as conditioning variable for a class-conditional latent flow/distribution
- query is classified by fit to the conditional latent distribution of each class

Target research claim:
    Existing few-shot methods often rely on support-derived point summaries or heuristic local matching.
    We instead model each class as a support-conditioned latent evidence distribution,
    and classify queries via conditional distribution fit.

====================================
3. EPISODIC SETUP AND NOTATION
====================================

Episode:
- N-way K-shot few-shot classification
- support set of class c:
      S_c = {x_{c,1}, ..., x_{c,K}}
- query image:
      q

Backbone tokenization:
- For any image x:
      Z(x) = [z_1, ..., z_M],    z_m in R^d
- support class pooled tokens:
      Z_c = concat_k Z(x_{c,k}) in R^{(K*M) x d}

We will build 5 modules:
1. backbone/tokenizer
2. support-set context encoder
3. latent evidence projector
4. conditional latent flow model
5. query-to-class distribution scoring module

====================================
4. MODEL DESIGN
====================================

4.1 Backbone / Tokenizer

Implement a standard backbone that outputs a spatial feature map and flatten it into tokens.

Minimal practical starting point:
- Conv64F or ResNet12 backbone already used in the few-shot pipeline
- output tokens:
      Z(x) = [z_1, ..., z_M] in R^{M x d}

Constraints:
- do not rewrite the training pipeline unnecessarily
- keep compatibility with episodic support/query batching
- preserve support/query tensor semantics cleanly

4.2 Support-Conditioned Class Context Encoder

For each class c, aggregate all support tokens into a permutation-invariant class context:

      h_c = Phi_set(Z_c) in R^{d_h}

Use a permutation-invariant encoder, such as:
- DeepSets-style weighted pooling
or
- lightweight Set Transformer if already stable

Preferred first implementation:
- token MLP phi(.)
- scalar attention score a_i from phi(z_i)
- normalized weighted sum:
      h_c = sum_i alpha_i phi(z_i)

Requirements:
- strictly invariant to support ordering
- class-level, not shot-routed inference
- no dependence on arbitrary support order

4.3 Latent Evidence Projector

Project image tokens into a latent evidence space:

      e_m = Psi(z_m) in R^{d_l}

This is the space where conditional distributions / flows live.

Requirements:
- simple 2-layer MLP with normalization
- output dimension d_l should be moderate and stable
- all downstream flow and scoring happens in this latent evidence space

Important:
- this latent evidence space is NOT for image generation
- it is a compact space for class-conditional evidence modeling

4.4 Conditional Latent Flow

For each class c, define a class-conditioned velocity field:

      v_theta(y, t; h_c),    y in R^{d_l}, t in [0,1]

This induces the ODE:

      d y_t / d t = v_theta(y_t, t; h_c),    y_0 ~ p0

Base distribution:
      p0 = N(0, I)

The time-1 pushforward distribution defines the class-conditional latent evidence distribution:

      mu_c(. | h_c)

Implementation requirement:
- small residual MLP / FiLM-conditioned MLP is enough
- input = concat(y_t, time_embedding(t), h_c)
- output = velocity in R^{d_l}

DO NOT:
- use an unnecessarily huge transformer here
- make the flow branch dominate the whole model
- confuse this with score matching or DDPM noise prediction

====================================
5. FLOW-MATCHING OBJECTIVE
====================================

You must carefully implement a FEW-SHOT ADAPTATION of flow matching.

Reference principle from Flow Matching:
- learn the vector field of a fixed conditional probability path

Our adaptation:

For each support latent token e = Psi(z), define noise epsilon ~ N(0, I).
Use a simple linear conditional path:

      y_t = (1 - t) * epsilon + t * e

Target velocity:

      u_t(epsilon, e) = e - epsilon

Then train the class-conditioned velocity field:

      L_FM =
      E_{c}
      E_{z in Z_c}
      E_{epsilon ~ N(0,I), t ~ Uniform(0,1)}
      || v_theta((1-t)epsilon + t Psi(z), t; h_c) - (Psi(z) - epsilon) ||_2^2

Important:
- this specific formula is our FEW-SHOT ADAPTATION
- do not falsely claim it is copied directly from any one paper
- but do implement it in the spirit of Flow Matching correctly

Code requirements:
- write a dedicated function for path sampling
- write a dedicated function for target velocity
- keep tensor shapes explicit and documented
- make training numerically stable

====================================
6. SUPPORT-CONDITIONED DISTRIBUTION REGULARIZATION
====================================

We want to transfer the KEY IDEA from GENTLE:
conditional distributions under low sample size need smooth / regularized learning.

We adapt this idea to episodic few-shot learning.

6.1 Support anchoring regularizer

The generated class-conditional distribution should stay anchored to observed support latent evidence.

Let support empirical latent distribution be:

      nu_c^sup = (1 / |Z_c|) sum_i delta_{Psi(z_i)}

Sample L particles from the class-conditioned flow to obtain empirical generated distribution:

      muhat_c = (1 / L) sum_j delta_{y_j^(c)}

Define an anchoring loss:

      L_align = D(muhat_c, nu_c^sup)

where D is initially:
- sliced Wasserstein preferred for implementation simplicity and stability
or
- entropic OT if already available and stable

Purpose:
- keep generated class distribution attached to actual support evidence
- prevent free-floating generative behavior

6.2 Optional context smoothness regularizer

If class contexts h_c and h_c' are close, the corresponding distributions should not vary wildly.

A simple first version:
- compute pairwise context similarity inside batch/episode
- for nearest context pairs, penalize:
      L_smooth = sum_{(c,c')} w_cc' * D(muhat_c, muhat_c')

This is OPTIONAL in version 1.
Keep it modular, but do not let it complicate the first working version too much.

====================================
7. QUERY REPRESENTATION AND CLASS SCORING
====================================

For query q:
- tokenize -> Z(q)
- project -> U_q = [u_1, ..., u_Mq], where u_m = Psi(z_m^q)

Represent the query as empirical latent evidence distribution:

      nu_q = (1 / Mq) sum_m delta_{u_m}

For each class c:
- sample L particles from mu_c(. | h_c)
- form empirical class distribution muhat_c

Then define class score:

      s_c(q) = - D(nu_q, muhat_c)

where D is initially:
- Sliced Wasserstein distance preferred first
- optionally entropic OT later

Classification logits:
      logits_c = temperature_scale * s_c(q)

CE loss:
      L_cls = CrossEntropy(logits, true_label)

IMPORTANT:
- this scoring is the core inference object
- score is distribution fit, NOT cosine to prototype
- if you add a global prototype branch for stabilization, keep it auxiliary and clearly separated

====================================
8. OPTIONAL STABILIZATION BRANCH
====================================

For the first implementation, it is acceptable to add a simple global prototype score:

      p_c = mean_k g(x_{c,k})
      s_c^glob = cosine(g(q), p_c)

Fuse with distribution score:

      s_c = alpha * s_c^glob + (1 - alpha) * s_c^dist

But:
- alpha should be fixed and conservative at first
- the paper contribution must remain the distribution branch
- do NOT let the prototype branch become the real classifier while the flow branch is decorative

====================================
9. FINAL LOSS
====================================

Version 1 total loss:

      L_total =
          L_cls
        + lambda_fm * L_FM
        + lambda_align * L_align
        + lambda_smooth * L_smooth   (optional, can be 0 initially)

Recommended initial strategy:
- first get a stable model with:
      L_cls + lambda_fm * L_FM + lambda_align * L_align
- only then turn on optional smoothness

====================================
10. WHAT MUST BE TRUE IN THE IMPLEMENTATION
====================================

10.1 Theoretical invariants
- support order invariance must hold
- class representation must be support-conditioned
- prototype matching must be recoverable as a degenerate special case conceptually
- query-class score must be a distribution-fit score

10.2 Engineering invariants
- no hidden tensor-shape ambiguity
- every main tensor shape documented in comments
- no silent broadcasting in important formulas
- no magic constants hardcoded without config exposure
- every loss implemented in a separate clean function
- every module independently testable

10.3 Scientific invariants
- no claim in comments/docstrings that overstates the paper references
- clearly separate:
    * paper-derived principle
    * our adaptation
    * optional engineering stabilizers

====================================
11. MINIMAL FILE / MODULE PLAN
====================================

Please implement with a clean file structure such as:

- net/sc_lfi.py
    main model

- net/modules/set_context.py
    permutation-invariant support context encoder

- net/modules/latent_projector.py
    latent evidence projector

- net/modules/conditional_flow.py
    class-conditioned velocity field + path helpers

- net/modules/distribution_distance.py
    sliced wasserstein / entropic OT wrappers

- net/modules/flow_losses.py
    FM loss and support anchoring loss

- tests/test_sc_lfi_shapes.py
    tensor shape and invariance tests

- tests/test_sc_lfi_losses.py
    sanity tests for losses

====================================
12. REQUIRED TESTS
====================================

Before declaring the implementation “done”, you must create sanity tests:

[A] Shape test
- verify support/query episodic shapes end-to-end

[B] Permutation invariance test
- shuffle support order, class score should remain unchanged up to tolerance

[C] FM loss sanity
- random batch should produce finite positive loss
- gradients should flow to context encoder, projector, and flow net

[D] Distribution score sanity
- score for query against its own class support should tend to improve after a few optimization steps on a toy task

[E] Degenerate behavior test
- if flow branch is replaced by a single learned mean particle per class, scoring should reduce conceptually toward prototype-like behavior

====================================
13. ABLATION HOOKS YOU MUST EXPOSE
====================================

Make config flags for:

- use_global_proto_branch
- use_flow_branch
- use_align_loss
- use_smooth_loss
- distance_type = {sw, entropic_ot}
- class_context_type = {deepsets, lightweight_set_transformer}
- flow_conditioning_type = {concat, film}
- num_flow_particles
- latent_dim
- fm_time_schedule
- score_temperature

These hooks are needed for paper ablations later.

====================================
14. WHAT TO WRITE IN CODE COMMENTS
====================================

In code comments and docstrings:
- explain WHY each module exists
- explain WHICH paper inspired the principle
- explain WHETHER the formula is directly standard or our few-shot adaptation

Example style:

- “This follows the flow-matching principle of regressing vector fields on fixed conditional probability paths.”
- “This support-conditioned context is our few-shot adaptation of conditional covariates.”
- “This anchoring loss is inspired by conditional distribution matching/regularization ideas, but adapted here to few-shot support-conditioned class inference.”

Do NOT write false comments like:
- “This is exactly Eq. X from paper Y”
unless it actually is.

====================================
15. DELIVERABLES
====================================

Deliver all of the following:

1. A concise architecture summary
2. Exact tensor shapes for every main stage
3. The implemented formulas in math notation and code mapping
4. A list of which paper/repo informed each module
5. A note of which pieces are:
   - core novelty
   - borrowed principle
   - engineering stabilizer
6. Initial training defaults that are conservative and stable
7. A short “known risks / failure modes” note

====================================
16. NON-NEGOTIABLE IMPLEMENTATION PHILOSOPHY
====================================

- Code the math carefully.
- Keep the model minimal but principled.
- Do not add fashionable modules unless they directly serve the theory.
- Prefer clean, explicit, testable formulas over clever shortcuts.
- If a formula is uncertain, inspect the referenced paper/repo first.
- If a part is OUR design adaptation, state it clearly and implement it transparently.

Your mission is to produce a correct, theory-faithful, few-shot-specific implementation draft suitable for a research codebase, not a flashy prototype.
Important:
- Reuse the current few-shot training pipeline when possible.
- Do not break episodic batching.
- Do not silently modify existing evaluation semantics.
- First produce a design note mapping formulas -> modules -> tensor shapes.
- Then implement the minimal stable version only.
- Do not over-engineer version 1.