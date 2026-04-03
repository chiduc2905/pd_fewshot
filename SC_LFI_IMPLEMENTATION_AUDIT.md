# SC-LFI Implementation Audit And Theory Alignment

This document is a deep implementation-level audit of the `sc_lfi` model added to `pulse_fewshot`.

Goal:

- explain exactly how the code works;
- map the intended theory in `latent.md` to the actual implementation;
- make it easy for another LLM or researcher to judge whether the current code is theoretically faithful;
- identify where the implementation is exact, where it is an engineering approximation, and where it may be limiting performance.

This is not a marketing note. It is an engineering/research audit.

## 1. Executive Verdict

Short answer:

- Yes, the current `sc_lfi` implementation is structurally aligned with the core theory in `latent.md`.
- The code really does implement support-conditioned class distributions, not plain prototype matching.
- The query/class score is genuinely a distribution-fit score, not cosine-to-prototype disguised as a new model.
- The few-shot flow-matching loss is implemented in the intended spirit and with the exact linear path formula specified in `latent.md`.

However:

- the current distribution scoring uses the repo's existing lightweight sliced Wasserstein estimator, not an exact paper-style SW estimator;
- inference uses a finite number of particles and a simple Euler integrator, so the class distribution is only approximated numerically;
- the flow is conditioned only on the class summary `h_c`, not on the full support token set at sampling time;
- the model is therefore theory-faithful at the structural level, but still conservative and approximate at the numerical level.

Practical conclusion:

- if the model is "promising but not yet high", the likely bottlenecks are not that the implementation collapsed back into ProtoNet semantics;
- the likely bottlenecks are numerical weakness, metric weakness, and limited conditional expressivity.

## 2. Source Of Truth

The intended theory comes from:

- `latent.md`

The actual implementation lives in:

- `net/sc_lfi.py`
- `net/modules/set_context.py`
- `net/modules/latent_projector.py`
- `net/modules/conditional_flow.py`
- `net/modules/distribution_distance.py`
- `net/modules/flow_losses.py`

Integration points:

- `net/model_factory.py`
- `main.py`

Tests:

- `tests/test_sc_lfi_shapes.py`
- `tests/test_sc_lfi_losses.py`

Backbone/token utility dependencies:

- `net/fewshot_common.py`
- `net/metrics/sliced_wasserstein.py`

## 3. The Theory That `latent.md` Asked For

The target design in `latent.md` can be summarized as:

1. Encode each image into spatial tokens.
2. Merge all support tokens of class `c`.
3. Build a permutation-invariant class context `h_c`.
4. Project tokens into a latent evidence space.
5. Define a class-conditioned velocity field `v_theta(y, t; h_c)`.
6. Train the flow using a fixed linear conditional path:

   `y_t = (1 - t) * epsilon + t * e`

   `u_t(epsilon, e) = e - epsilon`

7. Sample particles from the support-conditioned flow to form an empirical class distribution.
8. Represent query tokens as an empirical latent evidence distribution.
9. Score a query against a class by negative distribution discrepancy:

   `s_c(q) = -D(nu_q, muhat_c)`

10. Optimize:

   `L_total = L_cls + lambda_fm * L_FM + lambda_align * L_align + lambda_smooth * L_smooth`

The main scientific claim is:

- a class is modeled as a support-conditioned latent evidence distribution, not as a single support prototype.

## 4. High-Level File Map

### 4.1 Main model

File:

- `net/sc_lfi.py`

Main class:

- `SupportConditionedLatentFlowInferenceNet`

Responsibility:

- own the end-to-end episodic pipeline;
- build query/support tokens;
- build class contexts;
- project to latent evidence;
- sample class particles;
- compute query/class distribution-fit logits;
- compute auxiliary flow/alignment/smoothness losses.

### 4.2 Support context encoder

File:

- `net/modules/set_context.py`

Classes:

- `DeepSetsContextEncoder`
- `LightweightSetTransformerContextEncoder`
- `SupportSetContextEncoder`

Responsibility:

- convert support token sets into a permutation-invariant class context.

### 4.3 Latent evidence projector

File:

- `net/modules/latent_projector.py`

Class:

- `LatentEvidenceProjector`

Responsibility:

- map backbone tokens into latent evidence vectors.

### 4.4 Conditional flow

File:

- `net/modules/conditional_flow.py`

Main components:

- `sample_flow_times`
- `sample_linear_conditional_path`
- `target_linear_path_velocity`
- `SinusoidalTimeEmbedding`
- `_ConcatVelocityField`
- `_FiLMVelocityField`
- `ConditionalLatentFlowModel`

Responsibility:

- implement the time-conditioned class-conditioned velocity field;
- implement the linear flow-matching path and target velocity;
- sample latent particles via Euler integration.

### 4.5 Distribution distance

File:

- `net/modules/distribution_distance.py`

Classes:

- `UniformEntropicOTDistance`
- `DistributionDistance`

Responsibility:

- expose `distance_type in {sw, entropic_ot}`.

### 4.6 Losses

File:

- `net/modules/flow_losses.py`

Functions:

- `compute_flow_matching_loss`
- `compute_support_anchoring_loss`
- `compute_context_smoothness_loss`

Responsibility:

- keep the losses explicit and independently testable.

## 5. End-To-End Data Flow With Exact Tensor Semantics

This section is the most important for theory auditing.

### 5.1 Episode input shapes

The model inherits few-shot episode semantics from `BaseConv64FewShotModel` in `net/fewshot_common.py`.

Expected input shapes:

- `query`: `[B, NQ, C, H, W]`
- `support`: `[B, Way, Shot, C, H, W]`

These are validated by:

- `BaseConv64FewShotModel.validate_episode_inputs`

Code reference:

- `net/fewshot_common.py`, lines 121-137

### 5.2 Backbone to tokens

The model uses the existing repo backbone through `BaseConv64FewShotModel.encode`.

Feature map to token conversion:

- `feature_map_to_tokens` converts `[N, D, Hf, Wf] -> [N, Hf*Wf, D]`

Code reference:

- `net/fewshot_common.py`, lines 14-18

In `SC-LFI`:

- query tokenization happens in `SupportConditionedLatentFlowInferenceNet._encode_episode_tokens`
- support images are flattened over `Way * Shot`, encoded independently, then reshaped back

Code reference:

- `net/sc_lfi.py`, lines 142-159

Returned shapes:

- `query_tokens`: `[NumQuery, TokensPerImage, HiddenDim]`
- `support_tokens`: `[Way, Shot, TokensPerImage, HiddenDim]`

This is theory-faithful.

Why:

- support images are not pooled globally before tokenization;
- local token evidence is preserved;
- support/query episodic semantics remain intact.

### 5.3 Support token merging

Merged support tokens:

- `merge_support_tokens(..., merge_mode="concat")`
- output shape: `[Way, Shot * TokensPerImage, HiddenDim]`

Code reference:

- `net/fewshot_common.py`, lines 44-62
- `net/sc_lfi.py`, lines 166-170

This matters because `latent.md` explicitly asked for:

- class pooled support tokens `Z_c = concat_k Z(x_{c,k})`

This is implemented exactly.

### 5.4 Support-conditioned class context

Context is built from the merged class token set:

- `class_contexts = self.context_encoder(merged_support_tokens)`

Code reference:

- `net/sc_lfi.py`, line 169

Default implementation:

- `DeepSetsContextEncoder`

DeepSets logic:

1. each token is mapped by `token_encoder`;
2. each encoded token gets a scalar attention logit;
3. softmax weights over the set are computed;
4. weighted sum of encoded tokens gives the class context.

Code reference:

- `net/modules/set_context.py`, lines 11-58

Alternative:

- `LightweightSetTransformerContextEncoder`

Code reference:

- `net/modules/set_context.py`, lines 61-107

Theory check:

- This is permutation invariant with respect to support token order.
- It is also support-shot-order invariant because support shots are only concatenated before a set function is applied.
- No query-conditioned support routing is used.

Verdict:

- This matches the intended theory well.

### 5.5 Latent evidence projector

Query and support tokens are projected into latent evidence space through:

- `self.latent_projector`

Code reference:

- `net/sc_lfi.py`, lines 170-171
- `net/modules/latent_projector.py`, lines 9-35

Implementation:

- `LayerNorm -> Linear -> GELU -> Linear -> LayerNorm`

Output:

- `query_latents`: `[NumQuery, TokensPerImage, LatentDim]`
- `support_latents`: `[Way, Shot * TokensPerImage, LatentDim]`

Theory check:

- this is exactly the requested "simple 2-layer MLP with normalization";
- all downstream flow and distribution scoring happen in this latent space.

Verdict:

- This matches `latent.md` directly.

### 5.6 Conditional latent flow

The flow model is:

- `self.flow_model = ConditionalLatentFlowModel(...)`

Code reference:

- `net/sc_lfi.py`, lines 124-130

The velocity field takes:

- latent state `y`
- time embedding `t`
- class context `h_c`

and predicts:

- velocity in `R^{LatentDim}`

Code reference:

- `net/modules/conditional_flow.py`, lines 163-223

Conditioning options:

- `concat`
- `film`

Theory check:

- this is a class-conditioned velocity field;
- conditioning is via `h_c` only, which is consistent with the intended support-conditioned distribution view;
- there is no giant transformer or heavy flow branch.

Verdict:

- faithful to the intended minimal design.

### 5.7 Flow-matching training path

The critical formula from `latent.md` is:

`y_t = (1 - t) epsilon + t e`

`u_t(epsilon, e) = e - epsilon`

In code:

- `sample_linear_conditional_path`
- `target_linear_path_velocity`

Code reference:

- `net/modules/conditional_flow.py`, lines 30-62

Flow-matching inputs are generated in:

- `ConditionalLatentFlowModel.sample_flow_matching_inputs`

Code reference:

- `net/modules/conditional_flow.py`, lines 224-246

This function does:

1. sample `noise = torch.randn_like(evidence)`
2. sample `time_values`
3. construct `path_states = (1 - t) noise + t evidence`
4. construct `target_velocity = evidence - noise`

Theory check:

- This is not just "inspired by" the path in `latent.md`;
- it is exactly the same linear interpolation path and target velocity.

Verdict:

- exact implementation of the intended few-shot FM adaptation.

### 5.8 Flow-matching loss

The loss function is:

- `compute_flow_matching_loss`

Code reference:

- `net/modules/flow_losses.py`, lines 9-44

Logic:

1. flatten class-wise support latents from `[Way, SupportTokens, LatentDim]`
   to `[Way * SupportTokens, LatentDim]`
2. repeat the class context for every support latent token of the same class
3. sample `(noise, t, y_t, target_velocity)`
4. predict `v_theta(y_t, t; h_c)`
5. compute MSE with `target_velocity`

This corresponds to:

`L_FM = E || v_theta((1-t)epsilon + t e, t; h_c) - (e - epsilon) ||^2`

Theory check:

- class-conditioning is preserved correctly by repeating `h_c` across all support tokens of class `c`;
- support tokens are used as latent evidence targets;
- there is no prototype collapse inside the FM loss.

Verdict:

- exact match to the intended adaptation.

### 5.9 Class distribution sampling

Class particles are built in:

- `SupportConditionedLatentFlowInferenceNet._build_class_particles`

Code reference:

- `net/sc_lfi.py`, lines 183-210

If `use_flow_branch=True`:

- sample particles from `N(0, I)`
- Euler integrate through the learned velocity field
- output shape: `[Way, NumFlowParticles, LatentDim]`

Euler sampler implementation:

- `ConditionalLatentFlowModel.sample_particles`

Code reference:

- `net/modules/conditional_flow.py`, lines 248-293

Important details:

- particles are advanced with fixed step size `1 / num_steps`
- time points are midpoint-like constants `(step_idx + 0.5) * step_size`
- each class's particles receive the same class context `h_c`

Theory check:

- conceptually faithful: yes;
- numerically exact to the continuous ODE: no.

Why not exact:

- the code uses Euler integration, not a high-order ODE solver;
- the pushforward distribution is represented by a finite Monte Carlo particle set.

Verdict:

- theory-faithful approximation.

### 5.10 Degenerate prototype-like mode

If `use_flow_branch=False`, class particles are not sampled from the flow.
Instead:

- class mean latent `mean(support_latents, dim=1)` is repeated as all particles.

Code reference:

- `net/sc_lfi.py`, lines 209-210

This is intentionally useful because:

- it provides the "prototype-like degenerate special case" requested by `latent.md`;
- it lets us verify that the architecture can collapse toward prototype behavior conceptually.

Verdict:

- good engineering realization of the requested degeneracy hook.

### 5.11 Query-to-class distribution scoring

Distribution-fit scoring happens in:

- `SupportConditionedLatentFlowInferenceNet._score_query_against_classes`

Code reference:

- `net/sc_lfi.py`, lines 212-232

Logic:

1. each query image is represented by its token latent set:
   - `query_latents[q] = [u_1, ..., u_M]`
2. each class is represented by sampled particles:
   - `class_particles[c] = [y_1, ..., y_L]`
3. query/class pairs are expanded to batched distributions:
   - query side: `[NumQuery, Way, QueryTokens, LatentDim]`
   - class side: `[NumQuery, Way, FlowParticles, LatentDim]`
4. distance is computed per query/class pair
5. logits = negative distance times temperature

Formula implemented:

`logits_{q,c} = - score_temperature * D(nu_q, muhat_c)`

Theory check:

- this is the core requirement from `latent.md`;
- the score is genuinely a distribution-fit score;
- no prototype cosine is required for the core branch.

Verdict:

- exact match at the conceptual level.

### 5.12 Optional global prototype branch

Global branch:

- `_compute_global_proto_scores`

Code reference:

- `net/sc_lfi.py`, lines 234-241

Fusion:

- `logits = (1 - proto_branch_weight) * distribution_scores + proto_branch_weight * global_proto_scores`

Code reference:

- `net/sc_lfi.py`, lines 304-310

Theory check:

- this branch is optional and disabled by default in the conservative setup;
- it is clearly separated from the distribution branch;
- it does not secretly replace the main inference object unless the user explicitly turns it on and weights it too strongly.

Verdict:

- acceptable engineering stabilizer;
- does not violate the paper position if used conservatively.

## 6. Loss Structure

Auxiliary losses are computed in:

- `SupportConditionedLatentFlowInferenceNet._compute_auxiliary_losses`

Code reference:

- `net/sc_lfi.py`, lines 243-285

### 6.1 Flow-matching loss

`fm_loss = compute_flow_matching_loss(...)`

Weighted by:

- `lambda_fm`

### 6.2 Support anchoring loss

`align_loss = compute_support_anchoring_loss(class_particles, support_latents, distance_module)`

Code reference:

- `net/modules/flow_losses.py`, lines 47-59

This implements:

`L_align = D(muhat_c, nu_c^sup)`

Theory check:

- this is conceptually faithful to the intended support anchoring;
- the only caveat is that `D` is the repo's practical distance implementation, not always an exact OT metric.

### 6.3 Smoothness loss

`smooth_loss = compute_context_smoothness_loss(...)`

Code reference:

- `net/modules/flow_losses.py`, lines 62-100

This:

- computes nearest class contexts inside the episode;
- penalizes large distribution gaps for those nearest context pairs;
- weights the penalty by `exp(-context_distance)`.

Theory check:

- this matches the optional idea in `latent.md`;
- it is kept modular and off by default.

### 6.4 Total auxiliary loss

`aux_loss = lambda_fm * fm_loss + lambda_align * align_loss + lambda_smooth * smooth_loss`

Code reference:

- `net/sc_lfi.py`, lines 279-285

At training time, the main training script then adds:

- `CrossEntropy(logits, targets) + aux_loss`

This follows the standard repo logic for models returning `{logits, aux_loss}`.

## 7. Distribution Distance: Exactness Versus Convenience

This is one of the most important sections if performance is lower than hoped.

### 7.1 What the code currently uses by default

Default:

- `distance_type = "sw"`

This routes to:

- `DistributionDistance`
- then to `SlicedWassersteinDistance`

Code reference:

- `net/modules/distribution_distance.py`, lines 101-144
- `net/metrics/sliced_wasserstein.py`, lines 42-144

### 7.2 Important caveat

The current `SlicedWassersteinDistance` in this repo is explicitly described as:

- lightweight;
- legacy;
- not the same as an exact paper-style estimator.

Code reference:

- `net/metrics/sliced_wasserstein.py`, lines 1-13

Main approximation:

- if query and class distributions have different numbers of particles/tokens, sorted projected values are interpolated to a shared token count.

Code reference:

- `net/metrics/sliced_wasserstein.py`, lines 80-96 and 127-138

This is stable and convenient, but it is not the strongest possible choice if the research claim is centered on distribution fit.

Verdict:

- structurally okay;
- mathematically weaker than ideal;
- a likely source of suboptimal performance.

### 7.3 Entropic OT option

The code also includes:

- `UniformEntropicOTDistance`

Code reference:

- `net/modules/distribution_distance.py`, lines 22-98

This is a pure-PyTorch Sinkhorn implementation with uniform marginals.

Pros:

- closer in spirit to conditional OT regularization ideas;
- useful as an ablation hook.

Cons:

- fixed uniform masses only;
- not as optimized or numerically refined as specialized OT libraries;
- still not guaranteed to outperform a stronger SW implementation.

## 8. Why The Implementation Is Support-Order Invariant

This is a direct theoretical invariant required by `latent.md`.

### 8.1 Mechanism

Support order invariance comes from:

1. support images are encoded independently;
2. support tokens are concatenated without shot-specific routing;
3. class context is built through a permutation-invariant set encoder;
4. support latents are used as an unordered empirical set for anchoring;
5. class scoring compares the query distribution against sampled class particles only.

No operation in the SC-LFI branch depends on:

- support image position;
- shot index identity;
- any recurrent support ordering.

### 8.2 What the test verifies

Test:

- `tests/test_sc_lfi_shapes.py`, lines 78-99

What it does:

- swaps support shot order;
- checks that logits, distribution scores, class contexts, and generated particles stay unchanged within tolerance.

Additional engineering detail:

- evaluation particle sampling uses a fixed seed for determinism.

Code reference:

- `net/sc_lfi.py`, lines 190-207

Without this fixed evaluation seed, shot-order invariance could still hold in theory but appear noisy in tests due to Monte Carlo variation.

Verdict:

- support-order invariance is genuinely enforced by the implementation.

## 9. Exact Theory Compliance Checklist

This section answers the question:

"Did the code actually implement the architecture promised in `latent.md`?"

### 9.1 Backbone/tokenizer

Requirement:

- backbone outputs tokens.

Status:

- satisfied.

### 9.2 Permutation-invariant support context

Requirement:

- support set of class `c` becomes permutation-invariant context `h_c`.

Status:

- satisfied.

### 9.3 Latent evidence projector

Requirement:

- simple 2-layer MLP with normalization.

Status:

- satisfied.

### 9.4 Class-conditioned velocity field

Requirement:

- `v_theta(y, t; h_c)`.

Status:

- satisfied.

### 9.5 FM path and target velocity

Requirement:

- `y_t = (1-t) epsilon + t e`
- `u_t = e - epsilon`

Status:

- satisfied exactly.

### 9.6 Query scoring by distribution fit

Requirement:

- `s_c(q) = -D(nu_q, muhat_c)`

Status:

- satisfied.

### 9.7 Optional global prototype branch only as stabilizer

Requirement:

- optional and conservative.

Status:

- satisfied.

### 9.8 Separate loss functions and explicit modules

Requirement:

- separate clean functions/modules.

Status:

- satisfied.

### 9.9 Support smoothing optional

Requirement:

- optional.

Status:

- satisfied.

### 9.10 No accidental reversion to standard prototype model

Requirement:

- architecture must not secretly be a prototype classifier.

Status:

- satisfied.

Reason:

- main logits are distribution distances unless the user explicitly turns on or overweights the prototype branch.

## 10. Where The Current Code Is Only An Approximation

These are the most important caveats for a serious research review.

### 10.1 Euler integration

Current implementation:

- fixed-step Euler solver.

Code reference:

- `net/modules/conditional_flow.py`, lines 287-291

Meaning:

- class particle generation is an ODE approximation, not an exact continuous-time solution.

Impact:

- coarse trajectories;
- particle drift error;
- reduced fidelity if the learned vector field is sharp or highly nonlinear.

### 10.2 Finite particle approximation

Current implementation:

- the class distribution is represented by `num_flow_particles` samples.

Impact:

- Monte Carlo noise;
- unstable class scoring if `num_flow_particles` is too small;
- difficulty approximating multimodal class evidence distributions.

### 10.3 Lightweight SW distance

Current implementation:

- legacy SW with interpolation-based token count alignment.

Impact:

- weaker geometric fidelity;
- possibly smoother but less discriminative transport signal.

### 10.4 Context-only flow conditioning

Current implementation:

- the flow is conditioned on `h_c` only.

Meaning:

- once the class context is built, the flow does not see the full support token set directly at particle generation time.

Impact:

- if `h_c` compresses away too much structure, the flow branch can become under-expressive;
- multimodal support evidence may be partially lost before transport.

### 10.5 Uniform token masses

Current implementation:

- query/support latent evidence distributions are treated as uniform empirical sets.

Meaning:

- no learned token masses;
- no reliability weighting across support evidence tokens.

Impact:

- noisy or irrelevant local tokens may dilute the distribution-fit score.

## 11. Why Results May Be "Promising But Not High Yet"

This section is specifically for diagnosing your observation.

### 11.1 The model is conceptually right, but numerically weak

The biggest likely reason is:

- the architecture matches the theory,
- but the numerical machinery is still a first stable version.

The current implementation deliberately prioritized:

- minimality;
- explicitness;
- testability;
- theory correctness at the structural level.

It did not yet maximize:

- transport metric sharpness;
- flow solver quality;
- conditional expressivity;
- token reliability modeling.

### 11.2 The default SW implementation is probably the weakest major component

The class score is the core of the method:

- if `D` is weak, the whole classifier is weak.

Right now:

- `D` is stable and easy to use,
- but not the strongest version of sliced Wasserstein available in the repo ecosystem.

If performance is lower than expected, this is one of the first places to investigate.

### 11.3 Flow branch may be under-conditioned

The class distribution is generated from `h_c`, which is compact.

This is elegant, but risky:

- a single context vector may not be rich enough to parameterize class variability, especially in harder episodes;
- the flow may learn a smoothed average distribution rather than sharp support-conditioned evidence geometry.

### 11.4 Anchor loss may be too soft

Anchoring currently says:

- sampled class particles should stay near the support latent token set.

But:

- if the distance metric is weak,
- or if the support latent space is not well structured,
- the alignment signal may be insufficient to keep generated particles informative.

### 11.5 Query supervision reaches the flow only indirectly

The FM loss trains the flow on support latents.
The CE loss trains the classifier on query labels.

The flow therefore receives class discrimination pressure partly through:

- shared latent projector;
- shared class distribution scoring;
- auxiliary losses.

This is useful, but still a somewhat indirect route compared with methods whose local matching is explicitly query-adaptive.

## 12. What The Current Tests Actually Prove

The tests do not prove the model is optimal.
They prove the implementation is sane.

### 12.1 Shape and payload sanity

Test:

- `tests/test_sc_lfi_shapes.py`, lines 49-75

Proves:

- end-to-end episode forward works;
- auxiliary payload keys exist;
- tensor shapes are consistent.

### 12.2 Support permutation invariance

Test:

- `tests/test_sc_lfi_shapes.py`, lines 78-99

Proves:

- the model does not depend on support shot order.

### 12.3 Degenerate prototype-like behavior

Test:

- `tests/test_sc_lfi_shapes.py`, lines 102-130

Proves:

- when flow is disabled, class particles collapse to repeated class means;
- the distribution branch can reduce to a prototype-like scoring regime conceptually.

### 12.4 Flow-matching gradient reachability

Test:

- `tests/test_sc_lfi_losses.py`, lines 74-93

Proves:

- `fm_loss` is finite and positive;
- gradients reach:
  - context encoder
  - latent projector
  - flow model

This is important because it confirms the class context actually influences the FM objective.

### 12.5 Distribution score can improve on a toy task

Test:

- `tests/test_sc_lfi_losses.py`, lines 96-126

Proves:

- the main distribution-fit branch is not dead;
- a few optimization steps can improve same-class margin on a simple synthetic episode.

### 12.6 Factory/CLI integration works

Test:

- `tests/test_sc_lfi_losses.py`, lines 129-179

Proves:

- `build_model_from_args` and the config path are wired correctly.

## 13. What Another LLM Should Conclude

If another LLM reads the code carefully, the correct conclusion should be:

1. `sc_lfi` is not a fake rename of a prototype network.
2. The support-conditioned distribution idea is actually implemented.
3. The FM path is coded exactly as specified in `latent.md`.
4. The scoring rule is genuinely distributional.
5. The current implementation is a conservative, minimal, first stable draft.
6. Lower-than-desired accuracy should be interpreted as a performance optimization problem, not as proof that the architecture was implemented wrongly.

## 14. Priority Weak Points For The Next Iteration

If the objective is to improve accuracy while preserving the theory, the most defensible next steps are:

### 14.1 Replace the default SW with a stronger transport metric

Best immediate candidate:

- switch SC-LFI scoring and align loss to a more exact/paper-style SW estimator instead of the current legacy interpolated SW.

Why:

- the class score is the core classifier;
- improving `D` often gives larger returns than making the flow network deeper.

### 14.2 Increase conditional richness beyond a single context vector

Possible direction:

- keep `h_c`, but also let the flow attend to a small set of class memory tokens rather than only one summary vector.

Why:

- preserves support-conditioned distribution semantics;
- improves multimodal support representation.

### 14.3 Improve particle sampling fidelity

Possible direction:

- more integration steps;
- Heun / RK2 style sampler;
- deterministic evaluation with more particles.

Why:

- cleaner class distributions;
- less Monte Carlo noise during evaluation.

### 14.4 Add token reliability weighting in latent evidence space

Possible direction:

- learned masses on support/query latent tokens before transport scoring.

Why:

- some tokens are likely background/noise;
- uniform empirical measures may be too naive.

### 14.5 Strengthen the align loss

Possible direction:

- separate distance choices for scoring vs anchoring;
- exact SW for classification, entropic OT or weighted SW for anchoring.

Why:

- a single weak `D` may not serve both purposes well.

## 15. Things That Are Correct But Easy To Misread

### 15.1 The flow branch is not trained as a density model with likelihood

This is intentional.

The code follows the flow-matching principle:

- regress a velocity field on a fixed path.

It does not attempt:

- exact log-density evaluation;
- CNF likelihood training;
- image generation.

This is correct for the intended design.

### 15.2 The model does not generate images

The latent flow generates:

- latent evidence particles,

not:

- image pixels.

This is fully aligned with `latent.md`.

### 15.3 The FM loss is class-conditioned even though it uses support tokens

The key point is:

- every support latent token of class `c` is paired with the same class context `h_c`.

So the FM loss is not unconditional token denoising.

### 15.4 Query images do affect learning

Even though the flow loss is support-side, the overall model is still few-shot discriminative because:

- logits come from query-to-class distribution fit;
- cross-entropy is computed on query labels;
- the projector and context encoder are shared across all branches.

## 16. Theory-Faithfulness Labels By Component

This is a compact grading table.

### 16.1 Exact to intended formula

- support token concat per class
- permutation-invariant support context
- latent evidence projection
- linear FM path
- target velocity `e - epsilon`
- class score as negative distribution discrepancy
- optional prototype branch as separate stabilizer

### 16.2 Theory-faithful approximation

- finite-particle class distribution
- Euler flow integration
- fixed-seed deterministic evaluation particles
- optional Sinkhorn OT wrapper

### 16.3 Scientifically acceptable but likely performance bottlenecks

- legacy SW implementation
- conditioning flow only on one class summary vector
- uniform token masses
- relatively small default particle count / integration steps

## 17. Reproduction And Inspection Commands

To run only the SC-LFI tests:

```bash
pytest -q tests/test_sc_lfi_shapes.py tests/test_sc_lfi_losses.py
```

To train/evaluate the model through the main entrypoint, the relevant config hooks are exposed in:

- `main.py`
- `net/model_factory.py`

Key CLI flags:

- `--model sc_lfi`
- `--latent_dim`
- `--class_context_type`
- `--flow_conditioning_type`
- `--distance_type`
- `--use_global_proto_branch`
- `--use_flow_branch`
- `--use_align_loss`
- `--use_smooth_loss`
- `--num_flow_particles`
- `--fm_time_schedule`
- `--score_temperature`
- `--sc_lfi_num_flow_integration_steps`
- `--sc_lfi_lambda_fm`
- `--sc_lfi_lambda_align`
- `--sc_lfi_lambda_smooth`

## 18. Final Bottom-Line Assessment

If the question is:

"Does the current code make the architecture operate in a way that is consistent with the intended theory?"

The answer is:

- yes, at the architectural and loss-definition level;
- yes, at the invariance and episodic semantics level;
- yes, at the "distribution-fit instead of prototype" level.

If the question is:

"Is the current code already the strongest possible numerical realization of that theory?"

The answer is:

- no.

The current implementation is best understood as:

- a correct, minimal, modular, theory-faithful first draft;
- strong enough to validate the research direction;
- not yet optimized enough to claim the ceiling of the idea.
