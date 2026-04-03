# SC-LFI Design Note

## 1. Architecture Summary

SC-LFI (Support-Conditioned Latent Flow Inference) treats each class as a support-conditioned latent evidence distribution instead of a single prototype.

Pipeline per episode:

1. Backbone/tokenizer:
   - image `x -> Z(x) in R^{M x d}`
2. Support-set context encoder:
   - merged support tokens of class `c`: `Z_c in R^{(K*M) x d}`
   - context: `h_c = Phi_set(Z_c) in R^{d_h}`
3. Latent evidence projector:
   - token evidence: `e = Psi(z) in R^{d_l}`
4. Conditional latent flow:
   - velocity field: `v_theta(y, t; h_c)`
   - pushforward from `p0 = N(0, I)` produces class particles `muhat_c`
5. Query/class scoring:
   - query evidence distribution `nu_q`
   - score `s_c(q) = -D(nu_q, muhat_c)`

Version 1 keeps the implementation minimal:

- support-conditioned class context is permutation invariant;
- flow matching uses the linear path specified in `latent.md`;
- scoring uses sliced Wasserstein by default;
- entropic OT is available through a pure-PyTorch Sinkhorn wrapper;
- a conservative global prototype branch is optional and disabled by default.

## 2. Formula -> Module Mapping

| Formula / object | Module | Notes |
| --- | --- | --- |
| `h_c = Phi_set(Z_c)` | `net/modules/set_context.py` | DeepSets weighted pooling or lightweight set transformer |
| `e = Psi(z)` | `net/modules/latent_projector.py` | 2-layer MLP + normalization |
| `v_theta(y, t; h_c)` | `net/modules/conditional_flow.py` | concat or FiLM conditioning |
| `y_t = (1-t) * epsilon + t * e` | `net/modules/conditional_flow.py` | fixed conditional path, few-shot flow-matching adaptation |
| `u_t = e - epsilon` | `net/modules/conditional_flow.py` | exact target velocity for the linear path |
| `L_FM` | `net/modules/flow_losses.py` | mean squared velocity regression |
| `L_align = D(muhat_c, nu_c^sup)` | `net/modules/flow_losses.py` | support anchoring |
| `L_smooth` | `net/modules/flow_losses.py` | optional nearest-context regularizer |
| `s_c(q) = -D(nu_q, muhat_c)` | `net/modules/distribution_distance.py`, `net/sc_lfi.py` | distribution-fit classifier |

## 3. Tensor Shapes

Per episode, with:

- `Way = N`
- `Shot = K`
- `M` backbone tokens per image
- `d` backbone token dim
- `d_h` class-context dim
- `d_l` latent evidence dim
- `L` flow particles
- `Q` query images in the episode

Main tensors:

- `query`: `[Q, C, H, W]`
- `support`: `[Way, Shot, C, H, W]`
- `query_tokens`: `[Q, M, d]`
- `support_tokens`: `[Way, Shot, M, d]`
- merged support tokens `Z_c`: `[Way, Shot*M, d]`
- class contexts `h`: `[Way, d_h]`
- `query_latents`: `[Q, M, d_l]`
- `support_latents`: `[Way, Shot*M, d_l]`
- sampled class particles `muhat`: `[Way, L, d_l]`
- expanded query distributions for scoring: `[Q, Way, M, d_l]`
- expanded class distributions for scoring: `[Q, Way, L, d_l]`
- logits: `[Q, Way]`

## 4. Implemented Formulas And Code Mapping

Flow-matching target path:

`y_t = (1 - t) epsilon + t e`

Implemented by:

- `sample_linear_conditional_path(...)` in `net/modules/conditional_flow.py`

Target velocity:

`u_t(epsilon, e) = e - epsilon`

Implemented by:

- `target_linear_path_velocity(...)` in `net/modules/conditional_flow.py`

Few-shot flow-matching loss:

`L_FM = E || v_theta(y_t, t; h_c) - (e - epsilon) ||_2^2`

Implemented by:

- `compute_flow_matching_loss(...)` in `net/modules/flow_losses.py`

Support anchoring:

`L_align = D(muhat_c, nu_c^sup)`

Implemented by:

- `compute_support_anchoring_loss(...)` in `net/modules/flow_losses.py`

Query classification:

`s_c(q) = -D(nu_q, muhat_c)`

Implemented by:

- `_score_query_against_classes(...)` in `net/sc_lfi.py`

Optional fused score:

`s_c = (1 - alpha) s_c^dist + alpha s_c^glob`

Implemented by:

- `_compute_global_proto_scores(...)` and score fusion in `net/sc_lfi.py`

## 5. Reference Mapping

- Flow Matching for Generative Modeling:
  - informed the fixed conditional path and vector-field regression principle
- Flow Matching in Latent Space:
  - informed using a latent state space for velocity-field learning instead of pixel-space generation
- GENTLE:
  - informed the support-anchoring / conditional-distribution-regularization viewpoint
- Prototype-bias few-shot literature:
  - motivated replacing point prototypes with support-conditioned distributions

No module claims to directly reproduce an exact equation from those papers unless the formula is the standard linear flow-matching path itself.

## 6. Novelty / Borrowed Principle / Stabilizer

Core novelty:

- support-conditioned latent evidence distributions for few-shot class inference
- class scoring by conditional distribution fit instead of prototype cosine
- few-shot adaptation of flow matching over support-derived latent evidence

Borrowed principles:

- fixed conditional probability paths and velocity regression from flow matching
- latent-space state modeling from latent flow work
- conditional-distribution regularization idea from GENTLE

Engineering stabilizers:

- sliced Wasserstein default distance
- optional entropic OT Sinkhorn wrapper
- optional conservative global prototype branch
- Euler flow sampling with small configurable step count

## 7. Conservative Defaults

- `latent_dim = 64`
- `class_context_type = deepsets`
- `flow_conditioning_type = concat`
- `distance_type = sw`
- `num_flow_particles = 16`
- `num_flow_integration_steps = 8`
- `score_temperature = 8.0`
- `lambda_fm = 0.05`
- `lambda_align = 0.1`
- `lambda_smooth = 0.0`
- `use_global_proto_branch = false`

## 8. Known Risks / Failure Modes

- If `lambda_fm` is too large, the velocity field can dominate early episodic CE training.
- If `num_flow_particles` is too small, class distributions collapse toward a noisy Monte-Carlo estimate.
- If `distance_type = entropic_ot` and Sinkhorn regularization is too small, transport can become numerically stiff.
- If the backbone features are weak, the flow branch may overfit support noise rather than useful evidence structure.
- The optional prototype branch can hide distribution-branch failures if `proto_branch_weight` is set too high.
