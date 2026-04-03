# SC-LFI-v3 Implementation Report

## 1. Module-by-module changes

### New modules

- `net/modules/posterior_context_v3.py`
  - builds support empirical measure, support memory, class summary, episode context, uncertainty statistics, support-conditioned prior measure, shot-aware shrinkage coefficient, and posterior base measure

- `net/modules/posterior_transport_flow_v3.py`
  - implements support-basis residual transport flow over posterior base atoms
  - preserves masses under pushforward
  - uses zero-initialized residual velocity output for near-identity startup

- `net/modules/query_conditioned_transport_v3.py`
  - performs query-conditioned reweighting of class-side posterior masses
  - computes final weighted transport score

- `net/modules/posterior_losses_v3.py`
  - defines posterior flow-matching, posterior alignment, distributional margin, and anti-collapse entropy regularization

- `net/sc_lfi_v3.py`
  - integrates the full posterior evidence transport model

### Updated integration

- `net/model_factory.py`
  - adds `sc_lfi_v3` registry entry and builder

- `main.py`
  - adds CLI arguments for `sc_lfi_v3`
  - routes `forward_scores` correctly for `sc_lfi_v3`

### New tests

- `tests/test_sc_lfi_v3_shapes.py`
- `tests/test_sc_lfi_v3_losses.py`

## 2. Exact formulas implemented

### Support empirical measure

- `nu_c^sup = sum_i a_i^sup delta_{e_i}`

with:

- `e_i = Psi(z_i)`
- `a_i^sup = softmax(W_mass(e_i))`

### Support-conditioned prior measure

- `pi_c = sum_r b_r delta_{p_r}`

where:

- `p_r` are support-conditioned prior atoms predicted from class and episode context
- `b_r` are prior masses

### Shot-aware posterior base measure

- `alpha_c = sigma(g_phi(h_c, t_ctx, uncert_c) + s_shot log(1 + K) - s_uncert disp_c)`

- `mu_c^0 = alpha_c * nu_c^sup + (1 - alpha_c) * pi_c`

### Posterior transport

- `muhat_c = (T_theta,c)_# mu_c^0`

implemented as deterministic fixed-step ODE transport over base atoms

### Query-conditioned scoring

- `nu_q = sum_j rho_j delta_{u_j}`
- `muhat_c^q = Reweight(muhat_c ; omega(q,c))`
- `score_c(q) = -tau * D_score(nu_q, muhat_c^q)`

### Loss

- `L_total = L_cls + lambda_margin L_margin + lambda_fm L_fm + lambda_align L_align + lambda_reg L_reg`

with:

- `L_cls`: CE on final posterior transport logits
- `L_margin`: hard-negative distributional margin
- `L_fm`: barycentric posterior flow-matching from base atoms to support-consistent targets
- `L_align`: posterior predictive to support empirical compatibility
- `L_reg`: entropy-floor anti-collapse regularization on support/query/reweighted masses

## 3. What was removed from v2 and why

- heuristic `support_mix_min / support_mix_max`
  - removed because it is not posterior shrinkage

- Gaussian-noise-driven class generation as default class construction
  - removed because it is too generator-centric for few-shot

- anchor-plus-generated class measure as the central object
  - removed because the class should be a posterior predictive evidence measure

- support-fit CE on anchor branch
  - removed because it is a stabilizer, not the cleanest theory-consistent objective

- learned generated-particle masses as the core class measure mass mechanism
  - removed because masses are preserved under pushforward and query-conditioned reweighting now serves the discriminative role

## 4. What was rewritten and why

- support conditioning rewritten into posterior base-measure construction
  - to make the class object explicitly few-shot and uncertainty-aware

- flow rewritten as residual transport over the base measure
  - to preserve support basis semantics

- scoring rewritten with query-conditioned class-side reweighting
  - to absorb the strongest few-shot local matching lesson without losing class-posterior identity

- FM loss rewritten to use barycentric targets from posterior base atoms to support empirical atoms
  - to make flow matching posterior-consistent rather than Gaussian-to-support regression

## 5. Which weaknesses were fixed

- no explicit shot-aware shrinkage
  - fixed by `alpha_c`

- class too generator-centric
  - fixed by transporting a posterior base measure instead of sampling from pure noise

- weak query-conditioned evidence selection
  - fixed by query-conditioned reweighting of class-side masses

- over-trust of one support image in `1-shot`
  - reduced by support-prior shrinkage and entropy-floor regularization

- flow not sufficiently tied to support evidence
  - fixed by deterministic pushforward of support-prior base atoms

## 6. Which approximations still remain

- class measures are still finite empirical measures
- transport is still fixed-step numerical integration
- alignment and FM couplings still use entropic OT approximations
- prior measure is support-conditioned neural prediction, not a closed-form Bayesian prior
- transductive query-batch refinement is not implemented in the default `v3`

## 7. Ablations now enabled

- `use_transport_flow`
- `use_prior_measure`
- `use_episode_adapter`
- `use_query_reweighting`
- `use_support_barycenter_only`
- `use_global_proto_branch`
- solver type and integration steps
- prior atom count and scale
- score / alignment transport settings
- entropy target ratios

## 8. Next likely bottlenecks

- the prior measure is still generic neural shrinkage, not a more structured conditional prior family
- query-conditioned reweighting is mass-only; atom-level query-conditioned residual refinement could be explored later
- low-shot regularization may still benefit from support consistency under light augmentation
- backbone quality remains an upper bound once the head becomes more principled
