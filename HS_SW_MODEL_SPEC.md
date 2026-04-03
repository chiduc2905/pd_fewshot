# HS-SW Model Spec

## 1. Method name

**HS-SW**: Hierarchical Support Sliced-Wasserstein Inference

Current implementation in `pulse_fewshot`:

- model id: `hierarchical_support_sw_net`
- display name: `HS-SW`
- architecture summary:
  `Backbone token pyramid -> scale-budgeted shot measures -> exact weighted sliced Wasserstein -> support-aware hierarchical inference`

Main code files:

- `net/hierarchical_support_sw_net.py`
- `net/heads/hierarchical_support_sw_head.py`
- `net/metrics/hssw_sliced_wasserstein.py`
- `net/fewshot_common.py`
- `net/model_factory.py`
- `main.py`

## 2. Design goal

HS-SW is a few-shot classifier head built around **shot-preserving token measures**.

The core design is:

- each image is represented as a set of local tokens
- each support shot is a weighted empirical measure over tokens
- each class is a set of shot-level measures
- query-class scoring is computed by hierarchical inference over shot-level SW distances

The current version also upgrades the shot measure from a single local token set to a **multi-scale token measure**:

- local tokens from the original feature map
- coarse pooled tokens
- global pooled token

This change is part of the model contribution, not an optimization trick.

Reason:

- in `1-shot`, vanilla HS-SW collapses to a single query-shot local SW matcher
- that is too weak if only fine local tokens are used
- a multi-scale measure gives each shot a stronger structural representation while keeping the same HS-SW head

## 3. Contribution vs optional stabilizers

### 3.1 Core contribution

The contribution of the model is:

- multi-scale shot-level empirical measures
- weighted or unweighted token measures
- exact 1D Wasserstein transport on random projections
- sliced Wasserstein distance between token measures
- shot-level responsibilities within each class
- support consistency regularization
- support redundancy regularization

### 3.2 Optional stabilizers

The following are **not** the architectural contribution:

- token weight temperature
- uniform mixing into token masses
- token-mass regularization to uniform
- learnable class-logit scale

These exist only as optional implementation knobs. In the clean contribution path, they should remain off or neutral:

- `hssw_token_weight_temperature = 1.0`
- `hssw_token_weight_mix_alpha = 1.0`
- `hssw_token_mass_reg_weight = 0.0`
- `hssw_learn_logit_scale = false`

## 4. Episodic setting and tensor shapes

Standard episodic few-shot classification:

- support input: `[B, Way, Shot, C, H, W]`
- query input: `[B, NQ, C, H, W]`

Backbone output per image:

- feature map: `F(x) in R^(D x Hf x Wf)`

Per-image token groups:

- local tokens: `[M_local, D]`, where `M_local = Hf * Wf`
- pooled-grid tokens for each configured grid size `g`: `[g*g, D]`

After concatenating all scales:

- per-image tokens: `[M_total, D]`
- per-image token weights: `[M_total]`

Per episode:

- query tokens: `[NQ, M_total, D]`
- query weights: `[NQ, M_total]`
- support tokens: `[Way, Shot, M_total, D]`
- support weights: `[Way, Shot, M_total]`

Shot-level outputs:

- shot distances: `[NQ, Way, Shot]`
- shot responsibilities: `[NQ, Way, Shot]`
- base distance: `[NQ, Way]`
- consistency penalty: `[NQ, Way]`
- redundancy penalty: `[Way]`
- final class distance: `[NQ, Way]`
- logits: `[NQ, Way]`

Batch output returned by the model:

- logits for training/eval: `[B * NQ, Way]`

## 5. Multi-scale shot measure

Given one image `x`, let the backbone feature map be `F(x)`.

### 5.1 Token groups

Define a list of token groups:

- local group:
  `T^(0)(x) = { t_i^(0)(x) }_(i=1..M_0 )`
- pooled groups:
  for each configured grid size `g_s`,
  `T^(s)(x) = { t_i^(s)(x) }_(i=1..M_s )`

In code, the token groups are produced by:

- local tokens from the original feature map
- pooled tokens from `adaptive_avg_pool2d(feature_map, output_size=(g, g))`

### 5.2 Per-scale mass budgets

Let the scale budgets be:

`rho = (rho_0, rho_1, ..., rho_S)`

with:

- `rho_s > 0`
- `sum_s rho_s = 1`

These are normalized internally even if the raw config values do not sum to `1`.

### 5.3 Token masses within each scale

For each token group `T^(s)(x)`, a lightweight scorer outputs logits:

`z_i^(s)(x) = g_theta( t_i^(s)(x) )`

Then per-scale token masses are:

`alpha_i^(s)(x) = softmax_i( z_i^(s)(x) / T_w )`

where `T_w` is `hssw_token_weight_temperature`.

If uniform mixing is enabled, the masses become:

`alpha'_i^(s)(x) = (1 - beta) / M_s + beta * alpha_i^(s)(x)`

where `beta = hssw_token_weight_mix_alpha`.

The final transport mass assigned to token `i` in scale `s` is:

`w_i^(s)(x) = rho_s * alpha'_i^(s)(x)`

If `beta = 1` and `T_w = 1`, this reduces to the clean learned scale-local mass assignment.

### 5.4 Final image measure

Concatenate all token groups and their scale-budgeted masses:

`T(x) = concat( T^(0)(x), T^(1)(x), ..., T^(S)(x) )`

`w(x) = concat( w^(0)(x), w^(1)(x), ..., w^(S)(x) )`

This defines a weighted empirical measure:

`mu_x = sum_i w_i(x) delta_( t_i(x) )`

For a support class `c` and shot `k`:

`mu_c^k = mu_( x_c^k )`

For a query image `q`:

`nu_q = mu_q`

## 6. Exact sliced Wasserstein used in HS-SW

This implementation does **not** use a fake Wasserstein surrogate.

Given two token measures:

`mu = sum_i a_i delta_(x_i)`

`nu = sum_j b_j delta_(y_j)`

with:

- `a_i >= 0`, `sum_i a_i = 1`
- `b_j >= 0`, `sum_j b_j = 1`

### 6.1 Projection directions

Sample `L` random directions:

`theta_l in R^D`, `||theta_l||_2 = 1`, for `l = 1..L`

Each direction is sampled from a Gaussian vector and then L2-normalized.

Projection modes:

- `fixed`: sample once from a deterministic seed and reuse
- `resample`: sample a fresh projection bank each forward call

### 6.2 1D projected measures

For each projection `theta_l`:

`p_i^(l) = <x_i, theta_l>`

`q_j^(l) = <y_j, theta_l>`

Sort projected values and carry the corresponding masses with the same permutation.

### 6.3 Exact weighted 1D Wasserstein

For each projection `l`, define the projected 1D measures:

`P_l # mu`

`P_l # nu`

Their `p`-Wasserstein cost is computed exactly by quantile transport:

`W_p^p( P_l # mu, P_l # nu ) = int_0^1 | F_(mu,l)^(-1)(t) - F_(nu,l)^(-1)(t) |^p dt`

Implementation detail:

- build cumulative masses
- merge CDF breakpoints from both measures
- find quantile indices with `searchsorted`
- accumulate interval mass times quantile gap to the power `p`

### 6.4 Sliced Wasserstein

The final Monte Carlo sliced Wasserstein estimator is:

`SW_p(mu, nu) = ( (1 / L) sum_(l=1..L) W_p^p( P_l # mu, P_l # nu ) )^(1/p)`

This is the metric used for:

- query-to-shot distances
- support-shot redundancy distances

### 6.5 Unweighted variant

If `hssw_weighted_sw = false`, the same SW pipeline is used but each token set uses uniform masses:

`a_i = 1 / M_x`

`b_j = 1 / M_y`

## 7. Hierarchical Support SW inference rule

For each class `c` and shot `k`:

`d_(c,k)(q) = SW( nu_q, mu_c^k )`

For each class `c`, define shot responsibilities over shots only:

`gamma_(c,k)(q) = softmax_k( - d_(c,k)(q) / tau )`

Base class distance:

`D_base(c,q) = sum_k gamma_(c,k)(q) * d_(c,k)(q)`

Support consistency penalty:

`Omega_cons(c,q) = - sum_k log( eps + gamma_(c,k)(q) )`

Support redundancy penalty:

`Omega_red(c) = average_(k != l) exp( - SW( mu_c^k, mu_c^l ) )`

Final hierarchical class distance:

`D_HS_SW(c,q) = D_base(c,q) + lambda_cons * Omega_cons(c,q) + lambda_red * Omega_red(c)`

Default clean logit:

`logit(c,q) = - D_HS_SW(c,q)`

If optional logit scaling is enabled:

`logit(c,q) = - s * D_HS_SW(c,q)`

where `s > 0` is a learnable positive scalar.

Loss:

- standard cross-entropy over class logits

## 8. Current default config in code

The following are the current parser defaults in `main.py`.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--hssw_token_weight_hidden` | `128` | hidden width of token scorer MLP |
| `--hssw_token_weight_temperature` | `1.0` | temperature for token-mass softmax |
| `--hssw_token_weight_mix_alpha` | `1.0` | learned-vs-uniform mass mixing coefficient |
| `--hssw_token_mass_reg_weight` | `0.0` | auxiliary KL-to-uniform weight for token masses |
| `--hssw_multiscale_grids` | `"4,1"` | pooled grids appended after local tokens |
| `--hssw_multiscale_mass_budgets` | `"0.6,0.3,0.1"` | mass budgets for local, `4x4`, and `1x1` token groups |
| `--hssw_token_l2norm` | `true` | L2-normalize tokens before projection |
| `--hssw_weighted_sw` | `true` | use weighted token measures instead of uniform masses |
| `--hssw_train_num_projections` | `32` | number of train-time projections |
| `--hssw_eval_num_projections` | `64` | number of eval-time projections |
| `--hssw_sw_p` | `2.0` | Wasserstein power `p` |
| `--hssw_train_projection_mode` | `resample` | train projection-bank mode |
| `--hssw_eval_projection_mode` | `fixed` | eval projection-bank mode |
| `--hssw_eval_num_repeats` | `1` | number of repeated SW evaluations in eval mode |
| `--hssw_projection_seed` | `7` | deterministic seed for projection sampling |
| `--hssw_pairwise_chunk_size` | `0` | support chunking for pairwise SW, `0` means no chunking |
| `--hssw_tau` | `0.2` | shot responsibility temperature |
| `--hssw_lambda_cons` | `0.1` | weight of support consistency term |
| `--hssw_lambda_red` | `0.02` | weight of support redundancy term |
| `--hssw_learn_logit_scale` | `false` | enable learnable positive logit scale |
| `--hssw_logit_scale_init` | `10.0` | initial value if learnable logit scale is enabled |
| `--hssw_logit_scale_max` | `100.0` | clamp for learnable logit scale |
| `--hssw_eps` | `1e-6` | numeric epsilon used across the method |

## 9. Recommended clean contribution config

For contribution-focused benchmarking, keep the non-contribution stabilizers neutral:

```bash
python3 run_all_experiments.py \
  --models hierarchical_support_sw_net \
  --fewshot_backbone resnet12 \
  --hssw_weighted_sw true \
  --hssw_multiscale_grids 4,1 \
  --hssw_multiscale_mass_budgets 0.6,0.3,0.1 \
  --hssw_token_weight_temperature 1.0 \
  --hssw_token_weight_mix_alpha 1.0 \
  --hssw_token_mass_reg_weight 0.0 \
  --hssw_learn_logit_scale false
```

If strict train/eval estimator matching is desired:

```bash
--hssw_train_num_projections 64 \
--hssw_eval_num_projections 64 \
--hssw_train_projection_mode fixed \
--hssw_eval_projection_mode fixed
```

## 10. Notes on fairness and the all-samples issue

If `training_samples` increases, accuracy should ideally not get worse.

In the current benchmark protocol, there is an external limitation:

- `run_all_experiments.py` uses a fixed `query_num_train = 1`
- `main.py` rebuilds a `FewshotDataset` each epoch with a fixed `episode_num_train = 200`
- `training_samples` changes the pool of available training images
- but the number of episodes per epoch does not scale with the pool size

So the `All` setting may underuse the larger dataset relative to its size.

This is a **protocol-level issue**, not purely an HS-SW architecture issue.

The HS-SW contribution in this repo does **not** change that global training protocol, to stay fair with the other benchmarked models.

## 11. Why this version is defensible

This implementation is designed to be defensible on four points:

- the transport metric is a real sliced Wasserstein estimator, not a heuristic projection similarity
- support shots remain explicit objects throughout inference
- `1-shot` is strengthened by improving the shot measure itself, not by adding a separate classifier branch
- optional optimization stabilizers are separated from the contribution and can be disabled cleanly

In short:

- the head contribution is hierarchical support inference over shot-level SW distances
- the representation contribution is a multi-scale shot measure
- the benchmark path remains compatible with the surrounding `pulse_fewshot` episodic pipeline
