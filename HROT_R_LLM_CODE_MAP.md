# HROT-R LLM Code Map

This file is a single-reference Markdown document for understanding the `hrot_fsl` model with `--hrot_variant R` in this repository.

Goal:
- give another LLM enough context to reason about the implementation without re-reading the whole repo first
- explain the exact training-time call path
- show the main tensors, modules, and equations
- include the most important source-code excerpts in one place

Important scope note:
- `HROT-R` is not a separate class. It is `HROTFSL` with `variant="R"`.
- Variant `R` is "Q plus structure-consistent cost".
- Training uses the normal episodic sampler (`FewshotDataset`), not the robust final-test sampler.
- `--test_protocol robust` only changes the final-test dataset build path in `main.py`; it does not change the core `HROT-R` training forward pass.

---

## 1. Files You Actually Need

Core implementation:
- `net/hrot_fsl.py`
- `net/modules/episode_adaptive_mass.py`
- `net/modules/unbalanced_ot.py`
- `net/fewshot_common.py`
- `net/hyperbolic/poincare_ops.py`

Training and routing:
- `main.py`
- `net/model_factory.py`
- `run_all_experiments.py`

Episode sampling:
- `dataloader/dataloader.py`

Behavioral contract:
- `tests/test_hrot_fsl.py`

---

## 2. One-Screen Mental Model

`HROT-R` takes few-shot episodes and does this:

1. Encode query and support images into spatial tokens.
2. Keep two token spaces at once:
   - Euclidean tokens for the actual token-to-token transport cost.
   - Hyperbolic tokens for geometry-aware mass prediction and token priors.
3. Predict per-shot transport budget `rho` with two heads:
   - `eam`: H-style geodesic transported-mass predictor.
   - `q_eam`: Q/R-style predictor that also sees shot cross-attention.
4. Run a detached probe UOT to estimate token reliability.
5. Turn probe statistics into query/support token marginals.
6. For `R`, build an additional structure-consistency cost from the detached probe plan.
7. Add a learnable noise sink so unmatched evidence can be discarded.
8. Solve final UOT with:
   - custom token marginals
   - optional structure cost
   - noise sink
9. Score each shot with a thresholded mass-vs-cost rule.
10. Pool shots with learned weights.
11. Blend the Q/R enhanced score with an H-anchor score.
12. Train with cross-entropy on final logits plus a small auxiliary regularization term.

The most important implementation fact is this:

- Variant `R` still uses Euclidean token matching cost.
- Hyperbolic geometry is used to predict budgets, token priors, and shot features.
- `R` differs from `Q` only by adding the posterior-derived structure-consistency cost before the final transport solve.

---

## 3. Top-Level Call Graph

### 3.1 Batch runner

`run_all_experiments.py`

```python
cmd = [
    sys.executable,
    target_script,
    "--model", model,
    "--shot_num", str(shot),
    "--way_num", "4",
    "--query_num_train", str(TRAIN_QUERY_NUM),
    "--query_num_val", str(EVAL_QUERY_NUM),
    "--query_num_test", str(EVAL_QUERY_NUM),
    "--image_size", "84",
    "--mode", "train",
    "--project", project,
    "--dataset_path", dataset_path,
    "--dataset_name", dataset_name,
    "--path_weights", "checkpoints/",
    "--path_results", "results/",
    "--num_epochs", "100",
    "--lr", "5e-4",
    "--scheduler", "cosine",
    "--warmup_epochs", "5",
    "--min_lr", "1e-6",
    "--weight_decay", "5e-4",
    "--label_smoothing", "0.0",
    "--train_augment", "false",
    "--episode_num_train", str(TRAIN_EPISODES_PER_EPOCH),
    "--episode_num_val", str(TEST_EPISODES_PER_EPOCH),
    "--episode_num_test", str(TEST_EPISODES_PER_EPOCH),
]
```

Default benchmark settings from the runner:
- `way_num = 4`
- `query_num_train = query_num_val = query_num_test = 1`
- `image_size = 84`
- `num_epochs = 100`
- `lr = 5e-4`
- scheduler = cosine with 5 warmup epochs
- `weight_decay = 5e-4`
- train augmentation off
- `episode_num_train = 130`
- `episode_num_val = episode_num_test = 150`
- model selection split = `val`
- `merge_val_into_train = false`

### 3.2 Main entrypoint

`main.py`

```text
main()
  -> load_dataset(...)
  -> build episodic datasets/loaders
  -> net = get_model(args)
  -> train_loop(...)
       -> forward_scores(...)
            -> net(query, support, ..., return_aux=collect_diagnostics)
       -> loss = cross_entropy(logits, targets) + aux_loss
  -> load best checkpoint
  -> test_final(...)
```

### 3.3 Model creation

`net/model_factory.py`

```python
if args.model == "hrot_fsl":
    HROTFSL = _load_symbol("net.hrot_fsl", "HROTFSL")
    hidden_dim = fewshot_backbone_output_dim(fewshot_backbone)
    return HROTFSL(
        in_channels=3,
        hidden_dim=hidden_dim,
        token_dim=int(getattr(args, "hrot_token_dim", getattr(args, "token_dim", 128)) or 128),
        use_raw_backbone_tokens=_bool_flag(getattr(args, "hrot_use_raw_backbone_tokens", "false"), default=False),
        backbone_name=fewshot_backbone,
        image_size=image_size,
        variant=str(getattr(args, "hrot_variant", "E")),
        eam_hidden_dim=int(getattr(args, "hrot_eam_hidden_dim", 256)),
        curvature_init=float(getattr(args, "hrot_curvature_init", 1.0)),
        projection_scale=float(getattr(args, "hrot_projection_scale", 0.1)),
        token_temperature=float(getattr(args, "hrot_token_temperature", 0.1)),
        score_scale=float(getattr(args, "hrot_score_scale", 16.0)),
        tau_q=float(getattr(args, "hrot_tau_q", 0.5)),
        tau_c=float(getattr(args, "hrot_tau_c", 0.5)),
        sinkhorn_epsilon=float(getattr(args, "hrot_sinkhorn_epsilon", 0.1)),
        sinkhorn_iterations=int(getattr(args, "hrot_sinkhorn_iterations", 60)),
        sinkhorn_tolerance=float(getattr(args, "hrot_sinkhorn_tolerance", 1e-5)),
        fixed_mass=float(getattr(args, "hrot_fixed_mass", 0.8)),
        min_mass=float(getattr(args, "hrot_min_mass", 0.1)),
        mass_bonus_init=float(getattr(args, "hrot_mass_bonus_init", 1.0)),
        transport_cost_threshold_init=getattr(args, "hrot_transport_cost_threshold_init", None),
        lambda_rho=float(getattr(args, "hrot_lambda_rho", 0.01)),
        rho_target=float(getattr(args, "hrot_rho_target", 0.8)),
        lambda_rho_rank=float(getattr(args, "hrot_lambda_rho_rank", 0.05)),
        rho_rank_margin=float(getattr(args, "hrot_rho_rank_margin", 0.05)),
        rho_rank_temperature=float(getattr(args, "hrot_rho_rank_temperature", 0.05)),
        lambda_curvature=float(getattr(args, "hrot_lambda_curvature", 0.0)),
        min_curvature=float(getattr(args, "hrot_min_curvature", 0.05)),
        structure_cost_init=float(getattr(args, "hrot_structure_cost_init", 0.05)),
        normalize_euclidean_tokens=_bool_flag(getattr(args, "hrot_normalize_euclidean_tokens", "true"), default=True),
        normalize_rho=_bool_flag(getattr(args, "hrot_normalize_rho", "false"), default=False),
        eval_use_float64=_bool_flag(getattr(args, "hrot_eval_use_float64", "true"), default=True),
        hyperbolic_backend=str(getattr(args, "hrot_hyperbolic_backend", "auto")),
        ot_backend=str(getattr(args, "hrot_ot_backend", "native")),
        eps=float(getattr(args, "hrot_eps", 1e-6)),
    )
```

---

## 4. Training-Time Fact That Is Easy To Miss

`main.py` routes `hrot_fsl` the same way in train, val, and test:

```python
if args.model == "hrot_fsl":
    return net(
        query,
        support,
        query_targets=query_targets,
        support_targets=support_targets,
        return_aux=collect_diagnostics,
    )
```

Unlike `DeepEMD`, there is no special train-vs-val-vs-test branch in `forward_scores` for `HROT-R`.

That means:
- same model forward logic in train/val/test
- only `model.train()` / `model.eval()` changes behavior
- the main eval-only difference is that some hyperbolic computations switch to `float64` when `hrot_eval_use_float64=true`

Also important:
- per-epoch selection uses `FewshotDataset`, not `RobustFewshotDataset`
- robust protocol is only used in final test construction

---

## 5. Episode Sampler And Tensor Shapes

### 5.1 Standard training sampler

`FewshotDataset.__getitem__` returns:
- `query_images`
- `query_targets`
- `support_images`
- `support_targets`

It samples class-wise support first, then query:

```python
support_idx = shuffled[: self.shot_num]
query_idx = shuffled[self.shot_num : self.shot_num + self.query_num]
```

### 5.2 Batch shapes in `train_loop`

Inside `main.py`:

```python
support = support.view(batch_size, args.way_num, args.shot_num, channels, height, width)
query = query.to(device)
targets = q_labels.view(-1).to(device)
support_targets = support_labels.view(batch_size, args.way_num, args.shot_num).to(device)
```

Shape conventions:
- query batch entering the model: `(B, NumQuery, C, H, W)`
- support batch entering the model: `(B, Way, Shot, C, H, W)`
- flattened class targets used for CE: `(B * NumQuery,)`

Inside one episode (`HROTFSL._forward_episode`):
- `query`: `(NumQuery, C, H, W)`
- `support`: `(Way, Shot, C, H, W)`

After encoding:
- `query_euclidean`: `(NumQuery, T, D)`
- `query_hyperbolic`: `(NumQuery, T, D)`
- `support_euclidean`: `(Way, Shot, T, D)`
- `support_hyperbolic`: `(Way, Shot, T, D)`
- `class_euclidean`: `(Way, Shot*T, D)`
- `class_hyperbolic`: `(Way, Shot*T, D)`

Where:
- `T = feature_map_height * feature_map_width`
- `D = token_dim` unless `use_raw_backbone_tokens=true`, in which case `D = hidden_dim`

---

## 6. Variant Flags: Why `R` Behaves The Way It Does

In `HROTFSL.__init__`:

```python
self.uses_unbalanced_transport = variant in {"B", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S"}
self.uses_learned_mass = variant in {"E", "F", "G", "H", "I", "K", "L", "M", "N", "O", "P", "Q", "R", "S"}
self.uses_shot_decomposed_transport = variant in {"G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S"}
self.uses_geodesic_eam = variant in {"H", "K", "L", "M", "N", "O", "P", "Q", "R", "S"}
self.uses_cost_threshold_score = variant in {"H", "I", "J", "L", "M", "N", "O", "P", "Q", "R", "S"}
self.uses_noise_calibrated_transport = variant in {"Q", "R"}
self.uses_structure_consistent_transport = variant == "R"
```

What this means for `R`:
- yes: unbalanced transport
- yes: learned transported mass `rho`
- yes: shot-decomposed matching
- yes: geodesic EAM features
- yes: threshold-based scoring
- yes: Q-style noise-calibrated token marginals
- yes: structure-consistency cost
- no: direct hyperbolic token cost (`uses_hyperbolic_geometry` is false for `R`)
- no: rho rank loss (`uses_rho_rank_loss` is only true for variant `N`)
- no: S-style geodesic shot pooling variant flag (`uses_geodesic_shot_pooling` is only true for `S`)

The direct consequence is:
- `R` uses Euclidean cost matrices for token matching.
- Hyperbolic geometry is a control signal, not the final ground cost.

---

## 7. Learnable Modules And Parameters In Variant R

### 7.1 Backbone and tokenizer

- `BaseConv64FewShotModel.backbone`
- optional `backbone_adapter`
- `token_projector`:
  - identity if raw backbone tokens are used
  - otherwise `LayerNorm(hidden_dim) -> Linear(hidden_dim, token_dim, bias=False)`

### 7.2 Hyperbolic geometry

- `curvature` is learnable
- backend is `geoopt` if available and requested, otherwise native PyTorch Poincare-ball ops

### 7.3 Episode-adaptive mass heads

- `eam`: H-anchor budget head
- `q_eam`: Q/R budget head

`EpisodeAdaptiveMass` is a small MLP:

```python
self.network = nn.Sequential(
    nn.Linear(self.input_dim, hidden_dim),
    nn.LayerNorm(hidden_dim),
    nn.GELU(),
    nn.Linear(hidden_dim, max(1, hidden_dim // 2)),
    nn.LayerNorm(max(1, hidden_dim // 2)),
    nn.GELU(),
    nn.Linear(max(1, hidden_dim // 2), 1),
    nn.Sigmoid(),
)
```

### 7.4 Q/R-only learned parameters

Variant `R` adds or uses these learnable tensors:
- `query_reliability_weights`
- `support_reliability_weights`
- `query_token_attention_vector`
- `support_token_attention_vector`
- `raw_token_reliability_mix`
- `raw_support_consensus_mix`
- `raw_hyperbolic_token_prior_mix`
- `raw_eam_cross_attention_temperature`
- `raw_consensus_temperature`
- `raw_noise_sink_cost`
- `raw_shot_pool_temperature`
- `raw_shot_pool_mix`
- `raw_q_enhancement_mix`
- `raw_structure_cost_weight`
- `q_shot_pool_scorer`
- `q_threshold_scorer`

Parameterization style:
- positive scalars use `softplus`
- mixing weights use `sigmoid`

This keeps the public properties well behaved:
- `token_reliability_mix in (0, 1)`
- `support_consensus_mix in (0, 1)`
- `hyperbolic_token_prior_mix in (0, 1)`
- `shot_pool_mix in (0, 1)`
- `q_enhancement_mix in (0, 1)`
- `structure_cost_weight > 0`
- `noise_sink_cost > 0`
- `transport_cost_threshold > 0`

---

## 8. Exact Forward Pass For Variant R

This is the real heart of the model.

### 8.1 Episode encoding

`_forward_episode` starts with:

```python
query_euclidean, query_hyperbolic, query_hw = self._encode_images(query)
support_euclidean, support_hyperbolic, support_hw = self._encode_images(
    support.reshape(way_num * shot_num, *support.shape[-3:])
)
```

`_encode_images` does:
1. `self.encode(images)` to get feature maps
2. `feature_map_to_tokens(feature_map)` to convert `(N, C, H, W)` into `(N, H*W, C)`
3. `token_projector`
4. optional Euclidean L2 normalization
5. scale projected tokens by `projection_scale`
6. map into the Poincare ball using `safe_project_to_ball`

So every image produces:
- Euclidean token set
- Hyperbolic token set

### 8.2 Build the flat support tensor

Because `R` is shot-decomposed, it flattens support by shot:

```python
flat_support_euclidean = support_euclidean.reshape(way_num * shot_num, T, D)
flat_support_hyperbolic = support_hyperbolic.reshape(way_num * shot_num, T, D)
```

Base token cost is Euclidean:

```python
flat_cost = self._euclidean_cost(query_euclidean, flat_support_euclidean)
```

Shape:
- `flat_cost`: `(NumQuery, Way*Shot, Tq, Ts)`

### 8.3 Geodesic shot features for H and Q/R budget heads

`_build_q_geodesic_eam_features` returns:
- `q_geodesic_features`: the usual geodesic features plus a shot cross-attention scalar
- `geodesic_features`: the H-anchor features without the extra cross-attention channel

Base geodesic features are:
- mean hyperbolic distance between query mean and shot mean
- shot spread inside the class
- query variance
- support variance

For Q/R, an extra cross-attention feature is appended:

```python
distance = geodesic_features[..., 0]
inv_distance = distance.clamp_min(self.eps).reciprocal()
cross_attention = torch.softmax(inv_distance / temperature, dim=-1) * float(distance.shape[-1])
q_features = torch.cat([geodesic_features, cross_attention.unsqueeze(-1)], dim=-1)
```

### 8.4 Two budget predictions: H-anchor and Q/R

```python
h_anchor_rho = self.eam.forward_features(geodesic_features)
shot_rho = self.q_eam.forward_features(q_geodesic_features)
```

Both are normalized by `_normalize_rho_budget` if `hrot_normalize_rho=true`.

Interpretation:
- `h_anchor_rho`: conservative H-style transported mass
- `shot_rho`: richer Q/R transported mass

### 8.5 H-anchor scoring path

The model keeps an H-style anchor score:

```python
_h_anchor_plan, h_anchor_flat_cost, h_anchor_flat_mass = self._transport_match(
    flat_cost,
    flat_h_anchor_rho,
)

h_anchor_shot_logits = self.score_scale * (
    global_threshold * h_anchor_shot_transport_mass - h_anchor_shot_transport_cost
)
h_anchor_logits = h_anchor_shot_logits.mean(dim=-1)
```

This anchor branch does not use:
- token reliability
- structure cost
- noise sink

It acts as a stabilizing reference branch.

### 8.6 Probe transport for token reliability

`_compute_noise_calibrated_token_marginals(...)` is the key Q/R mechanism.

#### Step A: detached probe plan

```python
probe_plan, probe_cost, probe_mass = self._transport_match(flat_cost.detach(), flat_rho.detach())
```

From the detached probe plan it builds per-token features:
- row mass share
- row average cost
- row entropy
- row min cost
- and the support-side analogs

These are standardized over tokens.

#### Step B: reliability logits

```python
query_logits = torch.einsum("qstf,f->qst", query_features, self.query_reliability_weights)
support_logits = torch.einsum("qstf,f->qst", support_features, self.support_reliability_weights)
```

#### Step C: support consensus prior

Support consensus is computed from same-class support-shot agreement:

```python
pair_cost = (
    support_tokens_euc[:, :, None, :, None, :] - support_tokens_euc[:, None, :, None, :, :]
).pow(2).sum(dim=-1)
nearest_other = pair_cost.amin(dim=-1)
mean_nearest = nearest_other.sum(dim=2) / float(shot_num - 1)
scores = -mean_nearest / temperature
```

This says:
- a support token is more trusted if it finds similar tokens in other shots of the same class

#### Step D: hyperbolic token prior

`_compute_hyperbolic_token_prior_logits`:
- maps hyperbolic tokens back to tangent space
- projects them along learned direction vectors
- combines directional evidence and radial norm
- standardizes over tokens

This gives a geometry-based token importance prior.

#### Step E: mix reliability with uniform marginals

```python
query_attn = torch.softmax(query_logits / temperature, dim=-1)
support_attn = torch.softmax(support_logits / temperature, dim=-1)

query_uniform = ...
support_uniform = ...

query_weights = (1.0 - reliability_mix) * query_uniform + reliability_mix * query_attn
support_weights = (1.0 - reliability_mix) * support_uniform + reliability_mix * support_attn

query_mass = query_weights * flat_rho.unsqueeze(-1)
support_mass = support_weights * flat_rho.unsqueeze(-1)
```

This is critical:
- total transport budget per pair is still `rho`
- the model only redistributes that budget across tokens

### 8.7 Structure-consistency cost: the `R`-only addition

This is the single feature that differentiates `R` from `Q`.

Code:

```python
if self.uses_structure_consistent_transport:
    structure_cost, structure_payload = self._compute_structure_consistency_cost(
        flat_cost,
        query_euclidean,
        flat_support_euclidean,
        flat_rho,
        query_token_mass,
        support_token_mass,
    )
    final_cost = flat_cost + structure_cost
```

What `_compute_structure_consistency_cost` does:

1. Run a detached probe transport using the learned token marginals:

```python
probe_plan, _, _ = self._transport_match(
    flat_cost.detach(),
    flat_rho.detach(),
    a=query_mass.detach(),
    b=support_mass.detach(),
)
```

2. Build normalized intra-query and intra-support pairwise distance matrices:

```python
query_structure = self._normalize_structure_distance(self._pairwise_token_distance(query_tokens_euc))
support_structure = self._normalize_structure_distance(self._pairwise_token_distance(flat_support_tokens_euc))
```

3. Convert the detached plan into row/column mass:
- `row_mass = probe_plan.sum(dim=-1)`
- `col_mass = probe_plan.sum(dim=-2)`

4. Compute a posterior-derived structure mismatch tensor:

```python
query_term = torch.einsum("qti,qpi->qpt", query_structure.pow(2), row_mass)
support_term = torch.einsum("pkj,qpj->qpk", support_structure.pow(2), col_mass)
cross_term = torch.einsum("qti,qpij,pkj->qptk", query_structure, probe_plan, support_structure)
structure_cost = (query_term.unsqueeze(-1) + support_term.unsqueeze(-2) - 2.0 * cross_term).clamp_min(0.0)
structure_cost = structure_cost / structure_cost.mean(dim=(-1, -2), keepdim=True).clamp_min(self.eps)
structure_cost = self.structure_cost_weight * structure_cost
```

Interpretation:
- matching two tokens should be cheap not only when their features are similar
- it should also be cheap when the token-token match is consistent with each token's internal relational geometry
- the detached probe plan acts like a soft correspondence prior

So the final cost becomes:

```text
C_final = C_base + lambda_struct * C_structure
```

where `lambda_struct` is `structure_cost_weight`.

### 8.8 Noise sink

Before the final transport solve, the model appends a learnable sink:

```python
sink_mass = (1.0 - flat_rho).clamp_min(self.eps)
query_mass_with_sink = torch.cat([query_mass, sink_mass.unsqueeze(-1)], dim=-1)
support_mass_with_sink = torch.cat([support_mass, sink_mass.unsqueeze(-1)], dim=-1)
```

And expands the cost matrix with one extra row and column:

```python
cost_with_sink[..., :-1, :-1] = flat_cost_or_final_cost
cost_with_sink[..., -1, -1] = 0.0
```

Meaning:
- if the pair budget is less than 1, leftover evidence can go to sink
- unmatched noisy evidence is not forced into bad token-token matches

### 8.9 Final transport solve

Final solve:

```python
flat_plan_with_sink, _, _ = self._transport_match(
    cost_with_sink,
    flat_rho,
    a=query_mass_with_sink,
    b=support_mass_with_sink,
)
flat_plan = flat_plan_with_sink[..., :-1, :-1]
flat_transport_cost = compute_transport_cost(flat_plan, final_cost)
flat_transport_mass = compute_transported_mass(flat_plan)
```

The model discards the sink row and column when computing the scored transport cost and transported mass.

### 8.10 Adaptive threshold and shot scoring

Q/R uses an adaptive threshold derived from the Q geodesic features:

```python
adaptive_threshold = self._compute_q_adaptive_threshold(q_geodesic_features, shot_transport_cost)
shot_logits = self.score_scale * (adaptive_threshold * shot_transport_mass - shot_transport_cost)
```

This matters a lot:
- shot score is not just `-cost`
- it is "reward transported mass, but only if mass is obtained below a learned threshold"

### 8.11 Shot pooling

Shots are pooled with learned weights:

```python
q_logits, q_transport_cost, q_transport_mass, shot_pool_weights = self._pool_shot_scores(
    shot_logits,
    shot_transport_cost,
    shot_transport_mass,
    geodesic_features=q_geodesic_features,
)
```

If `shot_num > 1`, `_pool_shot_scores`:
- standardizes shot-level features over shots
- scores each shot with `q_shot_pool_scorer`
- turns that evidence into attention weights
- mixes attentive weights with a uniform distribution using `shot_pool_mix`

### 8.12 Final blend with H-anchor

The last step blends the stable H-anchor path with the Q/R-enhanced path:

```python
q_mix = self.q_enhancement_mix
logits = (1.0 - q_mix) * h_anchor_logits + q_mix * q_logits
transport_cost = (1.0 - q_mix) * h_anchor_transport_cost + q_mix * q_transport_cost
transport_mass = (1.0 - q_mix) * h_anchor_transport_mass + q_mix * q_transport_mass
```

This is another stabilizer:
- early in training the model can stay close to H
- later it can shift toward the full Q/R branch

---

## 9. Transport Solver Details

`_transport_match` builds the row/column marginals and calls one of:
- `sinkhorn_unbalanced_log`
- `sinkhorn_balanced_log`
- or POT-based equivalents if `hrot_ot_backend="pot"`

For `R`, transport is unbalanced by default, so the important solver is:

```python
pair_plan = sinkhorn_unbalanced_log(
    pair_cost,
    pair_a,
    pair_b,
    tau_q=self.tau_q,
    tau_c=self.tau_c,
    eps=self.sinkhorn_epsilon,
    max_iter=self.sinkhorn_iterations,
    tol=self.sinkhorn_tolerance,
)
```

Notes:
- `tau_q` and `tau_c` are KL relaxation strengths for row and column marginals.
- `eps` is the entropic regularization strength.
- `compute_transport_cost(plan, cost)` is simply `(plan * cost).sum(...)`.
- `compute_transported_mass(plan)` is simply `plan.sum(...)`.

---

## 10. Loss Used During Training

Training loss in `main.py` is:

```python
loss, cls_loss, aux_loss = compute_loss_breakdown(...)
```

For `HROT-R`, this resolves to:

```text
total_loss = cross_entropy(logits, targets) + aux_loss
```

where `aux_loss` from `HROTFSL._forward_episode` is:

```python
aux_loss = (
    self.lambda_rho * rho_regularization
    + self.lambda_rho_rank * rho_rank_loss
    + self.lambda_curvature * curvature_regularization
)
```

For variant `R`, the default active term is usually:
- `rho_regularization = (rho - rho_target)^2.mean()`

By default:
- `uses_rho_rank_loss` is false for `R`, so `rho_rank_loss` contributes zero
- `lambda_curvature` is usually zero unless manually enabled

Important:
- the structure-consistency penalty is not added as a separate auxiliary term
- it changes the transport ground cost directly

---

## 11. Optimization And Scheduler

From `train_loop`:

```python
optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = build_scheduler(optimizer, args)
...
loss.backward()
if args.grad_clip > 0:
    torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
optimizer.step()
...
scheduler.step()
```

Default scheduler path:
- linear warmup for `warmup_epochs`
- then cosine annealing to `min_lr`

So the standard benchmark optimizer stack is:
- optimizer: `AdamW`
- scheduler: `LinearLR -> CosineAnnealingLR`

---

## 12. What Changes Between Train, Val, And Test

For `HROT-R`:

### Train
- `net.train()`
- `forward_scores(... collect_diagnostics=True)`
- full payload returned
- gradients flow through:
  - backbone
  - token projector
  - curvature
  - `eam`
  - `q_eam`
  - token reliability parameters
  - adaptive threshold head
  - shot pooling head
  - structure-cost weight

### Val
- `net.eval()`
- still calls the same forward path
- no gradients
- diagnostics collected
- optional float64 hyperbolic calculations if enabled

### Final test
- `net.eval()`
- same model forward
- different episode loader if robust protocol is requested
- no extra HROT-specific test-time branch

This is one of the cleanest things about the implementation:
- no `if phase == "test"` inside `HROT-R` itself

---

## 13. Diagnostics Payload Returned By `HROT-R`

When `return_aux=True`, `HROTFSL.forward` returns a large dictionary. The most important keys for variant `R` are:

Core:
- `logits`
- `aux_loss`
- `class_scores`
- `total_distance`
- `transport_cost`
- `transported_mass`
- `rho`
- `rho_regularization`
- `rho_rank_loss`
- `curvature_regularization`
- `curvature`
- `mass_bonus`
- `transport_cost_threshold`

Shot-level:
- `shot_transport_cost`
- `shot_transported_mass`
- `shot_rho`

Probe / reliability:
- `transport_probe_cost`
- `transport_probe_mass`
- `transport_probe_entropy`
- `transport_probe_min_cost`
- `query_token_mass`
- `support_token_mass`
- `probe_query_reliability`
- `probe_support_reliability`
- `support_consensus`
- `query_hyperbolic_token_prior`
- `support_hyperbolic_token_prior`
- `adaptive_transport_cost_threshold`
- `shot_logits`
- `shot_pool_weights`
- `q_enhanced_logits`
- `h_anchor_logits`
- `h_anchor_shot_logits`
- `h_anchor_shot_transport_cost`
- `h_anchor_shot_transported_mass`
- `h_anchor_rho`
- `q_eam_cross_attention`

Noise sink:
- `noise_sink_query_mass`
- `noise_sink_support_mass`
- `noise_sink_self_mass`
- `noise_sink_cost`

R-only structure keys:
- `base_cost_matrix`
- `structure_cost`
- `structure_probe_mass`
- `structure_cost_weight`

Raw token outputs for analysis:
- `transport_plan`
- `cost_matrix`
- `query_euclidean_tokens`
- `support_euclidean_tokens`
- `query_hyperbolic_tokens`
- `support_hyperbolic_tokens`

These fields are what make the model especially analyzable by another LLM.

---

## 14. Behavioral Contract From Tests

The best "ground truth" summary of variant `R` is in the tests.

### 14.1 Structure cost really changes the final cost

`tests/test_hrot_fsl.py` checks:

```python
expected_cost = outputs["base_cost_matrix"] + outputs["structure_cost"]
assert torch.allclose(outputs["cost_matrix"], expected_cost, atol=1e-5, rtol=1e-5)
```

This proves:
- `R` is exactly "Q plus structure cost" at the final cost-matrix level

### 14.2 The scored shot logits use thresholded mass-vs-cost

The test also checks:

```python
expected_shot_logits = model.score_scale * (
    outputs["adaptive_transport_cost_threshold"] * outputs["shot_transported_mass"]
    - outputs["shot_transport_cost"]
)
assert torch.allclose(outputs["shot_logits"], expected_shot_logits, atol=1e-5, rtol=1e-5)
```

This is the exact scoring rule.

### 14.3 The structure weight is learnable

`test_hrot_fsl_variant_r_backpropagates_structure_weight` verifies:
- `model.raw_structure_cost_weight.grad is not None`
- gradient is finite and non-zero

So `structure_cost_weight` is not just a static hyperparameter after init; it is optimized.

---

## 15. Core Code Excerpts

### 15.1 Variant-R-only switch

```python
self.uses_noise_calibrated_transport = variant in {"Q", "R"}
self.uses_structure_consistent_transport = variant == "R"
```

### 15.2 Noise-calibrated token marginals

```python
def _compute_noise_calibrated_token_marginals(
    self,
    flat_cost,
    flat_rho,
    support_tokens_euc,
    query_tokens_hyp,
    flat_support_tokens_hyp,
):
    query_features, support_features, payload = self._build_probe_token_features(flat_cost, flat_rho)
    query_logits = torch.einsum("qstf,f->qst", query_features, self.query_reliability_weights)
    support_logits = torch.einsum("qstf,f->qst", support_features, self.support_reliability_weights)

    support_consensus = self._build_support_consensus_scores(support_tokens_euc)
    flat_consensus = support_consensus.reshape(way_num * shot_num, support_consensus.shape[-1])
    flat_consensus = flat_consensus.unsqueeze(0).expand(flat_cost.shape[0], -1, -1)
    support_logits = support_logits + consensus_mix * flat_consensus

    query_prior_logits, support_prior_logits = self._compute_hyperbolic_token_prior_logits(
        query_tokens_hyp,
        flat_support_tokens_hyp,
    )
    query_logits = query_logits + prior_mix * query_prior_logits
    support_logits = support_logits + prior_mix * support_prior_logits

    query_attn = torch.softmax(query_logits / temperature, dim=-1)
    support_attn = torch.softmax(support_logits / temperature, dim=-1)
    query_weights = (1.0 - reliability_mix) * query_uniform + reliability_mix * query_attn
    support_weights = (1.0 - reliability_mix) * support_uniform + reliability_mix * support_attn

    query_mass = query_weights * flat_rho.unsqueeze(-1)
    support_mass = support_weights * flat_rho.unsqueeze(-1)
    return query_mass, support_mass, payload
```

### 15.3 Structure-consistency cost

```python
def _compute_structure_consistency_cost(
    self,
    flat_cost,
    query_tokens_euc,
    flat_support_tokens_euc,
    flat_rho,
    query_mass,
    support_mass,
):
    with torch.no_grad():
        probe_plan, _, _ = self._transport_match(
            flat_cost.detach(),
            flat_rho.detach(),
            a=query_mass.detach(),
            b=support_mass.detach(),
        )
        row_mass = probe_plan.sum(dim=-1)
        col_mass = probe_plan.sum(dim=-2)

    query_structure = self._normalize_structure_distance(self._pairwise_token_distance(query_tokens_euc))
    support_structure = self._normalize_structure_distance(
        self._pairwise_token_distance(flat_support_tokens_euc)
    )

    query_term = torch.einsum("qti,qpi->qpt", query_structure.pow(2), row_mass)
    support_term = torch.einsum("pkj,qpj->qpk", support_structure.pow(2), col_mass)
    cross_term = torch.einsum("qti,qpij,pkj->qptk", query_structure, probe_plan, support_structure)
    structure_cost = (query_term.unsqueeze(-1) + support_term.unsqueeze(-2) - 2.0 * cross_term).clamp_min(0.0)
    structure_cost = structure_cost / structure_cost.mean(dim=(-1, -2), keepdim=True).clamp_min(self.eps)
    structure_cost = self.structure_cost_weight * structure_cost
    return structure_cost
```

### 15.4 Noise sink

```python
def _append_noise_sink(self, flat_cost, query_mass, support_mass, flat_rho):
    sink_cost = self.noise_sink_cost
    cost_with_sink = sink_cost.expand(
        flat_cost.shape[0],
        flat_cost.shape[1],
        flat_cost.shape[2] + 1,
        flat_cost.shape[3] + 1,
    ).clone()
    cost_with_sink[..., :-1, :-1] = flat_cost
    cost_with_sink[..., -1, -1] = 0.0

    sink_mass = (1.0 - flat_rho).clamp_min(self.eps)
    query_mass_with_sink = torch.cat([query_mass, sink_mass.unsqueeze(-1)], dim=-1)
    support_mass_with_sink = torch.cat([support_mass, sink_mass.unsqueeze(-1)], dim=-1)
    return cost_with_sink, query_mass_with_sink, support_mass_with_sink
```

### 15.5 Shot scoring and pooling

```python
adaptive_threshold = self._compute_q_adaptive_threshold(q_geodesic_features, shot_transport_cost)
shot_logits = self.score_scale * (adaptive_threshold * shot_transport_mass - shot_transport_cost)

q_logits, q_transport_cost, q_transport_mass, shot_pool_weights = self._pool_shot_scores(
    shot_logits,
    shot_transport_cost,
    shot_transport_mass,
    geodesic_features=q_geodesic_features,
)
```

### 15.6 Final blend

```python
logits = (1.0 - q_mix) * h_anchor_logits + q_mix * q_logits
transport_cost = (1.0 - q_mix) * h_anchor_transport_cost + q_mix * q_transport_cost
transport_mass = (1.0 - q_mix) * h_anchor_transport_mass + q_mix * q_transport_mass
```

---

## 16. Minimal Pseudocode For Another LLM

```text
for each episode:
    encode query/support images into Euclidean and hyperbolic tokens
    build Euclidean token cost for each query-shot pair

    build geodesic shot features
    rho_H = eam(geodesic_features)
    rho_QR = q_eam(geodesic_features + cross_attention)

    H-anchor logits = score(UOT(cost, rho_H))

    probe token reliability with detached UOT(cost, rho_QR)
    build token marginals from:
        probe reliability
        support consensus
        hyperbolic token prior

    if variant == R:
        structure_cost = posterior_derived_structure_cost(...)
        final_cost = base_cost + structure_cost
    else:
        final_cost = base_cost

    append noise sink using sink mass = 1 - rho_QR
    final_plan = UOT(final_cost_with_sink, rho_QR, custom_marginals)

    compute per-shot cost and transported mass from real token block
    compute adaptive threshold
    compute shot logits = score_scale * (threshold * mass - cost)
    pool shots

    final logits = blend(H-anchor logits, Q/R logits)
    aux_loss = lambda_rho * rho_reg + optional other regs
    total loss = CE(final logits, targets) + aux_loss
```

---

## 17. What Another LLM Should Focus On When Reviewing This Code

If the goal is analysis rather than reimplementation, the highest-value review targets are:

1. Whether the detached probe plan creates a train/test mismatch in how structure cost is formed.
2. Whether `structure_cost` is too strongly normalized by its own mean and therefore loses absolute calibration.
3. Whether mixing H-anchor and Q/R logits hides failure modes in the Q/R path.
4. Whether the same `rho` should govern both real-token marginals and sink mass as `1 - rho`.
5. Whether support consensus should be class-conditioned only, or query-conditioned as well.
6. Whether the Euclidean base cost plus hyperbolic control path is internally consistent, or conceptually split.
7. Whether the adaptive threshold head is learning calibration or simply another free rescaling path.
8. Whether the gradients into `raw_structure_cost_weight` are strong enough in real training, not just in unit tests.

---

## 18. Short Summary

`HROT-R` in this repo is:
- a shot-decomposed, unbalanced OT few-shot classifier
- with Euclidean token cost
- hyperbolic geometry for budget prediction and token priors
- Q-style noise-calibrated token marginals
- an explicit noise sink
- learned shot pooling
- and one extra `R`-only posterior-derived structure-consistency cost added to the final transport ground cost

If you only remember one exact sentence, use this:

`HROT-R = H-anchor path + Q-style noise-calibrated UOT path + structure cost from a detached posterior probe plan`.
