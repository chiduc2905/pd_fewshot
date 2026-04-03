You are modifying an existing few-shot learning repository.

Goal:
Implement a NEW model variant called `SPIFAEB` (Stable Partial Invariance with Adaptive Evidence Budget) as a separate file/module derived from the existing SPIFCE design, WITHOUT modifying the original SPIFCE behavior or code path.

Hard constraints:
1. Do NOT change the logic of the original SPIFCE class or existing SPIFCE experiments.
2. Create a NEW implementation file for the new model, e.g. `net/spif_aeb.py` (or a similarly clean new file).
3. Register the new model into the model factory / few-shot training pipeline so it can be trained exactly like other few-shot models.
4. Reuse as much existing SPIFCE encoder / episode pipeline code as possible if clean, but isolate all new logic inside the new model implementation.
5. Keep the training pipeline, episodic dataloading, CLI integration, logging, checkpointing, and evaluation flow consistent with existing few-shot models.
6. The new model must be selectable by a new model name such as `"spifaeb"` from the same factory / CLI path used by other few-shot models.
7. Do not break backward compatibility.

Research design to implement:
We want to replace the fixed local top-r partial matching branch in SPIFCE with a stability-aware adaptive evidence budget.

Reference behavior from SPIFCE:
- Keep the same backbone usage and tokenization flow.
- Keep stable / variant factorization.
- Keep stable gate.
- Keep the global prototype branch.
- Keep CE-only compatibility like SPIFCE fair variant unless config later enables extras.
- Preserve return_aux / diagnostics patterns if present.

Mathematical design:
Given:
- query stable tokens: T_q^s with shape [num_query, L, D]
- support stable tokens: T_s^s with episodic shape [way, shot, L, D]
- support stable global embeddings and query stable global embeddings as in SPIFCE

Build class token pool:
- For each class c, concatenate support stable tokens over shots:
  class_token_pool[c] -> shape [shot * L, D]

Similarity:
- For each query q and class c, compute
  A[q, c] = T_q^s[q] @ class_token_pool[c].T
  shape [L, M], where M = shot * L
- This is cosine-style similarity if the upstream token normalization in SPIFCE already ensures normalized tokens. Preserve the same normalization convention as SPIFCE local matching.

Adaptive evidence budget:
- Predict rho[q, c] in (0, 1) from stable global geometry only.
- Construct:
  u[q, c] = concat(
      zq_s,
      pc_s,
      abs(zq_s - pc_s),
      zq_s * pc_s
    )
- Use a lightweight MLP:
    Linear(4D -> hidden)
    GELU
    Linear(hidden -> 1)
    Sigmoid
- hidden dimension should be configurable, with a small default such as max(16, D//2) or similar.

Adaptive threshold:
For each row l in A[q, c]:
- mean mu[q, c, l] over M support tokens
- std  sigma[q, c, l] over M support tokens
- threshold:
    tau[q, c, l] = mu[q, c, l] + beta * (1 - rho[q, c]) * sigma[q, c, l]
- beta is a configurable scalar hyperparameter, default around 1.0

Sparse evidence weights:
- pre_weight = relu(A - tau)
- normalize over support-token dimension:
    w = pre_weight / (pre_weight.sum(dim=-1, keepdim=True) + eps)

Local score:
- local_score[q, c] = mean over query tokens l of sum_t w[q, c, l, t] * A[q, c, l, t]

Global score:
- same as SPIFCE:
    cosine between query stable global embedding and class stable prototype

Final logits:
- logits = alpha * global_score + (1 - alpha) * local_score
- keep alpha fixed by default, same spirit as SPIFCE fair version
- make alpha configurable, but default to the SPIFCE-like value if available

Implementation requirements:
1. Read the current SPIFCE code carefully and preserve:
   - episodic tensor shape conventions
   - normalization conventions
   - auxiliary returns
   - forward signatures
   - loss computation style
2. Reuse existing encoder class if possible, or subclass the SPIF base class cleanly.
3. Add a dedicated class, e.g. `SPIFAEB`, with a clean constructor and defaults.
4. Add any needed helper functions:
   - class token pool builder
   - budget predictor
   - adaptive local score computation
5. Add diagnostics in aux outputs when return_aux=True:
   - rho / budget values
   - active match counts per query-class if easy
   - local/global scores
6. Keep numerics stable:
   - epsilon in std and normalization
   - avoid NaNs when all pre_weight are zero
   - if all pre_weight are zero for a row, fall back safely:
       either use a tiny epsilon floor
       or revert to a normalized positive transform only for that row
   Choose the cleanest stable implementation and comment it.
7. Keep code style consistent with repo conventions.

Config / factory integration:
1. Register `"spifaeb"` in `net/model_factory.py` (or equivalent factory file).
2. Add constructor arguments / config plumbing from CLI if the repo already exposes SPIF arguments:
   - aeb_hidden
   - aeb_beta
   - maybe aeb_eps
3. If top_r is currently part of shared SPIF args, do not require it for SPIFAEB local matching, but keep compatibility if parser passes it.
4. Ensure `main.py` / training entry can instantiate and train `spifaeb` without changing existing experiment commands for other models.

Do NOT:
- edit SPIFCE logic for existing models
- refactor large unrelated parts of the repo
- introduce heavy external dependencies
- add cross-attention / transformer relation modules
- add OT solvers
- add unnecessary auxiliary losses in the first implementation

Expected deliverables:
1. New model file implementing `SPIFAEB`
2. Factory registration and minimal config wiring
3. Short inline comments explaining the adaptive evidence budget math
4. A concise summary of files changed and why
5. A note on any numerical fallback used when sparse weights become all zero

Validation checklist:
- Existing SPIFCE still works unchanged
- `spifaeb` trains through the normal few-shot pipeline
- Forward pass shapes are correct for 1-shot and 5-shot
- No NaN in local score when rows are fully filtered
- return_aux contains interpretable budget statistics