You are implementing a new few-shot learning model inside an existing episodic training codebase. Your task is to add two new models:

- SPIFCE: fair version, using only episodic cross-entropy
- SPIFMAX: stronger version, same core architecture plus lightweight regularization losses

The implementation must integrate into the existing repository using the SAME episodic training / evaluation pipeline as current baseline models. Do not modify dataset loading, episode sampler logic, trainer flow, logging conventions, or evaluation protocol unless strictly necessary for compatibility.

====================================================
1. DESIGN GOAL
====================================================

The model must be architecturally biased toward:

- strong 1-shot performance
- strong behavior under very small data
- low variance
- low overfitting by architectural design, not by heavy tricks
- remaining meaningful when shot = 1

This is critical.

Avoid the common failure mode where the model becomes too adaptive, too parameter-heavy, too query-conditioned, or too elaborate, and ends up losing to simple baselines like MatchingNet.

The model should solve this general few-shot problem:

"same-class samples often share only a subset of stable evidence, while a large portion of observed features is variant or nuisance. Therefore, the model should learn stable-vs-variant token decomposition and classify primarily using stable evidence, with a low-variance few-shot head combining global prototype similarity and lightweight local partial matching."

This is NOT:

- a heavy attention model
- a support-query cross-attention model
- a dynamic basis model
- a reconstruction model
- a transductive model
- an episode-level transformer model

====================================================
2. HIGH-LEVEL MODEL NAME
====================================================

Model family name:
SPIF = Stable Partial Invariance Few-shot Network

Implement two classes:
- SPIFCE
- SPIFMAX

====================================================
3. CORE PRINCIPLES
====================================================

The architecture must follow these principles:

A. 1-shot first
- The model must still make full sense when K=1
- No core module may rely on K>1 to be useful
- No shot-routing mechanism as the main idea
- No cross-shot adaptive design that degenerates at 1-shot

B. Invariance from feature geometry, not heavy adaptivity
- Prefer stable/variant decomposition
- Prefer stable evidence classification
- Prefer partial but lightweight matching
- Avoid deeply query-conditioned modules

C. Low-variance few-shot head
- Must remain mostly non-parametric
- No heavy learned matching heads
- No transductive batch refinement
- No support-query transformer layers

D. Preserve local evidence without dense greedy matching
- Keep local token evidence
- But do not build a large dense pairwise reasoning module

====================================================
4. BACKBONE
====================================================

Use the repository’s existing backbone conventions if possible.

Recommended default backbone:
- ResNet12 if already available
- fallback to Conv4 or ResNet18 if needed

Input image -> backbone -> feature map:
F: [B, C, H, W]

Then flatten spatially into token sequence:
X: [B, L, C]
where L = H * W

Do not add large extra trunk modules.

====================================================
5. STABLE-VARIANT TOKEN FACTORIZATION
====================================================

From token map X, compute two token branches:

- stable tokens Xs
- variant tokens Xv

Implementation requirements:
- shared backbone trunk
- two lightweight projection heads only
- each projection head should be very small

Recommended projection head structure:
- LayerNorm
- Linear
- GELU
- Linear

Example:
input X: [B, L, C]

output:
- Xs: [B, L, Ds]
- Xv: [B, L, Dv]

Recommended defaults:
- Ds = Dv = C
or
- Ds = Dv = C // 2

Keep this block small and stable.
Do NOT add deep MLP stacks.

====================================================
6. STABLE EVIDENCE GATE
====================================================

Add a lightweight per-token scalar gate computed only from stable tokens Xs.

Gate network:
- LayerNorm
- Linear(Ds -> hidden)
- GELU
- Linear(hidden -> 1)
- Sigmoid

Output:
g: [B, L, 1]

Apply:
Xs_hat = g * Xs

Important constraints:
- gate is per-sample only
- no support-query interaction
- no class-conditioned gate
- no cross-attention
- no query-conditioned adaptation

The gate should softly suppress nuisance tokens and preserve stable evidence tokens.

Recommended hidden:
- hidden = Ds // 4 if Ds is large
- otherwise hidden = Ds

====================================================
7. GLOBAL STABLE EMBEDDING
====================================================

From Xs_hat, compute weighted pooled stable embedding:

z = sum(Xs_hat over tokens) / (sum(g over tokens) + eps)

Implementation:
- output z: [B, Ds]

Then L2-normalize z before prototype similarity computation.

Also keep token-level normalized stable tokens for local matching:
T = LayerNorm(Xs_hat)

Optionally also L2-normalize T along the feature dimension.

This is important:
- use both a global embedding path and a local token path
- global path gives low variance
- local path preserves subtle evidence

====================================================
8. EPISODIC TENSOR ORGANIZATION
====================================================

Within an episode:
- encode support and query through the same network

Need these outputs:
- support global embeddings
- support stable token sets
- query global embeddings
- query stable token sets

Assume standard episode format:
- way = N
- shot = K
- query_per_class = Q

Shapes after reshaping:

support_global: [N, K, Ds]
support_tokens: [N, K, L, Ds]

query_global: [N*Q, Ds]
query_tokens: [N*Q, L, Ds]

====================================================
9. GLOBAL PROTOTYPE BRANCH
====================================================

Compute stable class prototypes from support global embeddings:

p_c = mean over shot dimension of support_global[c]

shape:
prototypes: [N, Ds]

Then compute cosine similarity between each query global embedding and each class prototype:

s_global[q, c] = cosine(query_global[q], prototypes[c])

This branch is critical for 1-shot stability.

Do NOT replace it with:
- a learned classifier head
- an MLP classifier
- a relation network head

====================================================
10. LOCAL PARTIAL MATCHING BRANCH
====================================================

This branch provides fine-grained matching on stable evidence, but must remain lightweight and mostly non-parametric.

For each class c:
- concatenate all support stable token sets across shots into one token pool

support_pool[c]:
shape [K * L, Ds]

For each query token set Tq: [L, Ds]
compute token-to-token cosine similarity matrix with support_pool[c]:

sim[q, c]:
shape [L, K*L]

Then for each query token:
- take top-r similarities over the support-token dimension
- average those top-r values

Then average over all query tokens:

s_local[q, c] = mean over query tokens of TopRMean(similarity row)

Recommended default:
- r = 3
or
- r = 5

Important:
- this must be exact lightweight top-r partial matching
- do NOT replace with full OT in SPIFCE
- do NOT add cross-attention
- do NOT add query-adaptive support routing
- do NOT learn a large matching module

The local branch must stay lightweight and stable.

====================================================
11. FINAL SCORE FUSION
====================================================

Fuse global and local scores:

s[q, c] = alpha * s_global[q, c] + (1 - alpha) * s_local[q, c]

For SPIFCE:
- alpha is fixed constant = 0.7

For SPIFMAX:
- alpha is a single learnable scalar parameter
- parameterize alpha via sigmoid(alpha_logit)
- initialize so alpha starts near 0.7

Important:
- no per-class fusion
- no per-query fusion MLP
- no support-conditioned fusion network

The fused score matrix is the final logits for episodic CE.

====================================================
12. LOSSES
====================================================

SPIFCE:
- use ONLY episodic cross-entropy on fused logits

No auxiliary loss.
No consistency loss.
No reconstruction loss.
No contrastive loss.
No prototype regularizer.
No entropy regularizer.

This is the fair version.

SPIFMAX:
Use the same core architecture, plus these lightweight losses:

1. L_ce
- episodic cross-entropy on fused logits

2. L_cons
- stable embedding consistency between two augmented views of the same sample
- use only on stable global embedding z
- simple L2 or cosine consistency is enough

3. L_decorr
- decorrelation penalty between stable and variant global embeddings
- keep it simple and lightweight
- for example, covariance penalty or cosine decorrelation

4. L_sparse
- average gate activation penalty
- encourages compact evidence usage

Total loss:
L = L_ce + lambda1 * L_cons + lambda2 * L_decorr + lambda3 * L_sparse

Recommended defaults:
- lambda1 = 0.1
- lambda2 = 0.01
- lambda3 = 0.001

Keep all these configurable.

Important:
- do not add extra losses beyond these unless behind an explicit flag and disabled by default
- do not add supervised contrastive by default

====================================================
13. ARCHITECTURAL ANTI-OVERFITTING CONSTRAINTS
====================================================

These constraints are mandatory:

1. Keep all new learnable modules shallow and lightweight
2. No support-query cross-attention anywhere
3. No dynamic basis generation
4. No transductive inference
5. No episode-level transformer
6. No multi-stage routing mechanism
7. No shot-selection module that degenerates when K=1
8. No reconstruction classifier
9. No learned class-specific classifier weights outside prototype computation
10. Preserve a strong low-variance path from support prototype to query classification

The whole point is to improve 1-shot by disciplined architectural bias, not by aggressive adaptation.

====================================================
14. CODE ORGANIZATION
====================================================

Implement in a new file if appropriate, for example:
- models/spif.py

Recommended classes:
- class SPIFEncoder(nn.Module)
- class SPIFCE(nn.Module)
- class SPIFMAX(nn.Module)

Or adapt to the repository’s naming / registry conventions if needed.

Suggested internal methods:
- encode_tokens(x)
- factorize_tokens(tokens)
- compute_gate(stable_tokens)
- pool_global(stable_tokens, gate)
- build_support_prototypes(support_global)
- build_support_token_pool(support_tokens)
- compute_global_scores(query_global, prototypes)
- compute_local_partial_scores(query_tokens, support_token_pool)
- forward_episode(support_x, support_y, query_x)

The code must be:
- readable
- dimension-safe
- heavily commented
- easy to ablate

====================================================
15. REQUIRED OUTPUTS
====================================================

The episodic forward should return at least:
- fused logits

Also return a diagnostic dictionary when possible containing:
- global_scores
- local_scores
- alpha
- mean_gate
- stable_global_embeddings
- variant_global_embeddings

This helps with debugging and ablation.

====================================================
16. ABLATION-FRIENDLY FLAGS
====================================================

Make these ablations easy to run:
- global_only
- local_only
- gate_on / gate_off
- factorization_on / factorization_off
- fixed_alpha / learnable_alpha
- top_r choice
- Ds / Dv choice
- SPIFCE / SPIFMAX

Do not hardcode the model so these ablations become painful.

====================================================
17. WHAT NOT TO DO
====================================================

Do NOT silently improve the model by adding:
- Transformers
- Mamba
- cross-attention
- reconstruction losses
- optimal transport solvers
- relation networks
- deep metric heads
- query-conditioned support weighting
- extra classification heads
- transductive query-batch refinement

If you believe one of these may help, leave a comment in code, but do not implement it in the main model.

====================================================
18. ENGINEERING EXPECTATIONS
====================================================

- Reuse existing backbone code when possible
- Match existing tensor conventions
- Match existing training-loop signatures
- Add model registry hooks if the repo uses a registry
- Ensure stable operation for both shot=1 and shot>1
- Avoid unnecessary memory-heavy pairwise ops
- Use vectorized similarity computation efficiently
- Keep the implementation faithful to the intended architectural bias

====================================================
19. FINAL INTENT
====================================================

This model is deliberately less greedy than flashy few-shot architectures.

Its intended advantages are:
- strong 1-shot behavior
- stable low-sample training
- clear invariant-learning story
- easy ablation
- one fair CE-only version
- one stronger regularized version

Implement it faithfully. Do not overcomplicate it.