You are an AI research engineer working inside an existing few-shot learning codebase.

Your task is to implement a NEW model variant in a NEW FILE ONLY.

CRITICAL FILE-SAFETY REQUIREMENT:
- Do NOT modify the original SPIFCE / SPIFMAX implementation files.
- Do NOT overwrite, refactor, rename, or patch the original SPIF files.
- Do NOT change existing baseline behavior.
- You must create a separate new model file for the new architecture.
- The original SPIFCE and SPIFMAX must remain fully intact and usable exactly as before.

Your job is to implement a new model family based on the existing SPIF design, but with a meaningful Mamba contribution:
- keep the original SPIFCE and SPIFMAX as baselines
- create a NEW model file for the Mamba-enhanced variant
- integrate it cleanly into the repository registry / training pipeline
- preserve compatibility with the existing episodic training code

====================================================
0. MAIN OBJECTIVE
====================================================

This new architecture must be designed intelligently, not greedily.

The purpose is NOT to “add Mamba because it is trendy”.
The purpose IS to solve a specific weakness of the original SPIF:

Original SPIF already learns stable-vs-variant features and uses a token gate to select stable evidence.
However, the original gate operates on relatively static token features.
A token may truly belong to class-stable evidence, but:
- it may be weak on its own
- it may only become meaningful when contextualized with neighboring / related tokens
- stable evidence may be distributed across multiple tokens
- nuisance tokens may look strong locally but should be suppressed after contextual propagation

Therefore, before stable evidence gating, we want a lightweight mechanism that propagates stable evidence across tokens in a selective, low-variance way.

This is the exact role of the new Mamba block.

The model should preserve the strengths of SPIF:
- strong 1-shot behavior
- low variance
- low overfitting by architectural bias
- no heavy support-query interaction
- no greedy dense adaptation

====================================================
1. MODEL FAMILY TO IMPLEMENT
====================================================

Create a NEW model family, for example with names like:

- SPIFMambaCE
- SPIFMambaMAX

You may adjust the exact class names to fit repository naming conventions, but they must clearly indicate:
- this is a new SPIF-based Mamba variant
- it is distinct from SPIFCE / SPIFMAX

Create a NEW FILE, for example:
- models/spif_mamba.py

Do not modify the original SPIF file.

====================================================
2. HIGH-LEVEL ARCHITECTURAL IDEA
====================================================

The architecture is:

1. Backbone trunk -> token map
2. Stable-variant factorization
3. Mamba-based stable evidence propagation on the stable token stream only
4. Stable evidence gate
5. Global stable embedding branch
6. Local partial invariant matching branch
7. Fused few-shot logits

Core conceptual claim:
- SPIF chooses stable evidence
- SPIF-Mamba first propagates stable evidence, then chooses it

This is the key new contribution.

The Mamba block must be treated as a real architectural contribution, not a decorative add-on.

====================================================
3. WHAT EXACTLY THE NEW MAMBA BLOCK SHOULD DO
====================================================

The Mamba block is inserted ONLY on the stable token branch, BEFORE the gate.

Original SPIF style:
X -> Xs, Xv -> gate(Xs) -> gated stable tokens

New SPIF-Mamba:
X -> Xs, Xv
Xs -> Mamba-based stable evidence propagator -> Hs
gate(Hs) -> gated stable evidence states

Interpretation:
- Xs = stable token candidates
- Hs = propagated stable evidence states

The Mamba block is there because the gate alone cannot create contextual evidence.
The gate only scores the current token representation.
The Mamba block should help:
- propagate weak but meaningful stable evidence
- reinforce distributed evidence patterns
- suppress nuisance through selective propagation
- provide contextualized stable evidence to the gate

====================================================
4. IMPORTANT DESIGN PRINCIPLES
====================================================

The design MUST respect all of the following:

A. 1-shot first
- The architecture must remain meaningful and strong when shot = 1
- Do not introduce any core module that only helps when shot > 1
- Do not add shot routing as the core idea

B. Low variance
- The new Mamba block should not turn the model into a heavy adaptive interaction model
- Avoid support-query coupling
- Avoid dynamic basis generation
- Avoid transductive refinement

C. Mamba only where it is justified
- Apply Mamba only to the stable token stream
- Do not replace the whole backbone with a full Mamba backbone
- Do not add Mamba to the local matching head
- Do not add Mamba to the support-query fusion stage

D. Preserve original SPIF strengths
- Keep the global prototype branch
- Keep the lightweight local partial matching branch
- Keep the overall inductive bias stable

====================================================
5. FILE IMPLEMENTATION REQUIREMENTS
====================================================

Create a completely NEW file, e.g.:
- models/spif_mamba.py

Inside it, implement new classes, for example:
- StableEvidenceMambaPropagator
- SPIFMambaEncoder
- SPIFMambaCE
- SPIFMambaMAX

You may adapt to repo naming conventions, but the separation must remain clear.

Again:
- DO NOT edit original SPIF file
- DO NOT move code out of original SPIF file
- DO NOT refactor baseline file
- NEW implementation only

====================================================
6. BACKBONE AND TOKENIZATION
====================================================

Reuse the same backbone conventions already used in the repository and in SPIF baselines whenever possible.

Recommended:
- use the same backbone options as SPIFCE / SPIFMAX
- keep fair comparability

Expected backbone output:
F: [B, C, H, W]

Flatten to tokens:
X: [B, L, C]
where L = H * W

Do not add a larger backbone than the original baseline by default.

====================================================
7. STABLE-VARIANT FACTORIZATION
====================================================

Preserve the original SPIF factorization idea.

From token map X, compute:
- Xs: stable token candidates
- Xv: variant token candidates

Use lightweight projection heads only.

Recommended structure for each projection head:
- LayerNorm
- Linear
- GELU
- Linear

Inputs:
X: [B, L, C]

Outputs:
- Xs: [B, L, Ds]
- Xv: [B, L, Dv]

Keep the heads shallow.

Why this exists:
- not all observed features are equally trustworthy for class identity
- stable branch should represent class-relevant evidence
- variant branch should absorb nuisance / style / background / acquisition variation

====================================================
8. NEW COMPONENT: STABLE EVIDENCE MAMBA PROPAGATOR
====================================================

This is the new contribution.

Implement a lightweight Mamba-based block that operates ONLY on Xs.

Input:
Xs: [B, L, Ds]

Output:
Hs: [B, L, Ds]

Hs should be interpreted as stable evidence states, not just static stable tokens.

Very important:
- keep this block shallow
- default should be ONE Mamba block only
- optionally allow stacking depth = 0 / 1 / 2 for ablation
- default to depth = 1

This module should:
- propagate context through the stable token stream
- allow weak but relevant evidence to become more identifiable
- improve the quality of token gating
- remain lightweight enough not to destroy 1-shot stability

Implementation guidance:
- if the repository already has a Mamba/SSM implementation, reuse it
- otherwise use a lightweight, well-structured implementation
- wrap it in a clearly named module such as:
  StableEvidenceMambaPropagator

Add residual connection and normalization if appropriate.

Suggested pattern:
- input norm
- Mamba block
- residual add
- optional small FFN
- residual add

But keep it small.
Do not build a deep transformer-like stack.

====================================================
9. WHY MAMBA IS USED HERE
====================================================

This rationale must be reflected in code comments and module naming:

The gate alone cannot solve the following:
- weak evidence tokens may be too weak individually
- distributed stable evidence may not be identifiable token-by-token
- nuisance tokens may appear strong locally
- stable evidence may only become clear after contextual propagation

Therefore, Mamba is inserted before the gate so that:
- the gate scores evidence states, not raw stable token candidates
- stable evidence can be selectively propagated
- nuisance can be suppressed by the propagated representation
- the final selected evidence is more reliable for both global pooling and local matching

This explanation is central.
The code structure should make this architectural logic obvious.

====================================================
10. STABLE EVIDENCE GATE
====================================================

After Mamba propagation, compute the stable evidence gate on Hs, not on raw Xs.

Gate network:
- LayerNorm
- Linear(Ds -> hidden)
- GELU
- Linear(hidden -> 1)
- Sigmoid

Output:
g: [B, L, 1]

Apply:
Hs_hat = g * Hs

Important:
- gate is per-sample only
- no support-query interaction
- no class-conditioned gate
- no cross-attention

Interpretation:
- Mamba creates contextualized stable evidence states
- gate selects how much of each evidence state should be kept

====================================================
11. GLOBAL STABLE EMBEDDING BRANCH
====================================================

Compute weighted pooled stable embedding from Hs_hat:

z = sum(Hs_hat over tokens) / (sum(g over tokens) + eps)

Output:
z: [B, Ds]

Then L2-normalize z.

This branch must remain the low-variance backbone of the classifier.

Why:
- strong 1-shot behavior
- stable class representation
- maintain prototype-based bias from SPIF

====================================================
12. LOCAL PARTIAL MATCHING BRANCH
====================================================

Preserve the original SPIF philosophy:
- lightweight
- largely non-parametric
- partial matching, not full greedy matching

Use the propagated-and-gated stable token states Hs_hat for local matching.

Create support token pool per class:
support_pool[c]: [K * L, Ds]

For each query token set:
- compute cosine similarity matrix with support_pool[c]
- for each query token, take top-r support-token similarities
- average top-r
- average over query tokens

This gives local class score:
s_local[q, c]

Why:
- same-class samples may share only partial stable evidence
- top-r partial matching is more appropriate than forcing full alignment
- keeps local detail without adding heavy pairwise attention

Do not replace this with:
- full OT solver by default
- dense cross-attention
- relation network
- reconstruction head

====================================================
13. GLOBAL PROTOTYPE BRANCH
====================================================

Using support global stable embeddings:
support_global: [N, K, Ds]

Compute class prototype:
p_c = mean_k(support_global[c, k])

Then compute cosine similarity with query global stable embeddings:
s_global[q, c]

Do not replace this with a learned classifier head.

This branch is essential for low variance and 1-shot robustness.

====================================================
14. FINAL FUSION
====================================================

Fuse scores as:
s[q, c] = alpha * s_global[q, c] + (1 - alpha) * s_local[q, c]

For CE version:
- fixed alpha, default 0.7 or 0.75

For MAX version:
- use a single learnable scalar alpha_logit
- alpha = sigmoid(alpha_logit)
- initialize near 0.7

Do not add:
- per-class alpha
- per-query alpha network
- fusion MLP

Keep fusion simple and stable.

====================================================
15. TWO MODEL VARIANTS TO IMPLEMENT
====================================================

A. SPIFMambaCE
Fair version:
- backbone
- stable/variant factorization
- stable evidence Mamba propagator
- gate
- global prototype branch
- local partial matching branch
- fused logits
- ONLY episodic cross-entropy

No extra losses.

B. SPIFMambaMAX
Stronger version:
same architecture as SPIFMambaCE, plus lightweight regularization losses:

1. L_cons
- stable embedding consistency across two augmented views

2. L_decorr
- decorrelation penalty between stable and variant global embeddings

3. L_sparse
- mean gate activation penalty to prevent gate from selecting everything

Total:
L = L_ce + lambda1 * L_cons + lambda2 * L_decorr + lambda3 * L_sparse

Recommended starting values:
- lambda1 = 0.05 or 0.1
- lambda2 = 0.005 or 0.01
- lambda3 = 0.0005 or 0.001

Keep configurable.

====================================================
16. REQUIRED ABLATIONS
====================================================

Please structure the code so these ablations are easy:

- original SPIFCE vs SPIFMambaCE
- original SPIFMAX vs SPIFMambaMAX
- no Mamba (identity)
- Mamba depth 1 vs 2
- gate before Mamba vs gate after Mamba
- local-only vs global-only vs fused
- fixed alpha vs learnable alpha
- top-r choice
- with and without factorization

This is important because the scientific claim is that Mamba helps specifically by improving stable evidence propagation before gating.

====================================================
17. WHAT MUST NOT BE DONE
====================================================

Do NOT:
- modify original SPIF source file
- change original SPIF behavior
- replace the whole backbone with VMamba
- add support-query cross-attention
- add transductive refinement
- add dynamic class basis generation
- add reconstruction classifier
- add full episode transformer
- add relation network heads
- add greedy heavy modules that increase variance

The model must remain disciplined.

====================================================
18. ENGINEERING EXPECTATIONS
====================================================

- Reuse original SPIF code patterns when appropriate, but by copying/adapting into the NEW file
- Do not import and monkey-patch original SPIF classes
- Keep tensor shapes explicit in comments
- Add clear code comments for each module
- Explain in comments WHY each component exists
- Preserve compatibility with existing train / eval entry points
- Register the new model names cleanly
- Ensure both shot=1 and shot>1 work without special hacks

====================================================
19. CLEAR ARCHITECTURAL RATIONALE FOR EACH COMPONENT
====================================================

When coding, document the intent of each component:

1. Stable-variant factorization
Why:
- separate class-relevant evidence from nuisance variation

2. Mamba stable evidence propagator
Why:
- raw stable token candidates may be weak, fragmented, or ambiguous
- evidence should be contextualized and propagated before gating

3. Gate on propagated stable evidence
Why:
- select reliable evidence states rather than static token guesses

4. Global prototype branch
Why:
- keep a low-variance, strong 1-shot classification path

5. Local partial matching branch
Why:
- preserve fine-grained stable evidence without forcing full similarity

6. Simple fusion
Why:
- combine stability of global prototypes with flexibility of local partial matching
- avoid over-parameterized fusion

====================================================
20. PREFERRED INTERNAL CLASS STRUCTURE
====================================================

Example structure inside the new file:

- class StableVariantProjector(nn.Module)
- class StableEvidenceMambaPropagator(nn.Module)
- class StableEvidenceGate(nn.Module)
- class SPIFMambaEncoder(nn.Module)
- class SPIFMambaCE(nn.Module)
- class SPIFMambaMAX(nn.Module)

Example helper methods:
- encode_backbone(x)
- flatten_tokens(feature_map)
- factorize_tokens(tokens)
- propagate_stable_tokens(stable_tokens)
- compute_gate(stable_states)
- pool_global(stable_states, gate)
- build_support_prototypes(support_global)
- build_support_token_pool(support_tokens)
- compute_global_scores(query_global, prototypes)
- compute_local_partial_scores(query_tokens, support_token_pool)
- forward_episode(...)

====================================================
21. OUTPUTS
====================================================

Forward should return:
- fused logits

Also return diagnostics when possible:
- global_scores
- local_scores
- alpha
- gate_mean
- stable_global_embeddings
- variant_global_embeddings
- optionally pre-Mamba and post-Mamba stable token summaries for debugging

====================================================
22. FINAL INSTRUCTION
====================================================

Implement this as a clean, faithful, non-greedy SPIF extension where Mamba is a real contribution:
- Mamba must improve stable evidence propagation before gating
- the architecture must remain strong in 1-shot
- the original SPIF files must remain untouched
- all new logic must live in a new file
- code must be accurate, readable, and ablation-friendly

Do not overcomplicate.
Do not silently redesign the method.
Follow this architecture precisely.