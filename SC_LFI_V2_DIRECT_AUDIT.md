# Direct Audit of SC-LFI-v2

This is a direct audit of the current `sc_lfi_v2` implementation against:

- the original SC-LFI claim,
- the stronger `SC_LFI_V3_THEORY_NOTE.md` formulation,
- and the empirical `1-shot` / `5-shot` failure modes already observed in training logs.

This note is intentionally direct.
The purpose is to decide what survives into `v3`, and what should be removed.

## 1. Bottom-Line Verdict

`sc_lfi_v2` is **not a bad model in the same way as `v1`**.
It fixed several obvious weaknesses:

- transport scoring is stronger;
- support masses are learned;
- solver quality is better;
- support memory conditioning exists;
- direct margin supervision exists.

But as a few-shot model, especially for `1-shot`, it is still **structurally wrong at the center**.

The central issue is:

> `v2` still defines the class too much as a mixed anchor-plus-generated particle cloud, instead of a posterior predictive evidence distribution inferred from a tiny support set.

That is why it can train, but still fails to become a truly strong few-shot method.

## 2. Empirical Symptoms That the Theory Is Still Wrong

### 2.1 `1-shot` peaks early and then collapses into overfitting

From the `60 samples, 1-shot` run:

- epoch 1: `Train=0.9475, Val=0.7458`
- epoch 2: `Train=1.0000, Val=0.7642`
- epoch 3: `Train=1.0000, Val=0.7675`
- epoch 4: `Train=0.6650, Val=0.3983`

See:

- [output.log](/mnt/disk2/nhatnc/res/scalogram_fewshot/pulse_fewshot/wandb/run-20260402_204424-fmeq6vpp/files/output.log#L19)
- [output.log](/mnt/disk2/nhatnc/res/scalogram_fewshot/pulse_fewshot/wandb/run-20260402_204424-fmeq6vpp/files/output.log#L37)

By epoch 100:

- `Train=1.0000`
- `Val=0.5775`

See:

- [output.log](/mnt/disk2/nhatnc/res/scalogram_fewshot/pulse_fewshot/wandb/run-20260402_204424-fmeq6vpp/files/output.log#L421)

This means:

- the model can fit episodes;
- but its class inference mechanism is still too brittle to generalize in low-shot.

### 2.2 `5-shot` does not show the expected gain

On the `60 samples, 5-shot` run, the first two epochs already show:

- epoch 1: `Val=0.7550`
- epoch 2: `Val=0.7558`

This is not clearly better than the best `1-shot` peak (`0.7675`).

See:

- [output.log](/mnt/disk2/nhatnc/res/scalogram_fewshot/pulse_fewshot/wandb/run-20260402_225442-hofa4k2w/files/output.log#L19)

That is a strong signal that the architecture is not exploiting additional support examples as effectively as a good few-shot model should.

### 2.3 The model is too expensive relative to the gain

`v2` only barely improved over `v1` in the low-shot benchmark that matters, while becoming much slower.

That is not an implementation bug.
It is an architecture-value problem.

## 3. Exact or Strong Parts of v2

These are the parts that are genuinely correct or strong enough to keep in spirit.

### 3.1 The core classifier claim is preserved

The top-level docstring is still conceptually correct:

- latent evidence tokens `e = Psi(z)`
- support-conditioned class measure
- query-class scoring as distribution fit

See:

- [sc_lfi_v2.py](/mnt/disk2/nhatnc/res/scalogram_fewshot/pulse_fewshot/net/sc_lfi_v2.py#L30)

Verdict:

- **Exact to intended claim**

### 3.2 Learned query/support token masses are correct in principle

`LatentEvidenceProjectorV2` maps tokens to latent evidence and token masses via:

- `e = Psi(z)`
- `a = softmax(W_mass(e))`

See:

- [latent_projector_v2.py](/mnt/disk2/nhatnc/res/scalogram_fewshot/pulse_fewshot/net/modules/latent_projector_v2.py#L10)

Verdict:

- **Exact in spirit**
- but incomplete because masses are not yet uncertainty-calibrated or query-conditioned

### 3.3 The flow solver abstraction is correct

The fixed-step Euler/Heun solver is clean and numerically much better than `v1`.

See:

- [conditional_flow_v2.py](/mnt/disk2/nhatnc/res/scalogram_fewshot/pulse_fewshot/net/modules/conditional_flow_v2.py#L106)

Verdict:

- **Good engineering abstraction**
- likely reusable in `v3`

### 3.4 Weighted transport scoring is a correct direction

The scoring distance uses weighted paper-style sliced Wasserstein.

See:

- [transport_distance_v2.py](/mnt/disk2/nhatnc/res/scalogram_fewshot/pulse_fewshot/net/modules/transport_distance_v2.py#L200)

Verdict:

- **Good**
- not the source of the main conceptual failure

### 3.5 Direct hard-negative margin loss is correct in spirit

The margin:

- `relu(m + d_true - d_neg)`

is a valid direct discriminative pressure.

See:

- [flow_losses_v2.py](/mnt/disk2/nhatnc/res/scalogram_fewshot/pulse_fewshot/net/modules/flow_losses_v2.py#L105)

Verdict:

- **Keep conceptually**

## 4. Acceptable Approximations in v2

These are not ideal, but they are acceptable if the architecture around them is right.

### 4.1 Finite-particle measure approximation

The class distribution is represented by finite particles.

See:

- [sc_lfi_v2.py](/mnt/disk2/nhatnc/res/scalogram_fewshot/pulse_fewshot/net/sc_lfi_v2.py#L50)

Verdict:

- **Acceptable engineering approximation**

### 4.2 Linear FM path

The path:

- `y_t = (1 - t) epsilon + t e`
- `u_t = e - epsilon`

is standard enough and can remain if the measure semantics are corrected.

See:

- [conditional_flow_v2.py](/mnt/disk2/nhatnc/res/scalogram_fewshot/pulse_fewshot/net/modules/conditional_flow_v2.py#L39)
- [flow_losses_v2.py](/mnt/disk2/nhatnc/res/scalogram_fewshot/pulse_fewshot/net/modules/flow_losses_v2.py#L20)

Verdict:

- **Acceptable**
- not the main issue

### 4.3 Compact support memory

Using compact memory tokens instead of all support tokens is acceptable if:

- the support basis is still preserved somewhere else.

See:

- [set_context_v2.py](/mnt/disk2/nhatnc/res/scalogram_fewshot/pulse_fewshot/net/modules/set_context_v2.py#L72)

Verdict:

- **Acceptable compression**
- but only if the class object is not reduced to memory-conditioned generation

## 5. Theoretically Weak Parts of v2

These are the main problems.

## 5.1 The class measure is still heuristic mixture, not posterior inference

Current formula:

- `muhat_c = rho_c * mu_anchor + (1 - rho_c) * mu_flow`

See:

- [sc_lfi_v2.py](/mnt/disk2/nhatnc/res/scalogram_fewshot/pulse_fewshot/net/sc_lfi_v2.py#L45)
- [sc_lfi_v2.py](/mnt/disk2/nhatnc/res/scalogram_fewshot/pulse_fewshot/net/sc_lfi_v2.py#L337)

Problem:

- `rho_c` is produced by a learned MLP over `class_summary`
- then clipped into `[support_mix_min, support_mix_max]`

See:

- [sc_lfi_v2.py](/mnt/disk2/nhatnc/res/scalogram_fewshot/pulse_fewshot/net/sc_lfi_v2.py#L294)

Why this is weak:

- it is a heuristic interpolation;
- it is not a support-posterior shrinkage rule;
- it does not explicitly depend on shot uncertainty;
- it does not distinguish `1-shot` and `5-shot` in a mathematically principled way.

Verdict:

- **Theoretically weak**
- must be replaced by shot-aware posterior shrinkage

## 5.2 The class object is still generator-centric

The flow branch samples particles from Gaussian noise:

- `particles = sample_particles(class_summary, support_memory, ...)`

See:

- [sc_lfi_v2.py](/mnt/disk2/nhatnc/res/scalogram_fewshot/pulse_fewshot/net/sc_lfi_v2.py#L324)
- [conditional_flow_v2.py](/mnt/disk2/nhatnc/res/scalogram_fewshot/pulse_fewshot/net/modules/conditional_flow_v2.py#L523)

Problem:

- the flow is still effectively a class-particle generator;
- support evidence influences it only through `class_summary` and `support_memory`;
- the transported object is not the support measure itself.

Why this is weak:

- a few-shot classifier should infer the class from support evidence;
- here the support mainly conditions a generator, rather than defining the base measure being transported.

Verdict:

- **Conceptually weak**
- the main novelty bottleneck of `v2`

## 5.3 The anchor measure is still too literal in `1-shot`

The anchor particles are:

- `[weighted_mean; memory_tokens]`

See:

- [set_context_v2.py](/mnt/disk2/nhatnc/res/scalogram_fewshot/pulse_fewshot/net/modules/set_context_v2.py#L133)

Problem:

- in `1-shot`, `weighted_mean` and the derived memory tokens still come from one support image;
- this gives no real posterior shrinkage against low support cardinality.

Why this is weak:

- the model still over-trusts single-support evidence;
- this matches the rapid early overfitting observed in the `1-shot` log.

Verdict:

- **Few-shot weak**

## 5.4 Query conditioning is too weak for a strong few-shot classifier

Current query usage:

- query masses are learned independently;
- pairwise transport distance is computed afterward.

See:

- [sc_lfi_v2.py](/mnt/disk2/nhatnc/res/scalogram_fewshot/pulse_fewshot/net/sc_lfi_v2.py#L395)

Problem:

- the class measure itself is built without query-conditioned reweighting;
- there is no DeepEMD-style cross-reference or CAN-style pairwise emphasis inside class evidence selection.

Why this is weak:

- support atoms that are irrelevant to a given query still remain active in the same class measure;
- few-shot local matching benefits are only partially realized.

Verdict:

- **Theoretically weak**

## 5.5 The support masses are learned, but not uncertainty-aware

Masses are simple softmax outputs:

- no entropy floor
- no consistency regularization
- no uncertainty statistic exposed for shrinkage

See:

- [latent_projector_v2.py](/mnt/disk2/nhatnc/res/scalogram_fewshot/pulse_fewshot/net/modules/latent_projector_v2.py#L79)

Problem:

- a single support image can collapse mass onto brittle local artifacts;
- there is no Bayesian or uncertainty-aware control.

Verdict:

- **Numerically and statistically weak**

## 5.6 The support-fit CE branch is useful but theory-unclean

`v2` adds:

- anchor score: `-tau D(nu_q, mu_anchor)`
- CE on that score

See:

- [flow_losses_v2.py](/mnt/disk2/nhatnc/res/scalogram_fewshot/pulse_fewshot/net/modules/flow_losses_v2.py#L140)
- [sc_lfi_v2.py](/mnt/disk2/nhatnc/res/scalogram_fewshot/pulse_fewshot/net/sc_lfi_v2.py#L454)

Problem:

- this branch was added mainly to stabilize training;
- it does not correspond cleanly to the posterior-evidence story;
- it risks reinforcing direct fitting to current support anchors rather than posterior generalization.

Verdict:

- **Useful stabilizer**
- but **not theory-clean**

## 6. Numerically Weak Parts of v2

## 6.1 No shot-aware prior means training dynamics are brittle

Because the class measure is forced to emerge from:

- support anchor,
- flow particles,
- and heuristic mixing,

the model has no proper low-shot prior to fall back on.

This is visible in the training curve:

- very high train accuracy almost immediately,
- unstable validation,
- then memorization.

See:

- [output.log](/mnt/disk2/nhatnc/res/scalogram_fewshot/pulse_fewshot/wandb/run-20260402_204424-fmeq6vpp/files/output.log#L20)
- [output.log](/mnt/disk2/nhatnc/res/scalogram_fewshot/pulse_fewshot/wandb/run-20260402_204424-fmeq6vpp/files/output.log#L422)

Verdict:

- **Numerically weak because the theory is weak**

## 6.2 Full model is trainable in a tiny-data regime with no protective structure

The run uses:

- `12,974,788` trainable parameters
- `label_smoothing=0`
- `train_augment=False`

See:

- [output.log](/mnt/disk2/nhatnc/res/scalogram_fewshot/pulse_fewshot/wandb/run-20260402_204424-fmeq6vpp/files/output.log#L17)

This is not just a training config issue.
It amplifies the architecture's lack of explicit low-shot shrinkage.

Verdict:

- **Practically weak**

## 7. Inconsistencies with the Stronger v3 Theory

This section states directly where `v2` fails against the intended `v3` formulation.

### 7.1 Missing posterior base measure

`v3` requires:

- `mu_c^0 = alpha_c * nusup_c + (1 - alpha_c) * pi_c`

`v2` does not have:

- a support-conditioned prior measure `pi_c`
- an explicit shrinkage coefficient `alpha_c`
- a shot-aware uncertainty statistic driving shrinkage

Verdict:

- **Inconsistent with v3 theory**

### 7.2 Missing support-measure pushforward semantics

`v3` requires:

- `muhat_c = (T_theta,c)_# mu_c^0`

`v2` instead does:

- sample noise -> generate particles -> mix with support anchor

Verdict:

- **Inconsistent with the intended posterior-evidence transport story**

### 7.3 Missing query-conditioned reweighted posterior scoring

`v3` requires:

- `muhat_c^q = Reweight(muhat_c; omega(q,c))`

`v2` has no such operation.

Verdict:

- **Inconsistent with the intended few-shot scoring theory**

## 8. What Should Survive into v3

These should survive mostly as components, not as the central architecture.

- latent projector idea
- weighted token masses as a base mechanism
- support memory attention utility
- fixed-step Euler/Heun solver
- strong weighted transport kernels
- hard-negative distribution margin loss

## 9. What Must Be Removed or Rewritten

These should not survive as central `v3` concepts.

- support/flow heuristic mixture via `support_mix_min/max`
- class generation from generic Gaussian noise as the default class construction path
- support-fit CE on anchor measure as a core objective
- support anchor defined only as weighted mean plus memory tokens
- support-only class measure with no query-conditioned reliability reweighting

## 10. Final Audit Classification

### Exact or strong enough

- distribution-fit classifier claim
- weighted token evidence
- stronger transport metric
- better solver abstraction
- direct margin loss

### Acceptable approximation

- finite particles
- linear FM path
- compact support memory

### Theoretically weak

- heuristic support/flow interpolation
- generator-centric class construction
- anchor over-trust in `1-shot`
- lack of query-conditioned class reweighting
- lack of explicit uncertainty-aware shrinkage

### Numerically weak

- low-shot training has no proper prior
- token masses can become brittle
- model cost is high relative to low-shot gain

### Inconsistent with the stronger target theory

- no posterior base measure
- no posterior pushforward semantics
- no query-conditioned posterior reliability scoring

## 11. Final Decision

`sc_lfi_v2` should **not** be patched into `v3`.

It should be treated as:

- a useful transitional model,
- a numerically improved version of `v1`,
- and a source of reusable submodules,

but **not** as the correct central architecture for an A-grade few-shot method.

The direct rewrite objective for `v3` should therefore be:

> replace mixed anchor-plus-generated class construction with posterior base-measure construction and support-basis residual transport, then score queries by query-conditioned transport fit to that posterior predictive class evidence measure.
