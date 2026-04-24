# CBCR-FSL Part 1: Formulation and Architecture Design Note

## Document Purpose

This note is written for LLM-based design review.

Its purpose is not to sell the idea rhetorically.
Its purpose is to expose the proposed model in a way that another model can audit:

- what the model actually changes relative to DeepEMD,
- which parts are the true source of novelty,
- which parts are only implementation choices,
- which claims are strong enough to defend in a paper,
- and which failure modes or reviewer attacks are likely.

This is **Part 1** only.
Scope of Part 1:

- problem formulation,
- research gap,
- architecture and scoring formulation,
- novelty boundaries,
- scientific claims and risks.

Out of scope for Part 1:

- exact code structure,
- training script integration,
- hyperparameter sweep plan,
- full experiment table design,
- implementation ablations in code.

Those belong in a later Part 2.

---

## 1. Working Name

**CBCR-FSL** = **Class-Barycentric Competitive Robust Transport for Few-Shot Learning**

Short meaning:

- **Class-Barycentric**: the class is modeled as a class-level support distribution, not as independent query-shot pairwise matches.
- **Competitive**: query evidence is not scored against each class in isolation only; class explanations compete for the same query evidence budget.
- **Robust Transport**: the class score is uncertainty-aware and supports rejectable / partial evidence instead of forced explanation.

---

## 2. One-Sentence Summary

CBCR-FSL replaces DeepEMD's pairwise query-support transport with an uncertainty-aware class-level transport formulation in which each class is represented by a shot-derived barycentric measure, query evidence is allocated competitively across classes, and unreliable evidence can be rejected through a sink mechanism.

---

## 3. Problem Setting

We remain in standard episodic few-shot classification.

For each episode:

- `N` classes,
- `K` support images per class,
- one or more query images,
- backbone produces a spatial token set per image.

Let:

- `T(q) = {u_i}` be query tokens,
- `T(c, k) = {v_{c,k,j}}` be support-shot tokens for class `c`, shot `k`.

DeepEMD solves a transport problem between query tokens and support-side dense features, with cross-reference weighting to suppress irrelevant regions.

CBCR-FSL keeps the local-evidence philosophy, but changes the object being compared.

DeepEMD compares:

- query vs support structure,
- then aggregates at class decision time.

CBCR-FSL compares:

- query distribution vs **class-level support measure**,
- with explicit uncertainty and competition built into the scoring rule.

---

## 4. Why DeepEMD Is Strong

DeepEMD is strong for three real reasons:

1. It does **not** collapse the image into a single vector too early.
2. It compares local evidence with **global assignment consistency**, not greedy nearest-neighbor matching.
3. Its cross-reference weighting suppresses irrelevant regions better than uniform local matching.

From the original paper, the cross-reference weighting is a meaningful gain, but the paper also shows that weighting alone is not the whole story: weighting without EMD helps only slightly, while weighting plus EMD gives the big gain.

So the actual DeepEMD lesson is:

> the winner is not "weights" alone and not "local descriptors" alone, but structured local matching over weighted evidence.

Any new model that ignores this lesson will likely lose to DeepEMD.

---

## 5. Research Gap in DeepEMD

The proposed model is not justified if it only says "DeepEMD is old, we add a new module."
The real gap must be structural.

### 5.1 Pairwise Formulation Gap

DeepEMD fundamentally operates through pairwise transport between a query and a support-side object.
Even when it handles `K-shot`, the final formulation does not fully treat the support set as a single uncertain class distribution.

This matters because in few-shot learning:

- the support set is small,
- shots may be heterogeneous,
- one shot may be noisy or only partially informative,
- and the class should be inferred from the joint support evidence, not only from pairwise similarity.

### 5.2 No Explicit Uncertainty Object

DeepEMD has weighting, but it does not explicitly represent:

- class uncertainty under low shot,
- support disagreement,
- support dispersion,
- or uncertainty-calibrated class acceptance.

This is especially problematic in:

- `1-shot`, where the single support image is an unreliable estimator,
- noisy scalograms,
- outlier support shots,
- and partial local evidence.

### 5.3 No Class-Competitive Explanation

DeepEMD scores classes largely independently.
But a query token that is weakly compatible with multiple classes should not be allowed to strongly help all of them simultaneously.

In other words:

- the model lacks an explicit **class-competitive evidence budget**,
- so ambiguous background structure can be over-explained by several classes at once.

### 5.4 No Native Rejectable Evidence

DeepEMD's weighting suppresses weak regions, but it still does not turn partial evidence into a first-class object in the scoring rule.
For noisy scalograms, this is not enough.

What is needed is not only "important tokens have lower or higher mass", but:

> some evidence should be explicitly allowed to remain unmatched or be assigned to a reject sink.

### 5.5 K-Shot Support Identity Is Not the Final Inference Object

DeepEMD improves k-shot handling with structured FC / refinement ideas, but it still does not cleanly formalize the class as:

- a set of shot measures,
- pooled into a class-level uncertain measure,
- with explicit support disagreement accounted for in the decision function.

This is the exact opening for a genuinely new formulation.

---

## 6. Core Design Goal

The goal is **not** to make a more complicated DeepEMD.

The goal is to change the few-shot object being scored:

- from pairwise query-support transport,
- to class-level uncertain transport with competitive evidence allocation.

If this is not the main story, the work will look incremental.

So the central paper claim should be:

> Few-shot classification with local transport should operate on a class-level uncertain support measure rather than on isolated pairwise query-support matches.

Everything else must serve this claim.

---

## 7. Proposed Formulation Overview

CBCR-FSL has five main components:

1. **Shot-level support measures**
2. **Class barycentric support measure**
3. **Class uncertainty radius**
4. **Competitive query evidence allocation**
5. **Robust score with reject sink**

The novelty is not any single module in isolation.
The novelty is the combination under one coherent formulation:

- class-level transport,
- uncertainty-aware decision rule,
- competitive explanation,
- rejectable partial evidence.

---

## 8. Component A: Shot-Level Support Measures

### 8.1 Definition

Each support shot is represented as an empirical weighted measure:

`mu_{c,k} = sum_j a_{c,k,j} delta(v_{c,k,j})`

where:

- `v_{c,k,j}` is the `j`-th token of support shot `k` in class `c`,
- `a_{c,k,j}` is the token mass / importance.

The query is also represented as a weighted measure:

`nu_q = sum_i b_i delta(u_i)`

### 8.2 Why This Exists

This preserves the strongest lesson from DeepEMD:

- classification should remain local-evidence aware,
- token importance matters,
- and background should not be treated uniformly.

### 8.3 What Is New Here

Strictly speaking, **this part alone is not the novelty**.
DeepEMD already uses weighted local evidence.

In CBCR-FSL, this block is a prerequisite, not the paper contribution.

### 8.4 Acceptable Mass Choices

Possible token mass choices:

- uniform,
- cross-reference style,
- support-only reliability,
- query-conditioned reliability,
- energy-based importance,
- physically informed mass from scalogram intensity or eventness.

For a clean paper story, token mass should stay lightweight.
The main paper should not depend on a complicated new token weighter.

### 8.5 Reviewer Risk

If the token weighting becomes too fancy, reviewers will say:

> this is just DeepEMD with a different weighting heuristic.

So this module must stay secondary.

---

## 9. Component B: Class Barycentric Support Measure

### 9.1 Definition

Given `K` support-shot measures for class `c`,

`{mu_{c,1}, ..., mu_{c,K}}`

we construct a class-level support measure:

`nu_c_hat = Barycenter(mu_{c,1}, ..., mu_{c,K})`

This barycenter can be:

- balanced OT barycenter,
- unbalanced OT barycenter,
- entropic approximate barycenter,
- or a lightweight support posterior measure that behaves like a barycentric class object.

### 9.2 Why This Exists

This is the first true break from DeepEMD.

The class is no longer treated mainly as a set of separate pairwise matches.
Instead, the support set produces a **class-level uncertain object** before query scoring.

This addresses:

- support-shot disagreement,
- support redundancy,
- shot complementarity,
- and the fact that class evidence should exist at the class level, not only at the pairwise level.

### 9.3 Scientific Claim

The key scientific claim here is:

> a class barycentric support measure is a better few-shot inference object than independent pairwise support matches when support shots are sparse, noisy, or partially inconsistent.

### 9.4 Why This Is Not Just Pooling

This is not simple averaging.

Simple averaging:

- destroys support geometry,
- over-smooths heterogeneous shots,
- and does not preserve transport semantics.

Barycentric pooling is different because it is defined in transport geometry.

### 9.5 Novelty Status

This is a **major novelty source**.

It changes the object being classified.

### 9.6 Reviewer Risk

The likely attack is:

> this is just support pooling in OT language.

So the defense must be:

- barycentric support object is not heuristic averaging,
- it is transport-geometric aggregation,
- and it leads to different behavior under shot conflict and partial support coverage.

---

## 10. Component C: Class Uncertainty Radius

### 10.1 Definition

For each class `c`, define an uncertainty radius `epsilon_c`.

This can depend on:

- shot count `K`,
- support dispersion,
- disagreement between support shots and the barycenter,
- or a lightweight learned predictor from support statistics.

The radius reflects how confident the model should be that `nu_c_hat` is a reliable class summary.

### 10.2 Why This Exists

Few-shot classification is not only about similarity.
It is also about confidence in the class estimate.

In `1-shot`, the class object is highly uncertain.
In `5-shot` with one corrupted support shot, uncertainty should also increase.

Without this, the model may be structurally elegant but still overconfident.

### 10.3 Robust Score

Instead of using only `W(q, c)`, we use a robust score such as:

`score(q, c) = - max(0, W(nu_q, nu_c_hat) - epsilon_c)`

Interpretation:

- if query evidence falls within the class uncertainty radius, the class should not be punished too aggressively,
- if query is far beyond the credible class envelope, the class should be rejected.

### 10.4 Novelty Status

This is a **major novelty source**, especially for few-shot robustness and calibration.

### 10.5 Reviewer Risk

The main attack:

> epsilon is just another threshold.

The defense:

- `epsilon_c` is class- and episode-dependent,
- tied to support uncertainty,
- and connected to the class inference object,
- unlike a single global scalar threshold.

---

## 11. Component D: Competitive Query Evidence Allocation

### 11.1 Definition

Query evidence should not be scored against every class in a fully independent way.

Instead, there should be a competitive mechanism so that ambiguous evidence cannot strongly support multiple classes simultaneously.

There are two implementation levels:

1. **Soft competition**:
   compute class scores jointly and normalize shared evidence contributions.
2. **Harder transport competition**:
   query mass is explicitly budgeted across classes through a shared or approximated multi-class transport allocation.

### 11.2 Why This Exists

This addresses a deep weakness of independent class scoring:

- a weak background token may look mildly compatible with many classes,
- and if each class is evaluated independently, all classes may overuse it.

Competition forces the model to answer:

> which class gets to explain this query evidence?

### 11.3 Novelty Status

This is a **major novelty source** because DeepEMD does not tell a strong class-competition story.

### 11.4 Reviewer Risk

The main risk is computational explosion or theoretical overreach.

So for Part 1, the document should stay honest:

- full exact multimarginal transport may be too expensive,
- but class competition is still the right formulation target,
- and practical approximations are acceptable if the paper clearly separates ideal formulation and tractable implementation.

---

## 12. Component E: Reject Sink for Partial Evidence

### 12.1 Definition

Add a reject / sink channel so that part of the query mass can remain unmatched when support evidence is insufficient or noisy.

### 12.2 Why This Exists

This is essential for noisy scalograms and partial evidence.

The problem is not only that some tokens are low importance.
The deeper issue is:

> some tokens should not be explained by any class at all.

The sink provides this option explicitly.

### 12.3 Novelty Status

This is a meaningful supporting novelty, but probably not the main headline by itself.

### 12.4 Reviewer Risk

If overemphasized, reviewers may say:

> this is just unbalanced OT / dustbin matching.

So the sink should be presented as a supporting mechanism that makes the class-level robust formulation viable, not as the core contribution.

---

## 13. Final Score Definition

A clean final scoring story is:

1. Build query measure `nu_q`.
2. Build class support measure `nu_c_hat`.
3. Allocate evidence competitively across classes.
4. Allow part of the evidence to go to reject sink.
5. Compute class transport distance `D_c`.
6. Apply class uncertainty radius `epsilon_c`.
7. Produce final logit:

`logit_c = - max(0, D_c - epsilon_c)`

This yields a score that is:

- local-evidence aware,
- class-level,
- uncertainty-aware,
- and robust to unmatched evidence.

---

## 14. What the Main Novelty Actually Is

The paper must not claim novelty in everything.
That weakens the story.

The clean novelty decomposition is:

### Primary novelty

- class-level barycentric support inference,
- class-dependent uncertainty-aware robust scoring,
- class-competitive evidence allocation.

### Secondary novelty

- reject sink for unmatched evidence,
- lightweight token masses adapted to scalogram evidence.

### Non-novel supporting choices

- standard backbone,
- ordinary token projection,
- conventional entropic OT solver,
- standard episodic training.

This separation is important.

---

## 15. What the Model Must Not Become

To stay scientifically clean, CBCR-FSL should **not** become:

- DeepEMD plus another attention block,
- a general-purpose transformer over all support and query tokens,
- a giant query-conditioned reasoning engine,
- a fully heuristic collection of tricks,
- or a solver paper disguised as a few-shot model.

If the implementation drifts there, the paper story collapses.

---

## 16. Direct Comparison Against DeepEMD

The clean comparison is:

### DeepEMD

- local weighted descriptors,
- pairwise transport,
- strong assignment structure,
- weak explicit treatment of class-level support uncertainty,
- weak explicit class competition.

### CBCR-FSL

- local weighted descriptors,
- class-level support measure,
- robust score under class uncertainty,
- explicit competition for query evidence,
- explicit unmatched evidence handling.

So the message is not:

> DeepEMD is wrong.

It is:

> DeepEMD solves local matching well, but still uses the wrong inference object for uncertain few-shot classes.

That is a much stronger and more defensible claim.

---

## 17. Expected Strengths

If the formulation is correct, CBCR-FSL should be especially strong in:

1. **1-shot calibration**
   because uncertainty is explicit instead of hidden.

2. **5-shot with support disagreement**
   because the class object is inferred from all shots jointly.

3. **noisy or partial-evidence scalograms**
   because rejectable evidence is first-class.

4. **support-outlier robustness**
   because shot conflict affects the barycenter and the uncertainty radius rather than being silently averaged away.

5. **background ambiguity**
   because class competition limits multi-class over-explanation.

---

## 18. Expected Weaknesses

The model will likely be weaker than simpler baselines when:

1. The dataset is extremely clean and low-noise.
   Then the extra robustness machinery may not help.

2. Support shots are already highly consistent.
   Then simple pairwise transport may already be good enough.

3. The barycenter approximation is poor.
   Then the class object can become an over-smoothed summary.

4. Competition is implemented too aggressively.
   Then the model may suppress useful ambiguous evidence.

5. The uncertainty radius is badly calibrated.
   Then the model may become too tolerant or too conservative.

This should be admitted early rather than hidden.

---

## 19. Reviewer Attack Points and Short Answers

### Attack 1

"This is just DeepEMD with support pooling."

Answer:

No. The class object and the decision rule both change.
The model does not merely pool support features before pairwise matching; it scores a query against an uncertainty-aware class measure and adds competition across classes.

### Attack 2

"Your epsilon is just a threshold."

Answer:

No. A global threshold is not tied to class uncertainty.
`epsilon_c` is class- and episode-dependent and derived from support uncertainty.

### Attack 3

"Competition is too expensive to be practical."

Answer:

Exact multimarginal transport may indeed be expensive.
But the scientific contribution is the formulation.
The implementation may use a controlled approximation while preserving the class-competitive principle.

### Attack 4

"The sink is not novel."

Answer:

Correct. The sink is not claimed as the core novelty.
It is a necessary supporting mechanism for partial-evidence robust scoring.

### Attack 5

"Why not simply use FRN / FEAT / MetaQDA?"

Answer:

Those models address different weaknesses:

- FEAT improves task adaptation,
- FRN treats support as a reconstruction basis,
- MetaQDA improves uncertainty and calibration,
- DeepEMD improves structured local matching.

CBCR-FSL is motivated by the missing combination:

- local transport structure,
- class-level uncertain support inference,
- and class-competitive explanation.

---

## 20. Minimum Claims That Are Safe to Defend

These are the claims most likely to survive reviewer scrutiny:

1. DeepEMD is a strong local matching baseline because it preserves weighted local evidence and solves a structured assignment problem.
2. DeepEMD still lacks an explicit class-level uncertain support inference object.
3. Few-shot support disagreement should influence both class representation and score confidence.
4. Competitive allocation of query evidence is more principled than fully independent class scoring when evidence is ambiguous.
5. A rejectable evidence mechanism is important for partial local evidence settings such as noisy scalograms.

These are strong, moderate, and defensible.

---

## 21. Claims That Are Too Aggressive Right Now

These should **not** be claimed yet:

1. "CBCR-FSL strictly generalizes DeepEMD."
2. "Competition is always better than independent scoring."
3. "Barycenter is the only correct class object."
4. "The robust score has complete theoretical guarantees in our implemented approximation."
5. "This is guaranteed to outperform DeepEMD on standard clean benchmarks."

These are too strong unless the experiments are overwhelming.

---

## 22. What an LLM Reviewer Should Evaluate

If another LLM reviews this design, it should answer these questions:

1. Is the proposed novelty truly formulation-level, or only module-level?
2. Is the distinction from DeepEMD clear and non-trivial?
3. Is the barycentric class object scientifically justified for few-shot support inference?
4. Is the uncertainty radius meaningful or merely a renamed threshold?
5. Is the class competition principle coherent and worth approximating?
6. Which component is indispensable for the story, and which are optional support mechanisms?
7. What is the weakest link in the formulation?
8. Which reviewer criticisms are most dangerous?

If the reviewer cannot answer those, the note is still too vague.

---

## 23. Bottom-Line Position

The proposed model is only worth pursuing if the work stays anchored on this single sentence:

> DeepEMD is strong because it solves local structured matching well, but it still scores the wrong few-shot object; CBCR-FSL changes the inference object from pairwise support matching to uncertainty-aware class-level competitive transport.

If the implementation and experiments remain faithful to that sentence, the model has a chance to look genuinely new.

If the implementation drifts into "DeepEMD plus more modules", it will look incremental and weak.

---

## 24. Proposed Next Document

Part 2 should define:

- exact tractable implementation,
- which approximation of barycenter to use,
- how to implement competition without exploding compute,
- training losses,
- diagnostics,
- and ablation table design.

