# Few-Shot Distributional Survey for Redesigning SC-LFI

This note is a theory-first survey of few-shot classification models that are most relevant to a redesign of SC-LFI.

The goal is not to list papers mechanically.
The goal is to extract the few-shot-specific inductive biases that actually matter for `1-shot` and `5-shot`, especially when the final classifier is intended to compare distributions rather than cosine prototypes.

The design target remains:

- each class should be represented as a support-conditioned latent evidence distribution;
- query-class scoring should remain a distribution-fit score;
- but the methodology must be genuinely few-shot-specific, not just a generic flow model attached to an episodic backbone.

## 1. Executive Conclusion

The strongest reusable lesson from the few-shot literature is this:

**good few-shot models do not treat the support set as a generic conditioning vector.**

Instead, they usually do one or more of the following:

- keep the support set as a set or basis, not a single summary;
- adapt representations at the task level before scoring;
- match query evidence to class-local evidence, not only to a class mean;
- use a probabilistic or distributional classifier that explicitly controls uncertainty in low-shot regimes;
- use query-batch structure or distribution alignment when sample bias is severe.

This is exactly where current `sc_lfi_v2` remains weak.
It is more mathematically disciplined than `sc_lfi`, but it is still not yet structurally aligned with what makes modern few-shot methods strong in `1-shot`.

The redesign direction that follows from the survey is:

- **support-anchored residual distribution modeling**, not unconditional latent generation from a compressed class vector;
- **query-conditioned local evidence selection**, not support-only class distribution construction;
- **task-level adaptation before transport scoring**;
- **shot-aware uncertainty control**, especially in `1-shot`;
- **distributional scoring over weighted local evidence**, not a global prototype with transport decoration.

## 2. Survey Axes

The relevant literature clusters into four families.

1. **Task-adaptive few-shot embedding models**
2. **Local descriptor / structured matching models**
3. **Probabilistic or distributional class modeling**
4. **Transductive distribution alignment over the query batch**

For SC-LFI redesign, the correct path is not to copy one family.
It is to combine the right pieces from all four.

## 3. Family A: Task-Adaptive Few-Shot Embedding Models

### 3.1 FEAT

Paper:
- CVPR 2020, *Few-Shot Learning via Embedding Adaptation with Set-to-Set Functions*
- Official repo: `https://github.com/Sha-Lab/FEAT`
- Paper URL: `https://openaccess.thecvf.com/content_CVPR_2020/papers/Ye_Few-Shot_Learning_via_Embedding_Adaptation_With_Set-to-Set_Functions_CVPR_2020_paper.pdf`

Core idea:

- first compute class prototypes from support;
- then adapt them with a permutation-invariant set-to-set transformer over the episode;
- then score query against the adapted class representation.

What matters for us:

- FEAT is not distributional in the OT sense, but it is very few-shot-specific.
- It explicitly models the **episode as a set**, not as independent class summaries.
- It improves classification because support-derived class representations should be **task-adapted before scoring**.

Why FEAT matters for `1-shot`:

- In `1-shot`, the single support sample is very noisy as a class estimate.
- FEAT stabilizes this by adapting class representations in the context of other classes in the episode.
- That is a direct antidote to single-support over-trust.

Lesson for redesign:

- SC-LFI should include a lightweight **episode adapter** before constructing class evidence distributions.
- The adapter must be permutation-invariant over support order and class-order aware only through set interaction.

### 3.2 CAN

Paper:
- CVPR 2020, *Cross Attention Network for Few-Shot Classification*
- Repo widely used; local project already contains a CAN implementation.

Core idea:

- support and query feature maps interact through cross-attention;
- class-relevant regions are emphasized pairwise for each query-class comparison.

What matters for us:

- CAN shows that few-shot performance improves when query and class evidence are **co-conditioned** before scoring.
- This is stronger than building a class object once and comparing everything to it later.

Lesson for redesign:

- SC-LFI should not construct a fixed class distribution completely independently of the query.
- The class evidence distribution used for scoring should include a **query-conditioned reweighting path**.

## 4. Family B: Local Descriptor / Structured Matching Models

### 4.1 DN4

Paper:
- CVPR 2019, *Revisiting Local Descriptor Based Image-to-Class Measure for Few-Shot Learning*

Core idea:

- avoid collapsing images into one vector;
- compare query local descriptors directly against the pooled support local descriptors of each class.

Why this matters:

- DN4 is one of the cleanest demonstrations that few-shot classification is often better viewed as **image-to-class local evidence matching** rather than point-to-point prototype matching.

Lesson for redesign:

- support should remain a **pool of evidence atoms**, not merely a context vector for a generator.
- any class distribution model that erases the support local basis too early is already misaligned with strong few-shot practice.

### 4.2 DeepEMD

Paper and official repo:
- CVPR 2020, *DeepEMD: Few-Shot Image Classification with Differentiable Earth Mover's Distance and Structured Classifiers*
- Official repo: `https://github.com/icoz69/DeepEMD`
- Paper URL: `https://arxiv.org/abs/2003.06777`

Official repo abstract states that DeepEMD:

- computes a structural distance between dense image representations with Earth Mover's Distance;
- uses a **cross-reference mechanism** to generate important weights for elements;
- alleviates cluttered background and large intra-class variation;
- handles `k-shot` with a structured classifier over dense representations.

Why DeepEMD is especially relevant:

- It is distributional in the transport sense.
- It is also truly few-shot-specific because the transport is defined over **support/query local structures**, not over generated latent particles detached from support evidence.

What DeepEMD teaches:

- transport-based scoring becomes strong only when the atoms being transported are semantically meaningful local descriptors;
- weights are crucial and should be **query-aware**, not uniform;
- the class object in few-shot should often be a **structured support set**, not a generated cloud from generic noise.

Lesson for redesign:

- SC-LFI should preserve support local evidence atoms into the scoring pipeline;
- token masses should be query-conditioned at scoring time, not only globally learned per image;
- the flow branch, if retained, should behave as a **residual support-aware augmentation of the support distribution**, not as the primary source of class particles.

### 4.3 FRN

Paper and official repo:
- CVPR 2021, *Few-Shot Classification with Feature Map Reconstruction Networks*
- Official repo: `https://github.com/Tsingularity/FRN`
- Paper URL: `https://arxiv.org/abs/2105.13009`

Core idea:

- treat the support class as a reconstruction basis;
- classify a query by how well its feature map can be reconstructed from the class support descriptors.

Why FRN matters:

- FRN is not an OT model, but it is deeply few-shot-aware.
- It assumes the class representation should be a **descriptor subspace / basis**, not a mean.
- This is especially useful in `1-shot`, because even one image still contains many local descriptors that define a class-conditioned basis.

Lesson for redesign:

- SC-LFI should have a support-basis view of the class.
- A good few-shot flow model should deform or enrich a support basis, not replace it.
- Support anchoring loss should look more like **basis coverage / reconstruction compatibility** than "fit support anchor with CE".

### 4.4 DeepBDC

Paper and official repo:
- CVPR 2022, *Joint Distribution Matters: Deep Brownian Distance Covariance for Few-Shot Classification*
- Official repo: `https://github.com/Fei-Long121/DeepBDC`
- Paper/README URL: `https://github.com/Fei-Long121/DeepBDC`

Official repo states that DeepBDC learns representations by measuring the discrepancy between:

- the joint distribution of embedded features, and
- the product of marginals.

Why it matters:

- DeepBDC improves the feature representation itself by emphasizing **joint distribution structure**.
- It is a strong reminder that richer second-order or distributional statistics can be useful before any classifier is applied.

Lesson for redesign:

- SC-LFI should not rely only on first-order support summaries for context.
- The support context block should expose **multi-component statistics**, not just a pooled mean.

## 5. Family C: Probabilistic and Distributional Class Modeling

### 5.1 Gaussian Prototypical Networks

Paper:
- 2018, *Gaussian Prototypical Networks for Few-Shot Learning on Omniglot*
- URL: `https://arxiv.org/abs/1708.02735`

Core idea:

- each embedding carries uncertainty;
- class distances are weighted using confidence-aware covariance structure.

Why it matters:

- This is an early but important signal that few-shot classification should model **confidence**, not only location.
- In `1-shot`, uncertainty is central.

Lesson for redesign:

- evidence masses should not only indicate importance;
- they should also induce **shot-aware uncertainty control** in the class distribution and in the scoring metric.

### 5.2 MetaQDA

Paper and official repo:
- ICCV 2021, *Shallow Bayesian Meta Learning for Real-World Few-Shot Recognition*
- Official repo: `https://github.com/Open-Debin/Bayesian_MQDA`
- Paper URL: `https://openaccess.thecvf.com/content/ICCV2021/papers/Zhang_Shallow_Bayesian_Meta_Learning_for_Real-World_Few-Shot_Recognition_ICCV_2021_paper.pdf`

The paper explicitly frames few-shot classification as inferring a **distribution over classifier parameters given the support set**.
It emphasizes:

- fast classifier-space meta-learning;
- calibration and uncertainty;
- strong cross-domain robustness.

Why MetaQDA matters:

- It is a clean probabilistic alternative to neural over-parameterization.
- It shows that in low-shot regimes, better uncertainty modeling often matters more than a more expressive nonlinear head.

Lesson for redesign:

- SC-LFI should carry explicit uncertainty or confidence signals tied to shot count and support dispersion.
- The score should not be dominated by aggressive auxiliary losses that force brittle certainty from a single support example.

### 5.3 Distribution Calibration

Paper and official repo:
- ICLR 2021 Oral, *Free Lunch for Few-Shot Learning: Distribution Calibration*
- Official repo: `https://github.com/ShuoYang-1998/Few_Shot_Distribution_Calibration`
- Paper URL: `https://openreview.net/forum?id=JWOiYxMG92s`

Core idea:

- estimate a better feature distribution for novel classes by transferring distribution statistics from related base classes;
- then sample calibrated features for classifier fitting.

Why it matters:

- Distribution calibration directly targets one of the central few-shot problems:
  the empirical support distribution is too poor to estimate a robust class distribution from scratch.

Lesson for redesign:

- a flow-based class distribution should not start from a completely unconstrained noise prior;
- it should start from a **support-anchored and base-informed prior**;
- support-only evidence in `1-shot` is too weak to define a reliable class distribution without prior structure.

### 5.4 DPGN

Paper:
- CVPR 2020, *DPGN: Distribution Propagation Graph Network for Few-Shot Learning*
- URL: `https://arxiv.org/abs/2003.14247`

Abstract contribution:

- explicitly models **distribution-level relation** of one example to all other examples in the task;
- combines distribution-level and instance-level relations.

Why it matters:

- DPGN reinforces that episode reasoning should include more than local pairwise similarity.
- Few-shot tasks have structure at the distribution level.

Lesson for redesign:

- SC-LFI should include episode-level coupling, not just per-class independent flow generation.

### 5.5 MixtFSL

Paper:
- ICCV 2021, *Mixture-Based Feature Space Learning for Few-Shot Image Classification*
- Paper URL: `https://openaccess.thecvf.com/content/ICCV2021/papers/Afrasiyabi_Mixture-Based_Feature_Space_Learning_for_Few-Shot_Image_Classification_ICCV_2021_paper.pdf`

Core idea:

- model base classes with multimodal mixtures in feature space;
- learn the feature extractor and mixture components jointly and online;
- avoid the unimodal assumption that one class should be one point.

Why it matters:

- MixtFSL is directly aligned with the claim that class structure is multimodal.
- It is stronger than prototype thinking, but still disciplined.

Lesson for redesign:

- SC-LFI should allow **multi-component support-conditioned class distributions** as the default, not as an optional decoration.
- The degenerate prototype case should remain available, but only as a limiting case.

## 6. Family D: Query-Batch Distribution Alignment / Transductive Few-Shot

### 6.1 LaplacianShot

Paper and official repo:
- ECCV 2020, *LaplacianShot: Laplacian Regularized Few-Shot Learning*
- Official repo: `https://github.com/imtiazziko/LaplacianShot`
- URL: `https://imtiazziko.github.io/publication/laplacianshot/`

Core idea:

- start from support-based predictions;
- refine query predictions using graph smoothness over the query manifold.

Why it matters:

- It exploits the structure of the unlabeled query set.
- In very low-shot regimes, query-batch geometry provides information missing from support.

Lesson for redesign:

- SC-LFI should expose an **optional transductive refinement head**.
- This should be optional, because episodic evaluation may be inductive in some benchmarks.

### 6.2 BECLR with OpTA

Paper/project:
- ICLR 2024, *BECLR: Batch Enhanced Contrastive Few-Shot Learning*
- Project URL: `https://stypoumic.github.io/BECLR/`
- Paper PDF already inspected during survey.

Key point from the paper/project:

- proposes **Optimal Transport-based Distribution Alignment (OpTA)**;
- aligns support prototypes and query features to reduce sample bias;
- authors note that this is especially helpful in low-shot scenarios.

Why it matters:

- It directly targets the exact pathology we are seeing:
  `1-shot` sample bias.
- It says that support and query should not be treated as independent clouds when there is obvious episode-level shift.

Lesson for redesign:

- even if the main classifier is inductive, the architecture should optionally support **query-to-class distribution alignment** before final scoring.
- This should be a refinement layer, not the only classification mechanism.

### 6.3 Transductive CLIP / EM-Dirichlet

Paper and official repo:
- CVPR 2024, *Transductive Zero-Shot and Few-Shot CLIP*
- Official repo: `https://github.com/SegoleneMartin/transductive-CLIP`

Official repo emphasizes:

- classifying groups of unlabeled images together;
- probability-feature classification;
- EM-like optimization with Dirichlet or Gaussian assumptions.

Why it matters:

- This is another recent sign that **distribution-level query batching** remains a strong path, even in the CLIP era.
- The reusable lesson is not CLIP itself.
- The reusable lesson is that few-shot accuracy can improve when the model reasons over the **query distribution jointly**.

Lesson for redesign:

- SC-LFI should keep an inductive core score, but leave room for a transductive refinement module if query batches are available.

### 6.4 VDC + CMDA

Paper:
- CVPR 2023, *Few-Shot Learning with Visual Distribution Calibration and Cross-Modal Distribution Alignment*
- Paper URL: `https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_Few-Shot_Learning_With_Visual_Distribution_Calibration_and_Cross-Modal_Distribution_Alignment_CVPR_2023_paper.pdf`

Core idea:

- calibrate visual distributions to suppress class-irrelevant clutter;
- align visual and language feature distributions with EMD.

Why it matters:

- Even though it is built on a vision-language setup, the important reusable idea is:
  **few-shot models overfit to clutter unless the evidence distribution is actively cleaned or calibrated**.

Lesson for redesign:

- token masses should reflect not just saliency, but **class-relevant reliability**;
- evidence weighting needs explicit regularization so that one noisy support image does not dominate the whole class distribution.

## 7. What Strong Few-Shot Models Consistently Do Better Than SC-LFI-v2

Across these papers, a clear set of common principles appears.

### 7.1 They keep support structure alive

Strong models do not immediately collapse support into one vector.
They keep at least one of the following:

- local descriptor pools;
- reconstruction bases;
- mixture components;
- probabilistic classifier parameters;
- query-conditioned support weights.

`sc_lfi_v2` still collapses too much structure too early.

### 7.2 They make the episode matter

FEAT, DPGN, LaplacianShot, BECLR, and transductive CLIP all make use of task-level or query-batch structure.
This is important because few-shot classification is not just classwise estimation.
It is an **episode inference problem**.

`sc_lfi_v2` remains too classwise and too support-only.

### 7.3 They treat `1-shot` as a special regime

The best `1-shot` methods do not behave as though a single support image is a stable empirical class distribution.
They add:

- stronger priors,
- uncertainty modeling,
- task adaptation,
- query-conditioned matching,
- or transductive refinement.

This is exactly the missing discipline in the current SC-LFI family.

### 7.4 They use weighting for a reason

DeepEMD, CAN, VDC, and transductive methods all imply that not all tokens or regions are equally useful.
But the weights are not arbitrary.
They are tied to:

- query relevance,
- structural matching,
- clutter suppression,
- or uncertainty.

This means SC-LFI token masses should not be static attention scores only.
They should become **query-conditioned reliability weights** at scoring time.

## 8. Redesign Consequences for the Next SC-LFI

The survey implies the next model should not be `sc_lfi_v2` with a few regularizers.
It should be a new architecture.

### 8.1 What to keep

- the core claim that class scoring is a distribution-fit score;
- the use of transport-based or distribution-based distance;
- the idea that support-conditioned latent evidence distributions can outperform point prototypes.

### 8.2 What to change fundamentally

#### A. Replace "single summary conditioned flow" with support-basis conditioned residual flow

The flow should not generate class particles from generic noise conditioned only on `h_c`.
Instead:

- build a support basis `B_c` from local support evidence atoms;
- optionally adapt it at the episode level;
- generate residual particles around support atoms or transport support atoms through a residual field.

This makes the flow genuinely few-shot-specific.

#### B. Add query-conditioned evidence selection

For each query-class pair:

- compute cross-reference reliability on support memory atoms;
- reweight support atoms using query-conditioned compatibility;
- then form the class measure used in the transport score.

This is the single most important lesson from DeepEMD, CAN, and FRN.

#### C. Make the class distribution multi-component by construction

The class measure should contain:

- support anchor atoms;
- adapted support memory atoms;
- optional residual flow particles;
- all with learned masses.

Prototype mode should exist only as a degenerate special case.

#### D. Use shot-aware priors and losses

`1-shot` and `5-shot` should not share the same support mixing pressure.
In `1-shot`:

- stronger uncertainty prior;
- weaker direct support-fit forcing;
- stronger episode adaptation;
- possibly stronger query-conditioned weighting.

#### E. Add optional transductive refinement

If the evaluation protocol allows it:

- refine class/query distributions using query-batch geometry or OT alignment.

This should be optional and isolated from the inductive core.

## 9. Recommended New Architecture Direction

The survey supports the following conceptual architecture:

### Stage 1: Local evidence extraction

- backbone outputs image tokens `Z(x)`;
- latent projector maps them to evidence atoms `E(x)`;
- reliability head produces base masses.

### Stage 2: Episode adaptation

- FEAT-style lightweight set-to-set adaptation over class summaries or compact class memories;
- no heavy transformer;
- the goal is task conditioning, not large-model capacity.

### Stage 3: Support basis construction

For each class:

- keep a compact set of support evidence atoms or memory slots;
- keep support atom masses;
- keep a support uncertainty summary.

### Stage 4: Query-conditioned class measure assembly

For each query-class pair:

- compute query-to-support cross-reference weights;
- produce a query-conditioned reweighted support measure;
- optionally add a residual flow-generated measure around the reweighted support basis.

### Stage 5: Distribution-fit scoring

- score with weighted OT / weighted sliced Wasserstein over local evidence atoms;
- allow separate distances for class scoring and support anchoring.

### Stage 6: Optional transductive refinement

- if allowed, refine predictions using query-batch graph or OT alignment.

## 10. Concrete Implication for SC-LFI Naming

The next architecture should be a new model, for example:

- `sc_lfi_fs`
- `sc_lfi_a`
- `sc_lfi_v3`

It should not overwrite `sc_lfi_v2`.

The name should reflect that the model is now:

- support-basis conditioned,
- query-conditioned during scoring,
- and explicitly designed around few-shot structure rather than generic latent generation.

## 11. Final Judgment

If the goal is an `A*`-quality few-shot design, the next SC-LFI must not be defined as:

- "a class summary vector plus a stronger flow plus a stronger transport metric."

That is still too generic.

It should instead be defined as:

- **a few-shot support-basis model whose class distribution is assembled from local support evidence, query-conditioned reliability, and residual support-aware transport.**

That framing is consistent with the strongest ideas from:

- FEAT
- DeepEMD
- FRN
- DeepBDC
- MetaQDA
- Distribution Calibration
- DPGN
- MixtFSL
- LaplacianShot
- BECLR / OpTA
- Transductive CLIP
- VDC + CMDA

This is the correct scientific direction for a new redesign.

## 12. References

- FEAT repo: `https://github.com/Sha-Lab/FEAT`
- FEAT paper: `https://openaccess.thecvf.com/content_CVPR_2020/papers/Ye_Few-Shot_Learning_via_Embedding_Adaptation_With_Set-to-Set_Functions_CVPR_2020_paper.pdf`
- DeepEMD repo: `https://github.com/icoz69/DeepEMD`
- DeepEMD repo README / abstract: `https://github.com/icoz69/DeepEMD`
- FRN repo: `https://github.com/Tsingularity/FRN`
- FRN paper: `https://arxiv.org/abs/2105.13009`
- DeepBDC repo: `https://github.com/Fei-Long121/DeepBDC`
- MetaQDA repo: `https://github.com/Open-Debin/Bayesian_MQDA`
- MetaQDA paper: `https://openaccess.thecvf.com/content/ICCV2021/papers/Zhang_Shallow_Bayesian_Meta_Learning_for_Real-World_Few-Shot_Recognition_ICCV_2021_paper.pdf`
- Distribution Calibration repo: `https://github.com/ShuoYang-1998/Few_Shot_Distribution_Calibration`
- Distribution Calibration paper: `https://openreview.net/forum?id=JWOiYxMG92s`
- DPGN paper: `https://arxiv.org/abs/2003.14247`
- MixtFSL paper: `https://openaccess.thecvf.com/content/ICCV2021/papers/Afrasiyabi_Mixture-Based_Feature_Space_Learning_for_Few-Shot_Image_Classification_ICCV_2021_paper.pdf`
- LaplacianShot project: `https://imtiazziko.github.io/publication/laplacianshot/`
- LaplacianShot repo: `https://github.com/imtiazziko/LaplacianShot`
- BECLR project: `https://stypoumic.github.io/BECLR/`
- BECLR paper PDF inspected during survey: `https://proceedings.iclr.cc/paper_files/paper/2024/file/08309150af77fc7c79ade0bf8bb6a562-Paper-Conference.pdf`
- Transductive CLIP repo: `https://github.com/SegoleneMartin/transductive-CLIP`
- VDC + CMDA paper: `https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_Few-Shot_Learning_With_Visual_Distribution_Calibration_and_Cross-Modal_Distribution_Alignment_CVPR_2023_paper.pdf`
- Gaussian Prototypical Networks: `https://arxiv.org/abs/1708.02735`
