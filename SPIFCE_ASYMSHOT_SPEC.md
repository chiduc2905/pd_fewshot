# SPIFCE-AsymShot: An Ablation-Grounded Redesign of SPIFCE

## 1. Purpose

This document specifies a new SPIF-family model derived from the current
SPIFCE ablation results in `pulse_fewshot`.

The target is not to "add more modules" to SPIFCE. The target is to redesign
SPIFCE so that:

- the strongest parts of the current model are kept,
- the parts that empirically conflict with each other are decoupled,
- the local branch becomes structurally correct for multi-shot episodes,
- the final model remains light, CE-only, and paper-friendly.

The proposed model name is:

`SPIFCE-AsymShot`

Short meaning:

- `Asym`: asymmetric global-local representation design
- `Shot`: shot-preserving local matching and shot-level aggregation

## 2. What The Current Ablation Actually Says

The current SPIFCE ablation suite was run in the 1-shot setting with
`spif_top_r = 3` over four training-sample regimes: `60`, `160`, `240`, and
`all`.

The important observations are:

### 2.1 The full current SPIFCE is not the best model across regimes

- `60 samples`: `base_shared_proto = 0.8183`, `full = 0.8058`
- `160 samples`: `full = 0.8925`, best on this split
- `240 samples`: `local_module_only = 0.9375`, `full = 0.8967`
- `all samples`: `gate_only = 0.9692`, `base_shared_proto = 0.9658`,
  `full = 0.9417`

So the current "full SPIFCE" does not consistently dominate its own
sub-modules.

### 2.2 The global branch is already very strong

The best global-style variants are:

- `60`: `base_shared_proto`
- `160`: `factor_gate`
- `240`: `factor_gate`
- `all`: `gate_only`

This means the global branch is not the bottleneck. The global branch is
already a strong classifier.

### 2.3 The local module is genuinely useful, but not in the current coupled form

At `240 samples`, the strongest model is:

- `local_module_only = 0.9375`

This is a very strong signal that the local top-r partial matching branch is
valuable. However, when combined into the current full SPIFCE:

- `full - local_module_only = -0.0408` at `240 samples`

So the local branch is good, but the current way of coupling it with the rest
of SPIFCE is not.

### 2.4 The current local branch prefers a simpler representation than the global branch

Within the local-only family:

- `shared_local_branch_only` is the best local-only variant at `60`, `160`,
  and `240`
- `full_local_branch_only` is only marginally better at `all`

This means the current local matcher does **not** benefit consistently from
using the same factorized-and-gated representation that helps the global
branch.

This is the main architectural clue from the ablation.

## 3. Core Interpretation

The ablation strongly suggests that the current SPIFCE uses one representation
policy for two tasks that have different needs:

- the `global` branch wants denoising and stable evidence selection,
- the `local` branch wants faithful correspondence geometry.

The present SPIFCE forces both branches to consume the same gated stable token
space. That is likely the structural reason why:

- the global branch is strong by itself,
- the local branch is strong by itself in some regimes,
- but the fused full model is unstable across regimes.

Therefore the redesign principle is:

> Do not make global and local branches share the same token semantics if the
> ablation says they prefer different representation behavior.

## 4. Proposed Model

## 4.1 Name

`SPIFCE-AsymShot`

## 4.2 High-level idea

The model has:

1. one shared image backbone,
2. one `global-stable` head specialized for prototype classification,
3. one separate `local-correspondence` head specialized for token matching,
4. shot-preserving local matching,
5. query-adaptive fusion between global and local evidence.

The proposed novelty is not a heavy relation network. It is a minimal,
ablation-grounded structural redesign:

- asymmetric global-local evidence routing,
- shot-preserving local partial matching,
- query-adaptive confidence fusion.

## 5. Design Principles

### Principle A: Keep the global branch close to what already works

The global path should keep the parts that are consistently strong:

- class prototype reasoning,
- stable evidence gating,
- lightweight geometry,
- CE-only training.

### Principle B: Stop forcing the local branch to use the same representation policy

The local branch should not be forced to use the same factorized-and-gated
tokens as the global branch. The ablation suggests that local matching wants a
less distorted token geometry.

### Principle C: Preserve the shot dimension until after local matching

The support shot dimension should remain explicit during local score
computation. The current shot-collapsed local score is the wrong inductive bias
for multi-shot episodes.

### Principle D: Fusion should depend on branch reliability, not a fixed alpha

The strongest branch changes with data regime. Therefore the fusion weight
should be query-adaptive rather than fixed globally.

## 6. Model Specification

## 6.1 Shared backbone and tokenization

Given an image `x`, let the backbone produce a feature map and token sequence:

$$
T = \{t_i\}_{i=1}^{L}, \qquad t_i \in \mathbb{R}^{d}
$$

The backbone remains the standard few-shot encoder already used in
`pulse_fewshot`.

## 6.2 Global branch: gated stable prototype branch

The global branch uses a dedicated stable head:

$$
z_i^g = h_g(t_i)
$$

and predicts a token gate:

$$
a_i = \sigma(g(z_i^g)), \qquad a_i \in (0,1)
$$

The query global embedding is:

$$
q^g = \frac{\sum_{i=1}^{L} a_i z_i^g}{\sum_{i=1}^{L} a_i + \varepsilon}
$$

For class `c` with `K` support shots:

$$
p_c^g = \frac{1}{K}\sum_{s=1}^{K}
\frac{\sum_{i=1}^{L} a_{csi} z_{csi}^g}{\sum_{i=1}^{L} a_{csi} + \varepsilon}
$$

The class score is cosine similarity:

$$
s_c^g = \tau_g \cdot \cos(q^g, p_c^g)
$$

Notes:

- this is the branch that keeps the SPIF stable-gate intuition,
- this branch may keep factorization,
- this branch stays light and mostly geometric.

## 6.3 Local branch: separate correspondence head

The local branch uses a different head:

$$
z_i^\ell = h_\ell(t_i)
$$

where `h_\ell` is **not** forced to share the same semantics as `h_g`.

Recommended design:

- lightweight LayerNorm -> Linear -> GELU -> Linear head,
- output L2-normalized tokens,
- no stable gate multiplication before local matching,
- no variant branch required in the first implementation.

This is the key asymmetric decision:

- global branch gets `gated stable tokens`,
- local branch gets `matching-friendly tokens`.

## 6.4 Shot-preserving local top-r matching

For a query `q`, class `c`, and support shot `s`, define the local score:

$$
\ell_{qcs} =
\frac{1}{L_q}
\sum_{u=1}^{L_q}
\operatorname{meanTopR}_{v \in \mathcal{S}_{cs}}
\left\langle z_{q,u}^{\ell}, z_{cs,v}^{\ell}\right\rangle
$$

where:

- `u` indexes query tokens,
- `v` indexes tokens inside one support image,
- `R` is the top-r hyperparameter.

This preserves the episode hierarchy:

- token-level alignment happens inside each support image,
- shot-level aggregation happens after per-shot matching.

For `1-shot`, this naturally reduces to a single local score.

## 6.5 Shot aggregation

Let the per-shot local scores be:

$$
\ell_{qc1}, \dots, \ell_{qcK}
$$

Two shot aggregations should be implemented.

### Mean aggregation

$$
s_c^\ell = \frac{1}{K}\sum_{s=1}^{K}\ell_{qcs}
$$

### Query-adaptive softmax aggregation

Because `\ell_{qcs}` is a similarity score, the aggregation should favor
larger values:

$$
\pi_{qcs} = \operatorname{softmax}_s(\beta \cdot \ell_{qcs})
$$

$$
s_c^\ell = \sum_{s=1}^{K} \pi_{qcs}\,\ell_{qcs}
$$

where `beta > 0` controls shot selectivity.

This is the main method. `mean` remains the clean ablation baseline.

## 6.6 Query-adaptive branch fusion

The current SPIFCE uses a fixed alpha:

$$
s_c = \alpha s_c^g + (1-\alpha)s_c^\ell
$$

The ablation says this is too rigid. The proposed model replaces this with a
query-adaptive fusion weight derived from branch confidence.

Define the top-1 minus top-2 margin for each branch:

$$
m_q^g = s_{q,(1)}^g - s_{q,(2)}^g
$$

$$
m_q^\ell = s_{q,(1)}^\ell - s_{q,(2)}^\ell
$$

Then define:

$$
\alpha_q = \sigma\big(\kappa (m_q^g - m_q^\ell)\big)
$$

and the final class logits:

$$
\text{logit}_{qc} = \alpha_q s_{qc}^g + (1-\alpha_q)s_{qc}^\ell
$$

Interpretation:

- if the global branch is more confident, trust it more,
- if the local branch is more confident, trust it more.

This remains parameter-light and fully episode-local.

## 7. What To Keep And What To Remove

### Keep

- shared backbone
- prototype-style global branch
- token gate for the global branch
- CE-only training as the default
- local top-r partial matching as the local mechanism

### Remove from the coupled design

- the assumption that the local branch must use the same gated stable tokens as
  the global branch
- the assumption that a single fixed `alpha` is adequate across all regimes
- the support-shot collapse before local matching

## 8. Why This Model Matches The Ablation

This redesign is directly implied by the ablation:

### 8.1 Why keep the gate in the global branch

Because `gate_only` is the best result at `all samples`, and the global-only
family is consistently strong.

### 8.2 Why not force factorization into the local branch

Because the local-only family prefers `shared_local_branch_only` over the
factorized local-only variants on `60`, `160`, and `240`.

### 8.3 Why preserve shots in the local branch

Because the current local branch collapses shots before matching, which is a
bad structural prior for multi-shot episodes and already showed weaknesses in
the current SPIF-family variants.

### 8.4 Why use adaptive fusion

Because the best branch changes by regime:

- low-data: simpler global branch wins,
- medium-data: full fusion helps,
- higher-data: local or gate-heavy variants dominate.

This is exactly the scenario where fixed fusion is too crude.

## 9. Novelty Positioning

The paper claim should be framed carefully.

A safe claim is:

> We identify that the current SPIFCE contributions are individually useful but
> not optimally coupled. We redesign the model with asymmetric global-local
> representation roles, shot-preserving local matching, and query-adaptive
> confidence fusion.

This is a strong and defensible claim because it is directly supported by the
ablation.

Do **not** claim "no prior paper has ever done anything similar" without a
separate literature sweep. The correct paper language is:

- "to our knowledge" only after related-work verification,
- and preferably emphasize the specific combination and the ablation-grounded
  reasoning rather than an absolute novelty claim.

## 10. Implementation Plan

Recommended new module:

- `net/spifce_asym_shot.py`

Recommended constructor arguments:

- `global_dim`
- `local_dim`
- `gate_hidden`
- `top_r`
- `shot_agg` in `{mean, softmax}`
- `shot_softmax_beta`
- `global_scale`
- `local_scale`
- `fusion_mode` in `{fixed, margin_adaptive}`
- `fusion_kappa`
- `learnable_global_scale` optional, default `false`
- `learnable_local_scale` optional, default `false`

Recommended diagnostics in `return_aux=True`:

- `global_scores`
- `local_scores`
- `per_shot_local_scores`
- `shot_aggregation_weights`
- `alpha_q`
- `global_margin`
- `local_margin`

## 11. Training Protocol

The first version should remain conservative:

- episodic cross-entropy only
- no new auxiliary losses
- no cross-attention
- no transport solver
- no memory bank
- no transductive adaptation

The first experimental version should isolate the architecture itself.

## 12. Primary Ablations For The New Model

The new model should be evaluated with the following ablations:

1. `global gate + fixed alpha + pooled local`
2. `global gate + fixed alpha + shot-preserving mean local`
3. `global gate + fixed alpha + shot-preserving softmax local`
4. `global gate + margin-adaptive alpha + shot-preserving softmax local`
5. `shared local head` vs `separate local head`

These ablations test the three actual claims:

- asymmetric representation helps,
- shot-preserving local matching helps,
- adaptive fusion helps.

## 13. Expected Outcome

The desired empirical pattern is:

- match or exceed `base_shared_proto` at very low data,
- match or exceed `full` at medium data,
- keep the strong local behavior seen in `local_module_only`,
- improve multi-shot robustness over the current shot-collapsed local SPIFCE,
- avoid the regime instability seen in the present full model.

## 14. Final Summary

The ablation does **not** support keeping the current full SPIFCE unchanged and
simply tuning it harder. It supports a cleaner architectural conclusion:

- the global branch is strong and should stay gated,
- the local branch is useful but should not share the same token policy,
- multi-shot local reasoning should preserve shots,
- global-local fusion should be query-adaptive.

That leads to a single concrete model proposal:

`SPIFCE-AsymShot = gated stable global prototype branch + separate
shot-preserving local correspondence branch + margin-adaptive fusion`

This is the most defensible next-step model after the current SPIFCE ablation.
