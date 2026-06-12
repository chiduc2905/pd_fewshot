# Minimal Rival-Aware Evidence UOT

## Motivation

A cheap local match is not necessarily class evidence. Background or noise can
be cheap for several candidate classes. The minimal RAE-UOT rule is:

> Use episode-rival evidence only to decide where UOT receives marginal mass.

It replaces EGSM. It does not modify the ground cost and does not replace the
original Ours-Final score. It is not combined with ECT-UOT, RV-UOT,
residual-aligned UOT, pulse masks, or other local guidance paths.

## Rival specificity

For local ground cost \(D_{c,s,i,j}\), first compute a soft minimum over the
support tokens in each rival shot:

\[
d_{c',s',i}
=-\tau\log\sum_{j'}\exp(-D_{c',s',i,j'}/\tau).
\]

Aggregate rival class-shot pairs with log-mean-exp:

\[
\bar D^{-c}_{i}
=-\tau\left[
\log\sum_{c'\ne c,s'}\exp(-d_{c',s',i}/\tau)
-\log N^{-c}
\right].
\]

The normalization makes the reference invariant to duplicating identical
rival shots, avoiding a systematic 1-shot versus 5-shot shift.

\[
g_{c,s,i,j}
=\sigma((\bar D^{-c}_{i}-D_{c,s,i,j})/\tau),
\qquad
r_{c,s,i,j}=\max(2g_{c,s,i,j}-1,0).
\]

An equal-cost candidate/rival match has zero specificity.

## Bidirectional pair marginals

Let \(q_i,z_{c,s,j}\) be normalized local descriptors and
\(\bar q,\bar z_{c,s}\) their pooled descriptors:

\[
x^q_{c,s,i}=[q_i^\top\bar z_{c,s}]_+,
\qquad
x^s_{c,s,j}=[z_{c,s,j}^\top\bar q]_+.
\]

Cross-reference and rival specificity define non-negative evidence:

\[
e^q_{c,s,i}=x^q_{c,s,i}\max_j r_{c,s,i,j},
\qquad
e^s_{c,s,j}=x^s_{c,s,j}\max_i r_{c,s,i,j}.
\]

Each evidence vector is normalized directly. If its total evidence is zero,
the marginal falls back exactly to uniform:

\[
a=\begin{cases}
e^q/\sum_i e^q_i,&\sum_i e^q_i>0,\\
u,&\text{otherwise},
\end{cases}
\qquad
b=\begin{cases}
e^s/\sum_j e^s_j,&\sum_j e^s_j>0,\\
u,&\text{otherwise}.
\end{cases}
\]

A uniform mixture provides a fixed exploration floor:

\[
\tilde a=(1-\lambda)u+\lambda a,\qquad
\tilde b=(1-\lambda)u+\lambda b.
\]

The only tunable values are rival temperature \(\tau\) and marginal mixture
\(\lambda\).

## Transport and score

RAE-UOT changes only the two marginals:

\[
P=\operatorname{UOT}(D,\tilde a,\tilde b,\rho=0.8).
\]

The ground cost and original Ours-Final local score remain unchanged:

\[
S_{\mathrm{local}}=T\,M(P)-\langle P,D\rangle.
\]

The existing global residual is optional:

\[
S=S_{\mathrm{local}}+0.1S_{\mathrm{global}}.
\]

The evidence determines where mass is offered, UOT determines which offered
mass is transported, and the original score evaluates the resulting plan.
The rival signal is therefore used once rather than repeated in the marginal,
ground cost, and score.

## Bias and falsification

RAE-UOT receives no pixel intensity, fixed time coordinate, frequency mask, or
hand-selected PD window. Token-position permutation equivariance verifies that
the mechanism itself has no fixed time prior. This does not prove that the
backbone cannot learn dataset artifacts.

The method should be rejected if:

1. Identical rival-shot duplication changes the rival gate.
2. Common-only evidence does not return uniform marginals.
3. Clean/noise accuracy or cross-seed stability is worse than the global
   residual baseline.
4. Visualization improves without corresponding classification improvement.

## Novelty boundary

Cross-reference weighting alone is not novel; DeepEMD already uses the other
image to weight local descriptors. The candidate contribution is the complete
combination of:

- bidirectional query-conditioned and support-conditioned pair marginals;
- episode-rival-normalized specificity with shot-count invariance;
- exact uniform fallback for episodes without class-specific local evidence;
- UOT mass rejection while retaining the original ground cost and score.

Novelty should be claimed only after ablation and literature comparison.
