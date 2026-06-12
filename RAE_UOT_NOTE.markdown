# Rival-Aware Evidence UOT (RAE-UOT)

## Motivation

A locally cheap match is not necessarily useful evidence. In a scalogram, a
background or noise pattern can be cheap for several candidate classes. Plain
UOT may retain that match because it only sees the candidate pair, while a
few-shot classifier needs evidence that is cheap for one class relative to the
other classes in the same episode.

RAE-UOT defines one local decision principle:

> Transport mass only where query and support cross-reference each other and
> the match is more compatible with the candidate class than with its rivals.

It replaces the default EGSM marginal. It is not stacked with ECT-UOT,
RV-UOT, residual-aligned UOT, pulse masks, or other local verification paths.
The global residual remains an optional, separate class-level stabilizer.

## Rival-normalized specificity

For query token \(i\), candidate class \(c\), shot \(s\), and support token
\(j\), let \(D_{c,s,i,j}\) be the local ground cost. The rival reference is
computed in two steps. First, take a soft minimum over support tokens inside
each rival shot:

\[
d_{c',s',i}
=-\tau\log\sum_{j'}\exp\left(-D_{c',s',i,j'}/\tau\right).
\]

Then aggregate rival class-shot pairs with a normalized soft minimum:

\[
\bar D^{-c}_{i}
=-\tau\left[
\log\sum_{c'\ne c,s'}\exp\left(-d_{c',s',i}/\tau\right)
-\log N^{-c}
\right].
\]

The \(-\log N^{-c}\) term makes the reference a log-mean-exp. Duplicating an
identical rival shot therefore does not change the gate. The inner soft
minimum still preserves an isolated cheap rival token instead of averaging it
away. This removes a systematic 1-shot versus 5-shot bias without hiding
common cheap matches.

\[
g_{c,s,i,j}
=\sigma\left((\bar D^{-c}_{i}-D_{c,s,i,j}-m)/\tau\right),
\qquad
r_{c,s,i,j}=\max(2g_{c,s,i,j}-1,0).
\]

An equal-cost candidate/rival match has \(g=0.5\) and \(r=0\). It is treated
as common evidence rather than class evidence.

## Bidirectional cross-reference marginals

Let \(q_i\) and \(z_{c,s,j}\) be normalized local descriptors. Let
\(\bar q\) and \(\bar z_{c,s}\) be their pooled descriptors.

\[
x^q_{c,s,i}=[q_i^\top\bar z_{c,s}]_+,
\qquad
x^s_{c,s,j}=[z_{c,s,j}^\top\bar q]_+.
\]

These are the symmetric cross-reference terms learned from DeepEMD's useful
observation: the other image should determine which local descriptors receive
mass. RAE-UOT adds episodic class competition:

\[
a_{c,s,i}\propto
\exp\left(
\log(x^q_{c,s,i}\,[\max_j r_{c,s,i,j}+\epsilon])/\tau_m
\right),
\]

\[
b_{c,s,j}\propto
\exp\left(
\log(x^s_{c,s,j}\,[\max_i r_{c,s,i,j}+\epsilon])/\tau_m
\right).
\]

Both marginals are pair-specific. A uniform mixture keeps a non-zero
exploration floor:

\[
\tilde a=(1-\lambda)u+\lambda a,\qquad
\tilde b=(1-\lambda)u+\lambda b.
\]

## One evidence variable for transport and score

The pair evidence is

\[
E_{c,s,i,j}
=r_{c,s,i,j}
\sqrt{\hat a_{c,s,i}\hat b_{c,s,j}},
\]

where juxtaposition denotes multiplication: the implemented expression is the
rival specificity times the geometric mean of peak-normalized marginal
probabilities.

The same evidence raises the cost of unsupported matches:

\[
D' = D+\gamma\,\operatorname{std}(D)\,(1-E).
\]

UOT with fixed transported budget \(\rho=0.8\) is then solved using
\(\tilde a,\tilde b,D'\). The local class score is the negative mean evidence
transport cost:

\[
S_{\text{local}}
=-\frac{\langle P,D'E\rangle}
{\langle P,E\rangle+\epsilon}.
\]

With the existing global residual:

\[
S=S_{\text{local}}+0.1S_{\text{global}}.
\]

The local branch answers "where is candidate-specific correspondence?" The
global branch answers "is the episode-level class prototype compatible?" They
operate at different statistical scales.

## Why this is not brightness/time bias

RAE-UOT receives no pixel intensity, fixed time coordinate, frequency mask, or
hand-selected PD window. Its decisions use learned feature compatibility and
relative class competition inside each episode. Translating a discharge event
does not directly change the rule. The uniform marginal floor also prevents a
single time region from becoming a hard prior.

This does not prove dataset independence. It creates falsifiable checks:

1. Rival gates must remain stable when identical rival shots are duplicated.
2. Common cheap matches must receive lower evidence than class-specific cheap
   matches in controlled tensors.
3. Evidence concentration and accuracy must be reported under time shift,
   noise, and clean protocols.
4. The method is rejected if visualization improves but clean/noise accuracy
   or cross-seed stability does not.

## Novelty boundary

The cross-reference idea alone is not novel; DeepEMD already uses the other
image to weight local descriptors. The candidate contribution is the unified
combination of:

- bidirectional query-conditioned and support-conditioned pair marginals;
- episode-rival-normalized specificity with shot-count invariance;
- unbalanced transport that can discard unsupported mass;
- one evidence variable shared by marginal construction, ground-cost shaping,
  and mass-normalized local scoring.

Novelty should be claimed for this complete local decision rule only after
ablation and literature checks, not for any individual component.
