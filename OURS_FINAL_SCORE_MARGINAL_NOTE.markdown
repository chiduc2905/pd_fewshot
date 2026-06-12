# Ours-Final Score-Aligned Marginals

## Scope

This mechanism starts from the original `ours_final` path:

- the same local cost matrix \(D\);
- the same UOT solver and transported budget \(\rho=0.8\);
- the same score \(16(T\,M(P)-\langle P,D\rangle)\);
- no RAE, EGSM, cost guidance, verification, or alternative scoring.

Only the uniform query and support marginals are replaced.

## Shared Query Marginal

For class \(c\) and query token \(i\), compute a normalized soft minimum over
all shots and support tokens:

\[
d_{c,i}=-\tau\left[
\log\sum_{s,j}\exp(-D_{c,s,i,j}/\tau)-\log(KL_s)
\right].
\]

The normalization removes dependence on duplicated shots. Let

\[
\Delta_{c,i}=\min_{c'\ne c}d_{c',i}-d_{c,i},
\qquad
r_{c,i}=1-\exp(-[\Delta_{c,i}]_+/\tau).
\]

The query evidence is aligned with the positive per-edge utility \(T-D\):

\[
e_i^q=\max_c r_{c,i}\,
\sigma\left((T-\min_{c,s,j}D_{c,s,i,j})/\tau\right).
\]

After normalization, the same query probability \(\pi^q\) is used for every
candidate class. This is required because class logits must evaluate the same
query measure.

## Support Marginal

For each candidate support token:

\[
E_{c,s,i,j}=r_{c,i}
\sigma((T-D_{c,s,i,j})/\tau)
\exp(-(D_{c,s,i,j}-\min_{j'}D_{c,s,i,j'})/\tau),
\]

\[
e^s_{c,s,j}=\sum_i\pi_i^qE_{c,s,i,j}.
\]

The query and support evidence are normalized directly. If all evidence is
zero, the corresponding marginal falls back to uniform. A uniform floor is
controlled by

\[
\tilde\pi=(1-\lambda)u+\lambda\pi.
\]

## Falsification

The implementation logs marginal entropy, peak probability, uniform fallback
share, threshold acceptance, and post-UOT L1 drift. It must be compared
directly against the original uniform `ours_final` under the same seeds and
clean/noise protocols.
