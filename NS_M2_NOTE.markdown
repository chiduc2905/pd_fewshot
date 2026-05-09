# NS-M2: Noise-Sink Single-Budget ECOT

NS-M2 is the preferred novelty direction for the J-ECOT-M2 line. The reason is
targeted: M2 is already competitive on clean testing and mainly loses to
DeepEMD on noise-test episodes. Token Saliency Reweighting (`--hrot_tsw_enable
true`) is therefore a poor headline direction because it adds an unanchored
token gate to the calibrated M2 cost table, changes the cost scale before OT,
and makes the architecture look heavier without addressing the specific noise
failure mode.

NS-M2 instead keeps the M2 backbone, fixed rho budget, threshold mass-cost
score, and episode-calibrated shot pooling. The only active change is the
transport problem solved inside each fixed-budget ECOT expert.

## Motivation

J-ECOT-M2 performs well on clean episodes because it uses a stable fixed
partial budget and avoids high-variance token-mass policies. Its weakness under
SNR-shifted noise is that every real query/support token still belongs to the
same real-token transport table. Unbalanced OT can reduce total transported
mass, but there is no explicit destination for unmatched background regions.

DeepEMD is strong under clutter because its local matching can avoid letting
background regions dominate the class score. NS-M2 targets the same failure
mode with a different, lower-variance mechanism: a dustbin/noise sink in the
ECOT solve.

The theoretical basis is robust partial/unbalanced transport. In noisy
scalograms, not every local region should be forced to explain a class. A null
destination gives unmatched query/support mass a bounded-cost route outside the
real-token matching table. The classifier still scores only real-to-real
transport, so sink mass behaves as rejected evidence rather than positive class
evidence.

## Mechanism

For every budget rho in the ECOT bank, the usual real-token marginals are built:

```text
a_real = rho * p_query
b_real = rho * p_support
```

where `p_query` and `p_support` are uniform in M2 unless another enabled ECOT
component supplies them.

NS-M2 appends one sink token to each side:

```text
a_aug = [a_real, 1 - rho]
b_aug = [b_real, 1 - rho]
```

and augments the cost matrix:

```text
C_aug =
[
  C_real      sink_cost
  sink_cost   0
]
```

The ECOT solver runs on `C_aug`, but the classifier scores only the real-token
block:

```text
P_real = P_aug[:-1, :-1]
C_score = <P_real, C_real>
M_score = sum(P_real)
E = score_scale * (threshold * M_score - C_score)
```

Thus sink transport is allowed to absorb unmatched tokens, but it is not counted
as positive class evidence.

## Why This Is Low Risk For Clean Accuracy

The old M2 path remains the default because `hrot_ecot_enable_noise_sink=false`.
When enabled, NS-M2 does not change the encoder, support-shot decomposition,
rho bank, threshold rule, or shot-pooling controller. The score still rewards
only low-cost real-token transported mass. On clean episodes, useful real-token
matches should remain cheap; on noisy episodes, unmatched regions have an
explicit sink route instead of forcing accidental real-token matches.

## CLI

Single-budget M2 with the noise sink:

```bash
python main.py \
  --model hrot_fsl \
  --hrot_variant J_ECOT \
  --hrot_ecot_rho_bank 0.80 \
  --hrot_ecot_base_rho 0.80 \
  --hrot_ecot_enable_noise_sink true
```

Ablation alias:

```bash
python run_jecot_ablation.py --variants m2_noise_sink
```

Recommended comparison against the actual weakness:

```bash
python run_jecot_ablation.py \
  --variants m2_baseline,m2_noise_sink \
  --test_protocol noise
```

Keep the old TSW variants only as negative controls:

```bash
python run_jecot_ablation.py --variants abl_m2_tsw,abl_m2_tsw_split
```

## Diagnostics

When `return_aux=True`, the path exposes:

```text
ecot_noise_sink_query_mass
ecot_noise_sink_support_mass
ecot_noise_sink_self_mass
ecot_noise_sink_query_mass_bank
ecot_noise_sink_support_mass_bank
ecot_noise_sink_self_mass_bank
ecot_noise_sink_cost
ecot_noise_sink_score_penalty
```

These should be checked against clean/noise splits. A useful sign is low sink
mass on clean episodes and higher sink mass on low-SNR episodes, without a
collapse of real-token transported mass for the true class.

## Claim Boundary

Safe claim:

```text
NS-M2 adds explicit noise-sink unbalanced transport to fixed-budget ECOT,
allowing unmatched scalogram regions to be absorbed outside the real-token
matching table while preserving M2's threshold-calibrated real-evidence score.
```

Do not claim a new OT solver, learned token reliability, or DeepEMD-style
cross-reference weighting.

Paper framing:

```text
Unlike token-gating extensions, NS-M2 modifies the transport feasible object
rather than the encoder or cost network: it augments each fixed-budget ECOT
expert with a null sink and scores only the real-token block. This gives noisy
scalogram regions an explicit rejection route while preserving the stable
single-budget M2 decision rule.
```
