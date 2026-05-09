# MEA-M2: Mutual-Evidence Attention M2

MEA-M2 is a compact attention extension for the fixed-budget J-ECOT-M2 path.
It avoids a new encoder, token gate, learned cost network, or extra transport
head.  The only learned quantities are a mixture strength and an attention
temperature.

## Motivation

The noise-sink route changes the feasible transport table by adding a null
token.  That can help low-SNR episodes, but it also changes how unmatched mass
is represented and can become sensitive to the sink cost.  MEA-M2 instead keeps
the original ECOT solve and score unchanged, and only replaces uniform token
marginals with an attention-derived prior over the real tokens.

## Definition

For each query/class/shot pair, let `C` be the same token-token cost table used
by ECOT.  MEA standardizes this cost per pair and forms bidirectional attention:

```text
A_s(j | i) = softmax_j(-standardize(C_ij) / tau)
A_q(i | j) = softmax_i(-standardize(C_ij) / tau)
```

The support and query token priors are the averaged attentions:

```text
p_s(j) = mean_i A_s(j | i)
p_q(i) = mean_j A_q(i | j)
```

Both are mixed with uniform mass:

```text
pi_s = (1 - eta) uniform_s + eta p_s
pi_q = (1 - eta) uniform_q + eta p_q
```

Every fixed-budget ECOT expert then uses:

```text
a_rho = rho pi_q
b_rho = rho pi_s
```

The UOT solver, threshold score, shot pooling, and M2 single-budget default
remain unchanged.

## Why This Is Defensible

MEA is an attention prior over the transport geometry, not a post-hoc logit
gate.  The attention table is induced by the same cost matrix that ECOT later
optimizes, so it is aligned with the transport objective.  It is also
budget-preserving: the attention only redistributes the fixed mass over real
tokens and cannot create extra evidence.

## Usage

```bash
python run_jecot_ablation.py --variants m2_baseline,m2_mea
```

Direct model flags:

```bash
--hrot_variant J_ECOT_M2
--hrot_ecot_rho_bank 0.80
--hrot_ecot_base_rho 0.80
--hrot_ecot_enable_mea_marginal true
--hrot_ecot_mea_eta_init 0.35
--hrot_ecot_mea_temperature_init 0.70
```

Diagnostics exposed in `return_aux=True` include `mea_query_marginal`,
`mea_support_marginal`, `mea_query_attention`, `mea_support_attention`,
`mea_eta`, `mea_temperature`, entropy, peak-ratio, and uniform-KL summaries.
