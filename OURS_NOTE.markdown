# Ours: standalone J-ECOT-M2 path with EGSM

In this codebase, **Ours** is the `--model ours` entry (`net/ours.py`): a
paper-facing stack built on **J-ECOT-M2** with fixed defaults (single budget
`rho=0.8`, unbalanced UOT, M2 threshold-mass score ablated unless overridden).
The distinguishing marginal module for the full design is **Episode-Gated
Shrinkage Marginals (EGSM)** (`net/modules/egsm_marginal.py`), enabled by
default for contribution modes that are not explicitly EGSM-off.

CRS-M2, NS-M2, and MEA-M2 notes describe *alternative* ECOT marginal extensions.
EGSM is **mutually exclusive** with MEA, CRS, NNCS, and CCDM on the same path
(see `hrot_fsl.py`). Ours keeps MEA/CCDM/CRS off and turns **EGSM on** for
`full`, `full_ot`, and `gap`.

## EGSM (Episode-Gated Shrinkage Marginals)

For each query/class/shot, ECOT needs token marginals before the UOT solve. With
EGSM, query and support priors are a **budget-preserving shrinkage** between a
uniform prior and a **cost-derived candidate** prior (same cost tensor the
solver uses):

```text
pi = (1 - kappa) * uniform + kappa * candidate
a = rho * pi_query
b = rho * pi_support
```

`kappa` in `[kappa_min, kappa_max]` is produced by a small MLP on **psi**
ambiguity statistics of the episode cost tensor, so the model commits more mass
to the candidate prior when costs look discriminative, and stays closer to
uniform when ambiguity is high. The current default is deliberately
conservative (`kappa_max=0.35`, `tau_q=tau_b=1.0`): candidate priors are built
from stop-gradient, episode-normalized costs so EGSM acts as a bounded
episode-conditioned residual over uniform marginals rather than replacing the
transport prior wholesale.

Optional **adaptive rho** (`--hrot_ecot_egsm_adaptive_rho true`): a second head
predicts a per-query offset to the transport mass budget, bounded around
`base_rho` (see `configs/ours_adaptive_rho_sweep.yaml`).

## Contribution ablations (`--ours_ablation`)

| Ablation   | EGSM | Notes |
|-----------|------|--------|
| `full`    | on   | Local descriptors + UOT + EGSM (default). |
| `full_ot` | on   | Balanced OT (`rho=1.0`) with EGSM. |
| `no_egsm` | off  | Uniform token marginals; control for EGSM. |
| `gap`     | on   | GAP-pooled tokens for the cost; same EGSM + UOT stack as `full`. |

Aliases such as `uniform_evidence` map to `no_egsm` (see `normalize_ours_ablation`).

## CLI (minimal)

Full Ours with EGSM (defaults applied inside `Ours` / `main.py` wiring):

```bash
python main.py --model ours --ours_ablation full
```

Explicit EGSM flag on the J_ECOT_M2 family (must be `true` or `false`, no auto):

```bash
--hrot_variant J_ECOT_M2
--hrot_ecot_enable_egsm true
```

Turn EGSM off for ablation:

```bash
python main.py --model ours --ours_ablation no_egsm
# or
--hrot_ecot_enable_egsm false
```

Sweep configs mentioning Ours + EGSM: `configs/ours_mncr_sweep.yaml`,
`configs/ours_adaptive_rho_sweep.yaml`, `configs/ours_cpm_sweep.yaml` (CPM
variant).

## Diagnostics (`return_aux=True`)

Keys from `egsm_marginal` include `egsm_kappa`, `egsm_psi`, `egsm_query_pi`,
`egsm_support_pi`, `egsm_candidate_tau_q`, `egsm_candidate_tau_b`,
`egsm_aux_loss`, and when enabled `egsm_rho_adaptive`.

## Claim boundary

Safe claim: **Ours** is the fixed-budget ECOT/UOT few-shot matcher with
**episode-conditioned token marginals (EGSM)** blending uniform and cost-based
priors, plus optional adaptive rho; **`no_egsm`** is the matched control with
uniform marginals.

Do not conflate EGSM with MEA (attention on standardized cost), CRS
(cross-reference + selective support prior), or NS-M2 (noise sink); those are
documented in their own note files.
