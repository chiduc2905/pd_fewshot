"""Posterior residual transport flow for SC-LFI v4.

This module is intentionally kept almost identical to `v3`.

Formula:
- posterior predictive measure:
  `muhat_c = (T_theta,c)_# mu_c^0`
- conditional velocity:
  `v_theta(y, t; h_c, M_c, t_ctx)`

Why this reuse is justified:
- the primary `v4` redesign is the few-shot object being transported;
- the fixed-step residual latent flow abstraction remains mathematically valid;
- keeping the transport operator stable avoids confounding architectural
  improvements in the posterior object with unrelated flow-parameterization
  changes.
"""

from __future__ import annotations

from net.modules.posterior_transport_flow_v3 import (
    PosteriorTransportFlowModelV3,
    PosteriorTransportVelocityFieldV3,
    sample_posterior_conditional_path_v3,
    target_posterior_transport_velocity_v3,
)


class PosteriorTransportVelocityFieldV4(PosteriorTransportVelocityFieldV3):
    """Unchanged residual velocity field reused for the `v4` posterior object."""


class PosteriorTransportFlowModelV4(PosteriorTransportFlowModelV3):
    """Unchanged fixed-step posterior transport solver reused in `v4`."""

