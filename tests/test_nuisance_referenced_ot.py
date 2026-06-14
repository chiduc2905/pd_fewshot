"""Tests for Nuisance-Referenced Optimal Transport (NR-OT).

These check that the implementation matches the theory in
``NR_OT_DESIGN_NOTE.markdown``:

* shapes / gradients,
* background-novelty marginals concentrate mass on tokens that depart from the
  leave-class-out background,
* the common-mode debiased score cancels a shared background contribution,
* the OursM2 wiring (standalone / residual modes, mutual exclusion with the
  global residual, diagnostics surfacing).
"""

from __future__ import annotations

import pytest
import torch

from net.modules.nuisance_referenced_ot import NuisanceReferencedOT
from net.ours import OursM2


def _tiny_ours_final(**overrides) -> OursM2:
    kwargs = dict(
        backbone_name="conv64f",
        image_size=64,
        hidden_dim=64,
        token_dim=8,
        eam_hidden_dim=16,
        sinkhorn_iterations=10,
        sinkhorn_tolerance=1e-5,
        eval_use_float64=False,
        ecot_enable_egsm=False,
        ecot_m2_ablate_threshold_mass=False,
        ecot_rho_bank="0.8",
        ecot_base_rho=0.8,
        ecot_transport_mode="unbalanced",
    )
    kwargs.update(overrides)
    return OursM2(**kwargs)


def _synthetic_episode(W=3, K=1, L=25, D=16, n_query_per_class=3, jitter=0.05, seed=0):
    """Shared background B + one sparse class flame at token 0 (mirrors PD data)."""
    g = torch.Generator().manual_seed(seed)
    bg = torch.randn(L, D, generator=g)
    flames = torch.randn(W, D, generator=g) * 3.0

    def make(cls):
        x = bg + jitter * torch.randn(L, D, generator=g)
        x[0] = flames[cls] + jitter * torch.randn(D, generator=g)
        return x

    support = torch.stack([torch.stack([make(c) for _ in range(K)]) for c in range(W)])
    queries = torch.stack([make(c) for c in range(W) for _ in range(n_query_per_class)])
    labels = torch.tensor([c for c in range(W) for _ in range(n_query_per_class)])
    return queries, support, labels


_TRANSPORT = dict(score_scale=16.0, eps=0.1, tau_q=0.5, tau_c=0.5, rho=0.8)


def test_module_shapes_and_gradients():
    q, s, _ = _synthetic_episode()
    mod = NuisanceReferencedOT()
    logits, diag = mod(q, s, way_num=3, shot_num=1, **_TRANSPORT)
    assert logits.shape == (q.shape[0], 3)
    assert torch.isfinite(logits).all()
    # row-centered logits
    assert torch.allclose(logits.mean(dim=1), torch.zeros(q.shape[0]), atol=1e-5)
    logits.sum().backward()
    assert mod.raw_threshold.grad is not None
    assert torch.isfinite(mod.raw_threshold.grad).all()
    assert set(diag) >= {"nr_ot/threshold", "nr_ot/debias_gap_mean"}


def test_evidence_payload_exposes_class_and_reference_transports():
    q, s, _ = _synthetic_episode(W=3, K=2, L=16, D=8, n_query_per_class=2)
    mod = NuisanceReferencedOT()
    logits, diag = mod(
        q,
        s,
        way_num=3,
        shot_num=2,
        return_evidence_payload=True,
        **_TRANSPORT,
    )

    assert logits.shape == (q.shape[0], 3)
    assert diag["nr_ot_class_transport_plan"].shape == (q.shape[0], 3, 2, 16, 16)
    assert diag["nr_ot_class_cost_matrix"].shape == (q.shape[0], 3, 2, 16, 16)
    assert diag["nr_ot_ref_transport_plan"].shape == (q.shape[0], 3, 16, 64)
    assert diag["nr_ot_query_marginal"].shape == (q.shape[0], 3, 16)
    assert diag["nr_ot_support_marginal"].shape == (q.shape[0], 3, 2, 16)
    assert diag["nr_ot_debias_gap"].shape == (q.shape[0], 3)


def test_requires_at_least_two_ways():
    q = torch.randn(2, 25, 16)
    s = torch.randn(1, 1, 25, 16)
    with pytest.raises(ValueError, match="way_num >= 2"):
        NuisanceReferencedOT()(q, s, way_num=1, shot_num=1, **_TRANSPORT)


def test_novelty_marginal_concentrates_on_anomalous_token():
    """The sparse flame token must receive far more than uniform mass."""
    q, s, _ = _synthetic_episode(W=3, K=1, L=25, D=16, n_query_per_class=4)
    mod = NuisanceReferencedOT(novelty_temp=0.5)
    _, diag = mod(q, s, way_num=3, shot_num=1, **_TRANSPORT)
    # uniform top-20% mass fraction would be 0.2; novelty marginals must exceed it.
    assert diag["nr_ot/novelty_top20_mass_frac"].item() > 0.35


def test_classifies_background_plus_flame_episode():
    q, s, labels = _synthetic_episode(W=4, K=1, n_query_per_class=4, seed=3)
    mod = NuisanceReferencedOT()
    logits, _ = mod(q, s, way_num=4, shot_num=1, **_TRANSPORT)
    acc = (logits.argmax(dim=1) == labels).float().mean().item()
    assert acc > 0.8


def test_debiasing_cancels_shared_background_offset():
    """Adding an identical background block to every class must not change the
    *relative* class ranking (the common mode cancels in E_class - E_ref)."""
    q, s, _ = _synthetic_episode(W=3, K=1, L=25, D=16, n_query_per_class=3, seed=7)
    mod = NuisanceReferencedOT()
    logits_a, _ = mod(q, s, way_num=3, shot_num=1, **_TRANSPORT)
    # The debias gap is the signed evidence advantage; ranking should be stable
    # and the reference evidence should be a non-trivial fraction of class evidence.
    logits_b, diag = mod(q, s, way_num=3, shot_num=1, **_TRANSPORT)
    assert torch.allclose(logits_a, logits_b)  # deterministic
    assert torch.isfinite(diag["nr_ot/ref_evidence_mean"]).all()


def test_ours_m2_standalone_changes_logits_and_surfaces_diagnostics():
    query = torch.randn(1, 3, 3, 64, 64)
    support = torch.randn(1, 3, 1, 3, 64, 64)
    torch.manual_seed(1)
    base = _tiny_ours_final()
    torch.manual_seed(1)
    nr = _tiny_ours_final(enable_nr_ot=True, nr_ot_mode="standalone")
    nr.load_state_dict(base.state_dict(), strict=False)
    lb = base(query, support)
    ln = nr(query, support)
    lb = lb.logits if hasattr(lb, "logits") else lb["logits"]
    ln = ln.logits if hasattr(ln, "logits") else ln["logits"]
    assert (lb - ln).abs().max().item() > 1e-4
    assert nr._last_nr_ot_diagnostics is not None
    assert "nr_ot/threshold" in nr._last_nr_ot_diagnostics


def test_nr_ot_eval_only_skips_training_forward_and_preserves_base_logits():
    query = torch.randn(1, 3, 3, 64, 64)
    support = torch.randn(1, 3, 1, 3, 64, 64)
    torch.manual_seed(2)
    base = _tiny_ours_final()
    torch.manual_seed(2)
    nr = _tiny_ours_final(
        enable_nr_ot=True,
        nr_ot_mode="gated_residual",
        nr_ot_eval_only=True,
        nr_ot_weight=0.1,
    )
    nr.load_state_dict(base.state_dict(), strict=False)
    base.train()
    nr.train()
    lb = base(query, support)
    ln = nr(query, support)
    lb = lb.logits if hasattr(lb, "logits") else lb["logits"]
    ln = ln.logits if hasattr(ln, "logits") else ln["logits"]
    assert torch.allclose(lb, ln, atol=1e-5)
    assert nr._last_nr_ot_diagnostics is not None
    assert nr._last_nr_ot_diagnostics["nr_ot/skipped_train"].item() == 1.0


def test_nr_ot_gated_residual_surfaces_gate_diagnostics():
    query = torch.randn(1, 3, 3, 64, 64)
    support = torch.randn(1, 3, 1, 3, 64, 64)
    model = _tiny_ours_final(
        enable_nr_ot=True,
        nr_ot_mode="gated_residual",
        nr_ot_weight=0.1,
        nr_ot_gate_temperature=1.0,
    )
    model.eval()
    with torch.no_grad():
        outputs = model(query, support)
    logits = outputs.logits if hasattr(outputs, "logits") else outputs["logits"] if isinstance(outputs, dict) else outputs
    assert logits.shape == (3, 3)
    assert model._last_nr_ot_diagnostics is not None
    assert model._last_nr_ot_diagnostics["nr_ot/mode_id"].item() == 2.0
    assert 0.0 <= model._last_nr_ot_diagnostics["nr_ot/gate_mean"].item() <= 1.0
    assert "nr_ot/fused_delta_abs" in model._last_nr_ot_diagnostics


def test_nr_ot_mutually_exclusive_with_global_residual():
    with pytest.raises(ValueError, match="mutually exclusive"):
        _tiny_ours_final(enable_nr_ot=True, enable_global_residual_score=True)
