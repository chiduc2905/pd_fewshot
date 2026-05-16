from __future__ import annotations

import torch

from net.modules.egsm_marginal import EpisodeGatedShrinkageMarginal


def test_egsm_marginal_rows_sum_to_rho():
    torch.manual_seed(0)
    nq, way, shot, lq, ls = 2, 3, 2, 5, 5
    p = way * shot
    flat = torch.rand(nq, p, lq, ls) * 0.5 + 0.1
    mod = EpisodeGatedShrinkageMarginal(
        hidden_dim=16,
        candidate_tau_q=0.5,
        candidate_tau_b=0.5,
        kappa_min=0.05,
        kappa_max=0.95,
    )
    rho = 0.8
    qm, sm, aux = mod(flat, way_num=way, shot_num=shot, rho=rho)
    assert qm.shape == (nq, p, lq)
    assert sm.shape == (nq, p, ls)
    assert torch.allclose(qm.sum(dim=-1), flat.new_full((nq, p), rho), atol=1e-5, rtol=0.0)
    assert torch.allclose(sm.sum(dim=-1), flat.new_full((nq, p), rho), atol=1e-5, rtol=0.0)
    kappa = aux["egsm_kappa"]
    assert kappa.shape == (nq,)
    assert (kappa >= 0.05).all() and (kappa <= 0.95).all()


def test_egsm_kappa_near_uniform_when_costs_flat():
    torch.manual_seed(1)
    nq, way, shot, lq, ls = 1, 5, 1, 4, 4
    p = way * shot
    flat = torch.ones(nq, p, lq, ls) * 0.5
    mod = EpisodeGatedShrinkageMarginal(hidden_dim=8)
    _, _, aux = mod(flat, way_num=way, shot_num=shot, rho=0.8)
    assert float(aux["egsm_kappa"].detach().mean()) < 0.5


def test_egsm_gate_mlp_receives_episode_gradients():
    torch.manual_seed(2)
    nq, way, shot, lq, ls = 2, 4, 1, 5, 5
    flat = torch.rand(nq, way * shot, lq, ls)
    mod = EpisodeGatedShrinkageMarginal(hidden_dim=8)
    _, _, aux = mod(flat, way_num=way, shot_num=shot, rho=0.8)

    aux["egsm_kappa"].sum().backward()

    first_weight_grad = mod.gate_mlp[0].weight.grad
    final_weight_grad = mod.gate_mlp[2].weight.grad
    assert first_weight_grad is not None and first_weight_grad.abs().sum() > 0
    assert final_weight_grad is not None and final_weight_grad.abs().sum() > 0
