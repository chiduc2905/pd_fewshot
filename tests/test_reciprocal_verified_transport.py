import torch

from net.modules.reciprocal_verified_transport import ReciprocalVerifiedTransport


def _cluster_indices(top: int, left: int, width: int) -> list[int]:
    return [
        top * width + left,
        top * width + left + 1,
        (top + 1) * width + left,
        (top + 1) * width + left + 1,
    ]


def test_reciprocal_verified_transport_suppresses_isolated_false_match():
    verifier = ReciprocalVerifiedTransport(
        beta=1.0,
        tau=0.10,
        ratio_threshold=0.20,
        kernel_size=3,
        cost_quantile=0.25,
        min_gate=0.0,
    )
    cost = torch.ones(1, 1, 1, 16, 16)
    plan = torch.zeros_like(cost)

    true_tokens = _cluster_indices(1, 1, 4)
    for query_token in true_tokens:
        for support_token in true_tokens:
            cost[..., query_token, support_token] = 0.04
    for token in true_tokens:
        plan[..., token, token] = 0.10
    # A misleading isolated correspondence: very low feature cost, but no
    # neighboring support in the query or support grid.
    cost[..., 0, 15] = 0.01
    plan[..., 0, 15] = 0.25

    verified, diagnostics = verifier(cost=cost, plan=plan, spatial_hw=(4, 4))

    false_retained = verified[..., 0, 15].item()
    true_retained = verified[..., true_tokens, true_tokens].sum().item()
    assert false_retained < 0.05
    assert true_retained > false_retained * 4.0
    assert diagnostics["rvuot/retained_mass_ratio"].item() < 1.0
    assert diagnostics["rvuot/removed_mass_mean"].item() > 0.0


def test_reciprocal_verified_transport_is_not_tied_to_absolute_time_position():
    verifier = ReciprocalVerifiedTransport(
        beta=1.0,
        tau=0.10,
        ratio_threshold=0.20,
        kernel_size=3,
        cost_quantile=0.25,
        min_gate=0.0,
    )

    def make_pair(top: int, left: int) -> tuple[torch.Tensor, torch.Tensor]:
        cost = torch.ones(1, 1, 1, 25, 25)
        plan = torch.zeros_like(cost)
        cluster = _cluster_indices(top, left, 5)
        for query_token in cluster:
            for support_token in cluster:
                cost[..., query_token, support_token] = 0.04
        for token in cluster:
            plan[..., token, token] = 0.10
        return cost, plan

    cost_a, plan_a = make_pair(1, 1)
    cost_b, plan_b = make_pair(2, 2)
    verified_a, _ = verifier(cost=cost_a, plan=plan_a, spatial_hw=(5, 5))
    verified_b, _ = verifier(cost=cost_b, plan=plan_b, spatial_hw=(5, 5))

    assert torch.allclose(verified_a.sum(), verified_b.sum(), atol=1e-6, rtol=1e-6)


def test_reciprocal_verified_transport_suppresses_common_mode_noise_match():
    verifier = ReciprocalVerifiedTransport(
        beta=1.0,
        tau=0.10,
        ratio_threshold=0.0,
        kernel_size=1,
        cost_quantile=0.60,
        min_gate=0.0,
        enable_rival_gate=True,
        rival_tau=0.10,
    )
    cost = torch.ones(1, 2, 1, 4, 4)
    plan = torch.zeros_like(cost)

    # Query token 0 is common-mode noise: it has an equally good match in both classes.
    cost[:, 0, 0, 0, 0] = 0.01
    cost[:, 1, 0, 0, 0] = 0.01
    plan[:, 0, 0, 0, 0] = 0.20
    plan[:, 1, 0, 0, 0] = 0.20

    # Query token 1 is class-specific evidence for class 0.
    cost[:, 0, 0, 1, 1] = 0.01
    cost[:, 1, 0, 1, 1] = 0.80
    plan[:, 0, 0, 1, 1] = 0.20

    verified, diagnostics = verifier(cost=cost, plan=plan, spatial_hw=(2, 2))

    common_noise = verified[:, 0, 0, 0].sum()
    class_specific = verified[:, 0, 0, 1].sum()
    assert class_specific > common_noise * 1.5
    assert diagnostics["rvuot/rival_gate_enabled"].item() == 1.0
    assert diagnostics["rvuot/rival_gate_mean"].item() < 1.0
