import torch

from net.hrot_fsl import HROTFSL


def _build_ecot_noise_sink_model(
    sink_cost: float = 1.0,
    *,
    variant: str = "J_ECOT",
    sink_cost_mode: str = "fixed",
) -> HROTFSL:
    return HROTFSL(
        backbone_name="conv64f",
        image_size=64,
        hidden_dim=64,
        token_dim=24,
        variant=variant,
        eam_hidden_dim=32,
        curvature_init=1.0,
        projection_scale=0.1,
        token_temperature=0.1,
        score_scale=8.0,
        tau_q=0.5,
        tau_c=0.5,
        sinkhorn_epsilon=0.02,
        sinkhorn_iterations=200,
        sinkhorn_tolerance=1e-7,
        fixed_mass=0.8,
        min_mass=0.1,
        mass_bonus_init=1.0,
        lambda_rho=0.05,
        rho_target=0.8,
        lambda_rho_rank=0.05,
        rho_rank_margin=0.05,
        rho_rank_temperature=0.05,
        lambda_curvature=0.01,
        min_curvature=0.05,
        normalize_euclidean_tokens=True,
        eval_use_float64=True,
        hyperbolic_backend="auto",
        ot_backend="native",
        eps=1e-6,
        ecot_rho_bank="0.80",
        ecot_base_rho=0.80,
        ecot_enable_noise_sink=True,
        ecot_noise_sink_cost_mode=sink_cost_mode,
        ecot_noise_sink_cost_init=sink_cost,
    )


def _solve_sink_plan(model: HROTFSL, cost: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rho = torch.full((1, 1), 0.80)
    query_mass = torch.full((1, 1, 4), 0.80 / 4.0)
    support_mass = torch.full((1, 1, 4), 0.80 / 4.0)
    cost_with_sink, query_with_sink, support_with_sink = model._append_noise_sink(
        cost,
        query_mass,
        support_mass,
        rho,
    )
    plan_with_sink, _, _ = model._transport_match(
        cost_with_sink,
        rho,
        a=query_with_sink,
        b=support_with_sink,
    )
    real_mass = plan_with_sink[..., :-1, :-1].sum()
    sink_mass = 0.5 * (
        plan_with_sink[..., :-1, -1].sum()
        + plan_with_sink[..., -1, :-1].sum()
    )
    return real_mass, sink_mass


def test_ecot_noise_sink_rejects_high_cost_real_matches_but_preserves_good_matches():
    model = _build_ecot_noise_sink_model(sink_cost=0.10)
    model.eval()

    bad_cost = torch.full((1, 1, 4, 4), 1.0)
    good_cost = bad_cost.clone()
    good_cost[..., torch.arange(4), torch.arange(4)] = 0.01

    good_real, good_sink = _solve_sink_plan(model, good_cost)
    bad_real, bad_sink = _solve_sink_plan(model, bad_cost)

    assert good_real > 0.70
    assert good_sink < 0.05
    assert bad_real < 0.05
    assert bad_sink > 0.30


def test_ecot_noise_sink_threshold_mode_uses_evidence_threshold_without_extra_parameter():
    model = _build_ecot_noise_sink_model(
        variant="J_ECOT_M2",
        sink_cost_mode="threshold",
    )
    model.eval()
    bad_cost = torch.full((1, 1, 4, 4), 1.0)
    good_cost = bad_cost.clone()
    good_cost[..., torch.arange(4), torch.arange(4)] = 0.01

    expected_threshold = model.transport_cost_threshold.detach()
    assert model.raw_noise_sink_cost is None
    assert torch.allclose(model.noise_sink_cost, expected_threshold)

    good_real, good_sink = _solve_sink_plan(model, good_cost)
    bad_real, bad_sink = _solve_sink_plan(model, bad_cost)

    assert good_real > 0.70
    assert good_sink < 0.05
    assert bad_real < 0.08
    assert bad_sink > 0.30


def test_default_ecot_noise_sink_cost_is_too_expensive_for_rejection_fixture():
    model = _build_ecot_noise_sink_model(sink_cost=1.0)
    model.eval()
    bad_cost = torch.full((1, 1, 4, 4), 1.0)

    bad_real, bad_sink = _solve_sink_plan(model, bad_cost)

    assert bad_real > 0.20
    assert bad_sink < 0.05
