from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from net.model_factory import build_model_from_args
from net.spif_ota import SPIFOTA


def _build_model(**overrides) -> SPIFOTA:
    kwargs = dict(
        backbone_name="conv64f",
        image_size=64,
        hidden_dim=64,
        transport_dim=24,
        projector_hidden_dim=32,
        mass_hidden_dim=16,
        mass_temperature=1.0,
        sinkhorn_epsilon=0.1,
        sinkhorn_iterations=40,
        shot_hidden_dim=16,
        shot_aggregation="learned",
        position_cost_weight=0.0,
        use_column_bias=True,
        lambda_mass=0.05,
        lambda_consistency=0.05,
        eps=1e-6,
    )
    kwargs.update(overrides)
    return SPIFOTA(**kwargs)


def test_spif_ota_forward_shapes_and_sinkhorn_marginals():
    torch.manual_seed(0)
    model = _build_model()
    model.eval()

    query = torch.randn(1, 2, 3, 64, 64)
    support = torch.randn(1, 3, 2, 3, 64, 64)
    targets = torch.tensor([0, 2], dtype=torch.long)

    with torch.no_grad():
        outputs = model(query, support, query_targets=targets, return_aux=True)

    assert outputs["logits"].shape == (2, 3)
    assert outputs["class_scores"].shape == (2, 3)
    assert outputs["total_distance"].shape == (2, 3)
    assert outputs["shot_scores"].shape == (2, 3, 2)
    assert outputs["shot_aggregation_weights"].shape == (2, 3, 2)
    assert outputs["transport_cost"].shape == (2, 3, 2)
    assert outputs["plan_entropy"].shape == (2, 3, 2)
    assert outputs["query_tokens"].shape[:2] == outputs["query_masses"].shape
    assert outputs["support_tokens"].shape[:4] == outputs["support_masses"].shape
    assert outputs["transport_plan"].shape[-2:] == (outputs["query_tokens"].shape[1], outputs["query_tokens"].shape[1])
    assert outputs["cost_matrix"].shape == outputs["transport_plan"].shape

    assert torch.allclose(outputs["query_masses"].sum(dim=-1), torch.ones(2), atol=1e-6, rtol=0.0)
    assert torch.allclose(
        outputs["support_masses"].sum(dim=-1),
        torch.ones_like(outputs["support_masses"].sum(dim=-1)),
        atol=1e-6,
        rtol=0.0,
    )
    assert torch.allclose(
        outputs["shot_aggregation_weights"].sum(dim=-1),
        torch.ones_like(outputs["shot_aggregation_weights"].sum(dim=-1)),
        atol=1e-6,
        rtol=0.0,
    )

    plan = outputs["transport_plan"]
    query_masses = outputs["query_masses"].unsqueeze(1).unsqueeze(1)
    support_masses = outputs["support_masses"][0]
    assert torch.allclose(plan.sum(dim=-1), query_masses.expand_as(plan.sum(dim=-1)), atol=1e-3, rtol=0.0)
    assert torch.allclose(plan.sum(dim=-2), support_masses.unsqueeze(0).expand_as(plan.sum(dim=-2)), atol=1e-3, rtol=0.0)

    for key in (
        "logits",
        "class_scores",
        "total_distance",
        "shot_scores",
        "shot_aggregation_weights",
        "transport_cost",
        "plan_entropy",
        "query_tokens",
        "support_tokens",
        "query_masses",
        "support_masses",
        "transport_plan",
        "cost_matrix",
        "aux_loss",
        "mean_query_mass_entropy",
        "mean_support_mass_entropy",
        "mass_regularization",
        "shot_consistency_loss",
    ):
        assert torch.isfinite(outputs[key]).all()


def test_spif_ota_is_support_permutation_invariant():
    torch.manual_seed(1)
    model = _build_model()
    model.eval()

    query = torch.randn(1, 3, 3, 64, 64)
    support = torch.randn(1, 4, 2, 3, 64, 64)
    support_permuted = support[:, :, torch.tensor([1, 0])]
    targets = torch.tensor([0, 1, 2], dtype=torch.long)

    with torch.no_grad():
        outputs = model(query, support, query_targets=targets, return_aux=True)
        permuted = model(query, support_permuted, query_targets=targets, return_aux=True)

    assert torch.allclose(outputs["logits"], permuted["logits"], atol=1e-5, rtol=0.0)
    assert torch.allclose(outputs["class_scores"], permuted["class_scores"], atol=1e-5, rtol=0.0)
    assert torch.allclose(outputs["total_distance"], permuted["total_distance"], atol=1e-5, rtol=0.0)
    assert torch.allclose(outputs["shot_scores"], permuted["shot_scores"].flip(-1), atol=1e-5, rtol=0.0)
    assert torch.allclose(
        outputs["shot_aggregation_weights"],
        permuted["shot_aggregation_weights"].flip(-1),
        atol=1e-5,
        rtol=0.0,
    )


def test_spif_ota_backpropagates_through_main_modules():
    torch.manual_seed(2)
    model = _build_model(lambda_mass=0.1, lambda_consistency=0.1)
    model.train()

    query = torch.randn(1, 2, 3, 64, 64)
    support = torch.randn(1, 3, 2, 3, 64, 64)
    targets = torch.tensor([1, 0], dtype=torch.long)

    outputs = model(query, support, query_targets=targets)
    loss = F.cross_entropy(outputs["logits"], targets) + outputs["aux_loss"]
    model.zero_grad(set_to_none=True)
    loss.backward()

    assert model.token_projector.network[1].weight.grad is not None
    assert model.mass_generator.score_head.weight.grad is not None
    assert model.shot_aggregator.network[0].weight.grad is not None
    assert torch.isfinite(model.token_projector.network[1].weight.grad).all()
    assert torch.isfinite(model.mass_generator.score_head.weight.grad).all()
    assert torch.isfinite(model.shot_aggregator.network[0].weight.grad).all()


def test_spif_ota_eval_without_aux_returns_plain_logits():
    torch.manual_seed(3)
    model = _build_model()
    model.eval()

    query = torch.randn(1, 4, 3, 64, 64)
    support = torch.randn(1, 5, 1, 3, 64, 64)

    with torch.no_grad():
        logits = model(query, support)

    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (4, 5)
    assert torch.isfinite(logits).all()


def test_spif_ota_model_factory_builds_and_runs():
    args = SimpleNamespace(
        model="spif_ota",
        device="cpu",
        image_size=64,
        fewshot_backbone="conv64f",
        token_dim=24,
        spif_ota_transport_dim=24,
        spif_ota_projector_hidden_dim=32,
        spif_ota_mass_hidden_dim=16,
        spif_ota_mass_temperature=1.0,
        spif_ota_sinkhorn_epsilon=0.1,
        spif_ota_sinkhorn_iterations=30,
        spif_ota_shot_hidden_dim=16,
        spif_ota_shot_aggregation="learned",
        spif_ota_position_cost_weight=0.0,
        spif_ota_use_column_bias="true",
        spif_ota_lambda_mass=0.0,
        spif_ota_lambda_consistency=0.0,
        spif_ota_eps=1e-6,
    )
    model = build_model_from_args(args)
    model.eval()

    query = torch.randn(1, 2, 3, 64, 64)
    support = torch.randn(1, 3, 2, 3, 64, 64)

    with torch.no_grad():
        outputs = model(query, support, return_aux=True)

    assert outputs["logits"].shape == (2, 3)
    assert outputs["support_tokens"].shape[-1] == 24
    assert torch.isfinite(outputs["logits"]).all()


def test_forward_scores_routes_query_targets_to_spif_ota():
    source = Path("main.py").read_text(encoding="utf-8")
    module = ast.parse(source)

    forward_scores = next(
        node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "forward_scores"
    )
    spif_ota_branch = None
    for node in ast.walk(forward_scores):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if (
            isinstance(test, ast.Compare)
            and isinstance(test.left, ast.Attribute)
            and test.left.attr == "model"
            and len(test.ops) == 1
            and isinstance(test.ops[0], ast.Eq)
            and len(test.comparators) == 1
            and isinstance(test.comparators[0], ast.Constant)
            and test.comparators[0].value == "spif_ota"
        ):
            spif_ota_branch = node
            break

    assert spif_ota_branch is not None, "forward_scores must keep a dedicated spif_ota branch"

    call = spif_ota_branch.body[0].value
    assert isinstance(call, ast.Call)
    keyword_names = {kw.arg for kw in call.keywords}
    assert "query_targets" in keyword_names
    assert "return_aux" in keyword_names
