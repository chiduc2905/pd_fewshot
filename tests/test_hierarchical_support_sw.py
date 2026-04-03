from types import SimpleNamespace

import torch

from net.fewshot_common import multiscale_feature_map_to_tokens
from net.heads.hierarchical_support_sw_head import HierarchicalSupportSlicedWassersteinHead
from net.hierarchical_support_sw_net import HSSWTokenWeighting, HierarchicalSupportSlicedWassersteinNet
from net.metrics.hssw_sliced_wasserstein import HSSWSlicedWassersteinDistance
from net.model_factory import build_model_from_args


class _CountingHSSWSlicedWassersteinDistance(HSSWSlicedWassersteinDistance):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.forward_calls = 0
        self.pairwise_calls = 0

    def forward(self, *args, **kwargs):
        self.forward_calls += 1
        return super().forward(*args, **kwargs)

    def pairwise_distance(self, *args, **kwargs):
        self.pairwise_calls += 1
        return super().pairwise_distance(*args, **kwargs)


def _build_metric(weighted=True):
    metric = HSSWSlicedWassersteinDistance(
        train_num_projections=8,
        eval_num_projections=8,
        p=2.0,
        weighted=weighted,
        normalize_tokens=True,
        train_projection_mode="fixed",
        eval_projection_mode="fixed",
        projection_seed=17,
    )
    metric.eval()
    return metric


def _build_head(weighted=True, lambda_cons=0.1, lambda_red=0.05):
    head = HierarchicalSupportSlicedWassersteinHead(
        sw_distance=_build_metric(weighted=weighted),
        tau=0.35,
        lambda_cons=lambda_cons,
        lambda_red=lambda_red,
        weighted_sw=weighted,
        eps=1e-6,
    )
    head.eval()
    return head


def test_hssw_head_matches_manual_formula_exactly():
    torch.manual_seed(0)
    head = _build_head(weighted=True, lambda_cons=0.13, lambda_red=0.07)

    query_tokens = torch.randn(3, 4, 5)
    query_weights = torch.rand(3, 4)
    support_tokens = torch.randn(2, 3, 4, 5)
    support_weights = torch.rand(2, 3, 4)

    support_state = head.build_support_state(support_tokens, support_weights)
    outputs = head(
        query_tokens=query_tokens,
        query_weights=query_weights,
        support_state=support_state,
        return_aux=True,
    )

    manual_shot_distances = torch.empty_like(outputs["shot_distances"])
    query_transport_weights = head._resolve_transport_weights(query_weights)
    support_transport_weights = head._resolve_transport_weights(support_weights)
    for query_idx in range(query_tokens.shape[0]):
        for class_idx in range(support_tokens.shape[0]):
            for shot_idx in range(support_tokens.shape[1]):
                manual_shot_distances[query_idx, class_idx, shot_idx] = head.sw_distance(
                    query_tokens[query_idx : query_idx + 1],
                    support_tokens[class_idx, shot_idx : shot_idx + 1],
                    x_weights=query_transport_weights[query_idx : query_idx + 1],
                    y_weights=support_transport_weights[class_idx, shot_idx : shot_idx + 1],
                    reduction="none",
                ).squeeze(0)

    manual_gamma = torch.softmax(-manual_shot_distances / head.tau, dim=-1)
    manual_base = torch.sum(manual_gamma * manual_shot_distances, dim=-1)
    manual_cons = -torch.log(manual_gamma + head.eps).sum(dim=-1)
    manual_red = torch.empty(support_tokens.shape[0], dtype=query_tokens.dtype)
    for class_idx in range(support_tokens.shape[0]):
        pair_terms = []
        for left_idx in range(support_tokens.shape[1]):
            for right_idx in range(support_tokens.shape[1]):
                if left_idx == right_idx:
                    continue
                pair_terms.append(
                    torch.exp(
                        -head.sw_distance(
                            support_tokens[class_idx, left_idx : left_idx + 1],
                            support_tokens[class_idx, right_idx : right_idx + 1],
                            x_weights=support_transport_weights[class_idx, left_idx : left_idx + 1],
                            y_weights=support_transport_weights[class_idx, right_idx : right_idx + 1],
                            reduction="none",
                        ).squeeze(0)
                    )
                )
        manual_red[class_idx] = torch.stack(pair_terms).mean()

    manual_total = manual_base + head.lambda_cons * manual_cons + head.lambda_red * manual_red.unsqueeze(0)
    logit_scale = head.get_logit_scale()

    assert torch.allclose(outputs["shot_distances"], manual_shot_distances, atol=1e-6, rtol=0.0)
    assert torch.allclose(outputs["shot_responsibilities"], manual_gamma, atol=1e-6, rtol=0.0)
    assert torch.allclose(outputs["base_distance"], manual_base, atol=1e-6, rtol=0.0)
    assert torch.allclose(outputs["consistency_penalty"], manual_cons, atol=1e-6, rtol=0.0)
    assert torch.allclose(outputs["redundancy_penalty"], manual_red, atol=1e-6, rtol=0.0)
    assert torch.allclose(outputs["total_distance"], manual_total, atol=1e-6, rtol=0.0)
    assert torch.allclose(outputs["logit_scale"], logit_scale, atol=1e-6, rtol=0.0)
    assert torch.allclose(outputs["logits"], -logit_scale * manual_total, atol=1e-6, rtol=0.0)


def test_hssw_head_is_support_order_invariant():
    torch.manual_seed(1)
    head = _build_head(weighted=True, lambda_cons=0.09, lambda_red=0.03)

    query_tokens = torch.randn(2, 5, 4)
    query_weights = torch.rand(2, 5)
    support_tokens = torch.randn(3, 4, 5, 4)
    support_weights = torch.rand(3, 4, 5)
    permutation = torch.tensor([2, 0, 3, 1])

    outputs = head(
        query_tokens=query_tokens,
        query_weights=query_weights,
        support_state=head.build_support_state(support_tokens, support_weights),
        return_aux=True,
    )
    permuted = head(
        query_tokens=query_tokens,
        query_weights=query_weights,
        support_state=head.build_support_state(support_tokens[:, permutation], support_weights[:, permutation]),
        return_aux=True,
    )

    assert torch.allclose(outputs["logits"], permuted["logits"], atol=1e-6, rtol=0.0)
    assert torch.allclose(outputs["total_distance"], permuted["total_distance"], atol=1e-6, rtol=0.0)
    assert torch.allclose(outputs["redundancy_penalty"], permuted["redundancy_penalty"], atol=1e-6, rtol=0.0)


def test_hssw_lambda_zero_reduces_to_base_distance():
    torch.manual_seed(2)
    head = _build_head(weighted=True, lambda_cons=0.0, lambda_red=0.0)
    query_tokens = torch.randn(3, 4, 6)
    query_weights = torch.rand(3, 4)
    support_tokens = torch.randn(2, 3, 4, 6)
    support_weights = torch.rand(2, 3, 4)

    outputs = head(
        query_tokens=query_tokens,
        query_weights=query_weights,
        support_state=head.build_support_state(support_tokens, support_weights),
        return_aux=True,
    )

    assert torch.allclose(outputs["total_distance"], outputs["base_distance"], atol=1e-6, rtol=0.0)
    assert torch.allclose(outputs["logits"], -outputs["logit_scale"] * outputs["base_distance"], atol=1e-6, rtol=0.0)


def test_uniform_responsibilities_reduce_to_mean_shot_distance():
    torch.manual_seed(3)
    head = _build_head(weighted=True)
    shot_distances = torch.randn(4, 3, 5).abs()
    uniform_gamma = torch.full_like(shot_distances, 1.0 / float(shot_distances.shape[-1]))

    reduced = head.compute_base_distance(shot_distances, shot_responsibilities=uniform_gamma)
    expected = shot_distances.mean(dim=-1)

    assert torch.allclose(reduced, expected, atol=1e-6, rtol=0.0)


def test_support_state_caches_support_only_redundancy():
    torch.manual_seed(4)
    metric = _CountingHSSWSlicedWassersteinDistance(
        train_num_projections=8,
        eval_num_projections=8,
        p=2.0,
        weighted=True,
        normalize_tokens=True,
        train_projection_mode="fixed",
        eval_projection_mode="fixed",
        projection_seed=19,
    )
    metric.eval()
    head = HierarchicalSupportSlicedWassersteinHead(
        sw_distance=metric,
        tau=0.25,
        lambda_cons=0.1,
        lambda_red=0.05,
        weighted_sw=True,
        eps=1e-6,
    )
    head.eval()

    support_state = head.build_support_state(
        shot_tokens=torch.randn(3, 3, 4, 6),
        shot_weights=torch.rand(3, 3, 4),
    )
    assert metric.forward_calls == 1
    assert metric.pairwise_calls == 0

    _ = head(
        query_tokens=torch.randn(5, 4, 6),
        query_weights=torch.rand(5, 4),
        support_state=support_state,
        return_aux=False,
    )
    assert metric.forward_calls == 1
    assert metric.pairwise_calls == 1


def test_hssw_unweighted_mode_uses_uniform_transport_weights():
    torch.manual_seed(5)
    head = _build_head(weighted=False, lambda_cons=0.1, lambda_red=0.02)
    support_state = head.build_support_state(
        shot_tokens=torch.randn(2, 3, 4, 5),
        shot_weights=torch.rand(2, 3, 4),
    )
    expected = torch.full_like(support_state.shot_transport_weights, 1.0 / 4.0)
    assert torch.allclose(support_state.shot_transport_weights, expected, atol=1e-6, rtol=0.0)


def test_hssw_token_weighter_mixes_with_uniform_floor():
    torch.manual_seed(5)
    mix_alpha = 0.8
    token_count = 6
    weighter = HSSWTokenWeighting(
        input_dim=4,
        hidden_dim=8,
        temperature=1.5,
        mix_alpha=mix_alpha,
    )
    weights = weighter(torch.randn(3, token_count, 4))

    expected_floor = (1.0 - mix_alpha) / float(token_count)
    assert torch.all(weights >= expected_floor - 1e-6)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(3), atol=1e-6, rtol=0.0)


def test_hssw_head_preserves_gradients_without_nan():
    torch.manual_seed(6)
    head = _build_head(weighted=True, lambda_cons=0.11, lambda_red=0.05)
    query_tokens = torch.randn(2, 4, 5, requires_grad=True)
    query_weights = torch.rand(2, 4, requires_grad=True)
    support_tokens = torch.randn(3, 2, 4, 5, requires_grad=True)
    support_weights = torch.rand(3, 2, 4, requires_grad=True)

    support_state = head.build_support_state(support_tokens, support_weights)
    outputs = head(
        query_tokens=query_tokens,
        query_weights=query_weights,
        support_state=support_state,
        return_aux=True,
    )
    outputs["logits"].sum().backward()

    assert torch.isfinite(query_tokens.grad).all()
    assert torch.isfinite(query_weights.grad).all()
    assert torch.isfinite(support_tokens.grad).all()
    assert torch.isfinite(support_weights.grad).all()


def test_hssw_model_smoke_and_invariance():
    torch.manual_seed(7)
    model = HierarchicalSupportSlicedWassersteinNet(
        backbone_name="conv64f",
        image_size=64,
        token_weight_hidden=16,
        weighted_sw=True,
        normalize_tokens_before_sw=True,
        train_num_projections=4,
        eval_num_projections=4,
        train_projection_mode="fixed",
        eval_projection_mode="fixed",
        tau=0.2,
        lambda_cons=0.1,
        lambda_red=0.03,
    )
    model.eval()

    query = torch.randn(2, 2, 3, 64, 64)
    support = torch.randn(2, 3, 2, 3, 64, 64)
    support_permuted = support[:, :, torch.tensor([1, 0])]

    with torch.no_grad():
        logits = model(query, support)
        aux = model(query, support, return_aux=True)
        logits_permuted = model(query, support_permuted)

    assert logits.shape == (query.shape[0] * query.shape[1], support.shape[1])
    assert aux["shot_distances"].shape == (query.shape[0] * query.shape[1], support.shape[1], support.shape[2])
    assert aux["redundancy_penalty"].shape == (query.shape[0], support.shape[1])
    assert aux["support_pairwise_sw"].shape == (query.shape[0], support.shape[1], support.shape[2], support.shape[2])
    assert aux["logit_scale"].item() > 0.0
    assert torch.allclose(logits, logits_permuted, atol=1e-6, rtol=0.0)


def test_hssw_multiscale_measure_uses_scale_mass_budgets():
    torch.manual_seed(8)
    model = HierarchicalSupportSlicedWassersteinNet(
        backbone_name="conv64f",
        image_size=64,
        token_weight_hidden=16,
        weighted_sw=True,
        multiscale_grids=(4, 1),
        multiscale_mass_budgets=(0.6, 0.3, 0.1),
        train_num_projections=4,
        eval_num_projections=4,
        train_projection_mode="fixed",
        eval_projection_mode="fixed",
    )
    model.eval()

    images = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        features = model.encode(images)
        token_groups = multiscale_feature_map_to_tokens(features, pooled_grids=model.multiscale_grids)
        tokens, weights = model.encode_token_measures(images)

    token_counts = [group.shape[1] for group in token_groups]
    assert tokens.shape[1] == sum(token_counts)
    assert weights.shape == tokens.shape[:-1]

    for scale_weights, scale_budget in zip(weights.split(token_counts, dim=-1), model.multiscale_mass_budgets):
        expected_budget = torch.full((images.shape[0],), float(scale_budget))
        assert torch.allclose(scale_weights.sum(dim=-1), expected_budget, atol=1e-6, rtol=0.0)


def test_hssw_model_returns_aux_loss_during_training():
    torch.manual_seed(9)
    model = HierarchicalSupportSlicedWassersteinNet(
        backbone_name="conv64f",
        image_size=64,
        token_weight_hidden=16,
        token_weight_temperature=1.5,
        token_weight_mix_alpha=0.8,
        token_mass_reg_weight=0.02,
        weighted_sw=True,
        normalize_tokens_before_sw=True,
        train_num_projections=4,
        eval_num_projections=4,
        train_projection_mode="fixed",
        eval_projection_mode="fixed",
        tau=0.2,
        lambda_cons=0.1,
        lambda_red=0.03,
        logit_scale_init=5.0,
    )
    model.train()

    query = torch.randn(1, 2, 3, 64, 64)
    support = torch.randn(1, 3, 2, 3, 64, 64)
    outputs = model(query, support)

    assert isinstance(outputs, dict)
    assert outputs["logits"].shape == (2, 3)
    assert outputs["aux_loss"].ndim == 0
    assert torch.isfinite(outputs["aux_loss"])
    assert outputs["aux_loss"].item() >= 0.0


def test_hssw_model_factory_builds_and_runs():
    args = SimpleNamespace(
        model="hierarchical_support_sw_net",
        device="cpu",
        image_size=64,
        fewshot_backbone="conv64f",
        hssw_token_weight_hidden=16,
        hssw_token_weight_temperature=1.0,
        hssw_token_weight_mix_alpha=1.0,
        hssw_token_mass_reg_weight=0.0,
        hssw_multiscale_grids="4,1",
        hssw_multiscale_mass_budgets="0.6,0.3,0.1",
        hssw_token_l2norm="true",
        hssw_weighted_sw="true",
        hssw_train_num_projections=4,
        hssw_eval_num_projections=4,
        hssw_sw_p=2.0,
        hssw_train_projection_mode="fixed",
        hssw_eval_projection_mode="fixed",
        hssw_eval_num_repeats=1,
        hssw_projection_seed=7,
        hssw_pairwise_chunk_size=0,
        hssw_tau=0.2,
        hssw_lambda_cons=0.1,
        hssw_lambda_red=0.03,
        hssw_learn_logit_scale="false",
        hssw_logit_scale_init=5.0,
        hssw_logit_scale_max=100.0,
        hssw_eps=1e-6,
    )
    model = build_model_from_args(args)
    model.eval()

    query = torch.randn(1, 2, 3, 64, 64)
    support = torch.randn(1, 3, 2, 3, 64, 64)
    with torch.no_grad():
        logits = model(query, support)
    assert logits.shape == (2, 3)
