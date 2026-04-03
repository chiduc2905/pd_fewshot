import random

import torch

from net.metrics.hssw_sliced_wasserstein import HSSWSlicedWassersteinDistance


def _exact_weighted_1d_cost(x, y, wx, wy, p):
    x = sorted(zip((float(v) for v in x), (float(w) for w in wx)), key=lambda item: item[0])
    y = sorted(zip((float(v) for v in y), (float(w) for w in wy)), key=lambda item: item[0])
    left_idx = 0
    right_idx = 0
    left_mass = x[0][1]
    right_mass = y[0][1]
    cost = 0.0
    while left_idx < len(x) and right_idx < len(y):
        mass = min(left_mass, right_mass)
        cost += mass * abs(x[left_idx][0] - y[right_idx][0]) ** p
        left_mass -= mass
        right_mass -= mass
        if left_mass <= 1e-12:
            left_idx += 1
            if left_idx < len(x):
                left_mass = x[left_idx][1]
        if right_mass <= 1e-12:
            right_idx += 1
            if right_idx < len(y):
                right_mass = y[right_idx][1]
    return cost


def _build_metric(weighted=True, pairwise_chunk_size=None):
    metric = HSSWSlicedWassersteinDistance(
        train_num_projections=8,
        eval_num_projections=8,
        p=2.0,
        weighted=weighted,
        normalize_tokens=True,
        train_projection_mode="fixed",
        eval_projection_mode="fixed",
        projection_seed=23,
        pairwise_chunk_size=pairwise_chunk_size,
    )
    metric.eval()
    return metric


def test_hssw_weighted_1d_transport_matches_exact_manual_solver():
    metric = HSSWSlicedWassersteinDistance(
        train_num_projections=1,
        eval_num_projections=1,
        p=2.0,
        weighted=True,
        normalize_tokens=False,
        train_projection_mode="fixed",
        eval_projection_mode="fixed",
        projection_seed=23,
    )
    metric.eval()
    max_gap = 0.0
    for seed in range(40):
        random.seed(seed)
        torch.manual_seed(seed)
        batch = random.randint(1, 4)
        nx = random.randint(2, 6)
        ny = random.randint(2, 6)
        x = torch.randn(batch, nx, 1)
        y = torch.randn(batch, ny, 1)
        wx = torch.rand(batch, nx)
        wy = torch.rand(batch, ny)
        wx = wx / wx.sum(dim=-1, keepdim=True)
        wy = wy / wy.sum(dim=-1, keepdim=True)
        got = metric(x, y, x_weights=wx, y_weights=wy, reduction="none")
        ref = torch.tensor(
            [_exact_weighted_1d_cost(x[b, :, 0].tolist(), y[b, :, 0].tolist(), wx[b].tolist(), wy[b].tolist(), 2.0) for b in range(batch)],
            dtype=got.dtype,
        ).pow(0.5)
        max_gap = max(max_gap, float((got - ref).abs().max()))
    assert max_gap < 1e-5


def test_hssw_pairwise_distance_matches_explicit_loop():
    torch.manual_seed(0)
    metric = _build_metric(weighted=True)
    query = torch.randn(3, 5, 7)
    support = torch.randn(4, 6, 7)
    query_weights = torch.rand(3, 5)
    support_weights = torch.rand(4, 6)

    pairwise = metric.pairwise_distance(
        query_tokens=query,
        support_tokens=support,
        query_weights=query_weights,
        support_weights=support_weights,
        reduction="none",
    )
    rows = []
    for query_idx in range(query.shape[0]):
        cols = []
        for support_idx in range(support.shape[0]):
            cols.append(
                metric(
                    query[query_idx : query_idx + 1],
                    support[support_idx : support_idx + 1],
                    x_weights=query_weights[query_idx : query_idx + 1],
                    y_weights=support_weights[support_idx : support_idx + 1],
                    reduction="none",
                ).squeeze(0)
            )
        rows.append(torch.stack(cols))
    loop = torch.stack(rows)
    assert torch.allclose(pairwise, loop, atol=1e-6, rtol=0.0)


def test_hssw_resample_mode_is_seeded_independently_of_global_rng():
    query = torch.randn(2, 5, 3)
    support = torch.randn(2, 6, 3)
    metric_a = HSSWSlicedWassersteinDistance(
        train_num_projections=8,
        eval_num_projections=8,
        p=2.0,
        weighted=True,
        normalize_tokens=True,
        train_projection_mode="resample",
        eval_projection_mode="resample",
        projection_seed=29,
    )
    metric_b = HSSWSlicedWassersteinDistance(
        train_num_projections=8,
        eval_num_projections=8,
        p=2.0,
        weighted=True,
        normalize_tokens=True,
        train_projection_mode="resample",
        eval_projection_mode="resample",
        projection_seed=29,
    )
    metric_a.train()
    metric_b.train()

    torch.manual_seed(0)
    out_a = metric_a(query, support, reduction="none")
    torch.manual_seed(999)
    out_b = metric_b(query, support, reduction="none")
    assert torch.allclose(out_a, out_b, atol=1e-6, rtol=0.0)


def test_hssw_unweighted_mode_ignores_provided_weights():
    torch.manual_seed(1)
    metric = _build_metric(weighted=False)
    query = torch.randn(2, 4, 5)
    support = torch.randn(3, 4, 5)
    arbitrary_query_weights = torch.rand(2, 4)
    arbitrary_support_weights = torch.rand(3, 4)
    uniform_query_weights = torch.full((2, 4), 0.25)
    uniform_support_weights = torch.full((3, 4), 0.25)

    got = metric.pairwise_distance(
        query_tokens=query,
        support_tokens=support,
        query_weights=arbitrary_query_weights,
        support_weights=arbitrary_support_weights,
        reduction="none",
    )
    ref = metric.pairwise_distance(
        query_tokens=query,
        support_tokens=support,
        query_weights=uniform_query_weights,
        support_weights=uniform_support_weights,
        reduction="none",
    )
    assert torch.allclose(got, ref, atol=1e-6, rtol=0.0)


def test_hssw_pairwise_chunking_matches_full_pairwise():
    torch.manual_seed(2)
    full_metric = _build_metric(weighted=True, pairwise_chunk_size=None)
    chunked_metric = _build_metric(weighted=True, pairwise_chunk_size=2)
    query = torch.randn(4, 5, 6)
    support = torch.randn(5, 5, 6)
    query_weights = torch.rand(4, 5)
    support_weights = torch.rand(5, 5)

    full = full_metric.pairwise_distance(
        query_tokens=query,
        support_tokens=support,
        query_weights=query_weights,
        support_weights=support_weights,
        reduction="none",
    )
    chunked = chunked_metric.pairwise_distance(
        query_tokens=query,
        support_tokens=support,
        query_weights=query_weights,
        support_weights=support_weights,
        reduction="none",
    )
    assert torch.allclose(full, chunked, atol=1e-6, rtol=0.0)
