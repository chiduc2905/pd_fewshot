import random

import torch

from net.metrics.sliced_wasserstein import SlicedWassersteinDistance
from net.metrics.sliced_wasserstein_paper import PaperSlicedWassersteinDistance
from net.metrics.sliced_wasserstein_weighted import WeightedPaperSlicedWassersteinDistance


def _exact_uniform_1d_cost(x, y, p):
    x = sorted(float(v) for v in x)
    y = sorted(float(v) for v in y)
    nx = len(x)
    ny = len(y)
    i = j = 0
    rem_x = 1.0 / nx
    rem_y = 1.0 / ny
    cost = 0.0
    while i < nx and j < ny:
        mass = min(rem_x, rem_y)
        cost += mass * abs(x[i] - y[j]) ** p
        rem_x -= mass
        rem_y -= mass
        if rem_x <= 1e-12:
            i += 1
            if i < nx:
                rem_x = 1.0 / nx
        if rem_y <= 1e-12:
            j += 1
            if j < ny:
                rem_y = 1.0 / ny
    return cost


def _exact_weighted_1d_cost(x, y, wx, wy, p):
    x = sorted(zip((float(v) for v in x), (float(w) for w in wx)), key=lambda item: item[0])
    y = sorted(zip((float(v) for v in y), (float(w) for w in wy)), key=lambda item: item[0])
    i = j = 0
    rem_x = x[0][1]
    rem_y = y[0][1]
    cost = 0.0
    while i < len(x) and j < len(y):
        mass = min(rem_x, rem_y)
        cost += mass * abs(x[i][0] - y[j][0]) ** p
        rem_x -= mass
        rem_y -= mass
        if rem_x <= 1e-12:
            i += 1
            if i < len(x):
                rem_x = x[i][1]
        if rem_y <= 1e-12:
            j += 1
            if j < len(y):
                rem_y = y[j][1]
    return cost


def test_paper_uniform_1d_cost_matches_exact_transport():
    metric = PaperSlicedWassersteinDistance(
        train_num_projections=8,
        eval_num_projections=8,
        p=2.0,
        reduction="none",
        normalize_inputs=False,
        train_projection_mode="fixed",
        eval_projection_mode="fixed",
    )

    max_gap = 0.0
    for seed in range(50):
        random.seed(seed)
        torch.manual_seed(seed)
        batch = random.randint(1, 4)
        nx = random.randint(2, 7)
        ny = random.randint(2, 7)
        x = torch.randn(batch, nx)
        y = torch.randn(batch, ny)
        got = metric._uniform_wasserstein_1d_cost(x, y)
        ref = torch.tensor(
            [_exact_uniform_1d_cost(x[b].tolist(), y[b].tolist(), 2.0) for b in range(batch)],
            dtype=got.dtype,
        )
        max_gap = max(max_gap, float((got - ref).abs().max()))

    assert max_gap < 1e-5


def test_weighted_1d_cost_matches_exact_transport():
    metric = WeightedPaperSlicedWassersteinDistance(
        train_num_projections=8,
        eval_num_projections=8,
        p=2.0,
        reduction="none",
        normalize_inputs=False,
        train_projection_mode="fixed",
        eval_projection_mode="fixed",
    )

    max_gap = 0.0
    for seed in range(50):
        random.seed(seed)
        torch.manual_seed(seed)
        batch = random.randint(1, 4)
        nx = random.randint(2, 7)
        ny = random.randint(2, 7)
        x = torch.randn(batch, nx)
        y = torch.randn(batch, ny)
        wx = torch.rand(batch, nx)
        wy = torch.rand(batch, ny)
        wx = wx / wx.sum(dim=-1, keepdim=True)
        wy = wy / wy.sum(dim=-1, keepdim=True)
        got = metric.projected_ot_cost(x, y, wx, wy)
        ref = torch.tensor(
            [
                _exact_weighted_1d_cost(x[b].tolist(), y[b].tolist(), wx[b].tolist(), wy[b].tolist(), 2.0)
                for b in range(batch)
            ],
            dtype=got.dtype,
        )
        max_gap = max(max_gap, float((got - ref).abs().max()))

    assert max_gap < 1e-5


def test_weighted_matches_paper_under_uniform_weights():
    paper = PaperSlicedWassersteinDistance(
        train_num_projections=16,
        eval_num_projections=16,
        p=2.0,
        reduction="none",
        normalize_inputs=False,
        train_projection_mode="fixed",
        eval_projection_mode="fixed",
    )
    weighted = WeightedPaperSlicedWassersteinDistance(
        train_num_projections=16,
        eval_num_projections=16,
        p=2.0,
        reduction="none",
        normalize_inputs=False,
        train_projection_mode="fixed",
        eval_projection_mode="fixed",
    )
    paper.eval()
    weighted.eval()

    max_gap = 0.0
    for seed in range(50):
        random.seed(seed)
        torch.manual_seed(seed)
        num_query = random.randint(1, 4)
        way_num = random.randint(1, 4)
        feature_dim = random.randint(2, 6)
        query_tokens = random.randint(2, 7)
        support_tokens = random.randint(2, 7)
        query = torch.randn(num_query, query_tokens, feature_dim)
        support = torch.randn(way_num, support_tokens, feature_dim)
        query_weights = torch.full((num_query, query_tokens), 1.0 / float(query_tokens))
        support_weights = torch.full((way_num, support_tokens), 1.0 / float(support_tokens))
        got = weighted.pairwise_distance(
            query,
            support,
            query_weights=query_weights,
            support_weights=support_weights,
            reduction="none",
        )
        ref = paper.pairwise_distance(query, support, reduction="none")
        max_gap = max(max_gap, float((got - ref).abs().max()))

    assert max_gap < 1e-5


def test_pairwise_weighted_matches_explicit_loop():
    metric = WeightedPaperSlicedWassersteinDistance(
        train_num_projections=12,
        eval_num_projections=12,
        p=2.0,
        reduction="none",
        normalize_inputs=True,
        train_projection_mode="fixed",
        eval_projection_mode="fixed",
    )
    metric.eval()

    torch.manual_seed(0)
    query = torch.randn(3, 5, 7)
    support = torch.randn(4, 6, 7)
    query_weights = torch.rand(3, 5)
    support_weights = torch.rand(4, 6)
    query_weights = query_weights / query_weights.sum(dim=-1, keepdim=True)
    support_weights = support_weights / support_weights.sum(dim=-1, keepdim=True)

    pairwise = metric.pairwise_distance(
        query,
        support,
        query_weights=query_weights,
        support_weights=support_weights,
        reduction="none",
    )
    loop = []
    for query_idx in range(query.shape[0]):
        row = []
        for class_idx in range(support.shape[0]):
            row.append(
                metric(
                    query[query_idx : query_idx + 1],
                    support[class_idx : class_idx + 1],
                    query_weights=query_weights[query_idx : query_idx + 1],
                    support_weights=support_weights[class_idx : class_idx + 1],
                    reduction="none",
                ).squeeze(0)
            )
        loop.append(torch.stack(row))
    loop = torch.stack(loop)

    assert torch.allclose(pairwise, loop, atol=1e-6, rtol=0.0)


def test_resample_mode_honors_projection_seed_independently_of_global_rng():
    query = torch.randn(2, 5, 3)
    support = torch.randn(2, 6, 3)

    metric_a = PaperSlicedWassersteinDistance(
        train_num_projections=8,
        eval_num_projections=8,
        p=2.0,
        reduction="none",
        normalize_inputs=False,
        train_projection_mode="resample",
        eval_projection_mode="resample",
        projection_seed=123,
    )
    metric_b = PaperSlicedWassersteinDistance(
        train_num_projections=8,
        eval_num_projections=8,
        p=2.0,
        reduction="none",
        normalize_inputs=False,
        train_projection_mode="resample",
        eval_projection_mode="resample",
        projection_seed=123,
    )
    metric_a.train()
    metric_b.train()

    torch.manual_seed(0)
    out_a = metric_a(query, support, reduction="none")
    torch.manual_seed(999)
    out_b = metric_b(query, support, reduction="none")

    assert torch.allclose(out_a, out_b, atol=1e-6, rtol=0.0)


def test_legacy_sw_is_distinct_from_paper_style_estimator():
    legacy = SlicedWassersteinDistance(
        num_projections=16,
        p=2.0,
        reduction="none",
        normalize_inputs=False,
    )
    paper = PaperSlicedWassersteinDistance(
        train_num_projections=16,
        eval_num_projections=16,
        p=2.0,
        reduction="none",
        normalize_inputs=False,
        train_projection_mode="fixed",
        eval_projection_mode="fixed",
    )
    legacy.eval()
    paper.eval()

    torch.manual_seed(7)
    query = torch.randn(3, 5, 4)
    support = torch.randn(3, 7, 4)
    legacy_dist = legacy(query, support, reduction="none")
    paper_dist = paper(query, support, reduction="none")

    assert not torch.allclose(legacy_dist, paper_dist, atol=1e-4, rtol=0.0)
