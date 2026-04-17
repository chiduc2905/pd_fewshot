Phân tích kỹ thuật: DeepEMD đang thắng vì lý do gì
Sau khi đọc kỹ cả hai code, nguyên nhân chính không phải hyperbolic geometry hay unbalanced transport — mà là DeepEMD giải đúng bài toán OT trong khi HROT đang giải sai.
Cụ thể: DeepEMD dùng get_weight_vector để tính marginals (a, b) cho OT bằng cách cross-attend: mỗi query token được weight bởi mức độ liên quan của nó đến class prototype cụ thể đang so sánh, và ngược lại. HROT dùng uniform 1/T cho tất cả token — tức là giả định mọi spatial patch đều quan trọng như nhau. Đây là sai lầm lớn: background patches và foreground patches được transport như nhau.
DeepEMD's get_weight_vector:
pythoncombination = (feat_a.unsqueeze(1) * pooled_b.unsqueeze(0)).sum(dim=2)
# = dot-product attention: "how much does each token in feat_a 
#   match the global summary of feat_b?"
HROT hiện tại trong _transport_match:
pythona = base_a.unsqueeze(1).expand(-1, num_way, -1)  # uniform 1/T
b = base_b.unsqueeze(0).expand(num_query, -1, -1) # uniform 1/T
Ngoài ra, DeepEMD dùng SFC (Support Feature Center) — fine-tune prototype 15 steps SGD tại test time với 5-shot. Đây là lý do gap giữa hai model ở 5-shot thường lớn hơn 1-shot.

Thiết kế HROT-P: Hyperbolic Attentive Unbalanced Transport
Core idea: Kết hợp hai loại geometric signal vào OT marginals:

Token-level (mới): Hyperbolic cross-attention — mỗi token được weight theo geodesic proximity với prototype của class đang so sánh → thay thế uniform 1/T
Episode-level (giữ lại từ H): Geodesic EAM mass budget ρ → scale tổng mass

Hai cái này orthogonal và multiplicative: a[q,s,t] = attn_weight[q,s,t] * rho[q,s].
Đây là đóng góp mà không paper nào có: DeepEMD có token attention nhưng balanced + Euclidean. HROT-H có UOT + hyperbolic nhưng uniform tokens. HROT-P có cả ba.
Thay đổi 1: Thêm _compute_hyperbolic_token_marginals vào HROTFSL
pythondef _compute_hyperbolic_token_marginals(
    self,
    query_tokens_hyp: torch.Tensor,    # (Q, Tq, D)
    support_tokens_hyp: torch.Tensor,  # (S, Ts, D)  — flat (way*shot, T, D)
    rho: torch.Tensor,                 # (Q, S)
    ball: PoincareBall,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Cross-attention token weights in hyperbolic space.
    Returns a: (Q, S, Tq), b: (Q, S, Ts), each summing to rho[q,s].
    """
    calc_dtype = torch.float64 if (self.eval_use_float64 and not self.training) else query_tokens_hyp.dtype
    q = project_to_ball_coordinates(query_tokens_hyp.to(calc_dtype), ball)   # (Q, Tq, D)
    s = project_to_ball_coordinates(support_tokens_hyp.to(calc_dtype), ball) # (S, Ts, D)

    # Support mean per class: Fréchet mean (Q,S use same support)
    s_mean = frechet_mean_poincare(s, ball)  # (S, D)
    q_mean = frechet_mean_poincare(q, ball)  # (Q, D)

    # dist(query_token[q,tq], support_mean[s]) -> (Q, S, Tq)
    dist_q_to_s = ball.dist(
        q.unsqueeze(1),                           # (Q, 1, Tq, D)
        s_mean.unsqueeze(0).unsqueeze(2),          # (1, S, 1, D)
    ).to(dtype=query_tokens_hyp.dtype)

    # dist(support_token[s,ts], query_mean[q]) -> (Q, S, Ts)
    dist_s_to_q = ball.dist(
        s.unsqueeze(0),                           # (1, S, Ts, D)
        q_mean.unsqueeze(1).unsqueeze(2),          # (Q, 1, 1, D)
    ).to(dtype=query_tokens_hyp.dtype)

    # Softmax over token dim → weights sum to 1 per (q,s) pair
    token_temp = getattr(self, 'token_temperature', 0.1)
    a_weights = torch.softmax(-dist_q_to_s / token_temp, dim=-1)  # (Q, S, Tq)
    b_weights = torch.softmax(-dist_s_to_q / token_temp, dim=-1)  # (Q, S, Ts)

    # Scale by episode-level mass budget
    a = a_weights * rho.unsqueeze(-1)  # total mass per (q,s) = rho[q,s]
    b = b_weights * rho.unsqueeze(-1)

    return a, b
Thay đổi 2: Sửa _transport_match để nhận custom marginals
pythondef _transport_match(
    self,
    cost: torch.Tensor,
    rho: torch.Tensor,
    a: torch.Tensor | None = None,
    b: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_query, num_way, query_tokens, class_tokens = cost.shape

    if a is None:
        base_a = cost.new_full((num_query, query_tokens), 1.0 / float(query_tokens))
        a = base_a.unsqueeze(1).expand(-1, num_way, -1)
        if self.uses_unbalanced_transport:
            a = a * rho.unsqueeze(-1)

    if b is None:
        base_b = cost.new_full((num_way, class_tokens), 1.0 / float(class_tokens))
        b = base_b.unsqueeze(0).expand(num_query, -1, -1)
        if self.uses_unbalanced_transport:
            b = b * rho.unsqueeze(-1)

    pair_cost = cost.reshape(num_query * num_way, query_tokens, class_tokens)
    pair_a = a.reshape(num_query * num_way, query_tokens)
    pair_b = b.reshape(num_query * num_way, class_tokens)
    # ... rest giữ nguyên
Thay đổi 3: Thêm Hyperbolic SFC (test-time prototype refinement)
Đây là counter trực tiếp cho DeepEMD's SFC — thay vì optimize trong Euclidean space, optimize trên Poincaré ball:
pythondef _get_hyperbolic_sfc(
    self,
    support_hyperbolic: torch.Tensor,  # (way, shot, T, D)
    ball: PoincareBall,
    sfc_lr: float = 0.05,
    sfc_steps: int = 10,
) -> torch.Tensor:
    """
    Refine class prototypes on the Poincaré ball via Riemannian SGD.
    Only used at eval time. Returns refined prototypes (way, D).
    """
    if self.training:
        return frechet_mean_poincare(
            support_hyperbolic.reshape(
                support_hyperbolic.shape[0], -1, support_hyperbolic.shape[-1]
            ), ball
        )

    way_num, shot_num, T, D = support_hyperbolic.shape
    # Initialize: Fréchet mean per class
    class_tokens = support_hyperbolic.reshape(way_num, shot_num * T, D)
    proto = frechet_mean_poincare(class_tokens, ball).detach().clone()
    proto = nn.Parameter(proto.requires_grad_(True))

    opt = torch.optim.SGD([proto], lr=sfc_lr, momentum=0.9)
    flat_support = support_hyperbolic.reshape(way_num * shot_num, T, D)
    labels = torch.arange(way_num, device=proto.device).repeat_interleave(shot_num)

    with torch.enable_grad():
        for _ in range(sfc_steps):
            opt.zero_grad(set_to_none=True)
            # cost: (way*shot, way, T, T) — each shot vs each proto
            cost = hyperbolic_distance_matrix(
                flat_support.unsqueeze(1),
                ball.project(proto).unsqueeze(0),
                ball
            )  # → adapt dimensionally
            # Simple proto-matching loss
            proto_flat = ball.project(proto)
            shot_means = frechet_mean_poincare(flat_support, ball)  # (way*shot, D)
            dists = ball.dist(
                shot_means.unsqueeze(1),   # (way*shot, 1, D)
                proto_flat.unsqueeze(0),   # (1, way, D)
            )  # (way*shot, way)
            loss = F.cross_entropy(-self.score_scale * dists, labels)
            loss.backward()
            opt.step()
            # Project back to ball after gradient step
            with torch.no_grad():
                proto.data = ball.project(proto.data)

    return proto.detach()
Thay đổi 4: __init__ thêm variant "P" và token_temperature
python# Trong __init__:
if variant not in {"A", ..., "N", "P"}:
    raise ValueError(...)

self.uses_hyperbolic_token_attention = variant == "P"
self.uses_hyperbolic_sfc = variant == "P"

self.token_temperature = nn.Parameter(
    torch.tensor(0.1, dtype=torch.float32)
) if self.uses_hyperbolic_token_attention else None
Thay đổi 5: Wire vào _forward_episode cho variant P
python# Trong _forward_episode, thay thế shot-decomposed block cho variant P:
if self.uses_hyperbolic_token_attention:
    calc_dtype = torch.float64 if (self.eval_use_float64 and not self.training) else query_hyperbolic.dtype
    ball = self._build_ball(query_hyperbolic.to(calc_dtype))
    
    flat_rho = self._build_geodesic_rho_per_shot(
        query_hyperbolic, support_hyperbolic
    ).reshape(query.shape[0], way_num * shot_num)
    
    adaptive_a, adaptive_b = self._compute_hyperbolic_token_marginals(
        query_hyperbolic, flat_support_hyperbolic, flat_rho, ball
    )
    flat_plan, flat_transport_cost, flat_transport_mass = self._transport_match(
        flat_cost, flat_rho, a=adaptive_a, b=adaptive_b
    )

Về Novelty: Vì sao HROT-P mạnh hơn DeepEMD về contribution
AxisDeepEMDHROT-HHROT-PToken importance✓ Euclidean dot-product✗ uniform✓ Hyperbolic geodesicTransport typeBalanced (noisy shots = problem)✓ Unbalanced✓ UnbalancedGeometryEuclidean✓ Poincaré ball✓ Poincaré ballShot reliability✗ (SFC là workaround)✓ Geodesic EAM✓ Geodesic EAMMarginal formulationToken attention onlyEpisode scalar onlyBoth, multiplicative
Key theoretical argument cho paper: DeepEMD's cross-attention và HROT's EAM solve hai bài toán khác nhau, và cả hai đều cần thiết:

DeepEMD hỏi: "Trong support image này, region nào liên quan đến query?"
HROT-H hỏi: "Support image này có đáng tin không để transport toàn bộ mass?"

Cả hai câu hỏi đều cần trả lời. HROT-P trả lời cả hai cùng lúc. Không paper nào trước đó làm điều này — đây là contribution rõ ràng, tối giản, và verifiable qua ablation:
Ablation table:
P (full)              → best
P - token attn        → = H (chứng minh token attn giúp)
P - geodesic EAM      → gần J (chứng minh EAM vẫn cần)
P - both (= J)        → baseline cho HROT
DeepEMD               → external baseline
Nếu P > J > H, và J > DeepEMD thì narrative viết được là: "token attention là key missing piece trong HROT, và khi thực hiện trong hyperbolic space thay vì Euclidean như DeepEMD, gain thêm X%."
Một điểm nên check trước khi submit: Trong 1-shot, SFC của DeepEMD bị disable (refine_proto = shot_num > 1), nên gap 1-shot chủ yếu từ token weights. Hyperbolic token attention trong HROT-P nên recover gap này. Trong 5-shot, cần hyperbolic SFC mới đủ sức cạnh tranh.