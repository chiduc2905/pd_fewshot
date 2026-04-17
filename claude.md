🔴 Yếu: Marginals đồng đều (uniform per-token)
a_{q,c,k}(i) = ρ_{q,c,k} / L    (mọi token i đều bằng nhau)
b_{q,c,k}(j) = ρ_{q,c,k} / L
Điều này có nghĩa là tất cả L token trong một shot đều có trọng số bằng nhau trước khi OT giải. Mass ρ chỉ scale toàn bộ marginal — không phân biệt token nào quan trọng hơn trong time-frequency scalogram. DeepEMD dùng cross-reference để weight từng patch, đây là lý do chính nó mạnh.

🟡 Trung bình: EAM với 4 geodesic features
EAM hiện tại dùng 4 scalar:
g_{q,c,k} = [d(μ_q, μ_{c,k}),  d(μ_{c,k}, μ_c),  σ²(Z^H_q),  σ²(Z^H_{c,k})]
Ưu điểm: interpretable, reviewer-friendly.
Nhược điểm: 4 scalar thô qua MLP không capture được fine-grained token-level compatibility. MLP trên 4 số này rất hạn chế về expressive power. Ngoài ra σ² tính trên toàn bộ token set (mean field approximation) — mất đi spatial locality.

🟡 Trung bình: Shot aggregation bằng mean đơn giản
ℓ_{q,c} = (1/K) Σ_k ℓ_{q,c,k}
Simple mean không phân biệt shot tốt/xấu. Trong khi g_{q,c,k} đã có d(μ_{c,k}, μ_c) để đo shot outlierness, thông tin đó không được dùng lại cho aggregation — chỉ dùng để predict ρ. Đây là wasted signal.

Phân tích chi tiết và đề xuất cụ thể
2. Non-uniform token attention marginals
Hiện tại a(i) = ρ/L đồng đều — tức là sau khi predict được ρ, mọi token vẫn được coi là như nhau. Cần thêm một lớp token-level attention weight:
a_{q,c,k}(i) = ρ_{q,c,k} · softmax_i(v^⊤ z^H_{q,i} / √d)
b_{q,c,k}(j) = ρ_{q,c,k} · softmax_j(v^⊤ z^H_{c,k,j} / √d)
Đây là hyperbolic self-attention đơn giản — dùng Möbius linear map để project trước khi dot product. Lý do phù hợp lý thuyết: trong PD scalogram, discriminative evidence tập trung ở một số vùng time-frequency cụ thể, không phải tất cả L token. Điều này giải quyết đúng điểm mạnh của DeepEMD (cross-reference weighting) bằng cơ chế native trong UOT marginal.
3. Hyperbolic cross-attention EAM
Thay MLP(4 scalars) bằng attention từ μ_q đến {μ_{c,k}}:
attn_{q,c,k} = softmax_k(  d_𝔻c(μ_q, μ_{c,k})⁻¹ / τ  )
ρ_{q,c,k} = clip(MLP(concat(g_{q,c,k}, attn_{q,c,k})), ρ_min, 1)
Hoặc mạnh hơn: dùng gyrovector-space attention (Shimizu et al. 2021) để compute compatibility trực tiếp trong Poincaré ball. Điều này khác biệt rõ với Variant G dùng tangent statistics thuần, và cho EAM signal richer hơn vì nó sees tất cả shots simultaneously.

Tier 2 — Nên làm (accuracy gain đáng kể)
4. Geodesic-weighted shot aggregation
Thay mean bằng:
w_{q,c,k} = softmax_k( MLP_w(g_{q,c,k}) )
ℓ_{q,c} = Σ_k w_{q,c,k} · ℓ_{q,c,k}
Signal d(μ_{c,k}, μ_c) đã có trong g_{q,c,k} — chỉ cần route nó ra thành aggregation weight thay vì bỏ vào ρ rồi ignore. Không tốn compute thêm, chỉ thêm một MLP_w nhỏ. Reviewer sẽ thấy đây là khai thác triệt để hơn của geodesic EAM.
5. Per-shot adaptive threshold
Thay scalar T toàn cục bằng:
T_{q,c,k} = softplus(MLP_T(g_{q,c,k}))
Giờ logit trở thành:
ℓ_{q,c,k} = s · m_{q,c,k} · (T_{q,c,k} - D̄_{q,c,k})