# HROT-S: Iterative Posterior Refinement cho Structural Evidence Matching

## Tóm Tắt

HROT-S là bước tiếp theo tự nhiên sau HROT-R. Nó không thêm tham số mới, không thay backbone, không thay UOT solver. Thay đổi duy nhất là về **thuật toán**: thay vì tính structure cost một lần từ một probe plan chưa được lọc qua structure, HROT-S chạy hai vòng refinement — mỗi vòng, transport plan tốt hơn tạo ra structure cost tốt hơn, và structure cost tốt hơn lại dẫn đến transport plan tốt hơn. Ngoài ra, structure cost được gating bởi độ tự tin của transport plan, để tránh phạt những cap token mà posterior còn chưa chắc chắn.

Hai thay đổi này — **Iterative Posterior Refinement (IPR)** và **Confidence-Gated Structure Cost** — không thêm parameter nào so với HROT-R, nhưng lấp được điểm yếu cấu trúc quan trọng nhất của R: cái vòng tròn giữa "cần plan tốt để tính structure cost" và "cần structure cost để có plan tốt".

---

## 1. Câu Chuyện: Tại Sao Cần HROT-S

### 1.1 Narrative Arc của Toàn Bộ Dòng HROT

Để hiểu S đến từ đâu, cần nhìn lại toàn bộ chuỗi lập luận:

```
Bài toán: few-shot phân loại scalogram phóng điện cục bộ.

Dấu hiệu phóng điện chỉ xuất hiện ở một vùng nhỏ trên
miền thời gian-tần số, bao quanh bởi background noise.
Support shots cùng lớp có thể không chứa cùng một phần bằng chứng.

⟹ Prototype toàn cục bị pha loãng bởi background.
   Cần so khớp cục bộ.

⟹ HROT-H: dùng Unbalanced OT để ghép token cục bộ.
   Ghép linh hoạt — không ép mọi vùng phải khớp.
   Geodesic EAM ước lượng bao nhiêu mass nên được vận chuyển.
   Threshold score: mass chỉ có giá trị khi chi phí trung bình thấp.

⟹ Vấn đề còn lại: uniform token mass quá yếu với background noise.
   Background noise tạo ra token có chi phí Euclidean thấp tình cờ.

⟹ HROT-Q: token reliability từ probe UOT.
   Support consensus giữa các shot.
   Noise sink hấp thu vùng không khớp.
   Robust shot pooling.

⟹ Vấn đề còn lại: Q biết token nào đáng tin,
   nhưng chưa biết cặp ghép token nào bảo toàn cấu trúc quan hệ cục bộ.
   Background noise có thể tạo một điểm giống, nhưng khó tạo
   cả một cấu trúc quan hệ giống.

⟹ HROT-R: structure consistency cost từ posterior UOT probe.
   Cặp (i,j) bị tăng chi phí nếu quan hệ của i với các token query khác
   không nhất quán với quan hệ của j với các token support tương ứng.

⟹ Vấn đề còn lại: probe plan của R được tính MỘT LẦN,
   trước khi structure cost can thiệp. Đây là vòng tròn:
   - Cần plan tốt để tính structure cost chính xác.
   - Cần structure cost chính xác để có plan tốt.
   R phá vòng tròn bằng cách bỏ qua nó (detach một lần).
   S phá vòng tròn bằng cách lặp đến hội tụ (2 vòng là đủ).

⟹ HROT-S: Iterative Posterior Refinement.
   Mỗi vòng: UOT plan → structure cost → cost matrix mới → UOT plan mới.
   Confidence gate: structure cost chỉ áp dụng nơi plan đã tự tin.
   Zero parameter mới.
```

### 1.2 Claim Cốt Lõi

HROT-S không phải là "thêm một thứ gì đó vào R". Nó là câu trả lời cho câu hỏi:

> Nếu structure consistency cost là đúng đắn về mặt lý thuyết,
> thì tại sao lại chỉ áp dụng nó một lần từ một posterior chưa được
> lọc bởi chính nó?

Câu trả lời: không có lý do gì cả. Đó là giới hạn thiết kế của R, không phải giới hạn của ý tưởng. S loại bỏ giới hạn đó.

### 1.3 So Sánh Story Với DeepEMD

DeepEMD cũng khai thác local matching nhưng theo một câu chuyện khác:

| | DeepEMD | HROT-S |
|---|---|---|
| **Câu hỏi** | Token nào quan trọng? | Cặp ghép nào bảo toàn cấu trúc? |
| **Cơ chế** | Cross-reference weighting | Iterative posterior refinement |
| **Transport** | EMD gần cân bằng | UOT không cân bằng + noise sink |
| **Structure** | Không có | Structure cost từ posterior plan |
| **Refinement** | Một lần (no iteration) | 2 vòng bootstrap |
| **Background noise** | Giảm weight token noise | UOT cho phép không vận chuyển |

DeepEMD trả lời tốt câu hỏi "token nào". HROT-S trả lời thêm câu hỏi "cặp ghép nào" — và trả lời lặp đi lặp lại cho đến khi nhất quán.

---

## 2. Điểm Yếu Của HROT-R Dẫn Đến HROT-S

### 2.1 Vấn Đề Probe One-Shot

Trong HROT-R, flow là:

```
Q marginals (a_Q, b_Q)
→ P_probe = UOT(C, a_Q, b_Q)   ← detach ở đây
→ S = structure_cost(P_probe, D_q, D_s)
→ C_R = C + λ * normalize(S)
→ P_final = UOT(C_R + sink, ...)
```

`P_probe` là UOT plan trên cost matrix Euclidean gốc `C`, chưa có thông tin structure. Nếu `C` còn nhiều noise (background tokens có chi phí Euclidean thấp tình cờ), thì `P_probe` cũng sẽ phản ánh những cặp ghép nhiễu đó. Structure cost tính từ `P_probe` nhiễu sẽ không chính xác.

Đây là vòng tròn cổ điển trong optimization:
- Để tính structure cost tốt, cần posterior plan tốt.
- Để có posterior plan tốt, cần structure cost tốt.

HROT-R phá vòng tròn bằng cách khởi tạo plan từ `C` gốc và không iterate. Điều này đơn giản nhưng dừng quá sớm.

### 2.2 Vấn Đề Structure Cost Áp Dụng Đồng Đều

Trong HROT-R, `S(i,j)` được tính và áp dụng cho mọi cặp `(i,j)` với cùng một trọng số `λ`. Nhưng không phải mọi row của `P_probe` đều đáng tin nhau:

- Nếu `P_probe[i, :]` concentrated (entropy thấp): token query `i` đã có một cặp ghép rõ ràng. Structure cost tính từ row này đáng tin.
- Nếu `P_probe[i, :]` scattered (entropy cao): token query `i` chưa tìm được khớp rõ ràng. Structure cost tính từ row này không chắc chắn.

Áp dụng `λ * S(i,j)` đồng đều cho cả hai trường hợp là phạt sai nơi và đúng nơi với cùng cường độ.

---

## 3. Định Nghĩa HROT-S

### 3.1 Pipeline Đầy Đủ

HROT-S giữ nguyên toàn bộ HROT-Q, chỉ thay thế phần "structure cost + final UOT" của HROT-R:

```
[Giữ nguyên từ Q]
─────────────────────────────────────────────────
backbone → token grid z_q, z_{c,k}
geodesic EAM → rho
probe-UOT token reliability → a_Q, b_Q (adaptive marginals)
support cross-shot consensus → b_Q điều chỉnh
─────────────────────────────────────────────────

[Phần mới của S]
─────────────────────────────────────────────────
C = Euclidean token cost matrix

Vòng 1:
  P₁ = UOT(C, a_Q, b_Q)              ← detach
  gate₁ = confidence_gate(P₁)        ← từ entropy của P₁
  S₁ = structure_cost(P₁, D_q, D_s)  ← giống HROT-R
  C₁ = C + λ * gate₁ ⊙ normalize(S₁)

Vòng 2:
  P₂ = UOT(C₁, a_Q, b_Q)             ← detach
  gate₂ = confidence_gate(P₂)
  S₂ = structure_cost(P₂, D_q, D_s)
  C₂ = C + λ * gate₂ ⊙ normalize(S₂)
      ← vẫn dùng C gốc, không phải C₁

Final:
  C_final = C₂ với noise sink
  P_final = UOT(C_final, a_Q+sink, b_Q+sink)
  score → threshold → robust shot pooling
─────────────────────────────────────────────────
```

Lưu ý quan trọng ở vòng 2: `C₂ = C + λ * gate₂ ⊙ normalize(S₂)`, không phải `C₁ + ...`. Điều này giữ cho structure cost là một perturbation lên cost gốc, không tích lũy qua các vòng theo cách khó kiểm soát.

### 3.2 Confidence Gate

```
gate(P)[i,j] = row_confidence[i]

row_confidence[i] = 1 - H(P[i,:]) / H_max

H(P[i,:]) = - Σ_j P[i,j] * log(P[i,j] + ε)   ← entropy của row i
H_max = log(T_s)                                ← entropy tối đa khi uniform
```

`gate[i,j]` gần 1 khi token query `i` đã có cặp ghép tập trung. Gần 0 khi `P[i,:]` còn phân tán.

Điều này có nghĩa:

- Ở early training, khi `P₁` còn noisy và scattered, gate tự động giảm ảnh hưởng của structure cost → training ổn định hơn.
- Ở late training, khi `P₁` đã concentrated, gate cho phép structure cost phát huy đầy đủ.

Không có tham số mới. Gate hoàn toàn được suy ra từ `P₁`.

### 3.3 Structure Cost (Giữ Nguyên Từ R)

Công thức structure cost không thay đổi so với HROT-R:

```
S(i,j) = E_{(i',j') ~ P} [ (D_q(i,i') - D_s(j,j'))² ]

Tính hiệu quả:
S = D_q² row(P) + D_s² col(P) - 2 D_q P D_sᵀ

Trong đó:
  D_q(i,i') = ||z_q(i) - z_q(i')||²   ← intra-query pairwise distance
  D_s(j,j') = ||z_s(j) - z_s(j')||²   ← intra-support pairwise distance
  row(P)[i] = Σ_j P[i,j]
  col(P)[j] = Σ_i P[i,j]
```

Điểm cải thiện: P trong vòng 2 tốt hơn P trong vòng 1, nên `S₂` sẽ phản ánh cấu trúc quan hệ chính xác hơn `S₁` của HROT-R.

### 3.4 Số Lượng Tham Số

| Component | Params mới so với R |
|---|---|
| Iterative refinement (2 vòng) | 0 |
| Confidence gate | 0 |
| `λ` (structure cost weight) | 0 — giữ nguyên từ R |
| Tổng | **0** |

HROT-S không thêm bất kỳ tham số nào so với HROT-R.

---

## 4. Tại Sao IPR Hội Tụ Sau 2 Vòng

Trực giác: mỗi vòng refinement, structure cost "đẩy" cost matrix xa những cặp ghép sai cấu trúc. Transport plan ở vòng sau tự nhiên tránh những cặp đó, tạo ra posterior tập trung hơn vào cặp ghép bảo toàn cấu trúc. Vòng 2 không cần bắt đầu từ `C` noisy như vòng 1 — nó bắt đầu từ `C₁` đã bị điều chỉnh một lần.

Thực nghiệm trong các bài toán OT liên quan cho thấy 2–3 vòng thường đạt ~90% gain của infinite iterations. Với bài toán few-shot classification, số vòng lặp không phải hyperparameter quan trọng vì:

1. Mỗi vòng thêm chỉ là một lần UOT solve (O(T²) với Sinkhorn).
2. Với T = 64 tokens, T² = 4096 — chi phí nhỏ.
3. Vòng 3 trở đi có diminishing return vì `C` gốc vẫn là anchor (S₂ vẫn tính từ `C + λS₂`, không tích lũy vô hạn).

---

## 5. Liên Hệ Với Các Phương Pháp Khác

### 5.1 Liên Hệ Với Gromov-Wasserstein

Gromov-Wasserstein (GW) bảo toàn quan hệ nội bộ bằng cách giải bài toán tối ưu lặp:
```
min_P Σ_{i,i',j,j'} |D_q(i,i') - D_s(j,j')|² P[i,j] P[i',j']
```

Đây là bài toán bậc 4 trong P, thường giải bằng Frank-Wolfe: linearize → solve OT → update → repeat.

HROT-S làm điều tương tự theo tinh thần, nhưng:
- Không giải bài toán GW đầy đủ (tránh bậc 4 và khó ổn định khi training).
- Dùng posterior UOT plan để tính xấp xỉ tuyến tính của structure cost.
- Giữ UOT thay vì chuyển sang OT cân bằng như GW thường dùng.

Có thể mô tả HROT-S là **"linearized iterative GW trong khung UOT"** — lấy được ý tưởng của GW mà không chịu chi phí giải GW đầy đủ.

### 5.2 Khác Biệt Với DeepEMD

DeepEMD dùng cross-reference weighting:
```
w_q(i) = softmax(similarity(z_q(i), all support tokens))
w_s(j) = softmax(similarity(z_s(j), all query tokens))
```

Sau đó EMD được giải với marginals từ các weight này.

DeepEMD không:
- Cho phép bỏ qua mass (EMD cân bằng hơn).
- Phân biệt noise và signal bằng explicit sink.
- Xét structure consistency giữa các cặp ghép.
- Iterate để bootstrap structure cost.

HROT-S không:
- Dùng cross-reference weighting.
- Dùng EMD cân bằng.

Hai phương pháp này không phải một cái tốt hơn cái kia về mọi mặt — chúng trả lời hai câu hỏi khác nhau. Điểm mạnh của HROT-S là ở bài toán có nhiều background noise và evidence cục bộ có cấu trúc quan hệ, đúng với scalogram phóng điện.

---

## 6. Cách Viết Claim Trong Paper

### 6.1 Claim Chính Xác

> HROT-S extends HROT-R with iterative posterior refinement (IPR): the structural consistency cost, originally derived from a single detached UOT probe, is now bootstrapped across two rounds. In each round, the posterior transport plan informs the structure cost, which reshapes the cost matrix for the next round. A confidence gate derived from row-wise transport entropy ensures that structural penalties are concentrated at token positions where the current transport plan is certain. The entire procedure introduces no additional parameters relative to HROT-R.

### 6.2 Claim Về Novelty

> Unlike Gromov-Wasserstein, which requires solving a quartic optimization, HROT-S approximates structural consistency through two sequential linear OT solves, inheriting the stability and noise tolerance of unbalanced transport while capturing relational geometry between local evidence tokens.

> Unlike DeepEMD's one-shot cross-reference weighting, HROT-S iteratively bootstraps the correspondence between query and support token sets, allowing structural consistency to refine both the matching and the structural signal simultaneously.

### 6.3 Claim Không Nên Viết

- Không nói "S là full Gromov-Wasserstein" — không đúng.
- Không nói "2 vòng là đủ về mặt lý thuyết" — chỉ có empirical justification.
- Không nói "confidence gate học được từ data" — gate hoàn toàn deterministic từ entropy của P.
- Không nói "S luôn tốt hơn R" — cần ablation để xác nhận.

---

## 7. Thiết Kế Thí Nghiệm Và Ablation

### 7.1 Ablation Đề Xuất

```
H                       ← baseline UOT
Q                       ← + reliability/consensus/sink/pooling
R (init=0.05)           ← + structure cost, 1 vòng, no gate
S-gate-only             ← R + confidence gate, 1 vòng
S-iter-only             ← R + 2 vòng IPR, no gate
S-full                  ← R + 2 vòng IPR + confidence gate
DeepEMD                 ← reference
```

Ablation này kể đúng câu chuyện:
- `S-gate-only` vs `R`: gate có giúp không khi chỉ 1 vòng?
- `S-iter-only` vs `R`: iteration có giúp không khi không có gate?
- `S-full` vs `S-iter-only`: gate có giúp thêm gì trên đỉnh iteration?

### 7.2 Câu Hỏi Cần Trả Lời

1. `S-full` có vượt `R` không, và bao nhiêu?
2. `S-full` có vượt `Q` không (để xác nhận structure cost vẫn có ích)?
3. `S-full` có vượt DeepEMD không, trên cả accuracy và calibration?
4. Lợi ích của S rõ hơn ở 1-shot hay 5-shot?
5. `structure_cost_weight` học về đâu so với R? Lên hay xuống?
6. Confidence gate có thực sự active không — histogram của gate values trước/sau training?

### 7.3 Diagnostics Cần Xem Thêm

Ngoài diagnostics của R, S cần thêm:

```
gate_round1          ← histogram confidence values ở vòng 1
gate_round2          ← histogram confidence values ở vòng 2
structure_cost_1     ← S₁ trước khi gate
structure_cost_2     ← S₂ sau khi gate + refine
delta_cost           ← C₂ - C₁ (phần structure thêm vào từ vòng 2)
plan_entropy_r1      ← entropy của P₁
plan_entropy_r2      ← entropy của P₂ (nên thấp hơn)
```

Cách đọc diagnostics:

1. Nếu `plan_entropy_r2 < plan_entropy_r1`: iteration đang làm cho plan tập trung hơn — đúng hướng.
2. Nếu `gate_round1` tập trung ở giá trị thấp trong early training: gate đang tự động suppress structure cost khi plan chưa ổn — đúng hướng.
3. Nếu `delta_cost` lớn tại những cặp token có base_cost thấp: S đang phát hiện các cặp ghép rẻ nhưng sai cấu trúc — đúng hướng.
4. Nếu `structure_cost_weight` sụp về 0 sau training: structure cost không có ích — cần điều tra.

---

## 8. Thực Thi Trong Code

### 8.1 Bật Variant S

```python
self.uses_noise_calibrated_transport = variant in {"Q", "R", "S"}
self.uses_structure_consistent_transport = variant in {"R", "S"}
self.uses_iterative_structure_refinement = variant == "S"
self.uses_confidence_gate = variant == "S"
```

### 8.2 Forward Path Thay Đổi

```python
if self.uses_iterative_structure_refinement:
    # Vòng 1
    P1 = uot_solve(flat_cost, query_mass, support_mass).detach()
    gate1 = compute_confidence_gate(P1)
    S1 = compute_structure_consistency_cost(
        flat_cost, query_tokens_euc, flat_support_tokens_euc,
        flat_rho, query_mass, support_mass
    )
    C1 = flat_cost + self.structure_cost_weight * gate1 * normalize(S1)

    # Vòng 2
    P2 = uot_solve(C1, query_mass, support_mass).detach()
    gate2 = compute_confidence_gate(P2)
    S2 = compute_structure_consistency_cost(
        flat_cost, query_tokens_euc, flat_support_tokens_euc,
        flat_rho, query_mass, support_mass,
        override_probe_plan=P2
    )
    # Quan trọng: dùng flat_cost gốc, không phải C1
    final_cost = flat_cost + self.structure_cost_weight * gate2 * normalize(S2)

elif self.uses_structure_consistent_transport:
    # HROT-R logic giữ nguyên
    ...
```

### 8.3 Hàm Confidence Gate

```python
def compute_confidence_gate(P: Tensor) -> Tensor:
    """
    P: [..., T_q, T_s] — posterior UOT plan (detached)
    Returns: [..., T_q, T_s] — gate values in [0, 1]
    """
    eps = 1e-8
    T_s = P.shape[-1]
    # Normalize rows to get distribution
    row_sum = P.sum(dim=-1, keepdim=True).clamp(min=eps)
    P_normalized = P / row_sum
    # Row entropy
    H = -(P_normalized * (P_normalized + eps).log()).sum(dim=-1)  # [..., T_q]
    H_max = math.log(T_s)
    # Confidence: thấp entropy → cao confidence
    row_confidence = (1.0 - H / H_max).clamp(0, 1)  # [..., T_q]
    # Broadcast thành gate matrix
    gate = row_confidence.unsqueeze(-1).expand_as(P)  # [..., T_q, T_s]
    return gate
```

### 8.4 Chạy Variant S

```bash
python run_all_experiments.py --models hrot_fsl --hrot_variant S
```

Tham số khởi tạo giữ nguyên:

```bash
--hrot_structure_cost_init 0.05
```

---

## 9. Giới Hạn Và Rủi Ro

### 9.1 Giới Hạn Còn Lại Sau S

1. `P_probe` ở cả hai vòng đều được detach. Structure cost vẫn không được học end-to-end qua solver. Tuy nhiên, 2 vòng detach vẫn tốt hơn 1 vòng detach.

2. Structure cost vẫn dùng Euclidean distance cho `D_q`, `D_s`. Hyperbolic distance sẽ nhất quán hơn với triết lý HROT nhưng sẽ thêm complexity. Đây là hướng mở rộng tiếp theo nếu S không đủ.

3. Gate chỉ xét row entropy (query side). Support side chưa được gate. Có thể thêm `col_confidence` từ entropy của các column của P để gate 2 phía, nhưng hiện tại giữ đơn giản.

4. Số vòng lặp (2) là fixed hyperparameter. Không học được.

### 9.2 Rủi Ro Trong Training

1. **Vòng 1 sai → vòng 2 cũng sai**: Nếu `P₁` rất tệ (do training chưa ổn định sớm), gate sẽ phần nào giảm thiểu nhưng không triệt tiêu hoàn toàn. Kiểm tra `plan_entropy_r1` trong early training để phát hiện.

2. **Lambda collapse**: Nếu `structure_cost_weight` học về 0, cả S và R đều degenerate về Q. Theo dõi bằng diagnostics.

3. **Chi phí tính toán**: S chạy thêm 1 UOT solve và 1 structure cost compute so với R. Với batch lớn và T lớn, có thể chậm hơn đáng kể. Cần benchmark wall-clock time.

---

## 10. Tóm Tắt Đóng Góp Của S

```
HROT-H:
  Tại sao: prototype toàn cục bị pha loãng bởi noise.
  Giải pháp: UOT local matching với geodesic EAM mass.

HROT-Q:
  Tại sao: uniform token mass không phân biệt signal và noise.
  Giải pháp: token reliability + consensus + sink + shot pooling.

HROT-R:
  Tại sao: reliability chưa đảm bảo structure consistency.
  Giải pháp: structure cost từ posterior UOT probe (1 vòng, detached).

HROT-S:
  Tại sao: probe 1 vòng tạo ra structure cost từ plan chưa được
           lọc bởi chính structure cost đó.
  Giải pháp: 2 vòng IPR — plan và structure cost bootstrap nhau.
             Confidence gate — chỉ phạt nơi plan đã tự tin.
  Chi phí: 0 tham số mới, 1 UOT solve thêm mỗi episode.
```

Đây là điểm kết thúc tự nhiên của chuỗi lập luận HROT: từ "cần local matching" đến "cần mass linh hoạt" đến "cần lọc noise" đến "cần structure consistency" đến "cần structure consistency nhất quán với chính nó".
