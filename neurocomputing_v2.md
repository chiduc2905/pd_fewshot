# SPIF-OTCCLS: Architecture Specification
## Target: Neurocomputing Q1 — Few-Shot Scalogram Classification

**Bài toán**: 4-way K-shot classification trên scalogram (CWT time-frequency representation).  
Trong cấu hình huấn luyện đề xuất cho model mới, mỗi run cố định `K ∈ {1, 5}`,
`Q_train = 5`, `Q_val = 10`, `Q_test = 10`, và ảnh đầu vào là RGB scalogram kích thước `84 × 84`.

**Tên đầy đủ**: Stable-Part Invariant Features with Optimal Transport Global Scoring  
and Confusability-Conditioned Local Scoring

---

## Mục lục

1. [Motivation và Research Gap](#1-motivation)
2. [Problem Setting](#2-problem-setting)
3. [Architecture Overview](#3-overview)
4. [Component 1 — Backbone](#4-backbone)
5. [Component 2 — Stable/Variant Factorization](#5-factorization)
6. [Component 3 — Global: Energy-Sliced Wasserstein (ESWG)](#6-global-eswg)
7. [Component 4 — Local: Confusability-Conditioned (CCLS)](#7-local-ccls)
8. [Component 5 — Fusion](#8-fusion)
9. [Loss Functions](#9-loss)
10. [Training Protocol](#10-training)
11. [Complementarity Analysis](#11-complementarity)
12. [Complete Forward Pass Pseudocode](#12-pseudocode)
13. [Hyperparameter Reference Table](#13-hyperparams)

---

## 1. Motivation

### Tại sao scalogram cần OT

Scalogram là $|W_\psi x(t, a)|^2$ — bình phương modulus của Continuous Wavelet Transform.  
Đây là phân phối năng lượng trên mặt phẳng (time × scale/frequency).  
Hai class tín hiệu khác nhau có **energy distribution khác nhau** trong không gian time-frequency.

Prototype averaging (ProtoNet) sụp đổ thông tin phân phối vào một điểm trung bình.  
Trung bình của nhiều Dirac measures khác nhau cho kết quả xấp xỉ Gaussian,  
mất đi cấu trúc multimodal của energy distribution.

**Optimal Transport** giữ nguyên cấu trúc phân phối: khoảng cách Wasserstein giữa  
query energy distribution và class energy distribution là metric tự nhiên và phù hợp nhất  
cho scalogram signals.

### Research gap của local branch

CTM (CVPR 2019) và mọi derivative tạo ra **một mask dùng chung cho N class** trong episode:  
`a_t = Fisher(V_inter, V_intra)` — không phân biệt tại position t,  
feature nào discriminative *cho class c cụ thể*.

CCLS tạo mask **riêng cho từng class** `a_{c,t}` dựa trên hardest negative competitor:  
vị trí t nguy hiểm nhất cho class c là vị trí c và c* (nearest competitor) gần nhau nhất.

### Sự bổ sung thực sự

| | ESWG (Global) | CCLS (Local) |
|---|---|---|
| Câu hỏi | Energy distribution của query giống class c không? | Query khớp class c tại các vị trí c dễ bị confuse không? |
| Thông tin | Tổng thể phân phối, K×M support tokens | Vị trí nguy hiểm cụ thể, per-class |
| Thay thế nhau được không? | Không — OT captures geometry | Không — pairwise competitive attention |
| Cần OT hay cosine? | OT: phân phối | Cosine: token matching |

---

## 2. Problem Setting

```
Episode / run trong cấu hình đề xuất cho model mới:
  - N = 4 classes (4-way)
  - K shots per class, K ∈ {1, 5} và được FIXED theo từng run
  - Q_train = 5 query images per class
  - Q_val   = 10 query images per class
  - Q_test  = 10 query images per class
  
  Support set: S = {(x_k^c, c) | c=1..N, k=1..K}
  Query set:   R = {(x_q, ?) | q=1..N·Q}

Input image:
  - RGB scalogram: (3, H_in, W_in)
  - H_in = W_in = 84

---

## 3. Architecture Overview

```
Input: (N·K + N·Q, 3, 84, 84)  ← support + query images
                │
         ┌──────▼──────┐
         │  Backbone    │  ResNet-12: (·, 3, 84, 84) → (·, 640, 5, 5)
         └──────┬───────┘
                │  F ∈ R^{B × 640 × 5 × 5}
         ┌──────▼──────────────────┐
         │  Stable/Variant Split   │  → T_s ∈ R^{B × 25 × C}  (C=256)
         └──────┬──────────────────┘    T_v ∈ R^{B × 25 × C}  (auxiliary)
                │
     ┌──────────┴─────────┐
     │                    │
  ┌──▼──────────────┐  ┌──▼─────────────────────┐
  │  Global: ESWG   │  │  Local: CCLS            │
  │                 │  │                         │
  │  Energy-Sliced  │  │  Confusability-         │
  │  Wasserstein    │  │  Conditioned Attention  │
  └──────┬──────────┘  └──────┬──────────────────┘
         │ S_global(q,c)       │ S_local(q,c)
         │    ∈ R^{N·Q × N}    │    ∈ R^{N·Q × N}
         └──────────┬──────────┘
                    │
            ┌───────▼───────┐
            │   Fusion      │  logit = S_global + β·S_local
            └───────┬───────┘
                    │
              logits ∈ R^{N·Q × N}
```

---

## 4. Component 1 — Backbone

### 4.1 Kiến trúc

**Backbone**: ResNet-12 (standard few-shot backbone, fair comparison với literature).

Trong implementation hiện tại của repo, scalogram được load dưới dạng **ảnh RGB 3-channel**,
vì vậy backbone dùng `conv1.in_channels = 3` như cấu hình chuẩn của ResNet-12.

```
ResNet-12 block structure:
  Block 1: Conv(3→64, 3×3) + BN + ReLU + Conv(64→64, 3×3) + BN + MaxPool(2×2)
  Block 2: Conv(64→128, 3×3) + BN + ReLU + Conv(128→128, 3×3) + BN + MaxPool(2×2)
  Block 3: Conv(128→256, 3×3) + BN + ReLU + Conv(256→256, 3×3) + BN + MaxPool(2×2)
  Block 4: Conv(256→640, 3×3) + BN + ReLU + Conv(640→640, 3×3) + BN

  Each block uses residual connection.
  Dropout(p=0.05) after each MaxPool (light dropout for low-data regime).
```

```
Input  : (B, 3, 84, 84)
Block 1: (B, 64, 40, 40)   → MaxPool → (B, 64, 20, 20)
Block 2: (B, 128, 20, 20)  → MaxPool → (B, 128, 10, 10)  
Block 3: (B, 256, 10, 10)  → MaxPool → (B, 256, 5, 5)
Block 4: (B, 640, 5, 5)    → NO pool

Output : F ∈ R^{B × 640 × 5 × 5}
```

## 5. Component 2 — Stable/Variant Factorization

### 5.1 Mục đích

Tách feature thành:
- **Stable branch** (`T_s`): features ổn định, ít bị ảnh hưởng bởi noise/phase  
- **Variant branch** (`T_v`): features biến động (chỉ dùng cho auxiliary loss khi training)

Điều này quan trọng cho scalogram vì tín hiệu thực tế có phase variation  
và noise không liên quan đến class label.

### 5.2 Các phép toán

```
Ký hiệu:
  F   : (B, C_b, h, w)   backbone output, C_b=640, h=w=5
  C   : 256               target feature dimension (giảm từ 640 để tránh overfit)
  M   : 25                số spatial tokens (= h × w = 5 × 5)
```

**Bước 1 — Projection heads** (linear, không có bias để tránh shift):

```python
W_s : Linear(640, 256, bias=False)   # stable projection
W_v : Linear(640, 256, bias=False)   # variant projection

F_flat = F.view(B, 640, 25).permute(0, 2, 1)   # (B, 25, 640)

F_s = W_s(F_flat)                               # (B, 25, 256)  stable features
F_v = W_v(F_flat)                               # (B, 25, 256)  variant features
```

**Bước 2 — Stability Gate** (element-wise gating trong feature space):

```python
# Gate MLP: 640 → 128 → 256, output in (0,1)
gate_mlp = nn.Sequential(
    nn.Linear(640, 128, bias=True),
    nn.ReLU(),
    nn.Linear(128, 256, bias=True),
    nn.Sigmoid()
)

G = gate_mlp(F_flat)        # (B, 25, 256), gates ∈ (0,1)
```

**Bước 3 — Gated stable tokens**:

```python
T_s = G * F_s               # (B, 25, 256)  element-wise product
                             # shape: (B, M, C) where M=25, C=256
```

**Bước 4 — Gated stable global embedding** (gated average pooling):

```python
# Numerator: weighted sum over spatial tokens
s_num = (G * T_s).sum(dim=1)      # (B, 256)

# Denominator: sum of gates
s_den = G.sum(dim=1) + 1e-8       # (B, 256), add epsilon for stability

s = s_num / s_den                 # (B, 256)  global stable embedding
```

**Output của Component 2**:
```
T_s : (B, M, C)  = (B, 25, 256)   stable spatial tokens
T_v : (B, M, C)  = (B, 25, 256)   variant tokens (training only)
s   : (B, C)     = (B, 256)       global stable embedding
```

---

## 6. Component 3 — Global: Energy-Sliced Wasserstein (ESWG)

### 6.1 Core Insight

Mỗi class c trong episode được đại diện bởi **một phân phối thực nghiệm**  
trên không gian feature, không phải một điểm prototype:

```
Phân phối của class c:
  μ_c = Σ_{k=1}^K Σ_{t=1}^M  e_{k,t}^c · δ_{T_{s,k,t}^c}

  trong đó e_{k,t}^c là trọng số năng lượng (xem bên dưới)
```

Score của query q với class c:

```
S_global(q, c) = -SWD(ν_q, μ_c) / τ_g
```

### 6.2 Energy-based prototype measure

**Tại sao energy weighting?**  
Scalogram token tại vị trí t có năng lượng `||T_{s,k,t}^c||²`.  
Vùng background (noise) có năng lượng thấp và không phân biệt class.  
Energy weighting tập trung "mass" của phân phối vào các vùng tín hiệu thực.

**Bước 1 — Tính energy weights cho support**:

```python
# T_s_sup: (N, K, M, C) support stable tokens, reorganized by class
# Tính norm² của mỗi token
norms_sq = (T_s_sup ** 2).sum(dim=-1)        # (N, K, M)  scalar energy per token

# Flatten K×M dimension
norms_flat = norms_sq.view(N, K * M)         # (N, K*M)

# Normalize to probability weights (energy-based measure)
e_c = norms_flat / (norms_flat.sum(dim=1, keepdim=True) + 1e-8)   # (N, K*M)
# e_c[c, k*M+t] = energy weight of token (k,t) for class c
```

**Bước 2 — Prototype feature set**:

```python
# Flatten support tokens (K×M tokens per class)
P_c = T_s_sup.view(N, K * M, C)             # (N, K*M, C)
# P_c[c] là tập K*M vectors, e_c[c] là trọng số của chúng
```

**Query measure** (uniform, không có prior về vị trí quan trọng của query):

```python
# T_s_qry: (N*Q, M, C) query stable tokens
# Uniform weights: 1/M cho mỗi token
e_q = torch.ones(N*Q, M, device=device) / M   # (N*Q, M)
```

### 6.3 Sliced Wasserstein Distance

**Công thức**:

```
SWD(ν, μ; L) = (1/L) Σ_{l=1}^L W_1(π_l ∘ ν, π_l ∘ μ)

  Trong đó:
  - π_l ∈ R^C là unit vector, lấy mẫu từ uniform(S^{C-1})
  - π_l ∘ μ là pushforward: 1D measure với atoms {(e_i, π_l · x_i)}
  - W_1(A, B) trên R là khoảng cách Wasserstein-1 (Earth Mover's Distance 1D)
```

**1D Wasserstein khi có weights**:  
Cho hai 1D discrete measures A = {(a_i, w_i)} và B = {(b_j, v_j)}:

```
W_1(A, B) = ∫₀¹ |F_A^{-1}(u) - F_B^{-1}(u)| du

  ≈ Σ_i,j |Q_A(u_i) - Q_B(u_i)| · Δu_i   (quantile approximation)
```

Trong thực tế, dùng thuật toán sau cho discrete weighted measures:

```
Algorithm: W1_weighted(a_vals, a_weights, b_vals, b_weights)
  1. Sort a_vals: a_sorted, idx_a → a_weights_sorted = a_weights[idx_a]
  2. Sort b_vals: b_sorted, idx_b → b_weights_sorted = b_weights[idx_b]
  3. Cumulative weight: cdf_a = cumsum(a_weights_sorted)
                        cdf_b = cumsum(b_weights_sorted)
  4. Merge all CDF values, interpolate both CDFs, integrate |CDF_A - CDF_B|
```

### 6.4 Implementation — batch SWD

```python
def sliced_wasserstein_score(
    q_feats,       # (Q_batch, M, C)   query tokens
    q_weights,     # (Q_batch, M)      query weights (uniform)
    p_feats,       # (N, KM, C)        prototype tokens per class
    p_weights,     # (N, KM)           prototype energy weights
    projections,   # (L, C)            fixed random projections
    tau_g,         # scalar            temperature
):
    """
    Returns:
        S_global: (Q_batch, N) score matrix
    """
    L = projections.shape[0]    # num projections
    Q = q_feats.shape[0]
    N = p_feats.shape[0]

    # Project all tokens onto L directions
    # q_proj: (Q, M, L)
    q_proj = q_feats @ projections.T          # (Q, M, L)
    # p_proj: (N, KM, L)
    p_proj = p_feats @ projections.T          # (N, KM, L)

    swd_matrix = torch.zeros(Q, N, device=q_feats.device)

    for l in range(L):
        q_1d = q_proj[:, :, l]               # (Q, M)
        p_1d = p_proj[:, :, l]               # (N, KM)

        for q_idx in range(Q):
            for c_idx in range(N):
                w1 = w1_weighted_1d(
                    q_1d[q_idx],              # (M,)
                    q_weights[q_idx],         # (M,) uniform
                    p_1d[c_idx],              # (KM,)
                    p_weights[c_idx]          # (KM,) energy
                )
                swd_matrix[q_idx, c_idx] += w1

    swd_matrix = swd_matrix / L              # average over projections
    S_global = -swd_matrix / tau_g           # negate: smaller distance → higher score
    return S_global                          # (Q, N)
```

**Efficient vectorized W1 (không có double loop)**:

```python
def w1_weighted_1d_batch(
    a_vals,      # (Q, M)     query projections
    a_w,         # (Q, M)     query weights
    b_vals,      # (N, KM)    prototype projections
    b_w,         # (N, KM)    prototype weights
):
    """
    Batch computation of 1D W1 between Q queries and N prototypes.
    Returns: (Q, N) matrix of W1 distances.
    
    Implementation note:
    Dùng quantile function approach:
      W1(A,B) = ∫₀¹ |Q_A(u) - Q_B(u)| du
    
    Xấp xỉ bằng cách sample R=100 quantile levels u_1,...,u_R
    và tính mean |Q_A(u_r) - Q_B(u_r)|.
    """
    R = 100  # quantile levels
    u = torch.linspace(0.01, 0.99, R, device=a_vals.device)  # (R,)

    # Compute quantile functions for A (queries)
    # Sort a_vals, compute cumulative weights, interpolate at u
    a_sorted, a_idx = a_vals.sort(dim=1)           # (Q, M)
    a_w_sorted = a_w.gather(1, a_idx)              # (Q, M)
    a_cdf = a_w_sorted.cumsum(dim=1) - a_w_sorted / 2   # (Q, M) midpoint CDF
    
    # Interpolate Q_A at quantile levels u: shape (Q, R)
    # For each q, find position in sorted values where cdf crosses u
    Q_A = batch_quantile_interp(a_sorted, a_cdf, u)  # (Q, R)

    # Compute quantile functions for B (prototypes)
    b_sorted, b_idx = b_vals.sort(dim=1)           # (N, KM)
    b_w_sorted = b_w.gather(1, b_idx)              # (N, KM)
    b_cdf = b_w_sorted.cumsum(dim=1) - b_w_sorted / 2   # (N, KM)
    Q_B = batch_quantile_interp(b_sorted, b_cdf, u)  # (N, R)

    # W1 ≈ mean over R quantile levels
    # Q_A: (Q, R), Q_B: (N, R)
    diff = Q_A.unsqueeze(1) - Q_B.unsqueeze(0)    # (Q, N, R)
    w1 = diff.abs().mean(dim=-1)                   # (Q, N)
    return w1
```

**Recommended implementation**: Sử dụng **POT library** (`pip install POT`):

```python
import ot

def swd_batch(q_feats, q_w, p_feats, p_w, projections):
    """
    q_feats: (Q, M, C)
    p_feats: (N, KM, C)
    projections: (L, C)
    Returns: (Q, N)
    """
    L, C = projections.shape
    Q, M, _ = q_feats.shape
    N, KM, _ = p_feats.shape

    # Project: (Q, M, L) and (N, KM, L)
    qp = q_feats @ projections.T    # (Q, M, L)
    pp = p_feats @ projections.T    # (N, KM, L)

    # Transpose to (L, Q, M) and (L, N, KM) for vectorization
    qp = qp.permute(2, 0, 1)        # (L, Q, M)
    pp = pp.permute(2, 0, 1)        # (L, N, KM)

    swd = torch.zeros(Q, N, device=q_feats.device)

    for l in range(L):
        # For each pair (q, c), compute W1 in 1D
        # Sort q projections
        qp_l = qp[l]   # (Q, M)
        pp_l = pp[l]   # (N, KM)

        # Vectorized W1 using sorted quantile approach
        for q in range(Q):
            for c in range(N):
                # ot.wasserstein_1d is O(n log n), exact for 1D
                w1_val = ot.wasserstein_1d(
                    qp_l[q].cpu().numpy(),
                    pp_l[c].cpu().numpy(),
                    q_w[q].cpu().numpy(),
                    p_w[c].cpu().numpy()
                )
                swd[q, c] += w1_val

    return swd / L
```

**Lưu ý quan trọng về gradient**: SWD qua POT có gradient nhờ AutoDiff.  
Nhưng để training ổn định hơn, dùng thêm **straight-through estimator** cho sort operations:

```python
# Thay vì dùng POT trực tiếp trong training,
# dùng soft-sorted version (Cuturi & Doucet 2017):
# sort approximation: soft-sort với temperature σ_sort = 0.1
```

### 6.5 Random projections setup

```python
# Cố định random seed để đảm bảo tái lập
# Khởi tạo một lần, KHÔNG train
torch.manual_seed(42)
projections = torch.randn(64, C)          # (L=64, C=256)
projections = F.normalize(projections, dim=1)  # unit sphere
projections = projections.to(device)
projections.requires_grad = False        # KHÔNG update qua gradient
```

**L=64**: đủ cho C=256 dimensions (theo lý thuyết Johnson-Lindenstrauss,  
L ≥ O(log(1/δ)/ε²) để đảm bảo ε-approximation với xác suất ≥ 1-δ.  
Với ε=0.1, δ=0.01: L ≥ ~450. Tuy nhiên trong thực tế, L=64 cho kết quả tốt  
vì features có structure thấp chiều hơn C=256 nominal dimensions).

**Learnable parameters trong ESWG**: chỉ `τ_g` (1 scalar).

---

## 7. Component 4 — Local: CCLS

### 7.1 Core Insight

CTM và EDLA tạo một mask `a_t` chung cho toàn episode.  
CCLS tạo mask `a_{c,t}` riêng cho từng class c — tập trung vào vị trí  
mà class c dễ bị confuse nhất với nearest competitor của nó.

**Không có overlap với CTM** vì:
- CTM: class-agnostic mask, Fisher criterion (population-level)
- CCLS: class-specific mask, hardest-negative criterion (pairwise)

### 7.2 Spatial prototype per class

```
Ký hiệu:
  T_s_sup: (N, K, M, C)  support stable tokens
  μ_{c,t}: mean token của class c tại vị trí t
```

```python
# T_s_sup: (N, K, M, C) → mean over K shots
mu_ct = T_s_sup.mean(dim=1)      # (N, M, C)  spatial prototypes per class
# mu_ct[c, t, :] = mean support token của class c tại vị trí t
```

### 7.3 Confusability score

**Bước 1 — Nearest competitor tại mỗi (c, t)**:

```python
# Pairwise distances giữa các class tại mỗi position t
# mu_ct: (N, M, C)
# Expand để tính pairwise: (N, 1, M, C) vs (1, N, M, C)
mu_i = mu_ct.unsqueeze(1)          # (N, 1, M, C)
mu_j = mu_ct.unsqueeze(0)          # (1, N, M, C)

pair_dist = (mu_i - mu_j).norm(dim=-1)   # (N, N, M)  ||μ_{c,t} - μ_{c',t}||
# pair_dist[c, c', t] = distance between class c and c' at position t

# Mask out diagonal (same class)
mask = torch.eye(N, device=device).bool().unsqueeze(-1).expand(N, N, M)
pair_dist_masked = pair_dist.masked_fill(mask, float('inf'))   # (N, N, M)

# Hardest negative: min over c' ≠ c
delta_ct, _ = pair_dist_masked.min(dim=1)   # (N, M)
# delta_ct[c, t] = distance to nearest competitor at position t
```

**Bước 2 — Class-specific confusability attention**:

```python
# Vị trí t có delta nhỏ → class c dễ bị confuse tại t → tập trung matching ở đó
# a_{c,t} ∝ 1/delta_{c,t} (inverse of distance to nearest competitor)

# Learnable temperature τ_L, initialized = 1.0
tau_L = F.softplus(tau_L_raw) + 0.1    # ensure τ_L > 0.1

# Inverse confusability (unnormalized)
inv_delta = 1.0 / (delta_ct + 1e-6)           # (N, M)

# Softmax over M positions (for each class c independently)
a_ct = F.softmax(inv_delta / tau_L, dim=1)    # (N, M)
# a_ct[c, t] ∈ (0,1), Σ_t a_ct[c,t] = 1 for each c
```

### 7.4 Local score

```python
# T_s_qry: (Q_batch, M, C) query stable tokens
# mu_ct: (N, M, C) support spatial prototypes
# a_ct: (N, M) class-specific confusability attention

# Cosine similarity at each position
# q_tokens: (Q, M, C), mu: (N, M, C)
# Expand: (Q, 1, M, C) vs (1, N, M, C)
q_expand   = T_s_qry.unsqueeze(1)             # (Q, 1, M, C)
mu_expand  = mu_ct.unsqueeze(0)               # (1, N, M, C)

# Cosine similarity per position: (Q, N, M)
cos_sim = F.cosine_similarity(q_expand, mu_expand, dim=-1)  # (Q, N, M)

# Weight by confusability attention a_{c,t}: (1, N, M)
a_expand = a_ct.unsqueeze(0)                  # (1, N, M)

# Weighted sum over M positions
S_local = (a_expand * cos_sim).sum(dim=-1)    # (Q, N)
```

**Output**: `S_local ∈ R^{Q × N}`, class-specific local scores.

---

## 8. Component 5 — Fusion

### 8.1 Adaptive fusion weight

Beta không cố định. Nó phụ thuộc vào **mean prototype reliability** của episode:

```python
# Reliability estimate: mean distance from each support sample to its class prototype
# Low mean distance → prototypes are reliable → global is trustworthy → local weight ↓
# High mean distance → prototypes uncertain → local adds more value → local weight ↑

# Compute reliability:
# T_s_sup: (N, K, M, C), global embeddings: s_sup (N, K, C)
# Use global embeddings for reliability
s_sup = ...      # (N, K, C) global embeddings, từ gated averaging
mu_global = s_sup.mean(dim=1)    # (N, C) mean prototype per class

scatter = ((s_sup - mu_global.unsqueeze(1)) ** 2).sum(dim=-1).mean()  # scalar D̄
rho_bar = torch.exp(-alpha * scatter / C)    # episode-level reliability, ∈ (0,1]

# Beta: higher when prototypes are uncertain
beta_0 = F.softplus(beta_raw)               # learnable base, init = log(exp(0.5)-1)
beta_eff = beta_0 * (1 - rho_bar)           # scalar, ∈ [0, beta_0)
```

### 8.2 Final logit

```python
logits = S_global + beta_eff * S_local      # (Q, N)
```

**Learnable scalars** trong fusion: `β_raw`, `α_raw` (2 scalars).

---

## 9. Loss Functions

### 9.1 Main loss — Episodic Cross-Entropy

```python
loss_CE = F.cross_entropy(
    logits,        # (Q, N)
    labels,        # (Q,) ground truth class indices
    label_smoothing=0.1,
    reduction='mean'
)
```

**Training note**: với model mới, bật `label_smoothing = 0.1`
để giảm overconfidence, đặc biệt ở `1-shot`.

### 9.2 Auxiliary loss 1 — Support compactness

Khuyến khích support embeddings của cùng class tập trung quanh prototype:

```python
# s_sup: (N, K, C) global stable embeddings
# mu_global: (N, C) mean prototype
diff = s_sup - mu_global.unsqueeze(1)        # (N, K, C)
loss_compact = (diff ** 2).sum(dim=-1).mean()   # scalar
```

**Rationale**: Compact support set → reliable prototype → reliable ESWG scores.

### 9.3 Auxiliary loss 2 — CCLS entropy regularization

Tránh collapse của attention về một position duy nhất:

```python
# a_ct: (N, M) class-specific attention
# Entropy: H(a_c) = -Σ_t a_{c,t} log a_{c,t}
entropy = -(a_ct * (a_ct + 1e-8).log()).sum(dim=1)   # (N,)
# Maximize entropy → prevent collapse to single position
loss_entropy = -entropy.mean()    # negative entropy (minimize = maximize entropy)
```

**Chú ý**: loss_entropy có hệ số nhỏ (λ_3=0.01) — chỉ là soft regularization.

### 9.4 Auxiliary loss 3 — Stable/Variant decorrelation

Khuyến khích T_s và T_v ít tương quan:

```python
# s_stable: (B, C) global stable embeddings
# s_variant: (B, C) global variant embeddings (computed similarly)
s_v_global = (G * T_v).sum(dim=1) / (G.sum(dim=1) + 1e-8)   # (B, C)

# Normalize
s_s_norm = F.normalize(s_stable, dim=-1)    # (B, C)
s_v_norm = F.normalize(s_v_global, dim=-1)  # (B, C)

# Correlation (should be near zero)
corr = (s_s_norm * s_v_norm).sum(dim=-1)    # (B,) ∈ [-1,1]
loss_decorr = (corr ** 2).mean()            # penalize correlation
```

### 9.5 Total loss

```
L = L_CE + λ₁·L_compact + λ₂·L_decorr + λ₃·L_entropy

λ₁ = 0.1     (compact support: primary auxiliary)
λ₂ = 0.05    (decorrelation: secondary)
λ₃ = 0.01    (entropy: soft regularizer)
```

**Tất cả λ đều cố định (không learn)** để giữ loss landscape đơn giản.

---

## 10. Training Protocol

### 10.1 Episodic training

```
Proposed training protocol for the new model:
  - N = 4 classes
  - K ∈ {1, 5}, FIXED theo từng run (không mixed-shot trong cùng một run)
  - Q_train = 5, Q_val = 10, Q_test = 10
  - Training sample modes: {60, 160, 240, all}
    → tương ứng {15, 40, 60, all} samples/class khi N = 4

Episodes per epoch : 200
Validation         : eval mỗi epoch trên 300 episodes
Test               : 400 episodes
Training epochs    : 100
Batch size         : 1 episode / step
```

### 10.2 Optimizer

```
Optimizer: AdamW

Main fair-comparison setting:
  - Single parameter group cho toàn bộ model
  - Global learning rate: lr = 5e-4
  - Mọi baseline và model đề xuất đều dùng cùng outer optimizer recipe này

Weight decay:
  - All parameters: wd = 5e-4

Scheduler:
  - CosineAnnealingLR(T_max=100, eta_min=1e-6)
  - Warmup: linear từ 0 đến full lr trong 5 epoch đầu

Optional tuned variant for the proposed model (report as ablation, not main table):
  - Backbone (ResNet-12):                 lr = 2e-4
  - Projection / token heads:            lr = 7e-4
    (stable_head, variant_head, shared_head, variant_align)
  - Gate MLP:                            lr = 7e-4
  - Scalar params (τ_g, τ_L, β, α):      lr = 1e-3
  - Weight decay for scalar params:      wd = 0.0

Regularization / runtime flags cho model mới:
  - label_smoothing = 0.1
  - grad_clip = 0.0  (disabled)
  - train_augment = false
  - time_shift_prob = 0.0
  - amp_scale_prob  = 0.0
  - time_mask_prob  = 0.0
  - freq_mask_prob  = 0.0
  - cudnn_deterministic = true
  - cudnn_benchmark    = false
```

### 10.4 Khởi tạo tham số

```python
# Scalar parameters (rất quan trọng để khởi tạo đúng)
tau_g_raw  = nn.Parameter(torch.tensor(2.303))   # softplus(2.303) ≈ 10.0 → tau_g = 10
tau_L_raw  = nn.Parameter(torch.tensor(0.541))   # softplus(0.541) ≈ 1.0  → tau_L = 1
beta_raw   = nn.Parameter(torch.tensor(0.405))   # softplus(0.405) ≈ 0.5  → beta_0 = 0.5
alpha_raw  = nn.Parameter(torch.tensor(0.541))   # softplus(0.541) ≈ 1.0  → alpha = 1

# Projections
nn.init.xavier_uniform_(W_s.weight)
nn.init.xavier_uniform_(W_v.weight)

# Gate MLP
for layer in gate_mlp:
    if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight)
        nn.init.zeros_(layer.bias)

# Random projections (frozen)
torch.manual_seed(42)
projections = F.normalize(torch.randn(64, 256), dim=1)
```

---

## 11. Complementarity Analysis

### 11.1 Thông tin mỗi nhánh mang

```
ESWG global:
  Input : K×M support tokens (K*25 điểm feature), M query tokens
  Computes: Sliced Wasserstein distance giữa HAI PHÂN PHỐI
  Câu hỏi: "energy distribution của query có khớp class c không?"
  K=1: SWD giữa 25 support tokens và 25 query tokens → vẫn ý nghĩa
  K=5: SWD giữa 125 support tokens và 25 query tokens → chính xác hơn

CCLS local:
  Input : spatial prototype μ_{c,t} (averaged over K), confusability δ_{c,t}
  Computes: weighted cosine matching tập trung vào vị trí nguy hiểm
  Câu hỏi: "query có khớp class c tại các vị trí c dễ bị nhầm không?"
  K=1: δ_{c,t} từ single sample → noisy nhưng vẫn computable
  K=5: δ_{c,t} stable hơn → attention mask đáng tin cậy hơn
```

### 11.2 Formal independence (không thể thay thế nhau)

**Trường hợp ESWG thắng, CCLS fail**:  
→ Classes khác nhau về global energy distribution nhưng locally confusable tại nhiều vị trí.  
→ CCLS phân tán attention ra nhiều vị trí có δ nhỏ → noisy matching.  
→ ESWG nắm bắt phân phối tổng thể → score chính xác.

**Trường hợp CCLS thắng, ESWG fail**:  
→ Classes có energy distribution gần nhau (tín hiệu cùng băng tần)  
   nhưng khác nhau tại một số vị trí đặc trưng.  
→ ESWG không đủ resolution → SWD nhỏ cho cả hai class.  
→ CCLS tập trung đúng vào vị trí discriminative → phân biệt được.

**Chứng minh formal**: ESWG là geometry-based (distributional distance),  
CCLS là discrimination-based (competitive local matching). Hai tiêu chí này  
trả lời câu hỏi độc lập nhau về mặt information-theoretic.

---

## 12. Complete Forward Pass Pseudocode

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class SPIF_OTCCLS(nn.Module):
    def __init__(
        self,
        backbone_out_dim = 640,
        feature_dim      = 256,     # C
        num_projections  = 64,      # L
        num_spatial      = 25,      # M = 5×5
    ):
        super().__init__()
        self.C = feature_dim
        self.M = num_spatial
        self.L = num_projections

        # ── Backbone ─────────────────────────────────────────────
        self.backbone = ResNet12(in_channels=3)   # repo benchmark uses RGB scalograms

        # ── Stable/Variant projections ────────────────────────────
        self.W_s = nn.Linear(backbone_out_dim, feature_dim, bias=False)
        self.W_v = nn.Linear(backbone_out_dim, feature_dim, bias=False)

        # ── Stability gate ────────────────────────────────────────
        self.gate_mlp = nn.Sequential(
            nn.Linear(backbone_out_dim, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim),
            nn.Sigmoid()
        )

        # ── Random projections (frozen) ───────────────────────────
        torch.manual_seed(42)
        proj = F.normalize(torch.randn(num_projections, feature_dim), dim=1)
        self.register_buffer('projections', proj)   # not a parameter

        # ── Learnable scalars ─────────────────────────────────────
        # Using softplus to ensure positivity
        self.tau_g_raw  = nn.Parameter(torch.tensor(2.303))  # → tau_g  ≈ 10
        self.tau_L_raw  = nn.Parameter(torch.tensor(0.541))  # → tau_L  ≈ 1
        self.beta_raw   = nn.Parameter(torch.tensor(0.405))  # → beta_0 ≈ 0.5
        self.alpha_raw  = nn.Parameter(torch.tensor(0.541))  # → alpha  ≈ 1

    # ──────────────────────────────────────────────────────────────
    def get_scalars(self):
        tau_g  = F.softplus(self.tau_g_raw)  + 0.1
        tau_L  = F.softplus(self.tau_L_raw)  + 0.1
        beta_0 = F.softplus(self.beta_raw)
        alpha  = F.softplus(self.alpha_raw)
        return tau_g, tau_L, beta_0, alpha

    # ──────────────────────────────────────────────────────────────
    def extract_features(self, x):
        """
        x    : (B, 3, 84, 84)
        Returns:
            T_s  : (B, M, C)   stable spatial tokens
            T_v  : (B, M, C)   variant tokens (auxiliary)
            s    : (B, C)      global stable embedding
        """
        # Backbone
        # Dataset tensors are already prepared like the other repo models,
        # so images go directly into the backbone without extra log-power preprocessing.
        F_feat = self.backbone(x)                      # (B, 640, 5, 5)

        # Flatten spatial
        B, C_b, h, w = F_feat.shape
        F_flat = F_feat.view(B, C_b, h*w).permute(0, 2, 1)  # (B, 25, 640)

        # Projections
        F_s = self.W_s(F_flat)    # (B, 25, 256)
        F_v = self.W_v(F_flat)    # (B, 25, 256)

        # Gate
        G   = self.gate_mlp(F_flat)  # (B, 25, 256), ∈ (0,1)

        # Gated stable tokens
        T_s = G * F_s              # (B, 25, 256)
        T_v = G * F_v              # (B, 25, 256)  [training auxiliary]

        # Global stable embedding (gated average)
        num = (G * T_s).sum(dim=1)             # (B, 256)
        den = G.sum(dim=1) + 1e-8              # (B, 256)
        s   = num / den                        # (B, 256)

        return T_s, T_v, s

    # ──────────────────────────────────────────────────────────────
    def forward(self, support, query, N, K, Q):
        """
        support : (N*K, 3, 84, 84)
        query   : (N*Q, 3, 84, 84)
        N       : num classes
        K       : num shots
        Q       : num queries per class
        
        Returns:
            logits      : (N*Q, N)
            aux_losses  : dict of auxiliary loss terms
        """
        tau_g, tau_L, beta_0, alpha = self.get_scalars()

        # ── Feature extraction ────────────────────────────────────
        all_imgs = torch.cat([support, query], dim=0)  # (N*K + N*Q, 3, 84, 84)
        T_s_all, T_v_all, s_all = self.extract_features(all_imgs)

        # Split support and query
        NK = N * K
        T_s_sup = T_s_all[:NK].view(N, K, self.M, self.C)   # (N, K, 25, 256)
        T_s_qry = T_s_all[NK:]                               # (N*Q, 25, 256)
        T_v_sup = T_v_all[:NK].view(N, K, self.M, self.C)

        s_sup   = s_all[:NK].view(N, K, self.C)             # (N, K, 256)
        s_qry   = s_all[NK:]                                 # (N*Q, 256)

        # ── ESWG Global Score ─────────────────────────────────────
        # Energy weights for support prototype measure
        norms_sq   = (T_s_sup ** 2).sum(dim=-1)             # (N, K, M)
        norms_flat = norms_sq.view(N, K * self.M)            # (N, K*M)
        e_c        = norms_flat / (norms_flat.sum(dim=1, keepdim=True) + 1e-8)  # (N, K*M)

        # Prototype feature set
        P_c = T_s_sup.view(N, K * self.M, self.C)           # (N, K*M, 256)

        # Query uniform weights
        NQ   = N * Q
        e_q  = torch.ones(NQ, self.M, device=T_s_qry.device) / self.M  # (N*Q, M)

        # Compute SWD
        swd = self._compute_swd(T_s_qry, e_q, P_c, e_c)    # (N*Q, N)
        S_global = -swd / tau_g                              # (N*Q, N)

        # ── CCLS Local Score ─────────────────────────────────────
        mu_ct = T_s_sup.mean(dim=1)    # (N, M, 256) spatial prototypes

        # Pairwise distances between class prototypes at each position
        mu_i = mu_ct.unsqueeze(1)      # (N, 1, M, 256)
        mu_j = mu_ct.unsqueeze(0)      # (1, N, M, 256)
        pair_dist = (mu_i - mu_j).norm(dim=-1)              # (N, N, M)

        # Mask diagonal, find nearest competitor
        eye_mask = torch.eye(N, device=pair_dist.device).bool()
        eye_mask = eye_mask.unsqueeze(-1).expand(N, N, self.M)
        pair_dist = pair_dist.masked_fill(eye_mask, float('inf'))
        delta_ct, _ = pair_dist.min(dim=1)                  # (N, M)

        # Class-specific attention
        inv_delta = 1.0 / (delta_ct + 1e-6)                 # (N, M)
        a_ct      = F.softmax(inv_delta / tau_L, dim=1)     # (N, M)

        # Cosine similarity: (N*Q, 1, M, C) vs (1, N, M, C)
        q_exp  = T_s_qry.unsqueeze(1)         # (N*Q, 1, M, C)
        mu_exp = mu_ct.unsqueeze(0)            # (1, N, M, C)
        cos_sim = F.cosine_similarity(q_exp, mu_exp, dim=-1)  # (N*Q, N, M)

        a_exp   = a_ct.unsqueeze(0)            # (1, N, M)
        S_local = (a_exp * cos_sim).sum(dim=-1)              # (N*Q, N)

        # ── Fusion ────────────────────────────────────────────────
        # Episode-level reliability
        mu_global = s_sup.mean(dim=1)          # (N, C) global prototype
        scatter   = ((s_sup - mu_global.unsqueeze(1)) ** 2).sum(dim=-1).mean()
        rho_bar   = torch.exp(-alpha * scatter / self.C)     # scalar ∈ (0,1]

        beta_eff  = beta_0 * (1 - rho_bar)    # scalar

        logits = S_global + beta_eff * S_local               # (N*Q, N)

        # ── Auxiliary loss terms ──────────────────────────────────
        aux = {}

        # Compactness loss
        diff_sup       = s_sup - mu_global.unsqueeze(1)      # (N, K, C)
        aux['compact'] = (diff_sup ** 2).sum(dim=-1).mean()

        # Decorrelation loss
        s_v_global = (
            (self._get_gate(T_s_sup.view(N*K,-1,640) if False else
             T_v_sup.view(N*K, self.M, self.C)) *
             T_v_sup.view(N*K, self.M, self.C)).sum(dim=1) /
            (self._get_gate_sum(T_v_sup.view(N*K, self.M, self.C)) + 1e-8)
        )
        # Simplified: directly compute s_v from T_v_all[:NK]
        T_v_flat = T_v_sup.view(N*K, self.M, self.C)
        G_flat   = ...  # (reuse gate from extract_features — store in forward)
        # (省略 for brevity — implement by storing G during extract_features)
        # aux['decorr'] = ...

        # Entropy regularization
        entropy         = -(a_ct * (a_ct + 1e-8).log()).sum(dim=1)  # (N,)
        aux['entropy']  = -entropy.mean()     # negative entropy

        return logits, aux

    # ──────────────────────────────────────────────────────────────
    def _compute_swd(self, q_feats, q_weights, p_feats, p_weights):
        """
        q_feats   : (Q, M, C)
        q_weights : (Q, M)
        p_feats   : (N, KM, C)
        p_weights : (N, KM)
        projections: (L, C)  [self.projections]
        Returns   : (Q, N) SWD matrix
        """
        Q  = q_feats.shape[0]
        N  = p_feats.shape[0]
        L  = self.projections.shape[0]

        # Project onto L random directions
        q_proj = q_feats @ self.projections.T   # (Q, M, L)
        p_proj = p_feats @ self.projections.T   # (N, KM, L)

        q_proj = q_proj.permute(2, 0, 1)        # (L, Q, M)
        p_proj = p_proj.permute(2, 0, 1)        # (L, N, KM)

        swd = torch.zeros(Q, N, device=q_feats.device)

        # Vectorized 1D Wasserstein using quantile approximation
        R    = 100
        u    = torch.linspace(0.01, 0.99, R, device=q_feats.device)  # (R,)

        for l in range(L):
            qp_l = q_proj[l]   # (Q, M)
            pp_l = p_proj[l]   # (N, KM)

            # Quantile functions
            Q_A = self._weighted_quantile(qp_l, q_weights, u)  # (Q, R)
            Q_B = self._weighted_quantile(pp_l, p_weights, u)  # (N, R)

            # W1 ≈ mean absolute quantile difference
            diff   = Q_A.unsqueeze(1) - Q_B.unsqueeze(0)       # (Q, N, R)
            swd   += diff.abs().mean(dim=-1)                    # (Q, N)

        return swd / L

    def _weighted_quantile(self, vals, weights, quantiles):
        """
        vals      : (B, M)   1D projection values
        weights   : (B, M)   ∑=1 probability weights
        quantiles : (R,)     target quantile levels ∈ (0,1)
        Returns   : (B, R)   quantile function at each level
        """
        B, M = vals.shape
        R    = quantiles.shape[0]

        # Sort values
        sorted_vals, sort_idx = vals.sort(dim=1)                # (B, M)
        sorted_w = weights.gather(1, sort_idx)                  # (B, M)

        # Cumulative CDF (at midpoints for better interpolation)
        cumw     = sorted_w.cumsum(dim=1)                       # (B, M)
        # Shift by half weight: CDF at atom position
        cdf_pts  = cumw - sorted_w / 2                          # (B, M)

        # Interpolate: for each quantile u_r, find val s.t. CDF(val)=u_r
        # Using linear interpolation
        # cdf_pts: (B, M), sorted_vals: (B, M), quantiles: (R,)
        # Expand for broadcasting
        cdf_exp  = cdf_pts.unsqueeze(2)                         # (B, M, 1)
        q_exp    = quantiles.view(1, 1, R)                      # (1, 1, R)
        val_exp  = sorted_vals.unsqueeze(2)                     # (B, M, 1)

        # For each quantile u_r, find the two CDF points bracketing it
        # (simplified: use linear interp over sorted CDF)
        # Output: (B, R)
        result = torch.zeros(B, R, device=vals.device)
        for r in range(R):
            u_r = quantiles[r]
            # For each batch item, find idx where cdf crosses u_r
            # Use torch.searchsorted
            idx = torch.searchsorted(cdf_pts, u_r.expand(B, 1)).clamp(0, M-1).squeeze(1)  # (B,)
            idx_lo = (idx - 1).clamp(0, M-1)
            idx_hi = idx

            cdf_lo = cdf_pts.gather(1, idx_lo.unsqueeze(1)).squeeze(1)   # (B,)
            cdf_hi = cdf_pts.gather(1, idx_hi.unsqueeze(1)).squeeze(1)   # (B,)
            val_lo = sorted_vals.gather(1, idx_lo.unsqueeze(1)).squeeze(1)
            val_hi = sorted_vals.gather(1, idx_hi.unsqueeze(1)).squeeze(1)

            # Linear interpolation
            t = ((u_r - cdf_lo) / (cdf_hi - cdf_lo + 1e-8)).clamp(0, 1)
            result[:, r] = val_lo + t * (val_hi - val_lo)

        return result   # (B, R)


# ── Training loop sketch ──────────────────────────────────────────
def train_epoch(model, dataloader, optimizer, lam=(0.1, 0.05, 0.01)):
    model.train()
    lam1, lam2, lam3 = lam

    for support, query, labels in dataloader:
        N, K = support.shape[:2]
        Q    = query.shape[1]

        # Reshape
        sup_flat = support.view(N*K, 3, 84, 84).to(device)
        qry_flat = query.view(N*Q, 3, 84, 84).to(device)
        labels   = labels.to(device)    # (N*Q,)

        optimizer.zero_grad()

        logits, aux = model(sup_flat, qry_flat, N=N, K=K, Q=Q)

        loss_ce      = F.cross_entropy(logits, labels, label_smoothing=0.1)
        loss_compact = aux['compact']
        loss_entropy = aux['entropy']

        loss = loss_ce + lam1 * loss_compact + lam3 * loss_entropy
        loss.backward()

        # Gradient clipping (critical for stability with SWD)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
```

---

## 13. Hyperparameter Reference Table

| Tham số | Ký hiệu | Giá trị | Learnable? | Ràng buộc |
|---|---|---|---|---|
| Feature dimension | C | 256 | No | Fixed |
| Spatial tokens | M | 25 | No | Fixed (5×5) |
| SWD projections | L | 64 | No | Fixed (random seed=42) |
| Quantile levels | R | 100 | No | Fixed |
| Global temperature | τ_g | init≈10.0 | Yes | softplus + 0.1 > 0 |
| Local temperature | τ_L | init≈1.0 | Yes | softplus + 0.1 > 0 |
| Fusion base | β_0 | init≈0.5 | Yes | softplus > 0 |
| Reliability scale | α | init≈1.0 | Yes | softplus > 0 |
| Compact loss weight | λ₁ | 0.1 | No | Fixed |
| Decorr loss weight | λ₂ | 0.05 | No | Fixed |
| Entropy loss weight | λ₃ | 0.01 | No | Fixed |
| Label smoothing | ε | 0.1 | No | New model training |
| Dropout (backbone) | p | 0.05 | No | Fixed |
| Image channels | — | 3 | No | RGB scalogram |
| Input size | — | 84×84 | No | Current protocol |
| Global LR (main fair table) | — | 5e-4 | — | AdamW, single parameter group |
| Weight decay (main fair table) | wd | 5e-4 | — | All parameters |
| Scheduler | — | CosineAnnealingLR(T_max=100, eta_min=1e-6) | — | Shared across all models |
| Gradient clip | — | 0.0 | — | Disabled |
| Episode K range | K | {1,5} | — | Fixed per run |
| Query/class (train/val/test) | Q | 5 / 10 / 10 | — | New model training |
| Train episodes/epoch | — | 200 | — | Current protocol |
| Val episodes/eval | — | 300 | — | Current protocol |
| Test episodes | — | 400 | — | New model training |
| Warmup epochs | — | 5 | — | Linear warmup |
| Training sample modes | — | {60,160,240,all} | — | Balanced by class |
| Variance floor | ε_v | 1e-6 | No | Energy weight denom |

Optional tuned ablation for the proposed model only:

| Tuned backbone LR | — | 2e-4 | — | AdamW param group |
| Tuned projection / token-head LR | — | 7e-4 | — | AdamW param group |
| Tuned gate MLP LR | — | 7e-4 | — | AdamW param group |
| Tuned scalar LR | — | 1e-3 | — | AdamW param group |
| Tuned scalar weight decay | — | 0.0 | — | Scalar params only |

---

## Contribution Summary cho Paper

```
Global (ESWG):
  Novel: SWD làm few-shot class scoring function — chưa có paper nào làm
  (DeepEMD dùng OT trong-episode, không dùng OT so sánh query vs class distribution)
  Energy weighting: scalogram-specific design choice, physically motivated

Local (CCLS):
  Novel: class-specific confusability mask — CTM chỉ có class-agnostic mask
  Verifiable: a_{c,t} ≠ a_{c',t} trong same episode → t-SNE visualization

Theory:
  Shot-scalability: cả SWD (more support tokens → richer distribution) và
  CCLS (more shots → stable δ_{c,t}) đều tốt hơn khi K tăng

Complementarity:
  Global = geometric comparison of distributions (OT)
  Local  = discriminative position matching (competitive attention)
  Không thể thay thế nhau → genuine dual-branch contribution
```

---

*Specification version 1.0 — SPIF-OTCCLS for Scalogram Few-Shot Classification*  
*Target: Neurocomputing Q1*
