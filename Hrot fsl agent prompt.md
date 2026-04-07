# HROT-FSL: Agent Coding Prompt
## Hyperbolic Relational Optimal Transport for Few-Shot Learning
### Full Implementation Prompt — From Scratch, A*-Ready

---

## ROLE & OBJECTIVE

You are an expert ML engineer implementing **HROT-FSL** (Hyperbolic Relational Optimal Transport for Few-Shot Learning), a novel few-shot image classification method for publication at CVPR/NeurIPS/ICLR. You will implement this from scratch in PyTorch, integrating into a standard few-shot learning benchmark codebase.

**Key novelty**: HROT-FSL performs token-level few-shot inference using hyperbolic unbalanced optimal transport with a **meta-learned Episode-Adaptive Mass (EAM)** network — the transported mass ρ is predicted per query-class pair from task-level hyperbolic statistics, not fixed as a scalar hyperparameter.

---


# Hyperbolic geometry library (MANDATORY)
pip install geoopt
https://github.com/geoopt/geoopt
  - geoopt.manifolds.PoincareBall           ← use this for all hyperbolic ops
  - geoopt.manifolds.PoincareBall.dist()    ← hyperbolic distance
  - geoopt.manifolds.PoincareBall.expmap0() ← Euclidean → hyperbolic projection
  - geoopt.manifolds.PoincareBall.logmap0() ← hyperbolic → Euclidean projection
  - geoopt.manifolds.PoincareBall.mobius_matvec()  ← Möbius linear layer
  - geoopt.optim.RiemannianAdam             ← optimizer for hyperbolic params

# Unbalanced OT library (MANDATORY)
pip install POT
https://github.com/PythonOT/POT
  - ot.unbalanced.sinkhorn_knopp_unbalanced()  ← main UOT solver
  - ot.partial.partial_wasserstein()           ← partial OT alternative
  - Parameters: a, b, M, reg (entropy), reg_m (KL marginal penalty)

# Backbone + model_factory reference
https://github.com/thuml/few-shot
  - few-shot/models/backbone.py   ← ResNet12 standard implementation
  - few-shot/config.py            ← config system
  - main.py --method [METHOD]     ← model_factory dispatch
```

---



## DETAILED IMPLEMENTATION SPECIFICATIONS

### 1. `models/hyperbolic/poincare_ops.py`

```python
"""
All hyperbolic operations use geoopt.manifolds.PoincareBall.
CRITICAL: Work in float64 for numerical stability near boundary.
CRITICAL: Clamp all vectors to norm < (1 - EPS) before any operation.
"""

import torch
import geoopt

EPS = 1e-5
BALL_EPS = {torch.float32: 4e-3, torch.float64: 1e-5}

def get_ball(curvature: float = 1.0) -> geoopt.PoincareBall:
    """Return a PoincareBall manifold with given curvature c > 0."""
    return geoopt.PoincareBall(c=curvature)

def safe_project_to_ball(x: torch.Tensor, ball: geoopt.PoincareBall) -> torch.Tensor:
    """
    Safely project Euclidean tensor x into the Poincaré ball.
    Use ball.expmap0(x) — exponential map at origin.
    Clamp norm to prevent boundary explosion.
    """
    # expmap0 maps from tangent space at 0 to manifold
    x_hyp = ball.expmap0(x)
    # safety clamp
    max_norm = 1.0 - BALL_EPS.get(x.dtype, 1e-5)
    norms = x_hyp.norm(dim=-1, keepdim=True).clamp(min=1e-10)
    x_hyp = torch.where(norms > max_norm, x_hyp / norms * max_norm, x_hyp)
    return x_hyp

def hyperbolic_distance_matrix(
    Z_q: torch.Tensor,   # (n_q, d) on Poincaré ball
    Z_c: torch.Tensor,   # (n_c, d) on Poincaré ball
    ball: geoopt.PoincareBall
) -> torch.Tensor:
    """
    Compute pairwise squared hyperbolic distances.
    Returns M of shape (n_q, n_c).
    
    Formula: d_H(u,v)^2 = (2/sqrt(c))^2 * arctanh(sqrt(c) * ||-u ⊕_c v||)^2
    where ⊕_c is Möbius addition.
    
    Implementation: use ball.dist(u, v) and square it.
    Vectorize via broadcasting: expand Z_q to (n_q, 1, d), Z_c to (1, n_c, d).
    """
    Z_q_exp = Z_q.unsqueeze(1)   # (n_q, 1, d)
    Z_c_exp = Z_c.unsqueeze(0)   # (1, n_c, d)
    # ball.dist() broadcasts correctly
    M = ball.dist(Z_q_exp, Z_c_exp) ** 2  # (n_q, n_c)
    return M

def frechet_mean_poincare(
    Z: torch.Tensor,          # (n, d) points on Poincaré ball
    ball: geoopt.PoincareBall,
    max_iter: int = 100,
    tol: float = 1e-5
) -> torch.Tensor:
    """
    Compute Fréchet mean on Poincaré ball via Riemannian gradient descent.
    
    Algorithm (Karcher flow):
      mu_0 = mean of logmap0(Z) then expmap0
      for t in range(max_iter):
        v = mean over i of logmap_{mu}(Z_i)  # tangent vector at mu
        mu = expmap_{mu}(lr * v)
        if ||v|| < tol: break
    
    Returns: mu of shape (d,)
    """
    # Initialize with Euclidean mean projected
    mu = ball.expmap0(ball.logmap0(Z).mean(dim=0))
    lr = 1.0
    for _ in range(max_iter):
        log_vecs = ball.logmap(mu.unsqueeze(0).expand_as(Z), Z)  # (n, d)
        v = log_vecs.mean(dim=0)  # (d,) tangent vector
        if v.norm() < tol:
            break
        mu = ball.expmap(mu, lr * v)
    return mu

def hyperbolic_variance(
    Z: torch.Tensor,          # (n, d) points on Poincaré ball
    mu: torch.Tensor,         # (d,) Fréchet mean
    ball: geoopt.PoincareBall
) -> torch.Tensor:
    """
    Variance = mean of squared hyperbolic distances from Fréchet mean.
    Returns scalar.
    """
    dists = ball.dist(Z, mu.unsqueeze(0).expand_as(Z))  # (n,)
    return (dists ** 2).mean()
```

---

### 2. `models/eam/episode_adaptive_mass.py`

```python
"""
Episode-Adaptive Mass Network (EAM).
Predicts ρ*(q,c) ∈ (0,1] — the fraction of mass to transport
between query token set and class-c support token set.

Input features (all computed in hyperbolic space):
  - mu_q: Fréchet mean of query tokens (d,)
  - mu_c: Fréchet mean of class-c support tokens (d,)
  - var_q: hyperbolic variance of query tokens (scalar)
  - var_c: hyperbolic variance of class-c support tokens (scalar)  
  - cos_sim: cosine similarity of logmap0(mu_q) and logmap0(mu_c) (scalar)
  
Total input dim = d + d + 1 + 1 + 1 = 2d + 3

Architecture: 3-layer MLP with LayerNorm and GELU activation
Output: sigmoid → (0, 1]

CRITICAL: EAM operates in EUCLIDEAN tangent space (logmap0 of hyperbolic embeddings).
DO NOT operate directly on Poincaré ball coordinates in MLP.
"""

import torch
import torch.nn as nn
import geoopt
from models.hyperbolic.poincare_ops import frechet_mean_poincare, hyperbolic_variance

class EpisodeAdaptiveMass(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int = 256):
        super().__init__()
        input_dim = 2 * embed_dim + 3
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        # Initialize to predict ~0.8 by default (near-balanced start)
        nn.init.zeros_(self.net[-2].weight)
        nn.init.constant_(self.net[-2].bias, 1.4)  # sigmoid(1.4) ≈ 0.8

    def forward(
        self,
        Z_q: torch.Tensor,        # (n_q, d) query tokens on Poincaré ball
        Z_c: torch.Tensor,        # (n_c, d) class support tokens on Poincaré ball
        ball: geoopt.PoincareBall
    ) -> torch.Tensor:
        """Returns ρ*(q,c) as a scalar tensor in (0, 1]."""
        # Fréchet means
        mu_q = frechet_mean_poincare(Z_q, ball)   # (d,)
        mu_c = frechet_mean_poincare(Z_c, ball)   # (d,)
        
        # Project means to tangent space for MLP input
        mu_q_tan = ball.logmap0(mu_q)   # (d,)
        mu_c_tan = ball.logmap0(mu_c)   # (d,)
        
        # Variances (scalars)
        var_q = hyperbolic_variance(Z_q, mu_q, ball).unsqueeze(0)  # (1,)
        var_c = hyperbolic_variance(Z_c, mu_c, ball).unsqueeze(0)  # (1,)
        
        # Cosine similarity of tangent means
        cos_sim = torch.nn.functional.cosine_similarity(
            mu_q_tan.unsqueeze(0), mu_c_tan.unsqueeze(0)
        )  # (1,)
        
        feat = torch.cat([mu_q_tan, mu_c_tan, var_q, var_c, cos_sim], dim=0)
        rho = self.net(feat.unsqueeze(0)).squeeze()  # scalar
        return rho.clamp(min=0.1, max=1.0)  # minimum 10% mass transport
```

---

### 3. `modules/unbalanced_ot.py`

```python
"""
Differentiable Unbalanced OT solver wrapper.
Uses PyTorch-native implementation (NOT ot.unbalanced which uses numpy).
Implements log-domain stabilized Sinkhorn-Knopp for unbalanced OT.

Objective:
  Π* = argmin_{Π≥0} <Π, M> + τ_q KL(Π1 || a) + τ_c KL(Π^T1 || b) - ε H(Π)

where:
  M: (n_q, n_c) cost matrix (hyperbolic squared distances)
  a: (n_q,) query marginal (uniform by default: 1/n_q)
  b: (n_c,) support marginal (uniform by default: 1/n_c)
  τ_q, τ_c: KL marginal penalty weights (learnable or fixed)
  ε: entropy regularization (controls plan smoothness)

Returns: Π* of shape (n_q, n_c) — differentiable via autograd.

IMPORTANT: Implement in log-domain for numerical stability.
DO NOT use ot.sinkhorn_unbalanced (numpy-based, not differentiable).
Reference: Chizat et al., "Scaling algorithms for unbalanced transport problems", 2018.
"""

import torch
import torch.nn.functional as F

def sinkhorn_unbalanced_log(
    M: torch.Tensor,        # (n_q, n_c) cost matrix
    a: torch.Tensor,        # (n_q,) source weights
    b: torch.Tensor,        # (n_c,) target weights
    tau_q: float,           # KL penalty on query marginal
    tau_c: float,           # KL penalty on support marginal
    eps: float,             # entropy regularization
    max_iter: int = 100,
    tol: float = 1e-6,
) -> torch.Tensor:
    """
    Log-domain stabilized unbalanced Sinkhorn.
    
    Dual variables (f, g) update:
      f ← -τ_q/(τ_q + ε) * log(Σ_j exp((g_j - M_ij)/ε) * b_j)  [softmin over j]
      g ← -τ_c/(τ_c + ε) * log(Σ_i exp((f_i - M_ij)/ε) * a_i)  [softmin over i]
    
    Transport plan:
      log Π_ij = (f_i + g_j - M_ij) / ε
      Π = exp(log_Π)
    
    Returns differentiable Π* (gradients flow through f, g via autograd).
    """
    n_q, n_c = M.shape
    log_a = torch.log(a + 1e-10)
    log_b = torch.log(b + 1e-10)
    
    # Scaling factors
    rho_q = tau_q / (tau_q + eps)
    rho_c = tau_c / (tau_c + eps)
    
    # Initialize dual potentials
    f = torch.zeros(n_q, dtype=M.dtype, device=M.device)
    g = torch.zeros(n_c, dtype=M.dtype, device=M.device)
    
    for _ in range(max_iter):
        f_prev = f.clone()
        
        # Update f: softmin_j over columns
        log_K_g = (g.unsqueeze(0) - M) / eps          # (n_q, n_c)
        f = -rho_q * torch.logsumexp(log_K_g + log_b.unsqueeze(0), dim=1)  # (n_q,)
        
        # Update g: softmin_i over rows
        log_K_f = (f.unsqueeze(1) - M) / eps          # (n_q, n_c)
        g = -rho_c * torch.logsumexp(log_K_f + log_a.unsqueeze(1), dim=0)  # (n_c,)
        
        # Convergence check
        if (f - f_prev).abs().max() < tol:
            break
    
    # Compute transport plan in log domain
    log_Pi = (f.unsqueeze(1) + g.unsqueeze(0) - M) / eps  # (n_q, n_c)
    Pi = torch.exp(log_Pi)
    return Pi


def compute_transport_cost(Pi: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    """<Π, M>_F — inner product (scalar)."""
    return (Pi * M).sum()


def compute_transported_mass(Pi: torch.Tensor) -> torch.Tensor:
    """1^T Π 1 — total transported mass (scalar)."""
    return Pi.sum()
```

---

### 4. `models/hrot_fsl.py` — MAIN MODEL

```python
"""
HROT-FSL: Hyperbolic Relational Optimal Transport for Few-Shot Learning

Architecture:
  1. Backbone φ: image → spatial feature map (H×W×d_feat)
  2. Projection head π: d_feat → d_hyp (Euclidean linear)
  3. Hyperbolic projection: expmap0 → Poincaré ball
  4. EAM: predict ρ*(q,c) per query-class pair
  5. UOT inference: compute Π* with hyperbolic cost + adaptive mass
  6. Classification: score = -<Π*, M> + β·ρ*(q,c)

Training stages:
  Stage 1 (pretrain): standard cross-entropy on base classes (backbone + projection)
  Stage 2 (meta-train): episodic training, full HROT-FSL objective
    - Loss: cross-entropy on episode scores
    - EAM is jointly trained during meta-training
    
Mode switching (as in DeepEMD):
  model.mode = 'encoder'  → returns feature maps
  model.mode = 'meta'     → full HROT-FSL inference
  model.mode = 'pretrain' → backbone + linear classifier
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import geoopt

from models.backbones.resnet12 import ResNet12
from models.hyperbolic.poincare_ops import (
    get_ball, safe_project_to_ball, hyperbolic_distance_matrix
)
from models.eam.episode_adaptive_mass import EpisodeAdaptiveMass
from modules.unbalanced_ot import sinkhorn_unbalanced_log, compute_transport_cost, compute_transported_mass


class HROT_FSL(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.mode = 'pretrain'
        
        # --- Backbone ---
        self.encoder = ResNet12(
            avg_pool=False,              # keep spatial tokens (5×5 or 7×7)
            drop_rate=args.drop_rate,
            dropblock_size=args.dropblock_size
        )
        d_feat = self.encoder.out_channels  # typically 640 for ResNet12
        
        # --- Hyperbolic projection ---
        self.d_hyp = args.hyp_dim          # e.g., 128
        self.proj = nn.Linear(d_feat, self.d_hyp, bias=False)
        
        # --- Poincaré ball ---
        # Curvature c is learnable (bounded positive)
        self.log_c = nn.Parameter(torch.zeros(1))  # c = softplus(log_c) > 0
        
        # --- Episode-Adaptive Mass network ---
        self.eam = EpisodeAdaptiveMass(
            embed_dim=self.d_hyp,
            hidden_dim=args.eam_hidden_dim  # e.g., 256
        )
        
        # --- UOT hyperparameters (fixed, not learned) ---
        self.tau_q = args.tau_q     # e.g., 0.5
        self.tau_c = args.tau_c     # e.g., 0.5
        self.eps = args.eps_ot      # e.g., 0.1
        
        # --- Score combination weight ---
        self.beta = nn.Parameter(torch.ones(1))  # mass bonus weight
        
        # --- Pretrain classifier (Stage 1 only) ---
        self.pretrain_classifier = None  # set after knowing n_base_classes
    
    @property
    def curvature(self) -> torch.Tensor:
        """Learnable positive curvature c = softplus(log_c)."""
        return F.softplus(self.log_c)
    
    @property
    def ball(self) -> geoopt.PoincareBall:
        """Poincaré ball with current curvature."""
        return geoopt.PoincareBall(c=self.curvature.item())
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 3, H, W) images
        Returns: (B, n_tokens, d_hyp) hyperbolic token embeddings
        """
        feat = self.encoder(x)              # (B, d_feat, h, w)
        B, d, h, w = feat.shape
        feat = feat.permute(0, 2, 3, 1)    # (B, h, w, d_feat)
        feat = feat.reshape(B, h*w, d)     # (B, n_tokens, d_feat)
        z_euc = self.proj(feat)            # (B, n_tokens, d_hyp)
        z_hyp = safe_project_to_ball(z_euc, self.ball)   # (B, n_tokens, d_hyp)
        return z_hyp
    
    def hrot_score(
        self,
        z_q: torch.Tensor,   # (n_q, d_hyp) query tokens
        z_c: torch.Tensor,   # (n_c, d_hyp) class support tokens
    ) -> dict:
        """
        Compute HROT score between query and one class.
        Returns dict with 'score', 'pi', 'rho', 'cost'.
        """
        ball = self.ball
        
        # Hyperbolic cost matrix
        M = hyperbolic_distance_matrix(z_q, z_c, ball)  # (n_q, n_c)
        
        # EAM: predict adaptive mass
        rho = self.eam(z_q, z_c, ball)  # scalar in (0.1, 1.0]
        
        # Uniform marginals scaled to ρ
        n_q, n_c = z_q.shape[0], z_c.shape[0]
        a = torch.full((n_q,), rho.item() / n_q, device=z_q.device, dtype=z_q.dtype)
        b = torch.full((n_c,), rho.item() / n_c, device=z_q.device, dtype=z_q.dtype)
        # Note: a, b may not sum to 1 — this is unbalanced OT's feature
        
        # Unbalanced Sinkhorn
        Pi = sinkhorn_unbalanced_log(
            M=M,
            a=a / a.sum(),   # normalize for Sinkhorn stability
            b=b / b.sum(),
            tau_q=self.tau_q,
            tau_c=self.tau_c,
            eps=self.eps,
        )
        
        # Transport cost
        cost = compute_transport_cost(Pi, M)             # scalar
        mass = compute_transported_mass(Pi)              # scalar
        
        # Classification score: lower cost = better match
        # + mass bonus: classes explaining more query mass score higher
        score = -cost + self.beta * mass
        
        return {'score': score, 'pi': Pi, 'rho': rho, 'cost': cost}
    
    def meta_forward(
        self,
        support: torch.Tensor,   # (n_way * n_shot, 3, H, W)
        query: torch.Tensor,     # (n_query, 3, H, W)
        n_way: int,
        n_shot: int,
    ) -> torch.Tensor:
        """
        Full HROT-FSL episode inference.
        Returns logits: (n_query, n_way)
        """
        n_query = query.shape[0]
        
        # Encode all images
        z_support = self.encode(support)  # (n_way*n_shot, n_tokens, d_hyp)
        z_query = self.encode(query)      # (n_query, n_tokens, d_hyp)
        
        # Reshape support: (n_way, n_shot*n_tokens, d_hyp)
        n_tokens = z_support.shape[1]
        z_support = z_support.reshape(n_way, n_shot * n_tokens, self.d_hyp)
        
        # Compute scores: for each query × each class
        logits = torch.zeros(n_query, n_way, device=query.device)
        
        for q_idx in range(n_query):
            z_q = z_query[q_idx]         # (n_tokens, d_hyp)
            for c_idx in range(n_way):
                z_c = z_support[c_idx]   # (n_shot*n_tokens, d_hyp)
                result = self.hrot_score(z_q, z_c)
                logits[q_idx, c_idx] = result['score']
        
        return logits
    
    def forward(self, x):
        if self.mode == 'encoder':
            return self.encode(x)
        elif self.mode == 'pretrain':
            z = self.encode(x)           # (B, n_tokens, d_hyp)
            # Global average pooling over tokens for pretrain
            z_pool = z.mean(dim=1)       # (B, d_hyp)
            return self.pretrain_classifier(z_pool)
        elif self.mode == 'meta':
            support, query, n_way, n_shot = x
            return self.meta_forward(support, query, n_way, n_shot)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
    
    def setup_pretrain_classifier(self, n_classes: int):
        """Call before Stage 1 training."""
        self.pretrain_classifier = nn.Linear(self.d_hyp, n_classes)
```

---

### 5. `models/model_factory.py`

```python
"""
model_factory.py — Register all few-shot models here.
Pattern from: https://github.com/thuml/few-shot/blob/master/few-shot/models/backbone.py

Usage:
  model = model_factory(args)
"""

from models.hrot_fsl import HROT_FSL

# Optional: import baseline models for comparison
# from models.protonet import ProtoNet
# from models.deepemd import DeepEMD

MODEL_REGISTRY = {
    'hrot_fsl': HROT_FSL,
    # 'protonet': ProtoNet,
    # 'deepemd': DeepEMD,
}

def model_factory(args):
    """
    args.method: str — name of the method
    Returns initialized model (not on GPU yet).
    """
    method = args.method.lower()
    if method not in MODEL_REGISTRY:
        raise ValueError(
            f"Method '{method}' not found. Available: {list(MODEL_REGISTRY.keys())}"
        )
    model = MODEL_REGISTRY[method](args)
    return model
```

---

### 6. `train/meta_train.py`

```python
"""
Episodic meta-training script.

Episode format (N-way K-shot):
  - support: (N*K, 3, 84, 84)
  - query: (N*Q, 3, 84, 84), Q=15 typically
  - labels: (N*Q,) in range [0, N)

Training procedure:
  1. Sample episode
  2. Forward: logits = model.meta_forward(support, query, N, K)
  3. Loss: F.cross_entropy(logits, labels)
  4. Backward + RiemannianAdam step (for hyperbolic params)
  5. Standard Adam step (for Euclidean params)

CRITICAL: Use two optimizers:
  - geoopt.optim.RiemannianAdam for model.log_c (curvature, on manifold)
  - torch.optim.Adam for all other parameters
  
  # Separate parameter groups:
  hyp_params = [model.log_c]
  euc_params = [p for n, p in model.named_parameters() if n != 'log_c']
  
  opt_riem = geoopt.optim.RiemannianAdam(hyp_params, lr=args.lr_hyp)
  opt_euc = torch.optim.Adam(euc_params, lr=args.lr)

Scheduler: CosineAnnealingLR on both optimizers.

EAM regularization: add L2 loss on ρ deviation from 0.8:
  rho_reg = args.lambda_rho * (rho - 0.8).pow(2).mean()
  (prevents EAM from collapsing to ρ=0.1 or ρ=1.0 trivially)
"""
```

---

### 7. Config `configs/mini_imagenet_5w5s.yaml`

```yaml
# HROT-FSL Configuration — miniImageNet 5-way 5-shot

method: hrot_fsl
dataset: mini_imagenet
data_dir: ./data/miniImageNet

# Episode setup
n_way: 5
n_shot: 5
n_query: 15

# Backbone
backbone: resnet12
drop_rate: 0.5
dropblock_size: 5

# Hyperbolic space
hyp_dim: 128
# Curvature: initialized to c=1, learned during meta-training

# EAM
eam_hidden_dim: 256

# Unbalanced OT
tau_q: 0.5
tau_c: 0.5
eps_ot: 0.1
ot_max_iter: 100

# Training Stage 1 (pretrain)
pretrain_epochs: 100
pretrain_lr: 0.05
pretrain_scheduler: cosine
pretrain_batch_size: 64

# Training Stage 2 (meta)
meta_epochs: 200
meta_episodes_per_epoch: 500
lr: 0.0001
lr_hyp: 0.001        # higher lr for curvature (Riemannian Adam)
lambda_rho: 0.01     # EAM regularization weight

# Evaluation
eval_episodes: 2000
confidence_interval: 0.95
```

---

## ABLATION VARIANTS TO IMPLEMENT

Implement the following 5 variants by modifying `hrot_fsl.py` via a `variant` argument:

| Variant | Geometry | Transport | Mass | How to set |
|---------|----------|-----------|------|------------|
| A | Euclidean | Balanced Sinkhorn | — | `args.variant='A'`: use L2 cost, balanced OT |
| B | Euclidean | Unbalanced | Fixed ρ=0.8 | `args.variant='B'`: L2 cost, UOT, fixed rho |
| C | Hyperbolic | Balanced Sinkhorn | — | `args.variant='C'`: hyp cost, balanced OT |
| D | Hyperbolic | Unbalanced | Fixed ρ=0.8 | `args.variant='D'`: hyp cost, UOT, fixed rho |
| E | Hyperbolic | Unbalanced | **Learned ρ*** | `args.variant='E'`: **full HROT-FSL** |

---

## DATASETS

Support the following datasets (same splits as DeepEMD):

```python
# miniImageNet: 64/16/20 classes for train/val/test
# tieredImageNet: 351/97/160 classes
# CUB-200-2011: 100/50/50 classes
# CIFAR-FS: 64/16/20 classes

# Data loading: use PIL + torchvision transforms
# Standard augmentation for training:
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(84),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
# No augmentation for eval (center crop only)
```

---

## REQUIREMENTS

```
# requirements.txt
torch>=2.0.0
torchvision>=0.15.0
geoopt>=0.5.0           # Riemannian optimization + Poincaré ball
POT>=0.9.0              # Python Optimal Transport (for testing/validation)
numpy>=1.24.0
scipy>=1.10.0
pyyaml>=6.0
tqdm>=4.65.0
tensorboard>=2.13.0
pillow>=9.5.0
einops>=0.6.0
```

---

## NUMERICAL STABILITY CHECKLIST

Implement all of these in `utils/numerical.py`:

```python
# 1. Ball boundary safety
def clamp_to_ball(x, ball, eps=1e-5):
    max_norm = 1.0 / (ball.c ** 0.5) - eps  # max norm in ball of curvature c
    ...

# 2. Log-domain Sinkhorn (already in spec above)

# 3. Curvature positivity
c = F.softplus(log_c)  # always positive

# 4. Float64 for hyperbolic ops in evaluation
z_hyp = z_hyp.double()  # cast before hyperbolic distance

# 5. Safe arctanh
def safe_arctanh(x, eps=1e-7):
    return torch.arctanh(x.clamp(-1 + eps, 1 - eps))

# 6. Fréchet mean convergence guard
# Clamp step size in Karcher flow to prevent leaving the ball
```

---

## EVALUATION PROTOCOL

```python
# Standard few-shot evaluation:
# - 2000 episodes, each randomly sampled
# - Report mean accuracy ± 95% confidence interval
# - 95% CI = 1.96 * std / sqrt(2000)
# 
# Datasets × Settings:
# - miniImageNet: 5-way 1-shot, 5-way 5-shot
# - tieredImageNet: 5-way 1-shot, 5-way 5-shot
# - CUB (fine-grained): 5-way 1-shot, 5-way 5-shot
#
# Comparison baselines (reproduce or use published numbers):
# ProtoNet, DeepEMD, BSSD, HypProtoNet, FEAT, CTM
```

---

## IMPORTANT IMPLEMENTATION NOTES

1. **Gradient flow through Sinkhorn**: the log-domain Sinkhorn above is differentiable via PyTorch autograd. Do NOT detach intermediate computations. Gradients flow through `f` and `g` back to `M`, back to `z_hyp`, back to the encoder.

2. **EAM gradients**: EAM forward uses `frechet_mean_poincare` which involves a for-loop. Wrap in `torch.no_grad()` for the Karcher flow iterations but let the final `mu` backpropagate via the last expmap step. Alternatively, use a simpler approximation: mean of `logmap0(Z)` then `expmap0` as a single-step Fréchet mean (faster, sufficient for training signal).

3. **Batched inference**: The nested loop over `n_query × n_way` is correct but slow. Vectorize the cost matrix computation across all query-class pairs using `einops.rearrange` for the final submission benchmark.

4. **Curvature learning**: Start with `log_c = 0` (c=log(2)≈0.693). Monitor curvature during training. If it collapses to 0, add a positivity regularizer `args.lambda_c * (c - 0.1).clamp(max=0).pow(2)`.

5. **Model mode switching**: Follow DeepEMD's pattern exactly. `model.mode` controls forward behavior. Meta-training sets `model.mode = 'meta'` for forward, `model.mode = 'encoder'` only if explicitly extracting features.

6. **Checkpoint saving**: Save `{encoder, proj, log_c, eam, beta}` separately. The pretrained backbone checkpoint (Stage 1) does NOT include EAM or beta.

---

## TESTING

After implementation, verify with unit tests:

```python
# test_hrot_fsl.py

def test_hyperbolic_distance():
    """Check d_H(x, x) = 0 and triangle inequality."""

def test_frechet_mean():
    """Check Fréchet mean of a single point = that point."""

def test_eam_output_range():
    """Check EAM output ∈ [0.1, 1.0]."""

def test_sinkhorn_unbalanced():
    """Check Π ≥ 0 and that lower τ_q/τ_c gives smaller marginal violations."""

def test_full_episode_forward():
    """
    Create toy episode: 5-way 5-shot, 15 queries, 25 tokens per image.
    model.meta_forward(support, query, 5, 5) → logits (75, 5).
    Check shape, no NaNs, gradients flow.
    """

def test_ablation_variants():
    """Each variant A-E should produce valid logits."""
```

---

*This prompt is complete and self-contained. The agent should implement all files in the order: poincare_ops.py → episode_adaptive_mass.py → unbalanced_ot.py → hrot_fsl.py → model_factory.py → train scripts → eval script.*
