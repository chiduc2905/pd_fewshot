"""Cosine Classifier (Baseline++) for few-shot learning.

Paper: "A Closer Look at Few-Shot Classification" (Chen et al., ICLR 2019)

Key differences from simple cosine similarity:
1. Learnable temperature/scale factor (crucial for good gradient flow)
2. Weight vectors as class prototypes (originally for transfer learning)
3. Can be used in both episodic and transfer learning settings

For few-shot episodic setting:
- Compute support prototypes (mean of support embeddings)
- Compute scaled cosine similarity to prototypes
- Apply softmax for classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from net.encoders.base_encoder import Conv64F_Encoder


class CosineClassifier(nn.Module):
    """Cosine Classifier (Baseline++) from Chen et al. ICLR 2019.
    
    Uses cosine similarity with learnable temperature scaling.
    Temperature is crucial - without it, cosine in [-1, 1] gives tiny gradients.
    """
    
    def __init__(self, temperature=10.0, learnable_temp=True, device='cuda'):
        """Initialize Cosine Classifier.
        
        Args:
            temperature: Initial temperature/scale factor (default 10.0 as in paper)
            learnable_temp: If True, temperature is a learnable parameter
            device: Device to use
        """
        super(CosineClassifier, self).__init__()
        
        # Encoder: 3x64x64 -> 64x16x16 -> 64 (after pooling)
        self.encoder = Conv64F_Encoder()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Temperature scaling (crucial for cosine classifier!)
        # Paper uses fixed scale=10 or learnable
        if learnable_temp:
            # Initialize with log to ensure positive after exp()
            self.temperature = nn.Parameter(torch.tensor(float(temperature)))
        else:
            self.register_buffer('temperature', torch.tensor(float(temperature)))
        
        self.learnable_temp = learnable_temp
        self.to(device)
    
    def encode(self, x):
        """Extract L2-normalized feature embedding.
        
        Args:
            x: (N, C, H, W) images
        Returns:
            (N, D) L2-normalized embeddings
        """
        feat = self.encoder(x)  # (N, 64, h, w)
        feat = self.avg_pool(feat)  # (N, 64, 1, 1)
        feat = feat.view(feat.size(0), -1)  # (N, 64)
        feat = F.normalize(feat, p=2, dim=1)  # L2 normalize
        return feat
    
    def forward(self, query, support):
        """Compute scaled cosine similarity scores.
        
        Args:
            query: (B, NQ, C, H, W) query images
            support: (B, Way, Shot, C, H, W) support images
        Returns:
            scores: (B*NQ, Way) scaled cosine similarities
        """
        B, NQ, C, H, W = query.size()
        _, Way, Shot, _, _, _ = support.size()
        
        # Encode all images
        q_flat = query.view(-1, C, H, W)
        s_flat = support.view(-1, C, H, W)
        
        q_emb = self.encode(q_flat)  # (B*NQ, D), normalized
        s_emb = self.encode(s_flat)  # (B*Way*Shot, D), normalized
        
        # Reshape for prototype computation
        s_emb = s_emb.view(B, Way, Shot, -1)  # (B, Way, Shot, D)
        q_emb = q_emb.view(B, NQ, -1)  # (B, NQ, D)
        
        # Compute class prototypes (mean of support embeddings)
        prototypes = s_emb.mean(dim=2)  # (B, Way, D)
        
        # Re-normalize prototypes (important after averaging!)
        prototypes = F.normalize(prototypes, p=2, dim=2)
        
        # Compute cosine similarity
        # q_emb: (B, NQ, D), prototypes: (B, Way, D)
        # (B, NQ, D) @ (B, D, Way) -> (B, NQ, Way)
        cos_sim = torch.bmm(q_emb, prototypes.transpose(1, 2))
        
        # Apply temperature scaling (this is the key difference!)
        # Without this, gradients are too small because cos ∈ [-1, 1]
        scores = self.temperature * cos_sim
        
        # Flatten for loss computation
        scores = scores.view(-1, Way)  # (B*NQ, Way)
        
        return scores


class CosineClassifierWithFC(nn.Module):
    """Cosine Classifier with additional projection layer.
    
    Some implementations add an FC layer for embedding projection
    before computing cosine similarity.
    """
    
    def __init__(self, embed_dim=64, temperature=10.0, learnable_temp=True, device='cuda'):
        """Initialize Cosine Classifier with FC.
        
        Args:
            embed_dim: Embedding dimension after projection
            temperature: Initial temperature/scale factor
            learnable_temp: If True, temperature is learnable
            device: Device to use
        """
        super(CosineClassifierWithFC, self).__init__()
        
        self.encoder = Conv64F_Encoder()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Projection layer (optional, some papers use this)
        self.projection = nn.Sequential(
            nn.Linear(64, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Temperature
        if learnable_temp:
            self.temperature = nn.Parameter(torch.tensor(float(temperature)))
        else:
            self.register_buffer('temperature', torch.tensor(float(temperature)))
        
        self.to(device)
    
    def encode(self, x):
        feat = self.encoder(x)
        feat = self.avg_pool(feat).view(feat.size(0), -1)
        feat = self.projection(feat)
        feat = F.normalize(feat, p=2, dim=1)
        return feat
    
    def forward(self, query, support):
        B, NQ, C, H, W = query.size()
        _, Way, Shot, _, _, _ = support.size()
        
        q_flat = query.view(-1, C, H, W)
        s_flat = support.view(-1, C, H, W)
        
        q_emb = self.encode(q_flat).view(B, NQ, -1)
        s_emb = self.encode(s_flat).view(B, Way, Shot, -1)
        
        prototypes = F.normalize(s_emb.mean(dim=2), p=2, dim=2)
        
        cos_sim = torch.bmm(q_emb, prototypes.transpose(1, 2))
        scores = self.temperature * cos_sim
        scores = scores.view(-1, Way)
        
        return scores
