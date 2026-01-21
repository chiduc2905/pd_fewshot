"""Siamese Network for few-shot learning.

Paper: "Siamese Neural Networks for One-shot Image Recognition" (Koch et al., ICML-W 2015)

Original Architecture:
- 4 convolutional layers with increasing channels
- L1 distance between embeddings
- Weighted L1 → Sigmoid → binary similarity score

This implementation uses Conv64F_Encoder (same as other few-shot models) for:
1. Fair comparison across all models
2. Consistent parameter count (~113K for encoder)
3. Works with any input size due to adaptive pooling

The key Siamese concepts preserved:
- Shared encoder (twin network)
- L1 distance metric
- Learned similarity function
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from net.encoders.base_encoder import Conv64F_Encoder


class SiameseEncoder(nn.Module):
    """Siamese encoder using Conv64F backbone for fair comparison.
    
    Uses the same Conv64F_Encoder as other few-shot models to ensure
    fair benchmarking. Outputs 64-dim embeddings via GAP.
    """
    
    def __init__(self, feat_dim=64):
        super(SiameseEncoder, self).__init__()
        
        self.feat_dim = feat_dim
        
        # Use standard Conv64F encoder (same as ProtoNet, CovaMNet, etc.)
        self.encoder = Conv64F_Encoder()
        
        # Global Average Pooling for 64-dim output
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Optional projection to different feature dimension
        if feat_dim != 64:
            self.projection = nn.Sequential(
                nn.Linear(64, feat_dim),
                nn.Sigmoid()
            )
        else:
            self.projection = nn.Sigmoid()  # Paper uses sigmoid activation
    
    def forward(self, x):
        """Extract feature embedding.
        
        Args:
            x: (N, C, H, W) images
        Returns:
            (N, feat_dim) embeddings
        """
        x = self.encoder(x)  # (N, 64, 16, 16)
        x = self.gap(x)  # (N, 64, 1, 1)
        x = x.view(x.size(0), -1)  # (N, 64)
        x = self.projection(x)  # (N, feat_dim)
        return x


class SiameseNet(nn.Module):
    """Koch et al. 2015 Siamese Network using Conv64F encoder.
    
    For few-shot episodic training:
    - For each query, computes pairwise similarity to each support sample
    - Average similarities per class to get class scores
    - Use CrossEntropyLoss on scores (adapted from binary verification)
    
    Key features:
    - Conv64F encoder (same as other few-shot models for fair comparison)
    - Feature dimension 64 (matching other models)
    - Sigmoid activations (paper-accurate)
    - L1 distance + weighted FC for similarity (paper-accurate)
    """
    
    def __init__(self, feat_dim=64, device='cuda'):
        """Initialize Siamese Network.
        
        Args:
            feat_dim: Feature dimension after encoder (default: 64)
            device: Device to use
        """
        super(SiameseNet, self).__init__()
        
        self.feat_dim = feat_dim
        
        # Use Conv64F encoder for fair comparison
        self.encoder = SiameseEncoder(feat_dim=feat_dim)
        
        # Similarity network: weighted L1 distance + sigmoid
        # Paper: learns weights α for |h1 - h2| * α, then sigmoid
        # We use a learnable linear layer which achieves the same
        self.similarity = nn.Sequential(
            nn.Linear(feat_dim, 1),  # Weighted sum of L1 distances
            nn.Sigmoid()  # Output similarity in [0, 1]
        )
        
        # Initialize similarity layer
        nn.init.normal_(self.similarity[0].weight, mean=0, std=0.2)
        nn.init.constant_(self.similarity[0].bias, 0)
        
        self.to(device)
    
    def encode(self, x):
        """Extract feature embedding from encoder."""
        return self.encoder(x)
    
    def compute_similarity(self, emb1, emb2):
        """Compute similarity between two embeddings.
        
        Paper uses: σ(Σ_j α_j |h1_j - h2_j|)
        
        Args:
            emb1: (N, D) first embeddings
            emb2: (N, D) second embeddings
        Returns:
            (N, 1) similarity scores in [0, 1]
        """
        # L1 distance (element-wise absolute difference)
        l1_dist = torch.abs(emb1 - emb2)  # (N, D)
        # Weighted sum + sigmoid
        sim = self.similarity(l1_dist)  # (N, 1)
        return sim
    
    def forward(self, query, support):
        """Compute similarity scores for few-shot classification.
        
        For N-way K-shot:
        1. Encode all images with shared encoder
        2. For each query, compute pairwise similarity to each support sample
        3. Average similarities per class to get class scores
        
        Args:
            query: (B, NQ, C, H, W) query images
            support: (B, Way, Shot, C, H, W) support images
        Returns:
            scores: (B*NQ, Way) similarity scores per class
        """
        B, NQ, C, H, W = query.size()
        _, Way, Shot, _, _, _ = support.size()
        
        # Flatten and encode
        query_flat = query.view(-1, C, H, W)  # (B*NQ, C, H, W)
        support_flat = support.view(-1, C, H, W)  # (B*Way*Shot, C, H, W)
        
        q_emb = self.encode(query_flat)  # (B*NQ, D)
        s_emb = self.encode(support_flat)  # (B*Way*Shot, D)
        
        # Reshape for batch processing
        q_emb = q_emb.view(B, NQ, -1)  # (B, NQ, D)
        s_emb = s_emb.view(B, Way, Shot, -1)  # (B, Way, Shot, D)
        
        D = q_emb.size(-1)
        
        # Compute pairwise similarities and average per class
        # Shape transformations for vectorized computation:
        # q_emb: (B, NQ, 1, 1, D)
        # s_emb: (B, 1, Way, Shot, D)
        q_exp = q_emb.unsqueeze(2).unsqueeze(3)  # (B, NQ, 1, 1, D)
        s_exp = s_emb.unsqueeze(1)  # (B, 1, Way, Shot, D)
        
        # Broadcast to (B, NQ, Way, Shot, D)
        q_exp = q_exp.expand(B, NQ, Way, Shot, D)
        s_exp = s_exp.expand(B, NQ, Way, Shot, D)
        
        # L1 distance for all pairs
        l1_dist = torch.abs(q_exp - s_exp)  # (B, NQ, Way, Shot, D)
        
        # Flatten for similarity computation
        l1_dist_flat = l1_dist.view(-1, D)  # (B*NQ*Way*Shot, D)
        
        # Compute pairwise similarities
        pair_sims = self.similarity(l1_dist_flat)  # (B*NQ*Way*Shot, 1)
        pair_sims = pair_sims.view(B, NQ, Way, Shot)  # (B, NQ, Way, Shot)
        
        # Average across shots to get class scores (paper compares to each support sample)
        class_scores = pair_sims.mean(dim=3)  # (B, NQ, Way)
        
        # Flatten for loss
        scores = class_scores.view(-1, Way)  # (B*NQ, Way)
        
        return scores


class SiameseNetFast(nn.Module):
    """Optimized Siamese Network using Conv64F encoder.
    
    Uses prototype-based comparison for efficiency while maintaining
    the paper's core concepts (L1 distance, learned similarity).
    """
    
    def __init__(self, feat_dim=64, device='cuda'):
        super(SiameseNetFast, self).__init__()
        
        self.feat_dim = feat_dim
        self.encoder = SiameseEncoder(feat_dim=feat_dim)
        
        # Similarity network
        self.similarity = nn.Sequential(
            nn.Linear(feat_dim, 1),
            nn.Sigmoid()
        )
        
        nn.init.normal_(self.similarity[0].weight, mean=0, std=0.2)
        nn.init.constant_(self.similarity[0].bias, 0)
        
        self.to(device)
    
    def encode(self, x):
        return self.encoder(x)
    
    def forward(self, query, support):
        """Vectorized forward pass using prototypes.
        
        Args:
            query: (B, NQ, C, H, W)
            support: (B, Way, Shot, C, H, W)
        Returns:
            scores: (B*NQ, Way)
        """
        B, NQ, C, H, W = query.size()
        _, Way, Shot, _, _, _ = support.size()
        
        # Encode all images
        q_flat = query.view(-1, C, H, W)
        s_flat = support.view(-1, C, H, W)
        
        q_emb = self.encode(q_flat).view(B, NQ, -1)  # (B, NQ, D)
        s_emb = self.encode(s_flat).view(B, Way, Shot, -1)  # (B, Way, Shot, D)
        
        D = q_emb.size(-1)
        
        # Compute prototypes (average support embeddings per class)
        prototypes = s_emb.mean(dim=2)  # (B, Way, D)
        
        # Expand for pairwise comparison
        q_exp = q_emb.unsqueeze(2).expand(-1, -1, Way, -1)  # (B, NQ, Way, D)
        p_exp = prototypes.unsqueeze(1).expand(-1, NQ, -1, -1)  # (B, NQ, Way, D)
        
        # L1 distance
        l1_dist = torch.abs(q_exp - p_exp)  # (B, NQ, Way, D)
        l1_dist = l1_dist.view(-1, D)  # (B*NQ*Way, D)
        
        # Compute similarity scores
        sims = self.similarity(l1_dist)  # (B*NQ*Way, 1)
        scores = sims.view(B, NQ, Way)  # (B, NQ, Way)
        scores = scores.view(-1, Way)  # (B*NQ, Way)
        
        return scores
