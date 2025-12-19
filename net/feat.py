"""FEAT: Few-Shot Embedding Adaptation with Transformer.

Paper: "FEAT: Few-Shot Embedding Adaptation with Transformer" (ICLR 2021, Ye et al.)

Key Idea:
- Adapt embeddings based on the specific task (support set) using attention
- Set-to-set function via Transformer
- Task-adaptive embeddings capture inter-class relationships
- Use adapted features with Euclidean distance (like ProtoNet)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from net.encoders.base_encoder import Conv64F_Encoder


class SetTransformer(nn.Module):
    """Simple Transformer block for set-to-set embedding adaptation.
    
    Takes a set of embeddings and adapts them based on the entire set context.
    """
    
    def __init__(self, dim, n_heads=4, dropout=0.1):
        """Initialize SetTransformer.
        
        Args:
            dim: Feature dimension
            n_heads: Number of attention heads
            dropout: Dropout rate
        """
        super(SetTransformer, self).__init__()
        
        # Multi-head self-attention
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        
        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(dim)
    
    def forward(self, x):
        """Apply self-attention adaptation.
        
        Args:
            x: (B, N, D) embeddings where N is set size
        Returns:
            (B, N, D) adapted embeddings
        """
        # Self-attention with residual
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        
        # Feed-forward with residual
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        
        return x


class FEAT(nn.Module):
    """Few-Shot Embedding Adaptation with Transformer.
    
    Uses a transformer to adapt embeddings based on the task context,
    then applies Euclidean distance for classification (like ProtoNet).
    """
    
    def __init__(self, temperature=0.2, device='cuda'):
        """Initialize FEAT.
        
        Args:
            temperature: Scaling factor for distance (default 0.2 as in paper)
            device: Device to use
        """
        super(FEAT, self).__init__()
        
        # Base encoder: 3x64x64 -> 64x16x16
        self.encoder = Conv64F_Encoder()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Feature dimension after pooling
        self.feat_dim = 64
        
        # Set transformer for embedding adaptation
        self.set_transformer = SetTransformer(
            dim=self.feat_dim,
            n_heads=4,
            dropout=0.1
        )
        
        # Temperature scaling for distances
        self.temperature = temperature
        
        self.to(device)
    
    def encode(self, x):
        """Extract feature embedding.
        
        Args:
            x: (N, C, H, W) images
        Returns:
            (N, feat_dim) embeddings
        """
        feat = self.encoder(x)  # (N, 64, h, w)
        feat = self.avg_pool(feat)  # (N, 64, 1, 1)
        feat = feat.view(feat.size(0), -1)  # (N, 64)
        return feat
    
    def forward(self, query, support):
        """Compute class scores with adapted embeddings.
        
        Process:
        1. Encode all images to get base embeddings
        2. Concatenate support and query embeddings as a set
        3. Apply set transformer to adapt all embeddings jointly
        4. Compute prototypes from adapted support embeddings
        5. Classify queries using Euclidean distance to prototypes
        
        Args:
            query: (B, NQ, C, H, W) query images
            support: (B, Way, Shot, C, H, W) support images
        Returns:
            scores: (B*NQ, Way) negative squared Euclidean distances
        """
        B, NQ, C, H, W = query.size()
        _, Way, Shot, _, _, _ = support.size()
        
        # Encode all images
        q_flat = query.view(-1, C, H, W)  # (B*NQ, C, H, W)
        s_flat = support.view(-1, C, H, W)  # (B*Way*Shot, C, H, W)
        
        q_emb = self.encode(q_flat).view(B, NQ, -1)  # (B, NQ, D)
        s_emb = self.encode(s_flat).view(B, Way * Shot, -1)  # (B, Way*Shot, D)
        
        # Concatenate for joint adaptation
        # Set: [support embeddings ; query embeddings]
        all_emb = torch.cat([s_emb, q_emb], dim=1)  # (B, Way*Shot + NQ, D)
        
        # Apply set transformer for task-adaptive embedding
        adapted_emb = self.set_transformer(all_emb)  # (B, Way*Shot + NQ, D)
        
        # Split back to support and query
        adapted_s = adapted_emb[:, :Way * Shot, :].view(B, Way, Shot, -1)  # (B, Way, Shot, D)
        adapted_q = adapted_emb[:, Way * Shot:, :]  # (B, NQ, D)
        
        # Compute prototypes from adapted support embeddings
        prototypes = adapted_s.mean(dim=2)  # (B, Way, D)
        
        # Compute squared Euclidean distance
        # adapted_q: (B, NQ, D), prototypes: (B, Way, D)
        dists = torch.cdist(adapted_q, prototypes).pow(2)  # (B, NQ, Way)
        
        # Return negative distances scaled by temperature
        scores = -dists / self.temperature
        scores = scores.view(-1, Way)  # (B*NQ, Way)
        
        return scores


class FEATContrastive(nn.Module):
    """FEAT with contrastive auxiliary loss support.
    
    Adds ability to use contrastive learning on embeddings.
    """
    
    def __init__(self, temperature=0.2, device='cuda'):
        super(FEATContrastive, self).__init__()
        
        self.encoder = Conv64F_Encoder()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.feat_dim = 64
        
        self.set_transformer = SetTransformer(
            dim=self.feat_dim,
            n_heads=4,
            dropout=0.1
        )
        
        self.temperature = temperature
        self.to(device)
    
    def encode(self, x):
        feat = self.encoder(x)
        feat = self.avg_pool(feat)
        feat = feat.view(feat.size(0), -1)
        return feat
    
    def forward(self, query, support, return_features=False):
        """Forward pass with optional feature return for auxiliary losses.
        
        Args:
            query: (B, NQ, C, H, W)
            support: (B, Way, Shot, C, H, W)
            return_features: If True, also return adapted features
        Returns:
            scores: (B*NQ, Way)
            (optional) adapted_q: (B*NQ, D) - if return_features=True
        """
        B, NQ, C, H, W = query.size()
        _, Way, Shot, _, _, _ = support.size()
        
        q_flat = query.view(-1, C, H, W)
        s_flat = support.view(-1, C, H, W)
        
        q_emb = self.encode(q_flat).view(B, NQ, -1)
        s_emb = self.encode(s_flat).view(B, Way * Shot, -1)
        
        all_emb = torch.cat([s_emb, q_emb], dim=1)
        adapted_emb = self.set_transformer(all_emb)
        
        adapted_s = adapted_emb[:, :Way * Shot, :].view(B, Way, Shot, -1)
        adapted_q = adapted_emb[:, Way * Shot:, :]
        
        prototypes = adapted_s.mean(dim=2)
        dists = torch.cdist(adapted_q, prototypes).pow(2)
        scores = -dists / self.temperature
        scores = scores.view(-1, Way)
        
        if return_features:
            return scores, adapted_q.view(-1, self.feat_dim)
        
        return scores
