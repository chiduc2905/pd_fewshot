"""Siamese Network for few-shot learning.

Paper: "Siamese Neural Networks for One-shot Image Recognition" (Koch et al., ICML-W 2015)

Architecture:
- Twin networks with shared weights (encoder)
- L1 distance between embeddings
- FC layers to predict similarity score
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from net.encoders.base_encoder import Conv64F_Encoder


class SiameseNet(nn.Module):
    """Few-shot classifier using learned pairwise similarity.
    
    For each query, computes similarity to each support sample,
    then averages per class to get class scores.
    """
    
    def __init__(self, device='cuda'):
        """Initialize Siamese Network.
        
        Args:
            device: Device to use
        """
        super(SiameseNet, self).__init__()
        
        # Shared encoder: 3x64x64 -> 64x16x16 (feature maps)
        self.encoder = Conv64F_Encoder()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Similarity network: takes L1 distance of embeddings
        # Input: 64-dim (absolute difference)
        # Output: 1 (similarity score)
        self.relation_net = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )
        
        self.to(device)
    
    def encode(self, x):
        """Extract feature embedding.
        
        Args:
            x: (N, C, H, W) images
        Returns:
            (N, 64) embeddings
        """
        feat = self.encoder(x)  # (N, 64, H, W)
        feat = self.avg_pool(feat)  # (N, 64, 1, 1)
        feat = feat.view(feat.size(0), -1)  # (N, 64)
        return feat
    
    def forward(self, query, support):
        """Compute similarity scores for classification.
        
        For N-way K-shot:
        1. Encode all images
        2. For each query, compute similarity to each support sample
        3. Average similarities per class
        
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
        
        q_emb = self.encode(query_flat)  # (B*NQ, 64)
        s_emb = self.encode(support_flat)  # (B*Way*Shot, 64)
        
        # Reshape for batch processing
        q_emb = q_emb.view(B, NQ, -1)  # (B, NQ, 64)
        s_emb = s_emb.view(B, Way, Shot, -1)  # (B, Way, Shot, 64)
        
        # Compute similarity for each query-support pair
        # Output: (B, NQ, Way)
        scores = []
        
        for b in range(B):
            batch_scores = []
            for q in range(NQ):
                q_feat = q_emb[b, q]  # (64,)
                class_sims = []
                
                for w in range(Way):
                    # Compute similarity to all shots of this class
                    shot_sims = []
                    for s in range(Shot):
                        s_feat = s_emb[b, w, s]  # (64,)
                        
                        # L1 distance as input to relation network
                        diff = torch.abs(q_feat - s_feat)  # (64,)
                        sim = self.relation_net(diff)  # (1,)
                        shot_sims.append(sim)
                    
                    # Average similarity across shots for this class
                    class_sim = torch.stack(shot_sims).mean()
                    class_sims.append(class_sim)
                
                batch_scores.append(torch.stack(class_sims))  # (Way,)
            
            scores.append(torch.stack(batch_scores))  # (NQ, Way)
        
        scores = torch.stack(scores)  # (B, NQ, Way)
        scores = scores.view(-1, Way)  # (B*NQ, Way)
        
        return scores


class SiameseNetFast(nn.Module):
    """Optimized Siamese Network using vectorized operations.
    
    This version avoids nested loops for better GPU utilization.
    """
    
    def __init__(self, device='cuda'):
        super(SiameseNetFast, self).__init__()
        
        self.encoder = Conv64F_Encoder()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Relation network (same as above)
        self.relation_net = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.to(device)
    
    def encode(self, x):
        feat = self.encoder(x)
        feat = self.avg_pool(feat)
        feat = feat.view(feat.size(0), -1)
        return feat
    
    def forward(self, query, support):
        """Vectorized forward pass.
        
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
        
        # Compute prototypes (average support embeddings per class) for efficiency
        # This gives us a proxy for class representation
        prototypes = s_emb.mean(dim=2)  # (B, Way, D)
        
        # Expand for pairwise comparison
        # q_emb: (B, NQ, D) -> (B, NQ, Way, D)
        # prototypes: (B, Way, D) -> (B, NQ, Way, D)
        q_exp = q_emb.unsqueeze(2).expand(-1, -1, Way, -1)
        p_exp = prototypes.unsqueeze(1).expand(-1, NQ, -1, -1)
        
        # L1 distance
        diff = torch.abs(q_exp - p_exp)  # (B, NQ, Way, D)
        diff = diff.view(-1, D)  # (B*NQ*Way, D)
        
        # Compute similarity scores
        sims = self.relation_net(diff)  # (B*NQ*Way, 1)
        scores = sims.view(B, NQ, Way)  # (B, NQ, Way)
        scores = scores.view(-1, Way)  # (B*NQ, Way)
        
        return scores
