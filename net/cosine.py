"""Cosine Similarity Network for few-shot learning."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from net.encoder import Conv64F_Encoder
from net.utils import init_weights


class CosineNet(nn.Module):
    """Few-shot classifier using cosine similarity metric."""
    
    def __init__(self, init_type='normal', device='cuda'):
        super(CosineNet, self).__init__()
        self.encoder = Conv64F_Encoder()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 64) 
        
        init_weights(self, init_type=init_type)
        self.to(device)

    def forward(self, query, support):
        """Compute cosine similarity scores.
        
        Args:
            query: (B, NQ, C, H, W) query images
            support: (B, Way, Shot, C, H, W) support images
        Returns:
            scores: (B*NQ, Way) cosine similarity
        """
        B, NQ, C, H, W = query.size()
        B_s, Way, Shot, C_s, H_s, W_s = support.size()
        
        # Flatten
        query_flat = query.view(-1, C, H, W)
        support_flat = support.view(-1, C, H, W)
        
        # Encode
        q_feat = self.encoder(query_flat)
        s_feat = self.encoder(support_flat)
        
        # Pool and Flatten
        q_feat = self.avg_pool(q_feat).view(q_feat.size(0), -1)
        s_feat = self.avg_pool(s_feat).view(s_feat.size(0), -1)
        
        # Apply FC layer (Embedding)
        q_feat = self.fc(q_feat)
        s_feat = self.fc(s_feat)
        
        # Reshape support
        s_feat = s_feat.view(B, Way, Shot, -1)
        
        # Prototypes (Mean of support features)
        prototypes = s_feat.mean(dim=2) # (B, Way, Feature_Dim)
        
        # Reshape query
        q_feat = q_feat.view(B, NQ, -1)
        
        # Compute Cosine Similarity
        # q: (B, NQ, D), p: (B, Way, D)
        # Normalize
        q_norm = F.normalize(q_feat, p=2, dim=2)
        p_norm = F.normalize(prototypes, p=2, dim=2)
        
        # Dot product
        # (B, NQ, D) @ (B, D, Way) -> (B, NQ, Way)
        scores = torch.bmm(q_norm, p_norm.transpose(1, 2))
        
        # Flatten to (B*NQ, Way)
        scores = scores.view(-1, Way)
        
        return scores
