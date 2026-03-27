"""Prototypical Network for few-shot learning."""
import torch
import torch.nn as nn
from net.encoders.protonet_encoder import Conv64F_Paper_Encoder


class ProtoNet(nn.Module):
    """Few-shot classifier using prototype-based Euclidean distance."""
    
    def __init__(self, image_size=64, device='cuda'):
        super(ProtoNet, self).__init__()
        self.encoder = Conv64F_Paper_Encoder(image_size=image_size)
        self.to(device)

    def forward(self, query, support):
        """Compute negative Euclidean distance to prototypes.
        
        Args:
            query: (B, NQ, C, H, W) query images
            support: (B, Way, Shot, C, H, W) support images
        Returns:
            scores: (B*NQ, Way) negative distance
        """
        B, NQ, C, H, W = query.size()
        B_s, Way, Shot, C_s, H_s, W_s = support.size()
        
        # Flatten and encode
        query_flat = query.view(-1, C, H, W)
        support_flat = support.view(-1, C, H, W)
        
        q_feat = self.encoder(query_flat)
        s_feat = self.encoder(support_flat)
        
        # Compute prototypes
        s_feat = s_feat.view(B, Way, Shot, -1)
        prototypes = s_feat.mean(dim=2)
        
        # Compute squared Euclidean distances (as in official ProtoNet paper)
        q_feat = q_feat.view(B, NQ, -1)
        dists = torch.cdist(q_feat, prototypes).pow(2)  # Squared Euclidean distance
        
        # Return negative distance as scores
        scores = -dists.view(-1, Way)
        return scores
