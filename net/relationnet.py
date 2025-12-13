"""Relation Network for few-shot learning."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from net.encoders.relationnet_encoder import RelationNetEncoder
from net.relation_block import RelationBlock
from net.utils import weights_init_relationnet


class RelationNet(nn.Module):
    """
    Relation Network for Few-Shot Learning.
    
    Paper: Sung et al. "Learning to Compare: Relation Network for Few-Shot Learning" (CVPR 2018)
    
    Key components:
    1. Embedding module (CNN encoder) - extracts features
    2. Relation module - learns to compare feature pairs
    3. Learns metric end-to-end instead of using fixed metric
    
    Architecture:
    - Encoder: 4-layer CNN -> (B, 64, 4, 4) feature maps
    - Relation: Concat query & support features -> MLP -> relation score
    """
    
    def __init__(self, device='cuda'):
        """
        Args:
            device: Device to use
        """
        super(RelationNet, self).__init__()
        
        self.encoder = RelationNetEncoder()  # Output: (B, 64, 4, 4)
        
        # Relation module: input is concat of query and support (128 channels)
        self.relation = RelationBlock(input_size=128, hidden_size=8)
        
        self.apply(weights_init_relationnet)  # Paper-specific init
        self.to(device)
    
    def forward(self, query, support):
        """
        Compute relation scores between query and support.
        
        Args:
            query: (B, NQ, C, H, W) query images
            support: (B, Way, Shot, C, H, W) support images
        Returns:
            scores: (B*NQ, Way) relation scores
        """
        B, NQ, C, H, W = query.size()
        B_s, Way, Shot, C_s, H_s, W_s = support.size()
        
        # Flatten and encode
        query_flat = query.view(-1, C, H, W)
        support_flat = support.view(-1, C, H, W)
        
        q_feat = self.encoder(query_flat)  # (B*NQ, 64, 4, 4)
        s_feat = self.encoder(support_flat)  # (B*Way*Shot, 64, 4, 4)
        
        # Reshape support features
        s_feat = s_feat.view(B, Way, Shot, 64, 4, 4)
        
        # Compute prototypes (mean over shot)
        prototypes = s_feat.mean(dim=2)  # (B, Way, 64, 4, 4)
        
        # Reshape query features
        q_feat = q_feat.view(B, NQ, 64, 4, 4)
        
        # Compute relation scores
        # For each query, concat with each prototype and pass through relation module
        scores_list = []
        
        for b in range(B):
            q_b = q_feat[b]  # (NQ, 64, 4, 4)
            p_b = prototypes[b]  # (Way, 64, 4, 4)
            
            # Expand and concatenate
            # q_b: (NQ, 1, 64, 4, 4) -> (NQ, Way, 64, 4, 4)
            # p_b: (1, Way, 64, 4, 4) -> (NQ, Way, 64, 4, 4)
            q_expanded = q_b.unsqueeze(1).expand(NQ, Way, 64, 4, 4)
            p_expanded = p_b.unsqueeze(0).expand(NQ, Way, 64, 4, 4)
            
            # Concatenate along channel dimension
            relation_pairs = torch.cat([q_expanded, p_expanded], dim=2)  # (NQ, Way, 128, 4, 4)
            
            # Reshape for relation module
            relation_pairs = relation_pairs.view(NQ * Way, 128, 4, 4)
            
            # Compute relation scores
            relations = self.relation(relation_pairs)  # (NQ*Way, 1)
            relations = relations.view(NQ, Way)  # (NQ, Way)
            
            scores_list.append(relations)
        
        scores = torch.cat(scores_list, dim=0)  # (B*NQ, Way)
        
        return scores
