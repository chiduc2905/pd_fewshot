import torch
import torch.nn as nn
import torch.nn.functional as F

class CosineBlock(nn.Module):
    def __init__(self):
        super(CosineBlock, self).__init__()
        
    def forward(self, query_feats, support_prototypes):
        """
        Args:
            query_feats: (N_query, Feature_Dim)
            support_prototypes: (N_way, Feature_Dim)
        Returns:
            scores: (N_query, N_way) Cosine similarity scores
        """
        # Normalize features
        q_norm = F.normalize(query_feats, p=2, dim=1)
        s_norm = F.normalize(support_prototypes, p=2, dim=1)
        
        # Compute Cosine Similarity Matrix
        # (N_query, D) @ (N_way, D).T -> (N_query, N_way)
        scores = torch.mm(q_norm, s_norm.t())
        
        return scores

