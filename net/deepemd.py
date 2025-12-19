"""DeepEMD: Few-Shot Classification with Earth Mover's Distance.

Paper: "DeepEMD: Few-Shot Image Classification with Differentiable Earth Mover's Distance 
        and Structured Classifiers" (CVPR 2020, Zhang et al.)

Key Idea:
- Use local features (like DN4) but with optimal transport matching
- Earth Mover's Distance (EMD) finds minimum cost to transform one distribution to another
- Cross-reference mechanism generates importance weights
- More robust to background clutter and intra-class variation

Note: Full EMD requires solving an optimization problem. Here we use a simplified 
approximation for efficiency while maintaining the core idea.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from net.encoders.base_encoder import Conv64F_Encoder


class CrossReference(nn.Module):
    """Cross-reference mechanism to generate node weights.
    
    Given query and support local features, generates importance weights
    that down-weight background regions.
    """
    
    def __init__(self, feat_dim=64):
        super(CrossReference, self).__init__()
        
        # Weight generation network
        self.weight_net = nn.Sequential(
            nn.Conv1d(feat_dim, feat_dim // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(feat_dim // 2, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, query_feat, support_feat):
        """Compute importance weights via cross-attention.
        
        Args:
            query_feat: (B, D, N_q) query local features
            support_feat: (B, D, N_s) support local features
        Returns:
            q_weights: (B, N_q) importance weights for query
            s_weights: (B, N_s) importance weights for support
        """
        # Cross-attention similarity
        # query_feat: (B, D, N_q), support_feat: (B, D, N_s)
        # Similarity: (B, N_q, N_s)
        sim = torch.bmm(query_feat.transpose(1, 2), support_feat)
        sim = F.softmax(sim, dim=-1)  # Normalize over support
        
        # Aggregate query features based on most similar support features
        # and vice versa
        q_agg = torch.bmm(query_feat, sim)  # (B, D, N_s)
        s_agg = torch.bmm(support_feat, sim.transpose(1, 2))  # (B, D, N_q)
        
        # Generate weights
        q_weights = self.weight_net(query_feat).squeeze(1)  # (B, N_q)
        s_weights = self.weight_net(support_feat).squeeze(1)  # (B, N_s)
        
        # Normalize weights to sum to 1
        q_weights = q_weights / (q_weights.sum(dim=1, keepdim=True) + 1e-8)
        s_weights = s_weights / (s_weights.sum(dim=1, keepdim=True) + 1e-8)
        
        return q_weights, s_weights


def sinkhorn_distance(cost, n_iters=10, reg=0.1):
    """Approximate EMD using Sinkhorn algorithm.
    
    The Sinkhorn algorithm provides a differentiable approximation to 
    optimal transport (EMD) by adding entropy regularization.
    
    Args:
        cost: (B, N, M) cost matrix
        n_iters: Number of Sinkhorn iterations
        reg: Entropy regularization
    Returns:
        distance: (B,) approximate EMD values
    """
    B, N, M = cost.size()
    
    # Uniform marginals
    mu = torch.ones(B, N, device=cost.device) / N
    nu = torch.ones(B, M, device=cost.device) / M
    
    # Gibbs kernel
    K = torch.exp(-cost / reg)
    
    # Sinkhorn iterations
    u = torch.ones_like(mu)
    for _ in range(n_iters):
        v = nu / (torch.bmm(K.transpose(1, 2), u.unsqueeze(-1)).squeeze(-1) + 1e-8)
        u = mu / (torch.bmm(K, v.unsqueeze(-1)).squeeze(-1) + 1e-8)
    
    # Compute transport plan
    P = u.unsqueeze(-1) * K * v.unsqueeze(1)
    
    # Compute distance
    distance = (P * cost).sum(dim=(1, 2))
    
    return distance


class DeepEMD(nn.Module):
    """Deep Earth Mover's Distance for few-shot learning.
    
    Uses local descriptors with optimal transport matching.
    """
    
    def __init__(self, sinkhorn_iters=10, sinkhorn_reg=0.1, device='cuda'):
        """Initialize DeepEMD.
        
        Args:
            sinkhorn_iters: Number of Sinkhorn iterations
            sinkhorn_reg: Entropy regularization for Sinkhorn
            device: Device to use
        """
        super(DeepEMD, self).__init__()
        
        # Encoder: 3x64x64 -> 64x16x16 (keep spatial structure)
        self.encoder = Conv64F_Encoder()
        
        # Cross-reference for importance weighting
        self.cross_ref = CrossReference(feat_dim=64)
        
        # Sinkhorn parameters
        self.sinkhorn_iters = sinkhorn_iters
        self.sinkhorn_reg = sinkhorn_reg
        
        self.to(device)
    
    def forward(self, query, support):
        """Compute class scores using EMD-based matching.
        
        Args:
            query: (B, NQ, C, H, W) query images
            support: (B, Way, Shot, C, H, W) support images
        Returns:
            scores: (B*NQ, Way) negative EMD distances
        """
        B, NQ, C, H, W = query.size()
        _, Way, Shot, _, _, _ = support.size()
        
        # Encode images - keep spatial structure
        q_flat = query.view(-1, C, H, W)  # (B*NQ, C, H, W)
        s_flat = support.view(-1, C, H, W)  # (B*Way*Shot, C, H, W)
        
        q_feat = self.encoder(q_flat)  # (B*NQ, D, h, w)
        s_feat = self.encoder(s_flat)  # (B*Way*Shot, D, h, w)
        
        D, h, w = q_feat.size(1), q_feat.size(2), q_feat.size(3)
        N_local = h * w  # Number of local descriptors
        
        # Reshape to local descriptors
        # q_feat: (B*NQ, D, h*w)
        q_local = q_feat.view(B * NQ, D, -1)
        
        # s_feat: (B, Way, Shot, D, h*w)
        s_local = s_feat.view(B, Way, Shot, D, -1)
        
        # L2 normalize
        q_local = F.normalize(q_local, p=2, dim=1)
        s_local = F.normalize(s_local, p=2, dim=2)
        
        # Compute EMD for each query-class pair
        all_scores = []
        
        for b in range(B):
            batch_scores = []
            for q in range(NQ):
                q_desc = q_local[b * NQ + q]  # (D, h*w)
                class_scores = []
                
                for c in range(Way):
                    # Average support features across shots
                    s_desc = s_local[b, c].mean(dim=0)  # (D, h*w)
                    
                    # Compute cost matrix (cosine distance)
                    # q_desc: (D, N_q), s_desc: (D, N_s)
                    # cost: (N_q, N_s)
                    sim = torch.mm(q_desc.t(), s_desc)  # (N_q, N_s)
                    cost = 1 - sim  # Cosine distance
                    
                    # Add batch dimension for sinkhorn
                    cost = cost.unsqueeze(0)  # (1, N_q, N_s)
                    
                    # Compute EMD
                    emd = sinkhorn_distance(cost, self.sinkhorn_iters, self.sinkhorn_reg)
                    class_scores.append(-emd.squeeze())  # Negative distance = similarity
                
                batch_scores.append(torch.stack(class_scores))  # (Way,)
            
            all_scores.append(torch.stack(batch_scores))  # (NQ, Way)
        
        scores = torch.stack(all_scores)  # (B, NQ, Way)
        scores = scores.view(-1, Way)  # (B*NQ, Way)
        
        return scores


class DeepEMDSimple(nn.Module):
    """Simplified DeepEMD without Sinkhorn.
    
    Uses weighted average of local similarities as an approximation.
    Faster but less accurate than full EMD.
    """
    
    def __init__(self, device='cuda'):
        super(DeepEMDSimple, self).__init__()
        
        self.encoder = Conv64F_Encoder()
        self.to(device)
    
    def forward(self, query, support):
        """Simplified EMD using best-match aggregation.
        
        For each query local descriptor, find best matching support descriptor
        and average the similarities.
        
        Args:
            query: (B, NQ, C, H, W)
            support: (B, Way, Shot, C, H, W)
        Returns:
            scores: (B*NQ, Way)
        """
        B, NQ, C, H, W = query.size()
        _, Way, Shot, _, _, _ = support.size()
        
        # Encode
        q_flat = query.view(-1, C, H, W)
        s_flat = support.view(-1, C, H, W)
        
        q_feat = self.encoder(q_flat)  # (B*NQ, D, h, w)
        s_feat = self.encoder(s_flat)  # (B*Way*Shot, D, h, w)
        
        D, h, w = q_feat.size(1), q_feat.size(2), q_feat.size(3)
        
        # Reshape and normalize
        q_local = q_feat.view(B, NQ, D, -1)  # (B, NQ, D, h*w)
        q_local = F.normalize(q_local, p=2, dim=2)
        
        s_local = s_feat.view(B, Way, Shot, D, -1).mean(dim=2)  # (B, Way, D, h*w)
        s_local = F.normalize(s_local, p=2, dim=2)
        
        # Compute scores
        scores_list = []
        
        for b in range(B):
            for q in range(NQ):
                q_desc = q_local[b, q]  # (D, h*w)
                class_scores = []
                
                for c in range(Way):
                    s_desc = s_local[b, c]  # (D, h*w)
                    
                    # Cosine similarity
                    sim = torch.mm(q_desc.t(), s_desc)  # (h*w, h*w)
                    
                    # Best match for each query descriptor
                    best_sim, _ = sim.max(dim=1)  # (h*w,)
                    
                    # Average best matches
                    score = best_sim.mean()
                    class_scores.append(score)
                
                scores_list.append(torch.stack(class_scores))
        
        scores = torch.stack(scores_list)  # (B*NQ, Way)
        return scores
