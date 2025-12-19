"""DN4: Deep Nearest Neighbor Neural Network for few-shot learning.

Paper: "Revisiting Local Descriptor based Image-to-Class Measure for Few-shot Learning"
       (CVPR 2019, Li et al.)

Key Idea:
- Keep spatial feature maps (local descriptors) instead of global pooling
- For each query local descriptor, find k-nearest neighbors in support set
- Vote for class based on neighbor labels
- Well-suited for texture/pattern recognition like PD scalograms
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from net.encoders.base_encoder import Conv64F_Encoder


class DN4(nn.Module):
    """Deep Nearest Neighbor Neural Network.
    
    Uses local descriptors and k-NN for few-shot classification.
    """
    
    def __init__(self, k_neighbors=3, device='cuda'):
        """Initialize DN4.
        
        Args:
            k_neighbors: Number of nearest neighbors for voting
            device: Device to use
        """
        super(DN4, self).__init__()
        
        # Encoder: 3x64x64 -> 64x16x16 (keep spatial structure)
        self.encoder = Conv64F_Encoder()
        self.k = k_neighbors
        
        self.to(device)
    
    def forward(self, query, support):
        """Compute class scores using local descriptor k-NN.
        
        Args:
            query: (B, NQ, C, H, W) query images
            support: (B, Way, Shot, C, H, W) support images
        Returns:
            scores: (B*NQ, Way) class scores
        """
        B, NQ, C, H, W = query.size()
        _, Way, Shot, _, _, _ = support.size()
        
        # Encode all images - keep spatial structure
        q_flat = query.view(-1, C, H, W)  # (B*NQ, C, H, W)
        s_flat = support.view(-1, C, H, W)  # (B*Way*Shot, C, H, W)
        
        q_feat = self.encoder(q_flat)  # (B*NQ, D, h, w) where D=64
        s_feat = self.encoder(s_flat)  # (B*Way*Shot, D, h, w)
        
        D, h, w = q_feat.size(1), q_feat.size(2), q_feat.size(3)
        num_local = h * w  # Number of local descriptors per image
        
        # Reshape to local descriptors
        # q_feat: (B*NQ, D, h, w) -> (B, NQ, D, h*w) -> (B, NQ, h*w, D)
        q_local = q_feat.view(B, NQ, D, -1).permute(0, 1, 3, 2)  # (B, NQ, h*w, D)
        
        # s_feat: (B*Way*Shot, D, h, w) -> (B, Way, Shot, D, h*w) -> (B, Way, Shot*h*w, D)
        s_local = s_feat.view(B, Way, Shot, D, -1).permute(0, 1, 2, 4, 3)  # (B, Way, Shot, h*w, D)
        s_local = s_local.view(B, Way, -1, D)  # (B, Way, Shot*h*w, D)
        
        # L2 normalize descriptors for cosine similarity
        q_local = F.normalize(q_local, p=2, dim=-1)
        s_local = F.normalize(s_local, p=2, dim=-1)
        
        # Compute scores for each query
        all_scores = []
        
        for b in range(B):
            batch_scores = []
            for q in range(NQ):
                q_desc = q_local[b, q]  # (h*w, D)
                class_scores = []
                
                for c in range(Way):
                    s_desc = s_local[b, c]  # (Shot*h*w, D)
                    
                    # Compute cosine similarity: (h*w, Shot*h*w)
                    sim = torch.mm(q_desc, s_desc.t())  # (h*w, Shot*h*w)
                    
                    # For each query descriptor, find top-k similarities
                    # Then sum over all query descriptors
                    topk_sim, _ = sim.topk(self.k, dim=1)  # (h*w, k)
                    class_score = topk_sim.sum()  # Aggregate all local matches
                    class_scores.append(class_score)
                
                batch_scores.append(torch.stack(class_scores))  # (Way,)
            
            all_scores.append(torch.stack(batch_scores))  # (NQ, Way)
        
        scores = torch.stack(all_scores)  # (B, NQ, Way)
        scores = scores.view(-1, Way)  # (B*NQ, Way)
        
        return scores


class DN4Fast(nn.Module):
    """Optimized DN4 using batched operations.
    
    More memory efficient and GPU-friendly.
    """
    
    def __init__(self, k_neighbors=3, device='cuda'):
        super(DN4Fast, self).__init__()
        
        self.encoder = Conv64F_Encoder()
        self.k = k_neighbors
        
        self.to(device)
    
    def forward(self, query, support):
        """Vectorized DN4 forward pass.
        
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
        # Query: (B, NQ, h*w, D)
        q_local = q_feat.view(B, NQ, D, -1).permute(0, 1, 3, 2)
        q_local = F.normalize(q_local, p=2, dim=-1)
        
        # Support: (B, Way, Shot*h*w, D)
        s_local = s_feat.view(B, Way, Shot, D, -1).permute(0, 1, 2, 4, 3)
        s_local = s_local.reshape(B, Way, -1, D)
        s_local = F.normalize(s_local, p=2, dim=-1)
        
        # Compute similarity between all query and support descriptors
        # For memory efficiency, process one query at a time per batch
        scores_list = []
        
        for b in range(B):
            q_b = q_local[b]  # (NQ, h*w, D)
            s_b = s_local[b]  # (Way, Shot*h*w, D)
            
            # Each query vs each class
            query_scores = []
            for q_idx in range(NQ):
                q_desc = q_b[q_idx]  # (h*w, D)
                class_scores = []
                
                for c_idx in range(Way):
                    s_desc = s_b[c_idx]  # (Shot*h*w, D)
                    
                    # Cosine similarity
                    sim = torch.mm(q_desc, s_desc.t())  # (h*w, Shot*h*w)
                    
                    # Top-k per query descriptor
                    k_actual = min(self.k, sim.size(1))
                    topk_sim, _ = sim.topk(k_actual, dim=1)
                    score = topk_sim.sum()
                    class_scores.append(score)
                
                query_scores.append(torch.stack(class_scores))
            
            scores_list.append(torch.stack(query_scores))
        
        scores = torch.stack(scores_list).view(-1, Way)
        return scores
