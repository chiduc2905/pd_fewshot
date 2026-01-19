"""Siamese Network for few-shot learning.

Paper: "Siamese Neural Networks for One-shot Image Recognition" (Koch et al., ICML-W 2015)

Original Architecture (for Omniglot 105x105):
- Conv1: 64 filters, 10x10, ReLU → MaxPool 2x2
- Conv2: 128 filters, 7x7, ReLU → MaxPool 2x2
- Conv3: 128 filters, 4x4, ReLU → MaxPool 2x2
- Conv4: 256 filters, 4x4, ReLU (no pooling)
- Flatten → FC 4096 with sigmoid
- L1 distance between embeddings
- Weighted L1 → Sigmoid → binary similarity score

Adapted for 128x128 RGB images while preserving paper architecture ratios.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SiameseEncoder(nn.Module):
    """Koch et al. 2015 paper-accurate Siamese encoder.
    
    Architecture follows Figure 3 of the original paper:
    - 4 convolutional layers with increasing channels (64→128→128→256)
    - ReLU activations (paper uses ReLU for conv layers)
    - MaxPool after first 3 conv layers
    - Flatten → FC → 4096 features with sigmoid
    
    Adapted for 128x128 RGB inputs (vs original 105x105 grayscale).
    """
    
    def __init__(self, in_channels=3, feat_dim=4096):
        super(SiameseEncoder, self).__init__()
        
        self.feat_dim = feat_dim
        
        # Paper architecture (adapted kernel sizes for 128x128 input):
        # Original uses 10x10, 7x7, 4x4, 4x4 for 105x105
        # We use 10x10, 7x7, 4x4, 4x4 which works for 128x128 too
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=10, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # 128 -> (128-10+1)=119 -> 119/2=59
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=7, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # 59 -> (59-7+1)=53 -> 53/2=26
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=4, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # 26 -> (26-4+1)=23 -> 23/2=11
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=0),
            nn.ReLU(inplace=True)
            # No pooling after last conv (as per paper)
        )
        # 11 -> (11-4+1)=8
        
        # Adaptive pooling to handle varying spatial sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        # Output: 256 * 4 * 4 = 4096
        
        # FC layer with sigmoid (as per paper)
        self.fc = nn.Sequential(
            nn.Linear(256 * 4 * 4, feat_dim),
            nn.Sigmoid()  # Paper uses sigmoid in FC layer
        )
        
        # Initialize weights (paper uses normal init with specific std)
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights as per paper recommendations."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Paper: normal init with mean=0, std specific to layer
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.2)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Extract feature embedding.
        
        Args:
            x: (N, C, H, W) images
        Returns:
            (N, feat_dim) embeddings
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.adaptive_pool(x)  # Ensure consistent 4x4 output
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x


class SiameseNet(nn.Module):
    """Koch et al. 2015 paper-accurate Siamese Network.
    
    For few-shot episodic training:
    - For each query, computes pairwise similarity to each support sample
    - Average similarities per class to get class scores
    - Use CrossEntropyLoss on scores (adapted from binary verification)
    
    Key differences from incorrect previous implementation:
    - Proper encoder with 64→128→128→256 channels (not 4x64)
    - Feature dimension 4096 (not 64)
    - Sigmoid activations in FC (not ReLU everywhere)
    - L1 distance + weighted FC for similarity (paper-accurate)
    """
    
    def __init__(self, feat_dim=4096, device='cuda'):
        """Initialize Siamese Network.
        
        Args:
            feat_dim: Feature dimension after encoder (paper uses 4096)
            device: Device to use
        """
        super(SiameseNet, self).__init__()
        
        self.feat_dim = feat_dim
        
        # Paper-accurate encoder
        self.encoder = SiameseEncoder(in_channels=3, feat_dim=feat_dim)
        
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
    """Optimized Siamese Network with paper-accurate architecture.
    
    Uses prototype-based comparison for efficiency while maintaining
    the paper's encoder architecture.
    """
    
    def __init__(self, feat_dim=4096, device='cuda'):
        super(SiameseNetFast, self).__init__()
        
        self.feat_dim = feat_dim
        self.encoder = SiameseEncoder(in_channels=3, feat_dim=feat_dim)
        
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
