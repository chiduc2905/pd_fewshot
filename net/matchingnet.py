"""Matching Networks for One Shot Learning."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from net.encoders.matchingnet_encoder import MatchingNetEncoder


class AttentionLSTM(nn.Module):
    """
    Attention LSTM for processing query with attention over support set.
    
    From Matching Networks paper (Vinyals et al., NIPS 2016)
    """
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(AttentionLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size + hidden_size, hidden_size, num_layers, batch_first=True)
        
    def forward(self, x, support_embeddings):
        """
        Args:
            x: (B, feature_dim) query embedding
            support_embeddings: (B, Way*Shot, feature_dim) support embeddings
        Returns:
            (B, hidden_size) attended query representation
        """
        B = x.size(0)
        h = torch.zeros(self.num_layers, B, self.hidden_size).to(x.device)
        c = torch.zeros(self.num_layers, B, self.hidden_size).to(x.device)
        
        # Read support set with attention
        for step in range(support_embeddings.size(1)):
            # Attention over support
            # Compute attention scores
            attention_scores = torch.bmm(
                h[-1].unsqueeze(1),  # (B, 1, hidden_size)
                support_embeddings.transpose(1, 2)  # (B, feature_dim, Way*Shot)
            ).squeeze(1)  # (B, Way*Shot)
            
            attention_weights = F.softmax(attention_scores, dim=1)  # (B, Way*Shot)
            
            # Attended support representation
            r = torch.bmm(
                attention_weights.unsqueeze(1),  # (B, 1, Way*Shot)
                support_embeddings  # (B, Way*Shot, feature_dim)
            ).squeeze(1)  # (B, feature_dim)
            
            # LSTM step: input is concat(x, r)
            lstm_input = torch.cat([x, r], dim=1).unsqueeze(1)  # (B, 1, input_size+hidden_size)
            _, (h, c) = self.lstm(lstm_input, (h, c))
        
        return h[-1]  # (B, hidden_size)


class MatchingNet(nn.Module):
    """
    Matching Networks for One Shot Learning (Full version with LSTM).
    
    Paper: Vinyals et al. "Matching Networks for One Shot Learning" (NIPS 2016)
    
    Key components:
    1. CNN encoder (shared for support and query)
    2. Bidirectional LSTM for full context embeddings of support set (g)
    3. Attention LSTM for query encoding with attention over support (f)
    4. Cosine similarity-based attention kernel
    5. Weighted prediction over support labels
    
    This implementation follows the original gitabcworld implementation.
    """
    
    def __init__(self, backbone='conv64f', device='cuda', image_size=128):
        """
        Args:
            backbone: 'conv64f' (original, dynamic dim) or 'resnet12' (512->4096 projection)
            device: Device to use
            image_size: Input image size (default: 128)
        """
        super(MatchingNet, self).__init__()
        
        self.backbone = backbone
        
        # Select backbone and determine feature dimension
        if backbone == 'resnet12':
            from net.encoders.resnet12_encoder import ResNet12Encoder
            self.encoder = ResNet12Encoder()
            encoder_dim = 512  # ResNet12 uses GAP -> fixed 512 dim
            # Project to same dimension as conv64f for fair comparison
            spatial_size = image_size // 16
            feat_dim = 64 * spatial_size * spatial_size  # 4096 for 128x128
            self.projection = nn.Linear(encoder_dim, feat_dim)
        else:  # conv64f - original implementation
            self.encoder = MatchingNetEncoder()
            self.projection = None
            # Original: flatten directly, feat_dim = 64 * (H/16)^2
            spatial_size = image_size // 16
            feat_dim = 64 * spatial_size * spatial_size
            # 28x28 -> 64, 84x84 -> 1600, 128x128 -> 4096
        
        self.feat_dim = feat_dim
        
        # Full contextual embeddings (dynamic dimensions based on backbone)
        self.support_lstm = nn.LSTM(feat_dim, feat_dim // 2, batch_first=True, bidirectional=True)
        self.query_attention_lstm = AttentionLSTM(input_size=feat_dim, hidden_size=feat_dim)
        
        # Paper uses PyTorch default init
        self.to(device)
    
    def forward(self, query, support):
        """
        Compute matching scores using attention mechanism.
        
        Args:
            query: (B, NQ, C, H, W) query images
            support: (B, Way, Shot, C, H, W) support images
        Returns:
            scores: (B*NQ, Way) matching scores
        """
        B, NQ, C, H, W = query.size()
        B_s, Way, Shot, C_s, H_s, W_s = support.size()
        
        # Flatten and encode
        query_flat = query.view(-1, C, H, W)
        support_flat = support.view(-1, C, H, W)
        
        q_encoded = self.encoder(query_flat)  # (B*NQ, encoder_dim)
        s_encoded = self.encoder(support_flat)  # (B*Way*Shot, encoder_dim)
        
        # Apply projection for resnet12 (to match conv64f dimension)
        if self.projection is not None:
            q_encoded = self.projection(q_encoded)  # (B*NQ, feat_dim)
            s_encoded = self.projection(s_encoded)  # (B*Way*Shot, feat_dim)
        
        # Reshape
        q_encoded = q_encoded.view(B, NQ, -1)  # (B, NQ, feat_dim)
        s_encoded = s_encoded.view(B, Way, Shot, -1)  # (B, Way, Shot, feat_dim)
        
        # Full context embeddings with LSTM (paper-compliant)
        # Support set: bidirectional LSTM (g function)
        s_flat_for_lstm = s_encoded.view(B, Way * Shot, -1)  # (B, Way*Shot, 1024)
        s_context, _ = self.support_lstm(s_flat_for_lstm)  # (B, Way*Shot, 1024)
        s_context = s_context.view(B, Way, Shot, -1)  # (B, Way, Shot, 1024)
        
        # Query: Attention LSTM (f function)
        q_context = []
        for i in range(NQ):
            q_i = q_encoded[:, i, :]  # (B, 1024)
            q_i_context = self.query_attention_lstm(q_i, s_flat_for_lstm)  # (B, 1024)
            q_context.append(q_i_context)
        q_context = torch.stack(q_context, dim=1)  # (B, NQ, 1024)
        
        # Compute attention kernel (cosine similarity)
        # For each query, compare with each support example
        scores_list = []
        
        for b in range(B):
            q_b = q_context[b]  # (NQ, 1024)
            s_b = s_context[b]  # (Way, Shot, 1024)
            
            # Normalize
            q_norm = F.normalize(q_b, p=2, dim=1)  # (NQ, 1024)
            s_norm = F.normalize(s_b.view(Way * Shot, -1), p=2, dim=1)  # (Way*Shot, 1024)
            
            # Cosine similarity
            similarity = torch.mm(q_norm, s_norm.t())  # (NQ, Way*Shot)
            
            # Attention weights
            attention = F.softmax(similarity, dim=1)  # (NQ, Way*Shot)
            
            # Weighted vote: each support example votes for its class
            # Create one-hot labels for support set
            support_labels = torch.arange(Way).repeat_interleave(Shot).to(query.device)  # (Way*Shot,)
            one_hot = F.one_hot(support_labels, num_classes=Way).float()  # (Way*Shot, Way)
            
            # Weighted prediction
            scores_b = torch.mm(attention, one_hot)  # (NQ, Way)
            scores_list.append(scores_b)
        
        scores = torch.cat(scores_list, dim=0)  # (B*NQ, Way)
        
        # Convert probabilities to logits for CrossEntropy loss
        # CrossEntropy expects logits, not probabilities
        # log(probs) converts probs to log-probs which work like logits
        scores = torch.log(scores + 1e-8)  # Add epsilon to avoid log(0)
        
        return scores
