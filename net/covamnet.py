"""Covariance Metric Network for few-shot learning."""
import torch
import torch.nn as nn
import functools
from net.encoder import Conv64F_Encoder, get_norm_layer
from net.cova_block import CovaBlock
from net.utils import init_weights


class CovarianceNet(nn.Module):
    """Few-shot classifier using covariance-based similarity."""
    
    def __init__(self, norm_layer=nn.BatchNorm2d, num_classes=5, init_type='normal', device='cuda', input_size=64):
        super(CovarianceNet, self).__init__()

        if type(norm_layer) == str:
             norm_layer = get_norm_layer(norm_layer)
        elif norm_layer is None:
             norm_layer = nn.BatchNorm2d

        # Check norm_layer to decide bias
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.features = Conv64F_Encoder(norm_layer=norm_layer)
        self.encoder = self.features  # Alias for t-SNE feature extraction
        self.covariance = CovaBlock()

        # Feature map size: input_size / 4 (2 max-pool layers)
        self.feature_h = input_size // 4
        self.feature_w = input_size // 4
        kernel_size = self.feature_h * self.feature_w

        # Learnable classifier
        self.classifier = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Dropout(),
            nn.Conv1d(1, 1, kernel_size=kernel_size, stride=kernel_size, bias=use_bias),
        )
        
        init_weights(self, init_type=init_type)
        self.to(device)

    def forward(self, query, support):
        """Compute covariance-based similarity scores.
        
        Args:
            query: (B, NQ, C, H, W) query images
            support: (B, Way, Shot, C, H, W) support images
        Returns:
            scores: (B*NQ, Way) similarity scores
        """
        B, NQ, C, H, W = query.shape
        B_s, Way, Shot, C_s, H_s, W_s = support.shape
        
        scores_list = []
        
        for b in range(B):
            q_b = query[b]
            s_b = support[b]
            
            # Extract features
            q_feat = self.features(q_b)
            
            s_feats = []
            for w in range(Way):
                sf = self.features(s_b[w])
                s_feats.append(sf)
            
            # Compute covariance similarity
            x1 = self.covariance(q_feat, s_feats)

            # Apply classifier
            x1 = self.classifier(x1.view(x1.size(0), 1, -1))
            output = x1.squeeze(1)
            
            scores_list.append(output)
            
        scores = torch.cat(scores_list, dim=0) # (B*NQ, Way)
        
        return scores

# Expose CovaMNet as CovarianceNet
CovaMNet = CovarianceNet
