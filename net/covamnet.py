import torch
import torch.nn as nn
import functools
from net.encoder import Conv64F_Encoder, get_norm_layer
from net.cova_block import CovaBlock
from net.utils import init_weights

class CovarianceNet(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d, num_classes=5, init_type='normal', use_gpu=True, input_size=64):
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
        
        self.covariance = CovaBlock()
        
        # Determine feature map dimensions
        # Encoder has 2 max pooling layers (stride 2), so downsample factor is 4
        self.feature_h = input_size // 4
        self.feature_w = input_size // 4
        kernel_size = self.feature_h * self.feature_w
        
        # Restore Classifier as per reference
        self.classifier = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Dropout(),
            nn.Conv1d(1, 1, kernel_size=kernel_size, stride=kernel_size, bias=use_bias),
        )
        
        init_weights(self, init_type=init_type)
        if use_gpu and torch.cuda.is_available():
            self.cuda()

    def forward(self, query, support):
        """
        Args:
            query: (B, NQ, C, H, W) 
            support: (B, Way, Shot, C, H, W)
        Returns:
            scores: (B * NQ, Way)
        """
        B, NQ, C, H, W = query.shape
        B_s, Way, Shot, C_s, H_s, W_s = support.shape
        
        scores_list = []
        
        for b in range(B):
            # Query for this batch: (NQ, C, H, W)
            q_b = query[b] 
            
            # Support for this batch: (Way, Shot, C, H, W)
            s_b = support[b]
            
            # Extract features for Query
            q_feat = self.features(q_b) # (NQ, 64, h, w)
            
            # Extract features for Support
            s_feats = []
            for w in range(Way):
                # s_b[w] is (Shot, C, H, W)
                sf = self.features(s_b[w]) # (Shot, 64, h, w)
                s_feats.append(sf)
            
            # Calculate similarity matrix using CovaBlock
            # returns (NQ, Way * h * w) or (NQ, 1, Way * h * w) depending on CovaBlock details
            # In our updated CovaBlock, it returns (NQ, Way * h * w) if using cat(0) on view(1,-1) rows?
            # Let's check updated CovaBlock:
            # Cova_Sim.append(mea_sim.view(1, -1)) -> cat(0) results in (NQ, Way*h*w)
            
            x1 = self.covariance(q_feat, s_feats) 
            
            # Apply classifier
            # Reference: x1 = self.classifier1(x1.view(x1.size(0), 1, -1))
            # Reshape x1 to (NQ, 1, Way*h*w) for Conv1d
            x1 = self.classifier(x1.view(x1.size(0), 1, -1))
            
            # Reference: output = x1.squeeze(1)
            output = x1.squeeze(1) # (NQ, Way)
            
            scores_list.append(output)
            
        scores = torch.cat(scores_list, dim=0) # (B*NQ, Way)
        
        return scores

# Expose CovaMNet as CovarianceNet
CovaMNet = CovarianceNet
