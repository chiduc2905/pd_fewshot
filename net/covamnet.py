import torch
import torch.nn as nn
import functools
from net.encoder import Conv64F_Encoder, get_norm_layer
from net.cova_block import CovaBlock
from net.utils import init_weights

class CovarianceNet(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d, num_classes=5, init_type='normal', use_gpu=True, input_size=224):
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
        
        # Determine kernel size for classifier based on input size
        # Encoder has 2 max pooling layers (stride 2), so downsample factor is 4
        self.feature_h = input_size // 4
        self.feature_w = input_size // 4
        kernel_size = self.feature_h * self.feature_w
        
        self.classifier = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Dropout(),
            nn.Conv1d(1, 1, kernel_size=kernel_size, stride=kernel_size, bias=use_bias),
        )
        
        init_weights(self, init_type=init_type)
        if use_gpu and torch.cuda.is_available():
            self.cuda()

    def forward(self, query, support):
        # query: (batch, query_num, C, H, W) 
        # support: (way, batch, shot, C, H, W)
        
        q_shape = query.shape
        s_shape = support.shape
        
        # Flatten query to (N_query, C, H, W)
        # N_query = batch * query_num
        input1 = query.view(-1, q_shape[-3], q_shape[-2], q_shape[-1])
        
        # Flatten support to list of tensors per class
        # support has shape (way, batch, shot, C, H, W)
        input2 = []
        num_ways = s_shape[0]
        
        for i in range(num_ways):
            # Flatten batch and shot dimensions for each way
            s_way = support[i].view(-1, s_shape[-3], s_shape[-2], s_shape[-1])
            input2.append(s_way)

        # extract features of input1--query image
        q = self.features(input1)
        
        # extract features of input2--support set
        S = []
        for i in range(len(input2)):
            S.append(self.features(input2[i]))
            
        x = self.covariance(q, S) # get Batch*1*(h*w*num_classes)
        x = self.classifier(x)    # get Batch*1*num_classes
        x = x.squeeze(1)          # get Batch*num_classes
        
        return x

# Expose CovaMNet as CovarianceNet for compatibility if needed
CovaMNet = CovarianceNet
