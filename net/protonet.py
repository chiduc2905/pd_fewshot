import torch
import torch.nn as nn
import torch.nn.functional as F

class ProtoNet(nn.Module):
    def __init__(self):
        super(ProtoNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(128, 64)
    
    def forward(self, query, support):
        # query: (way, C, H, W) -> (way, 1, C, H, W) for broadcasting if needed, but here we process batch
        # support: (way, shot, C, H, W)
        
        # Extract query features
        # query shape is actually (batch_size, C, H, W) in training loop usually, 
        # but based on train_1shot.py it seems to be (way, C, H, W) for 1-shot 1-query?
        # Let's check train_1shot.py again. 
        # In train_1shot.py:
        # q = query_images.permute(1, 0, 2, 3, 4) -> (batch, way, C, H, W) if batch_size > 1?
        # Wait, batch_size is 1.
        # query_images: (batch, way, query_num, C, H, W)
        # q = query_images.permute(1, 0, 2, 3, 4) -> (way, batch, query_num, C, H, W)
        # q[i] -> (batch, query_num, C, H, W) -> (1, 1, C, H, W) -> squeeze -> (C, H, W)
        # Actually in train_1shot.py:
        # q = query_images.permute(1, 0, 2, 3, 4) -> (way, batch, query_num, C, H, W)
        # for i in range(len(q)):
        #   scores = net(q[i], s)
        # q[i] is (batch, query_num, C, H, W). batch=1, query_num=1.
        # So q[i] is (1, 1, C, H, W).
        # In CosineNet: q_feat = self.encoder(query) -> query must be (N, C, H, W).
        # So q[i] should be reshaped to (N, C, H, W).
        
        # Let's assume input query is (N_query, C, H, W) and support is (N_way, N_shot, C, H, W)
        # In train_1shot.py, q[i] is passed. If batch=1, query=1, it is (1, 1, C, H, W).
        # The original code might be relying on implicit squeezing or the encoder handles 5D input?
        # CosineNet encoder: nn.Conv2d(3, 64...). Expects 4D.
        # So q[i] must be 4D. (1, 1, 3, 224, 224) -> fail.
        # Let's look at train_1shot.py again.
        # q = query_images.permute(1, 0, 2, 3, 4)
        # query_images from DataLoader: (batch, way, query_num, C, H, W)
        # permute(1, 0, 2, 3, 4) -> (way, batch, query_num, C, H, W)
        # q[i] -> (batch, query_num, C, H, W).
        # If batch=1, query_num=1 -> (1, 1, C, H, W).
        # If the original code works, maybe query_images was squeezed somewhere?
        # Or maybe batch_size is not 1? args.batch_size default is 1.
        
        # Wait, in train_1shot.py:
        # q = query_images.permute(1, 0, 2, 3, 4)
        # for i in range(len(q)):
        #   scores = net(q[i], s)
        # If q has shape (way, batch, query_num, C, H, W), then len(q) is way.
        # So we iterate over way? That seems wrong for standard few-shot.
        # Usually we iterate over episodes (batches).
        # Ah, the dataloader yields (query_images, query_targets, support_images, support_targets).
        # In standard few-shot, one batch is one episode.
        # query_images: (batch, way, query_num, C, H, W)
        # support_images: (batch, way, shot_num, C, H, W)
        
        # In train_1shot.py:
        # q = query_images.permute(1, 0, 2, 3, 4) -> (way, batch, query_num, C, H, W)
        # This permutation seems to mix 'way' and 'batch'.
        # If batch=1, it's (way, 1, query_num, C, H, W).
        # Loop over 'way'?
        # for i in range(len(q)):
        #   scores = net(q[i], s)
        #   target = targets[i]
        # This implies we are classifying each class's query images against the support set?
        # But 'targets' is also permuted: targets = query_targets.permute(1, 0) -> (way, batch).
        # So for each class 'i' (from 0 to way-1), we take its query images q[i].
        # And we classify them.
        
        # This is a bit unusual structure but I must stick to it to be compatible.
        # So input 'query' to forward is (batch, query_num, C, H, W).
        # Input 'support' to forward is (way, batch, shot_num, C, H, W) - wait, s is also permuted.
        # s = support_images.permute(1, 0, 2, 3, 4) -> (way, batch, shot_num, C, H, W).
        # In CosineNet:
        # q_feat = self.encoder(query)
        # If query is (1, 1, C, H, W), encoder will fail.
        # Unless... the dataloader produces something else.
        # Let's assume the user's code works and I should handle inputs similarly.
        # I will add .squeeze(0) if needed or view(-1, C, H, W).
        
        # Handling dimensions:
        # query: (batch, query_num, C, H, W)
        # support: (way, batch, shot_num, C, H, W)
        # Since batch=1 in default, let's flatten batch and query_num/shot_num.
        
        q_shape = query.shape
        s_shape = support.shape
        
        # Flatten query to (N_query, C, H, W)
        query = query.view(-1, q_shape[-3], q_shape[-2], q_shape[-1])
        
        # Flatten support to (N_support, C, H, W)
        # support is (way, batch, shot, C, H, W) passed as 's' in train loop?
        # In train_1shot.py: s = support_images.permute(1, 0, 2, 3, 4)
        # passed to net(q[i], s).
        # So 's' in forward is (way, batch, shot, C, H, W).
        
        # We need to compute prototypes for each way.
        # Reshape support to (way, batch * shot, C, H, W) -> then encode -> average.
        
        num_ways = s_shape[0]
        # Reshape to (num_ways * rest, C, H, W) to pass through encoder
        support_reshaped = support.view(-1, s_shape[-3], s_shape[-2], s_shape[-1])
        
        # Encode
        q_feat = self.encoder(query) # (N_query, 128, 1, 1)
        q_feat = q_feat.view(q_feat.size(0), -1)
        q_feat = self.fc(q_feat) # (N_query, 64)
        
        s_feat = self.encoder(support_reshaped) # (Total_support, 128, 1, 1)
        s_feat = s_feat.view(s_feat.size(0), -1)
        s_feat = self.fc(s_feat) # (Total_support, 64)
        
        # Reshape support features back to (way, batch*shot, 64) to compute prototypes
        # Note: 'batch' here is from the dataloader batch size, which is 1.
        # So effectively (way, shot, 64).
        s_feat = s_feat.view(num_ways, -1, s_feat.size(-1))
        
        # Compute prototypes: mean over shots
        prototypes = s_feat.mean(dim=1) # (way, 64)
        
        # Compute distances (Euclidean)
        # q_feat: (N_query, 64)
        # prototypes: (way, 64)
        # dist: (N_query, way)
        
        dists = torch.cdist(q_feat, prototypes)
        
        # Output should be scores (logits). For ProtoNet, usually -distance.
        scores = -dists
        
        # If the training loop expects (way,) output for single query?
        # In train_1shot.py: 
        # scores = net(q[i], s)
        # target = targets[i]
        # loss = loss_fn(scores, target)
        # ContrastiveLoss expects:
        # if output.dim() == 1: ...
        # else: ...
        
        # If q[i] has 1 sample, scores will be (1, way).
        # If q[i] has multiple, scores will be (N, way).
        # The code seems to handle both.
        
        return scores

