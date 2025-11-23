import torch
import torch.nn as nn

class CosineNet(nn.Module):
    def __init__(self):
        super(CosineNet, self).__init__()
        # Placeholder - Replace with actual pam_mamba architecture
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
        # query: (way, C, H, W)
        # support: (way, shot, C, H, W)
        q_feat = self.encoder(query)  # (way, 128, 1, 1)
        q_feat = q_feat.view(q_feat.size(0), -1)  # (way, 128)
        q_feat = self.fc(q_feat)  # (way, 64)
        
        support_feats = []
        for i, s in enumerate(support):
            # s shape: (shot, C, H, W)
            s_feat = self.encoder(s)  # (shot, 128, 1, 1)
            s_feat = s_feat.view(s_feat.size(0), -1)  # (shot, 128)
            s_feat = self.fc(s_feat)  # (shot, 64)
            support_feats.append(s_feat.mean(0, keepdim=True))  # (1, 64)
        
        support_feats = torch.cat(support_feats, dim=0)  # (way, 64)
        
        # Compute similarity scores
        scores = torch.nn.functional.cosine_similarity(q_feat, support_feats)
        return scores
