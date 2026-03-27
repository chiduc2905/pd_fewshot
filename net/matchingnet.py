"""Matching Networks with full context embeddings."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from net.encoders.matchingnet_encoder import MatchingNetEncoder


class FullyContextualEmbedding(nn.Module):
    """Attention LSTM query encoder from the original Matching Networks paper."""

    def __init__(self, feat_dim):
        super().__init__()
        self.lstm_cell = nn.LSTMCell(feat_dim * 2, feat_dim)
        self.softmax = nn.Softmax(dim=1)
        self.feat_dim = feat_dim

    def forward(self, query, support_context):
        if query.dim() == 1:
            query = query.unsqueeze(0)
        hidden = query
        cell = torch.zeros_like(query)
        support_t = support_context.transpose(0, 1)
        steps = support_context.size(0)

        for _ in range(steps):
            attn_logits = hidden.mm(support_t)
            attn = self.softmax(attn_logits)
            readout = attn.mm(support_context)
            lstm_input = torch.cat([query, readout], dim=1)
            hidden, cell = self.lstm_cell(lstm_input, (hidden, cell))
            hidden = hidden + query

        return hidden


class MatchingNet(nn.Module):
    """Paper-style Matching Networks with BiLSTM support encoding and FCE query encoding."""

    def __init__(self, image_size=64, device="cuda"):
        super().__init__()
        self.encoder = MatchingNetEncoder(image_size=image_size)
        self.feat_dim = self.encoder.out_dim
        self.support_encoder = nn.LSTM(
            self.feat_dim,
            self.feat_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.query_encoder = FullyContextualEmbedding(self.feat_dim)
        self.scale_factor = 100.0
        self.to(device)

    def encode_support_set(self, support_embeddings):
        out = self.support_encoder(support_embeddings.unsqueeze(0))[0].squeeze(0)
        support_context = support_embeddings + out[:, : self.feat_dim] + out[:, self.feat_dim :]
        support_norm = F.normalize(support_context, p=2, dim=1)
        return support_context, support_norm

    def forward(self, query, support):
        batch_size, num_query, channels, height, width = query.size()
        _, way_num, shot_num, _, _, _ = support.size()

        query_flat = query.view(-1, channels, height, width)
        support_flat = support.view(-1, channels, height, width)

        query_embeddings = self.encoder(query_flat).view(batch_size, num_query, -1)
        support_embeddings = self.encoder(support_flat).view(batch_size, way_num * shot_num, -1)

        scores = []
        for batch_idx in range(batch_size):
            support_context, support_norm = self.encode_support_set(support_embeddings[batch_idx])
            adapted_queries = []
            for query_idx in range(num_query):
                adapted_queries.append(self.query_encoder(query_embeddings[batch_idx, query_idx], support_context).squeeze(0))
            adapted_queries = torch.stack(adapted_queries, dim=0)
            query_norm = F.normalize(adapted_queries, p=2, dim=1)

            similarities = F.relu(query_norm.mm(support_norm.t())) * self.scale_factor
            attention = F.softmax(similarities, dim=1)
            support_labels = torch.arange(way_num, device=query.device).repeat_interleave(shot_num)
            support_one_hot = F.one_hot(support_labels, num_classes=way_num).float()
            probs = attention.mm(support_one_hot)
            scores.append(torch.log(probs + 1e-6))

        return torch.cat(scores, dim=0)
