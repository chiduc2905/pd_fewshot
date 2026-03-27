"""FEAT: Few-Shot Embedding Adaptation with Transformer."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from net.encoders.smnet_conv64f_encoder import build_resnet12_family_encoder


class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention from the official FEAT implementation."""

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        return torch.bmm(attn, v)


class MultiHeadAttention(nn.Module):
    """Official FEAT multi-head attention block."""

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.5):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0.0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0.0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0.0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        _, len_k, _ = k.size()
        _, len_v, _ = v.size()

        residual = q
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)

        output = self.attention(q, k, v)
        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)
        output = self.dropout(self.fc(output))
        return self.layer_norm(output + residual)


class FEAT(nn.Module):
    """Paper-style FEAT forward path with episodic auxiliary regularization."""

    def __init__(
        self,
        image_size=64,
        temperature=64.0,
        temperature2=64.0,
        aux_balance=0.01,
        fewshot_backbone="resnet12",
        device="cuda",
    ):
        super().__init__()
        self.encoder = build_resnet12_family_encoder(
            image_size=image_size,
            backbone_name=fewshot_backbone,
            pool_output=True,
            variant="fewshot",
            drop_rate=0.1,
        )
        self.feat_dim = self.encoder.out_dim
        self.slf_attn = MultiHeadAttention(1, self.feat_dim, self.feat_dim, self.feat_dim, dropout=0.5)
        self.temperature = float(temperature)
        self.temperature2 = float(temperature2)
        self.aux_balance = float(aux_balance)
        self.to(device)

    def encode(self, x):
        return self.encoder(x)

    def _episode_aux_loss(self, support_emb, query_emb):
        batch_size, way_num, shot_num, feat_dim = support_emb.shape
        num_query = query_emb.size(1)
        query_per_class = max(1, num_query // way_num)

        aux_losses = []
        for batch_idx in range(batch_size):
            query_by_class = query_emb[batch_idx].view(way_num, query_per_class, feat_dim)
            class_sequences = []
            class_targets = []
            for class_idx in range(way_num):
                seq = torch.cat([support_emb[batch_idx, class_idx], query_by_class[class_idx]], dim=0)
                class_sequences.append(seq)
                class_targets.append(
                    torch.full(
                        (seq.size(0),),
                        class_idx,
                        device=query_emb.device,
                        dtype=torch.long,
                    )
                )

            aux_task = torch.stack(class_sequences, dim=0)
            aux_emb = self.slf_attn(aux_task, aux_task, aux_task)
            aux_centers = aux_emb.mean(dim=1)
            aux_samples = aux_emb.view(-1, feat_dim)
            aux_targets = torch.cat(class_targets, dim=0)
            aux_logits = -torch.sum(
                (aux_samples.unsqueeze(1) - aux_centers.unsqueeze(0)) ** 2,
                dim=2,
            ) / self.temperature2
            aux_losses.append(F.cross_entropy(aux_logits, aux_targets))

        return torch.stack(aux_losses).mean()

    def forward(self, query, support, query_targets=None, support_targets=None):
        del query_targets, support_targets
        batch_size, num_query, channels, height, width = query.size()
        _, way_num, shot_num, _, _, _ = support.size()

        query_emb = self.encode(query.view(-1, channels, height, width)).view(batch_size, num_query, -1)
        support_emb = self.encode(support.view(-1, channels, height, width)).view(batch_size, way_num, shot_num, -1)

        prototypes = support_emb.mean(dim=2)
        prototypes = self.slf_attn(prototypes, prototypes, prototypes)

        expanded_proto = prototypes.unsqueeze(1).expand(batch_size, num_query, way_num, self.feat_dim)
        expanded_proto = expanded_proto.contiguous().view(batch_size * num_query, way_num, self.feat_dim)
        expanded_query = query_emb.contiguous().view(batch_size * num_query, -1).unsqueeze(1)
        logits = -torch.sum((expanded_proto - expanded_query) ** 2, dim=2) / self.temperature

        if self.training and self.aux_balance > 0:
            aux_loss = self._episode_aux_loss(support_emb, query_emb)
            return {"logits": logits, "aux_loss": self.aux_balance * aux_loss}

        return logits
