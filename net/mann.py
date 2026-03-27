"""Paper-style MANN with external memory adapted to support/query episodes."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from net.encoders.protonet_encoder import Conv64F_Paper_Encoder


def _batched_cosine_similarity(keys: torch.Tensor, memory: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Cosine similarity between `(B, R, D)` keys and `(B, N, D)` memory slots."""

    keys = F.normalize(keys, p=2, dim=-1, eps=eps)
    memory = F.normalize(memory, p=2, dim=-1, eps=eps)
    return torch.einsum("brd,bnd->brn", keys, memory)


class MANNNet(nn.Module):
    """External-memory MANN following the Santoro et al. design direction."""

    def __init__(
        self,
        image_size: int = 64,
        way_num: int = 4,
        cell_size: int = 200,
        memory_slots: int = 128,
        memory_dim: int = 40,
        num_reads: int = 4,
        gamma: float = 0.95,
        device: str = "cuda",
    ) -> None:
        super().__init__()
        self.encoder = Conv64F_Paper_Encoder(image_size=image_size)
        self.feat_dim = self.encoder.out_dim
        self.way_num = way_num
        self.cell_size = cell_size
        self.memory_slots = memory_slots
        self.memory_dim = memory_dim
        self.num_reads = num_reads
        self.gamma = gamma
        self.null_label_id = way_num

        controller_input_dim = self.feat_dim + cell_size + num_reads * memory_dim
        self.label_embedding = nn.Embedding(way_num + 1, cell_size)
        self.controller = nn.LSTMCell(controller_input_dim, cell_size)

        self.key_proj = nn.Linear(cell_size, num_reads * memory_dim)
        self.add_proj = nn.Linear(cell_size, num_reads * memory_dim)
        self.sigma_proj = nn.Linear(cell_size, 1)
        self.hidden_classifier = nn.Linear(cell_size, way_num)
        self.read_classifier = nn.Linear(num_reads * memory_dim, way_num)
        self.to(device)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def _initial_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
        return {
            "memory": torch.zeros(batch_size, self.memory_slots, self.memory_dim, device=device, dtype=dtype),
            "cell": torch.zeros(batch_size, self.cell_size, device=device, dtype=dtype),
            "hidden": torch.zeros(batch_size, self.cell_size, device=device, dtype=dtype),
            "readout": torch.zeros(batch_size, self.num_reads * self.memory_dim, device=device, dtype=dtype),
            "read_weight": torch.zeros(batch_size, self.num_reads, self.memory_slots, device=device, dtype=dtype),
            "used_weight": torch.zeros(batch_size, self.memory_slots, device=device, dtype=dtype),
        }

    def _least_used_mask(self, used_weight: torch.Tensor) -> torch.Tensor:
        _, least_used = torch.topk(used_weight, k=self.num_reads, dim=1, largest=False)
        mask = used_weight.new_zeros(used_weight.shape)
        mask.scatter_(1, least_used, 1.0)
        return mask

    def _step(
        self,
        x_t: torch.Tensor,
        prev_label_ids: torch.Tensor,
        state: Dict[str, torch.Tensor],
    ) -> tuple[Dict[str, torch.Tensor], torch.Tensor]:
        label_embed = self.label_embedding(prev_label_ids)
        controller_input = torch.cat([x_t, label_embed, state["readout"]], dim=-1)
        hidden, cell = self.controller(controller_input, (state["hidden"], state["cell"]))

        key = self.key_proj(hidden).view(-1, self.num_reads, self.memory_dim)
        add = torch.tanh(self.add_proj(hidden)).view(-1, self.num_reads, self.memory_dim)
        sigma = torch.sigmoid(self.sigma_proj(hidden)).view(-1, 1, 1)

        least_used_mask = self._least_used_mask(state["used_weight"])
        write_from_read = sigma * state["read_weight"]
        write_from_unused = (1.0 - sigma) * least_used_mask.unsqueeze(1)
        write_weight = write_from_read + write_from_unused

        cleared_memory = state["memory"] * (1.0 - least_used_mask.unsqueeze(-1))
        written_memory = torch.einsum("brn,brd->bnd", write_weight, add)
        memory = cleared_memory + written_memory

        read_logits = _batched_cosine_similarity(key, memory)
        read_weight = F.softmax(read_logits, dim=-1)
        used_weight = self.gamma * state["used_weight"] + read_weight.sum(dim=1) + write_weight.sum(dim=1)
        readout = torch.einsum("brn,bnd->brd", read_weight, memory).reshape(x_t.shape[0], -1)

        logits = self.hidden_classifier(hidden) + self.read_classifier(readout)
        next_state = {
            "memory": memory,
            "cell": cell,
            "hidden": hidden,
            "readout": readout,
            "read_weight": read_weight,
            "used_weight": used_weight,
        }
        return next_state, logits

    def _build_prev_label_sequence(
        self,
        way_num: int,
        shot_num: int,
        num_query: int,
        device: torch.device,
    ) -> torch.Tensor:
        support_labels = torch.arange(way_num, device=device, dtype=torch.long).repeat_interleave(shot_num)
        support_prev = torch.full_like(support_labels, self.null_label_id)
        if support_prev.numel() > 1:
            support_prev[1:] = support_labels[:-1]

        if num_query == 0:
            return support_prev

        query_prev = torch.full((num_query,), self.null_label_id, device=device, dtype=torch.long)
        query_prev[0] = support_labels[-1]
        return torch.cat([support_prev, query_prev], dim=0)

    def forward(self, query: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        batch_size, num_query, channels, height, width = query.size()
        _, way_num, shot_num, _, _, _ = support.size()
        if way_num != self.way_num:
            raise ValueError(f"MANNNet initialized for way_num={self.way_num}, got episode way_num={way_num}")

        support_flat = support.reshape(-1, channels, height, width)
        query_flat = query.reshape(-1, channels, height, width)
        support_feat = self.encoder(support_flat).view(batch_size, way_num * shot_num, -1)
        query_feat = self.encoder(query_flat).view(batch_size, num_query, -1)

        sequence_feat = torch.cat([support_feat, query_feat], dim=1)
        prev_labels = self._build_prev_label_sequence(way_num, shot_num, num_query, query.device)
        prev_labels = prev_labels.unsqueeze(0).expand(batch_size, -1)

        state = self._initial_state(batch_size, device=query.device, dtype=sequence_feat.dtype)
        logits = []
        for step_idx in range(sequence_feat.shape[1]):
            state, step_logits = self._step(sequence_feat[:, step_idx], prev_labels[:, step_idx], state)
            logits.append(step_logits)

        all_logits = torch.stack(logits, dim=1)
        query_logits = all_logits[:, support_feat.shape[1] :, :]
        return query_logits.reshape(batch_size * num_query, way_num)
