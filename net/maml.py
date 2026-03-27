"""MAML-style episodic learner for few-shot classification."""

from collections import OrderedDict
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F


class MAMLNet(nn.Module):
    """Conv4 MAML with paper-style defaults for few-shot classification."""

    def __init__(self, image_size=64, inner_lr=0.01, inner_steps=5, first_order=True, max_way_num=32, device="cuda"):
        super().__init__()
        hidden_dim = 32
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.first_order = first_order
        self.max_way_num = max_way_num

        self.conv1 = nn.Conv2d(3, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.BatchNorm2d(hidden_dim, affine=True, track_running_stats=False)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(hidden_dim, affine=True, track_running_stats=False)
        self.conv3 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.norm3 = nn.BatchNorm2d(hidden_dim, affine=True, track_running_stats=False)
        self.conv4 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.norm4 = nn.BatchNorm2d(hidden_dim, affine=True, track_running_stats=False)
        spatial_dim = max(1, image_size // 16)
        self.classifier = nn.Linear(hidden_dim * spatial_dim * spatial_dim, max_way_num)

        self._reset_parameters()
        self.to(device)

    def _reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def _conv_block(self, x, params, idx):
        x = F.conv2d(x, params[f"conv{idx}.weight"], params[f"conv{idx}.bias"], stride=1, padding=1)
        x = F.batch_norm(
            x,
            running_mean=None,
            running_var=None,
            weight=params[f"norm{idx}.weight"],
            bias=params[f"norm{idx}.bias"],
            training=True,
            momentum=1.0,
        )
        x = F.relu(x, inplace=True)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        return x

    def _functional_features(self, x, params):
        x = self._conv_block(x, params, 1)
        x = self._conv_block(x, params, 2)
        x = self._conv_block(x, params, 3)
        x = self._conv_block(x, params, 4)
        return x.view(x.size(0), -1)

    def _functional_logits(self, x, params, way_num):
        features = self._functional_features(x, params)
        weight = params["classifier.weight"][:way_num]
        bias = params["classifier.bias"][:way_num]
        return F.linear(features, weight, bias)

    @staticmethod
    def _inner_objective(logits, targets, way_num, label_smoothing=0.0):
        """Match the authors' support-set loss scaling for MiniImageNet MAML."""
        shot_num = max(1, targets.numel() // max(1, way_num))
        per_example = F.cross_entropy(
            logits,
            targets,
            reduction="none",
            label_smoothing=label_smoothing,
        )
        return per_example.sum() / shot_num

    def _adapt_episode(self, support_images, support_targets, way_num, label_smoothing=0.0):
        fast_params = OrderedDict(self.named_parameters())
        for _ in range(self.inner_steps):
            support_logits = self._functional_logits(support_images, fast_params, way_num)
            support_loss = self._inner_objective(
                support_logits,
                support_targets,
                way_num,
                label_smoothing=label_smoothing,
            )
            grads = torch.autograd.grad(
                support_loss,
                tuple(fast_params.values()),
                create_graph=not self.first_order,
            )
            fast_params = OrderedDict(
                (name, param - self.inner_lr * grad)
                for (name, param), grad in zip(fast_params.items(), grads)
            )
        return fast_params

    def extract_features(self, x):
        params = OrderedDict(self.named_parameters())
        return self._functional_features(x, params)

    def forward(self, query, support, label_smoothing=0.0):
        grad_ctx = nullcontext() if torch.is_grad_enabled() else torch.enable_grad()
        with grad_ctx:
            batch_size, num_query, channels, height, width = query.size()
            _, way_num, shot_num, _, _, _ = support.size()

            if way_num > self.max_way_num:
                raise ValueError(f"way_num={way_num} exceeds MAML max_way_num={self.max_way_num}")

            outputs = []
            support_targets = torch.arange(way_num, device=query.device).repeat_interleave(shot_num)

            for batch_idx in range(batch_size):
                support_images = support[batch_idx].reshape(way_num * shot_num, channels, height, width)
                fast_params = self._adapt_episode(support_images, support_targets, way_num, label_smoothing)
                query_images = query[batch_idx].reshape(num_query, channels, height, width)
                outputs.append(self._functional_logits(query_images, fast_params, way_num))

            return torch.cat(outputs, dim=0)
