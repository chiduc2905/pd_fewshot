"""ADC-HOT: adaptive distribution calibration with hierarchical OT."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression

from net.encoders.smnet_conv64f_encoder import build_resnet12_family_encoder


def _softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.clip(np.sum(exp_x, axis=axis, keepdims=True), a_min=1e-12, a_max=None)


class _CosineClassifier(nn.Module):
    """Lightweight classifier mirroring the cosine head used in the official codebase."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        self.scale = 2.0 if out_dim <= 200 else 10.0
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.normalize(x, p=2, dim=-1)
        weight = F.normalize(self.weight, p=2, dim=-1)
        return self.scale * torch.matmul(x, weight.transpose(0, 1))


class _SinkhornBase(nn.Module):
    """Minimal, device-safe rewrite of the Sinkhorn solvers from the authors' repo."""

    def __init__(self, eps: float, max_iter: int, thresh: float = 1e-1) -> None:
        super().__init__()
        self.eps = float(eps)
        self.max_iter = int(max_iter)
        self.thresh = float(thresh)

    @staticmethod
    def _cosine_cost(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = F.normalize(x, p=2, dim=-1)
        y = F.normalize(y, p=2, dim=-1)
        return 1.0 - torch.matmul(x, y.transpose(-2, -1))

    def _run(self, cost: torch.Tensor, mu: torch.Tensor | None = None, sum_dims=(-2, -1)):
        squeeze = False
        if cost.dim() == 2:
            cost = cost.unsqueeze(0)
            squeeze = True

        batch_size, x_points, y_points = cost.shape
        device = cost.device
        dtype = cost.dtype

        if mu is None:
            mu = torch.full((batch_size, x_points), 1.0 / x_points, device=device, dtype=dtype)
        else:
            mu = mu.to(device=device, dtype=dtype)
            if mu.dim() == 1:
                mu = mu.unsqueeze(0)
            mu = mu / mu.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        nu = torch.full((batch_size, y_points), 1.0 / y_points, device=device, dtype=dtype)
        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)

        for _ in range(self.max_iter):
            prev_u = u
            modified = self._modified_cost(cost, u, v)
            u = self.eps * (torch.log(mu + 1e-8) - torch.logsumexp(modified, dim=-1)) + u
            modified = self._modified_cost(cost, u, v)
            v = self.eps * (torch.log(nu + 1e-8) - torch.logsumexp(modified.transpose(-2, -1), dim=-1)) + v
            if (u - prev_u).abs().mean().item() < self.thresh:
                break

        pi = torch.exp(self._modified_cost(cost, u, v))
        reduced_cost = torch.sum(pi * cost, dim=sum_dims)
        if squeeze:
            reduced_cost = reduced_cost.squeeze(0)
            pi = pi.squeeze(0)
            cost = cost.squeeze(0)
        return reduced_cost, pi, cost

    def _modified_cost(self, cost: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return (-cost + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps


class _SinkhornDistance(_SinkhornBase):
    def forward(self, x: torch.Tensor, y: torch.Tensor):
        return self._run(self._cosine_cost(x, y))


class _SinkhornDistanceAnotherOneToMulti(_SinkhornBase):
    def forward(self, x: torch.Tensor, y: torch.Tensor, mu: torch.Tensor):
        return self._run(self._cosine_cost(x, y), mu=mu, sum_dims=(-2,))


class _SinkhornDistanceGivenCost(_SinkhornBase):
    def forward(self, x: torch.Tensor, y: torch.Tensor, given_cost: torch.Tensor):
        return self._run(given_cost)


class ADCHOTNet(nn.Module):
    """Paper-style ADC-HOT adaptation with dataset-level base statistics and OT calibration.

    The original paper evaluates on fixed pretrained features. In this benchmark repo we keep
    the paper head/calibration logic, while obtaining the base features from the project's own
    supervised backbone pretraining stage.
    """

    def __init__(
        self,
        image_size=64,
        way_num=4,
        beta=0.8,
        alpha=0.21,
        lambd=0.3,
        sample_temperature=0.3,
        transport_eps=0.01,
        transport_max_iter=200,
        samples_per_class=750,
        jitter=1e-5,
        fewshot_backbone="resnet12",
        device="cuda",
    ):
        super().__init__()
        self.encoder = build_resnet12_family_encoder(
            image_size=image_size,
            backbone_name=fewshot_backbone,
            pool_output=True,
            variant="fewshot",
            drop_rate=0.0,
        )
        self.pretrain_head = _CosineClassifier(self.encoder.out_dim, way_num)
        self.way_num = int(way_num)
        self.beta = float(beta)
        self.alpha = float(alpha)
        self.lambd = float(lambd)
        self.sample_temperature = float(sample_temperature)
        self.transport_eps = float(transport_eps)
        self.transport_max_iter = int(transport_max_iter)
        self.samples_per_class = int(samples_per_class)
        self.jitter = float(jitter)
        self._rng = np.random.default_rng(0)

        feat_dim = self.encoder.out_dim
        self.register_buffer("base_means", torch.empty(0, feat_dim))
        self.register_buffer("base_cov", torch.empty(0, feat_dim, feat_dim))
        self.register_buffer("base_feature_bank", torch.empty(0, 0, feat_dim))
        self.register_buffer("base_sample_prob", torch.empty(0, 0))
        self.register_buffer("base_stats_ready", torch.tensor(False, dtype=torch.bool))

        self.sinkhorn = _SinkhornDistance(transport_eps, transport_max_iter)
        self.sinkhorn_inner = _SinkhornDistanceAnotherOneToMulti(transport_eps, transport_max_iter)
        self.sinkhorn_given_cost = _SinkhornDistanceGivenCost(transport_eps, transport_max_iter)
        self.to(device)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def pretrain_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.pretrain_head(self.extract_features(x))

    def pretrain_parameters(self):
        return list(self.encoder.parameters()) + list(self.pretrain_head.parameters())

    def _tukey_transform_tensor(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp_min(1e-6)
        if self.beta == 0.0:
            return torch.log(x)
        return torch.pow(x, self.beta)

    def _tukey_transform_np(self, x: np.ndarray) -> np.ndarray:
        x = np.clip(x, a_min=1e-6, a_max=None)
        if self.beta == 0.0:
            return np.log(x)
        return np.power(x, self.beta)

    def fit_base_statistics(self, images: torch.Tensor, labels: torch.Tensor, batch_size: int = 32) -> None:
        if images.numel() == 0:
            raise ValueError("Cannot fit ADC-HOT base statistics on an empty training split.")

        device = next(self.parameters()).device
        was_training = self.training
        self.eval()

        features = []
        with torch.no_grad():
            for start in range(0, len(images), batch_size):
                batch = images[start : start + batch_size].to(device)
                features.append(self.extract_features(batch).cpu())

        feats_np = torch.cat(features, dim=0).numpy().astype(np.float64)
        feats_np = self._tukey_transform_np(feats_np)
        labels_np = labels.cpu().numpy().astype(np.int64)

        class_features = []
        base_means = []
        base_cov = []
        max_count = 0
        feat_dim = feats_np.shape[1]
        eye = np.eye(feat_dim, dtype=np.float64)

        for class_id in range(self.way_num):
            cls_feat = feats_np[labels_np == class_id]
            if cls_feat.size == 0:
                raise ValueError(f"ADC-HOT requires class {class_id} in the base split, but none were found.")
            class_features.append(cls_feat)
            max_count = max(max_count, cls_feat.shape[0])
            base_means.append(cls_feat.mean(axis=0))
            if cls_feat.shape[0] > 1:
                cov = np.cov(cls_feat.T)
            else:
                cov = np.zeros((feat_dim, feat_dim), dtype=np.float64)
            cov = np.asarray(cov, dtype=np.float64).reshape(feat_dim, feat_dim)
            base_cov.append(cov + self.jitter * eye)

        classifier = LogisticRegression(max_iter=1000)
        classifier.fit(feats_np, labels_np)

        padded_features = np.zeros((self.way_num, max_count, feat_dim), dtype=np.float64)
        sample_prob = np.zeros((self.way_num, max_count), dtype=np.float64)
        for class_id, cls_feat in enumerate(class_features):
            if cls_feat.shape[0] < max_count:
                pad_count = max_count - cls_feat.shape[0]
                cls_feat_padded = np.concatenate([cls_feat, cls_feat[:pad_count]], axis=0)
            else:
                cls_feat_padded = cls_feat
            padded_features[class_id] = cls_feat_padded
            predict_prob = classifier.predict_proba(cls_feat_padded)
            sample_prob[class_id] = _softmax_np(
                predict_prob[:, class_id] / max(self.sample_temperature, 1e-6),
                axis=0,
            )

        self.base_means = torch.from_numpy(np.stack(base_means).astype(np.float32)).to(device)
        self.base_cov = torch.from_numpy(np.stack(base_cov).astype(np.float32)).to(device)
        self.base_feature_bank = torch.from_numpy(padded_features.astype(np.float32)).to(device)
        self.base_sample_prob = torch.from_numpy(sample_prob.astype(np.float32)).to(device)
        self.base_stats_ready = torch.tensor(True, dtype=torch.bool, device=device)

        if was_training:
            self.train()

    def _distribution_calibration(
        self,
        prototype: np.ndarray,
        transport_prob: np.ndarray,
        n_lsamples: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        weights = np.asarray(transport_prob, dtype=np.float64)
        weights = weights / np.clip(weights.sum(), a_min=1e-12, a_max=None)
        scaled = (n_lsamples * weights).reshape(-1, 1)

        base_means = self.base_means.detach().cpu().numpy().astype(np.float64)
        base_cov = self.base_cov.detach().cpu().numpy().astype(np.float64)
        feat_dim = base_means.shape[1]

        calibrated_mean = (1.0 - self.lambd) * np.sum(scaled * base_means, axis=0) + self.lambd * prototype
        calibrated_cov = np.sum(scaled.reshape(-1, 1, 1) * base_cov, axis=0) + self.alpha
        calibrated_cov = calibrated_cov + self.jitter * np.eye(feat_dim, dtype=np.float64)
        return calibrated_mean, calibrated_cov

    def _episode_classifier(self, support_feat: np.ndarray, shot_num: int) -> LogisticRegression:
        n_support = support_feat.shape[0]
        support_labels = np.repeat(np.arange(self.way_num), shot_num)
        sampled_data = []
        sampled_labels = []

        base_means = self.base_means
        support_torch = torch.from_numpy(support_feat.astype(np.float32)).to(base_means.device)
        if self.base_feature_bank.numel() == 0 or self.base_sample_prob.numel() == 0:
            raise RuntimeError("ADC-HOT base statistics have not been fitted.")

        support_each = support_torch.unsqueeze(0)
        cost_inner, _, _ = self.sinkhorn_inner(self.base_feature_bank, support_each, self.base_sample_prob)
        _, pi, _ = self.sinkhorn_given_cost(base_means, support_torch, cost_inner)

        num_sampled = max(1, int(self.samples_per_class / max(1, shot_num)))
        for support_idx in range(n_support):
            mean, cov = self._distribution_calibration(
                support_feat[support_idx],
                pi[:, support_idx].detach().cpu().numpy(),
                n_support,
            )
            try:
                sampled = self._rng.multivariate_normal(mean=mean, cov=cov, size=num_sampled)
            except np.linalg.LinAlgError:
                cov = cov + (10 * self.jitter) * np.eye(cov.shape[0], dtype=np.float64)
                sampled = self._rng.multivariate_normal(mean=mean, cov=cov, size=num_sampled)
            sampled_data.append(sampled)
            sampled_labels.extend([support_labels[support_idx]] * num_sampled)

        sampled_data = np.concatenate(sampled_data, axis=0)
        augmented_x = np.concatenate([support_feat, sampled_data], axis=0)
        augmented_y = np.concatenate([support_labels, np.asarray(sampled_labels, dtype=np.int64)], axis=0)

        classifier = LogisticRegression(max_iter=1000)
        classifier.fit(augmented_x, augmented_y)
        return classifier

    def forward(self, query: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        if not bool(self.base_stats_ready.item()):
            raise RuntimeError(
                "ADC-HOT base statistics are missing. Run fit_base_statistics(...) after backbone pretraining "
                "before validation/test inference."
            )

        batch_size, num_query, channels, height, width = query.size()
        _, way_num, shot_num, _, _, _ = support.size()
        if way_num != self.way_num:
            raise ValueError(f"ADCHOTNet initialized for way_num={self.way_num}, got episode way_num={way_num}")

        query_feat = self.extract_features(query.view(-1, channels, height, width)).view(batch_size, num_query, -1)
        support_feat = self.extract_features(support.view(-1, channels, height, width)).view(
            batch_size,
            way_num,
            shot_num,
            -1,
        )
        query_feat = self._tukey_transform_tensor(query_feat)
        support_feat = self._tukey_transform_tensor(support_feat)

        outputs = []
        for batch_idx in range(batch_size):
            support_np = support_feat[batch_idx].reshape(way_num * shot_num, -1).detach().cpu().numpy().astype(np.float64)
            query_np = query_feat[batch_idx].detach().cpu().numpy().astype(np.float64)

            classifier = self._episode_classifier(support_np, shot_num)
            log_prob = classifier.predict_log_proba(query_np)
            episode_logits = np.full((num_query, way_num), -np.inf, dtype=np.float32)
            for cls_index, cls_id in enumerate(classifier.classes_):
                episode_logits[:, int(cls_id)] = log_prob[:, cls_index]
            outputs.append(torch.from_numpy(episode_logits).to(query.device))

        return torch.cat(outputs, dim=0)
