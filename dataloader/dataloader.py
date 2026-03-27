"""Episodic sampler for N-way K-shot few-shot learning."""

import torch
from torch.utils.data import Dataset


class FewshotDataset(Dataset):
    """N-way K-shot episode generator."""

    def __init__(
        self,
        data,
        labels,
        episode_num,
        way_num,
        shot_num,
        query_num,
        seed=None,
        hard_pool=None,
        hard_ratio=0.0,
        augment=False,
        augment_cfg=None,
        return_indices=False,
    ):
        self.data = data
        self.labels = labels
        self.episode_num = episode_num
        self.way_num = way_num
        self.shot_num = shot_num
        self.query_num = query_num
        self.seed = seed if seed is not None else 0
        self.hard_ratio = float(hard_ratio)
        self.augment = bool(augment)
        self.return_indices = bool(return_indices)

        self.hard_pool = {}
        if hard_pool is not None:
            for class_id in range(self.way_num):
                indices = hard_pool.get(class_id, [])
                if isinstance(indices, torch.Tensor):
                    indices = indices.long().cpu()
                else:
                    indices = torch.tensor(list(indices), dtype=torch.long)
                self.hard_pool[class_id] = indices

        default_aug = {
            "time_shift_max": 4,
            "time_shift_prob": 0.5,
            "amp_scale_min": 0.9,
            "amp_scale_max": 1.1,
            "amp_scale_prob": 0.5,
            "time_mask_width": 4,
            "time_mask_prob": 0.25,
            "freq_mask_width": 4,
            "freq_mask_prob": 0.25,
        }
        self.augment_cfg = default_aug
        if augment_cfg:
            self.augment_cfg.update(augment_cfg)

        self.class_indices = {}
        for class_id in range(way_num):
            self.class_indices[class_id] = (labels == class_id).nonzero(as_tuple=True)[0]

        self._validate()

    def _validate(self):
        required = self.shot_num + self.query_num
        for class_id in range(self.way_num):
            available = len(self.class_indices[class_id])
            if available < required:
                print(f"Warning: Class {class_id} has {available} samples, need {required}")

    def __len__(self):
        return self.episode_num

    def _sample_hard_index(self, class_id, available, generator):
        if class_id not in self.hard_pool:
            return None
        hard_idx = self.hard_pool[class_id]
        if hard_idx.numel() == 0:
            return None

        hard_set = set(hard_idx.tolist())
        candidates = [idx.item() for idx in available if idx.item() in hard_set]
        if not candidates:
            return None

        pos = torch.randint(0, len(candidates), (1,), generator=generator).item()
        return int(candidates[pos])

    def _augment_one(self, image, generator):
        cfg = self.augment_cfg
        out = image.clone()
        _, height, width = out.shape

        if torch.rand(1, generator=generator).item() < cfg["time_shift_prob"]:
            shift = int(
                torch.randint(
                    -cfg["time_shift_max"],
                    cfg["time_shift_max"] + 1,
                    (1,),
                    generator=generator,
                ).item()
            )
            if shift != 0:
                out = torch.roll(out, shifts=shift, dims=2)

        if torch.rand(1, generator=generator).item() < cfg["amp_scale_prob"]:
            scale = cfg["amp_scale_min"] + (
                cfg["amp_scale_max"] - cfg["amp_scale_min"]
            ) * torch.rand(1, generator=generator).item()
            out = out * scale

        if cfg["time_mask_width"] > 0 and torch.rand(1, generator=generator).item() < cfg["time_mask_prob"]:
            width_mask = min(int(cfg["time_mask_width"]), width)
            start = int(torch.randint(0, max(1, width - width_mask + 1), (1,), generator=generator).item())
            out[:, :, start : start + width_mask] = 0.0

        if cfg["freq_mask_width"] > 0 and torch.rand(1, generator=generator).item() < cfg["freq_mask_prob"]:
            height_mask = min(int(cfg["freq_mask_width"]), height)
            start = int(torch.randint(0, max(1, height - height_mask + 1), (1,), generator=generator).item())
            out[:, start : start + height_mask, :] = 0.0

        return out

    def _augment_batch(self, batch, generator):
        if not self.augment:
            return batch
        return torch.stack([self._augment_one(image, generator) for image in batch], dim=0)

    def __getitem__(self, index):
        generator = torch.Generator()
        generator.manual_seed(self.seed * 10000 + index)

        support_images, support_targets = [], []
        query_images, query_targets = [], []
        support_indices, query_indices = [], []

        for class_id in range(self.way_num):
            indices = self.class_indices[class_id]
            perm = torch.randperm(len(indices), generator=generator)
            shuffled = indices[perm]

            hard_idx = None
            use_hard = (
                self.query_num > 0
                and self.hard_ratio > 0
                and torch.rand(1, generator=generator).item() < self.hard_ratio
            )
            if use_hard:
                hard_idx = self._sample_hard_index(class_id, shuffled, generator)

            if hard_idx is not None:
                hard_idx_t = torch.tensor([hard_idx], dtype=shuffled.dtype)
                remain = shuffled[shuffled != hard_idx]
                support_idx = remain[: self.shot_num]
                extra_query = remain[self.shot_num : self.shot_num + max(self.query_num - 1, 0)]
                query_idx = torch.cat([hard_idx_t, extra_query], dim=0)
            else:
                support_idx = shuffled[: self.shot_num]
                query_idx = shuffled[self.shot_num : self.shot_num + self.query_num]

            support_images.append(self.data[support_idx])
            query_images.append(self.data[query_idx])
            support_indices.append(support_idx)
            query_indices.append(query_idx)

            support_targets.append(torch.full((len(support_idx),), class_id, dtype=torch.long))
            query_targets.append(torch.full((len(query_idx),), class_id, dtype=torch.long))

        query_images = torch.cat(query_images)
        query_targets = torch.cat(query_targets)
        support_images = torch.cat(support_images)
        support_targets = torch.cat(support_targets)
        query_indices = torch.cat(query_indices)
        support_indices = torch.cat(support_indices)

        if self.augment:
            query_images = self._augment_batch(query_images, generator)
            support_images = self._augment_batch(support_images, generator)

        if self.return_indices:
            return (
                query_images,
                query_targets,
                support_images,
                support_targets,
                query_indices,
                support_indices,
            )

        return query_images, query_targets, support_images, support_targets
