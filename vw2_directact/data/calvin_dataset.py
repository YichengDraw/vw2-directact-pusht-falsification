from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .common import H5SequenceWindowDataset, IMAGENET_MEAN, IMAGENET_STD


class CalvinSequenceDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        *,
        path: str,
        image_size: int,
        plan_horizon: int,
        action_horizon: int,
        train: bool,
        train_split: float,
        seed: int,
        stride: int,
        max_samples: int | None,
        use_h5: bool | None = None,
    ) -> None:
        super().__init__()
        target = Path(path)
        if use_h5 is None:
            use_h5 = target.suffix == ".h5"

        if use_h5:
            self.dataset: Dataset[dict[str, torch.Tensor]] = H5SequenceWindowDataset(
                path=str(target),
                dataset_name=None,
                cache_dir=None,
                image_key="rgb_static",
                gripper_key="rgb_gripper",
                proprio_key="robot_obs",
                state_key="scene_obs",
                action_key="actions",
                language_key="language_embedding",
                sequence_length=plan_horizon + 1,
                action_horizon=action_horizon,
                train=train,
                train_split=train_split,
                seed=seed,
                stride=stride,
                max_samples=max_samples,
                image_size=image_size,
            )
            self.records: list[tuple[Path, int]] = []
            self.image_size = image_size
            self.plan_horizon = plan_horizon
            self.action_horizon = action_horizon
            return

        files = sorted(target.glob("*.npz"))
        if not files:
            raise FileNotFoundError(f"No CALVIN episode files found under {target}.")

        self.dataset = None
        self.image_size = image_size
        self.plan_horizon = plan_horizon
        self.action_horizon = action_horizon
        self.records: list[tuple[Path, int]] = []
        for file_path in files:
            with np.load(file_path) as episode:
                length = int(episode["actions"].shape[0])
            max_start = length - (plan_horizon + 1) + 1
            for start in range(0, max_start, stride):
                self.records.append((file_path, start))

        rng = np.random.default_rng(seed)
        ordered = [self.records[i] for i in rng.permutation(len(self.records))]
        split_at = int(len(ordered) * train_split)
        split = ordered[:split_at] if train else ordered[split_at:]
        if max_samples is not None:
            split = split[:max_samples]
        self.records = split

    def __len__(self) -> int:
        if self.dataset is not None:
            return len(self.dataset)
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if self.dataset is not None:
            return self.dataset[index]

        file_path, start = self.records[index]
        with np.load(file_path) as episode:
            end = start + self.plan_horizon + 1
            static = torch.from_numpy(episode["rgb_static"][start:end]).permute(0, 3, 1, 2).float() / 255.0
            gripper = torch.from_numpy(episode["rgb_gripper"][start:end]).permute(0, 3, 1, 2).float() / 255.0
            static = torch.nn.functional.interpolate(static, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
            gripper = torch.nn.functional.interpolate(gripper, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
            static = (static - IMAGENET_MEAN) / IMAGENET_STD
            gripper = (gripper - IMAGENET_MEAN) / IMAGENET_STD

            sample = {
                "pixels": static,
                "gripper_pixels": gripper,
                "proprio": torch.from_numpy(episode["robot_obs"][start:end]).float(),
                "action": torch.from_numpy(episode["actions"][start : start + self.action_horizon]).float(),
            }
            if "scene_obs" in episode:
                sample["state"] = torch.from_numpy(episode["scene_obs"][start:end]).float()
            if "language_embedding" in episode:
                language = torch.from_numpy(episode["language_embedding"]).float()
                if language.ndim == 1:
                    language = language.unsqueeze(0).expand(self.plan_horizon + 1, -1)
                sample["language"] = language[: self.plan_horizon + 1]
            return sample
