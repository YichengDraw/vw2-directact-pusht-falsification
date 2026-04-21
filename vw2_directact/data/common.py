from __future__ import annotations

import os
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)


def resolve_h5_path(path: str | None, dataset_name: str | None, cache_dir: str | None) -> Path:
    if path:
        return Path(path)
    if not dataset_name:
        raise ValueError("Either 'path' or 'dataset_name' must be provided.")
    root = cache_dir or os.environ.get("STABLEWM_HOME") or os.path.join(Path.home(), ".stable-wm")
    return Path(root) / f"{dataset_name}.h5"


class H5SequenceWindowDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        *,
        path: str | None,
        dataset_name: str | None,
        cache_dir: str | None,
        image_key: str,
        gripper_key: str | None,
        proprio_key: str | None,
        state_key: str | None,
        action_key: str,
        language_key: str | None,
        sequence_length: int,
        action_horizon: int,
        train: bool,
        train_split: float,
        seed: int,
        stride: int,
        max_samples: int | None,
        image_size: int,
    ) -> None:
        super().__init__()
        self.path = resolve_h5_path(path, dataset_name, cache_dir)
        self.image_key = image_key
        self.gripper_key = gripper_key
        self.proprio_key = proprio_key
        self.state_key = state_key
        self.action_key = action_key
        self.language_key = language_key
        self.sequence_length = sequence_length
        self.action_horizon = action_horizon
        self.image_size = image_size

        with h5py.File(self.path, "r") as handle:
            episode_key = "episode_idx" if "episode_idx" in handle.keys() else "ep_idx"
            episode_index = np.asarray(handle[episode_key])

        max_start = len(episode_index) - self.sequence_length + 1
        candidates = np.arange(0, max_start, stride, dtype=np.int64)
        valid_mask = episode_index[candidates] == episode_index[candidates + self.sequence_length - 1]
        starts = candidates[valid_mask]

        rng = np.random.default_rng(seed)
        shuffled = starts[rng.permutation(len(starts))]
        split_at = int(len(shuffled) * train_split)
        split = shuffled[:split_at] if train else shuffled[split_at:]
        if max_samples is not None:
            split = split[:max_samples]
        self.starts = split

    def __len__(self) -> int:
        return int(self.starts.shape[0])

    def _load_array(self, array: np.ndarray | None) -> torch.Tensor | None:
        if array is None:
            return None
        return torch.from_numpy(array)

    def _normalize_images(self, images: np.ndarray | None) -> torch.Tensor | None:
        if images is None:
            return None
        tensor = torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255.0
        if tensor.shape[-1] != self.image_size or tensor.shape[-2] != self.image_size:
            tensor = F.interpolate(tensor, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
        return tensor

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        start = int(self.starts[index])
        with h5py.File(self.path, "r") as handle:
            pixels = np.asarray(handle[self.image_key][start : start + self.sequence_length])
            actions = np.asarray(handle[self.action_key][start : start + self.action_horizon])
            gripper = np.asarray(handle[self.gripper_key][start : start + self.sequence_length]) if self.gripper_key else None
            proprio = np.asarray(handle[self.proprio_key][start : start + self.sequence_length]) if self.proprio_key else None
            state = np.asarray(handle[self.state_key][start : start + self.sequence_length]) if self.state_key else None
            language = np.asarray(handle[self.language_key][start : start + self.sequence_length]) if self.language_key else None

        sample: dict[str, torch.Tensor] = {
            "pixels": self._normalize_images(pixels),
            "action": self._load_array(actions).float(),
            "index": torch.tensor(start, dtype=torch.long),
        }
        if gripper is not None:
            sample["gripper_pixels"] = self._normalize_images(gripper)
        if proprio is not None:
            sample["proprio"] = self._load_array(proprio).float()
        if state is not None:
            sample["state"] = self._load_array(state).float()
        if language is not None:
            sample["language"] = self._load_array(language).float()
        return sample

    def close(self) -> None:
        return None


def build_torch_split(
    dataset: Dataset[dict[str, torch.Tensor]],
    *,
    batch_size: int,
    num_workers: int,
    train: bool,
) -> torch.utils.data.DataLoader[dict[str, torch.Tensor]]:
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        drop_last=train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
