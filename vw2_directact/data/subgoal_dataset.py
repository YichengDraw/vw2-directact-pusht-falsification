from __future__ import annotations

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from .common import IMAGENET_MEAN, IMAGENET_STD, resolve_h5_path


class PushTSubgoalDataset(torch.utils.data.Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        *,
        path: str | None = None,
        dataset_name: str = "pusht_expert_train",
        cache_dir: str | None = None,
        image_size: int = 96,
        history_steps: int = 4,
        future_horizon: int = 8,
        action_horizon: int = 4,
        train: bool = True,
        train_split: float = 0.98,
        seed: int = 7,
        stride: int = 8,
        max_samples: int | None = None,
    ) -> None:
        super().__init__()
        self.path = resolve_h5_path(path, dataset_name, cache_dir)
        self.image_size = image_size
        self.history_steps = history_steps
        self.future_horizon = future_horizon
        self.action_horizon = action_horizon

        with h5py.File(self.path, "r") as handle:
            self.episode_key = "episode_idx" if "episode_idx" in handle.keys() else "ep_idx"
            episode_index = np.asarray(handle[self.episode_key])
            self.has_step_idx = "step_idx" in handle
            self._supports_stablewm = "ep_len" in handle and "ep_offset" in handle
        self._swm_dataset = None

        candidate_current = np.arange(history_steps - 1, len(episode_index) - future_horizon, stride, dtype=np.int64)
        valid_mask = episode_index[candidate_current - (history_steps - 1)] == episode_index[candidate_current + future_horizon]
        starts = candidate_current[valid_mask]

        rng = np.random.default_rng(seed)
        shuffled = starts[rng.permutation(len(starts))]
        split_at = int(len(shuffled) * train_split)
        split = shuffled[:split_at] if train else shuffled[split_at:]
        if max_samples is not None:
            split = split[:max_samples]
        self.current_indices = split

    def __len__(self) -> int:
        return int(self.current_indices.shape[0])

    def _normalize_images(self, images: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255.0
        if tensor.shape[-2:] != (self.image_size, self.image_size):
            tensor = F.interpolate(tensor, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        return (tensor - IMAGENET_MEAN) / IMAGENET_STD

    def _ensure_dataset(self):
        if self._swm_dataset is None:
            import stable_worldmodel as swm

            self._swm_dataset = swm.data.HDF5Dataset(
                self.path.stem,
                cache_dir=self.path.parent,
            )
        return self._swm_dataset

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        current = int(self.current_indices[index])
        history_start = current - self.history_steps + 1
        future_end = current + self.future_horizon + 1

        if self._supports_stablewm:
            dataset = self._ensure_dataset()
            indices = np.arange(history_start, future_end)
            rows = dataset.get_row_data(indices.tolist())
            pixels = np.asarray(rows["pixels"])
            actions = np.asarray(rows["action"][: self.history_steps - 1 + self.action_horizon]).astype(np.float32)
            proprio = np.asarray(rows["proprio"]).astype(np.float32) if "proprio" in rows else None
            state = np.asarray(rows["state"]).astype(np.float32) if "state" in rows else None
            episode_idx = int(np.asarray(rows[self.episode_key])[self.history_steps - 1])
            step_idx = int(np.asarray(rows["step_idx"])[self.history_steps - 1]) if "step_idx" in rows else current
        else:
            with h5py.File(self.path, "r") as handle:
                pixels = np.asarray(handle["pixels"][history_start:future_end])
                actions = np.asarray(handle["action"][history_start : current + self.action_horizon]).astype(np.float32)
                proprio = np.asarray(handle["proprio"][history_start:future_end]).astype(np.float32) if "proprio" in handle else None
                state = np.asarray(handle["state"][history_start:future_end]).astype(np.float32) if "state" in handle else None
                episode_idx = int(handle[self.episode_key][current])
                step_idx = int(handle["step_idx"][current]) if self.has_step_idx else current

        history_pixels = self._normalize_images(pixels[: self.history_steps])
        future_pixels = self._normalize_images(pixels[self.history_steps :])

        prev_actions = np.zeros((self.history_steps, actions.shape[-1]), dtype=np.float32)
        if self.history_steps > 1:
            prev_actions[1:] = actions[: self.history_steps - 1]
        target_actions = actions[self.history_steps - 1 : self.history_steps - 1 + self.action_horizon]

        sample: dict[str, torch.Tensor] = {
            "history_pixels": history_pixels,
            "future_pixels": future_pixels,
            "prev_actions": torch.from_numpy(prev_actions).float(),
            "action": torch.from_numpy(target_actions).float(),
            "horizon": torch.tensor(self.future_horizon, dtype=torch.long),
            "index": torch.tensor(current, dtype=torch.long),
            "episode_idx": torch.tensor(episode_idx, dtype=torch.long),
            "step_idx": torch.tensor(step_idx, dtype=torch.long),
        }
        if proprio is not None:
            sample["history_proprio"] = torch.from_numpy(proprio[: self.history_steps]).float()
            sample["future_proprio"] = torch.from_numpy(proprio[self.history_steps :]).float()
        if state is not None:
            sample["history_state"] = torch.from_numpy(state[: self.history_steps]).float()
            sample["future_state"] = torch.from_numpy(state[self.history_steps :]).float()
        return sample

    def close(self) -> None:
        self._swm_dataset = None
