from __future__ import annotations

import h5py
import numpy as np
import torch

from .common import H5SequenceWindowDataset


class PushTSequenceDataset(H5SequenceWindowDataset):
    def __init__(
        self,
        *,
        path: str | None = None,
        dataset_name: str = "pusht_expert_train",
        cache_dir: str | None = None,
        image_size: int = 96,
        plan_horizon: int = 8,
        action_horizon: int = 4,
        train: bool = True,
        train_split: float = 0.98,
        seed: int = 7,
        stride: int = 8,
        max_samples: int | None = None,
    ) -> None:
        super().__init__(
            path=path,
            dataset_name=dataset_name,
            cache_dir=cache_dir,
            image_key="pixels",
            gripper_key=None,
            proprio_key="proprio",
            state_key="state",
            action_key="action",
            language_key=None,
            sequence_length=plan_horizon + 1,
            action_horizon=action_horizon,
            train=train,
            train_split=train_split,
            seed=seed,
            stride=stride,
            max_samples=max_samples,
            image_size=image_size,
        )
        self._swm_dataset = None
        with h5py.File(self.path, "r") as handle:
            self._supports_stablewm = "ep_len" in handle and "ep_offset" in handle

    def _ensure_dataset(self):
        if self._swm_dataset is None:
            import stable_worldmodel as swm

            self._swm_dataset = swm.data.HDF5Dataset(
                self.path.stem,
                cache_dir=self.path.parent,
            )
        return self._swm_dataset

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if not self._supports_stablewm:
            return super().__getitem__(index)
        start = int(self.starts[index])
        dataset = self._ensure_dataset()
        indices = np.arange(start, start + self.sequence_length)
        rows = dataset.get_row_data(indices.tolist())
        sample: dict[str, torch.Tensor] = {
            "pixels": self._normalize_images(np.asarray(rows["pixels"])),
            "action": torch.as_tensor(np.asarray(rows["action"][: self.action_horizon])).float(),
            "index": torch.tensor(start, dtype=torch.long),
        }
        if "proprio" in rows:
            sample["proprio"] = torch.as_tensor(np.asarray(rows["proprio"])).float()
        if "state" in rows:
            sample["state"] = torch.as_tensor(np.asarray(rows["state"])).float()
        return sample

    def close(self) -> None:
        self._swm_dataset = None
