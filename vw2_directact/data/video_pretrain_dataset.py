from __future__ import annotations

from torch.utils.data import Dataset

from .calvin_dataset import CalvinSequenceDataset
from .pusht_dataset import PushTSequenceDataset


class VideoPretrainDataset(Dataset):
    def __init__(self, *, dataset_type: str, **kwargs) -> None:
        super().__init__()
        if dataset_type == "pusht":
            self.dataset = PushTSequenceDataset(**kwargs)
        elif dataset_type == "calvin":
            self.dataset = CalvinSequenceDataset(**kwargs)
        else:
            raise ValueError(f"Unsupported dataset_type={dataset_type!r}.")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        return self.dataset[index]
