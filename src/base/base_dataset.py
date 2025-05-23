from abc import abstractmethod

from torch.utils.data import DataLoader, Dataset

class BaseDataset(Dataset):

    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, idx):
        raise NotImplementedError

    @classmethod
    def get_dataloader(cls, dataset_kwargs, dataloader_kwargs, collate_fn=None):
        return DataLoader(
            cls(**dataset_kwargs), collate_fn=collate_fn, **dataloader_kwargs
        )
