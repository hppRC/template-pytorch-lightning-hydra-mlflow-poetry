import torch


class Dataset(torch.utils.data.Dataset):
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, key):
        return self.dataset[key]


class TrainDataset(Dataset):
    def __init__(self) -> None:
        super(Dataset, self).__init__()
        self.dataset = [torch.randn(64, 20) for _ in range(1000)]


class ValDataset(Dataset):
    def __init__(self) -> None:
        super(Dataset, self).__init__()
        self.dataset = [torch.randn(64, 20) for _ in range(20)]


class TestDataset(Dataset):
    def __init__(self) -> None:
        super(Dataset, self).__init__()
        self.dataset = [torch.randn(64, 20) for _ in range(20)]