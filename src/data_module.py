import os
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from typing import Optional
from src.dataset import TrainDataset, ValDataset, TestDataset


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
    ):
        super(DataModule, self).__init__()

        self.batch_size = batch_size
        self.train = ...
        self.val = ...
        self.test = ...


    # will be called in every GPUs
    def setup(self, stage: Optional[str] = None) -> None:
        self.train = TrainDataset()
        self.val = ValDataset()
        self.test = TestDataset()


    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=os.cpu_count(),
            pin_memory=True,
        )


    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            num_workers=os.cpu_count(),
            pin_memory=True,
        )


    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            num_workers=os.cpu_count(),
            pin_memory=True,
        )