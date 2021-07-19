import os
import torch
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from typing import Optional


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
        self.train = [torch.randn(64, 20) for _ in range(1000)]
        self.val = [torch.randn(64, 20) for _ in range(20)]
        self.test = [torch.randn(64, 20) for _ in range(20)]


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