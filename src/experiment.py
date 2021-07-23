import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning.callbacks import LearningRateMonitor
from src.data_module import DataModule
from src.model import VAE
from torch import Tensor
from torch.optim import Optimizer


class Experiment(pl.LightningModule):
    def __init__(self, config: DictConfig) -> None:
        super(Experiment, self).__init__()
        self.config: DictConfig = config
        logger = instantiate(config.logger)
        self.trainer = instantiate(
            config.trainer,
            logger=logger,
            callbacks=[
                LearningRateMonitor(logging_interval="step"),
            ],
        )
        self.model = VAE(latent_dim=config.latent_dim)
        self.data_module = DataModule(batch_size=config.batch_size)

    def configure_optimizers(self):
        params = self.model.parameters()
        optimizer: Optimizer = instantiate(self.config.optimizer, params=params)
        scheduler = instantiate(self.config.scheduler, optimizer=optimizer)
        return [optimizer], [scheduler]

    def loss_fn(self, recon_x: Tensor, x: Tensor, mu: Tensor, logvar: Tensor) -> Tensor:
        BCE = F.binary_cross_entropy_with_logits(
            recon_x, x.view(-1, 784), reduction="sum"
        )
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD

    def training_step(self, batch: Tensor, batch_idx: int):
        recon_batch, mu, logvar = self.model(batch)
        loss = self.loss_fn(recon_batch, batch, mu, logvar)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: Tensor, batch_idx: int):
        recon_batch, mu, logvar = self.model(batch)
        loss = self.loss_fn(recon_batch, batch, mu, logvar)
        self.log("val_loss", loss)
        return loss

    # train your model
    def fit(self):
        self.trainer.fit(self, self.data_module)
        self.logger.log_hyperparams(
            {
                "batch_size": self.config.batch_size,
                "lr": self.config.lr,
            }
        )
        self.log_artifact(".hydra/config.yaml")
        self.log_artifact(".hydra/hydra.yaml")
        self.log_artifact(".hydra/overrides.yaml")
        self.log_artifact("main.log")

    # run your whole experiments
    def run(self):
        self.fit()

    def log_artifact(self, artifact_path: str):
        self.logger.experiment.log_artifact(self.logger.run_id, artifact_path)
