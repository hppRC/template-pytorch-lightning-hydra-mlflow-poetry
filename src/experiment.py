import torch
from torch import Tensor
from torch.optim import Optimizer
import torch.nn.functional as F
import pytorch_lightning as pl

from omegaconf import DictConfig
from hydra.utils import instantiate

from src.model import VAE


class Experiment(pl.LightningModule):
    def __init__(self, config: DictConfig) -> None:
        super(Experiment, self).__init__()
        self.config: DictConfig = config
        self.model = VAE()
        self.save_hyperparameters()


    def configure_optimizers(self):
        optimizer: Optimizer = instantiate(self.config.optimizer, params=self.model.parameters())
        # scheduler = 
        # return instantiate(self.config.optimizer.optimizer_config, optimizer=optimizer)
        return 


    def loss_fn(self, recon_x: Tensor, x: Tensor, mu: Tensor, logvar: Tensor):
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD


    def training_step(self, batch: Tensor, batch_idx: int):
        recon_batch, mu, logvar = self.model(batch)
        loss = self.loss_fn(recon_batch, batch, mu, logvar)
        return loss