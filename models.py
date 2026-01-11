
"""
@author: NUOJIN
"""

from typing import List, Any, Tuple
from torch import nn, Tensor
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import copy
import tqdm
import numpy as np
import random


def seed_worker(worker_id):
    worker_seed = 42 + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(42)

class BaseVAE(nn.Module):
    
    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError

    def sample(self, batch_size:int, current_device: int, **kwargs) -> Tensor:
        raise NotImplementedError

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError
        
class VAE(BaseVAE):

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 beta: float = 1.0,
                 dev: str = 'cuda',
                 **kwargs) -> None:
        super(VAE, self).__init__()

        self.latent_dim = latent_dim
        self.beta = beta
        input_dim = in_channels
        
        self.dev = dev

        modules = []
        if hidden_dims is None:
            hidden_dims = [128, 64]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_channels, h_dim),
                    nn.GELU())
                    # nn.ReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)


        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1])

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i],hidden_dims[i + 1]),
                    nn.GELU())
                    # nn.ReLU())
            )



        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Linear(hidden_dims[-1], input_dim)

    def encode(self, x: Tensor) -> List[Tensor]:
        result = self.encoder(x)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        result = self.decoder_input(z)
        # result = F.relu(result)
        result = F.gelu(result)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        torch.manual_seed(42)  # DETERMINISTIC SEED 
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), x, mu, log_var]
    
    def loss_function(self,
                      data: Tensor
                      ) -> dict:
        mu,log_var = self.encode(data)
        
        data_hat = self.forward(data)[0]
        recons_loss = ((data - data_hat)**2).sum()

        kld_loss = torch.sum(-0.5 * (1 + log_var - mu ** 2 - torch.exp(2 * log_var)))
        
        # normalize by batch size
        recons_loss /= len(data)
        kld_loss /= len(data)

        loss = recons_loss + self.beta * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':kld_loss.detach()}

    def sample(self,
               num_samples:int,
               **kwargs) -> Tensor:
        z = torch.randn(num_samples,
                        self.latent_dim).to(self.dev)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        return self.forward(x)[0]
    
    def train(self, 
              trn_data: Tensor,
              batch_size: int = 64,
              epochs: int = 500):
        trn_dataloader = DataLoader(
            trn_data, 
            batch_size=batch_size, 
            shuffle=False,  # Important for deterministic order
            worker_init_fn=seed_worker,
            generator=g
        )
        opt = torch.optim.Adam(self.parameters(), lr=1e-3, betas= (0.9, 0.99)) # not related to beta in loss function
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', factor = 0.9, patience = 500, eps = 1e-5, cooldown = 2500, verbose = True)
        losses = []
        rec_losses = []
        kld_losses = []
        with tqdm.tqdm(range(epochs), unit=' Epoch') as tepoch:
            epoch_loss = 0
            epoch_rec_loss = 0
            epoch_kld_loss = 0
            for epoch in tepoch:
                for batch_index, data in enumerate(trn_dataloader):
                    opt.zero_grad()
                    batch_losses = self.loss_function(data)
                    trn_loss = batch_losses['loss']
                    rec_loss = batch_losses['Reconstruction_Loss']
                    kld_loss = batch_losses['KLD']
                    trn_loss.backward()
                    epoch_loss += trn_loss.item()
                    epoch_rec_loss += rec_loss.item()
                    epoch_kld_loss += kld_loss.item()
                    opt.step()
    
                    # epoch_loss += trn_loss
                    # epoch_rec_loss += rec_loss
                    # epoch_kld_loss += kld_loss
                epoch_loss /= len(trn_dataloader)
                epoch_rec_loss /= len(trn_dataloader)
                epoch_kld_loss /= len(trn_dataloader)
                sch.step(epoch_loss)
                # losses.append(np.copy(epoch_loss.cpu().detach().numpy()))
                # rec_losses.append(np.copy(epoch_rec_loss.cpu().detach().numpy()))
                # kld_losses.append(np.copy(epoch_kld_loss.cpu().detach().numpy()))
                # tepoch.set_postfix(loss=epoch_loss.cpu().detach().numpy())
                losses.append(epoch_loss)
                rec_losses.append(epoch_rec_loss)
                kld_losses.append(epoch_kld_loss)
                tepoch.set_postfix(loss=epoch_loss)
    
        return losses, rec_losses, kld_losses