import torch
import torch.nn as nn
from arch.mlp import MLP

class SSVAE(nn.Module):
    def __init__(self, input0_dim, input1_dim, hidden_dims, latent_dim, target_dim):
        super(SSVAE, self).__init__()
        self.encoder_mu = MLP(input0_dim + input1_dim + target_dim, hidden_dims, latent_dim)
        self.encoder_logvar = MLP(input0_dim + input1_dim + target_dim, hidden_dims, latent_dim)
        self.x0_decoder = MLP(latent_dim + target_dim, hidden_dims, input0_dim)
        self.x1_decoder_mu = nn.Linear(latent_dim + target_dim, input1_dim)
        self.x1_decoder_logprec = nn.Linear(latent_dim + target_dim, input1_dim)

    def posterior_params(self, x0, x1, y):
        xy = torch.hstack((x0, x1, y))
        return self.encoder_mu(xy), self.encoder_logvar(xy)

    def sample_z(self, mu, logvar, n_samples=1):
        if self.training:
            sd = torch.exp(logvar / 2) # Same as sqrt(exp(logvar))
            eps = torch.randn((n_samples, len(sd))) if n_samples > 1 else torch.randn_like(sd)
            return mu + eps * sd
        else:
            return mu

    def forward(self, x0, x1, y):
        mu, logvar = self.posterior_params(x0, x1, y)
        z = self.sample_z(mu, logvar)
        x0_reconst = self.x0_decoder(torch.hstack((z, y)))
        x1_mu = self.x1_decoder_mu(torch.hstack((z, y)))
        x1_logprec = self.x1_decoder_logprec(torch.hstack((z, y)))
        return x0_reconst, x1_mu, x1_logprec, mu, logvar