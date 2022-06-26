import torch
import torch.nn as nn
from arch.mlp import MLP

class VAE(nn.Module):
    def __init__(self, x0_dim, x1_dim, h_dim, h_rep, z_dim, y_dim):
        super(VAE, self).__init__()
        self.x_to_mu = MLP(x0_dim + x1_dim, h_dim, h_rep, z_dim)
        self.x_to_logvar = MLP(x0_dim + x1_dim, h_dim, h_rep, z_dim)
        self.x0_decoder = MLP(z_dim, h_dim, h_rep, x0_dim)
        self.x1_decoder = MLP(z_dim, h_dim, h_rep, x1_dim)

    def posterior_params(self, x0, x1):
        x = torch.cat((x0, x1))
        return self.x_to_mu(x), self.x_to_logvar(x)

    def sample_z(self, mu, logvar, n_samples=1):
        if self.training:
            sd = torch.exp(logvar / 2) # Same as sqrt(exp(logvar))
            eps = torch.randn((n_samples, len(sd))) if n_samples > 1 else torch.randn_like(sd)
            return mu + eps * sd
        else:
            return mu

    def forward(self, x0, x1):
        mu, logvar = self.posterior_params(x0, x1)
        z = self.sample_z(mu, logvar)
        x0_reconst = self.x0_decoder(z)
        x1_reconst = self.x1_decoder(z)
        return x0_reconst, x1_reconst, mu, logvar

class SSVAE(nn.Module):
    def __init__(self, x0_dim, x1_dim, h_dim, h_reps, z_dim, y_dim):
        super(SSVAE, self).__init__()
        self.encoder_mu = MLP(x0_dim + x1_dim + y_dim, h_dim, h_reps, z_dim)
        self.encoder_logvar = MLP(x0_dim + x1_dim + y_dim, h_dim, h_reps, z_dim)
        self.x0_decoder = MLP(z_dim + y_dim, h_dim, h_reps, x0_dim)
        self.x1_decoder = MLP(z_dim + y_dim, h_dim, h_reps, x1_dim)

    def posterior_params(self, x0, x1, y):
        xy = torch.hstack((x0, x1, y))
        return self.encoder_mu(xy), self.encoder_logvar(xy)

    def sample_z(self, mu, logvar, n_samples=1):
        sd = torch.exp(logvar / 2) # Same as sqrt(exp(logvar))
        eps = torch.randn((n_samples, len(sd))) if n_samples > 1 else torch.randn_like(sd)
        return mu + eps * sd

    def forward(self, x0, x1, y):
        mu, logvar = self.posterior_params(x0, x1, y)
        z = self.sample_z(mu, logvar)
        x0_reconst = self.x0_decoder(torch.hstack((z, y)))
        x1_reconst = self.x1_decoder(torch.hstack((z, y)))
        return x0_reconst, x1_reconst, mu, logvar