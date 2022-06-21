import torch
import torch.nn as nn
from arch.mlp import MLP

class TwoScalarVAE(nn.Module):
    def __init__(self, hidden_dim, n_hidden, latent_dim):
        super(TwoScalarVAE, self).__init__()
        self.x_to_mu = MLP(2, hidden_dim, n_hidden, latent_dim)
        self.x_to_logvar = MLP(2, hidden_dim, n_hidden, latent_dim)
        self.x0_decoder = MLP(latent_dim, hidden_dim, n_hidden, 1)
        self.x1_decoder = MLP(latent_dim, hidden_dim, n_hidden, 1)

    def posterior_params(self, x0, x1):
        x = torch.cat((x0, x1))
        return self.x_to_mu(x), self.x_to_logvar(x)

    def sample_latents(self, mu, logvar, n_samples=1):
        sd = torch.exp(logvar / 2) # Same as sqrt(exp(logvar))
        eps = torch.randn((n_samples, len(sd))) if n_samples > 1 else torch.randn_like(sd)
        return mu + eps * sd

    def forward(self, x0, x1):
        mu, logvar = self.posterior_params(x0, x1)
        latent = self.sample_latents(mu, logvar)
        x0_reconst = self.x0_decoder(latent)
        x1_reconst = self.x1_decoder(latent)
        return x0_reconst, x1_reconst, mu, logvar

class TwoScalarSSVAE(nn.Module):
    def __init__(self, hidden_dim, n_hidden, latent_dim):
        super(TwoScalarSSVAE, self).__init__()
        self.xy_to_mu = MLP(3, hidden_dim, n_hidden, latent_dim)
        self.xy_to_logvar = MLP(3, hidden_dim, n_hidden, latent_dim)
        self.x0_decoder = MLP(latent_dim + 1, hidden_dim, n_hidden, 1)
        self.x1_decoder = MLP(latent_dim + 1, hidden_dim, n_hidden, 1)

    def posterior_params(self, x0, x1, y):
        xy = torch.hstack((x0, x1, y))
        return self.xy_to_mu(xy), self.xy_to_logvar(xy)

    def sample_latents(self, mu, logvar, n_samples=1):
        sd = torch.exp(logvar / 2) # Same as sqrt(exp(logvar))
        eps = torch.randn((n_samples, len(sd))) if n_samples > 1 else torch.randn_like(sd)
        return mu + eps * sd

    def forward(self, x0, x1, y):
        mu, logvar = self.posterior_params(x0, x1, y)
        latent = self.sample_latents(mu, logvar)
        x0_reconst = self.x0_decoder(torch.hstack((latent, y)))
        x1_reconst = self.x1_decoder(torch.hstack((latent, y)))
        return x0_reconst, x1_reconst, mu, logvar