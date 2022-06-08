import torch
import torch.nn as nn
from mlp import MLP

class VAE(nn.Module):
    def __init__(self, x_dim, hidden_dim, n_hidden, latent_dim):
        super(VAE, self).__init__()
        self.x_to_mu = MLP(x_dim, hidden_dim, n_hidden, latent_dim)
        self.x_to_logvar = MLP(x_dim, hidden_dim, n_hidden, latent_dim)
        self.decoder = MLP(latent_dim, hidden_dim, n_hidden, x_dim)

    def posterior_params(self, x):
        return self.x_to_mu(x), self.x_to_logvar(x)

    def sample_latents(self, mu, logvar, n_samples=1):
        sd = torch.exp(logvar / 2) # Same as sqrt(exp(logvar))
        eps = torch.randn((n_samples, len(sd))) if n_samples > 1 else torch.randn_like(sd)
        return mu + eps * sd

    def forward(self, x):
        mu, logvar = self.posterior_params(x)
        latent = self.sample_latents(mu, logvar)
        x_reconst = self.decoder(latent)
        return x_reconst, mu, logvar

class VAE_SSL(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_dim, n_hidden, latent_dim):
        super(VAE_SSL, self).__init__()
        self.xy_to_mu = MLP(x_dim + y_dim, hidden_dim, n_hidden, latent_dim)
        self.xy_to_logvar = MLP(x_dim + y_dim, hidden_dim, n_hidden, latent_dim)
        self.decoder = MLP(latent_dim + y_dim, hidden_dim, n_hidden, x_dim)

    def posterior_params(self, x, y):
        xy = torch.hstack((x, y))
        return self.xy_to_mu(xy), self.xy_to_logvar(xy)

    def sample_latents(self, mu, logvar, n_samples=1):
        sd = torch.exp(logvar / 2) # Same as sqrt(exp(logvar))
        eps = torch.randn((n_samples, len(sd))) if n_samples > 1 else torch.randn_like(sd)
        return mu + eps * sd

    def forward(self, x, y):
        mu, logvar = self.posterior_params(x, y)
        latent = self.sample_latents(mu, logvar)
        x_reconst = self.decoder(torch.hstack((latent, y)))
        return x_reconst, mu, logvar