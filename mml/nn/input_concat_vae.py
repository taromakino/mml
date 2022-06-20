import torch
import torch.nn as nn
from nn.mlp import MLP

import inspect

class InputConcatVAE(nn.Module):
    def __init__(self, x0_dim, x1_dim, hidden_dim, n_hidden, latent_dim):
        super(InputConcatVAE, self).__init__()
        self.x_to_mu = MLP(x0_dim + x1_dim, hidden_dim, n_hidden, latent_dim)
        self.x_to_logvar = MLP(x0_dim + x1_dim, hidden_dim, n_hidden, latent_dim)
        self.x0_decoder = MLP(latent_dim, hidden_dim, n_hidden, x0_dim)
        self.x1_decoder = MLP(latent_dim, hidden_dim, n_hidden, x1_dim)

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

class InputConcatSSVAE(nn.Module):
    def __init__(self, x0_dim, x1_dim, y_dim, hidden_dim, n_hidden, latent_dim):
        super(InputConcatSSVAE, self).__init__()
        print(inspect.getsource(MLP))
        self.xy_to_mu = MLP(x0_dim + x1_dim + y_dim, hidden_dim, n_hidden, latent_dim)
        self.xy_to_logvar = MLP(x0_dim + x1_dim + y_dim, hidden_dim, n_hidden, latent_dim)
        self.x0_decoder = MLP(latent_dim + y_dim, hidden_dim, n_hidden, x0_dim)
        self.x1_decoder = MLP(latent_dim + y_dim, hidden_dim, n_hidden, x1_dim)

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