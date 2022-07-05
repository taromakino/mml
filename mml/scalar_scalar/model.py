import torch
import torch.nn as nn
from utils.ml import swish

class Encoder(nn.Module):
    def __init__(self, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(3, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)
        self.fc22 = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = swish(self.fc1(x))
        return self.fc21(h), self.fc22(h)

class SemiSupervisedVae(nn.Module):
    def __init__(self, hidden_dim, latent_dim):
        super(SemiSupervisedVae, self).__init__()
        self.encoder = Encoder(hidden_dim, latent_dim)
        self.x0_decoder_mu = nn.Linear(latent_dim + 1, 1)
        self.x0_decoder_logprec = nn.Linear(latent_dim + 1, 1)
        self.x1_decoder_mu = nn.Linear(latent_dim + 1, 1)
        self.x1_decoder_logprec = nn.Linear(latent_dim + 1, 1)

    def encode(self, x0, x1, y):
        return self.encoder(torch.hstack((x0, x1, y)))

    def sample_z(self, mu, logvar):
        if self.training:
            sd = torch.exp(logvar / 2) # Same as sqrt(exp(logvar))
            eps = torch.randn_like(sd)
            return mu + eps * sd
        else:
            return mu

    def forward(self, x0, x1, y):
        mu, logvar = self.encode(x0, x1, y)
        z = self.sample_z(mu, logvar)
        x0_mu = self.x0_decoder_mu(torch.hstack((z, y)))
        x0_logprec = self.x0_decoder_logprec(torch.hstack((z, y)))
        x1_mu = self.x1_decoder_mu(torch.hstack((z, y)))
        x1_logprec = self.x1_decoder_logprec(torch.hstack((z, y)))
        return x0_mu, x0_logprec, x1_mu, x1_logprec, mu, logvar