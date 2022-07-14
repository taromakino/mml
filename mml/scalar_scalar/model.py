import torch
import torch.nn as nn
from utils.ml import swish

class ScalarEncoder(nn.Module):
    def __init__(self, hidden_dim, latent_dim):
        super(ScalarEncoder, self).__init__()
        self.fc1 = nn.Linear(3, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc31 = nn.Linear(hidden_dim, latent_dim)
        self.fc32 = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = swish(self.fc1(x))
        h = swish(self.fc2(h))
        return self.fc31(h), self.fc32(h)

class ScalarDecoder(nn.Module):
    def __init__(self, hidden_dim, latent_dim):
        super(ScalarDecoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim + 1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = swish(self.fc1(x))
        h = swish(self.fc2(h))
        return self.fc3(h)

class SemiSupervisedVae(nn.Module):
    def __init__(self, hidden_dim, latent_dim):
        super(SemiSupervisedVae, self).__init__()
        self.encoder = ScalarEncoder(hidden_dim, latent_dim)
        self.x0_decoder_mu = ScalarDecoder(hidden_dim, latent_dim)
        self.x0_decoder_logprec = ScalarDecoder(hidden_dim, latent_dim)
        self.x1_decoder_mu = ScalarDecoder(hidden_dim, latent_dim)
        self.x1_decoder_logprec = ScalarDecoder(hidden_dim, latent_dim)

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