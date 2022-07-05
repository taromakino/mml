import torch
import torch.nn as nn
from utils.ml import Swish

class ImageEncoder(nn.Module):
    def __init__(self, image_dim, hidden_dim, latent_dim):
        super(ImageEncoder, self).__init__()
        self.fc1 = nn.Linear(image_dim + 1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc31 = nn.Linear(hidden_dim, latent_dim)
        self.fc32 = nn.Linear(hidden_dim, latent_dim)
        self.swish = Swish()

    def forward(self, x):
        h = self.swish(self.fc1(x))
        h = self.swish(self.fc2(h))
        return self.fc31(h), self.fc32(h)

class ImageDecoder(nn.Module):
    def __init__(self, image_dim, hidden_dim, latent_dim):
        super(ImageDecoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim + 1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, image_dim)
        self.swish = Swish()

    def forward(self, z):
        h = self.swish(self.fc1(z))
        h = self.swish(self.fc2(h))
        h = self.swish(self.fc3(h))
        return self.fc4(h)

class SemiSupervisedVae(nn.Module):
    def __init__(self, x0_dim, x1_dim, hidden_dim, latent_dim):
        super(SemiSupervisedVae, self).__init__()
        self.encoder = ImageEncoder(x0_dim + x1_dim, hidden_dim, latent_dim)
        self.x0_decoder = ImageDecoder(x0_dim, hidden_dim, latent_dim)
        self.x1_decoder = ImageDecoder(x1_dim, hidden_dim, latent_dim)

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
        x0_reconst = self.x0_decoder(torch.hstack((z, y)))
        x1_reconst = self.x0_decoder(torch.hstack((z, y)))
        return x0_reconst, x1_reconst, mu, logvar