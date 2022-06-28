import torch
import torch.nn as nn
from arch.mlp import MLP
from torch.autograd import Variable

class SSVAE(nn.Module):
    def __init__(self, x1_dim, h_dim, z_dim, y_dim):
        super(SSVAE, self).__init__()
        self.latent_dim = z_dim
        self.x0_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, h_dim - x1_dim - y_dim),
            nn.LeakyReLU())
        self.encoder_mu = nn.Linear(h_dim, z_dim)
        self.encoder_logvar = nn.Linear(h_dim, z_dim)
        self.x0_decoder = nn.Sequential(
            nn.Linear(z_dim + 1, 64 * 4 * 4),
            nn.LeakyReLU(),
            nn.Unflatten(1, (64, 4, 4)),
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, output_padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=5, stride=2, output_padding=1))
        self.x1_decoder = nn.Linear(z_dim + 1, x1_dim)

    def posterior_params(self, x0, x1, y):
        xy = torch.hstack((self.x0_encoder(x0), x1, y))
        return self.encoder_mu(xy), self.encoder_logvar(xy)

    def sample_z(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x0, x1, y):
        mu, logvar = self.posterior_params(x0, x1, y)
        z = self.sample_z(mu, logvar)
        x0_reconst = self.x0_decoder(torch.hstack((z, y)))
        x1_reconst = self.x1_decoder(torch.hstack((z, y)))
        return x0_reconst, x1_reconst, mu, logvar