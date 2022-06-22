import torch
import torch.nn as nn
from arch.mlp import MLP
from torch.autograd import Variable
from utils.const import EPSILON
from utils.ml import make_device

class ImageScalarSSVAE(nn.Module):
    def __init__(self, hidden_dim, n_hidden, latent_dim):
        super(ImageScalarSSVAE, self).__init__()
        self.latent_dim = latent_dim
        self.img_encoder = ImageEncoder(latent_dim)
        self.img_decoder = nn.Sequential(
            nn.Linear(latent_dim + 2, 64 * 4 * 4),
            Swish(),
            nn.Unflatten(1, (64, 4, 4)),
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, output_padding=1),
            Swish(),
            nn.ConvTranspose2d(32, 3, kernel_size=5, stride=2, output_padding=1))
        self.scalar_encoder = ScalarEncoder(hidden_dim, n_hidden, latent_dim)
        self.scalar_decoder = MLP(latent_dim + 2, hidden_dim, n_hidden, 1)
        self.experts = ProductOfExperts()
        self.device = make_device()

    def reparametrize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x_img, x_scalar, y):
        mu, logvar = self.infer(x_img, x_scalar, y)
        z = self.reparametrize(mu, logvar)
        img_reconst = self.img_decoder(torch.hstack((z, y)))
        scalar_reconst = self.scalar_decoder(torch.hstack((z, y)))
        return img_reconst, scalar_reconst, mu, logvar

    def infer(self, x_img, x_scalar, y):
        batch_size = x_img.size(0) if x_img is not None else x_scalar.size(0)
        # Prior
        mu, logvar = prior_expert((1, batch_size, self.latent_dim + 1), self.device)
        # Image
        image_mu, image_logvar = self.img_encoder(x_img, y)
        mu = torch.cat((mu, image_mu.unsqueeze(0)), dim=0)
        logvar = torch.cat((logvar, image_logvar.unsqueeze(0)), dim=0)
        # Scalar
        attrs_mu, attrs_logvar = self.scalar_encoder(x_scalar, y)
        mu = torch.cat((mu, attrs_mu.unsqueeze(0)), dim=0)
        logvar = torch.cat((logvar, attrs_logvar.unsqueeze(0)), dim=0)
        # Combine
        mu, logvar = self.experts(mu, logvar)
        return mu, logvar

class ImageEncoder(nn.Module):
    def __init__(self, latent_dim):
        super(ImageEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2),
            Swish(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2),
            Swish(),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 256),
            Swish(),
            nn.Linear(256, 2 * latent_dim))

    def forward(self, x, y):
        z = self.layers(x)
        return torch.hstack((z[:, :self.latent_dim], y)), torch.hstack((z[:, self.latent_dim:], y))

class ScalarEncoder(nn.Module):
    def __init__(self, hidden_dim, n_hidden, latent_dim):
        super(ScalarEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.layers = MLP(1, hidden_dim, n_hidden, 2 * latent_dim)

    def forward(self, x, y):
        z = self.layers(x)
        return torch.hstack((z[:, :self.latent_dim], y)), torch.hstack((z[:, self.latent_dim:], y))

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class ProductOfExperts(nn.Module):
    def forward(self, mu, logvar):
        var = torch.exp(logvar) + EPSILON
        T = 1. / var
        pd_mu = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var = 1. / torch.sum(T, dim=0)
        pd_logvar = torch.log(pd_var)
        return pd_mu, pd_logvar

def prior_expert(dim, device):
    mu, logvar = Variable(torch.zeros(dim)), Variable(torch.log(torch.ones(dim)))
    return mu.to(device), logvar.to(device)