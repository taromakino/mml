import torch
import torch.nn as nn
from torch.autograd import Variable

def conv_size(input_size, kernel_size, padding, stride):
    return int(((input_size - kernel_size + 2 * padding) / stride) + 1)

class SSVAE(nn.Module):
    def __init__(self, input0_dim, input1_dim, hidden_dims, latent_dim, target_dim):
        super(SSVAE, self).__init__()
        modules = []
        input_c, input_h, input_w = input0_dim
        input_size = input_h
        prev_dim = input_c
        for hidden_dim in hidden_dims[:-1]:
            modules.append(nn.Conv2d(prev_dim, hidden_dim, kernel_size=5, stride=2))
            input_size = conv_size(input_size, 5, 0, 2)
            modules.append(nn.BatchNorm2d(hidden_dim))
            modules.append(nn.LeakyReLU())
            prev_dim = hidden_dim
        modules.append(nn.Flatten())
        modules.append(nn.Linear(hidden_dims[-2] * input_size ** 2, hidden_dims[-1] - input1_dim - target_dim))
        modules.append(nn.LeakyReLU())
        self.x0_encoder = nn.Sequential(*modules)
        self.encoder_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.encoder_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        hidden_dims.reverse()
        modules = []
        modules.append(nn.Linear(latent_dim + 1, hidden_dims[1] * input_size ** 2))
        modules.append(nn.LeakyReLU())
        modules.append(nn.Unflatten(1, (hidden_dims[1], input_size, input_size)))
        prev_dim = hidden_dims[1]
        for hidden_dim in hidden_dims[2:]:
            modules.append(nn.ConvTranspose2d(prev_dim, hidden_dim, kernel_size=5, stride=2, output_padding=1))
            modules.append(nn.LeakyReLU())
            prev_dim = hidden_dim
        modules.append(nn.ConvTranspose2d(prev_dim, input_c, kernel_size=5, stride=2, output_padding=1))
        self.x0_decoder = nn.Sequential(*modules)
        self.x1_decoder_mu = nn.Linear(latent_dim + 1, input1_dim)
        self.x1_decoder_logprec = nn.Linear(latent_dim + 1, input1_dim)

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
        x1_mu = self.x1_decoder_mu(torch.hstack((z, y)))
        x1_logprec = self.x1_decoder_logprec(torch.hstack((z, y)))
        return x0_reconst, x1_mu, x1_logprec, mu, logvar