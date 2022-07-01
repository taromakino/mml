import torch
import torch.nn as nn
from torch.autograd import Variable

class SSVAE(nn.Module):
    def __init__(self):
        super(SSVAE, self).__init__()
        self.encoder = Encoder()
        self.x0_decoder = Decoder()
        self.x1_decoder_mu = nn.Linear(256 + 1, 1)
        self.x1_decoder_logprec = nn.Linear(256 + 1, 1)

    def sample_z(self, mu, logvar):
        if self.training:
            sd = torch.exp(logvar / 2) # Same as sqrt(exp(logvar))
            eps = torch.randn_like(sd)
            return mu + eps * sd
        else:
            return mu

    def forward(self, x0, x1, y):
        mu, logvar = self.encoder(torch.hstack((x0, x1, y)))
        z = self.sample_z(mu, logvar)
        x0_reconst = self.x0_decoder(torch.hstack((z, y)))
        x1_mu = self.x1_decoder_mu(torch.hstack((z, y)))
        x1_logprec = self.x1_decoder_mu(torch.hstack((z, y)))
        return x0_reconst, x1_mu, x1_logprec, mu, logvar

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(784 * 3 + 2, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc31 = nn.Linear(512, 256)
        self.fc32 = nn.Linear(512, 256)
        self.swish = Swish()

    def forward(self, x):
        h = self.swish(self.fc1(x))
        h = self.swish(self.fc2(h))
        return self.fc31(h), self.fc32(h)

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(256 + 1, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 784 * 3)
        self.swish = Swish()

    def forward(self, z):
        h = self.swish(self.fc1(z))
        h = self.swish(self.fc2(h))
        h = self.swish(self.fc3(h))
        return self.fc4(h)

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)