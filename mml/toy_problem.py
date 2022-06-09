import matplotlib.pyplot as plt
import numpy as np
import os
import random
import torch
import torch.nn.functional as F
from const import *
from mlp import MLP
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from toy_problem_data import make_data
from vae import VAE, VAE_SSL

plt.rcParams.update({'font.size': 14})

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def vae_loss(x_reconst, x, mu, logvar):
    reconst_loss = F.mse_loss(x_reconst, x, reduction='sum')
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return reconst_loss + kl_div

def train(epoch, train_data, model, optimizer):
    model.train()
    loss_epoch = []
    for x_batch, y_batch in train_data:
        optimizer.zero_grad()
        loss_batch = F.binary_cross_entropy(torch.sigmoid(model(x_batch)), y_batch)
        loss_batch.backward()
        loss_epoch.append(loss_batch.item())
        optimizer.step()
    print(f"Epoch={epoch} Loss={np.mean(loss_epoch):.6f}")

def train_vae(epoch, train_data, model, optimizer, is_ssl):
    model.train()
    loss_epoch = []
    for x_batch, y_batch in train_data:
        optimizer.zero_grad()
        x_reconst, mu, logvar = model(x_batch, y_batch) if is_ssl else model(x_batch)
        loss_batch = vae_loss(x_reconst, x_batch, mu, logvar)
        loss_batch.backward()
        loss_epoch.append(loss_batch.item())
        optimizer.step()
    print(f"Epoch={epoch} Loss={np.mean(loss_epoch):.6f}")

def plot_binary_scatter(ax, x, y):
    neg_idxs, pos_idxs = np.where(y == 0)[0], np.where(y == 1)[0]
    ax.scatter(x[neg_idxs, 0], x[neg_idxs, 1], s=0.1, color="red")
    ax.scatter(x[pos_idxs, 0], x[pos_idxs, 1], s=0.1, color="blue")

seed = 2
set_seed(seed)
rng = np.random.RandomState(seed)

n_examples = 100000
n_epochs = 50
batch_size = 500
uy_prior_train = np.array([
    [0.25, 0.25],
    [0.25, 0.25]])
sigma = 0.9

x_dim = 2
y_dim = 1
hidden_dim = 128
n_hidden = 3
latent_dim = 128

mlp = MLP(x_dim, hidden_dim, n_hidden, y_dim)
vae = VAE(x_dim, hidden_dim, n_hidden, latent_dim)
vae_ssl = VAE_SSL(x_dim, y_dim, hidden_dim, n_hidden, latent_dim)
critic = MLP(latent_dim, hidden_dim, n_hidden, y_dim)

optim_mlp = Adam(mlp.parameters())
optim_vae = Adam(vae.parameters())
optim_vae_ssl = Adam(vae_ssl.parameters())
optim_critic = Adam(critic.parameters())

x_train, y_train = make_data(rng, n_examples, uy_prior_train, sigma)
x_mean, x_sd = x_train.mean(0), x_train.std(0)
x_train = (x_train - x_mean) / x_sd
x_train, y_train = torch.tensor(x_train), torch.tensor(y_train)
train_data = DataLoader(TensorDataset(x_train, y_train[:, None]), batch_size=batch_size, shuffle=True)

for epoch in range(n_epochs):
    train(epoch, train_data, mlp, optim_mlp)
    train_vae(epoch, train_data, vae, optim_vae, False)
    train_vae(epoch, train_data, vae_ssl, optim_vae_ssl, True)
torch.save(mlp.state_dict(), os.path.join("results", "mlp.pt"))
torch.save(vae.state_dict(), os.path.join("results", "vae.pt"))
torch.save(vae_ssl.state_dict(), os.path.join("results", "vae_ssl.pt"))

# mlp.load_state_dict(torch.load(os.path.join("results", "mlp.pt")))
# vae.load_state_dict(torch.load(os.path.join("results", "vae.pt")))
# vae_ssl.load_state_dict(torch.load(os.path.join("results", "vae_ssl.pt")))

mlp.eval()
vae.eval()
vae_ssl.eval()

pred = torch.sigmoid(mlp(x_train))
pred = (pred > 0.5).squeeze().int()
print(f"{(pred == y_train.squeeze()).float().mean():.3f}")

u_train, pseudo_y_train = [], []
with torch.no_grad():
    for x_batch, y_batch in train_data:
        u_neg = vae.sample_latents(*vae.posterior_params(x_batch))
        u_pos = vae_ssl.sample_latents(*vae_ssl.posterior_params(x_batch, y_batch))
        u_batch = torch.vstack((u_neg, u_pos))
        pseudo_y_batch = torch.cat((torch.zeros(len(u_neg)), torch.ones(len(u_pos))))
        u_train.append(u_batch)
        pseudo_y_train.append(pseudo_y_batch)
u_train = torch.vstack(u_train)
u_mean, u_sd = u_train.mean(0), u_train.std(0)
u_train = (u_train - u_mean) / u_sd
pseudo_y_train = torch.cat(pseudo_y_train)
critic_data = DataLoader(TensorDataset(u_train, pseudo_y_train[:, None]), batch_size=batch_size, shuffle=True)

for epoch in range(n_epochs):
    train(epoch, critic_data, critic, optim_critic)
torch.save(critic.state_dict(), os.path.join("results", "critic.pt"))

# critic.load_state_dict(torch.load(os.path.join("results", "critic.pt")))

critic.eval()

pred = torch.sigmoid(critic(u_train))
pred = (pred > 0.5).squeeze().int()
print(f"{(pred == pseudo_y_train.squeeze()).float().mean():.3f}")

uy_prior_unlabeled = np.array([
    [0.25, 0.25],
    [0.25, 0.25]])
n_unlabeled_examples = 10000
x_unlabeled, y_unlabeled = make_data(rng, n_unlabeled_examples, uy_prior_unlabeled, sigma)
x_unlabeled = (x_unlabeled - x_mean) / x_sd
x_unlabeled, y_unlabeled = torch.tensor(x_unlabeled), torch.tensor(y_unlabeled)
unlabeled_data = DataLoader(TensorDataset(x_unlabeled, y_unlabeled[:, None]))

n_samples = 500
n_subset = n_unlabeled_examples // 2
uy_mutual_info = []
with torch.no_grad():
    for x_elem, y_elem in unlabeled_data:
        # y_elem = torch.bernoulli(torch.sigmoid(mlp(x_elem)))
        u_neg = vae.sample_latents(*vae.posterior_params(x_elem), n_samples)
        u_pos = vae_ssl.sample_latents(*vae_ssl.posterior_params(x_elem, y_elem), n_samples)
        u_batch = torch.vstack((u_neg, u_pos))
        pred = torch.clip(torch.sigmoid(critic(u_batch)), EPSILON, 1 - EPSILON)
        uy_mutual_info.append((torch.log(pred) - torch.log(1 - pred)).mean().item())
sorted_idxs = np.argsort(uy_mutual_info)
min_idxs = sorted_idxs[:n_subset]
max_idxs = sorted_idxs[-n_subset:]
x_unlabeled_unnorm = (x_unlabeled * x_sd) + x_mean
x_min_subset = x_unlabeled_unnorm[min_idxs]
x_max_subset = x_unlabeled_unnorm[max_idxs]
y_min_subset = y_unlabeled[min_idxs]
y_max_subset = y_unlabeled[max_idxs]
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
plot_binary_scatter(axes[0], x_min_subset, y_min_subset)
plot_binary_scatter(axes[1], x_max_subset, y_max_subset)
x_lim = (-4, 6)
y_lim = (-5, 5)
for ax in axes:
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    ax.grid(True)
axes[0].set_title("Lowest I(U, Y | x, x')")
axes[1].set_title("Highest I(U, Y | x, x')")
plt.tight_layout()