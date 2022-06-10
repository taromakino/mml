import os
from functools import partial
from utils.ml import *
from torch.optim import Adam
from data import make_data
from nn.vae import VAE_SSL

seed = 0
set_seed(seed)
rng = np.random.RandomState(seed)

n_examples = 100000
n_epochs = 50
batch_size = 500
split_ratios = [0.8, 0.1, 0.1]

uy_prior_spurious = np.array([
    [0.5, 0],
    [0, 0.5]])
uy_prior_nonspurious = np.array([
    [0.25, 0.25],
    [0.25, 0.25]])
sigma = 0.9

x_dim = 2
y_dim = 1
hidden_dim = 128
n_hidden = 3
latent_dim = 128

model_spurious = VAE_SSL(x_dim, y_dim, hidden_dim, n_hidden, latent_dim)
model_union = VAE_SSL(x_dim, y_dim, hidden_dim, n_hidden, latent_dim)
optimizer_spurious = Adam(model_spurious.parameters())
optimizer_union = Adam(model_union.parameters())

x_spurious, y_spurious = make_data(rng, n_examples, uy_prior_spurious, sigma)
x_nonspurious, y_nonspurious = make_data(rng, n_examples, uy_prior_nonspurious, sigma)

(x_train_spurious, y_train_spurious), (x_val_spurious, y_val_spurious), (x_test_spurious, y_test_spurious) = \
    split_data(torch.tensor(x_spurious), torch.tensor(y_spurious[:, None]), split_ratios)
(x_train_nonspurious, y_train_nonspurious), (x_val_nonspurious, y_val_nonspurious), (x_test_nonspurious, y_test_nonspurious) = \
    split_data(torch.tensor(x_nonspurious), torch.tensor(y_nonspurious[:, None]), split_ratios)

x_train_union, y_train_union = torch.vstack((x_train_spurious, x_train_nonspurious)), torch.cat((y_train_spurious,
    y_train_nonspurious))
x_val_union, y_val_union = torch.vstack((x_val_spurious, x_val_nonspurious)), torch.cat((y_val_spurious,
    y_val_nonspurious))
x_test_union, y_test_union = torch.vstack((x_test_spurious, x_test_nonspurious)), torch.cat((y_test_spurious,
    y_test_nonspurious))

data_spurious = make_dataloaders(x_train_spurious, y_train_spurious, x_val_spurious, y_val_spurious, x_test_spurious,
    y_test_spurious, batch_size)
data_union = make_dataloaders(x_train_union, y_train_union, x_val_union, y_val_union, x_test_union,
    y_test_union, batch_size)

train_f = partial(train_epoch_vae, is_ssl=True)
eval_f = partial(eval_epoch_vae, is_ssl=True)

train_losses_spurious, val_losses_spurious, test_loss_spurious = train_eval_loop(*data_spurious, model_spurious,
    optimizer_spurious, train_f, eval_f, n_epochs)
train_losses_union, val_losses_union, test_loss_union = train_eval_loop(*data_union, model_union,
    optimizer_union, train_f, eval_f, n_epochs)

torch.save(model_spurious.state_dict(), os.path.join("results", "model_spurious.pt"))
torch.save(model_union.state_dict(), os.path.join("results", "model_union.pt"))

kldivs_spurious, kldivs_union = [], []
data_test_union = data_union[-1]
for x_batch, y_batch in data_test_union:
    kldivs_spurious.append(vae_kldiv(*model_spurious.posterior_params(x_batch, y_batch)).item())
    kldivs_union.append(vae_kldiv(*model_union.posterior_params(x_batch, y_batch)).item())
print(f"spurious={np.mean(kldivs_spurious):.3f}, union={np.mean(kldivs_union):.3f}")