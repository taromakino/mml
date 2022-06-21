from functools import partial
from utils.ml import *
from torch.optim import Adam
from toy_problem.data import make_data
from arch.two_scalar_vae import TwoScalarSSVAE

seed = 0
set_seed(seed)
rng = np.random.RandomState(seed)

n_examples = 100000
trainval_ratios = [0.8, 0.1]

uy_prior_spurious = np.array([
    [0.5, 0],
    [0, 0.5]])
uy_prior_nonspurious = np.array([
    [0.25, 0.25],
    [0.25, 0.25]])
sigma = 0.9

hidden_dim = 128
n_hidden = 3
latent_dim = 128

n_epochs = 50
batch_size = 500

x_spurious, y_spurious = make_data(rng, n_examples, uy_prior_spurious, sigma)
x_nonspurious, y_nonspurious = make_data(rng, n_examples, uy_prior_nonspurious, sigma)

(x_train_spurious, y_train_spurious), (x_val_spurious, y_val_spurious), (x_test_spurious, y_test_spurious) = \
    split_data(x_spurious, y_spurious, trainval_ratios)
(x_train_nonspurious, y_train_nonspurious), (x_val_nonspurious, y_val_nonspurious), (x_test_nonspurious, y_test_nonspurious) = \
    split_data(x_nonspurious, y_nonspurious, trainval_ratios)

x_train_union = np.vstack((x_train_spurious, x_train_nonspurious))
y_train_union = np.concatenate((y_train_spurious, y_train_nonspurious))

x_val_union = np.vstack((x_val_spurious, x_val_nonspurious))
y_val_union = np.concatenate((y_val_spurious, y_val_nonspurious))

x_test_union = np.vstack((x_test_spurious, x_test_nonspurious))
y_test_union = np.concatenate((y_test_spurious, y_test_nonspurious))

x0_train_spurious, x1_train_spurious = x_train_spurious[:, 0], x_train_spurious[:, 1]
x0_val_spurious, x1_val_spurious = x_val_spurious[:, 0], x_val_spurious[:, 1]
x0_test_spurious, x1_test_spurious = x_test_spurious[:, 0], x_test_spurious[:, 1]

x0_train_union, x1_train_union = x_train_union[:, 0], x_train_union[:, 1]
x0_val_union, x1_val_union = x_val_union[:, 0], x_val_union[:, 1]
x0_test_union, x1_test_union = x_test_union[:, 0], x_test_union[:, 1]

data_train_spurious = torch.tensor(x0_train_spurious)[:, None], torch.tensor(x1_train_spurious)[:, None], \
    torch.tensor(y_train_spurious)[:, None]
data_val_spurious = torch.tensor(x0_val_spurious)[:, None], torch.tensor(x1_val_spurious)[:, None], \
    torch.tensor(y_val_spurious)[:, None]
data_test_spurious = torch.tensor(x0_test_spurious)[:, None], torch.tensor(x1_test_spurious)[:, None], \
    torch.tensor(y_test_spurious)[:, None]

data_train_union = torch.tensor(x0_train_union)[:, None], torch.tensor(x1_train_union)[:, None], \
    torch.tensor(y_train_union)[:, None]
data_val_union = torch.tensor(x0_val_union)[:, None], torch.tensor(x1_val_union)[:, None], \
    torch.tensor(y_val_union)[:, None]
data_test_union = torch.tensor(x0_test_union)[:, None], torch.tensor(x1_test_union)[:, None], \
    torch.tensor(y_test_union)[:, None]

data_spurious = make_dataloaders(data_train_spurious, data_val_spurious, data_test_spurious, batch_size)
data_union = make_dataloaders(data_train_union, data_val_union, data_test_union, batch_size)

train_f = partial(train_epoch_vae, is_ssl=True)
eval_f = partial(eval_epoch_vae, is_ssl=True)

model_spurious = TwoScalarSSVAE(hidden_dim, n_hidden, latent_dim)
model_union = TwoScalarSSVAE(hidden_dim, n_hidden, latent_dim)
optimizer_spurious = Adam(model_spurious.parameters())
optimizer_union = Adam(model_union.parameters())

train_losses_spurious, val_losses_spurious, test_loss_spurious = train_eval_loop(*data_spurious, model_spurious,
    optimizer_spurious, train_f, eval_f, n_epochs)
train_losses_union, val_losses_union, test_loss_union = train_eval_loop(*data_union, model_union,
    optimizer_union, train_f, eval_f, n_epochs)

kldivs_spurious, kldivs_union = [], []
data_test_union = data_union[-1]
for x0_batch, x1_batch, y_batch in data_test_union:
    kldivs_spurious.append(vae_kldiv(*model_spurious.posterior_params(x0_batch, x1_batch, y_batch)).item())
    kldivs_union.append(vae_kldiv(*model_union.posterior_params(x0_batch, x1_batch, y_batch)).item())
print(f"spurious={np.mean(kldivs_spurious):.3f}, union={np.mean(kldivs_union):.3f}")