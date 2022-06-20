import numpy as np
import random
import torch
import torch.nn.functional as F
from copy import deepcopy
from torch.utils.data import DataLoader, TensorDataset

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def split_data(x, y, trainval_ratios):
    n_train, n_val = [int(len(x) * trainval_ratio) for trainval_ratio in trainval_ratios]
    x_train, y_train = x[:n_train], y[:n_train]
    x_val, y_val = x[n_train:n_train + n_val], y[n_train:n_train + n_val]
    x_mean, x_sd = x_train.mean(0), x_train.std(0)
    x_train = (x_train - x_mean) / x_sd
    x_val = (x_val - x_mean) / x_sd
    if sum(trainval_ratios) == 1:
        return (x_train, y_train), (x_val, y_val)
    else:
        x_test, y_test = x[n_train + n_val:], y[n_train + n_val:]
        x_test = (x_test - x_mean) / x_sd
        return (x_train, y_train), (x_val, y_val), (x_test, y_test)

def make_dataloaders(data_train, data_val, data_test, batch_size):
    data_train = DataLoader(TensorDataset(*data_train), batch_size=batch_size, shuffle=True)
    data_val = DataLoader(TensorDataset(*data_val), batch_size=batch_size)
    data_test = DataLoader(TensorDataset(*data_test), batch_size=batch_size)
    return data_train, data_val, data_test

def train_epoch_vanilla(train_data, model, optimizer):
    model.train()
    loss_epoch = []
    for x_batch, y_batch in train_data:
        optimizer.zero_grad()
        loss_batch = F.binary_cross_entropy(torch.sigmoid(model(x_batch)), y_batch)
        loss_batch.backward()
        loss_epoch.append(loss_batch.item())
        optimizer.step()
    return np.mean(loss_epoch)

def eval_epoch_vanilla(eval_data, model):
    model.eval()
    loss_epoch = []
    with torch.no_grad():
        for x_batch, y_batch in eval_data:
            loss_batch = F.binary_cross_entropy(torch.sigmoid(model(x_batch)), y_batch)
            loss_epoch.append(loss_batch.item())
    return np.mean(loss_epoch)

def vae_kldiv(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

def vae_loss(x0, x1, x0_reconst, x1_reconst, mu, logvar):
    x0_reconst_loss = F.mse_loss(x0_reconst, x0, reduction='sum')
    x1_reconst_loss = F.mse_loss(x1_reconst, x1, reduction='sum')
    return x0_reconst_loss + x1_reconst_loss + vae_kldiv(mu, logvar)

def train_epoch_vae(train_data, model, optimizer, is_ssl):
    model.train()
    loss_epoch = []
    for x0_batch, x1_batch, y_batch in train_data:
        optimizer.zero_grad()
        x0_reconst, x1_reconst, mu, logvar = model(x0_batch, x1_batch, y_batch) if is_ssl else model(x0_batch, x1_batch)
        loss_batch = vae_loss(x0_batch, x1_batch, x0_reconst, x1_reconst, mu, logvar)
        loss_batch.backward()
        loss_epoch.append(loss_batch.item())
        optimizer.step()
    return np.mean(loss_epoch)

def eval_epoch_vae(eval_data, model, is_ssl):
    model.eval()
    loss_epoch = []
    with torch.no_grad():
        for x0_batch, x1_batch, y_batch in eval_data:
            x0_reconst, x1_reconst, mu, logvar = model(x0_batch, x1_batch, y_batch) if is_ssl else model(x0_batch, x1_batch)
            loss_batch = vae_loss(x0_batch, x1_batch, x0_reconst, x1_reconst, mu, logvar)
            loss_epoch.append(loss_batch.item())
    return np.mean(loss_epoch)

def train_eval_loop(data_train, data_val, data_test, model, optimizer, train_f, eval_f, n_epochs):
    min_val_loss = np.inf
    optim_weights = deepcopy(model.load_state_dict)
    train_losses, val_losses = [], []
    for epoch in range(n_epochs):
        train_losses.append(train_f(data_train, model, optimizer))
        val_loss = eval_f(data_val, model)
        val_losses.append(val_loss)
        if val_loss < min_val_loss:
            optim_weights = deepcopy(model.state_dict())
    model.load_state_dict(optim_weights)
    test_loss = eval_f(data_test, model)
    return train_losses, val_losses, test_loss