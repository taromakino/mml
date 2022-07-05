import datetime
import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from torch.utils.data import DataLoader, TensorDataset
from utils.file import write

class Swish(nn.Module):
    def forward(self, x):
        return swish(x)

def swish(x):
    return x * torch.sigmoid(x)

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def make_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def split_data(trainval_ratios, *arrays):
    n_train, n_val = [int(len(arrays[0]) * split_ratio) for split_ratio in trainval_ratios]
    arrays_train = [array[:n_train] for array in arrays]
    arrays_val = [array[n_train:n_train + n_val] for array in arrays]
    if sum(trainval_ratios) == 1:
        return arrays_train, arrays_val
    else:
        arrays_test = [array[n_train + n_val:] for array in arrays]
        return arrays_train, arrays_val, arrays_test

def to_1d_torch(*arrs):
    return [torch.tensor(arr)[:, None] for arr in arrs]

def make_dataloader(data_tuple, batch_size, is_train):
    return DataLoader(TensorDataset(*data_tuple), batch_size=batch_size, shuffle=is_train)

def make_dataloaders(data_train, data_val, data_test, batch_size):
    data_train = DataLoader(TensorDataset(*data_train), batch_size=batch_size, shuffle=True)
    data_val = DataLoader(TensorDataset(*data_val), batch_size=batch_size)
    data_test = DataLoader(TensorDataset(*data_test), batch_size=batch_size)
    return data_train, data_val, data_test

def train_epoch_vanilla(train_data, model, optimizer):
    device = make_device()
    model.train()
    loss_epoch = []
    for x_batch, y_batch in train_data:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        loss_batch = F.binary_cross_entropy(torch.sigmoid(model(x_batch)), y_batch)
        loss_batch.backward()
        loss_epoch.append(loss_batch.item())
        optimizer.step()
    return np.mean(loss_epoch)

def eval_epoch_vanilla(eval_data, model):
    device = make_device()
    model.eval()
    loss_epoch = []
    with torch.no_grad():
        for x_batch, y_batch in eval_data:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            loss_batch = F.binary_cross_entropy(torch.sigmoid(model(x_batch)), y_batch)
            loss_epoch.append(loss_batch.item())
    return np.mean(loss_epoch)

def posterior_kldiv(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

def gaussian_nll(x, mu, logprec):
    return 0.5 * torch.log(2 * torch.tensor(np.pi)) - 0.5 * logprec + 0.5 * torch.exp(logprec) * (x - mu) ** 2

def image_image_elbo(x0, x1, x0_reconst, x1_reconst, mu, logvar):
    x0_reconst_loss = F.binary_cross_entropy_with_logits(x0_reconst, x0, reduction="none").sum(dim=1)
    x1_reconst_loss = F.binary_cross_entropy_with_logits(x1_reconst, x1, reduction="none").sum(dim=1)
    return x0_reconst_loss, x1_reconst_loss, posterior_kldiv(mu, logvar)

def image_scalar_elbo(x0, x1, x0_reconst, x1_mu, x1_logprec, mu, logvar):
    x0_reconst_loss = F.binary_cross_entropy_with_logits(x0_reconst, x0, reduction="none").sum(dim=1)
    x1_reconst_loss = gaussian_nll(x1, x1_mu, x1_logprec)
    return x0_reconst_loss, x1_reconst_loss, posterior_kldiv(mu, logvar)

def scalar_scalar_elbo(x0, x1, x0_mu, x0_logprec, x1_mu, x1_logprec, mu, logvar):
    x0_reconst_loss = gaussian_nll(x0, x0_mu, x0_logprec)
    x1_reconst_loss = gaussian_nll(x1, x1_mu, x1_logprec)
    return x0_reconst_loss, x1_reconst_loss, posterior_kldiv(mu, logvar)

def train_epoch_vae(train_data, model, optimizer, epoch, loss_fn, n_anneal_epochs):
    n_batches = len(train_data)
    device = make_device()
    model.train()
    loss_epoch_x0, loss_epoch_x1, loss_epoch_kldiv, loss_epoch = [], [], [], []
    for batch_idx, (x0_batch, x1_batch, y_batch) in enumerate(train_data):
        x0_batch, x1_batch, y_batch = x0_batch.to(device), x1_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        model_output = model(x0_batch, x1_batch, y_batch)
        loss_batch_x0, loss_batch_x1, loss_batch_kldiv = loss_fn(x0_batch, x1_batch, *model_output)
        anneal_mult = (batch_idx + epoch * n_batches) / (n_anneal_epochs * n_batches) if epoch < n_anneal_epochs else 1
        loss_batch = (loss_batch_x0 + loss_batch_x1 + anneal_mult * loss_batch_kldiv).mean()
        loss_batch.backward()
        loss_epoch_x0.append(loss_batch_x0.mean().item())
        loss_epoch_x1.append(loss_batch_x1.mean().item())
        loss_epoch_kldiv.append(loss_batch_kldiv.mean().item())
        loss_epoch.append(loss_batch.item())
        optimizer.step()
    return np.mean(loss_epoch_x0), np.mean(loss_epoch_x1), np.mean(loss_epoch_kldiv), np.mean(loss_epoch)

def eval_epoch_vae(eval_data, model, loss_fn):
    device = make_device()
    model.eval()
    loss_epoch_x0, loss_epoch_x1, loss_epoch_kldiv, loss_epoch = [], [], [], []
    with torch.no_grad():
        for x0_batch, x1_batch, y_batch in eval_data:
            x0_batch, x1_batch, y_batch = x0_batch.to(device), x1_batch.to(device), y_batch.to(device)
            model_output = model(x0_batch, x1_batch, y_batch)
            loss_batch_x0, loss_batch_x1, loss_batch_kldiv = loss_fn(x0_batch, x1_batch, *model_output)
            loss_batch = (loss_batch_x0 + loss_batch_x1 + loss_batch_kldiv).mean()
            loss_epoch_x0.append(loss_batch_x0.mean().item())
            loss_epoch_x1.append(loss_batch_x1.mean().item())
            loss_epoch_kldiv.append(loss_batch_kldiv.mean().item())
            loss_epoch.append(loss_batch.item())
    return np.mean(loss_epoch_x0), np.mean(loss_epoch_x1), np.mean(loss_epoch_kldiv), np.mean(loss_epoch)

def train_eval_loop(data_train, data_val, model, optimizer, train_f, eval_f, dpath, n_epochs, n_early_stop_epochs):
    train_fpath = os.path.join(dpath, "train_summary.txt")
    val_fpath = os.path.join(dpath, "val_summary.txt")
    min_val_loss = np.inf
    optimal_weights = deepcopy(model.load_state_dict)
    optimal_epoch = 0
    for epoch in range(n_epochs):
        train_loss_x0, train_loss_x1, train_loss_kldiv, train_loss = train_f(data_train, model, optimizer, epoch)
        val_loss_x0, val_loss_x1, val_loss_kldiv, val_loss = eval_f(data_val, model)
        train_loss_str = f"{train_loss_x0:.6f}, {train_loss_x1:.6f}, {train_loss_kldiv:.6f}, {train_loss:.6f}"
        val_loss_str = f"{val_loss_x0:.6f}, {val_loss_x1:.6f}, {val_loss_kldiv:.6f}, {val_loss:.6f}"
        write(train_fpath, f"{timestamp()}, {epoch}, {train_loss_str}")
        write(val_fpath, f"{timestamp()}, {epoch}, {val_loss_str}")
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            optimal_weights = deepcopy(model.state_dict())
            optimal_epoch = epoch
            print(f"epoch={epoch}, optimal_epoch={optimal_epoch}")
        if epoch - optimal_epoch == n_early_stop_epochs:
            break
    torch.save(optimal_weights, os.path.join(dpath, "optimal_weights.pt"))
    model.load_state_dict(optimal_weights)

def timestamp():
    return datetime.datetime.now().strftime('%H:%M:%S')