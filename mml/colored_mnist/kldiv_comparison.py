from functools import partial
from utils.ml import *
from torch.optim import Adam
from colored_mnist.data import make_data
from arch.mlp_vae import SSVAE
from argparse import ArgumentParser

def split_data(x0, x1, y, trainval_ratios):
    assert sum(trainval_ratios) == 1
    x0 /= 255.
    n_train, n_val = [int(len(x0) * split_ratio) for split_ratio in trainval_ratios]
    x0_train, x1_train, y_train = x0[:n_train], x1[:n_train], y[:n_train]
    x0_val, x1_val, y_val = x0[n_train:n_train + n_val], x1[n_train:n_train + n_val], y[n_train:n_train + n_val]
    x1_mean, x1_sd = x1_train.mean(0), x1_train.std(0)
    x1_train = (x1_train - x1_mean) / x1_sd
    x1_val = (x1_val - x1_mean) / x1_sd
    if sum(trainval_ratios) == 1:
        return (x0_train, x1_train, y_train), (x0_val, x1_val, y_val)
    else:
        x0_test, x1_test, y_test = x0[n_train + n_val:], x1[n_train + n_val:], y[n_train + n_val:]
        x1_test = (x1_test - x1_mean) / x1_sd
        return (x0_train, x1_train, y_train), (x0_val, x1_val, y_val), (x0_test, x1_test, y_test)

def main(args):
    set_seed(args.seed)

    x0_trainval_det, x1_trainval_det, y_trainval_det = make_data(True, 0, args.sigma)
    x0_trainval_nondet, x1_trainval_nondet, y_trainval_nondet = make_data(True, args.p_flip_color, args.sigma)

    x0_test_det, x1_test_det, y_test_det = make_data(False, 0, args.sigma)
    x0_test_nondet, x1_test_nondet, y_test_nondet = make_data(False, args.p_flip_color, args.sigma)

    (x0_train_det, x1_train_det, y_train_det), (x0_val_det, x1_val_det, y_val_det) = \
        split_data(x0_trainval_det, x1_trainval_det, y_trainval_det, args.trainval_ratios)
    (x0_train_nondet, x1_train_nondet, y_train_nondet), (x0_val_nondet, x1_val_nondet, y_val_nondet) = \
        split_data(x0_trainval_nondet, x1_trainval_nondet, y_trainval_nondet, args.trainval_ratios)

    x0_train_det, x1_train_det = torch.tensor(x0_train_det), torch.tensor(x1_train_det)[:, None]
    x0_val_det, x1_val_det = torch.tensor(x0_val_det), torch.tensor(x1_val_det)[:, None]
    x0_test_det, x1_test_det = torch.tensor(x0_test_det), torch.tensor(x1_test_det)[:, None]
    y_train_det, y_val_det, y_test_det = torch.tensor(y_train_det)[:, None], torch.tensor(y_val_det)[:, None], \
        torch.tensor(y_test_det)[:, None]

    x0_train_nondet, x1_train_nondet = torch.tensor(x0_train_nondet), torch.tensor(x1_train_nondet)[:, None]
    x0_val_nondet, x1_val_nondet = torch.tensor(x0_val_nondet), torch.tensor(x1_val_nondet)[:, None]
    x0_test_nondet, x1_test_nondet = torch.tensor(x0_test_nondet), torch.tensor(x1_test_nondet)[:, None]
    y_train_nondet, y_val_nondet, y_test_nondet = torch.tensor(y_train_nondet)[:, None], torch.tensor(y_val_nondet)[:,None], \
        torch.tensor(y_test_nondet)[:, None]

    x0_train_union, x1_train_union, y_train_union = \
        torch.vstack((x0_train_det, x0_train_nondet)), \
        torch.vstack((x1_train_det, x1_train_nondet)), \
        torch.vstack((y_train_det, y_train_nondet))
    x0_val_union, x1_val_union, y_val_union = \
        torch.vstack((x0_val_det, x0_val_nondet)), \
        torch.vstack((x1_val_det, x1_val_nondet)), \
        torch.vstack((y_val_det, y_val_nondet))
    x0_test_union, x1_test_union, y_test_union = \
        torch.vstack((x0_test_det, x0_test_nondet)), \
        torch.vstack((x1_test_det, x1_test_nondet)), \
        torch.vstack((y_test_det, y_test_nondet))

    data_det = make_dataloaders(
        (x0_train_det, x1_train_det, y_train_det),
        (x0_val_det, x1_val_det, y_val_det),
        (x0_test_det, x1_test_det, y_test_det), args.batch_size)
    data_union = make_dataloaders(
        (x0_train_union, x1_train_union, y_train_union),
        (x0_val_union, x1_val_union, y_val_union),
        (x0_test_union, x1_test_union, y_test_union), args.batch_size)

    train_f = partial(train_epoch_vae, loss_mults=args.loss_mults, n_anneal_epochs=args.n_anneal_epochs)

    x0_dim = x0_train_det.shape[1]
    x1_dim = x1_train_det.shape[1]
    y_dim = y_train_det.shape[1]

    model_det = SSVAE(x0_dim, x1_dim, args.hidden_dims, args.latent_dim, y_dim)
    model_union = SSVAE(x0_dim, x1_dim, args.hidden_dims, args.latent_dim, y_dim)
    model_det.to(make_device())
    model_union.to(make_device())
    optimizer_det = Adam(model_det.parameters(), lr=args.lr)
    optimizer_union = Adam(model_union.parameters(), lr=args.lr)

    dpath_spurious = os.path.join(args.dpath, "spurious")
    dpath_union = os.path.join(args.dpath, "union")
    os.makedirs(dpath_spurious, exist_ok=True)
    os.makedirs(dpath_union, exist_ok=True)

    train_eval_loop(*data_det, model_det, optimizer_det, train_f, eval_epoch_vae, dpath_spurious, args.n_epochs)
    train_eval_loop(*data_union, model_union, optimizer_union, train_f, eval_epoch_vae, dpath_union, args.n_epochs)

    device = make_device()
    kldivs_det, kldivs_union = [], []
    data_test_union = data_union[-1]
    model_det.eval()
    model_union.eval()
    for x0_batch, x1_batch, y_batch in data_test_union:
        x0_batch, x1_batch, y_batch = x0_batch.to(device), x1_batch.to(device), y_batch.to(device)
        kldivs_det.append(posterior_kldiv(*model_det.posterior_params(x0_batch, x1_batch, y_batch)).item())
        kldivs_union.append(posterior_kldiv(*model_union.posterior_params(x0_batch, x1_batch, y_batch)).item())
    print(f"det={np.mean(kldivs_det):.3f}, union={np.mean(kldivs_union):.3f}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dpath", type=str, default="results")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--p-flip-color", type=float, default=0.5)
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--trainval-ratios", nargs="+", type=float, default=[0.8, 0.2])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--loss-mults", nargs="+", type=float, default=[1, 1])
    parser.add_argument("--n-epochs", type=int, default=100)
    parser.add_argument("--n-anneal-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[256, 256, 256])
    parser.add_argument("--latent-dim", type=int, default=256)
    main(parser.parse_args())