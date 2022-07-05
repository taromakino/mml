from argparse import ArgumentParser
from functools import partial
from utils.ml import *
from torch.optim import Adam
from scalar_scalar.data import make_data
from scalar_scalar.model import SemiSupervisedVae

def main(args):
    set_seed(args.seed)
    rng = np.random.RandomState(args.seed)
    device = make_device()

    n_examples = 10000
    trainval_ratios = [0.8, 0.1]

    uy_prior_det = np.array([
        [0.5, 0],
        [0, 0.5]])
    uy_prior_nondet = np.array([
        [0.25, 0.25],
        [0.25, 0.25]])
    sigma = 0.9

    # Make numpy data
    x_det, y_det = make_data(rng, n_examples, uy_prior_det, sigma)
    x_nondet, y_nondet = make_data(rng, n_examples, uy_prior_nondet, sigma)

    # Split train/val/test
    (x_train_det, y_train_det), (x_val_det, y_val_det), (x_test_det, y_test_det) = \
        split_data(trainval_ratios, x_det, y_det)
    (x_train_nondet, y_train_nondet), (x_val_nondet, y_val_nondet), (x_test_nondet, y_test_nondet) = \
        split_data(trainval_ratios, x_nondet, y_nondet)

    # Normalize
    x_mean_det, x_sd_det = x_train_det.mean(0), x_train_det.std(0)
    x_mean_nondet, x_sd_nondet = x_train_nondet.mean(0), x_train_nondet.std(0)

    x_train_det = (x_train_det - x_mean_det) / x_sd_det
    x_val_det = (x_val_det - x_mean_det) / x_sd_det
    x_test_det = (x_test_det - x_mean_det) / x_sd_det

    x_train_nondet = (x_train_nondet - x_mean_nondet) / x_sd_nondet
    x_val_nondet = (x_val_nondet - x_mean_nondet) / x_sd_nondet
    x_test_nondet = (x_test_nondet - x_mean_nondet) / x_sd_nondet

    # To torch

    x0_train_det, x1_train_det = to_1d_torch(x_train_det[:, 0], x_train_det[:, 1])
    x0_val_det, x1_val_det = to_1d_torch(x_val_det[:, 0], x_val_det[:, 1])
    x0_test_det, x1_test_det = to_1d_torch(x_test_det[:, 0], x_test_det[:, 1])
    y_train_det, y_val_det, y_test_det = to_1d_torch(y_train_det, y_val_det, y_test_det)

    x0_train_nondet, x1_train_nondet = to_1d_torch(x_train_nondet[:, 0], x_train_nondet[:, 1])
    x0_val_nondet, x1_val_nondet = to_1d_torch(x_val_nondet[:, 0], x_val_nondet[:, 1])
    x0_test_nondet, x1_test_nondet = to_1d_torch(x_test_nondet[:, 0], x_test_nondet[:, 1])
    y_train_nondet, y_val_nondet, y_test_nondet = to_1d_torch(y_train_nondet, y_val_nondet, y_test_nondet)

    # Make union
    x0_train_union, x1_train_union, y_train_union = \
        torch.vstack((x0_train_det, x0_train_nondet)), \
        torch.vstack((x1_train_det, x1_train_nondet)), \
        torch.vstack((y_train_det, y_train_nondet))
    x0_val_union, x1_val_union, y_val_union = \
        torch.vstack((x0_val_det, x0_val_nondet)), \
        torch.vstack((x1_val_det, x1_val_nondet)), \
        torch.vstack((y_val_det, y_val_nondet))

    train_subset_idxs = rng.choice(np.arange(len(x0_train_union)), len(x0_train_det), replace=False)
    val_subset_idxs = rng.choice(np.arange(len(x0_val_union)), len(x0_val_det), replace=False)

    x0_train_union, x1_train_union, y_train_union = x0_train_union[train_subset_idxs], x1_train_union[
        train_subset_idxs], y_train_union[train_subset_idxs]
    x0_val_union, x1_val_union, y_val_union = x0_val_union[val_subset_idxs], x1_val_union[val_subset_idxs], \
        y_val_union[val_subset_idxs]

    # Run experiment
    data_train_det = make_dataloader((x0_train_det, x1_train_det, y_train_det), args.batch_size, True)
    data_val_det = make_dataloader((x0_val_det, x1_val_det, y_val_det), args.batch_size, False)

    data_train_union = make_dataloader((x0_train_union, x1_train_union, y_train_union), args.batch_size, True)
    data_val_union = make_dataloader((x0_val_union, x1_val_union, y_val_union), args.batch_size, False)

    data_test_det = make_dataloader((x0_test_det, x1_test_det, y_test_det), args.batch_size, False)
    data_test_nondet = make_dataloader((x0_test_nondet, x1_test_nondet, y_test_nondet), args.batch_size, False)

    train_f = partial(train_epoch_vae, loss_fn=scalar_scalar_elbo, n_anneal_epochs=args.n_anneal_epochs)
    eval_f = partial(eval_epoch_vae, loss_fn=scalar_scalar_elbo)

    model_det = SemiSupervisedVae(args.hidden_dim, args.latent_dim)
    model_union = SemiSupervisedVae(args.hidden_dim, args.latent_dim)
    model_det.to(device)
    model_union.to(device)
    optimizer_det = Adam(model_det.parameters(), lr=args.lr)
    optimizer_union = Adam(model_union.parameters(), lr=args.lr)

    dpath_spurious = os.path.join(args.dpath, "spurious")
    dpath_union = os.path.join(args.dpath, "union")
    os.makedirs(dpath_spurious, exist_ok=True)
    os.makedirs(dpath_union, exist_ok=True)

    train_eval_loop(data_train_det, data_val_det, model_det, optimizer_det, train_f, eval_f, dpath_spurious,
        args.n_epochs, args.n_early_stop_epochs)
    train_eval_loop(data_train_union, data_val_union, model_union, optimizer_union, train_f, eval_f, dpath_union,
        args.n_epochs, args.n_early_stop_epochs)

    test_fpath = os.path.join(args.dpath, "test_summary.txt")
    model_det.eval()
    model_union.eval()
    kldivs_det, kldivs_union = [], []
    for x0_batch, x1_batch, y_batch in data_test_det:
        x0_batch, x1_batch, y_batch = x0_batch.to(device), x1_batch.to(device), y_batch.to(device)
        kldivs_det.append(posterior_kldiv(*model_det.encode(x0_batch, x1_batch, y_batch)).item())
        kldivs_union.append(posterior_kldiv(*model_union.encode(x0_batch, x1_batch, y_batch)).item())
    write(test_fpath, f"data_test_det, model_det={np.mean(kldivs_det):.3f}, model_union={np.mean(kldivs_union):.3f}")
    kldivs_det, kldivs_union = [], []
    for x0_batch, x1_batch, y_batch in data_test_nondet:
        x0_batch, x1_batch, y_batch = x0_batch.to(device), x1_batch.to(device), y_batch.to(device)
        kldivs_det.append(posterior_kldiv(*model_det.encode(x0_batch, x1_batch, y_batch)).item())
        kldivs_union.append(posterior_kldiv(*model_union.encode(x0_batch, x1_batch, y_batch)).item())
    write(test_fpath, f"data_test_nondet, model_det={np.mean(kldivs_det):.3f}, model_union={np.mean(kldivs_union):.3f}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dpath", type=str, default="results")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n-epochs", type=int, default=200)
    parser.add_argument("--n-early-stop-epochs", type=int, default=20)
    parser.add_argument("--n-anneal-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=256)
    main(parser.parse_args())