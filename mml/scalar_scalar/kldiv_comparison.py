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

    uy_prior_s = np.array([
        [0.5, 0],
        [0, 0.5]])
    uy_prior_ns = np.array([
        [0.25, 0.25],
        [0.25, 0.25]])
    sigma = 0.9

    # Make numpy data
    x_s, y_s = make_data(rng, args.n_examples, uy_prior_s, sigma)
    x_ns, y_ns = make_data(rng, args.n_examples, uy_prior_ns, sigma)

    # Split train/val/test
    (x_train_s, y_train_s), (x_val_s, y_val_s), (x_test_s, y_test_s) = split_data(args.trainval_ratios, x_s, y_s)
    (x_train_ns, y_train_ns), (x_val_ns, y_val_ns), (x_test_ns, y_test_ns) = split_data(args.trainval_ratios, x_ns, y_ns)

    # Normalize
    x_mean_s, x_sd_s = x_train_s.mean(0), x_train_s.std(0)
    x_mean_ns, x_sd_ns = x_train_ns.mean(0), x_train_ns.std(0)

    x_train_s = (x_train_s - x_mean_s) / x_sd_s
    x_val_s = (x_val_s - x_mean_s) / x_sd_s
    x_test_s = (x_test_s - x_mean_s) / x_sd_s

    x_train_ns = (x_train_ns - x_mean_ns) / x_sd_ns
    x_val_ns = (x_val_ns - x_mean_ns) / x_sd_ns
    x_test_ns = (x_test_ns - x_mean_ns) / x_sd_ns

    # To torch

    x0_train_s, x1_train_s = to_1d_torch(x_train_s[:, 0], x_train_s[:, 1])
    x0_val_s, x1_val_s = to_1d_torch(x_val_s[:, 0], x_val_s[:, 1])
    x0_test_s, x1_test_s = to_1d_torch(x_test_s[:, 0], x_test_s[:, 1])
    y_train_s, y_val_s, y_test_s = to_1d_torch(y_train_s, y_val_s, y_test_s)

    x0_train_ns, x1_train_ns = to_1d_torch(x_train_ns[:, 0], x_train_ns[:, 1])
    x0_val_ns, x1_val_ns = to_1d_torch(x_val_ns[:, 0], x_val_ns[:, 1])
    x0_test_ns, x1_test_ns = to_1d_torch(x_test_ns[:, 0], x_test_ns[:, 1])
    y_train_ns, y_val_ns, y_test_ns = to_1d_torch(y_train_ns, y_val_ns, y_test_ns)

    # Run experiment
    data_train_s = make_dataloader((x0_train_s, x1_train_s, y_train_s), args.batch_size, True)
    data_val_s = make_dataloader((x0_val_s, x1_val_s, y_val_s), args.batch_size, False)

    data_train_ns = make_dataloader((x0_train_ns, x1_train_ns, y_train_ns), args.batch_size, True)
    data_val_ns = make_dataloader((x0_val_ns, x1_val_ns, y_val_ns), args.batch_size, False)

    data_test_s = make_dataloader((x0_test_s, x1_test_s, y_test_s), args.batch_size, False)
    data_test_ns = make_dataloader((x0_test_ns, x1_test_ns, y_test_ns), args.batch_size, False)

    train_f = partial(train_epoch_vae, loss_fn=scalar_scalar_elbo, n_anneal_epochs=args.n_anneal_epochs)
    eval_f = partial(eval_epoch_vae, loss_fn=scalar_scalar_elbo)

    model_s = SemiSupervisedVae(args.hidden_dim, args.latent_dim)
    model_ns = SemiSupervisedVae(args.hidden_dim, args.latent_dim)
    model_s.to(device)
    model_ns.to(device)
    optimizer_s = Adam(model_s.parameters(), lr=args.lr)
    optimizer_ns = Adam(model_ns.parameters(), lr=args.lr)

    dpath_s = os.path.join(args.dpath, "spurious")
    dpath_ns = os.path.join(args.dpath, "nonspurious")
    os.makedirs(dpath_s, exist_ok=True)
    os.makedirs(dpath_ns, exist_ok=True)

    train_eval_loop(data_train_s, data_val_s, model_s, optimizer_s, train_f, eval_f, dpath_s, args.n_epochs,
        args.n_early_stop_epochs)
    train_eval_loop(data_train_ns, data_val_ns, model_ns, optimizer_ns, train_f, eval_f, dpath_ns, args.n_epochs,
        args.n_early_stop_epochs)

    test_fpath = os.path.join(args.dpath, "test_summary.txt")
    result = scalar_scalar_marginal_likelihood(data_test_s, model_s, args.n_samples)
    write(test_fpath, f"model_s, data_test_s, marginal_likelihood={result:.3f}")
    result = scalar_scalar_marginal_likelihood(data_test_ns, model_s, args.n_samples)
    write(test_fpath, f"model_s, data_test_ns, marginal_likelihood={result:.3f}")
    result = scalar_scalar_marginal_likelihood(data_test_s, model_ns, args.n_samples)
    write(test_fpath, f"model_ns, data_test_s, marginal_likelihood={result:.3f}")
    result = scalar_scalar_marginal_likelihood(data_test_ns, model_ns, args.n_samples)
    write(test_fpath, f"model_ns, data_test_ns, marginal_likelihood={result:.3f}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dpath", type=str, default="results")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-examples", type=int, default=100000)
    parser.add_argument("--trainval-ratios", nargs="+", type=float, default=[0.8, 0.1])
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n-epochs", type=int, default=200)
    parser.add_argument("--n-early-stop-epochs", type=int, default=20)
    parser.add_argument("--n-anneal-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--n-samples", type=int, default=1000)
    main(parser.parse_args())