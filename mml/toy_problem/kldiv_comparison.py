from argparse import ArgumentParser
from functools import partial
from utils.ml import *
from torch.optim import Adam
from toy_problem.data import make_data
from arch.two_scalar_vae import TwoScalarSSVAE

def main(args):
    set_seed(args.seed)
    rng = np.random.RandomState(args.seed)

    n_examples = 10000
    trainval_ratios = [0.8, 0.1]

    uy_prior_det = np.array([
        [0.5, 0],
        [0, 0.5]])
    uy_prior_nondet = np.array([
        [0.25, 0.25],
        [0.25, 0.25]])
    sigma = 0.9

    hidden_dim = 256
    n_hidden = 3
    latent_dim = 256

    x_det, y_det = make_data(rng, n_examples, uy_prior_det, sigma)
    x_nondet, y_nondet = make_data(rng, n_examples, uy_prior_nondet, sigma)

    (x_train_det, y_train_det), (x_val_det, y_val_det), (x_test_det, y_test_det) = \
        split_data(x_det, y_det, trainval_ratios)
    (x_train_nondet, y_train_nondet), (x_val_nondet, y_val_nondet), (x_test_nondet, y_test_nondet) = \
        split_data(x_nondet, y_nondet, trainval_ratios)

    x_train_union = np.vstack((x_train_det, x_train_nondet))
    y_train_union = np.concatenate((y_train_det, y_train_nondet))

    x_val_union = np.vstack((x_val_det, x_val_nondet))
    y_val_union = np.concatenate((y_val_det, y_val_nondet))

    x_test_union = np.vstack((x_test_det, x_test_nondet))
    y_test_union = np.concatenate((y_test_det, y_test_nondet))

    x0_train_det, x1_train_det = x_train_det[:, 0], x_train_det[:, 1]
    x0_val_det, x1_val_det = x_val_det[:, 0], x_val_det[:, 1]
    x0_test_det, x1_test_det = x_test_det[:, 0], x_test_det[:, 1]

    x0_train_union, x1_train_union = x_train_union[:, 0], x_train_union[:, 1]
    x0_val_union, x1_val_union = x_val_union[:, 0], x_val_union[:, 1]
    x0_test_union, x1_test_union = x_test_union[:, 0], x_test_union[:, 1]

    data_train_det = torch.tensor(x0_train_det)[:, None], torch.tensor(x1_train_det)[:, None], \
        torch.tensor(y_train_det)[:, None]
    data_val_det = torch.tensor(x0_val_det)[:, None], torch.tensor(x1_val_det)[:, None], \
        torch.tensor(y_val_det)[:, None]
    data_test_det = torch.tensor(x0_test_det)[:, None], torch.tensor(x1_test_det)[:, None], \
        torch.tensor(y_test_det)[:, None]

    data_train_union = torch.tensor(x0_train_union)[:, None], torch.tensor(x1_train_union)[:, None], \
        torch.tensor(y_train_union)[:, None]
    data_val_union = torch.tensor(x0_val_union)[:, None], torch.tensor(x1_val_union)[:, None], \
        torch.tensor(y_val_union)[:, None]
    data_test_union = torch.tensor(x0_test_union)[:, None], torch.tensor(x1_test_union)[:, None], \
        torch.tensor(y_test_union)[:, None]

    data_det = make_dataloaders(data_train_det, data_val_det, data_test_det, args.batch_size)
    data_union = make_dataloaders(data_train_union, data_val_union, data_test_union, args.batch_size)

    train_f = partial(train_epoch_vae, loss_fn0=F.mse_loss, loss_fn1=F.mse_loss, is_ssl=True)
    eval_f = partial(eval_epoch_vae, loss_fn0=F.mse_loss, loss_fn1=F.mse_loss, is_ssl=True)

    model_det = TwoScalarSSVAE(hidden_dim, n_hidden, latent_dim)
    model_union = TwoScalarSSVAE(hidden_dim, n_hidden, latent_dim)
    optimizer_det = Adam(model_det.parameters())
    optimizer_union = Adam(model_union.parameters())

    dpath_det = os.path.join(args.dpath, "spurious")
    dpath_union = os.path.join(args.dpath, "union")
    os.makedirs(dpath_det, exist_ok=True)
    os.makedirs(dpath_union, exist_ok=True)

    train_eval_loop(*data_det, model_det, optimizer_det, train_f, eval_f, dpath_det, args.n_epochs)
    train_eval_loop(*data_union, model_union, optimizer_union, train_f, eval_f, dpath_union, args.n_epochs)

    kldivs_det, kldivs_union = [], []
    data_test_union = data_union[-1]
    for x0_batch, x1_batch, y_batch in data_test_union:
        kldivs_det.append(vae_kldiv(*model_det.posterior_params(x0_batch, x1_batch, y_batch)).item())
        kldivs_union.append(vae_kldiv(*model_union.posterior_params(x0_batch, x1_batch, y_batch)).item())
    print(f"det={np.mean(kldivs_det):.3f}, union={np.mean(kldivs_union):.3f}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dpath", type=str)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--n-epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    main(parser.parse_args())