from functools import partial
from utils.ml import *
from torch.optim import Adam
from image_image.data import make_data
from image_image.model import SemiSupervisedVae
from argparse import ArgumentParser

def main(args):
    set_seed(args.seed)
    rng = np.random.RandomState(args.seed)
    device = make_device()

    x0_trainval_s, x1_trainval_s, y_trainval_s = make_data(rng, True, 0)
    x0_test_s, x1_test_s, y_test_s = make_data(rng, False, 0)

    x0_trainval_ns, x1_trainval_ns, y_trainval_ns = make_data(rng, True, args.p_shuffle_u)
    x0_test_ns, x1_test_ns, y_test_ns = make_data(rng, False, args.p_shuffle_u)

    (x0_train_s, x1_train_s, y_train_s), (x0_val_s, x1_val_s, y_val_s) = \
        split_data(args.trainval_ratios, x0_trainval_s, x1_trainval_s, y_trainval_s)
    (x0_train_ns, x1_train_ns, y_train_ns), (x0_val_ns, x1_val_ns, y_val_ns) = \
        split_data(args.trainval_ratios, x0_trainval_ns, x1_trainval_ns, y_trainval_ns)

    x0_train_s, x1_train_s = torch.tensor(x0_train_s), torch.tensor(x1_train_s)
    x0_val_s, x1_val_s = torch.tensor(x0_val_s), torch.tensor(x1_val_s)
    x0_test_s, x1_test_s = torch.tensor(x0_test_s), torch.tensor(x1_test_s)
    y_train_s, y_val_s, y_test_s = torch.tensor(y_train_s)[:, None], torch.tensor(y_val_s)[:, None], \
        torch.tensor(y_test_s)[:, None]

    x0_train_ns, x1_train_ns = torch.tensor(x0_train_ns), torch.tensor(x1_train_ns)
    x0_val_ns, x1_val_ns = torch.tensor(x0_val_ns), torch.tensor(x1_val_ns)
    x0_test_ns, x1_test_ns = torch.tensor(x0_test_ns), torch.tensor(x1_test_ns)
    y_train_ns, y_val_ns, y_test_ns = torch.tensor(y_train_ns)[:, None], torch.tensor(y_val_ns)[:,None], \
        torch.tensor(y_test_ns)[:, None]

    data_train_s = make_dataloader((x0_train_s, x1_train_s, y_train_s), args.batch_size, True)
    data_val_s = make_dataloader((x0_val_s, x1_val_s, y_val_s), args.batch_size, False)

    data_train_ns = make_dataloader((x0_train_ns, x1_train_ns, y_train_ns), args.batch_size, True)
    data_val_union = make_dataloader((x0_val_ns, x1_val_ns, y_val_ns), args.batch_size, False)

    data_test_s = make_dataloader((x0_test_s, x1_test_s, y_test_s), args.batch_size, False)
    data_test_ns = make_dataloader((x0_test_ns, x1_test_ns, y_test_ns), args.batch_size, False)

    train_f = partial(train_epoch_vae, loss_fn=image_image_elbo, n_anneal_epochs=args.n_anneal_epochs)
    eval_f = partial(eval_epoch_vae, loss_fn=image_image_elbo)

    model_s = SemiSupervisedVae(x0_train_s.shape[1], x1_train_s.shape[1], args.hidden_dim, args.latent_dim)
    model_ns = SemiSupervisedVae(x0_train_s.shape[1], x1_train_s.shape[1], args.hidden_dim, args.latent_dim)
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
    train_eval_loop(data_train_ns, data_val_union, model_ns, optimizer_ns, train_f, eval_f, dpath_ns, args.n_epochs,
        args.n_early_stop_epochs)

    test_fpath = os.path.join(args.dpath, "test_summary.txt")
    result = eval_marginal_likelihood(data_test_s, model_s, args.n_samples)
    write(test_fpath, f"model_s, data_test_s, marginal_likelihood={result:.3f}")
    result = eval_marginal_likelihood(data_test_ns, model_s, args.n_samples)
    write(test_fpath, f"model_s, data_test_ns, marginal_likelihood={result:.3f}")
    result = eval_marginal_likelihood(data_test_s, model_ns, args.n_samples)
    write(test_fpath, f"model_ns, data_test_s, marginal_likelihood={result:.3f}")
    result = eval_marginal_likelihood(data_test_ns, model_ns, args.n_samples)
    write(test_fpath, f"model_ns, data_test_ns, marginal_likelihood={result:.3f}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dpath", type=str, default="results")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--p-shuffle-u", type=float, default=1)
    parser.add_argument("--trainval-ratios", nargs="+", type=float, default=[0.8, 0.2])
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n-epochs", type=int, default=200)
    parser.add_argument("--n-early-stop-epochs", type=int, default=20)
    parser.add_argument("--n-anneal-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--n-samples", type=int, default=100)
    main(parser.parse_args())