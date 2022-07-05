from functools import partial
from utils.ml import *
from torch.optim import Adam
from image_scalar.data import make_data
from image_scalar.model import SemiSupervisedVae
from argparse import ArgumentParser

def main(args):
    set_seed(args.seed)
    rng = np.random.RandomState(args.seed)
    device = make_device()

    # Make numpy data
    x0_trainval_det, x1_trainval_det, y_trainval_det = make_data(args.dataset_name, rng, True, 0, args.sigma)
    x0_trainval_nondet, x1_trainval_nondet, y_trainval_nondet = make_data(args.dataset_name, rng, True,
        args.p_flip_color, args.sigma)

    x0_test_det, x1_test_det, y_test_det = make_data(args.dataset_name, rng, False, 0, args.sigma)
    x0_test_nondet, x1_test_nondet, y_test_nondet = make_data(args.dataset_name, rng, False, args.p_flip_color, args.sigma)

    # Split train/val/test
    (x0_train_det, x1_train_det, y_train_det), (x0_val_det, x1_val_det, y_val_det) = \
        split_data(args.trainval_ratios, x0_trainval_det, x1_trainval_det, y_trainval_det)
    (x0_train_nondet, x1_train_nondet, y_train_nondet), (x0_val_nondet, x1_val_nondet, y_val_nondet) = \
        split_data(args.trainval_ratios, x0_trainval_nondet, x1_trainval_nondet, y_trainval_nondet)

    # Normalize
    x1_mean_det, x1_sd_det = x1_train_det.mean(), x1_train_det.std()
    x1_mean_nondet, x1_sd_nondet = x1_train_nondet.mean(), x1_train_nondet.std()

    x1_train_det = (x1_train_det - x1_mean_det) / x1_sd_det
    x1_val_det = (x1_val_det - x1_mean_det) / x1_sd_det
    x1_test_det = (x1_test_det - x1_mean_det) / x1_sd_det

    x1_train_nondet = (x1_train_nondet - x1_mean_nondet) / x1_sd_nondet
    x1_val_nondet = (x1_val_nondet - x1_mean_nondet) / x1_sd_nondet
    x1_test_nondet = (x1_test_nondet - x1_mean_nondet) / x1_sd_nondet

    # To torch
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

    x0_train_union, x1_train_union, y_train_union = x0_train_union[train_subset_idxs], x1_train_union[train_subset_idxs], \
        y_train_union[train_subset_idxs]
    x0_val_union, x1_val_union, y_val_union = x0_val_union[val_subset_idxs], x1_val_union[val_subset_idxs], \
        y_val_union[val_subset_idxs]

    # Run experiment
    data_train_det = make_dataloader((x0_train_det, x1_train_det, y_train_det), args.batch_size, True)
    data_val_det = make_dataloader((x0_val_det, x1_val_det, y_val_det), args.batch_size, False)

    data_train_union = make_dataloader((x0_train_union, x1_train_union, y_train_union), args.batch_size, True)
    data_val_union = make_dataloader((x0_val_union, x1_val_union, y_val_union), args.batch_size, False)

    data_test_det = make_dataloader((x0_test_det, x1_test_det, y_test_det), args.batch_size, False)
    data_test_nondet = make_dataloader((x0_test_nondet, x1_test_nondet, y_test_nondet), args.batch_size, False)

    train_f = partial(train_epoch_vae, loss_fn=image_scalar_elbo, n_anneal_epochs=args.n_anneal_epochs)
    eval_f = partial(eval_epoch_vae, loss_fn=image_scalar_elbo)

    model_det = SemiSupervisedVae(x0_train_det.shape[1], args.hidden_dim, args.latent_dim)
    model_union = SemiSupervisedVae(x0_train_det.shape[1], args.hidden_dim, args.latent_dim)
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
        kldivs_det.append(posterior_kldiv(*model_det.encode(x0_batch, x1_batch, y_batch)).detach().numpy())
        kldivs_union.append(posterior_kldiv(*model_union.encode(x0_batch, x1_batch, y_batch)).detach().numpy())
    kldivs_det, kldivs_union = np.array(kldivs_det), np.array(kldivs_union)
    write(test_fpath, f"data_test_det, model_det={np.mean(kldivs_det):.3f}, model_union={np.mean(kldivs_union):.3f}")
    kldivs_det, kldivs_union = [], []
    for x0_batch, x1_batch, y_batch in data_test_nondet:
        x0_batch, x1_batch, y_batch = x0_batch.to(device), x1_batch.to(device), y_batch.to(device)
        kldivs_det.append(posterior_kldiv(*model_det.encode(x0_batch, x1_batch, y_batch)).detach().numpy())
        kldivs_union.append(posterior_kldiv(*model_union.encode(x0_batch, x1_batch, y_batch)).detach().numpy())
    kldivs_det, kldivs_union = np.array(kldivs_det), np.array(kldivs_union)
    write(test_fpath, f"data_test_nondet, model_det={np.mean(kldivs_det):.3f}, model_union={np.mean(kldivs_union):.3f}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dpath", type=str, default="results")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset-name", type=str, default="MNIST")
    parser.add_argument("--p-flip-color", type=float, default=0.5)
    parser.add_argument("--sigma", type=float, default=1)
    parser.add_argument("--trainval-ratios", nargs="+", type=float, default=[0.8, 0.2])
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n-epochs", type=int, default=200)
    parser.add_argument("--n-early-stop-epochs", type=int, default=20)
    parser.add_argument("--n-anneal-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--latent-dim", type=int, default=256)
    main(parser.parse_args())