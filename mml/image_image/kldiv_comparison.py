from functools import partial
from utils.ml import *
from torch.optim import Adam
from image_image.data import make_data
from image_image.model import SemiSupervisedVae
from argparse import ArgumentParser

def main(args):
    set_seed(args.seed)
    rng = np.random.RandomState(args.seed)

    x0_trainval_det, x1_trainval_det, y_trainval_det = make_data(rng, True, 0)
    x0_test_det, x1_test_det, y_test_det = make_data(rng, False, 0)

    x0_trainval_nondet, x1_trainval_nondet, y_trainval_nondet = make_data(rng, True, args.p_shuffle_u)
    x0_test_nondet, x1_test_nondet, y_test_nondet = make_data(rng, False, args.p_shuffle_u)

    (x0_train_det, x1_train_det, y_train_det), (x0_val_det, x1_val_det, y_val_det) = \
        split_data(args.trainval_ratios, x0_trainval_det, x1_trainval_det, y_trainval_det)
    (x0_train_nondet, x1_train_nondet, y_train_nondet), (x0_val_nondet, x1_val_nondet, y_val_nondet) = \
        split_data(args.trainval_ratios, x0_trainval_nondet, x1_trainval_nondet, y_trainval_nondet)

    x0_train_det, x1_train_det = torch.tensor(x0_train_det), torch.tensor(x1_train_det)
    x0_val_det, x1_val_det = torch.tensor(x0_val_det), torch.tensor(x1_val_det)
    x0_test_det, x1_test_det = torch.tensor(x0_test_det), torch.tensor(x1_test_det)
    y_train_det, y_val_det, y_test_det = torch.tensor(y_train_det)[:, None], torch.tensor(y_val_det)[:, None], \
        torch.tensor(y_test_det)[:, None]

    x0_train_nondet, x1_train_nondet = torch.tensor(x0_train_nondet), torch.tensor(x1_train_nondet)
    x0_val_nondet, x1_val_nondet = torch.tensor(x0_val_nondet), torch.tensor(x1_val_nondet)
    x0_test_nondet, x1_test_nondet = torch.tensor(x0_test_nondet), torch.tensor(x1_test_nondet)
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

    train_f = partial(train_epoch_vae, loss_fn=image_image_elbo, n_anneal_epochs=args.n_anneal_epochs)
    eval_f = partial(eval_epoch_vae, loss_fn=image_image_elbo)

    model_det = SemiSupervisedVae(x0_train_det.shape[1], x1_train_det.shape[1], args.hidden_dim, args.latent_dim)
    model_union = SemiSupervisedVae(x0_train_det.shape[1], x1_train_det.shape[1], args.hidden_dim, args.latent_dim)
    model_det.to(make_device())
    model_union.to(make_device())
    optimizer_det = Adam(model_det.parameters(), lr=args.lr)
    optimizer_union = Adam(model_union.parameters(), lr=args.lr)

    dpath_spurious = os.path.join(args.dpath, "spurious")
    dpath_union = os.path.join(args.dpath, "union")
    os.makedirs(dpath_spurious, exist_ok=True)
    os.makedirs(dpath_union, exist_ok=True)

    train_eval_loop(*data_det, model_det, optimizer_det, train_f, eval_f, dpath_spurious, args.n_epochs)
    train_eval_loop(*data_union, model_union, optimizer_union, train_f, eval_f, dpath_union, args.n_epochs)

    device = make_device()
    kldivs_det, kldivs_union = [], []
    data_test_det, data_test_union = data_det[-1], data_union[-1]
    model_det.eval()
    model_union.eval()
    for x0_batch, x1_batch, y_batch in data_test_det:
        x0_batch, x1_batch, y_batch = x0_batch.to(device), x1_batch.to(device), y_batch.to(device)
        kldivs_det.append(posterior_kldiv(*model_det.posterior_params(x0_batch, x1_batch, y_batch)).item())
    for x0_batch, x1_batch, y_batch in data_test_union:
        x0_batch, x1_batch, y_batch = x0_batch.to(device), x1_batch.to(device), y_batch.to(device)
        kldivs_union.append(posterior_kldiv(*model_union.posterior_params(x0_batch, x1_batch, y_batch)).item())
    print(f"det={np.mean(kldivs_det):.3f}, union={np.mean(kldivs_union):.3f}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dpath", type=str, default="results")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset-name", type=str, default="MNIST")
    parser.add_argument("--p-shuffle-u", type=float, default=0.5)
    parser.add_argument("--trainval-ratios", nargs="+", type=float, default=[0.8, 0.2])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n-epochs", type=int, default=100)
    parser.add_argument("--n-anneal-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--latent-dim", type=int, default=256)
    main(parser.parse_args())