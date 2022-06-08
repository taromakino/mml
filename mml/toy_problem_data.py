import numpy as np

def make_data(rng, n_examples, uy_distr, sigma):
    p00, p01, p10, p11 = uy_distr.flatten()
    ub00 = p00
    ub01 = ub00 + p01
    ub10 = ub01 + p10
    uy = []
    for _ in range(n_examples):
        p = np.random.random()
        if p <= ub00:
            uy.append([0, 0])
        elif p <= ub01:
            uy.append([0, 1])
        elif p <= ub10:
            uy.append([1, 0])
        else:
            uy.append([1, 1])
    uy = np.array(uy)
    u, y = uy[:, 0], uy[:, 1]

    idxs_y1_u0 = np.where((y == 1) & (u == 0))[0]
    idxs_y1_u1 = np.where((y == 1) & (u == 1))[0]
    idxs_y0_u0 = np.where((y == 0) & (u == 0))[0]
    idxs_y0_u1 = np.where((y == 0) & (u == 1))[0]

    cov = np.array([
        [1, sigma],
        [sigma, 1]])

    x = np.full((n_examples, 2), np.nan)
    x[idxs_y1_u0] = rng.multivariate_normal(mean=np.array([2, 0]), cov=cov, size=len(idxs_y1_u0))
    x[idxs_y1_u1] = rng.multivariate_normal(mean=np.array([2, 0]), cov=np.eye(2), size=len(idxs_y1_u1))
    x[idxs_y0_u0] = rng.multivariate_normal(mean=np.array([0, 0]), cov=np.eye(2), size=len(idxs_y0_u0))
    x[idxs_y0_u1] = rng.multivariate_normal(mean=np.array([0, 0]), cov=cov, size=len(idxs_y0_u1))

    x, y = x.astype("float32"), y.astype("float32")
    return x, y