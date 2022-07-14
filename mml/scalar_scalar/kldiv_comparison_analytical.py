import numpy as np
from utils.const import *
from utils.plotting import *
from scipy.special import softmax
from scipy.stats import entropy, multivariate_normal as mvn
from scalar_scalar.data import make_data

def posterior_entropy(x_test, y_test, uy_prior_train, sigma):
    u_posterior = []
    for u in range(2):
        x_mean_y0 = np.array([0, 0])
        x_mean_y1 = np.array([2, 0])
        x_cov_spurious = np.array([
            [1, 0],
            [0, 1]])
        x_cov_nonspurious = np.array([
            [1, sigma],
            [sigma, 1]])
        idxs_y0_spurious = np.where((y_test == 0) & (y_test == u))[0]
        idxs_y0_nonspurious = np.where((y_test == 0) & (y_test != u))[0]
        idxs_y1_spurious = np.where((y_test == 1) & (y_test == u))[0]
        idxs_y1_nonspurious = np.where((y_test == 1) & (y_test != u))[0]
        logdensity_x = np.full(len(x_test), np.nan)
        logdensity_x[idxs_y0_spurious] = mvn.logpdf(x_test[idxs_y0_spurious], x_mean_y0, x_cov_spurious)
        logdensity_x[idxs_y0_nonspurious] = mvn.logpdf(x_test[idxs_y0_nonspurious], x_mean_y0, x_cov_nonspurious)
        logdensity_x[idxs_y1_spurious] = mvn.logpdf(x_test[idxs_y1_spurious], x_mean_y1, x_cov_spurious)
        logdensity_x[idxs_y1_nonspurious] = mvn.logpdf(x_test[idxs_y1_nonspurious], x_mean_y1, x_cov_nonspurious)
        logprob_uy = np.full(len(y_test), np.nan)
        idxs_y0 = np.where(y_test == 0)[0]
        idxs_y1 = np.where(y_test == 1)[0]
        logprob_uy[idxs_y0] = uy_prior_train[u, 0]
        logprob_uy[idxs_y1] = uy_prior_train[u, 1]
        logprob_uy = np.clip(logprob_uy, EPSILON, 1 - EPSILON)
        logprob_uy = np.log(logprob_uy)
        u_posterior.append(logdensity_x + logprob_uy)
    u_posterior = np.column_stack(u_posterior)
    u_posterior = softmax(u_posterior, axis=1)
    return entropy(u_posterior, axis=1)

rng = np.random.RandomState(0)
n_examples = 100000
n_epochs = 50
uy_prior_test = np.array([
    [0.25, 0.25],
    [0.25, 0.25]])
sigma = 0.9

x_test, y_test = make_data(rng, n_examples, uy_prior_test, sigma)

values = []
prob_spurious_range = np.linspace(0.5, 1, 10)
for prob_spurious in prob_spurious_range:
    prob_nonspurious = 1 - prob_spurious
    uy_prior_train = np.array([
        [prob_spurious / 2, prob_nonspurious / 2],
        [prob_nonspurious / 2, prob_spurious / 2]])
    values.append(posterior_entropy(x_test, y_test, uy_prior_train, sigma).mean())
plt.plot(prob_spurious_range, values)
plt.xlabel(r"$P(U = Y)$")
plt.ylabel(r"$H(U \mid X, X', Y)$")
plt.grid(True)