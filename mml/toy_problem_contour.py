import numpy as np
from scipy.stats import multivariate_normal as mvn
from utils.plotting import *

sigma = 0.9
uy_prior = np.array([
    [0.25, 0.25],
    [0.25, 0.25]])

x_lim = (-4, 6)
y_lim = (-5, 5)
w = np.linspace(*x_lim)
h = np.linspace(*y_lim)
ww, hh = np.meshgrid(w, h)
x = np.column_stack((ww.flatten(), hh.flatten()))

density_x = 0
for u in range(2):
    for y in range(2):
        x_mean = np.array([2 * y, 0])
        x_cov = np.array([
            [1, sigma * (u != y)],
            [sigma * (u != y), 1]])
        density_x += mvn.pdf(x, x_mean, x_cov) * uy_prior[u, y]
density_x = density_x.reshape(ww.shape)

marginal_w = density_x.sum(axis=0)
marginal_h = density_x.sum(axis=1)
density_x_indep = np.outer(marginal_w, marginal_h)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].contourf(ww, hh, density_x)
axes[1].contourf(ww, hh, density_x_indep)
for ax in axes:
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
axes[0].set_title(r"p(x, x')")
axes[1].set_title(r"p(x)p(x')")

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.contourf(ww, hh, density_x - density_x_indep)