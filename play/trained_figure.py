"""Regenerate figure 3.3: DTC GP before / after joint optimisation.

Both panels use the same 11 inducing points; the right panel optimises the
hyper-parameters and the inducing positions (GPy SparseGPRegression, DTC
approximation, 5 restarts), so the comparison isolates the effect of training.
"""

from __future__ import annotations

import argparse
import os

import GPy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from GPy.kern import RBF

from model.gp_model import SparseDTCGP
from play.pred_benchmark import NOISE_STD, data, make_opts, metrics


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    x, y, u, x_pred, y_true = data()
    grid = np.linspace(-5.0, 25.0, 201).reshape(-1, 1)

    gp = SparseDTCGP(RBF(input_dim=1, variance=1.0, lengthscale=1.0), make_opts())
    gp.fit(x, y, u, NOISE_STD ** 2)
    mu_u, var_u = gp.predict(grid, method="cholesky")
    mu_u, var_u = np.asarray(mu_u).reshape(-1), np.asarray(var_u).reshape(-1)
    print("untrained  : %.3f / %.3f / %.3f" % metrics(
        *gp.predict(x, method="cholesky"), y))

    np.random.seed(0)
    m = GPy.models.SparseGPRegression(x, y, kernel=RBF(1, variance=1.0, lengthscale=1.0),
                                      normalizer=False, Z=np.asarray(u).copy())
    m.optimize_restarts(num_restarts=5, verbose=False)
    mu_t, var_t = m.predict(grid)
    mu_t, var_t = np.asarray(mu_t).reshape(-1), np.asarray(var_t).reshape(-1)
    print("trained    : %.3f / %.3f / %.3f (M=%d)" % (
        *metrics(*m.predict(x), y), m.Z.shape[0]))

    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
                         "legend.frameon": False, "axes.linewidth": 0.8})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    # outside [0, 20] the trained model has a huge prior variance (std ~ 50), so
    # its mean and band explode; the view is restricted to the data region
    x_lo, x_hi = -1.0, 21.0
    inside = (grid[:, 0] >= x_lo) & (grid[:, 0] <= x_hi)
    lo = min(float(np.asarray(y).min()), float(mu_u[inside].min()), float(mu_t[inside].min()))
    hi = max(float(np.asarray(y).max()), float(mu_u[inside].max()), float(mu_t[inside].max()))
    pad = 0.08 * (hi - lo)
    ylim = (lo - pad, hi + pad)

    panels = ((axes[0], mu_u, var_u, np.asarray(u).reshape(-1),
               np.asarray(y)[::5].reshape(-1), "Untrained"),
              (axes[1], mu_t, var_t, np.asarray(m.Z).reshape(-1),
               np.asarray(m.predict(m.Z)[0]).reshape(-1), "Trained"))
    for ax, mu, var, zx, zy, title in panels:
        std = np.sqrt(np.maximum(var, 0.0))
        ax.plot(x, y, "xk", ms=5, mew=1.2, label="Training data")
        ax.plot(zx, zy, "ob", ms=5, mew=1.0, label="Inducing points")
        ax.plot(grid, mu, "r", lw=1.2, label="Mean")
        ax.fill_between(grid.ravel(), mu - 2 * std, mu + 2 * std,
                        color="red", alpha=0.15, label=r"$\pm 2\sigma$")
        ax.set_title(title, fontsize=10)
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(*ylim)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
    axes[0].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    os.makedirs(args.outdir, exist_ok=True)
    path = os.path.join(args.outdir, "trained.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    main()