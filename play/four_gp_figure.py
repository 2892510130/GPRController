"""Regenerate figure 3.2: exact GP vs DTC vs FITC vs variational FITC.

All four panels share the same data (51 samples of x sin x on [0, 20]) and the
same 11 inducing points; the view is restricted to the data region because the
sparse models' bands explode outside it.
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from GPy.kern import RBF

from model.gp_model import SparseDTCGP, SparseFITCGP
from play.pred_benchmark import NOISE_STD, ExactGP, data, make_opts, metrics

X_LO, X_HI = -1.0, 21.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    x, y, u, _, _ = data()
    grid = np.linspace(X_LO, X_HI, 241).reshape(-1, 1)
    opts = make_opts()
    variance = NOISE_STD ** 2

    panels = []
    gp = ExactGP(RBF(input_dim=1, variance=1.0, lengthscale=1.0))
    gp.fit(x, y, variance)
    mu, var = gp.predict(grid)
    panels.append(("Exact GP", np.asarray(mu).reshape(-1),
                   np.asarray(var).reshape(-1), None))
    mu_x, var_x = gp.predict(x)
    print("Exact GP         : %.3f / %.3f / %.3f" % metrics(mu_x, var_x, y))

    for name, cls, method in (("DTC GP", SparseDTCGP, "cholesky"),
                              ("FITC GP", SparseFITCGP, "cholesky"),
                              ("Variational FITC GP", SparseFITCGP, "mu_su")):
        gp = cls(RBF(input_dim=1, variance=1.0, lengthscale=1.0), opts)
        gp.fit(x, y, u, variance)
        mu, var = gp.predict(grid, method=method)
        mu = np.asarray(mu).reshape(-1)
        var = np.asarray(var).reshape(-1)
        panels.append((name, mu, var, np.asarray(gp.U).reshape(-1)))
        print(f"{name:<17}: %.3f / %.3f / %.3f" % metrics(
            *gp.predict(x, method=method), x * np.sin(x)))

    lo = min(float(np.asarray(y).min()), *(float(p[1].min()) for p in panels))
    hi = max(float(np.asarray(y).max()), *(float(p[1].max()) for p in panels))
    pad = 0.08 * (hi - lo)
    ylim = (lo - pad, hi + pad)

    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
                         "legend.frameon": False, "axes.linewidth": 0.8})
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (name, mu, var, uz) in zip(axes.ravel(), panels):
        std = np.sqrt(np.maximum(var, 0.0))
        ax.plot(x, y, "xk", ms=5, mew=1.2, label="Training data")
        if uz is not None:
            ax.plot(uz, np.interp(uz, np.asarray(x).reshape(-1),
                                  np.asarray(y).reshape(-1)),
                    "ob", ms=4.5, mew=1.0, label="Inducing points")
        ax.plot(grid, mu, "r", lw=1.2, label="Mean")
        ax.fill_between(grid.ravel(), mu - 2 * std, mu + 2 * std,
                        color="red", alpha=0.15, label=r"$\pm 2\sigma$")
        ax.set_title(name, fontsize=10)
        ax.set_xlim(X_LO, X_HI)
        ax.set_ylim(*ylim)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
    axes[0, 0].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    os.makedirs(args.outdir, exist_ok=True)
    path = os.path.join(args.outdir, "compare_4gp.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    main()