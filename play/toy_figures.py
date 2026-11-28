"""Regenerate the three chapter-3 comparison figures."""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from GPy.kern import RBF
from GPy.models import GPRegression

from model.gp_model import SparseFITCGP
from play.toy_benchmark import NOISE_STD, make_data, make_opts

PANELS = [("standard", "Before update"), ("osvgp", "OSVGP"),
          ("bijl", "Bijl"), ("agp_window", "ASGP")]


def run(case: str, method: str, incremental: bool):
    x, y, x_update, y_update, u_idx = make_data(case)
    if method == "standard":
        gp = GPRegression(x, y, RBF(input_dim=1, variance=1.0, lengthscale=1.0),
                          noise_var=NOISE_STD ** 2)
        return x, y, x_update, y_update, None, gp

    gp = SparseFITCGP(RBF(input_dim=1, variance=1.0, lengthscale=1.0), make_opts())
    gp.fit(x, y, u_idx, NOISE_STD ** 2, lambda_=0.99 if method == "agp_window" else 1.0)
    for xi, yi in zip(x_update, y_update):
        xi = xi.reshape(1, -1)
        if incremental:
            var_new, aa = gp.check_new_inducing_var(xi)
            if var_new > aa * 0.01:
                gp.add_new_inducing(xi)
        if method == "osvgp":
            gp.update(xi, yi)
        elif method == "bijl":
            gp.update_mu_su_bijl(xi, yi)
        elif method == "agp_window":
            gp.add_new_data(xi, np.atleast_1d(yi).reshape(1, 1))
            gp.remove_data(1)
    return x, y, x_update, y_update, gp, None


def predict(model, gp, grid):
    if gp is None:
        mu, var = model.predict(grid)
    else:
        mu, var = gp.predict(grid, method="mu_su")
        if gp.M > 0:
            pass
    return np.asarray(mu).reshape(-1), np.asarray(var).reshape(-1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--incremental", action="store_true")
    ap.add_argument("--name", required=True)
    args = ap.parse_args()

    case = "same" if args.name == "compare_in_same" else "cross"
    grid = np.linspace(-5.0, 25.0, 201).reshape(-1, 1)

    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
                         "legend.frameon": False, "axes.linewidth": 0.8})
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    order = [0, 2, 1, 3]  # before, bijl / osvgp, agp  -> matches 2x2 layout

    results = []
    lo, hi = np.inf, -np.inf
    for idx in order:
        method, title = PANELS[idx]
        x, y, x_up, y_up, gp, exact = run(case, method, args.incremental)
        mu, var = predict(exact, gp, grid)
        std = np.sqrt(np.maximum(var, 0.0))
        results.append((method, title, x, y, x_up, y_up, gp, exact, mu, std))
        lo = min(lo, float(np.asarray(y).min()), float(np.asarray(y_up).min()),
                 float(mu.min()), float(np.percentile(mu - 2 * std, 2)))
        hi = max(hi, float(np.asarray(y).max()), float(np.asarray(y_up).max()),
                 float(mu.max()), float(np.percentile(mu + 2 * std, 98)))
    pad = 0.06 * (hi - lo)
    ylim = (lo - pad, hi + pad)
    print(f"  shared ylim = ({ylim[0]:+.1f}, {ylim[1]:+.1f})")

    for ax, (method, title, x, y, x_up, y_up, gp, exact, mu, std) in zip(axes.ravel(), results):

        x0 = np.asarray(x).reshape(-1)
        y0 = np.asarray(y).reshape(-1)
        u0 = x0[::5]
        ax.plot(u0, y0[::5], "ob", ms=4.5, mew=1.0, label="Inducing data")
        if gp is not None and gp.M > len(u0):
            extra = np.asarray(gp.U[len(u0):, 0], dtype=float)
            ax.plot(extra, np.interp(extra, np.asarray(x_up).reshape(-1),
                                     np.asarray(y_up).reshape(-1)),
                    "^b", ms=4.5, label="Added inducing")
        ax.plot(x, y, "xk", ms=5, mew=1.2, label="Training data")
        ax.plot(x_up, y_up, ".", color="0.55", ms=3, label="Update data")
        ax.plot(grid, mu, "r", lw=1.2, label="Mean")
        ax.fill_between(grid.ravel(), mu - 2 * std, mu + 2 * std,
                        color="red", alpha=0.15, label=r"$\pm 2\sigma$")
        ax.set_title(title, fontsize=10)
        ax.set_xlim(-5, 25)
        ax.set_ylim(*ylim)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
    axes[0, 0].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    os.makedirs(args.outdir, exist_ok=True)
    path = os.path.join(args.outdir, f"{args.name}.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    main()