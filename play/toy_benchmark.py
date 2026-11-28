"""Reproduce the chapter-3 toy comparisons inside GPRController.

Three settings from section 3.5 of the thesis (see tables 3.3-3.5):

``same``   training and update data both from x in [0, 20]; after the update the
           target shrinks to 0.33 x sin(x)  -> table 3.3 (non-incremental)
``cross``  training x in [0, 10], update x in [5, 20], target 0.3 x sin(x)
           -> table 3.4 (no expansion) and table 3.5 (with expansion)

Methods implemented with the classes of this repository:

``standard``    exact GPy GP, no online update
``osvgp``       SparseFITCGP.update            (rank-one variational update)
``bijl``        SparseFITCGP.update_mu_su_bijl (non-variational FITC update)
``agp_window``  SparseFITCGP.add_new_data/remove_data with lambda = 0.99
                (sliding window, what Graduate_Data calls AGP)
``agp_addrm``   variational update with lambda = 0.99 plus marginal-contribution
                inducing add/prune (the "real" AGP of this repository)

Metrics follow Graduate_Data/graduate.py: RMSE, MAE and NLL of the prediction at
the update inputs against the update targets; timing is the mean per-update wall
clock over the update sequence.

Usage::

    python -m play.toy_benchmark                 # all settings, 1 repetition
    python -m play.toy_benchmark --repeats 20    # averaged timing
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from GPy.kern import RBF
from GPy.models import GPRegression

from model.gp_model import SparseFITCGP
from utils import GaussianConfig

NOISE_STD = 0.33


def make_opts() -> GaussianConfig:
    return GaussianConfig(
        path="", save_path="log/_toy", load_path="log/_toy",
        total_data_number=0, inducing_data_number=0, input_window=1,
        gpr_method=1, sparse_method=3, threshold=0.95,
        normalize_y=False, lagacy=False, enable_test=False,
        update_mu_su=True, add_mu_su=False, read_model=False, params=[],
    )


def make_data(case: str):
    np.random.seed(0)
    if case == "same":
        n = 51
        x = np.linspace(0.0, 20.0, n).reshape(-1, 1)
        x_update = x.copy()
        amp = 0.33
    else:
        n = 26
        x = np.linspace(0.0, 10.0, n).reshape(-1, 1)
        x_update = np.linspace(5.0, 20.0, 2 * n).reshape(-1, 1)
        amp = 0.30

    y = x * np.sin(x) + NOISE_STD * np.random.randn(n, 1)
    y_update = amp * x_update * np.sin(x_update) + NOISE_STD * np.random.randn(len(x_update), 1)
    return x, y, x_update, y_update, x[::5]


def metrics(mu, var, target):
    mu = np.asarray(mu).reshape(-1)
    var = np.asarray(var).reshape(-1)
    target = np.asarray(target).reshape(-1)
    rmse = float(np.sqrt(np.mean((mu - target) ** 2)))
    mae = float(np.mean(np.abs(mu - target)))
    nll = float(np.mean(0.5 * np.log(2 * np.pi * var) + (target - mu) ** 2 / (2 * var)))
    return rmse, mae, nll


def run_method(case: str, method: str, incremental: bool):
    x, y, x_update, y_update, u_idx = make_data(case)
    kernel = RBF(input_dim=1, variance=1.0, lengthscale=1.0)
    opts = make_opts()

    if method == "standard":
        gp = GPRegression(x, y, kernel, noise_var=NOISE_STD ** 2)
        mu, var = gp.predict(x_update)
        return metrics(mu, var, y_update), 0.0

    gp = SparseFITCGP(kernel, opts)
    gp.fit(x, y, u_idx, NOISE_STD ** 2, lambda_=0.99 if method == "agp_window" else 1.0)

    t0 = time.perf_counter()
    for xi, yi in zip(x_update, y_update):
        xi = xi.reshape(1, -1)
        if method == "osvgp":
            if incremental:
                var_new, aa = gp.check_new_inducing_var(xi)
                if var_new > aa * 0.01:
                    gp.add_new_inducing(xi)
            gp.update(xi, yi)
        elif method == "bijl":
            if incremental:
                var_new, aa = gp.check_new_inducing_var(xi)
                if var_new > aa * 0.01:
                    gp.add_new_inducing(xi)
            gp.update_mu_su_bijl(xi, yi)
        elif method == "agp_window":
            gp.add_new_data(xi, np.atleast_1d(yi).reshape(1, 1))
            gp.remove_data(1)
        elif method == "agp_addrm":
            gp.update(xi, yi)
            var_new, aa = gp.check_new_inducing_var(xi)
            if var_new > aa * 0.01 and gp.M < 60:
                gp.add_new_inducing(xi)
            elif gp.M >= 60:
                gp.remove_inducing(int(np.argmin(gp.contributions())))
        else:
            raise ValueError(method)
    elapsed = time.perf_counter() - t0

    mu, var = gp.predict(x_update, method="mu_su")
    return metrics(mu, var, y_update), 1000.0 * elapsed / len(x_update)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", nargs="*", default=["same", "cross"])
    ap.add_argument("--methods", nargs="*",
                    default=["standard", "osvgp", "bijl", "agp_window", "agp_addrm"])
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--incremental", action="store_true")
    args = ap.parse_args()

    for case in args.cases:
        print(f"=== case: {case}  incremental={args.incremental} ===")
        print(f"{'method':<12}{'RMSE':>10}{'MAE':>10}{'NLL':>12}{'ms/update':>12}")
        for method in args.methods:
            rows = [run_method(case, method, args.incremental) for _ in range(args.repeats)]
            (rmse, mae, nll), ms = rows[0]
            ms_avg = float(np.mean([r[1] for r in rows]))
            print(f"{method:<12}{rmse:>10.4f}{mae:>10.4f}{nll:>12.4f}{ms_avg:>12.3f}")
        print()


if __name__ == "__main__":
    main()