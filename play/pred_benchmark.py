"""Reproduce the pure-prediction comparison of section 3.4 (tables 3.1 and 3.2).

Data: 51 samples of y = x sin(x) + 0.33 noise on [0, 20], 11 inducing points,
prediction grid on [-5, 25] against the noise-free function (as in
Graduate_Data/graduate.py::basic_prediction).  Table 3.2 additionally optimises
the hyper-parameters and the inducing positions of the DTC model.

Usage::

    python -m play.pred_benchmark
"""

from __future__ import annotations

import time

import numpy as np
from GPy.kern import RBF
from GPy.models import SparseGPRegression

from model.gp_model import SparseDTCGP, SparseFITCGP
from play.toy_benchmark import make_opts

NOISE_STD = 0.33
N = 51


class ExactGP:
    """Exact GP with a fixed kernel, matching Graduate_Data/exact_gp.py."""

    def __init__(self, kernel):
        self.kernel = kernel

    def fit(self, x, y, variance):
        self.x, self.y, self.variance = np.atleast_2d(x), np.atleast_2d(y), variance
        k = self.kernel.K(self.x) + (1e-6 + variance) * np.eye(len(self.x))
        self.L = np.linalg.cholesky(k)
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, self.y))

    def predict(self, xs):
        ks = self.kernel.K(self.x, xs)
        v = np.linalg.solve(self.L, ks)
        return ks.T @ self.alpha, np.diag(self.kernel.K(xs) - v.T @ v)


def metrics(mu, var, target):
    mu = np.asarray(mu).reshape(-1)
    var = np.asarray(var).reshape(-1)
    target = np.asarray(target).reshape(-1)
    return (float(np.sqrt(np.mean((mu - target) ** 2))),
            float(np.mean(np.abs(mu - target))),
            float(np.mean(0.5 * np.log(2 * np.pi * var) + (target - mu) ** 2 / (2 * var))))


def data():
    np.random.seed(0)
    x = np.linspace(0.0, 20.0, N).reshape(-1, 1)
    y = x * np.sin(x) + NOISE_STD * np.random.randn(N, 1)
    x_pred = np.linspace(-5.0, 25.0, 101).reshape(-1, 1)
    return x, y, x[::5], x_pred, x_pred * np.sin(x_pred)


def timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, 1000.0 * (time.perf_counter() - t0)


def main() -> None:
    kernel = RBF(input_dim=1, variance=1.0, lengthscale=1.0)
    variance = NOISE_STD ** 2
    opts = make_opts()
    x, y, u, x_pred, y_true = data()

    print(f"{'model':<18}{'RMSE':>10}{'MAE':>10}{'NLL':>11}{'ms':>9}")

    gp = ExactGP(kernel.copy())
    gp.fit(x, y, variance)
    (mu, var), ms = timed(lambda: gp.predict(x_pred))
    r, m_, n = metrics(mu, var, y_true)
    print(f"{'Exact GP':<18}{r:>10.3f}{m_:>10.3f}{n:>11.3f}{ms:>9.3f}")

    for name, cls, method in (("DTC GP", SparseDTCGP, "cholesky"),
                              ("FITC GP", SparseFITCGP, "cholesky"),
                              ("Variational FITC", SparseFITCGP, "mu_su")):
        gp = cls(kernel.copy(), opts)
        gp.fit(x, y, u, variance)
        (mu, var), ms = timed(lambda: gp.predict(x_pred, method=method))
        r, m, n = metrics(mu, var, y_true)
        print(f"{name:<18}{r:>10.3f}{m:>10.3f}{n:>11.3f}{ms:>9.3f}")
        mu_t, var_t = (gp.predict(x, method=method) if name != "Exact GP" else gp.predict(x))
        ra, ma, na = metrics(mu_t, var_t, y)
        rb, mb, nb = metrics(mu_t, var_t, x * np.sin(x))
        print(f"    on train X (noisy Y)  {ra:>10.3f}{ma:>10.3f}{na:>11.3f}")
        print(f"    on train X (clean Y)  {rb:>10.3f}{mb:>10.3f}{nb:>11.3f}")

    # table 3.2: DTC before / after joint optimisation of hyper-parameters and
    # inducing positions (GPy SparseGPRegression with the same 11 inducing points)
    m = SparseGPRegression(x, y, kernel=GPy_RBF(), num_inducing=11,
                           Z=np.asarray(u).copy(), noise_var=variance)
    mu, var = m.predict(x_pred)
    r, ma, n = metrics(mu, np.diag(var)[:, None], y_true)
    print(f"{'DTC untrained':<18}{r:>10.3f}{ma:>10.3f}{n:>11.3f}")
    m.optimize()
    mu, var = m.predict(x_pred)
    r, ma, n = metrics(mu, np.diag(var)[:, None], y_true)
    print(f"{'DTC trained':<18}{r:>10.3f}{ma:>10.3f}{n:>11.3f}")


def GPy_RBF():
    return RBF(input_dim=1, variance=1.0, lengthscale=1.0)


if __name__ == "__main__":
    main()
