import numpy as np
import pytest

from model.gp_model import GaussianModel, _kern_K, _kern_Kdiag
from utils import Config

CONFIG_PATH = "config/mpc/linear_mpc_frenet.yaml"


def _gp():
    config = Config.from_yaml(CONFIG_PATH)
    config.gaussian.read_model = False
    return GaussianModel(config.gaussian).gps[0]


def test_fast_rbf_kernel_matches_gpy():
    gp = _gp()
    X = gp.X[3:7] + 0.01

    np.testing.assert_allclose(
        _kern_K(gp.kern, gp.U, X), gp.kern.K(gp.U, X), rtol=1e-12, atol=1e-14
    )
    np.testing.assert_allclose(
        _kern_K(gp.kern, X), gp.kern.K(X), rtol=1e-12, atol=1e-14
    )
    np.testing.assert_allclose(
        _kern_Kdiag(gp.kern, X), gp.kern.Kdiag(X), rtol=1e-12, atol=1e-14
    )


def test_analytic_rbf_gradient_matches_gpy():
    gp = _gp()
    X = gp.X[3:4] + 0.003

    expected = np.column_stack([
        gp.kern.dK_dX2(gp.U, X, i).ravel() for i in range(gp.U.shape[1])
    ])
    np.testing.assert_allclose(gp.dk_dx(X), expected, rtol=1e-10, atol=1e-12)

    # The mean gradient must still match a finite-difference check.
    eps = 1e-6
    analytic = gp.dmu_dx(X).ravel()
    numeric = np.zeros_like(analytic)
    for j in range(X.shape[1]):
        Xp, Xm = X.copy(), X.copy()
        Xp[0, j] += eps
        Xm[0, j] -= eps
        mp, _ = gp.predict_noiseless(Xp, full_cov=True)
        mm, _ = gp.predict_noiseless(Xm, full_cov=True)
        numeric[j] = (mp - mm).item() / (2 * eps)
    np.testing.assert_allclose(analytic, numeric, rtol=1e-6, atol=1e-8)


def test_dk_dx_rejects_batch_inputs():
    gp = _gp()
    X = np.vstack([gp.X[0], gp.X[1]])
    with pytest.raises(ValueError, match="single test input"):
        gp.dk_dx(X)
