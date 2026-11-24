import uuid
from pathlib import Path

import numpy as np
import pytest

from model.gp_model import GaussianModel
from utils import Config

ROOT = Path(__file__).resolve().parents[1]
BUILD_DIR = ROOT / "build"


@pytest.fixture
def model_prefix():
    """Workspace-local file prefix (the system temp dir is sandboxed here)."""
    BUILD_DIR.mkdir(exist_ok=True)
    prefix = BUILD_DIR / f"gp-model-{uuid.uuid4().hex}"
    try:
        yield prefix
    finally:
        for path in BUILD_DIR.glob(f"{prefix.name}*"):
            path.unlink(missing_ok=True)


def _gp_config(prefix: Path, *, sparse_method: int = 3,
               normalize_y: bool = False, read_model: bool = False):
    config = Config.from_yaml(str(ROOT / "config/mpc/linear_mpc_frenet.yaml")).gaussian
    config.sparse_method = sparse_method
    config.normalize_y = normalize_y
    config.read_model = read_model
    config.save_path = str(prefix)
    config.load_path = str(prefix)
    config.path = str(ROOT / config.path)
    return config


def _save_all(model: GaussianModel) -> None:
    for i, gp in enumerate(model.gps):
        gp.save_model(f"_{i}")


def _save_legacy(gp, path: Path, kind: str) -> None:
    """Write the pre-versioning flat np.save layout."""
    with open(path, "wb") as file:
        for value in (gp.N, gp.M, gp.variance, gp.lambda_, gp.X, gp.Y, gp.U):
            np.save(file, value)
        if gp.gaussian_opts.normalize_y:
            np.save(file, gp.normalizer.mean)
            np.save(file, gp.normalizer.std)
        for value in (gp.Luu, gp.mu, gp.Su, gp.Kuf, gp.W):
            np.save(file, value)
        if kind == "FITC":
            np.save(file, gp.diag)
            np.save(file, gp.diag_inv)
        for value in (gp.W_d_inv_y, gp.L):
            np.save(file, value)


@pytest.mark.parametrize("normalize_y", [False, True])
def test_save_load_roundtrip(model_prefix, normalize_y):
    model = GaussianModel(_gp_config(model_prefix, normalize_y=normalize_y))
    Xs = model.gps[0].X[:5] + 0.01
    expected = [gp.predict_noiseless(Xs, full_cov=True) for gp in model.gps]
    _save_all(model)

    loaded = GaussianModel(_gp_config(model_prefix, normalize_y=normalize_y, read_model=True))
    for (mu, var), gp in zip(expected, loaded.gps):
        assert gp.cholesky_valid
        loaded_mu, loaded_var = gp.predict_noiseless(Xs, full_cov=True)
        np.testing.assert_allclose(loaded_mu, mu, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(loaded_var, var, rtol=1e-10, atol=1e-12)


def test_save_load_roundtrip_dtc(model_prefix):
    model = GaussianModel(_gp_config(model_prefix, sparse_method=0))
    Xs = model.gps[0].X[:5] + 0.01
    expected = [gp.predict_noiseless(Xs, full_cov=True) for gp in model.gps]
    _save_all(model)

    loaded = GaussianModel(_gp_config(model_prefix, sparse_method=0, read_model=True))
    for (mu, var), gp in zip(expected, loaded.gps):
        assert gp.cholesky_valid
        loaded_mu, loaded_var = gp.predict_noiseless(Xs, full_cov=True)
        np.testing.assert_allclose(loaded_mu, mu, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(loaded_var, var, rtol=1e-10, atol=1e-12)


def test_quick_mode_is_mu_su_only_and_can_be_synced(model_prefix):
    gp = GaussianModel(_gp_config(model_prefix)).gps[0]
    Xs = gp.X[:4] + 0.01
    mu_before, var_before = gp.predict_noiseless(Xs, full_cov=True)

    gp.add_new_inducing_quick(gp.X[:1] + 0.02)
    assert gp.cholesky_valid is False

    mu_quick, var_quick = gp.predict_noiseless(Xs, full_cov=True)
    assert mu_quick.shape == mu_before.shape
    assert np.all(np.isfinite(mu_quick)) and np.all(np.isfinite(var_quick))
    np.testing.assert_allclose(gp.predict(Xs)[0], mu_quick)

    with pytest.raises(RuntimeError, match="mu_su"):
        gp.predict(Xs, method="cholesky")
    with pytest.raises(RuntimeError, match="mu_su"):
        gp.add_new_data(gp.X[:1], gp.Y[:1])
    with pytest.raises(RuntimeError, match="mu_su"):
        gp.add_new_inducing(gp.X[:1] + 0.03)

    gp.sync_cholesky()
    assert gp.cholesky_valid is True
    mu_chol, var_chol = gp.predict(Xs, method="cholesky")
    assert np.all(np.isfinite(mu_chol)) and np.all(np.isfinite(var_chol))
    # mu_su keeps the incremental (approximate) posterior, so the Cholesky
    # prediction is not expected to match it bit-for-bit.


def test_quick_mode_save_load_keeps_mu_su_state(model_prefix):
    model = GaussianModel(_gp_config(model_prefix))
    gp = model.gps[0]
    gp.add_new_inducing_quick(gp.X[:1] + 0.02)
    Xs = gp.X[:4] + 0.01
    expected_mu, expected_var = gp.predict_noiseless(Xs, full_cov=True)
    _save_all(model)

    loaded = GaussianModel(_gp_config(model_prefix, read_model=True))
    gp_loaded = loaded.gps[0]
    assert gp_loaded.cholesky_valid is False
    loaded_mu, loaded_var = gp_loaded.predict_noiseless(Xs, full_cov=True)
    np.testing.assert_allclose(loaded_mu, expected_mu, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(loaded_var, expected_var, rtol=1e-10, atol=1e-12)

    with pytest.raises(RuntimeError, match="mu_su"):
        gp_loaded.predict(Xs, method="cholesky")
    gp_loaded.sync_cholesky()
    assert gp_loaded.cholesky_valid is True
    assert np.all(np.isfinite(gp_loaded.predict(Xs, method="cholesky")[0]))


def test_full_add_inducing_save_load(model_prefix):
    model = GaussianModel(_gp_config(model_prefix))
    gp = model.gps[0]
    gp.add_new_inducing(gp.X[:1] + 0.02)
    Xs = gp.X[:4] + 0.01
    expected_mu, expected_var = gp.predict_noiseless(Xs, full_cov=True)
    _save_all(model)

    loaded = GaussianModel(_gp_config(model_prefix, read_model=True))
    gp_loaded = loaded.gps[0]
    assert gp_loaded.cholesky_valid is True
    loaded_mu, loaded_var = gp_loaded.predict_noiseless(Xs, full_cov=True)
    np.testing.assert_allclose(loaded_mu, expected_mu, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(loaded_var, expected_var, rtol=1e-10, atol=1e-12)
    cholesky_mu, _ = gp_loaded.predict(Xs, method="cholesky")
    assert np.all(np.isfinite(cholesky_mu))


def test_load_rejects_mismatched_model_metadata(model_prefix):
    model = GaussianModel(_gp_config(model_prefix))
    _save_all(model)

    bad_kernel = _gp_config(model_prefix, read_model=True)
    bad_kernel.params[0][1] = float(bad_kernel.params[0][1]) * 2.0
    with pytest.raises(ValueError, match="hyperparameter"):
        GaussianModel(bad_kernel)

    bad_normalize = _gp_config(model_prefix, normalize_y=True, read_model=True)
    with pytest.raises(ValueError, match="normalize_y"):
        GaussianModel(bad_normalize)

    bad_method = _gp_config(model_prefix, sparse_method=0, read_model=True)
    with pytest.raises(ValueError, match="FITC"):
        GaussianModel(bad_method)


def test_load_legacy_model_file(model_prefix):
    model = GaussianModel(_gp_config(model_prefix))
    legacy_prefix = Path(str(model_prefix) + "-legacy")
    for i, gp in enumerate(model.gps):
        _save_legacy(gp, Path(f"{legacy_prefix}_{i}.npy"), "FITC")

    legacy_config = _gp_config(model_prefix, read_model=True)
    legacy_config.load_path = str(legacy_prefix)
    loaded = GaussianModel(legacy_config)

    Xs = loaded.gps[0].X[:3] + 0.01
    mu, var = loaded.gps[0].predict_noiseless(Xs, full_cov=True)
    assert np.all(np.isfinite(mu)) and np.all(np.isfinite(var))
    assert loaded.gps[0].cholesky_valid is True
