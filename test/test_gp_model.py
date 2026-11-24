import numpy as np

from utils import Config

CONFIG_PATH = "config/mpc/linear_mpc_frenet.yaml"


def _check_predictions(model):
    for gp in model.gps:
        mu, var = gp.predict_noiseless(gp.X[:10], full_cov=True)
        assert mu.shape == (10, 1)
        assert var.shape == (10,)
        assert np.all(np.isfinite(mu))
        assert np.all(np.isfinite(var))


def test_gp_model_error():
    config = Config.from_yaml(CONFIG_PATH)
    gp_config = config.gaussian
    gp_config.read_model = False

    from model.gp_model import GaussianModel, test_gaussian_model

    print(gp_config.path, gp_config.total_data_number)
    model = GaussianModel(gp_config)
    test_gaussian_model(model)
    _check_predictions(model)

    # The DTC implementation shares the same online-update machinery.
    gp_config.sparse_method = 2
    model = GaussianModel(gp_config)
    test_gaussian_model(model)
    _check_predictions(model)
