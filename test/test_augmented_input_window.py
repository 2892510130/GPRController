import uuid
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import controller.linear_mpc as linear_mpc_module

from controller import LinearMPC
from model.frenet_gp_model import AugmentedFrenetGaussianModel
from utils import Config

ROOT = Path(__file__).resolve().parents[1]
BUILD_DIR = ROOT / "build"


class _DummyLogger:
    def __init__(self, _file_name: str) -> None:
        pass

    def debug(self, message, data) -> None:
        pass

    def info(self, message, data) -> None:
        pass

    def empty_line(self, times: int) -> None:
        pass

    def close_log(self) -> None:
        pass


@pytest.fixture
def synthetic_gp_data():
    BUILD_DIR.mkdir(exist_ok=True)
    path = BUILD_DIR / f"augmented-data-{uuid.uuid4().hex}.txt"
    try:
        yield path
    finally:
        path.unlink(missing_ok=True)


def _write_synthetic_data(path: Path, feature_size: int, total: int = 40, inducing: int = 6):
    rng = np.random.default_rng(0)
    train = np.hstack([
        rng.normal(size=(total, feature_size)),
        rng.normal(scale=0.01, size=(total, 2)),
    ])
    inducing_inputs = rng.normal(size=(2 * inducing, feature_size))
    with open(path, "w", encoding="utf-8") as file:
        for row in train:
            file.write(" ".join(map(str, row)) + "\n")
        for row in inducing_inputs:
            file.write(" ".join(map(str, row)) + "\n")


def _config(model_name: str, window: int, nx: int, data_path: Path) -> Config:
    config = Config.from_yaml(str(ROOT / "config/mpc/linear_mpc_frenet.yaml"))
    config.gaussian.path = str(data_path)
    config.gaussian.total_data_number = 40
    config.gaussian.inducing_data_number = 6
    config.gaussian.input_window = window
    config.gaussian.sparse_method = 3
    config.gaussian.read_model = False
    config.gaussian.add_mu_su = False
    config.gaussian.update_mu_su = False
    config.gaussian.gpr_method = 0
    config.gaussian.normalize_y = False
    config.gaussian.enable_test = False
    config.gaussian.params = [
        [1e-4, 1.0] + [1.0] * (4 * window),
        [1e-4, 1.0] + [1.0] * (4 * window),
    ]
    config.mpc.model = model_name
    config.mpc.mpcWindow = 4
    config.mpc.Q = np.diag([1.0] * nx)
    config.mpc.Q_final = np.diag([1.0] * nx)
    config.mpc.record_data = False
    config.mpc.propagete_cov = False
    config.mpc.variance_aware = False
    return config


def _circle_path(n: int = 200, radius: float = 5.0, velocity: float = 1.0):
    s = np.linspace(0.0, 10.0, n)
    yaw = s / radius
    path = np.column_stack([
        radius * np.sin(yaw),
        radius * (1.0 - np.cos(yaw)),
        yaw,
        np.full(n, 1.0 / radius),
    ])
    speed_profile = np.column_stack([
        s / velocity,
        s,
        np.full(n, velocity),
        np.zeros(n),
        np.full(n, velocity / radius),
        np.zeros(n),
    ])
    return path, speed_profile


def _initial_state(path, speed_profile):
    return np.array([
        [path[0, 0]],
        [path[0, 1]],
        [path[0, 2]],
        [speed_profile[0, 2]],
        [speed_profile[0, 4]],
    ])


@pytest.mark.parametrize("window", [2, 4])
@pytest.mark.parametrize(
    "model_name,base_nx",
    [
        ("Kinematic Augmented Frenet Gaussian", 4),
        ("Kinematic Augmented Cartesian Gaussian", 5),
    ],
)
def test_augmented_models_support_other_windows(
    model_name, base_nx, window, synthetic_gp_data, monkeypatch
):
    monkeypatch.setattr(linear_mpc_module, "Logger", _DummyLogger)

    _write_synthetic_data(synthetic_gp_data, feature_size=4 * window)
    nx = base_nx + 4 * (window - 1)
    config = _config(model_name, window, nx, synthetic_gp_data)

    path, speed_profile = _circle_path()
    mpc = LinearMPC(config, path, speed_profile)
    try:
        mpc.initialize(_initial_state(path, speed_profile), config.mpc.control_dt)
        assert mpc.augmented_u0.shape == (2 * window, 1)
        assert mpc.x0.shape == (nx, 1)
        u = mpc.solve()
    finally:
        mpc.logger.close_log()

    assert u.shape == (mpc.model.nu * config.mpc.mpcWindow, 1)
    assert np.all(np.isfinite(u))
    assert mpc.pred_traj.shape == (config.mpc.mpcWindow, 5)


def test_change_vector_to_model2_rejects_inconsistent_buffers():
    # Old W=4 buffer sizes: state 10 + control history 7 != nx 16.
    with pytest.raises(ValueError, match="must equal nx"):
        AugmentedFrenetGaussianModel.change_vector_to_model2(
            SimpleNamespace(nx=16), np.zeros((10, 1)), np.zeros((9, 1))
        )
