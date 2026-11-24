from pathlib import Path

import numpy as np
import pytest

import controller.linear_mpc as linear_mpc_module

from controller import LinearMPC
from utils import Config

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config/mpc/linear_mpc_frenet.yaml"
MPC_WINDOW = 8


class _DummyLogger:
    """Logger stand-in so the tests do not touch the filesystem."""

    def __init__(self, _file_name: str) -> None:
        self.lines = []

    def debug(self, message, data) -> None:
        self.lines.append((message, data))

    def info(self, message, data) -> None:
        self.lines.append((message, data))

    def empty_line(self, times: int) -> None:
        pass

    def close_log(self) -> None:
        pass


def _config(model_name: str, nx: int) -> Config:
    config = Config.from_yaml(str(CONFIG_PATH))
    config.mpc.model = model_name
    config.mpc.mpcWindow = MPC_WINDOW
    config.mpc.Q = np.diag([1.0] * nx)
    config.mpc.Q_final = np.diag([1.0] * nx)
    config.mpc.record_data = False
    config.mpc.propagete_cov = False
    config.mpc.variance_aware = False
    config.gaussian.read_model = False
    # Keep the GP data lookup independent of the current working directory.
    if not Path(config.gaussian.path).is_absolute():
        config.gaussian.path = str(ROOT / config.gaussian.path)
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


@pytest.mark.parametrize(
    "model_name,nx",
    [
        ("Kinematic Frenet", 4),
        ("Kinematic Frenet Gaussian", 4),
        ("Kinematic Augmented Frenet Gaussian", 12),
        ("Kinematic Cartesian Gaussian", 5),
        ("Kinematic Augmented Cartesian Gaussian", 13),
    ],
)
def test_controller_models_solve(model_name, nx, monkeypatch):
    monkeypatch.setattr(linear_mpc_module, "Logger", _DummyLogger)

    config = _config(model_name, nx)
    path, speed_profile = _circle_path()
    mpc = LinearMPC(config, path, speed_profile)

    # The controller always receives the full Cartesian measurement
    # [x, y, yaw, v, w]; Frenet models convert it internally.
    x0 = np.array([
        [path[0, 0]],
        [path[0, 1]],
        [path[0, 2]],
        [speed_profile[0, 2]],
        [speed_profile[0, 4]],
    ])

    try:
        mpc.initialize(x0, config.mpc.control_dt)
        u = mpc.solve()
    finally:
        mpc.logger.close_log()

    assert u.shape == (mpc.model.nu * MPC_WINDOW, 1)
    assert np.all(np.isfinite(u))
    assert np.max(np.abs(u)) <= max(config.mpc.max_input) + 1e-6
