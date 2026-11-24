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


class _FakeSolver:
    def __init__(self, stats):
        self._stats = stats

    def stats(self):
        return self._stats


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


def _initial_state(path, speed_profile):
    # The controller always receives the full Cartesian measurement
    # [x, y, yaw, v, w]; Frenet models convert it internally.
    return np.array([
        [path[0, 0]],
        [path[0, 1]],
        [path[0, 2]],
        [speed_profile[0, 2]],
        [speed_profile[0, 4]],
    ])


def _solve_once(mpc, config):
    path, speed_profile = _circle_path()
    mpc.initialize(_initial_state(path, speed_profile), config.mpc.control_dt)
    return mpc.solve()


@pytest.mark.parametrize(
    "model_name,nx",
    [
        ("Kinematic Frenet", 4),
        ("Kinematic Frenet Gaussian", 4),
        ("Kinematic Augmented Frenet Gaussian", 12),
        ("Kinematic Cartesian", 5),
        ("Kinematic Cartesian Delay", 5),
        ("Kinematic Cartesian Gaussian", 5),
        ("Kinematic Augmented Cartesian Gaussian", 13),
    ],
)
def test_controller_models_solve(model_name, nx, monkeypatch):
    monkeypatch.setattr(linear_mpc_module, "Logger", _DummyLogger)

    config = _config(model_name, nx)
    path, speed_profile = _circle_path()
    mpc = LinearMPC(config, path, speed_profile)
    try:
        u = _solve_once(mpc, config)
    finally:
        mpc.logger.close_log()

    assert u.shape == (mpc.model.nu * MPC_WINDOW, 1)
    assert np.all(np.isfinite(u))
    assert np.max(np.abs(u)) <= max(config.mpc.max_input) + 1e-6


@pytest.mark.parametrize(
    "model_name,nx",
    [
        ("Kinematic Augmented Frenet Gaussian", 12),
        ("Kinematic Augmented Cartesian Gaussian", 13),
    ],
)
def test_variance_aware_solve(model_name, nx, monkeypatch):
    monkeypatch.setattr(linear_mpc_module, "Logger", _DummyLogger)

    config = _config(model_name, nx)
    config.mpc.mpcWindow = 6
    config.mpc.aware_len = 6
    config.mpc.variance_aware = True
    config.mpc.lambda_var = [1.0, 1.0]
    config.gaussian.gpr_method = 1

    path, speed_profile = _circle_path()
    mpc = LinearMPC(config, path, speed_profile)
    try:
        u = _solve_once(mpc, config)
    finally:
        mpc.logger.close_log()

    assert np.all(np.isfinite(u))
    assert mpc.A_sigma.shape == (1, nx * config.mpc.mpcWindow)
    assert mpc.B_sigma.shape == (1, mpc.model.nu * config.mpc.mpcWindow)


def test_variance_aware_requires_first_order_gp(monkeypatch):
    monkeypatch.setattr(linear_mpc_module, "Logger", _DummyLogger)

    config = _config("Kinematic Augmented Frenet Gaussian", 12)
    config.mpc.mpcWindow = 4
    config.mpc.variance_aware = True
    config.gaussian.gpr_method = 0

    path, speed_profile = _circle_path()
    with pytest.raises(ValueError, match="gpr_method"):
        LinearMPC(config, path, speed_profile)


def test_variance_aware_requires_augmented_model(monkeypatch):
    monkeypatch.setattr(linear_mpc_module, "Logger", _DummyLogger)

    config = _config("Kinematic Frenet", 4)
    config.mpc.mpcWindow = 4
    config.mpc.variance_aware = True
    config.gaussian.gpr_method = 1

    path, speed_profile = _circle_path()
    with pytest.raises(ValueError, match="variance_aware"):
        LinearMPC(config, path, speed_profile)


def test_non_gaussian_controller_without_gaussian_section(monkeypatch):
    monkeypatch.setattr(linear_mpc_module, "Logger", _DummyLogger)

    config = Config.from_yaml(str(CONFIG_PATH))
    config.gaussian = None
    config.mpc.model = "Kinematic Frenet"
    config.mpc.mpcWindow = 6
    config.mpc.Q = np.diag([1.0] * 4)
    config.mpc.Q_final = np.diag([1.0] * 4)
    config.mpc.record_data = False
    config.mpc.propagete_cov = False

    path, speed_profile = _circle_path()
    mpc = LinearMPC(config, path, speed_profile)
    try:
        u = _solve_once(mpc, config)
    finally:
        mpc.logger.close_log()

    assert np.all(np.isfinite(u))


def test_augmented_models_require_at_least_two_steps():
    config = _config("Kinematic Augmented Frenet Gaussian", 12)
    config.gaussian.input_window = 1

    path, speed_profile = _circle_path()
    with pytest.raises(ValueError, match="input_window"):
        LinearMPC(config, path, speed_profile)


def test_solver_is_reused_between_solves(monkeypatch):
    monkeypatch.setattr(linear_mpc_module, "Logger", _DummyLogger)

    config = _config("Kinematic Frenet", 4)
    config.mpc.mpcWindow = 6
    path, speed_profile = _circle_path()
    mpc = LinearMPC(config, path, speed_profile)
    try:
        _solve_once(mpc, config)
        first_solver = mpc._cached_solver
        assert first_solver is not None
        _solve_once(mpc, config)
        assert mpc._cached_solver is first_solver
    finally:
        mpc.logger.close_log()


def test_solver_result_check_rejects_failure():
    with pytest.raises(RuntimeError, match="maximum iterations"):
        LinearMPC._check_solver_result(
            _FakeSolver({"success": False, "return_status": "maximum iterations reached"}),
            {"x": np.zeros((2, 1))},
            "test",
        )


def test_solver_result_check_rejects_non_finite():
    with pytest.raises(RuntimeError, match="non-finite"):
        LinearMPC._check_solver_result(
            _FakeSolver({"success": True, "return_status": "solved"}),
            {"x": np.array([[np.nan]])},
            "test",
        )
