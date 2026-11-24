from types import SimpleNamespace

import numpy as np
import pytest

from controller.linear_mpc import LinearMPC


def _straight_path(n: int = 200, length: float = 20.0) -> np.ndarray:
    x = np.linspace(0.0, length, n)
    return np.column_stack([
        x,
        np.zeros(n),
        np.zeros(n),
        np.zeros(n),
    ])


def _state(x, y, yaw=0.0, v=1.0, w=0.0):
    return np.array([[x], [y], [yaw], [v], [w]])


def test_compute_error_returns_scalars_and_local_index():
    path = _straight_path()
    mpc = SimpleNamespace(path=path, index=100)

    lat, head, index = LinearMPC.compute_error(mpc, _state(path[100, 0], path[100, 1]))

    assert isinstance(lat, float)
    assert isinstance(head, float)
    assert index == 100
    assert abs(lat) < 1e-12
    assert abs(head) < 1e-12


def test_compute_error_lateral_offset_is_signed_scalar():
    path = _straight_path()
    mpc = SimpleNamespace(path=path, index=100)

    lat, head, index = LinearMPC.compute_error(mpc, _state(path[100, 0], 0.2))

    assert index == 100
    assert lat == pytest.approx(0.2, abs=1e-12)
    assert isinstance(lat, float)


def test_compute_error_falls_back_to_global_search():
    path = _straight_path()
    mpc = SimpleNamespace(path=path, index=0)

    lat, head, index = LinearMPC.compute_error(mpc, _state(path[150, 0], path[150, 1]))

    assert index == 150
    assert isinstance(lat, float)
    assert isinstance(head, float)
