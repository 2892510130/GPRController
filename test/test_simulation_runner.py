from pathlib import Path

import numpy as np

import controller.linear_mpc as linear_mpc_module

from controller import LinearMPC
from play.mpc_closed_loop import SimulationRunner
from trajectory import Trajectory
from utils import Config

ROOT = Path(__file__).resolve().parents[1]


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


def _config(start_index: int) -> Config:
    config = Config.from_yaml(str(ROOT / "config/mpc/linear_mpc_frenet.yaml"))
    config.mpc.model = "Kinematic Frenet"
    config.mpc.mpcWindow = 5
    config.mpc.Q = np.diag([1.0] * 4)
    config.mpc.Q_final = np.diag([1.0] * 4)
    config.mpc.record_data = False
    config.mpc.propagete_cov = False
    config.sim.start_index = start_index
    config.sim.SIM_TIMES = 5
    return config


def test_start_index_places_vehicle_on_track(monkeypatch):
    monkeypatch.setattr(linear_mpc_module, "Logger", _DummyLogger)

    config = _config(start_index=25)
    track = Trajectory(str(ROOT / "config/track/full_track_opt.yaml"))
    mpc = LinearMPC(config, track.path, track.speed_profile)
    runner = SimulationRunner(mpc, config)
    try:
        index = config.sim.start_index
        expected = np.array([
            track.path[index, 0],
            track.path[index, 1],
            track.path[index, 2],
            track.speed_profile[index, 2],
            track.speed_profile[index, 4],
        ])
        np.testing.assert_allclose(runner.sim_x0[:5, 0], expected)

        state = runner.run_step(runner.get_initial_state())
        assert np.all(np.isfinite(state))
    finally:
        mpc.logger.close_log()
