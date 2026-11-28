"""Dump closed-loop trajectories for the figure set (nominal / zero-order / first-order)."""

from __future__ import annotations

import argparse
import os

import numpy as np

from controller import MPCList
from play.mpc_closed_loop import SimulationRunner
from trajectory import Trajectory
from utils import Config

BASE_CONFIG = "config/mpc/linear_mpc_frenet.yaml"
TRACK_CONFIG = "config/track/full_track_opt.yaml"
OUT_DIR = "log/sweep/traj"

SCENARIOS = {
    "delay": dict(delay_window=2, bad_wheel=False, wheel_left=0.975),
    "bad_wheel": dict(delay_window=1, bad_wheel=True, wheel_left=0.925),
}

RUNS = {
    "nominal": (-1, False, False),
    "gp0": (0, True, False),
    "gp1": (1, True, True),
}


def read_params(tag: str) -> tuple[str, int, list]:
    path = f"log/sweep/{tag}_params.txt"
    data_path, n, params = None, None, []
    for line in open(path):
        line = line.strip()
        if line.startswith("data="):
            data_path = line.split("=", 1)[1]
        elif line.startswith("n="):
            n = int(line.split("=", 1)[1])
        elif line.startswith("gp") and "=" in line:
            params.append([float(v) for v in line.split("=", 1)[1].split()])
    return data_path, n, params


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenario", required=True, choices=list(SCENARIOS))
    ap.add_argument("--tag", required=True, help="params tag, e.g. delay3 / bad_wheel925")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data_path, n_data, params = read_params(args.tag)
    os.makedirs(OUT_DIR, exist_ok=True)
    track = Trajectory(TRACK_CONFIG)

    for name, (gpr_method, update_mu_su, add_mu_su) in RUNS.items():
        cfg = Config.from_yaml(BASE_CONFIG)
        for key, value in SCENARIOS[args.scenario].items():
            setattr(cfg.sim, key, value)
        cfg.mpc.R = np.diag([0.1, 0.1])
        cfg.mpc.max_input = [1.0, 1.5]
        cfg.mpc.record_data = False
        cfg.mpc.print_time = False
        cfg.mpc.log_path = f"{OUT_DIR}/log_{args.scenario}_{name}.txt"
        cfg.gaussian.gpr_method = gpr_method
        cfg.gaussian.update_mu_su = update_mu_su
        cfg.gaussian.add_mu_su = add_mu_su
        cfg.gaussian.path = data_path
        cfg.gaussian.total_data_number = n_data
        cfg.gaussian.params = params
        cfg.gaussian.read_model = False
        np.random.seed(args.seed)

        mpc = MPCList[cfg.mpc.mpc_name](cfg, track.path, track.speed_profile)
        runner = SimulationRunner(mpc, cfg)
        runner.run_full_simulation()
        mpc.logger.close_log()

        state = np.asarray(runner.state, dtype=float)
        sim_path = np.asarray(runner.sim_path, dtype=float).reshape(-1)
        sim_path = sim_path.reshape(len(state), -1)
        control = np.asarray(runner.control, dtype=float)
        np.savez(
            f"{OUT_DIR}/{args.scenario}_{name}.npz",
            state=state, sim_path=sim_path, control=control,
            ref=np.asarray(track.path, dtype=float),
        )
        print(
            f"{args.scenario}/{name}: steps={state.shape[0]} "
            f"mean|lat|={np.abs(state[:,0]).mean():.5f} max|lat|={np.abs(state[:,0]).max():.5f}"
        )


if __name__ == "__main__":
    main()