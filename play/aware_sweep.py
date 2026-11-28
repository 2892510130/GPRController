"""Sensitivity sweep of the variance-aware cost weights (table 4.4).

First-order GP-MPC on the communication-delay scenario, ``(lambda_v, lambda_omega)``
in {(0,0) baseline, (1,1), (5,5), (10,10)}, with and without observation noise.
The GP model is the one trained for section 4.6.2.

Usage::

    python -m play.aware_sweep --params-tag delay3 --repeats 1
"""

from __future__ import annotations

import argparse
import csv
import os
import time

import numpy as np

from controller import MPCList
from play.mpc_closed_loop import SimulationRunner
from trajectory import Trajectory
from utils import Config

BASE_CONFIG = "config/mpc/linear_mpc_frenet.yaml"
TRACK = "config/track/full_track_opt.yaml"
OUT_DIR = "log/sweep"

SCENARIO = dict(delay_window=2, bad_wheel=False, wheel_left=0.975)
LAMBDAS = (0, 0.05, 0.1, 0.25, 0.5)

FIELDS = ["lam", "noise", "seed", "steps", "mean_lat", "max_lat",
          "mean_head", "max_head", "ms_per_step"]


def read_params(tag: str) -> tuple[str, int, list]:
    data_path, n, params = None, None, []
    for line in open(f"log/sweep/{tag}_params.txt"):
        line = line.strip()
        if line.startswith("data="):
            data_path = line.split("=", 1)[1]
        elif line.startswith("n="):
            n = int(line.split("=", 1)[1])
        elif line.startswith("gp") and "=" in line:
            params.append([float(v) for v in line.split("=", 1)[1].split()])
    return data_path, n, params


def run(lam, noise, seed, track, data_path, n_data, params):
    cfg = Config.from_yaml(BASE_CONFIG)
    for key, value in SCENARIO.items():
        setattr(cfg.sim, key, value)
    cfg.sim.add_noise = noise
    cfg.mpc.R = np.diag([0.1, 0.1])
    cfg.mpc.max_input = [1.0, 1.5]
    cfg.mpc.record_data = False
    cfg.mpc.print_time = False
    cfg.mpc.log_path = f"{OUT_DIR}/aware_{lam}_{'n' if noise else 'nn'}.txt"
    cfg.mpc.variance_aware = lam > 0
    cfg.mpc.lambda_var = [float(lam), float(lam)]
    cfg.mpc.aware_len = 40

    cfg.gaussian.gpr_method = 1
    cfg.gaussian.update_mu_su = True
    cfg.gaussian.add_mu_su = True
    cfg.gaussian.update_method = "vfe"
    cfg.gaussian.lambda_ = 1.0
    cfg.gaussian.path = data_path
    cfg.gaussian.total_data_number = n_data
    cfg.gaussian.params = params
    cfg.gaussian.read_model = False
    np.random.seed(seed)

    t0 = time.perf_counter()
    mpc = MPCList[cfg.mpc.mpc_name](cfg, track.path, track.speed_profile)
    runner = SimulationRunner(mpc, cfg)
    runner.run_full_simulation()
    sim_s = time.perf_counter() - t0
    mpc.logger.close_log()

    err = np.abs(np.asarray(runner.state, dtype=float))
    steps = int(err.shape[0])
    return dict(
        lam=lam, noise=int(noise), seed=seed, steps=steps,
        mean_lat=float(err[:, 0].mean()), max_lat=float(err[:, 0].max()),
        mean_head=float(err[:, 1].mean()), max_head=float(err[:, 1].max()),
        ms_per_step=round(1000.0 * sim_s / max(steps, 1), 2),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--params-tag", default="delay3")
    ap.add_argument("--repeats", type=int, default=1)
    args = ap.parse_args()

    data_path, n_data, params = read_params(args.params_tag)
    track = Trajectory(TRACK)
    os.makedirs(OUT_DIR, exist_ok=True)

    csv_path = os.path.join(OUT_DIR, "aware_sweep.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for noise in (True, False):
            for lam in LAMBDAS:
                for rep in range(args.repeats):
                    print(f"[lam={lam} noise={noise} seed={rep}]", flush=True)
                    row = run(lam, noise, rep, track, data_path, n_data, params)
                    writer.writerow(row)
                    fh.flush()
                    print("    mean|lat|={:.6f} max|lat|={:.6f} mean|head|={:.6f} "
                          "steps={} {:.1f} ms/step".format(
                              row["mean_lat"], row["max_lat"], row["mean_head"],
                              row["steps"], row["ms_per_step"]), flush=True)
    print(f"\nresults -> {csv_path}")


if __name__ == "__main__":
    main()