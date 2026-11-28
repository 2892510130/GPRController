"""Variance-aware weight sweep in scenarios where the uncertainty is informative.

Two cases are supported:

``cross``     GP trained on the original loop (delay), deployed on the mirrored
              loop (delay) -- the regressors fall outside the training support so
              the predicted variance is larger and unevenly distributed;
``unlearn0``  GP trained on the wheel-fault recording, deployed on the normal
              condition without online update (round 0 of section 4.6.2.3), i.e.
              a wrong prior with heterogeneous uncertainty.

Usage::

    python -m play.aware_scenarios --case cross
    python -m play.aware_scenarios --case unlearn0
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
OUT_DIR = "log/sweep"
LAMBDAS = (0, 1, 5, 10)

CASES = {
    "cross": dict(
        params="delay3",
        track="config/track/full_track_mirror.yaml",
        scenario=dict(delay_window=2, bad_wheel=False, wheel_left=0.975),
        gpr_method=1, update=True, add=True, noise=True,
    ),
    "unlearn0": dict(
        params="bad_wheel925",
        track="config/track/full_track_opt.yaml",
        scenario=dict(delay_window=1, bad_wheel=False, wheel_left=0.975),
        gpr_method=1, update=False, add=False, noise=True,
    ),
}

FIELDS = ["case", "lam", "seed", "steps", "mean_lat", "max_lat",
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


def run(case, lam, seed, track, data_path, n_data, params):
    spec = CASES[case]
    cfg = Config.from_yaml(BASE_CONFIG)
    for key, value in spec["scenario"].items():
        setattr(cfg.sim, key, value)
    cfg.sim.add_noise = spec["noise"]
    cfg.mpc.R = np.diag([0.1, 0.1])
    cfg.mpc.max_input = [1.0, 1.5]
    cfg.mpc.record_data = False
    cfg.mpc.print_time = False
    cfg.mpc.log_path = f"{OUT_DIR}/aws_{case}_{lam}.txt"
    cfg.mpc.variance_aware = lam > 0
    cfg.mpc.lambda_var = [float(lam), float(lam)]
    cfg.mpc.aware_len = 40

    cfg.gaussian.gpr_method = spec["gpr_method"]
    cfg.gaussian.update_mu_su = spec["update"]
    cfg.gaussian.add_mu_su = spec["add"]
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
        case=case, lam=lam, seed=seed, steps=steps,
        mean_lat=float(err[:, 0].mean()), max_lat=float(err[:, 0].max()),
        mean_head=float(err[:, 1].mean()), max_head=float(err[:, 1].max()),
        ms_per_step=round(1000.0 * sim_s / max(steps, 1), 2),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", required=True, choices=list(CASES))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data_path, n_data, params = read_params(CASES[args.case]["params"])
    track = Trajectory(CASES[args.case]["track"])
    os.makedirs(OUT_DIR, exist_ok=True)

    csv_path = os.path.join(OUT_DIR, f"aware_{args.case}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for lam in LAMBDAS:
            print(f"[{args.case} lam={lam}]", flush=True)
            row = run(args.case, lam, args.seed, track, data_path, n_data, params)
            writer.writerow(row)
            fh.flush()
            print("    mean|lat|={:.6f} max|lat|={:.6f} mean|head|={:.6f} "
                  "steps={} {:.1f} ms/step".format(
                      row["mean_lat"], row["max_lat"], row["mean_head"],
                      row["steps"], row["ms_per_step"]), flush=True)
    print(f"\nresults -> {csv_path}")


if __name__ == "__main__":
    main()