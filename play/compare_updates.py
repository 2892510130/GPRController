"""Closed-loop comparison of the online update rules.

Seven configurations share the same trained per-scenario GP model:

    nominal      nominal MPC (no GP)
    frozen       zero-order GP-MPC, posterior fixed after training
    osvgp        zero-order GP-MPC, rank-one variational (Kalman) update
    osvgp_add    osvgp + threshold-triggered inducing-point expansion
    first        first-order GP-MPC, osvgp_add rule
    bijl         zero-order GP-MPC, non-variational FITC rank-one update
    agp          zero-order GP-MPC, fixed-length sliding window of training
                 data with the posterior recomputed every step

Usage::

    python -m play.compare_updates --scenario delay --params-tag delay3 --repeats 3
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
TRACK_CONFIG = "config/track/full_track_opt.yaml"
OUT_DIR = "log/sweep"

SCENARIOS = {
    "delay": dict(delay_window=2, bad_wheel=False, wheel_left=0.975),
    "bad_wheel": dict(delay_window=1, bad_wheel=True, wheel_left=0.925),
}

# name -> (gpr_method, update_mu_su, add_mu_su, update_method, lambda_)
VARIANTS = {
    "nominal": (-1, False, False, "vfe", 1.0),
    "frozen": (0, False, False, "vfe", 1.0),
    "osvgp": (0, True, False, "vfe", 1.0),
    "osvgp_add": (0, True, True, "vfe", 1.0),
    "first": (1, True, True, "vfe", 1.0),
    "bijl": (0, True, False, "bijl", 1.0),
    "agp_window": (0, True, False, "window", 1.0),
    "agp": (0, True, False, "agp", 0.99),
}

FIELDS = ["variant", "scenario", "seed", "steps", "terminated",
          "mean_lat", "max_lat", "mean_head", "max_head", "m_final",
          "init_s", "sim_s", "ms_per_step"]


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


def run_variant(scenario, variant, seed, track, data_path, n_data, params):
    cfg = Config.from_yaml(BASE_CONFIG)
    for key, value in SCENARIOS[scenario].items():
        setattr(cfg.sim, key, value)
    cfg.mpc.R = np.diag([0.1, 0.1])
    cfg.mpc.max_input = [1.0, 1.5]
    cfg.mpc.record_data = False
    cfg.mpc.print_time = False
    cfg.mpc.log_path = f"{OUT_DIR}/upd_{scenario}_{variant}_{seed}.txt"

    gpr_method, update_mu_su, add_mu_su, update_method, lambda_ = VARIANTS[variant]
    cfg.gaussian.gpr_method = gpr_method
    cfg.gaussian.update_mu_su = update_mu_su
    cfg.gaussian.add_mu_su = add_mu_su
    cfg.gaussian.update_method = update_method
    cfg.gaussian.lambda_ = lambda_
    cfg.gaussian.path = data_path
    cfg.gaussian.total_data_number = n_data
    cfg.gaussian.params = params
    cfg.gaussian.read_model = False
    np.random.seed(seed)

    t0 = time.perf_counter()
    mpc = MPCList[cfg.mpc.mpc_name](cfg, track.path, track.speed_profile)
    init_s = time.perf_counter() - t0
    runner = SimulationRunner(mpc, cfg)
    t1 = time.perf_counter()
    runner.run_full_simulation()
    sim_s = time.perf_counter() - t1
    mpc.logger.close_log()

    err = np.abs(np.asarray(runner.state, dtype=float))
    steps = int(err.shape[0])
    try:
        m_final = "/".join(str(gp.M) for gp in mpc.model.gaussian_model.gps)
    except Exception:
        m_final = "-"
    return dict(
        variant=variant, scenario=scenario, seed=seed, steps=steps,
        terminated=bool(getattr(runner, "simulation_finished", False)),
        mean_lat=float(err[:, 0].mean()), max_lat=float(err[:, 0].max()),
        mean_head=float(err[:, 1].mean()), max_head=float(err[:, 1].max()),
        m_final=m_final, init_s=round(init_s, 2), sim_s=round(sim_s, 2),
        ms_per_step=round(1000.0 * sim_s / steps, 2),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenario", required=True, choices=list(SCENARIOS))
    ap.add_argument("--params-tag", required=True)
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--repeats", type=int, default=1)
    args = ap.parse_args()

    data_path, n_data, params = read_params(args.params_tag)
    track = Trajectory(TRACK_CONFIG)
    os.makedirs(OUT_DIR, exist_ok=True)

    csv_path = os.path.join(OUT_DIR, f"updates_{args.scenario}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for variant in args.variants:
            for rep in range(args.repeats):
                print(f"[{variant} seed={rep}]", flush=True)
                row = run_variant(args.scenario, variant, rep, track,
                                  data_path, n_data, params)
                writer.writerow(row)
                fh.flush()
                print("    mean|lat|={:.5f} max|lat|={:.5f} mean|head|={:.5f} "
                      "steps={} ms/step={:.1f} M={}".format(
                          row["mean_lat"], row["max_lat"], row["mean_head"],
                          row["steps"], row["ms_per_step"], row["m_final"]),
                      flush=True)
    print(f"\nresults -> {csv_path}")


if __name__ == "__main__":
    main()