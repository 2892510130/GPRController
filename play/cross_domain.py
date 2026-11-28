"""Cross-domain closed-loop experiment.

Train the GP on data recorded on track A (nominal MPC, fault enabled) and deploy
the closed loop on *another* track B under the same fault.  The curvature regime
of B is the mirror image of A, so the regressors fall outside the training
support and the inducing-point mechanism has something to do.

Usage::

    python -m play.cross_domain --train-track config/track/full_track_opt.yaml \
        --test-track config/track/full_track_mirror.yaml --scenario delay
"""

from __future__ import annotations

import argparse
import csv
import os
import time

import numpy as np

from controller import MPCList
from play.mpc_closed_loop import SimulationRunner
from play.pipeline_sweep import record_nominal, train_from_record
from trajectory import Trajectory
from utils import Config

BASE_CONFIG = "config/mpc/linear_mpc_frenet.yaml"
OUT_DIR = "log/sweep"

SCENARIOS = {
    "delay": dict(delay_window=2, bad_wheel=False, wheel_left=0.975),
    "bad_wheel": dict(delay_window=1, bad_wheel=True, wheel_left=0.925),
    "normal": dict(delay_window=1, bad_wheel=False, wheel_left=0.975),
}

VARIANTS = {
    "nominal": (-1, False, False, "vfe", 1.0),
    "frozen": (0, False, False, "vfe", 1.0),
    "osvgp": (0, True, False, "vfe", 1.0),
    "osvgp_add": (0, True, True, "vfe", 1.0),
    "bijl": (0, True, False, "bijl", 1.0),
    "agp": (0, True, False, "agp", 0.99),
    "agp_window": (0, True, False, "window", 1.0),
}

FIELDS = ["variant", "scenario", "seed", "steps", "terminated", "mean_lat",
          "max_lat", "mean_head", "max_head", "m_final", "m_max",
          "ms_per_step"]


def run_on_track(scenario, variant, seed, track, data_path, n_data, params):
    cfg = Config.from_yaml(BASE_CONFIG)
    for key, value in SCENARIOS[scenario].items():
        setattr(cfg.sim, key, value)
    cfg.mpc.R = np.diag([0.1, 0.1])
    cfg.mpc.max_input = [1.0, 1.5]
    cfg.mpc.record_data = False
    cfg.mpc.print_time = False
    cfg.mpc.log_path = f"{OUT_DIR}/cd_{scenario}_{variant}_{seed}.txt"

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

    mpc = MPCList[cfg.mpc.mpc_name](cfg, track.path, track.speed_profile)
    runner = SimulationRunner(mpc, cfg)
    t0 = time.perf_counter()
    runner.run_full_simulation()
    sim_s = time.perf_counter() - t0
    mpc.logger.close_log()

    err = np.abs(np.asarray(runner.state, dtype=float))
    steps = int(err.shape[0])
    try:
        m_final = "/".join(str(gp.M) for gp in mpc.model.gaussian_model.gps)
        m_max = max(gp.M for gp in mpc.model.gaussian_model.gps)
    except Exception:
        m_final, m_max = "-", -1
    return dict(
        variant=variant, scenario=scenario, seed=seed, steps=steps,
        terminated=bool(getattr(runner, "simulation_finished", False)),
        mean_lat=float(err[:, 0].mean()), max_lat=float(err[:, 0].max()),
        mean_head=float(err[:, 1].mean()), max_head=float(err[:, 1].max()),
        m_final=m_final, m_max=m_max,
        ms_per_step=round(1000.0 * sim_s / max(steps, 1), 2),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-track", required=True)
    ap.add_argument("--test-track", required=True)
    ap.add_argument("--scenario", default="delay", choices=list(SCENARIOS))
    ap.add_argument("--tag", default="cross")
    ap.add_argument("--restarts", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    train_track = Trajectory(args.train_track)
    test_track = Trajectory(args.test_track)

    raw = record_nominal(args.scenario, f"{args.tag}_train", train_track)
    data_path, n_data, params = train_from_record(
        raw, f"{args.tag}_train", 40, 3, 0, args.restarts
    )

    csv_path = os.path.join(OUT_DIR, f"{args.tag}_{args.scenario}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for variant in args.variants:
            for rep in range(args.repeats):
                print(f"[{variant} seed={rep}]", flush=True)
                row = run_on_track(args.scenario, variant, rep, test_track,
                                   data_path, n_data, params)
                writer.writerow(row)
                fh.flush()
                print("    mean|lat|={:.5f} max|lat|={:.5f} steps={} "
                      "M={} (max {})".format(
                          row["mean_lat"], row["max_lat"], row["steps"],
                          row["m_final"], row["m_max"]), flush=True)
    print(f"\nresults -> {csv_path}")


if __name__ == "__main__":
    main()