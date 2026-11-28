"""Closed-loop ablation sweep for the GP-MPC online-update mechanism.

Runs a matrix of (controller variant x scenario x seed) headless closed-loop
simulations and appends the tracking-error metrics to ``log/sweep/<tag>.csv``.

Run from the repository root, e.g.::

    python -m play.sweep_closed_loop --tag smoke --variants nominal \
        --scenarios delay --repeats 1
    python -m play.sweep_closed_loop --tag full --repeats 3

Variants (gpr_method, update_mu_su, add_mu_su):
    nominal          (-1, F, F)  nominal MPC, no GP at all
    gp0_frozen       ( 0, F, F)  zero-order GP, posterior frozen after init
    gp0_update       ( 0, T, F)  zero-order GP, variational update only
    gp0_update_add   ( 0, T, T)  zero-order GP, update + inducing expansion
    gp1_update_add   ( 1, T, T)  first-order GP, update + inducing expansion
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

VARIANTS = {
    "nominal": (-1, False, False),
    "gp0_frozen": (0, False, False),
    "gp0_update": (0, True, False),
    "gp0_update_add": (0, True, True),
    "gp1_update_add": (1, True, True),
}

SCENARIOS = {
    # 100 ms delay == 2 control steps at dt = 0.05 s
    "delay": dict(delay_window=2, bad_wheel=False),
    # one side keeps 97.5% of the commanded wheel speed
    "bad_wheel": dict(delay_window=1, bad_wheel=True, wheel_left=0.975, wheel_right=1.0),
}

FIELDS = [
    "variant", "scenario", "seed", "steps", "terminated", "reason",
    "mean_lat", "max_lat", "final_lat", "mean_head", "max_head", "final_head",
    "init_s", "sim_s", "ms_per_step",
]


def build_config(variant: str, scenario: str, seed: int, paper_params: bool = False, read_model: bool = False) -> Config:
    cfg = Config.from_yaml(BASE_CONFIG)
    gpr_method, update_mu_su, add_mu_su = VARIANTS[variant]
    cfg.gaussian.gpr_method = gpr_method
    cfg.gaussian.update_mu_su = update_mu_su
    cfg.gaussian.add_mu_su = add_mu_su
    cfg.gaussian.read_model = read_model
    for key, value in SCENARIOS[scenario].items():
        setattr(cfg.sim, key, value)
    if paper_params:
        # match section 4.6.2.1 of the thesis: R = diag(0.1, 0.1), |a| <= 1, |d| <= 1.5
        cfg.mpc.R = np.diag([0.1, 0.1])
        cfg.mpc.max_input = [1.0, 1.5]
    # per-run log files so that parallel/repeated runs do not clobber each other
    cfg.mpc.record_data = False
    cfg.mpc.print_time = False
    cfg.mpc.log_path = f"{OUT_DIR}/mpc_log_{variant}_{scenario}_{seed}.txt"
    return cfg


def run_once(variant: str, scenario: str, seed: int, track: Trajectory, paper_params: bool = False, read_model: bool = False) -> dict:
    cfg = build_config(variant, scenario, seed, paper_params, read_model)
    np.random.seed(seed)

    t0 = time.perf_counter()
    mpc = MPCList[cfg.mpc.mpc_name](cfg, track.path, track.speed_profile)
    init_s = time.perf_counter() - t0

    runner = SimulationRunner(mpc, cfg)
    t1 = time.perf_counter()
    runner.run_full_simulation()
    sim_s = time.perf_counter() - t1

    err = np.abs(np.asarray(runner.state, dtype=float))
    steps = int(err.shape[0])
    row = dict(
        variant=variant,
        scenario=scenario,
        seed=seed,
        steps=steps,
        terminated=bool(getattr(runner, "simulation_finished", False)),
        reason=str(getattr(runner, "termination_reason", "") or ""),
        mean_lat=float(err[:, 0].mean()) if steps else float("nan"),
        max_lat=float(err[:, 0].max()) if steps else float("nan"),
        final_lat=float(err[-1, 0]) if steps else float("nan"),
        mean_head=float(err[:, 1].mean()) if steps else float("nan"),
        max_head=float(err[:, 1].max()) if steps else float("nan"),
        final_head=float(err[-1, 1]) if steps else float("nan"),
        init_s=round(init_s, 2),
        sim_s=round(sim_s, 2),
        ms_per_step=round(1000.0 * sim_s / steps, 2) if steps else float("nan"),
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="run", help="csv file name under log/sweep")
    parser.add_argument("--variants", nargs="*", default=list(VARIANTS))
    parser.add_argument("--scenarios", nargs="*", default=list(SCENARIOS))
    parser.add_argument("--repeats", type=int, default=1, help="number of seeds per cell")
    parser.add_argument("--seed0", type=int, default=0)
    parser.add_argument("--read-model", action="store_true",
                        help="load the trained GP from log/save_*.npy instead of fitting on the fly")
    parser.add_argument("--paper-params", action="store_true",
                        help="use R = diag(0.1,0.1) and |a| <= 1, |d| <= 1.5 as in section 4.6.2.1")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, f"{args.tag}.csv")
    write_header = not os.path.exists(csv_path)

    track = Trajectory(TRACK_CONFIG)
    total = len(args.variants) * len(args.scenarios) * args.repeats
    done = 0

    with open(csv_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
        for scenario in args.scenarios:
            for variant in args.variants:
                for rep in range(args.repeats):
                    seed = args.seed0 + rep
                    done += 1
                    print(f"[{done}/{total}] {variant} / {scenario} / seed={seed} ...", flush=True)
                    t0 = time.perf_counter()
                    try:
                        row = run_once(variant, scenario, seed, track, args.paper_params, args.read_model)
                    except Exception as exc:  # keep the sweep alive
                        print(f"    FAILED: {type(exc).__name__}: {exc}", flush=True)
                        continue
                    writer.writerow(row)
                    fh.flush()
                    print(
                        "    mean|lat|={:.5f} max|lat|={:.5f} mean|head|={:.5f} "
                        "steps={} ms/step={:.1f} ({} s)".format(
                            row["mean_lat"], row["max_lat"], row["mean_head"],
                            row["steps"], row["ms_per_step"],
                            round(time.perf_counter() - t0, 1),
                        ),
                        flush=True,
                    )

    print(f"\nresults -> {csv_path}")


if __name__ == "__main__":
    main()