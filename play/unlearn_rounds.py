"""Round-based unlearning experiment for section 4.6.2.3.

Protocol
--------
1. record the residual data with the *nominal* controller under an abnormal
   condition (single-side wheel fault);
2. fit the two sparse GPs on that recording -> trained hyper-parameters;
3. deploy on the *normal* condition for ``--rounds`` episodes:
   - round 0: ``fit`` (read_model = False), **no online update**;
   - round r >= 1: ``read`` the model saved at the end of round r-1 and run the
     variational update (plus inducing-point expansion for the "incremental"
     variants);
4. four classes = {trained, default} hyper-parameters x {non-incremental,
   incremental}.

Usage::

    python -m play.unlearn_rounds --rounds 10 --figure-dir <thesis>/figure/gpmpc
"""

from __future__ import annotations

import argparse
import csv
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from controller import MPCList
from play.mpc_closed_loop import SimulationRunner
from play.pipeline_sweep import record_nominal, train_from_record
from trajectory import Trajectory
from utils import Config

BASE_CONFIG = "config/mpc/linear_mpc_frenet.yaml"
TRACK = "config/track/full_track_opt.yaml"
OUT_DIR = "log/sweep"

# abnormal condition used for recording, normal condition used for deployment
TRAIN_SCENARIO = dict(delay_window=1, bad_wheel=True, wheel_left=0.925, wheel_right=1.0)
TEST_SCENARIO = dict(delay_window=1, bad_wheel=False, wheel_left=0.975, wheel_right=1.0)

CLASSES = {
    "trained-non-inc": dict(kind="trained", incremental=False),
    "trained-inc": dict(kind="trained", incremental=True),
    "default-non-inc": dict(kind="default", incremental=False),
    "default-inc": dict(kind="default", incremental=True),
}

FIELDS = ["cls", "round", "steps", "mean_lat", "max_lat", "mean_head",
          "max_head", "m_final", "ms_per_step"]


def default_params() -> list:
    """Hyper-parameters used by the 'default' class: unit signal variance and
    unit ARD length-scales, tiny likelihood noise (same convention as the yaml)."""
    return [[1e-8, 1.0] + [1.0] * 12 for _ in range(2)]


def run_round(cls, rnd, track, data_path, n_data, params, prefix):
    cfg = Config.from_yaml(BASE_CONFIG)
    for key, value in TEST_SCENARIO.items():
        setattr(cfg.sim, key, value)
    cfg.mpc.R = np.diag([0.1, 0.1])
    cfg.mpc.max_input = [1.0, 1.5]
    cfg.mpc.record_data = False
    cfg.mpc.print_time = False
    cfg.mpc.log_path = f"{OUT_DIR}/ul_{cls}_{rnd}.txt"

    incremental = CLASSES[cls]["incremental"]
    cfg.gaussian.gpr_method = 0
    cfg.gaussian.update_mu_su = rnd > 0          # round 0: no online update
    cfg.gaussian.add_mu_su = incremental and rnd > 0
    cfg.gaussian.update_method = "vfe"
    cfg.gaussian.lambda_ = 1.0
    cfg.gaussian.path = data_path
    cfg.gaussian.total_data_number = n_data
    cfg.gaussian.params = params
    cfg.gaussian.read_model = rnd > 0            # round 0: fit, later: read
    cfg.gaussian.save_path = prefix
    cfg.gaussian.load_path = prefix
    np.random.seed(0)

    t0 = time.perf_counter()
    mpc = MPCList[cfg.mpc.mpc_name](cfg, track.path, track.speed_profile)
    runner = SimulationRunner(mpc, cfg)
    runner.run_full_simulation()
    sim_s = time.perf_counter() - t0
    mpc.save_model()
    mpc.logger.close_log()

    err = np.abs(np.asarray(runner.state, dtype=float))
    steps = int(err.shape[0])
    try:
        m_final = "/".join(str(gp.M) for gp in mpc.model.gaussian_model.gps)
    except Exception:
        m_final = "-"
    return dict(
        cls=cls, round=rnd, steps=steps,
        mean_lat=float(err[:, 0].mean()), max_lat=float(err[:, 0].max()),
        mean_head=float(err[:, 1].mean()), max_head=float(err[:, 1].max()),
        m_final=m_final, ms_per_step=round(1000.0 * sim_s / max(steps, 1), 2),
    )


def plot(rows, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.2))
    for cls in CLASSES:
        sel = [r for r in rows if r["cls"] == cls]
        rounds = [r["round"] for r in sel]
        axes[0].plot(rounds, [r["mean_lat"] * 1000 for r in sel], marker="o", ms=3, label=cls)
        axes[1].plot(rounds, [r["mean_head"] for r in sel], marker="o", ms=3, label=cls)
    axes[0].set_xlabel("round")
    axes[0].set_ylabel("mean $|d|$ (mm)")
    axes[1].set_xlabel("round")
    axes[1].set_ylabel(r"mean $|\theta_e|$ (rad)")
    for ax in axes:
        ax.grid(alpha=0.3, lw=0.5)
        ax.legend(fontsize=7)
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    for name in ("lateral_error.pdf", "heading_error.pdf"):
        pass
    fig.savefig(os.path.join(out_dir, "online_learning_rounds.pdf"), bbox_inches="tight")

    # the thesis places two separate panels side by side
    for idx, name in ((0, "lateral_error.pdf"), (1, "heading_error.pdf")):
        single, ax = plt.subplots(figsize=(4.6, 3.0))
        for cls in CLASSES:
            sel = [r for r in rows if r["cls"] == cls]
            ys = ([r["mean_lat"] * 1000 for r in sel] if idx == 0
                  else [r["mean_head"] for r in sel])
            ax.plot([r["round"] for r in sel], ys, marker="o", ms=3, label=cls)
        ax.set_xlabel("round")
        ax.set_ylabel("mean $|d|$ (mm)" if idx == 0 else r"mean $|\theta_e|$ (rad)")
        ax.grid(alpha=0.3, lw=0.5)
        ax.legend(fontsize=7)
        single.tight_layout()
        single.savefig(os.path.join(out_dir, name), bbox_inches="tight")
        plt.close(single)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--restarts", type=int, default=2)
    ap.add_argument("--tag", default="ul")
    ap.add_argument("--figure-dir", default=None)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    track = Trajectory(TRACK)

    raw = record_nominal("bad_wheel", f"{args.tag}_train", track, wheel_left=0.925)
    data_path, n_data, trained = train_from_record(
        raw, f"{args.tag}_train", 40, 3, 0, args.restarts
    )
    param_sets = {"trained": trained, "default": default_params()}

    rows = []
    csv_path = os.path.join(OUT_DIR, f"{args.tag}_rounds.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for cls, spec in CLASSES.items():
            params = param_sets[spec["kind"]]
            prefix = f"log/save_{args.tag}_{cls.replace('-', '_')}"
            for rnd in range(args.rounds):
                print(f"[{cls} round={rnd}]", flush=True)
                row = run_round(cls, rnd, track, data_path, n_data, params, prefix)
                rows.append(row)
                writer.writerow(row)
                fh.flush()
                print("    mean|lat|={:.5f} mean|head|={:.5f} M={} {:.1f} ms/step".format(
                    row["mean_lat"], row["mean_head"], row["m_final"],
                    row["ms_per_step"]), flush=True)

    print(f"\nresults -> {csv_path}")
    if args.figure_dir:
        plot(rows, args.figure_dir)
        print(f"figures -> {args.figure_dir}/lateral_error.pdf, heading_error.pdf")


if __name__ == "__main__":
    main()