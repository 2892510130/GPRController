"""Residual-prediction accuracy of the trained GPs (noise on / off).

For every run the MPC log gives the augmented regressors ``z_k`` and the
measured residual ``d_k``.  The trained posterior is re-evaluated on each
``z_k`` so that the prediction can be compared with the residual of the *same*
step:

    nominal predictor : d_hat = 0            -> error = mean |d_k|
    GP predictor      : d_hat = mu(z_k)      -> error = mean |d_k - mu(z_k)|
    improvement (%)   : 1 - GP error / nominal error

Both approximations (zero / first order) share the same GP posterior, so the
table reports one row per scenario and channel, and the script also refreshes
the residual-prediction figures of chapter 4.
"""

from __future__ import annotations

import argparse
import csv
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from controller import MPCList
from model.gp_model import GaussianModel
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
NUM = re.compile(r"-?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?")


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


def parse_log(path: str) -> tuple[np.ndarray, np.ndarray]:
    nominal, inputs = [], []
    for line in open(path, encoding="utf-8", errors="ignore"):
        if "Delta State Nominal" in line:
            nominal.append([float(v) for v in NUM.findall(line)])
        elif "GPR Input" in line:
            inputs.append([float(v) for v in NUM.findall(line)])
    return np.asarray(nominal, dtype=float), np.asarray(inputs, dtype=float)


def run_cell(scenario: str, variant: str, noise: bool, seed: int,
             data_path: str, n_data: int, params: list, track: Trajectory) -> str:
    cfg = Config.from_yaml(BASE_CONFIG)
    for key, value in SCENARIOS[scenario].items():
        setattr(cfg.sim, key, value)
    cfg.sim.add_noise = noise
    cfg.mpc.R = np.diag([0.1, 0.1])
    cfg.mpc.max_input = [1.0, 1.5]
    cfg.mpc.record_data = False
    cfg.mpc.print_time = False
    cfg.mpc.log_path = f"{OUT_DIR}/res_{scenario}_{variant}_{'n' if noise else 'nn'}.txt"
    gpr_method, update_mu_su, add_mu_su = {"gp0": (0, True, False),
                                           "gp1": (1, True, True)}[variant]
    cfg.gaussian.gpr_method = gpr_method
    cfg.gaussian.update_mu_su = update_mu_su
    cfg.gaussian.add_mu_su = add_mu_su
    cfg.gaussian.path = data_path
    cfg.gaussian.total_data_number = n_data
    cfg.gaussian.params = params
    cfg.gaussian.read_model = False
    np.random.seed(seed)

    mpc = MPCList[cfg.mpc.mpc_name](cfg, track.path, track.speed_profile)
    runner = SimulationRunner(mpc, cfg)
    runner.run_full_simulation()
    mpc.logger.close_log()
    return cfg.mpc.log_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenario", required=True, choices=list(SCENARIOS))
    ap.add_argument("--params-tag", required=True)
    ap.add_argument("--figure-dir", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data_path, n_data, params = read_params(args.params_tag)
    track = Trajectory(TRACK_CONFIG)

    gp_cfg = Config.from_yaml(BASE_CONFIG).gaussian
    gp_cfg.path = data_path
    gp_cfg.total_data_number = n_data
    gp_cfg.params = params
    gp_cfg.read_model = False
    gp = GaussianModel(gp_cfg)

    rows = []
    for noise in (True, False):
        log_path = run_cell(args.scenario, "gp0", noise, args.seed,
                            data_path, n_data, params, track)
        nominal, inputs = parse_log(log_path)
        k = min(len(nominal), len(inputs))
        nominal, inputs = nominal[:k], inputs[:k]

        mu = np.zeros((k, 2))
        for i, z in enumerate(inputs):
            for c in range(2):
                mu[i, c] = float(np.ravel(
                    gp.gps[c].predict_noiseless(z.reshape(1, -1), full_cov=False)[0]
                )[0])

        base = np.mean(np.abs(nominal), axis=0)
        err = np.mean(np.abs(nominal - mu), axis=0)
        imp = (base - err) / base * 100.0
        for c, name in enumerate(("$\\Delta v$", "$\\Delta\\omega$")):
            rows.append(dict(
                scenario=args.scenario, noise="有噪声" if noise else "无噪声",
                channel=name, baseline=float(base[c]), gp=float(err[c]),
                improvement=float(imp[c]), steps=k,
            ))
        print(f"{args.scenario}/{'noise' if noise else 'clean'}: "
              f"dv {imp[0]:+.1f}%  dw {imp[1]:+.1f}%  (baseline {base[0]:.4f} {base[1]:.4f})")

        if noise and args.figure_dir:
            os.makedirs(args.figure_dir, exist_ok=True)
            suffix = "delay" if args.scenario == "delay" else "badwheel"
            out = os.path.join(args.figure_dir, f"sim_residual_{suffix}.pdf")
            fig, axes = plt.subplots(2, 1, figsize=(6.2, 4.0), sharex=True)
            t = np.arange(k) * 0.05
            for ax, c, color in ((axes[0], 0, "#d62728"), (axes[1], 1, "#1f77b4")):
                ax.plot(t, nominal[:, c], color="0.55", lw=1.0, label="measured residual")
                ax.plot(t, mu[:, c], color=color, lw=1.2, label="GP prediction")
                ax.grid(alpha=0.3, lw=0.5)
                ax.legend(fontsize=7)
            axes[0].set_ylabel(r"$\Delta v$")
            axes[1].set_ylabel(r"$\Delta \omega$")
            axes[1].set_xlabel("$t$ (s)")
            fig.tight_layout()
            fig.savefig(out, bbox_inches="tight")
            plt.close(fig)
            print(f"figure -> {out}")

    csv_path = os.path.join(OUT_DIR, f"residual_{args.scenario}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"csv -> {csv_path}")


if __name__ == "__main__":
    main()