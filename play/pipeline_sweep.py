"""Full pipeline for the closed-loop update-method comparison.

For one scenario it

1. runs the nominal controller (``gpr_method = -1``) with data recording on,
2. turns the recorded residuals into the GP training file (features + responses,
   followed by the ``2 * M`` inducing inputs, i.e. the format read by
   ``GaussianModel.read_data``),
3. fits the two FITC sparse GPs with GPy and keeps the learned hyper-parameters,
4. runs the closed-loop ablation with that per-scenario model:

   nominal / gp0_frozen / gp0_update / gp0_update_add / gp1_update_add

Usage (from the repository root)::

    python -m play.pipeline_sweep --scenario delay   --tag delay
    python -m play.pipeline_sweep --scenario bad_wheel --tag bad_wheel --restarts 3
"""

from __future__ import annotations

import argparse
import csv
import os
import time

import GPy
import numpy as np

from controller import MPCList
from play.mpc_closed_loop import SimulationRunner
from train.train_gp import _create_sparse_kernel, get_pre_processed_data
from trajectory import Trajectory
from utils import Config

BASE_CONFIG = "config/mpc/linear_mpc_frenet.yaml"
TRACK_CONFIG = "config/track/full_track_opt.yaml"
REC_DIR = "log/record"
DATA_DIR = "log/processed_data"
OUT_DIR = "log/sweep"

SCENARIOS = {
    # 100 ms delay == 2 control steps at dt = 0.05 s
    "delay": dict(delay_window=2, bad_wheel=False),
    # one side keeps 97.5% of the commanded wheel speed
    "bad_wheel": dict(delay_window=1, bad_wheel=True, wheel_left=0.975, wheel_right=1.0),
}

VARIANTS = {
    "nominal": (-1, False, False),
    "gp0_frozen": (0, False, False),
    "gp0_update": (0, True, False),
    "gp0_update_add": (0, True, True),
    "gp1_update_add": (1, True, True),
}

FIELDS = [
    "variant", "scenario", "seed", "steps", "terminated", "reason",
    "mean_lat", "max_lat", "mean_head", "max_head",
    "init_s", "sim_s", "ms_per_step", "m_final",
]


def make_config(scenario: str, window: int = 3, wheel_left: float | None = None) -> Config:
    cfg = Config.from_yaml(BASE_CONFIG)
    for key, value in SCENARIOS[scenario].items():
        setattr(cfg.sim, key, value)
    if wheel_left is not None:
        cfg.sim.wheel_left = wheel_left
    # match section 4.6.2.1: R = diag(0.1, 0.1), |a| <= 1, |delta| <= 1.5
    cfg.mpc.R = np.diag([0.1, 0.1])
    cfg.mpc.max_input = [1.0, 1.5]
    cfg.gaussian.input_window = window
    cfg.mpc.record_data = False
    cfg.mpc.print_time = False
    return cfg


def record_nominal(scenario: str, tag: str, track: Trajectory, wheel_left: float | None = None) -> str:
    """Run the nominal controller and dump (residual, v, w, a, delta) per step."""
    os.makedirs(REC_DIR, exist_ok=True)
    raw = f"{REC_DIR}/{tag}.txt"
    cfg = make_config(scenario, wheel_left=wheel_left)
    cfg.gaussian.gpr_method = -1
    cfg.mpc.record_data = True
    cfg.mpc.record_data_path = raw
    cfg.mpc.log_path = f"{OUT_DIR}/record_{tag}.txt"

    print(f"[record] nominal run, scenario={scenario} -> {raw}")
    np.random.seed(0)  # reproducible recording run
    mpc = MPCList[cfg.mpc.mpc_name](cfg, track.path, track.speed_profile)
    runner = SimulationRunner(mpc, cfg)
    runner.run_full_simulation()
    mpc.logger.close_log()

    if getattr(mpc, "gpr_file", None):
        mpc.gpr_file.close()
    n = sum(1 for _ in open(raw))
    print(f"[record] {n} samples")
    return raw


def train_from_record(raw: str, tag: str, n_inducing: int, window: int,
                      gap: int, restarts: int) -> tuple[str, int, list]:
    """Build the GP training file and fit the two sparse FITC GPs."""
    features, responses, _ = get_pre_processed_data(100000, window, raw, gap)
    n = int(features.shape[0])
    out = f"{DATA_DIR}/{tag}.txt"
    os.makedirs(DATA_DIR, exist_ok=True)

    with open(out, "w") as fh:
        for feat, resp in zip(features, responses):
            fh.write(" ".join(map(str, feat)) + " " + " ".join(map(str, resp)) + "\n")

    print(f"[train] {n} training rows -> {out}")

    params = []
    for i in range(2):
        kernel = _create_sparse_kernel(4 * (window - gap))
        model = GPy.models.SparseGPRegression(
            features, responses[:, i].reshape(-1, 1),
            kernel=kernel, normalizer=False, num_inducing=n_inducing,
        )
        model.inference_method = GPy.inference.latent_function_inference.FITC()
        model.parameters_changed()
        if restarts > 0:
            model.optimize_restarts(num_restarts=restarts, verbose=False)

        with open(out, "a") as fh:
            for inducing in model.inducing_inputs[:n_inducing]:
                fh.write(" ".join(map(str, inducing)) + "\n")

        params.append(
            [float(model.likelihood.variance[0]), float(model.kern.variance[0])]
            + [float(v) for v in np.asarray(model.kern.lengthscale).ravel()]
        )
        print(
            "[train] gp{}: noise={:.3e} var={:.4g} ls=[{:.3g} .. {:.3g}]".format(
                i, params[-1][0], params[-1][1], min(params[-1][2:]), max(params[-1][2:])
            )
        )
    return out, n, params


def run_variant(scenario: str, variant: str, seed: int, track: Trajectory,
                data_path: str, n_data: int, params: list,
                wheel_left: float | None = None) -> dict:
    cfg = make_config(scenario, wheel_left=wheel_left)
    gpr_method, update_mu_su, add_mu_su = VARIANTS[variant]
    cfg.gaussian.gpr_method = gpr_method
    cfg.gaussian.update_mu_su = update_mu_su
    cfg.gaussian.add_mu_su = add_mu_su
    cfg.gaussian.path = data_path
    cfg.gaussian.total_data_number = n_data
    cfg.gaussian.params = params
    cfg.gaussian.read_model = False
    cfg.mpc.log_path = f"{OUT_DIR}/mpc_{scenario}_{variant}_{seed}.txt"
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
        reason=str(getattr(runner, "termination_reason", "") or ""),
        mean_lat=float(err[:, 0].mean()), max_lat=float(err[:, 0].max()),
        mean_head=float(err[:, 1].mean()), max_head=float(err[:, 1].max()),
        init_s=round(init_s, 2), sim_s=round(sim_s, 2),
        ms_per_step=round(1000.0 * sim_s / steps, 2), m_final=m_final,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenario", required=True, choices=list(SCENARIOS))
    ap.add_argument("--tag", default=None, help="artifact prefix, defaults to scenario")
    ap.add_argument("--restarts", type=int, default=3, help="GPy optimize restarts")
    ap.add_argument("--window", type=int, default=3)
    ap.add_argument("--gap", type=int, default=0)
    ap.add_argument("--n-inducing", type=int, default=40)
    ap.add_argument("--wheel-left", type=float, default=None)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    args = ap.parse_args()

    tag = args.tag or args.scenario
    os.makedirs(OUT_DIR, exist_ok=True)
    track = Trajectory(TRACK_CONFIG)

    raw = record_nominal(args.scenario, tag, track, args.wheel_left)
    data_path, n_data, params = train_from_record(
        raw, tag, args.n_inducing, args.window, args.gap, args.restarts
    )

    csv_path = os.path.join(OUT_DIR, f"{tag}_ablation.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for variant in args.variants:
            for rep in range(args.repeats):
                seed = rep
                print(f"[ablate] {variant} seed={seed} ...", flush=True)
                row = run_variant(
                    args.scenario, variant, seed, track, data_path, n_data, params,
                    args.wheel_left,
                )
                writer.writerow(row)
                fh.flush()
                print(
                    "    mean|lat|={:.5f} max|lat|={:.5f} mean|head|={:.5f} "
                    "steps={} ms/step={:.1f}".format(
                        row["mean_lat"], row["max_lat"], row["mean_head"],
                        row["steps"], row["ms_per_step"],
                    ),
                    flush=True,
                )

    with open(os.path.join(OUT_DIR, f"{tag}_params.txt"), "w") as fh:
        fh.write(f"data={data_path}\nn={n_data}\n")
        for i, p in enumerate(params):
            fh.write(f"gp{i}=" + " ".join(f"{v:.10g}" for v in p) + "\n")
    print(f"\nresults -> {csv_path}")


if __name__ == "__main__":
    main()