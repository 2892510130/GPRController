"""Probe: which future step does the trained GP actually predict?

The GP is trained on (history at i) -> (residual at i + input_window).  This
script replays a run, pushes the logged GPR inputs through the trained model
and reports the prediction error for several alignment offsets, so that the
metric used in the thesis can be defined correctly.
"""

from __future__ import annotations

import re

import numpy as np

from model.gp_model import GaussianModel
from utils import Config

LOG = "log/sweep/res_delay_gp0_n.txt"
PARAMS = "log/sweep/delay3_params.txt"
NUM = re.compile(r"-?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?")


def read_params(path):
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
    nominal, inputs = [], []
    for line in open(LOG, encoding="utf-8", errors="ignore"):
        if "Delta State Nominal" in line:
            nominal.append([float(v) for v in NUM.findall(line)])
        elif "GPR Input" in line:
            inputs.append([float(v) for v in NUM.findall(line)])
    nominal = np.asarray(nominal, dtype=float)
    inputs = np.asarray(inputs, dtype=float)
    print("steps:", len(nominal), "inputs:", len(inputs))

    data_path, n_data, params = read_params(PARAMS)
    cfg = Config.from_yaml("config/mpc/linear_mpc_frenet.yaml")
    cfg.gaussian.path = data_path
    cfg.gaussian.total_data_number = n_data
    cfg.gaussian.params = params
    cfg.gaussian.read_model = False
    gp = GaussianModel(cfg.gaussian)

    mu = np.zeros((len(inputs), 2))
    for i, x in enumerate(inputs):
        mu[i, 0] = float(np.ravel(gp.gps[0].predict_noiseless(x.reshape(1, -1), full_cov=False)[0])[0])
        mu[i, 1] = float(np.ravel(gp.gps[1].predict_noiseless(x.reshape(1, -1), full_cov=False)[0])[0])

    k = min(len(nominal), len(mu))
    baseline = np.mean(np.abs(nominal[:k]), axis=0)
    print(f"baseline mean|residual| = {baseline[0]:.5f} {baseline[1]:.5f}")
    for shift in (0, 1, 2, 3, 4):
        if shift >= k:
            break
        target = nominal[shift:k]
        pred = mu[: k - shift]
        err = np.mean(np.abs(target - pred), axis=0)
        imp = (baseline - err) / baseline * 100
        print(
            f"shift={shift}: error={err[0]:.5f} {err[1]:.5f} "
            f"improvement={imp[0]:+6.1f}% {imp[1]:+6.1f}%"
        )


if __name__ == "__main__":
    main()