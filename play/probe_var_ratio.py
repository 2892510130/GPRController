"""Probe the posterior-variance ratio on the mirrored track.

Prints how far the predictive variance at the cross-domain operating points sits
above the training-set reference, which is what the "variance" inducing-point
criterion compares against.  Used to pick ``variance_ratio``.
"""

from __future__ import annotations

import re

import numpy as np

from model.gp_model import GaussianModel
from utils import Config

LOG = "log/sweep/cd_delay_osvgp_add_0.txt"
PARAMS = "log/sweep/cross_train_params.txt"
NUM = re.compile(r"-?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?")


def main() -> None:
    data_path = n_data = None
    params = []
    for line in open(PARAMS):
        line = line.strip()
        if line.startswith("data="):
            data_path = line.split("=", 1)[1]
        elif line.startswith("n="):
            n_data = int(line.split("=", 1)[1])
        elif line.startswith("gp") and "=" in line:
            params.append([float(v) for v in line.split("=", 1)[1].split()])

    cfg = Config.from_yaml("config/mpc/linear_mpc_frenet.yaml").gaussian
    cfg.path = data_path
    cfg.total_data_number = n_data
    cfg.params = params
    cfg.read_model = False
    gp = GaussianModel(cfg)

    zs = []
    for line in open(LOG, encoding="utf-8", errors="ignore"):
        if "GPR Input" in line:
            zs.append([float(v) for v in NUM.findall(line)])
    zs = np.asarray(zs, dtype=float)
    print(f"regressors: {zs.shape}, var_ref = {gp.gps[0].var_ref:.5f}")

    ratios = []
    for z in zs:
        v = float(np.ravel(gp.gps[0].predict_noiseless(z.reshape(1, -1), full_cov=False)[1])[0])
        ratios.append(v / max(gp.gps[0].var_ref, 1e-12))
    r = np.asarray(ratios)
    qs = [0, 5, 25, 50, 75, 95, 100]
    print("ratio percentiles: " + ", ".join(f"p{q}={np.percentile(r, q):.2f}" for q in qs))
    print(f"fraction above 2x: {(r > 2).mean():.2%}, 4x: {(r > 4).mean():.2%}, "
          f"10x: {(r > 10).mean():.2%}, 50x: {(r > 50).mean():.2%}")


if __name__ == "__main__":
    main()