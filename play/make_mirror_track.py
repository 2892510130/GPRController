"""Mirror a track file across the x axis (y -> -y, yaw/curvature/omega -> -).

The mirrored loop has exactly the same shape, length and speed profile as the
original but turns the other way, which makes it a clean covariate shift for the
inducing-point mechanism: the dynamics (and the fault) stay identical while the
curvature regime flips sign.
"""

from __future__ import annotations

import argparse

import numpy as np
import yaml

SIGN_FLIP = ("y", "yaw", "curvature", "omega", "acc_omega")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default="config/track/full_track_opt.yaml")
    ap.add_argument("--dst", default="config/track/full_track_mirror.yaml")
    args = ap.parse_args()

    with open(args.src) as fh:
        data = yaml.safe_load(fh)

    out = {}
    for key, value in data.items():
        arr = np.asarray(value, dtype=float)
        out[key] = (-arr if key in SIGN_FLIP else arr).tolist()

    with open(args.dst, "w") as fh:
        yaml.safe_dump(out, fh, default_flow_style=None, sort_keys=False)

    x, y = np.asarray(out["x"]), np.asarray(out["y"])
    k = np.asarray(out["curvature"])
    print(f"{args.src} -> {args.dst}")
    print(f"  N={len(x)}  start=({x[0]:+.2f},{y[0]:+.2f})  end=({x[-1]:+.2f},{y[-1]:+.2f})")
    print(f"  curvature range: [{k.min():+.3f}, {k.max():+.3f}]  (source "
          f"[{-k.max():+.3f}, {-k.min():+.3f}])")


if __name__ == "__main__":
    main()