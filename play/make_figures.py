"""Render the closed-loop figures for the thesis from the dumped trajectories."""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TRAJ_DIR = "log/sweep/traj"

STYLE = {
    "ref": dict(color="#1f77b4", lw=1.2),
    "sim": dict(color="#d62728", lw=1.4),
    "lat": dict(color="#d62728", lw=1.2),
    "head": dict(color="#ff7f0e", lw=1.2),
}

plt.rcParams.update({
    "font.family": "Microsoft YaHei",
    "axes.unicode_minus": False,
    "font.size": 9,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "legend.frameon": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
})


def load(scenario: str, run: str) -> dict:
    return dict(np.load(f"{TRAJ_DIR}/{scenario}_{run}.npz"))


def figure_traj(scenario: str, runs, labels, out: str) -> None:
    fig, axes = plt.subplots(1, len(runs), figsize=(9.0, 3.0))
    for ax, run, label in zip(axes, runs, labels):
        data = load(scenario, run)
        ref = data["ref"]
        sim = data["sim_path"]
        ax.plot(ref[:, 0], ref[:, 1], **STYLE["ref"], label="参考轨迹")
        ax.plot(sim[:, 0], sim[:, 1], **STYLE["sim"], label="车辆轨迹")
        ax.scatter([sim[0, 0]], [sim[0, 1]], s=18, c="k", marker="o", zorder=5)
        ax.set_title(label, fontsize=9)
        ax.set_aspect("equal")
        ax.set_xlabel("$x$ (m)")
        ax.set_ylabel("$y$ (m)")
    axes[0].legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def figure_err(scenario: str, runs, labels, out: str) -> None:
    fig, axes = plt.subplots(1, len(runs), figsize=(9.0, 3.0))
    for ax, run, label in zip(axes, runs, labels):
        state = load(scenario, run)["state"]
        t = np.arange(state.shape[0]) * 0.05
        ax.plot(t, state[:, 0], **STYLE["lat"], label="横向误差")
        ax.plot(t, state[:, 1], **STYLE["head"], label="航向误差")
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("$t$ (s)")
        ax.set_ylabel("误差")
        ax.set_ylim(-0.25, 0.25)
    axes[0].legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", required=True, help="where the PDFs are written")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    runs = ["nominal", "gp0", "gp1"]
    labels = ["名义模型", "零阶 GP-MPC", "一阶 GP-MPC"]
    for scenario, tag in (("delay", "delay"), ("bad_wheel", "badwheel")):
        figure_traj(scenario, runs, labels, os.path.join(args.outdir, f"sim_traj_{tag}.pdf"))
        figure_err(scenario, runs, labels, os.path.join(args.outdir, f"sim_err_{tag}.pdf"))
        print(f"rendered {tag}")


if __name__ == "__main__":
    main()