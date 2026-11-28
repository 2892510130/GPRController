"""Re-render the round curves from the CSV with readable labels."""

import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = r"C:\Users\pc\Desktop\Personal\Code\GPRController\log\sweep\ul_rounds.csv"
OUT = r"C:\Users\pc\Desktop\Personal\Graduation\zjuthesis\figure\gpmpc"

LABELS = {
    "trained-non-inc": "trained params, no expansion",
    "trained-inc": "trained params + expansion",
    "default-non-inc": "default params, no expansion",
    "default-inc": "default params + expansion",
}
STYLE = {
    "trained-non-inc": dict(color="#7f7f7f", ls="--", marker="s"),
    "trained-inc": dict(color="#d62728", ls="-", marker="^"),
    "default-non-inc": dict(color="#1f77b4", ls="--", marker="o"),
    "default-inc": dict(color="#2ca02c", ls="-", marker="D"),
}

rows = list(csv.DictReader(open(CSV)))
plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.8,
                     "legend.frameon": False, "axes.grid": True,
                     "grid.alpha": 0.3, "grid.linewidth": 0.5})

for key, idx, ylabel, scale in (("lat", 1, "mean $|d|$ (mm)", 1000.0),
                                ("head", 2, r"mean $|\theta_e|$ (rad)", 1.0)):
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for cls, style in STYLE.items():
        sel = sorted((r for r in rows if r["cls"] == cls),
                     key=lambda r: int(r["round"]))
        x = [int(r["round"]) for r in sel]
        col = "mean_lat" if idx == 1 else "mean_head"
        y = [float(r[col]) * scale for r in sel]
        ax.plot(x, y, lw=1.3, ms=3.2, **style, label=LABELS[cls])
    ax.set_xlabel("round")
    ax.set_ylabel(ylabel)
    ax.set_xlim(-0.3, 9.3)
    ax.set_xticks(range(0, 10, 1) if idx == 1 else range(0, 10, 2))
    ax.legend(fontsize=6.2, loc="upper right")
    fig.tight_layout()
    name = "lateral_error.pdf" if idx == 1 else "heading_error.pdf"
    fig.savefig(f"{OUT}/{name}", bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)