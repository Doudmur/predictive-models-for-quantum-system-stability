"""Figure for the slides: forecast skill against persistence as a function of horizon."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from forecast import TRAIN_DAYS, f_ar1, f_moving_avg, mae
from pathlib import Path

from simulate import simulate

ROOT = Path(__file__).resolve().parent.parent

BLUE, ORANGE, GRAY, INK, MUTED, BG = "#1F6CAB", "#D9692B", "#8A939C", "#13202E", "#5F6A75", "#F6F5F1"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 15, "axes.edgecolor": "#C9C4B8",
    "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": BG,
    "axes.facecolor": BG, "axes.grid": True, "grid.color": "#E2DED4", "grid.linewidth": 1,
})

t1_df, _ = simulate(2026)
logv = np.log(t1_df.to_numpy())
n_t, n_q = logv.shape
start = TRAIN_DAYS * 8

# ---------------- skill vs horizon (persistence as reference) ----------------
H = range(1, 9)
skill = {"AR(1)": [], "Moving average, 24 h": []}
for h in H:
    y, fp, far, fma = [], [], [], []
    for origin in range(start - 1, n_t - 8):            # same origins for every h
        y.append(logv[origin + h])
        fp.append(logv[origin])
        far.append([f_ar1(logv[:origin + 1, q], h) for q in range(n_q)])
        fma.append([f_moving_avg(logv[:origin + 1, q], h) for q in range(n_q)])
    y, fp, far, fma = (np.exp(np.array(a)) for a in (y, fp, far, fma))
    skill["AR(1)"].append(1 - mae(y, far) / mae(y, fp))
    skill["Moving average, 24 h"].append(1 - mae(y, fma) / mae(y, fp))

fig, ax = plt.subplots(figsize=(9.6, 5.6), dpi=200)
hours = [3 * h for h in H]
ax.axhline(0, color=GRAY, lw=2, ls=(0, (4, 3)))
ax.text(24, -2.5, "persistence = last calibration", color=MUTED, fontsize=13, va="top", ha="right")
for name, color in (("AR(1)", BLUE), ("Moving average, 24 h", ORANGE)):
    ax.plot(hours, 100 * np.array(skill[name]), color=color, lw=2.5, marker="o", ms=8,
            markeredgecolor=BG, markeredgewidth=2)
    ax.text(24.6, 100 * skill[name][-1], name, color=INK, fontsize=14, va="center")
ax.set_xlabel("forecast horizon, hours")
ax.set_ylabel("MAE improvement over persistence, %")
ax.set_xticks(hours)
ax.set_xlim(2, 34)
fig.tight_layout()
(ROOT / "figures").mkdir(exist_ok=True)
fig.savefig(ROOT / "figures" / "fig_skill_vs_horizon.png", facecolor=BG)
json.dump({"hours": hours, **{k: v for k, v in skill.items()}}, open(ROOT / "results" / "skill_vs_horizon.json", "w"), indent=1)
print("skill vs horizon", {k: [round(x, 3) for x in v] for k, v in skill.items()})
