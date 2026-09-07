"""Draw the two scaling axes measured by scaling_study.py."""

# Run-from-anywhere: put this dir (siblings) then the package root (core/, pipeline/) on the path.
import sys as _sys
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve().parent
_sys.path[:0] = [str(_HERE), str(_HERE.parent)]
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pipeline import style

rows = json.load(open(_HERE / "scaling_study.json"))
A = [r for r in rows if r["label"] == "channels"]
B = [r for r in rows if r["label"] == "parameters"]


def slope(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = (x > 0) & (y > 0) & np.isfinite(y)
    return np.polyfit(np.log(x[ok]), np.log(y[ok]), 1)[0]


fig, ax = plt.subplots(1, 3, figsize=(11.2, 3.5))

# --- panel 1: where the time goes, against channel count ---
Nc = np.array([r["Nc"] for r in A], float)
for key, lab, col, ls in (("t_solver", "DBMM", "black", "-"),
                          ("t_rbm", "RBM", style.BLUE, "--"),
                          ("t_lrom", "LROM", style.WINE, "-.")):
    y = np.array([r[key] for r in A]) * 1e3
    ax[0].plot(Nc, y, marker="o", ms=3.5, color=col, ls=ls,
               label=f"{lab}  $\\propto N_c^{{{slope(Nc, y):.2f}}}$")
ax[0].set(xscale="log", yscale="log", xlabel="channels $N_c$",
          ylabel="online time (ms)", title="cost against channel count")
ax[0].legend(fontsize=6.5, frameon=False)

# --- panel 2: the rank is what decides ---
nb = np.array([r["nb"] for r in A], float)
ax[1].plot(Nc, nb, marker="o", ms=3.5, color=style.WINE,
           label=f"$n_b \\propto N_c^{{{slope(Nc, nb):.2f}}}$")
ax[1].plot(Nc, np.array([r["dim"] for r in A]), ls=":", color="0.5",
           label=r"ambient $N_cN$")
g = np.array([r["gain_rbm"] for r in A])
ax[1].plot(Nc, g, marker="s", ms=3.5, color=style.BLUE,
           label=f"RBM gain $\\propto N_c^{{{slope(Nc, g):.2f}}}$")
ax[1].set(xscale="log", yscale="log", xlabel="channels $N_c$",
          title="rank grows with the channels")
ax[1].legend(fontsize=6.5, frameon=False)

# --- panel 3: the parameter axis, at fixed matrix ---
p = np.array([r["p"] for r in B], float)
ax[2].plot(p, [r["nb"] for r in B], marker="o", ms=3.5, color=style.WINE,
           label=r"$n_b$")
ax[2].plot(p, [r["gain_rbm"] for r in B], marker="s", ms=3.5, color=style.BLUE,
           label="RBM gain")
ax2 = ax[2].twinx()
ax2.plot(p, [r["errsig_rbm"] for r in B], marker="^", ms=3.5, ls="--",
         color="0.35", label=r"error on $\sigma$")
ax2.set_yscale("log")
ax2.set_ylabel(r"median error on $\sigma$", fontsize=7)
ax2.tick_params(labelsize=6)
ax[2].set(xlabel="free parameters $p$", title="parameters, at fixed $N_cN=360$")
ax[2].legend(fontsize=6.5, frameon=False, loc="center right")
ax2.legend(fontsize=6.5, frameon=False, loc="lower right")

for a in ax:
    a.tick_params(labelsize=7)
    a.title.set_fontsize(8)
    a.xaxis.label.set_size(7.5)
    a.yaxis.label.set_size(7.5)

fig.suptitle("how the gain scales: one axis at a time, one physical family",
             fontsize=10)
fig.tight_layout()
fig.savefig(_HERE.parent / "figures_out" / "f12_scaling.png", dpi=200)
print("written figures_out/f12_scaling.png")
