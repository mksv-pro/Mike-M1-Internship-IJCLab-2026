"""Figures that depend on no cached block, drawn through the shared style.

One of them: f13_coulomb, the Coulomb functions. Reads no `results/`, so
`api.figures()` draws it without a run.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from core.specfun import SF

from . import style
from .draw import out_dir

style.setup()

def _save(fig, name, outdir):
    target = Path(outdir) if outdir is not None else out_dir()
    target.mkdir(parents=True, exist_ok=True)
    path = target / (name + ".png")
    fig.savefig(path)
    plt.close(fig)
    return path



# --- Coulomb functions ------------------------------------------------
def coulomb(outdir=None, eta=0.5, ells=(0, 1, 2, 3)):
    """F_l and G_l near the origin, and |H_l^+| approaching unity.

    Colours are the sweep ramp so l reads as an ordered variable.
    """
    rho = np.linspace(1e-3, 20.0, 1200)
    colours = style.sweep_colors(len(ells))

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.0, 3.4))
    for c, l in zip(colours, ells):
        F = SF.coulomb_f_mesh(l, eta, rho)
        # coulomb_hplus is scalar in rho, which is what the boundary row needs;
        # a figure over a mesh loops it once here rather than vectorising it there.
        Hp = np.array([SF.coulomb_hplus(l, eta, float(r))[0] for r in rho])
        G = np.real(Hp)                      # H+ = G + iF, so G is its real part
        axL.plot(rho, F, color=c, lw=1.3, label=rf"$\ell={l}$")
        axL.plot(rho, G, color=c, lw=1.0, ls="--")
        axR.plot(rho, np.abs(Hp), color=c, lw=1.3)

    axL.set_xlim(0, 12)
    axL.set_ylim(-2.5, 2.5)
    axL.set_xlabel(r"$\rho$")
    axL.set_ylabel(r"$F_\ell$ (solid), $G_\ell$ (dashed)")
    axL.axhline(0.0, color=style.GREY, lw=0.6, zorder=0)
    axL.legend(frameon=False, fontsize=7.5)

    axR.axhline(1.0, color=style.GREY, lw=0.8, ls=":")
    axR.set_xlim(0, 20)
    axR.set_ylim(0.8, 4.0)
    axR.set_xlabel(r"$\rho$")
    axR.set_ylabel(r"$|H^+_\ell|$")

    fig.tight_layout()
    return _save(fig, "f13_coulomb", outdir)


STATIC = {"coulomb": coulomb}
