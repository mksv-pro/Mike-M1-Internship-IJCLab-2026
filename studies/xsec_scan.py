"""sigma(E) for every benchmark on one dense grid, with the time each one costs.

The smallest thing the project can do: no emulator, no cache, no figure
pipeline. One solver, one grid, one plot. Run it to see what the reference
calculation costs before anything is reduced.

    python xsec_scan.py
"""

# Run-from-anywhere: put this dir (siblings) then the package root (core/, pipeline/) on the path.
import sys as _sys
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve().parent
_sys.path[:0] = [str(_HERE), str(_HERE.parent)]
import time

import numpy as np
import matplotlib.pyplot as plt

from core import dbmm, specfun
from pipeline.observables import cross_section, solver_for
from pipeline.systems import SYSTEMS

# NumbaBackend ahead of Arb for the mesh: ~3x, at the price of 7e-12 error.
specfun.SF = specfun.SpecialFunctions(chain=(
    specfun.ScipyBackend, specfun.NumbaBackend,
    specfun.FlintBackend, specfun.MpmathBackend))
dbmm.SF = specfun.SF                                    # dbmm did `from .specfun import SF`
specfun.SF.coulomb_f_mesh(0, 1.0, np.linspace(0.1, 5, 10))   # JIT warm-up, outside the timers

E = np.linspace(0, 50.0, 1000)
E[0] = 1e-1   # E = 0 gives k = 0 and a division by zero

KEYS = ["p12C", "alpha_d", "n40Ca", "alpha24Mg", "n238U", "alpha12C_band"]

for key in KEYS:
    s = SYSTEMS[key]
    theta_c = s.preset_fn()[2]
    kind = "elastic"                # reaction where absorptive, elastic where not
    t0 = time.perf_counter()
    solver = solver_for(s, theta_c, energies=E)
    sigma = [cross_section(solver, e, solver.solve(e), kind) for e in E]
    dt = time.perf_counter() - t0
    print(f"{key:14s} {dt:6.2f} s ({1e3 * dt / len(E):.1f} ms/point)  {kind}")
    plt.plot(E, sigma, label=s.label)
    
# ligne au minimum affiché
plt.axhline(np.exp(0), color="k", ls="--")
plt.xlabel(r"$E\;(\mathrm{MeV})$")
plt.ylabel(r"$\sigma\;(\mathrm{fm}^2)$")
plt.yscale("log")
plt.ylim(bottom=1e-6)   # borne inférieure
plt.legend()
plt.savefig(_HERE / "xsec_scan.png", dpi=150)
