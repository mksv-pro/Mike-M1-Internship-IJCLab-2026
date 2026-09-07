"""How the two emulators scale against the solver, in channels and in parameters.

The four-system suite varies size, coupling, parameter count and physics at
once, so it cannot separate what drives the gain. This script moves ONE axis at
a time on a single physical family, the alpha + even-even rotor of
`preset_rotational_band`.

  axis A, channels     Nc = 2 ... 12 at fixed p = 6 and fixed mesh N.
                       The solver is O((Nc N)^3); the reduced solve is O(nb^3).
                       The question is whether nb follows Nc or saturates.

  axis B, parameters   p = 2 ... 6 at fixed Nc and fixed mesh, by freezing the
                       remaining parameters at their central value. Same matrix,
                       same physics, same cost per solve: only the dimension of
                       the box the manifold is swept over changes.

Everything measured, never asserted: the error is reported twice, as the
Frobenius norm on U and as the relative error on the elastic cross section --
neither has the near-zero denominator that makes `err_report` unreadable on a
rotor.

    python scaling_study.py            # runs both axes, writes scaling_study.json
"""

# Run-from-anywhere: put this dir (siblings) then the package root (core/, pipeline/) on the path.
import sys as _sys
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve().parent
_sys.path[:0] = [str(_HERE), str(_HERE.parent)]
import json
import time

import numpy as np

from core.calibration import EPS_POT, EPS_SOL, _k_from_spectral, _nb_from_energy
from core.dbmm import DBMMSolver
from core.lrom_dbmm import LROMEmulator
from core.potentials import preset_rotational_band
from core.rbm import RBMEmulator, latin_hypercube
from pipeline import metrics
from pipeline.observables import cross_section

N_MESH = 60
R_MAX = 15.0
E_DEMO = 12.0
N_VALID = 25
MIN_REPEATS = 7
SEED_TRAIN, SEED_VALID = 11, 23


def timed(fn):
    """Minimum of MIN_REPEATS runs: the floor, not the mean, as in compute.py."""
    best = np.inf
    for _ in range(MIN_REPEATS):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def solver_for(channels, theta, model):
    V_diag, V_coup = model(theta)
    return DBMMSolver(N_MESH, R_MAX, channels, V_diag, V_coup)


def calibrate(model, theta_c, bounds, solver, coupled, n_probe=80):
    """The report's own rule: nb from the solution spectrum, K from the potential."""
    from core.calibration import _potential_svd, _solution_svd
    sv_d, sv_c = _potential_svd(model, theta_c, bounds, solver, n_probe, 0, E_DEMO)
    K_diag = max(_k_from_spectral(sv_d, EPS_POT), 2)
    K_coup = max(_k_from_spectral(sv_c, EPS_POT), 2) if coupled and len(sv_c) else 0
    th = latin_hypercube(n_probe, bounds, seed=1)
    nb = _nb_from_energy(_solution_svd(model, th, solver, E_DEMO), EPS_SOL)
    n_train = max(int(np.ceil(5 * nb / 10) * 10), 50)
    return int(nb), int(K_diag), int(K_coup), int(n_train)


def one_point(Nc, free, label):
    """Measure solver, RBM and LROM on one (Nc, p) configuration."""
    channels, model_full, theta_c, bounds_full = preset_rotational_band(Nc=Nc)
    p_all = theta_c.size
    free = list(free)

    # Freezing a parameter = collapsing its box to the central value. The model
    # signature is untouched, so the physics and the matrix are identical; only
    # the dimension of the swept box changes.
    bounds = bounds_full.copy()
    for j in range(p_all):
        if j not in free:
            bounds[j] = (theta_c[j], theta_c[j])

    base = solver_for(channels, theta_c, model_full)
    nb, Kd, Kc, n_train = calibrate(model_full, theta_c, bounds, base,
                                    coupled=Nc > 1)

    th_train = latin_hypercube(n_train, bounds, seed=SEED_TRAIN)
    th_valid = latin_hypercube(N_VALID, bounds, seed=SEED_VALID)

    S_ref = np.array([solver_for(channels, t, model_full).solve(E_DEMO)
                      for t in th_valid])
    sig_ref = np.array([cross_section(base, E_DEMO, S, "elastic") for S in S_ref])

    t_solver = float(np.median([
        timed(lambda t=t: (base.set_potential(*model_full(t)), base.solve(E_DEMO)))
        for t in th_valid[:8]]))

    out = dict(Nc=Nc, p=len(free), label=label, dim=Nc * N_MESH, nb=nb,
               K=Kd + Kc, n_train=n_train, t_solver=t_solver)

    t0 = time.perf_counter()
    rbm = RBMEmulator(base, E_DEMO)
    rbm.fit(model_full, th_train, eps_tol=1e-14, nb_max=nb)
    out["t_off_rbm"] = time.perf_counter() - t0
    S = np.array([rbm.predict(t) for t in th_valid])
    out["err_rbm"] = float(np.nanmedian(metrics.err_global(S, S_ref)))
    sig = np.array([cross_section(base, E_DEMO, s_, "elastic") for s_ in S])
    out["errsig_rbm"] = float(np.nanmedian(metrics.err_observable(sig, sig_ref)))
    out["t_rbm"] = float(np.median([timed(lambda t=t: rbm.predict(t))
                                    for t in th_valid[:8]]))

    t0 = time.perf_counter()
    lrom = LROMEmulator(base, E_DEMO)
    lrom.fit(model_full, th_train, theta_c, K_diag=Kd, K_coup=Kc,
             eps_tol=1e-14, nb_max=nb)
    out["t_off_lrom"] = time.perf_counter() - t0
    S = np.array([lrom.predict(t) for t in th_valid])
    out["err_lrom"] = float(np.nanmedian(metrics.err_global(S, S_ref)))
    sig = np.array([cross_section(base, E_DEMO, s_, "elastic") for s_ in S])
    out["errsig_lrom"] = float(np.nanmedian(metrics.err_observable(sig, sig_ref)))
    out["t_lrom"] = float(np.median([timed(lambda t=t: lrom.predict(t))
                                     for t in th_valid[:8]]))

    out["gain_rbm"] = t_solver / out["t_rbm"]
    out["gain_lrom"] = t_solver / out["t_lrom"]
    print(f"  Nc={Nc:3d} p={len(free)}  dim={out['dim']:5d}  nb={nb:4d}  K={Kd+Kc:3d}  "
          f"solver={t_solver*1e3:8.2f} ms  RBM x{out['gain_rbm']:6.1f} "
          f"({out['err_rbm']:.1e})  LROM x{out['gain_lrom']:6.1f} ({out['err_lrom']:.1e})",
          flush=True)
    return out


if __name__ == "__main__":
    rows = []
    print("=== axis A: channels, p = 6 fixed", flush=True)
    for Nc in (2, 3, 4, 6, 8, 10, 12):
        rows.append(one_point(Nc, range(6), "channels"))

    print("\n=== axis B: parameters, Nc = 6 fixed", flush=True)
    # Freed in this order: depth, then absorption, then geometry, then coupling.
    order = [0, 1, 3, 4, 5, 2]
    for p in (2, 3, 4, 5, 6):
        rows.append(one_point(6, order[:p], "parameters"))

    with open(_HERE / "scaling_study.json", "w") as f:
        json.dump(rows, f, indent=1)
    print("\nwritten scaling_study.json")
