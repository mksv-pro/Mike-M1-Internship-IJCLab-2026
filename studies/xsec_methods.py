"""sigma(E) by six methods, with the offline (fit) and online times kept apart.

The evaluation theta is NOT in the training sample: otherwise the emulators are
queried at a point they have seen, and the error means nothing. The timings
would stay valid either way.

Run from the package directory:  python xsec_methods.py
"""

# Run-from-anywhere: put this dir (siblings) then the package root (core/, pipeline/) on the path.
import sys as _sys
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve().parent
_sys.path[:0] = [str(_HERE), str(_HERE.parent)]
import time

import numpy as np
import matplotlib.pyplot as plt

from core import dbmm, lrom_dbmm, python_rmatrix as rmat, rbm as rbm_mod
from core import rbm_et, specfun
from pipeline.observables import cross_section, sample_theta, solver_for
from pipeline.settings import declared, xsec_kind
from pipeline.systems import SYSTEMS

# NumbaBackend ahead of Arb for the mesh: ~3x, at the price of 7e-12 error.
specfun.SF = specfun.SpecialFunctions(chain=(
    specfun.ScipyBackend, specfun.NumbaBackend,
    specfun.FlintBackend, specfun.MpmathBackend))
dbmm.SF = specfun.SF
specfun.SF.coulomb_f_mesh(0, 1.0, np.linspace(0.1, 5, 10))   # JIT warm-up, outside the timers

KEYS = ["n40Ca", "alpha24Mg", "n238U"]
N_E = 200
N_ET = 25             # (theta, E) samples per window, as figures/fitting.py does
E = np.linspace(0.0, 50.0, N_E)
E[0] = 1e-1           # E = 0 gives k = 0 and a division by zero


# --- the methods: each returns (list of S, t_offline, t_online) ---

def m_dbmm(s, st, theta):
    t0 = time.perf_counter()
    sv = solver_for(s, theta, energies=E)          # construction + Coulomb precompute
    t_off = time.perf_counter() - t0
    t0 = time.perf_counter()
    S = [sv.solve(e) for e in E]
    return S, t_off, time.perf_counter() - t0


def m_rmatrix(s, st, theta):
    channels, model, _, _ = s.preset_fn()
    V_diag, V_coup = model(theta)
    if V_coup is None:
        def V_coup(r, ca, cb):
            return np.zeros_like(r, dtype=complex)

    # H does not depend on E, only the Bloch matrix does: building it inside
    # the loop would cost a factor N_E and skew the comparison.
    t0 = time.perf_counter()
    mesh, _ = rmat.compute_lagrange_mesh(s.N, s.R)
    lag_b = rmat.eval_lag_at_boundary(mesh, s.R)
    H = rmat.compute_CC_ham(mesh, s.R, channels, V_diag, V_coup)
    t_off = time.perf_counter() - t0

    t0 = time.perf_counter()
    S = []
    for e in E:
        B = rmat.make_boundary_matrix(mesh, s.R, channels, e)
        Se, _ = rmat.compute_S_matrix(H + B, lag_b, channels, e, s.R, mesh,
                                      compute_wfs=False)
        S.append(Se)
    return S, t_off, time.perf_counter() - t0


def _per_energy(s, theta, build, n_train):
    """RBM and LROM: one fit per energy, hence N_E fits for a single curve."""
    _, model, theta_c, _ = s.preset_fn()
    th_tr = sample_theta(s, n_train, seed=1)
    S, t_off, t_on = [], 0.0, 0.0
    for e in E:
        t0 = time.perf_counter()
        emu = build(solver_for(s, theta_c, energies=[e]), e, model, th_tr, theta_c)
        t_off += time.perf_counter() - t0
        t0 = time.perf_counter()
        S.append(emu.predict(theta))
        t_on += time.perf_counter() - t0
    return S, t_off, t_on


def m_rbm(s, st, theta):
    def build(base, e, model, th_tr, theta_c):
        emu = rbm_mod.RBMEmulator(base, e)
        emu.fit(model, th_tr, eps_tol=1e-10)
        return emu
    # the caps from figures/fig_cross_sections.py: N_E fits, they have to be affordable
    return _per_energy(s, theta, build, min(30, st.n_train_rbm))


def m_lrom(s, st, theta):
    def build(base, e, model, th_tr, theta_c):
        emu = lrom_dbmm.LROMEmulator(base, e)
        emu.fit(model, th_tr, theta_c, K_diag=st.K_diag,
                K_coup=st.K_coup if s.coupled else 0, eps_tol=1e-10)
        return emu
    return _per_energy(s, theta, build, min(40, st.n_train_lrom))


def _et(s, st, theta, parametric):
    """ET/ETP: one fit per threshold window, then a dense curve.

    fit() raises ValueError if E_train crosses a threshold: the number of open
    channels changes, and a reduced basis cannot span both sides. The dispatch
    by window is the one WindowedEmulatorET performs, inlined here.
    """
    _, model, theta_c, bounds = s.preset_fn()
    cuts = [t for t in s.thresholds if E[0] < t < E[-1]]
    win = np.searchsorted(cuts, E, side="right")

    S, t_off, t_on = [None] * len(E), 0.0, 0.0
    for w in np.unique(win):
        idx = np.where(win == w)[0]
        Ew = E[idx]
        # (theta, E) drawn jointly by Latin hypercube over the extended bounds.
        # Pairing LHS thetas with a monotone linspace in E would sweep only a
        # 1-D curve of the (theta, E) plane, and the reduced basis collapses.
        smp = rbm_mod.latin_hypercube(
            N_ET, np.vstack([bounds, [[Ew[0], Ew[-1]]]]), seed=1)
        th_tr, E_tr = smp[:, :-1], smp[:, -1]

        t0 = time.perf_counter()
        base = solver_for(s, theta_c, energies=np.concatenate([E_tr, Ew]))
        if parametric:
            emu = rbm_et.RBMEmulatorETP(base)
            emu.fit(model, th_tr, E_tr, theta_c, eps_tol=1e-10,
                    K_diag=st.K_diag, K_coup=st.K_coup if s.coupled else 0)
        else:
            emu = rbm_et.RBMEmulatorET(base)
            emu.fit(model, th_tr, E_tr, eps_tol=1e-10)
        t_off += time.perf_counter() - t0

        t0 = time.perf_counter()
        for i, e in zip(idx, Ew):
            S[i] = emu.predict(theta, e)
        t_on += time.perf_counter() - t0
    return S, t_off, t_on


def m_et(s, st, theta):
    return _et(s, st, theta, parametric=False)


def m_etp(s, st, theta):
    return _et(s, st, theta, parametric=True)


METHODS = [("DBMM", m_dbmm), ("R-matrix", m_rmatrix), ("RBM", m_rbm), ("LROM", m_lrom), ("RBM_ET", m_et), ("RBM_ETP", m_etp)]


# --- main loop ---

for key in KEYS:
    s = SYSTEMS[key]
    st = declared(key)                  # the settings the cached results were built with
    kind = xsec_kind(s, st)
    theta = sample_theta(s, 1, seed=999, include_center=False)[0]   # outside the training set
    ref = solver_for(s, theta, energies=E)     # converts S -> sigma for every method
    print(f"\n=== {key}  (Nc={s.n_channels}, N={s.N}, {kind}) ===")
    # Median AND max: the median is the project's convention (pipeline/metrics.py),
    # but the max alone misleads the other way. On a grid reaching below the
    # Coulomb barrier it is set by the first point, where sigma is 1e-11 fm^2
    # and a 100% relative error represents nothing.
    print(f"{'method':10s} {'offline':>9s} {'online':>9s} {'ms/pt':>8s} "
          f"{'err med':>9s} {'err max':>9s}")

    sig_ref, curves = None, {}
    for name, fn in METHODS:
        try:
            S, t_off, t_on = fn(s, st, theta)
        except Exception as exc:                # one failing method must not kill the run
            print(f"{name:10s} failed: {type(exc).__name__}: {exc}")
            continue
        sigma = np.array([cross_section(ref, e, Se, kind)
                          for e, Se in zip(E, S)])   # conversion, outside the timers
        if sig_ref is None:
            sig_ref = sigma
        err = np.abs(sigma - sig_ref) / (np.abs(sig_ref) + 1e-30)
        curves[name] = sigma
        print(f"{name:10s} {t_off:8.2f}s {t_on:8.2f}s {1e3*t_on/N_E:7.2f}  "
              f"{np.nanmedian(err):8.1e} {np.nanmax(err):9.1e}")

    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for name, sigma in curves.items():
        ax.plot(E, sigma, lw=1.0, label=name)
    ax.set_xlabel(r"$E_{\rm c.m.}$ (MeV)")
    ax.set_ylabel(r"$\sigma$ (fm$^2$)")
    ax.set_yscale("log")
    ax.set_title(s.label)
    ax.legend(fontsize=8, frameon=False)
    fig.savefig(f"xsec_methods_{key}.png", dpi=150)
    plt.close(fig)
