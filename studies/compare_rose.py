"""Solver against solver on the shared n+40Ca case.

Giuliani's ROSE (Runge-Kutta on the radial equation, tolerance 1e-9) against the
DBMM (one linear system on a Lagrange-Legendre mesh), on his central values and
box. The two share no code, so this is an independent check of the DBMM.

ROSE passes the potential into numba-compiled code, so the forms are rewritten
here with @njit -- the same expressions as core/potentials.py.

Run with a virtualenv that has ROSE installed.
"""

# Run-from-anywhere: put this dir (siblings) then the package root (core/, pipeline/) on the path.
import sys as _sys
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve().parent
_sys.path[:0] = [str(_HERE), str(_HERE.parent)]
from pathlib import Path
import sys
import time

import numpy as np
from numba import njit

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import rose
from rose import SchroedingerEquation
from rose.interaction import Interaction
from rose.spin_orbit import SpinOrbitTerm

from core.potentials import (build_channels, woods_saxon, woods_saxon_surface,
                             thomas_so, ls_coupling)
from core.constants import AMU
from core.dbmm import DBMMSolver
from pipeline.presets import preset_n40Ca_woods_saxon

E_LAB = 14.1
TARGET, PROJECTILE = (40, 20), (1, 0)


# No clipping: numba cannot vectorise a two-argument min/max, and rose calls
# these both pointwise inside the integrator and on a whole mesh through
# Interaction.tilde. On this mesh the exponent stays under 60, so exp does not
# overflow and plain arithmetic works for a scalar and for an array alike.
@njit
def _ws(r, R, a):
    return 1.0 / (1.0 + np.exp((r - R) / a))


@njit
def _surface(r, R, a):
    ex = np.exp((r - R) / a)
    return 4.0 * ex / (1.0 + ex) ** 2


@njit
def _thomas(r, R, a):
    ex = np.exp((r - R) / a)
    return ex / (1.0 + ex) ** 2 / (a * r)


@njit
def central(r, alpha):
    """Real and imaginary volume plus imaginary surface, as in the preset."""
    return (-(alpha[0] + 1j * alpha[1]) * _ws(r, alpha[4], alpha[7])
            - 1j * alpha[2] * _surface(r, alpha[5], alpha[8]))


@njit
def spin_orbit(r, alpha, ell_dot_s):
    return -alpha[3] * _thomas(r, alpha[6], alpha[9]) * ell_dot_s


def our_smatrix(theta, E_cm, level):
    """Diagonal S-matrix element of one spin-orbit channel, from the DBMM."""
    channels, _ = build_channels(A1=1, A2=40, z1z2=0, level_specs=[level])
    Vv, Wv, Wd, Vso, Rv, Rd, Rso, av, ad, aso = theta

    def V_diag(r, ch):
        return (-(Vv + 1j * Wv) * woods_saxon(r, Rv, av)
                - 1j * Wd * woods_saxon_surface(r, Rd, ad)
                - Vso * thomas_so(r, Rso, aso)
                * ls_coupling(ch["l"], ch["j"], ch.get("s", 0.5)))

    solver = DBMMSolver(N=64, R=15.0, channels=channels, V_diag=V_diag)
    return complex(np.atleast_2d(solver.solve(E_cm))[0, 0])


def rose_solver(level, mu, e_com):
    ldots = ls_coupling(level["l"], level["j"], level.get("s", 0.5))
    interaction = Interaction(
        ell=level["l"],
        coordinate_space_potential=central,
        n_theta=10, mu=mu, energy=e_com, is_complex=True,
        spin_orbit_term=SpinOrbitTerm(spin_orbit, ldots),
    )
    base = SchroedingerEquation.make_base_solver(rk_tols=[1e-9, 1e-9])
    return base.clone_for_new_interaction(interaction)


def main():
    _, _, theta_c, bounds = preset_n40Ca_woods_saxon()
    mu_rose, e_com, _k, _eta = rose.kinematics(target=TARGET,
                                               projectile=PROJECTILE,
                                               E_lab=E_LAB)
    # ROSE builds the reduced mass from measured masses, we build it from mass
    # NUMBERS (Appendix F). The two differ by 1.5 per cent, which moves k by
    # 0.7 per cent and the S-matrix by a few per cent. That is a convention and
    # not a numerical error, so ROSE is given our mu here: the comparison then
    # measures the two integrators and not the two mass tables.
    mu = 40.0 / 41.0 * AMU
    print("E_lab = %.1f MeV, E_cm = %.4f MeV" % (E_LAB, e_com))
    print("reduced mass: ours %.3f, ROSE %.3f MeV/c^2; ours used for both\n"
          % (mu, mu_rose))

    rng = np.random.default_rng(20260819)
    draws = [theta_c] + [
        bounds[:, 0] + rng.random(theta_c.size) * (bounds[:, 1] - bounds[:, 0])
        for _ in range(20)
    ]

    for level in (dict(l=1, s=0.5, j=1.5), dict(l=1, s=0.5, j=0.5)):
        solver = rose_solver(level, mu, e_com)
        rel, t_ours, t_rose = [], [], []
        print("channel l=%d j=%.1f" % (level["l"], level["j"]))
        for i, theta in enumerate(draws):
            t0 = time.perf_counter()
            s_ours = our_smatrix(theta, e_com, level)
            t_ours.append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            s_rose = complex(solver.smatrix(np.asarray(theta, dtype=float)))
            t_rose.append(time.perf_counter() - t0)

            rel.append(abs(s_ours - s_rose) / abs(s_rose))
            if i < 3:
                print("   draw %-2d DBMM % .6f%+.6fj   ROSE % .6f%+.6fj   rel %.2e"
                      % (i, s_ours.real, s_ours.imag,
                         s_rose.real, s_rose.imag, rel[-1]))
        rel = np.array(rel)
        print("   over %d draws: median %.2e   max %.2e" % (rel.size, np.median(rel), rel.max()))
        print("   solve time: DBMM %.2f ms   ROSE %.2f ms\n"
              % (1e3 * np.median(t_ours[1:]), 1e3 * np.median(t_rose[1:])))




def convergence():
    """Is the residual ours? Refine the mesh and see whether it falls."""
    _, _, theta_c, _ = preset_n40Ca_woods_saxon()
    _mu_rose, e_com, _k, _eta = rose.kinematics(target=TARGET,
                                                projectile=PROJECTILE,
                                                E_lab=E_LAB)
    mu = 40.0 / 41.0 * AMU
    level = dict(l=1, s=0.5, j=1.5)
    solver = rose_solver(level, mu, e_com)
    s_rose = complex(solver.smatrix(np.asarray(theta_c, dtype=float)))

    Vv, Wv, Wd, Vso, Rv, Rd, Rso, av, ad, aso = theta_c
    channels, _ = build_channels(A1=1, A2=40, z1z2=0, level_specs=[level])

    def V_diag(r, ch):
        return (-(Vv + 1j * Wv) * woods_saxon(r, Rv, av)
                - 1j * Wd * woods_saxon_surface(r, Rd, ad)
                - Vso * thomas_so(r, Rso, aso)
                * ls_coupling(ch["l"], ch["j"], ch.get("s", 0.5)))

    print("mesh convergence of the DBMM against ROSE, central parameters")
    for N in (48, 64, 96, 128, 160):
        s = complex(np.atleast_2d(
            DBMMSolver(N=N, R=15.0, channels=channels, V_diag=V_diag).solve(e_com))[0, 0])
        print("   N = %3d   rel = %.2e" % (N, abs(s - s_rose) / abs(s_rose)))


if __name__ == "__main__":
    main()
    convergence()
