"""Emulator against emulator on the shared n+40Ca case.

The two LROM implementations, compared on the radial wave function (what both
predict) in relative L2 norm against each code's own high-fidelity solver, which
avoids the matching conventions that separate their quoted observables.

Run at matched rank, not each code's calibrated setting, so what is measured is
the two implementations, not the two offline budgets. Only simple_lrom.py is
used from Giuliani's package; the other demos import a module it does not ship.

Run with a virtualenv that has Giuliani's LROM package installed.
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

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent / "Learning-Reduced-Order-Models-main"
sys.path.insert(0, str(REPO))

import rose
from rose import SchroedingerEquation
from rose.basis import CustomBasis
from rose.interaction import Interaction
from rose.spin_orbit import SpinOrbitTerm

from lrom_demo import simple_lrom as sl

from compare_rose import central, spin_orbit, E_LAB, TARGET, PROJECTILE
from core.constants import AMU
from core.potentials import (build_channels, woods_saxon, woods_saxon_surface,
                             thomas_so, ls_coupling)
from core.lrom_dbmm import LROMEmulator
from core.dbmm import DBMMSolver
from pipeline.presets import preset_n40Ca_woods_saxon

LEVEL = dict(l=1, s=0.5, j=1.5)
N_TRAIN = 100
N_VALID = 30
RANKS = (4, 8, 16, 22)
RHO_MAX = 8 * np.pi
#: Both errors are read on this radial grid, in fm, and on the TOTAL wave
#: function. Without that the two are not comparable: his mesh runs to about
#: 31 fm in 700 uniform points and ours to 15 fm in 64 Gauss-Legendre points,
#: and our emulator returns the scattered wave by default.
R_GRID = np.linspace(0.1, 15.0, 200)


def draws(bounds, n, seed):
    rng = np.random.default_rng(seed)
    return bounds[:, 0] + rng.random((n, bounds.shape[0])) * (bounds[:, 1] - bounds[:, 0])


def rose_side(theta_c, train, valid, mu, e_com, rank, K=10):
    """His LROM: CustomBasis on the central solution, delta-maxvol, central fit."""
    ldots = ls_coupling(LEVEL["l"], LEVEL["j"], LEVEL["s"])
    interaction = Interaction(
        ell=LEVEL["l"], coordinate_space_potential=central, n_theta=10,
        mu=mu, energy=e_com, is_complex=True,
        spin_orbit_term=SpinOrbitTerm(spin_orbit, ldots))
    solver = SchroedingerEquation.make_base_solver(
        rk_tols=[1e-9, 1e-9]).clone_for_new_interaction(interaction)
    rho = np.linspace(1e-8, RHO_MAX, 700)

    phi_train = np.array([solver.phi(a, rho) for a in train])
    phi_c = solver.phi(theta_c, rho)
    basis = CustomBasis(solutions=phi_train.T.copy(), phi_0=phi_c.copy(),
                        rho_mesh=rho, n_basis=rank, solver=solver,
                        subtract_phi0=True, use_svd=True, center=False, scale=False)

    X = basis.vectors                                   # (n_rho, rank)
    def coeffs_of(phis):
        return np.linalg.lstsq(X, (phis - phi_c[None, :]).T, rcond=None)[0].T

    pack = sl.delta_maxvol_predictor_pack(interaction, train, theta_c, rho,
                                          n_predictors=K, min_s=0.5)
    p_train = sl.centered_potential_predictors(interaction, train, pack)
    model = sl.fit_central_lrom("rose", p_train, coeffs_of(phi_train))

    p_valid = sl.centered_potential_predictors(interaction, valid, pack)
    t0 = time.perf_counter()
    a_pred = sl.predict_coefficients(model, p_valid)
    t_on = (time.perf_counter() - t0) / len(valid)

    phi_pred = phi_c[None, :] + a_pred @ X.T
    phi_exact = np.array([solver.phi(a, rho) for a in valid])
    # onto the common grid: his mesh is in rho = k r
    rho_common = interaction.k * R_GRID
    def on_grid(rows):
        return np.array([np.interp(rho_common, rho, row) for row in rows])
    err = sl.relative_l2_rows(on_grid(phi_pred), on_grid(phi_exact))
    return np.median(err), t_on


def ours_side(theta_c, train, valid, e_com, rank, K=10):
    """Our LROM on the same channel, same draws, forced to the same rank."""
    channels, _ = build_channels(A1=1, A2=40, z1z2=0, level_specs=[LEVEL])

    def model_fn(theta):
        Vv, Wv, Wd, Vso, Rv, Rd, Rso, av, ad, aso = theta

        def V_diag(r, ch):
            return (-(Vv + 1j * Wv) * woods_saxon(r, Rv, av)
                    - 1j * Wd * woods_saxon_surface(r, Rd, ad)
                    - Vso * thomas_so(r, Rso, aso)
                    * ls_coupling(ch["l"], ch["j"], ch["s"]))
        return V_diag, None

    base = DBMMSolver(N=64, R=15.0, channels=channels,
                      V_diag=model_fn(theta_c)[0])
    emu = LROMEmulator(base, e_com)
    # eps_tol below anything reachable, nb_max as the cap: the rank is then
    # exactly `rank`, which is what makes the two columns comparable.
    emu.fit(model_fn, train, theta_c, K_diag=K, K_coup=0,
            eps_tol=1e-16, nb_max=rank)

    errs = []
    t = []
    for theta in valid:
        V_diag, _ = model_fn(theta)
        exact = np.ravel(DBMMSolver(N=64, R=15.0, channels=channels,
                                    V_diag=V_diag).wavefunction(
                                        e_com, r_eval=R_GRID, total=True)[1])
        # Time the reduced solve alone, which is what his side times: his
        # lifting a_pred @ X.T sits outside his timer, so timing our
        # wavefunction(), which lifts onto 200 radii and adds the incident
        # wave, would not be the same measurement.
        t0 = time.perf_counter()
        emu.predict_coefficients(theta)
        t.append(time.perf_counter() - t0)
        pred = np.ravel(emu.wavefunction(theta, r_eval=R_GRID, total=True)[1])
        errs.append(np.linalg.norm(pred - exact) / np.linalg.norm(exact))
    return np.median(errs), np.median(t)


def main():
    _, _, theta_c, bounds = preset_n40Ca_woods_saxon()
    _mu, e_com, _k, _eta = rose.kinematics(target=TARGET, projectile=PROJECTILE,
                                           E_lab=E_LAB)
    mu = 40.0 / 41.0 * AMU
    train = draws(bounds, N_TRAIN, 20260819)
    valid = draws(bounds, N_VALID, 20260820)

    print("n+40Ca, l=1 j=3/2, E_cm = %.4f MeV, common reduced mass %.2f" % (e_com, mu))
    print("%d training draws, %d validation draws, K = 10 predictors both sides\n"
          % (N_TRAIN, N_VALID))
    print("%-6s %-32s %-32s" % ("rank", "repository LROM", "this work"))
    for rank in RANKS:
        try:
            e_r, t_r = rose_side(theta_c, train, valid, mu, e_com, rank)
            r_txt = "err %.2e   online %.3f ms" % (e_r, 1e3 * t_r)
        except Exception as exc:
            r_txt = "failed: %s" % type(exc).__name__
        try:
            e_o, t_o = ours_side(theta_c, train, valid, e_com, rank)
            o_txt = "err %.2e   online %.3f ms" % (e_o, 1e3 * t_o)
        except Exception as exc:
            o_txt = "failed: %s" % type(exc).__name__
        print("%-6d %-32s %-32s" % (rank, r_txt, o_txt))


def sweep_K():
    """Is the K = 10 of the matched run handicapping either side?"""
    _, _, theta_c, bounds = preset_n40Ca_woods_saxon()
    _mu, e_com, _k, _eta = rose.kinematics(target=TARGET, projectile=PROJECTILE,
                                           E_lab=E_LAB)
    mu = 40.0 / 41.0 * AMU
    train = draws(bounds, N_TRAIN, 20260819)
    valid = draws(bounds, N_VALID, 20260820)
    print()
    print("rank 22, predictor budget swept")
    print("%-4s %-26s %-26s" % ("K", "repository LROM", "this work"))
    for K in (6, 10, 16, 24):
        try:
            e_r, _ = rose_side(theta_c, train, valid, mu, e_com, 22, K=K)
            r = "err %.2e" % e_r
        except Exception as exc:
            r = "failed: %s" % type(exc).__name__
        try:
            e_o, _ = ours_side(theta_c, train, valid, e_com, 22, K=K)
            o = "err %.2e" % e_o
        except Exception as exc:
            o = "failed: %s" % type(exc).__name__
        print("%-4d %-26s %-26s" % (K, r, o))


if __name__ == "__main__":
    main()
    sweep_K()
