"""Sum J^pi blocks on alpha+24Mg, solver against both emulators.

The relative error on a sum of blocks is bounded by the worst of the errors on
the terms; the pipeline uses that bound but never exercises it. This does: the
same interaction solved at every J^pi from 0 to J_MAX with the statistical
weight (2J+1) a reaction cross section carries, each block emulated on its own
basis, and the weighted sums compared.

Below the grazing angular momentum (J up to ~12 at 54 MeV here) every block sits
at the unitarity limit pi/k^2 and barely responds to theta; all the parameter
sensitivity lives in the grazing region, J of 14 to 22, where the weighted sum
gets its error. Summing a few low blocks tests nothing.

    python block_sum_study.py

Writes nothing. Prints a table.
"""

# Run-from-anywhere: put this dir (siblings) then the package root (core/, pipeline/) on the path.
import sys as _sys
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve().parent
_sys.path[:0] = [str(_HERE), str(_HERE.parent)]
import time

import numpy as np

from core.dbmm import DBMMSolver
from core.lrom_dbmm import LROMEmulator
from core.potentials import (build_channels, rotational_channels,
                             rotational_xfac, woods_saxon, woods_saxon_surface)
from core.rbm import RBMEmulator, latin_hypercube, observables_from_coeffs

E_2PLUS, E_4PLUS = 1.369, 4.123
SPINS = [0, 2, 4]
THRESHOLDS = [0.0, E_2PLUS, E_4PLUS]
A1, A2, Z1Z2 = 4, 24, 2 * 12
N, R, E_DEMO = 100, 15.0, 54.0

R0_C = 1.2 * A2 ** (1 / 3)
THETA_C = np.array([100.0, 10.0, 20.0, R0_C, 0.65, 0.43, -0.11])
BOUNDS = np.array([[70.0, 140.0], [5.0, 20.0], [8.0, 30.0],
                   [R0_C * 0.85, R0_C * 1.15], [0.50, 0.80],
                   [0.35, 0.51], [-0.17, -0.05]])

J_MAX = 26
N_TRAIN, N_VALID = 190, 40
SEED_TRAIN, SEED_VALID = 20260601, 20260602
EPS_POST = 5e-3


def build(J):
    """channels, model for one J^pi block of the same interaction."""
    specs = rotational_channels(J, SPINS, THRESHOLDS)
    X2 = rotational_xfac(specs, J=J, lam=2)
    X4 = rotational_xfac(specs, J=J, lam=4)
    channels, _ = build_channels(A1=A1, A2=A2, z1z2=Z1Z2,
                                 uniform_sphere_coulomb=True, level_specs=specs)

    def model(theta):
        V0, W0, Wd0, R0, a0, b2, b4 = theta

        def _xf(a, b, R0):
            return X2[a, b] * b2 * R0 + X4[a, b] * b4 * R0

        def V_diag(r, ch):
            i = ch['idx']
            return (-(V0 + 1j * W0) * woods_saxon(r, R0, a0)
                    - 1j * Wd0 * woods_saxon_surface(r, R0, a0)
                    - _xf(i, i, R0) * woods_saxon_surface(r, R0, a0))

        def V_coup(r, ch_a, ch_b):
            return -_xf(ch_a['idx'], ch_b['idx'], R0) * woods_saxon_surface(r, R0, a0)

        return V_diag, V_coup

    return channels, model


def sigma_from_S(solver, E, S):
    """Block reaction cross section in the entrance channel, unit weight.

    Entrance is index 0: rotational_channels emits the I = 0 level first, and
    its single l = J channel is the elastic entrance of the block.
    """
    if not np.all(np.isfinite(S)):
        return np.nan
    if float(np.linalg.svd(S, compute_uv=False)[0]) > 1.1:
        return np.nan
    return float(solver.reaction_cross_section(E, S=S)[0])


def main():
    th_train = latin_hypercube(N_TRAIN, BOUNDS, seed=SEED_TRAIN)
    th_valid = np.vstack([THETA_C[None, :],
                          latin_hypercube(N_VALID - 1, BOUNDS, seed=SEED_VALID)])

    rows, tot_ref, tot_rbm, tot_lrom = [], 0.0, 0.0, 0.0
    t_start = time.perf_counter()
    for J in range(J_MAX + 1):
        channels, model = build(J)
        Nc = len(channels)
        Vd, Vc = model(THETA_C)
        base = DBMMSolver(N, R, channels, Vd, Vc)
        base.precompute_coulomb(np.array([E_DEMO]))

        # Reference through the emulator's own snapshot path: one K per block,
        # so the reference costs one LU per theta and not one Coulomb precompute.
        probe = RBMEmulator(base, E_DEMO)
        C_val = probe.snapshots(model, th_valid)
        ref = np.array([sigma_from_S(
            base, E_DEMO,
            observables_from_coeffs(C_val[:, i * Nc:(i + 1) * Nc], base, E_DEMO).S)
            for i in range(len(th_valid))])

        rbm = RBMEmulator(base, E_DEMO)
        rbm.fit(model, th_train, eps_tol=1e-8)
        nb = rbm.nb
        K = min(16, max(6, nb // 3))
        lrom = LROMEmulator(base, E_DEMO)
        lrom.fit(model, th_train, THETA_C, K_diag=K, K_coup=0,
                 eps_tol=1e-14, nb_max=nb)

        srbm = np.array([sigma_from_S(base, E_DEMO, rbm.predict(t)) for t in th_valid])
        slrom = np.array([sigma_from_S(base, E_DEMO, lrom.predict(t)) for t in th_valid])

        w = 2 * J + 1
        tot_ref = tot_ref + w * ref
        tot_rbm = tot_rbm + w * srbm
        tot_lrom = tot_lrom + w * slrom

        er = float(np.nanmedian(np.abs(srbm - ref) / np.abs(ref)))
        el = float(np.nanmedian(np.abs(slrom - ref) / np.abs(ref)))
        spread = float((np.nanmax(ref) - np.nanmin(ref)) / np.nanmedian(ref))
        rows.append((J, Nc, nb, w, ref[0], w * ref[0], spread, er, el))
        print("  J=%-3d Nc=%d nb=%-4d w=%-3d sigma=%8.5f  spread=%7.1e  "
              "RBM %.2e  LROM %.2e   [%.0f s]"
              % (J, Nc, nb, w, ref[0], spread, er, el,
                 time.perf_counter() - t_start), flush=True)

    e_rbm = np.abs(tot_rbm - tot_ref) / np.abs(tot_ref)
    e_lrom = np.abs(tot_lrom - tot_ref) / np.abs(tot_ref)
    worst_rbm = max(r[7] for r in rows)
    worst_lrom = max(r[8] for r in rows)

    print("\n" + "=" * 78)
    print("sigma_R at theta_c, summed over J = 0..%d with weight (2J+1): %.3f fm^2"
          % (J_MAX, tot_ref[0]))
    print("the last block adds %.4f fm^2, that is %.2e of the total"
          % (rows[-1][5], rows[-1][5] / tot_ref[0]))
    print("\nerror on the SUM, over %d validation draws" % len(th_valid))
    print("  median   RBM %.2e   LROM %.2e" % (np.nanmedian(e_rbm), np.nanmedian(e_lrom)))
    print("  p95      RBM %.2e   LROM %.2e"
          % (np.nanpercentile(e_rbm, 95), np.nanpercentile(e_lrom, 95)))
    print("\nworst single block, median")
    print("           RBM %.2e   LROM %.2e" % (worst_rbm, worst_lrom))
    print("\nEq. (blocksum): error(sum) <= max_J error(block)")
    print("  RBM  %s      LROM  %s"
          % (np.nanmedian(e_rbm) <= worst_rbm, np.nanmedian(e_lrom) <= worst_lrom))
    print("eps_post = %.0e, the sum clears it:  RBM %s   LROM %s"
          % (EPS_POST, np.nanmedian(e_rbm) < EPS_POST, np.nanmedian(e_lrom) < EPS_POST))
    print("total %.0f s" % (time.perf_counter() - t_start))


if __name__ == "__main__":
    main()
