"""Invariants the solution must possess whatever code produced it.

These are the checks that caught what the code comparison did not. Two
solvers written in the same group, sharing a mesh and a set of conventions,
can agree with each other and both be wrong; a property required by the
physics cannot be talked into agreement.

Run with:  python -m pytest tests/ -v
       or:  python tests/test_invariants.py
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.systems import SYSTEMS
from pipeline.observables import solver_for
from core.dbmm import DBMMSolver, flux_factor_disabled
from pipeline.metrics import symmetry_violation, unitarity_violation

COUPLED = ["alpha12C_band", "alpha24Mg", "n238U"]


def _distinct_thresholds(system):
    """Guard. On degenerate thresholds the velocity factor is unity, so both
    invariants would pass for the wrong reason and prove nothing."""
    thr = {float(c["threshold"]) for c in system.channels}
    return len(thr) > 1


def _all_open_energy(system):
    thr = system.thresholds
    lo, hi, _ = system.E_range
    return float(min(hi * 0.97, max(thr) * 1.15)) if thr else float(hi * 0.97)


@pytest.mark.parametrize("key", COUPLED)
def test_thresholds_are_distinct(key):
    """The guard itself, asserted rather than assumed."""
    assert _distinct_thresholds(SYSTEMS[key]), (
        f"{key} has degenerate thresholds, so the symmetry and unitarity tests "
        "below would pass whether or not the flux factor is applied")


@pytest.mark.parametrize("key", COUPLED)
def test_collision_matrix_is_symmetric(key):
    """Time reversal invariance requires U_ab = U_ba."""
    system = SYSTEMS[key]
    _, _, theta_c, _ = system.preset_fn()
    E = _all_open_energy(system)
    S = solver_for(system, theta_c, energies=[E]).solve(E)
    assert symmetry_violation(S) < 1e-3, (
        f"{key}: max |U_ab - U_ba| = {symmetry_violation(S):.2e}")


@pytest.mark.parametrize("key", COUPLED)
def test_flux_is_conserved_without_absorption(key):
    """With a real potential U must be unitary.

    The absorption is removed by taking the real part of the interaction, not by
    zeroing a parameter: some systems carry a surface absorption W_d0 as well as
    a volume one, and zeroing one index leaves the potential absorptive.

    Evaluated on the open sub-block: a closed channel carries an identically
    zero row and column, which contributes an exact floor of 1.0 to the full
    matrix norm whatever the physics.
    """
    system = SYSTEMS[key]
    channels, model, theta_c, _ = system.preset_fn()
    V_diag, V_coup = model(theta_c)

    def real_diag(r, ch):
        return np.asarray(V_diag(r, ch), dtype=complex).real.astype(complex)

    real_coup = None
    if V_coup is not None:
        def real_coup(r, ca, cb):
            return np.asarray(V_coup(r, ca, cb), dtype=complex).real.astype(complex)

    E = _all_open_energy(system)
    solver = DBMMSolver(system.N, system.R, channels, real_diag, real_coup)
    S = solver.solve(E)
    assert unitarity_violation(S) < 1e-2, (
        f"{key}: ||U^H U - 1|| = {unitarity_violation(S):.2e}")


@pytest.mark.parametrize("key", COUPLED)
def test_the_missing_factor_is_not_a_discretisation_error(key):
    """Removing the velocity factor leaves an error that does NOT fall with N.

    This replaces a test that asserted `bad > 100 * good` at the declared mesh.
    That ratio is not a measure of discriminating power: `good` is the
    discretisation error, which falls with N, so the ratio grows without bound
    as the mesh is refined. alpha+24Mg read 32x at N = 100 and 116x at N = 160 --
    the same physics, passing or failing depending on a mesh size. The threshold
    was arbitrary and the test was measuring convergence.

    What F3 actually claims is mesh-independent, and is asserted here: between
    the two finest meshes the corrected error keeps falling while the
    uncorrected one sits on a floor. Measured across the three coupled systems,
    the floor moves by 0.5 to 0.8 per cent while the corrected error falls by
    factors of 1.9 to 187.
    """
    system = SYSTEMS[key]
    channels, model, theta_c, _ = system.preset_fn()
    E = _all_open_energy(system)
    V_diag, V_coup = model(theta_c)

    def violation(N, disabled):
        sv = DBMMSolver(N, system.R, channels, V_diag, V_coup)
        if not disabled:
            return symmetry_violation(sv.solve(E))
        with flux_factor_disabled():
            return symmetry_violation(sv.solve(E))

    coarse, fine = 120, 160
    good_c, good_f = violation(coarse, False), violation(fine, False)
    bad_c, bad_f = violation(coarse, True), violation(fine, True)

    assert good_f < good_c / 1.5, (
        f"{key}: with the flux factor the asymmetry did not fall with N "
        f"({good_c:.2e} at N={coarse} -> {good_f:.2e} at N={fine}), so this "
        f"test cannot distinguish a floor from convergence")
    assert bad_f > bad_c / 1.3, (
        f"{key}: without the flux factor the asymmetry FELL with N "
        f"({bad_c:.2e} -> {bad_f:.2e}), which would mean the missing factor is "
        f"a discretisation error after all")


def test_discretisation_error_falls_with_mesh_size():
    """A discretisation error falls with N. An error that does not is in the
    formulation, which is the diagnostic that identified the missing factor."""
    system = SYSTEMS["alpha12C_band"]
    channels, model, theta_c, _ = system.preset_fn()
    E = _all_open_energy(system)
    V_diag, V_coup = model(theta_c)
    viol = [symmetry_violation(
        DBMMSolver(N, system.R, channels, V_diag, V_coup).solve(E))
        for N in (60, 90, 120)]
    assert viol[-1] < viol[0] / 10, (
        f"asymmetry did not fall with N: {viol}")


def test_matches_published_reference_above_the_highest_threshold():
    """alpha+12C against Example 4 of the reference package.

    Compared only above the highest threshold. Below it the published values
    are not converged in channel radius: across the three radii the reference
    itself reports, 9, 10 and 11 fm, its closed-channel amplitudes move by
    factors of two to three, while this calculation is stable to 3e-4 from
    14 fm out to 30 fm.
    """
    published = {                       # output4.txt, channel radius 11 fm
        16.0: [4.3124e-2, 2.7057e-2, 2.9530e-2, 3.4450e-2],
        20.0: [2.8037e-2, 1.9167e-2, 1.9080e-2, 2.0027e-2],
    }
    system = SYSTEMS["alpha12C_band"]
    _, _, theta_c, _ = system.preset_fn()
    solver = solver_for(system, theta_c, energies=list(published))
    for E, ref in published.items():
        got = np.abs(solver.solve(E)[:len(ref), 0])
        dev = float(np.max(np.abs(got - ref) / np.array(ref)))
        assert dev < 2e-3, f"E = {E} MeV: max relative deviation {dev:.2e}"


def test_uncoupled_system_has_no_inelastic_error():
    """n+40Ca solves two spin-orbit blocks that are never coupled, so every
    off-diagonal element of S is structurally zero and any relative error on
    them is meaningless. The guard must return NaN rather than a number."""
    from pipeline.metrics import coupling_mask, err_inelastic
    system = SYSTEMS["n40Ca"]
    _, _, theta_c, _ = system.preset_fn()
    E = float(system.E_demo)
    S = solver_for(system, theta_c, energies=[E]).solve(E)
    assert not coupling_mask(S[None]).any()
    assert np.isnan(err_inelastic(S[None], S[None]))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--no-header", "-x"]))
