"""What the emulators must satisfy, independently of any cached result.

Properties, not regressions: they hold for any mesh, parameter box and
tolerance, so they cannot be satisfied by refreezing a number. Without them an
RBM that silently stopped converging is caught only by a later recompute.

Run with:  python -m pytest tests/ -q
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.lrom_dbmm import LROMEmulator
from core.rbm import RBMEmulator, latin_hypercube
from pipeline.observables import solver_for
from pipeline.plan_calibration import calibration_for
from pipeline.systems import SYSTEMS

#: Small and fast, and one of each kind: uncoupled neutral, uncoupled charged,
#: coupled with a tensor force. The properties below do not depend on size, so
#: paying for alpha+12C to assert them would buy nothing.
CHEAP = ["p12C", "n40Ca", "alpha_d"]


def _fixture(key, n_train=60, seed=1):
    system = SYSTEMS[key]
    channels, model, theta_c, bounds = system.preset_fn()
    E = float(system.E_demo)
    base = solver_for(system, theta_c, energies=[E])
    th_train = latin_hypercube(n_train, bounds, seed=seed)
    # A different seed from the training draws: an emulator scored where it was
    # trained is not being scored.
    th_valid = latin_hypercube(6, bounds, seed=seed + 500)
    return system, model, theta_c, base, E, th_train, th_valid


@pytest.mark.parametrize("key", CHEAP)
def test_rbm_reproduces_the_cross_section_in_sample(key):
    """A training parameter must come back through the emulator.

    Measured on the cross section, deliberately, and NOT on the full matrix
    norm at "full rank". Two things get in the way of the tempting version:

      * `pod_basis` thresholds the cumulative energy, and 1 - eps_tol saturates
        at 1.0 in double precision for any eps_tol below about 1e-16. Asking for
        more modes than that stops working -- on n+40Ca the rank stops at 43
        whatever tolerance is passed -- so "full rank" is not reachable through
        the tolerance and a test that assumes it is testing the arithmetic of
        the criterion, not the emulator.
      * past that point the reduced operator is built from modes whose singular
        values span 1e-8, and the error on the full matrix is no longer monotone
        in rank (1.4e-4 at n_b = 20, 8.0e-3 at 38, 2.2e-4 at 43 on n+40Ca) while
        the cross section is unaffected (5e-6, 1.9e-5, 1.0e-5) -- the property
        worth asserting.

    The pipeline operates well below that regime, because the calibrated rank
    binds before the tolerance does.
    """
    system, model, theta_c, base, E, th_train, _ = _fixture(key)
    from pipeline.observables import cross_section
    from pipeline.settings import xsec_kind
    kind = xsec_kind(system)
    emu = RBMEmulator(base, E).fit(model, th_train, eps_tol=1e-14)
    for theta in th_train[:3]:
        sol = solver_for(system, theta, energies=[E])
        ref = cross_section(sol, E, sol.solve(E), kind)
        got = cross_section(sol, E, emu.predict(theta), kind)
        rel = abs(got - ref) / abs(ref)
        assert rel < 1e-3, f"{key}: in-sample error on sigma {rel:.2e}"


@pytest.mark.parametrize("key", CHEAP)
def test_rbm_error_falls_with_rank(key):
    """More modes, less error -- out of sample, which is the only claim worth
    making. Measured on the median over draws, because the worst is set by
    whichever matrix element sits nearest zero."""
    system, model, theta_c, base, E, th_train, th_valid = _fixture(key)
    S_ref = np.array([solver_for(system, t, energies=[E]).solve(E)
                      for t in th_valid])
    errs = []
    for nb in (2, 4, 12):
        emu = RBMEmulator(base, E).fit(model, th_train, eps_tol=1e-14, nb_max=nb)
        S = np.array([emu.predict(t) for t in th_valid])
        errs.append(float(np.median(
            np.max(np.abs(S - S_ref), axis=(-2, -1))
            / np.max(np.abs(S_ref), axis=(-2, -1)))))
    assert errs[-1] < errs[0], f"{key}: error did not fall with rank: {errs}"


@pytest.mark.parametrize("key", CHEAP)
def test_lrom_is_exact_at_the_central_parameter(key):
    """The central gauge sets a(theta_c) = 0, so the prediction at theta_c is
    the stored snapshot itself.

    This is why every figure is drawn at a validation draw, never theta_c: an
    emulator that cannot fail there proves nothing.
    """
    system, model, theta_c, base, E, th_train, _ = _fixture(key)
    emu = LROMEmulator(base, E).fit(model, th_train, theta_c, K_diag=6,
                                    K_coup=4 if system.coupled else 0,
                                    eps_tol=1e-14, nb_max=12)
    S_ref = solver_for(system, theta_c, energies=[E]).solve(E)
    rel = np.max(np.abs(emu.predict(theta_c) - S_ref)) / np.max(np.abs(S_ref))
    assert rel < 1e-10, f"{key}: LROM is not exact at theta_c ({rel:.2e})"


@pytest.mark.parametrize("key", ["alpha12C_band"])
def test_emulators_do_not_break_the_symmetry_of_U(key):
    """Time reversal does not stop applying because the solve was reduced.

    Against the SOLVER's own asymmetry, not against zero. The solver is
    symmetric only up to its discretisation error -- 1.6e-04 on alpha+d at
    N = 60 -- so a threshold pulled out of the air tests the mesh rather than
    the emulator. What has to hold is that the reduction adds nothing: the
    emulator may inherit that floor and must not exceed it by much.

    And on a system where the factor is far from unity AND at a rank where the
    emulator is accurate enough for an invariant to be visible. Both matter.
    On alpha+24Mg the thresholds sit within 4 MeV of a 54 MeV demonstration
    energy, so sqrt(v_a/v_b) = 1.04 and dropping it moves the asymmetry from
    8.6e-07 to 5.5e-06, which is below the emulator's own error; and at a rank
    below the calibrated one the approximation error swamps the invariant
    whatever the system. On alpha+12C at n_b = 104 the two separate by a factor
    of 350, which is what makes the assertion below mean something.

    Both emulators reconstruct U through core.dbmm.channel_velocities, the same
    single funnel as the solver. This is what asserts that they still do.
    """
    cal = calibration_for(key)
    system, model, theta_c, base, E, th_train, th_valid = _fixture(key,
                                                                  n_train=cal.Ns)
    thresholds = sorted({float(c["threshold"]) for c in system.channels})
    assert len(thresholds) > 1, (
        f"{key} has degenerate thresholds, so this test would pass whether or "
        "not the flux factor is applied")
    # The factor is sqrt(v_a / v_b); at 1.04 it is invisible under the
    # emulator's own error, which is what made this test vacuous before.
    spread = max((E - thresholds[0]) / (E - t) for t in thresholds) ** 0.5
    assert spread > 1.5, (
        f"{key}: sqrt(v_a/v_b) reaches only {spread:.2f}, too close to unity "
        "for this test to distinguish a dropped factor from the emulator error")

    solver_asym = float(np.max(np.abs(
        (lambda S: S - S.T)(solver_for(system, th_valid[0], energies=[E]).solve(E)))))
    rbm = RBMEmulator(base, E).fit(model, th_train, eps_tol=1e-14, nb_max=cal.nb)
    lrom = LROMEmulator(base, E).fit(model, th_train, theta_c,
                                     K_diag=cal.K_diag, K_coup=cal.K_coup,
                                     eps_tol=1e-14, nb_max=cal.nb)
    for emu, name in ((rbm, "RBM"), (lrom, "LROM")):
        S = emu.predict(th_valid[0])
        asym = float(np.max(np.abs(S - S.T)))
        assert asym < max(10 * solver_asym, 1e-6), (
            f"{key}: {name} asymmetry {asym:.2e} against the solver's own "
            f"{solver_asym:.2e}")


@pytest.mark.parametrize("key", CHEAP)
def test_boundary_shortcut_agrees_with_the_full_lift(key):
    """predict() must return what the full-mesh lift returns.

    U depends on the interior solution only through psi(R), so predict() takes
    the basis already contracted with the boundary row instead of lifting the
    reduced coefficients over the mesh. The two paths are the same arithmetic
    in a different order and must agree to roundoff.
    """
    from core.rbm import s_matrix_from_coeffs

    system, model, theta_c, base, E, th_train, th_valid = _fixture(key)
    for emu in (RBMEmulator(base, E).fit(model, th_train),
                LROMEmulator(base, E).fit(model, th_train, theta_c,
                                          K_diag=6, K_coup=4)):
        for th in th_valid:
            fast = emu.predict(th)
            slow = s_matrix_from_coeffs(emu.predict_coefficients(th), base, E)
            assert np.allclose(fast, slow, rtol=0, atol=1e-11), (
                f"{key}: {type(emu).__name__} boundary shortcut disagrees with "
                f"the lift by {np.abs(fast - slow).max():.2e}")


@pytest.mark.parametrize("key", CHEAP)
def test_boundary_shortcut_follows_a_change_of_rank(key):
    """The shortcut must not cache a basis its caller has replaced.

    block_manifold sweeps the rank by assigning a slice of the basis to
    emu.X_r between calls to predict(). A contraction stored at fit time is
    stale from the first slice onwards, and the pipeline failed with a shape
    error on four systems before this was fixed.
    """
    from core.rbm import s_matrix_from_coeffs

    system, model, theta_c, base, E, th_train, th_valid = _fixture(key)
    emu = RBMEmulator(base, E).fit(model, th_train)
    X_full, K_full = emu.X_r, emu.K
    th = th_valid[0]

    for nb in (max(2, emu.nb // 4), max(3, emu.nb // 2), emu.nb):
        emu.X_r = X_full[:, :nb]
        emu.K_r = emu.X_r.conj().T @ K_full @ emu.X_r
        fast = emu.predict(th)
        slow = s_matrix_from_coeffs(emu.predict_coefficients(th), base, E)
        assert np.allclose(fast, slow, rtol=0, atol=1e-11), (
            f"{key}: the shortcut is stale at rank {nb}")


@pytest.mark.parametrize("key", CHEAP)
def test_predictions_do_not_depend_on_the_order_of_the_queries(key):
    """No hidden state between predictions.

    Both emulators memoise -- the basis blocks, the conjugate blocks, the
    solver's per-energy arrays -- and a memo keyed on the wrong thing shows up
    exactly here: the second query returning something that depends on the
    first. Cheap to assert, and it is the failure mode that a parallel map over
    energy nodes would turn into a heisenbug.
    """
    system, model, theta_c, base, E, th_train, th_valid = _fixture(key)
    emu = RBMEmulator(base, E).fit(model, th_train, eps_tol=1e-14, nb_max=10)
    forward = [emu.predict(t) for t in th_valid]
    backward = [emu.predict(t) for t in reversed(th_valid)][::-1]
    for a, b in zip(forward, backward):
        assert np.array_equal(a, b), f"{key}: prediction depends on query order"


def test_solver_energy_memo_is_transparent():
    """The per-energy memo added to DBMMSolver must change nothing.

    It caches the boundary rows and the regular Coulomb function on the mesh --
    both theta-independent -- so a warm solver and a cold one have to return
    bit-identical matrices. Asserted rather than assumed: the whole point of the
    memo is that it is invisible, and an invisible change is one nothing would
    otherwise report.
    """
    system = SYSTEMS["alpha_d"]
    _, model, theta_c, _ = system.preset_fn()
    E = float(system.E_demo)
    warm = solver_for(system, theta_c, energies=[E])
    warm.solve(E)                                   # fill the memo
    cold = solver_for(system, theta_c, energies=None)
    assert np.array_equal(warm._build_matrix(E), cold._build_matrix(E))
    assert np.array_equal(warm._build_rhs(E), cold._build_rhs(E))
    assert np.array_equal(warm.solve(E), cold.solve(E))


def test_physical_constants_have_one_definition():
    """core.dbmm, core.python_rmatrix and core.potentials each held their own
    copy, and pipeline.compute drew Figure 0's Coulomb potential with a fourth
    value. They are re-exported from core.constants now, and this asserts that
    they still are rather than having drifted apart again."""
    from core import constants, dbmm, potentials, python_rmatrix
    assert dbmm.E2 is constants.E2
    assert python_rmatrix.E2 is constants.E2
    assert dbmm.HBARC is constants.HBARC
    assert python_rmatrix.HBARC is constants.HBARC
    assert potentials.AMU is constants.AMU
