"""The computations, each writing one cached block.

Every block takes (system, settings): the physics it runs on, and the experiment
it runs. Ranks, predictor budgets and training sizes are in neither -- they are
read from `plan_calibration`, so every block of a run shares one configuration
and none of them can quietly choose its own.
"""
import os
import time
from concurrent.futures import ThreadPoolExecutor

from dataclasses import asdict

import numpy as np

from .settings import xsec_kind as _xsec

from core.dbmm import (DBMMSolver, _coulomb_sphere_arr, flux_factor_disabled)
from core.rbm import latin_hypercube

from . import cache
from .observables import (cross_section, eigenphase_branch, sample_theta,
                          solver_for)

# Seeds are fixed per role, not per call, so that the training set of one block
# is never accidentally reused as the validation set of another.
SEED_TRAIN = 11
SEED_VALID = 907
SEED_BOX = 4242

#: Threads for the parallel maps. Two cores left free so a timing measured in
#: the same process is not competing with a full fan-out. Capped at 12: past
#: that the threads contend on the shared Coulomb dict and the allocator, not
#: the arithmetic. Recorded in every block's metadata.
N_WORKERS = max(1, min(12, (os.cpu_count() or 2) - 2))

#: Energy nodes at which `excitation` and `interior` refit an emulator. It is
#: the one number that sets the price of a run: those two blocks pay
#: (configurations) x n_E x N_s full solves, and every other block pays n_E = 1.
N_ENERGY_NODES = 20


def _pmap(fn, items):
    """Thread-parallel map. numpy releases the GIL inside LAPACK, where nearly
    all the arithmetic time goes, so threads give real speedup without the
    pickling that processes would need for the model closures.

    Always call _warm_coulomb first: on a cold cache every worker would compute
    the same Coulomb values N_WORKERS times over.
    """
    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        return list(pool.map(fn, items))


def _warm_coulomb(system, energies, theta=None):
    """Fill the shared Coulomb cache serially, once, before any parallel work.

    The Coulomb functions depend on (l, eta, k, r) and never on the optical
    potential parameters, so warming at any single theta serves every draw.
    """
    from . import coulomb
    coulomb.warm()
    if theta is None:
        theta = system.preset_fn()[2]
    solver_for(system, theta, energies=np.atleast_1d(energies))
    return solver_for


def _write(system, settings, block, arrays, meta=None, **kw):
    """cache.write, plus the provenance every block owes the planner.

    `excitation` and `interior` (run.METHOD_BLOCKS) keep one record per emulator,
    since a merged block can hold RBM at 200 samples beside LROM at 400; the
    other blocks carry one record for the block itself, which `force="if-changed"`
    needs something to compare.
    """
    meta = dict(meta or {})
    prov = dict(settings=asdict(settings), system=system.fingerprint)
    # A third axis for the blocks that consumed a calibration: settings and
    # fingerprint do not see nb, K or N_s, which are derived and set the size of
    # everything the emulator blocks compute, so a moved calibration would be
    # invisible to force="if-changed". Only recorded when a calibration was used.
    if meta.get("calibrated"):
        prov["calibration"] = {k: meta[k] for k in
                               ("nb", "K_diag", "K_coup", "Ns", "eps_sol",
                                "eps_pot", "n_probe") if k in meta}
    meta["provenance"] = prov
    return cache.write(system.key, block, arrays, meta, **kw)


def _cal_record(cal):
    """The calibration as the planner compares it. Same fields as _write puts in
    a block-level provenance, so a method record and a block record are read the
    same way."""
    return dict(nb=cal.nb, K_diag=cal.K_diag, K_coup=cal.K_coup, Ns=cal.Ns,
                eps_sol=cal.eps_sol, eps_pot=cal.eps_pot, n_probe=cal.n_probe)


def _nanmedian(x):
    """Median ignoring NaN, silent when everything is NaN.

    An all-NaN slice is the expected inelastic error of an uncoupled system
    (n+40Ca has structurally-zero off-diagonal S), and numpy's warning about it
    is noise.
    """
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)
    if not finite.any():
        return float("nan")
    return float(np.median(x[finite]))


# --- F1: the benchmark suite -------------------------------------------

def block_sweep(system, settings, n_sweep=9, n_box=200, verbose=True):
    """Potential components at theta_c, and how far the observables travel.

    Two statements about the parameter box, deliberately both:
      - a rainbow as ONE parameter walks its range, everything else fixed;
      - the p5-p95 envelope over the FULL box, all p parameters varying at once.
    The first is readable, the second is what the emulator actually has to
    cover, and the gap between them is why training uses a Latin hypercube
    rather than a one-at-a-time scan.
    """
    channels, model, theta_c, bounds = system.preset_fn()
    E = system.E_grid
    Nc = len(channels)

    r = np.linspace(1e-3, system.R, 400)
    V_diag_c, V_coup_c = model(theta_c)
    V_channels = np.array([np.asarray(V_diag_c(r, ch), dtype=complex)
                           for ch in channels])
    V_couplings = np.full((Nc, Nc, r.size), np.nan, dtype=complex)
    # A pair the model declines to evaluate stays NaN (draws as a gap); the
    # failure is recorded so a missing coupling curve is attributable.
    coup_failures = []
    if V_coup_c is not None:
        for a in range(Nc):
            for b in range(Nc):
                if a == b:
                    continue
                try:
                    V_couplings[a, b] = np.asarray(
                        V_coup_c(r, channels[a], channels[b]), dtype=complex)
                except Exception as exc:               # noqa: BLE001 - recorded
                    coup_failures.append(f"({a},{b}): {type(exc).__name__}: {exc}")

    # The interaction term by term, from the affine structure the emulator
    # exploits, not from hard-coded term names (eleven presets, no shared
    # parameterisation). Where V is affine in theta_j the term it carries is
    # theta_j * dV/dtheta_j, and the terms sum to V. R_0 and a_0 enter
    # nonlinearly: detected (derivative not constant in the step) and left out.
    def _terms(evaluate, baseline):
        """[(label, curve)] for every parameter the interaction is affine in."""
        out = []
        for k in range(theta_c.size):
            h = 1e-4 * max(abs(theta_c[k]), 1.0)
            th = theta_c.copy(); th[k] += h
            d1 = (evaluate(th) - baseline) / h
            big = 0.25 * (bounds[k, 1] - bounds[k, 0]) or h
            th = theta_c.copy(); th[k] += big
            d2 = (evaluate(th) - baseline) / big
            scale = max(np.max(np.abs(d1)), 1e-30)
            if np.max(np.abs(d1 - d2)) > 1e-6 * scale:
                continue                       # nonlinear: R_0, a_0, ...
            term = theta_c[k] * d1
            if np.max(np.abs(term)) > 1e-12 * max(np.max(np.abs(baseline)), 1e-30):
                out.append((system.theta_labels[k], term))
        return out

    terms_diag = _terms(
        lambda th: np.asarray(model(th)[0](r, channels[0]), dtype=complex),
        V_channels[0])
    terms_coup = []
    if V_coup_c is not None and Nc > 1:
        base_c = np.asarray(V_coup_c(r, channels[0], channels[1]), dtype=complex)
        terms_coup = _terms(
            lambda th: np.asarray(
                model(th)[1](r, channels[0], channels[1]), dtype=complex),
            base_c)

    # Coulomb potential of a uniformly charged sphere: lives in the channel
    # definition, not the optical model, so a V_diag-only figure never showed it.
    # Drawn through the solver's own routine so it matches the interaction used.
    V_coulomb = np.zeros_like(r)
    z1z2 = channels[0].get("z1z2") or 0
    Rc = channels[0].get("R_coulomb")
    if z1z2 and Rc:
        V_coulomb = _coulomb_sphere_arr(r, z1z2, Rc).real

    j = settings.sweep_idx
    sweep_vals = np.linspace(bounds[j, 0], bounds[j, 1], n_sweep)

    _warm_coulomb(system, E)

    def run(theta):
        sv = solver_for(system, theta, energies=E)
        S = np.array([sv.solve(e) for e in E])
        delta, _ = eigenphase_branch(S)
        sigma = np.array([cross_section(sv, e, s, _xsec(system, settings))
                          for e, s in zip(E, S)])
        return delta, sigma

    thetas_sweep = []
    for v in sweep_vals:
        th = theta_c.copy()
        th[j] = v
        thetas_sweep.append(th)

    t0 = time.perf_counter()
    out = _pmap(run, thetas_sweep)
    delta = np.array([o[0] for o in out])
    sigma = np.array([o[1] for o in out])

    # The interior solution along the same sweep: sigma and delta are boundary
    # values of it, so a sweep of only those never shows the object the emulator
    # must reproduce. One extra solve per sweep value, at E_demo.
    def psi_at(theta):
        sv = solver_for(system, theta, energies=[system.E_demo])
        r_psi, chi, _ = sv.wavefunction(float(system.E_demo), total=True)
        return np.asarray(r_psi, dtype=float), np.asarray(chi)[0, :, 0]

    psi_out = _pmap(psi_at, thetas_sweep)
    r_psi = psi_out[0][0]
    psi_sweep = np.array([p[1] for p in psi_out])

    thetas_box = latin_hypercube(n_box, bounds, seed=SEED_BOX)
    sigma_box = np.array([o[1] for o in _pmap(run, thetas_box)])
    seconds = time.perf_counter() - t0

    with np.errstate(invalid="ignore", divide="ignore"):
        span = np.nanmax(sigma_box, axis=0) / np.nanmin(sigma_box, axis=0)

    return _write(system, settings, "sweep", dict(
        r=r, E=E, V_channels=V_channels, V_couplings=V_couplings,
        sweep_vals=sweep_vals, delta=delta, sigma=sigma, sigma_box=sigma_box,
        theta_c=theta_c, bounds=bounds,
        r_psi=r_psi, psi_sweep=psi_sweep,
        V_coulomb=V_coulomb,
        terms_diag=np.array([t for _, t in terms_diag]) if terms_diag
                   else np.zeros((0, r.size), dtype=complex),
        terms_coup=np.array([t for _, t in terms_coup]) if terms_coup
                   else np.zeros((0, r.size), dtype=complex),
    ), dict(
        terms_diag_labels=[n for n, _ in terms_diag],
        terms_coup_labels=[n for n, _ in terms_coup],
        n_channels=Nc, N=system.N, R=system.R, p=theta_c.size,
        n_sweep=n_sweep, n_box=n_box, seed_box=SEED_BOX,
        sweep_label=settings.sweep_label, sweep_idx=j,
        xsec_kind=_xsec(system, settings), thresholds=system.thresholds,
        threshold_labels=system.threshold_labels,
        span_max=float(np.nanmax(span)), seconds=seconds,
        coupling_failures=coup_failures[:8],
        # Draws whose S-matrix was rejected as unphysical, out of n_box * n_E.
        # A p5-p95 envelope silently computed over fewer draws than it claims is
        # a narrower envelope, so the count belongs beside it.
        n_unphysical_box=int(np.sum(~np.isfinite(sigma_box))),
        n_box_points=int(sigma_box.size),
    ))


# --- F2: solver agreement --------------------------------------------

def block_validation(system, settings, n_theta=40, n_E=60, verbose=True):
    """DBMM against the R-matrix reference on sigma, on S, and on chi(r).

    The three quantities are nested: sigma is a scalar, S is the full matrix,
    chi(r) is the interior solution. Agreement on sigma alone would be
    compatible with compensating errors, which is why all three are cached.
    """
    from core import python_rmatrix as rmat

    channels, model, theta_c, bounds = system.preset_fn()
    lo, hi, _ = system.E_range
    E = np.linspace(lo, hi, n_E)
    thetas = sample_theta(system, n_theta, seed=SEED_VALID)
    Nc, N = len(channels), system.N

    _warm_coulomb(system, E)

    # The Lagrange mesh and the boundary values depend on nothing but (N, R),
    # so they are built once for the whole block rather than per solve.
    mesh, _ = rmat.compute_lagrange_mesh(N, system.R)
    lag_b = rmat.eval_lag_at_boundary(mesh, system.R)
    B_of_E = {float(e): rmat.make_boundary_matrix(mesh, system.R, channels, e)
              for e in E}

    def rmatrix_grid(theta):
        """S at every energy for one theta, building the Hamiltonian once.

        The R-matrix Hamiltonian depends on theta but not on E (only the Bloch
        boundary matrix does), so rebuilding it inside the energy loop would
        cost a factor n_E for nothing.
        """
        V_diag, V_coup = model(theta)
        if V_coup is None:
            def V_coup(r, ca, cb):
                return np.zeros_like(r, dtype=complex)
        H = rmat.compute_CC_ham(mesh, system.R, channels, V_diag, V_coup)
        out = np.empty((n_E, Nc, Nc), dtype=complex)
        for i, e in enumerate(E):
            S, _ = rmat.compute_S_matrix(H + B_of_E[float(e)], lag_b, channels,
                                         e, system.R, mesh, compute_wfs=False)
            out[i] = S
        return out

    # Physics only, no timing: costs are timed serially in `cat` and `scaling`
    # (cache.MACHINE_DEPENDENT). A wall clock measured inside a 12-thread map
    # carries the contention of whatever else is running.
    def run(theta):
        sv = solver_for(system, theta, energies=E)
        S_d = np.array([sv.solve(e) for e in E])
        S_r = rmatrix_grid(theta)

        sig_d = np.array([cross_section(sv, e, s, _xsec(system, settings))
                          for e, s in zip(E, S_d)])
        sig_r = np.array([cross_section(sv, e, s, _xsec(system, settings))
                          for e, s in zip(E, S_r)])

        # interior wave function at the demo energy, in a common gauge
        e0 = float(E[np.argmin(np.abs(E - system.E_demo))])
        _, chi, _ = sv.wavefunction(e0, r_eval=sv.r, total=True)
        psi_d = chi[0, :, 0]
        psi_r = _rmatrix_interior(rmat, system, channels, model, theta, e0, psi_d)
        return S_d, S_r, sig_d, sig_r, psi_d, psi_r

    t0 = time.perf_counter()
    out = _pmap(run, thetas)
    seconds = time.perf_counter() - t0

    pack = lambda i: np.array([o[i] for o in out])
    return _write(system, settings, "validation", dict(
        E=E, thetas=thetas,
        S_dbmm=pack(0), S_rmat=pack(1),
        sigma_dbmm=pack(2), sigma_rmat=pack(3),
        psi_dbmm=pack(4), psi_rmat=pack(5),
    ), dict(
        n_theta=n_theta, n_E=n_E, seed=SEED_VALID,
        n_channels=Nc, N=N, R=system.R, p=theta_c.size,
        xsec_kind=_xsec(system, settings), thresholds=system.thresholds,
        threshold_labels=system.threshold_labels,
        E_wf=float(system.E_demo), seconds=seconds,
        note=("physics only: this block carries no timing. Costs come from "
              "`cat` and `scaling` -- cache.MACHINE_DEPENDENT."),
    ))


def _rmatrix_interior(rmat, system, channels, model, theta, E, psi_ref):
    """R-matrix interior solution, gauge-matched onto the DBMM normalisation.

    The two codes fix the overall constant differently, so an uncalibrated
    comparison measures that convention rather than the physics. The scale is
    obtained by least squares against the DBMM solution, which leaves the shape
    entirely free to disagree.
    """
    N, R = system.N, system.R
    V_diag, V_coup = model(theta)
    if V_coup is None:
        def V_coup(r, ca, cb):
            return np.zeros_like(r, dtype=complex)
    mesh, _ = rmat.compute_lagrange_mesh(N, R)
    H = rmat.compute_CC_ham(mesh, R, channels, V_diag, V_coup)
    lag_b = rmat.eval_lag_at_boundary(mesh, R)
    B = rmat.make_boundary_matrix(mesh, R, channels, E)
    Htot = H + B
    shift = np.repeat([E - ch['threshold'] for ch in channels], N)
    Cinv = np.linalg.inv(Htot - np.diag(shift))
    cand = (Cinv[0:N, 0:N] @ lag_b)
    from core.dbmm import _gl_mesh
    lam = _gl_mesh(N, R)[1]
    cand = cand / np.sqrt(R * lam)
    scale = np.vdot(cand, psi_ref) / np.vdot(cand, cand)
    return scale * cand


# --- F3: the invariants ---------------------------------------------

def block_invariants(system, settings, verbose=True):
    """Symmetry and unitarity against mesh refinement, with and without the
    velocity factor of core.dbmm.channel_velocities.

    Supports the claim that a discretisation error falls with N and an error
    that does not is in the formulation. Both curves come from the same code,
    switched by core.dbmm.flux_factor_disabled.
    """
    from .metrics import symmetry_violation, unitarity_violation

    channels, model, theta_c, bounds = system.preset_fn()
    Ns = np.array(settings.N_scan)

    # An energy at which every channel is open, so the unitarity test is not
    # partly vacuous, and one at the demo energy for the symmetry test.
    lo, hi, _ = system.E_range
    thr = system.thresholds
    E_open = float(min(hi * 0.97, (max(thr) * 1.10) if thr else hi * 0.97))
    E_sym = float(system.E_demo)

    def real_potential(theta):
        """model(theta) with the absorption removed exactly, by taking the real
        part of every potential term.

        Not by zeroing a parameter: some systems carry a surface absorption
        W_d0 as well as a volume one, so zeroing one index leaves the potential
        absorptive and the unitarity panel then measures real absorption
        (~0.5-0.8) rather than a formulation error.
        """
        V_diag, V_coup = model(theta)

        def rd(r, ch):
            return np.asarray(V_diag(r, ch), dtype=complex).real.astype(complex)

        rc = None
        if V_coup is not None:
            def rc(r, ca, cb):
                return np.asarray(V_coup(r, ca, cb), dtype=complex).real.astype(complex)
        return rd, rc

    def measure(N, theta, E, disabled, real=False):
        V_diag, V_coup = real_potential(theta) if real else model(theta)
        sv = DBMMSolver(N, system.R, channels, V_diag, V_coup)
        if disabled:
            with flux_factor_disabled():
                S = sv.solve(E)
        else:
            S = sv.solve(E)
        return S

    sym_on, sym_off, uni_on, uni_off = [], [], [], []
    for N in Ns:
        sym_on.append(symmetry_violation(measure(N, theta_c, E_sym, False)))
        sym_off.append(symmetry_violation(measure(N, theta_c, E_sym, True)))
        uni_on.append(unitarity_violation(
            measure(N, theta_c, E_open, False, real=True)))
        uni_off.append(unitarity_violation(
            measure(N, theta_c, E_open, True, real=True)))

    # Panel C: the ratio the uncorrected matrix exhibits, against the kinematic
    # ratio that names the missing factor. With the uncorrected extraction
    # S_ab = U_ab sqrt(v_b / v_a) and the symmetry |U_ab| = |U_ba|,
    #     |S_ba| / |S_ab| = v_a / v_b = k_a / k_b   (shared reduced mass).
    # The scan starts well clear of the highest threshold, where k_b -> 0 makes
    # the ratio diverge for unrelated kinematic reasons.
    from core.dbmm import _channel_kinematics
    E_start = max(lo, (min(thr) * 1.35) if thr else lo)
    E_ratio = np.linspace(E_start, hi * 0.99, 40)
    sv = solver_for(system, theta_c, energies=E_ratio)
    ratio_S, ratio_k = [], []
    for e in E_ratio:
        op = [a for a, ch in enumerate(channels) if e > ch['threshold']]
        if len(op) < 2:
            ratio_S.append(np.nan)
            ratio_k.append(np.nan)
            continue
        with flux_factor_disabled():
            S = sv.solve(e)
        a, b = op[0], op[1]
        den = abs(S[a, b])
        ratio_S.append(abs(S[b, a]) / den if den > 0 else np.nan)
        ka = _channel_kinematics(channels[a], e)[0].real
        kb = _channel_kinematics(channels[b], e)[0].real
        ratio_k.append(ka / kb if kb > 0 else np.nan)

    return _write(system, settings, "invariants", dict(
        N_scan=Ns,
        sym_on=np.array(sym_on), sym_off=np.array(sym_off),
        uni_on=np.array(uni_on), uni_off=np.array(uni_off),
        E_ratio=E_ratio, ratio_S=np.array(ratio_S), ratio_k=np.array(ratio_k),
    ), dict(
        E_sym=E_sym, E_open=E_open, n_channels=len(channels),
        thresholds=thr, theta_c=list(map(float, theta_c)),
        note=("uni_* are evaluated on the open sub-block: a closed channel "
              "contributes an exact floor of 1.0 to the full-matrix norm."),
    ))


# --- F4, F5: the manifold, and the quantity the error is measured on ---

def block_manifold(system, settings, n_valid=40, verbose=True):
    """POD spectrum, the truncation criterion, and the error actually achieved.

    One block feeds F4 (is the manifold thin, does the criterion predict the
    error) and F5 (is the error measured on the right quantity), both statements
    about one scan over rank. Validation draws use a different seed from
    training draws, so every error is out-of-sample.
    """
    from core.rbm import RBMEmulator
    from . import metrics
    from .plan_calibration import as_meta, calibration_for

    channels, model, theta_c, bounds = system.preset_fn()
    E = float(system.E_demo)
    Nc = len(channels)

    cal = calibration_for(system.key)
    n_train = cal.Ns

    th_train = latin_hypercube(n_train, bounds, seed=SEED_TRAIN)
    th_valid = latin_hypercube(n_valid, bounds, seed=SEED_VALID)

    base = solver_for(system, theta_c, energies=[E])
    S_ref = np.array([solver_for(system, t, energies=[E]).solve(E)
                      for t in th_valid])
    sig_ref = np.array([cross_section(base, E, S, _xsec(system, settings))
                        for S in S_ref])
    mask = metrics.coupling_mask(S_ref)

    t0 = time.perf_counter()
    emu = RBMEmulator(base, E)
    emu.fit(model, th_train, eps_tol=1e-14)
    t_offline = time.perf_counter() - t0

    sv = emu.singular_values
    energy = np.cumsum(sv ** 2) / np.sum(sv ** 2)
    discarded = 1.0 - energy                      # r(k), discarded energy

    X_full, K_full = emu.X_r, emu.K
    nb_scan = np.array([n for n in cal.ladder() if n <= X_full.shape[1]])

    err_g, err_e, err_i, err_s = [], [], [], []
    for nb in nb_scan:
        emu.X_r = X_full[:, :nb]
        emu.K_r = emu.X_r.conj().T @ K_full @ emu.X_r
        S_emu = np.array([emu.predict(t) for t in th_valid])
        sig_emu = np.array([cross_section(base, E, S, _xsec(system, settings))
                            for S in S_emu])
        err_g.append(metrics.err_global(S_emu, S_ref))
        err_e.append(metrics.err_elastic(S_emu, S_ref))
        err_i.append(metrics.err_inelastic(S_emu, S_ref, mask=mask))
        err_s.append(metrics.err_observable(sig_emu, sig_ref))
    emu.X_r = X_full
    emu.K_r = X_full.conj().T @ K_full @ X_full

    # Panel C of F4: does the rank saturate, and is N_s enough?
    ns_scan = np.unique(np.clip(
        (np.array([0.1, 0.2, 0.35, 0.5, 0.75, 1.0]) * n_train).astype(int),
        10, n_train))
    nb_of_ns, err_of_ns, err_sigma_of_ns = [], [], []
    # The operated rank, read from the calibration (max over the energy window),
    # not recomputed here from one energy and one sample. The block's own
    # reading is kept below as a metadata cross-check, never as the source.
    nb_probe = min(cal.nb, int(X_full.shape[1]))
    nb_at_demo = int(np.argmax(discarded < cal.eps_sol) + 1) \
        if (discarded < cal.eps_sol).any() else int(X_full.shape[1])
    for ns in ns_scan:
        e2 = RBMEmulator(base, E)
        # the same eps_sol the operated rank is read at, so the criterion panel
        # plots the rank everything else uses.
        e2.fit(model, th_train[:ns], eps_tol=cal.eps_sol)
        nb_of_ns.append(e2.nb)
        e3 = RBMEmulator(base, E)
        e3.fit(model, th_train[:ns], eps_tol=1e-14, nb_max=nb_probe)
        S_emu = np.array([e3.predict(t) for t in th_valid])
        err_of_ns.append(_nanmedian((
            metrics.err_report(S_emu, S_ref, mask=mask))))
        # Same question on the cross section. err_report is the inelastic error
        # where coupling exists, which on alpha+12C is relative to matrix
        # elements 3e-03 of the diagonal and reads hundreds of per cent; sigma
        # is what the emulator is for, measured here at the same ranks.
        sig_ns = np.array([cross_section(base, E, S, _xsec(system, settings))
                           for S in S_emu])
        err_sigma_of_ns.append(_nanmedian(
            metrics.err_observable(sig_ns, sig_ref)))

    return _write(system, settings, "manifold", dict(
        singular_values=sv, energy=energy, discarded=discarded,
        nb_scan=nb_scan,
        err_global=np.array(err_g), err_elastic=np.array(err_e),
        err_inelastic=np.array(err_i), err_sigma=np.array(err_s),
        ns_scan=ns_scan, nb_of_ns=np.array(nb_of_ns),
        err_of_ns=np.array(err_of_ns),
        err_sigma_of_ns=np.array(err_sigma_of_ns), th_valid=th_valid,
        # The magnitudes the errors above are relative to, without which the
        # inelastic curve is uninterpretable (410% relative on elements 2e-03 of
        # the diagonal). Stored raw, per validation draw.
        abs_S_ref=np.abs(S_ref), coupling_mask=mask,
    ), dict(
        E=E, n_train=n_train, n_valid=n_valid,
        seed_train=SEED_TRAIN, seed_valid=SEED_VALID,
        n_channels=Nc, N=system.N, p=int(theta_c.size), dim=Nc * system.N,
        # nb_probe: the operated rank. nb_at_demo: this block's reading of the
        # same criterion at E_demo alone, kept so the gap with the calibration's
        # window maximum stays visible.
        nb_full=int(X_full.shape[1]), nb_probe=nb_probe, nb_at_demo=nb_at_demo,
        **as_meta(cal),
        has_coupling=bool(mask.any()), t_offline=t_offline,
        xsec_kind=_xsec(system, settings),
    ))


# --- F9, F10: the interior solution and the phase shift vs the emulators ---

def block_interior(system, settings, methods=("rbm", "lrom"), n_E=N_ENERGY_NODES,
                   n_resid=30, verbose=True):
    """psi(r) at one energy and delta(E) over the grid, solver against emulators.

    The two observables the cross section hides: an emulator can reproduce the
    scalar sigma while getting chi(r) wrong, because errors that cancel in the
    flux integral do not cancel pointwise. Everything is at one validation-seed
    parameter, never seen in training (at theta_c the LROM is exact by
    construction and the RBM nearly so). delta(E) needs one emulator per energy,
    n_E fits per method, which is why n_E is modest and this is its own block.
    """
    from core.rbm import RBMEmulator
    from core.lrom_dbmm import LROMEmulator
    from .plan_calibration import as_meta, calibration_for

    channels, model, theta_c, bounds = system.preset_fn()
    Nc = len(channels)
    E_demo = float(system.E_demo)
    lo, hi, _ = system.E_range
    E_grid = np.linspace(lo, hi, n_E)

    cal = calibration_for(system.key)
    nb = cal.nb
    K_diag, K_coup = cal.budgets(1)[0]

    # n_resid validation-seed draws, centre excluded. The first is plotted as a
    # curve; the rest are what the p5-p95 band is computed over.
    th_valid = sample_theta(system, n_resid, seed=SEED_VALID, include_center=False)
    theta_show = th_valid[0]
    n_train = cal.Ns
    th_train = sample_theta(system, n_train, seed=SEED_TRAIN)

    _warm_coulomb(system, np.concatenate([E_grid, [E_demo]]))

    # --- psi(r) at E_demo ---
    ref = solver_for(system, theta_show, energies=[E_demo])
    r_eval, chi_ref, incident = ref.wavefunction(E_demo, total=True)
    r_eval = np.asarray(r_eval, dtype=float)

    def first_incident(chi):
        """One column of the interior solution: entrance channel, first
        incident wave. The full object is (Nc, Np, n_inc) and a figure that
        draws all of it draws nothing legible."""
        return np.asarray(chi, dtype=complex)[0, :, 0]

    psi = {"ref": first_incident(chi_ref)}

    def psi_ref_at(theta):
        return first_incident(solver_for(system, theta, energies=[E_demo])
                              .wavefunction(E_demo, r_eval=r_eval, total=True)[1])

    psi_all = {"ref": np.array(_pmap(psi_ref_at, th_valid))}
    base = solver_for(system, theta_c, energies=[E_demo])
    t_fit = {}
    for name in methods:
        t0 = time.perf_counter()
        if name == "rbm":
            emu = RBMEmulator(base, E_demo)
            emu.fit(model, th_train, eps_tol=1e-14, nb_max=nb)
        elif name == "lrom":
            emu = LROMEmulator(base, E_demo)
            emu.fit(model, th_train, theta_c, K_diag=K_diag, K_coup=K_coup,
                    eps_tol=1e-14, nb_max=nb)
        else:
            continue
        t_fit[name] = time.perf_counter() - t0
        psi[name] = first_incident(emu.wavefunction(theta_show, r_eval=r_eval,
                                                   total=True)[1])
        psi_all[name] = np.array([first_incident(
            emu.wavefunction(t, r_eval=r_eval, total=True)[1]) for t in th_valid])

    # delta(E) over the grid. One solver per theta, not one per (theta, energy):
    # the solver build (K0, diagonals, Coulomb warm-up) does not depend on E.
    def ref_grid(theta):
        sv = solver_for(system, theta, energies=E_grid)
        return np.array([sv.solve(e) for e in E_grid])

    S_ref_all = np.array(_pmap(ref_grid, th_valid))
    delta_all = {"ref": np.array([eigenphase_branch(S)[0] for S in S_ref_all])}
    delta = {"ref": delta_all["ref"][0]}

    # Independent per-energy fits, run concurrently. The per-node base solver is
    # built inside the worker because RBMEmulator keeps a reference to it.
    def fit_node(job):
        name, i, e = job
        b = solver_for(system, theta_c, energies=[e])
        if name == "rbm":
            emu = RBMEmulator(b, e)
            emu.fit(model, th_train, eps_tol=1e-14, nb_max=nb)
        else:
            emu = LROMEmulator(b, e)
            emu.fit(model, th_train, theta_c, K_diag=K_diag, K_coup=K_coup,
                    eps_tol=1e-14, nb_max=nb)
        return name, i, np.array([emu.predict(t) for t in th_valid])

    node_jobs = [(name, i, e) for name in methods if name in ("rbm", "lrom")
                 for i, e in enumerate(E_grid)]
    S_emu = {name: np.empty_like(S_ref_all) for name in methods
             if name in ("rbm", "lrom")}
    t_nodes = time.perf_counter()
    for name, i, block in _pmap(fit_node, node_jobs):
        S_emu[name][:, i] = block
    t_nodes = time.perf_counter() - t_nodes

    for name in S_emu:
        # Same branch tracking as the reference: eigenphase_branch is a property
        # of the grid, so both grids must be built the same way.
        delta_all[name] = np.array([eigenphase_branch(S)[0] for S in S_emu[name]])
        delta[name] = delta_all[name][0]

    arrays = dict(r=r_eval, E=E_grid, theta_show=theta_show, th_valid=th_valid,
                  psi_ref=psi["ref"], delta_ref=delta["ref"])
    for name in methods:
        if name in psi:
            arrays[f"psi_{name}"] = psi[name]
            arrays[f"delta_{name}"] = delta[name]
            # The residual over every draw, which is what the band summarises.
            scale = np.max(np.abs(psi_all["ref"]), axis=1, keepdims=True)
            arrays[f"resid_psi_{name}"] = np.abs(
                psi_all[name] - psi_all["ref"]) / np.maximum(scale, 1e-300)
            arrays[f"resid_delta_{name}"] = np.abs(
                delta_all[name] - delta_all["ref"])

    method_meta = {m: dict(settings=asdict(settings), system=system.fingerprint,
                           calibration=_cal_record(cal),
                           seconds=t_fit.get(m, float("nan")))
                   for m in methods if m in psi}
    return _write(system, settings, "interior", arrays, methods=method_meta,
                  merge=True, meta=dict(
        E=E_demo, n_train=n_train, n_channels=Nc,
        n_resid=n_resid,
        seed_train=SEED_TRAIN, seed_valid=SEED_VALID,
        **as_meta(cal, n_E=n_E),
        thresholds=system.thresholds,
        threshold_labels=system.threshold_labels,
        # Wall clock of the whole (concurrent) delta(E) stage, not a per-method
        # cost. `seconds` in each method's record is the serial psi fit at E_demo.
        t_nodes_wall=t_nodes, n_workers=N_WORKERS,
        note="theta_show is a validation draw: no method has seen it",
    ))


# --- CAT: the cost-accuracy cloud over the parameter box --------------

#: Minimum repetitions every timed call gets, on top of the 5 ms floor. The
#: floor alone buys ~32 repetitions of a 0.16 ms prediction but only one of a
#: 75 ms solve, so it estimates the two paired quantities with different care.
#:
#: 7, not more, from measurement: the spread of `min over R` across five re-runs
#: does not converge as R grows (4-140% at R=7, still 10-35% at R=60). That
#: variance is the processor's clock state changing over a longer window, not
#: sampling noise. Hence the paired timings below -- an absolute time here is
#: reproducible only to tens of per cent, so a ratio must be measured with its
#: numerator and denominator next to each other.
MIN_REPEATS = 7


def _timed(call, floor=5e-3, warmup=True, min_reps=MIN_REPEATS):
    """Seconds for one `call()`, measured for something this short.

    Repeat until the accumulated time clears `floor` AND at least `min_reps`
    samples are taken: the floor protects a cheap call from the clock's
    resolution, `min_reps` stops an expensive one being estimated from a single
    sample. Returns the minimum, not the median -- the computation is
    deterministic, so its true cost is the smallest time observed and every
    excess is OS interference. At ~200 us a prediction is close to Python's own
    call overhead, so a ratio off this is end to end, not a flop count.
    """
    if warmup:
        call()                                  # first touch, not measured
    best, total, n = float("inf"), 0.0, 0
    while total < floor or n < min_reps:
        t0 = time.perf_counter()
        call()
        dt = time.perf_counter() - t0
        best = min(best, dt)
        total += dt
        n += 1
        if n > 2000:
            break
    return best


# --- F11: online cost against mesh size, where the LROM's premise is tested ---

def block_scaling(system, settings, n_valid=30, verbose=True):
    """Online cost of both emulators against the mesh size N, physics fixed.

    The RBM online stage evaluates and projects the potential at all N-1
    interior points, so its cost carries a term linear in N; the LROM reads K
    offline-chosen points, K independent of N. That advantage is asymptotic, and
    on this suite (N 40-120) the term is not yet what costs: at the calibrated
    configuration the LROM is 1.4-3.6x the slower on every system. Same two
    emulators, same rank and budget, rebuilt at each N, with the solver's own
    cost measured alongside so any crossing can be read, not extrapolated.
    Timed through `_timed`, the same path as `cat`.
    """
    from core.rbm import RBMEmulator
    from core.lrom_dbmm import LROMEmulator
    from .plan_calibration import as_meta, calibration_for

    channels, model, theta_c, bounds = system.preset_fn()
    E = float(system.E_demo)
    cal = calibration_for(system.key)
    K_diag, K_coup = cal.budgets(1)[0]

    th_train = latin_hypercube(cal.Ns, bounds, seed=SEED_TRAIN)
    th_valid = sample_theta(system, n_valid, seed=SEED_VALID, include_center=False)

    N_scan = [int(n) for n in settings.N_scan]
    rows = []
    for N in N_scan:
        # Coulomb functions precomputed per solver, as everywhere: in the
        # critical path they would make this a measurement of the special
        # -function backend against N, not of the two methods.
        base = solver_for(system, theta_c, energies=[E], N=N)

        rbm = RBMEmulator(base, E)
        rbm.fit(model, th_train, eps_tol=1e-14, nb_max=cal.nb)
        lrom = LROMEmulator(base, E)
        lrom.fit(model, th_train, theta_c, K_diag=K_diag, K_coup=K_coup,
                 eps_tol=1e-14, nb_max=cal.nb)

        th = th_valid[0]
        t_rbm = _timed(lambda: rbm.predict(th))
        t_lrom = _timed(lambda: lrom.predict(th))
        # The three solver baselines, at this N, through exactly block_cat's
        # calls: t_asm (matrix already assembled), t_V (K kept, V re-evaluated
        # and reassembled -- the honest baseline), t_full (rebuilt from theta).
        sv = solver_for(system, theta_c, energies=[E], N=N)
        t_asm = _timed(lambda: sv.solve(E))
        t_V = _timed(lambda: (sv.set_potential(*model(th)), sv.solve(E)))
        t_full = _timed(lambda: solver_for(system, th, energies=[E], N=N).solve(E))

        rows.append((N, t_rbm, t_lrom, t_V, t_full, t_asm, rbm.nb, lrom.nb))
        if verbose:
            print(f"    N={N:4d}  RBM {t_rbm*1e3:7.3f} ms   LROM {t_lrom*1e3:7.3f} ms"
                  f"   ratio {t_lrom/t_rbm:5.2f}   nb {rbm.nb}/{lrom.nb}")

    a = np.array(rows, dtype=float)
    return _write(system, settings, "scaling", dict(
        N=a[:, 0], t_rbm=a[:, 1], t_lrom=a[:, 2],
        t_dbmm_V=a[:, 3], t_dbmm_full=a[:, 4], t_dbmm_asm=a[:, 5],
        nb_rbm=a[:, 6], nb_lrom=a[:, 7],
    ), dict(
        E=E, n_valid=n_valid, seed_train=SEED_TRAIN, seed_valid=SEED_VALID,
        n_channels=len(channels), N_scan=N_scan, min_repeats=MIN_REPEATS,
        **as_meta(cal),
        note=("online cost against mesh size at fixed physics; the rank and the "
              "predictor budget are the calibrated ones and do not vary with N. "
              "t_dbmm_asm, t_dbmm_V and t_dbmm_full are the same three calls "
              "block_cat times, so the two blocks are comparable."),
    ))


def block_cat(system, settings, n_valid=2000, n_time=25, n_rbm=2, n_lrom=3,
              verbose=True):
    """The cost-accuracy cloud: one point per (configuration, validation draw).

    A median error and cost per configuration hides what the figure is for: an
    emulator whose error spans two decades across the box is not the same object
    as one that does not. Here every draw keeps its own time and error, so the
    cloud has width on both axes. Few configurations: the two largest RBM ranks
    and three largest LROM budgets of the calibrated ladder.
    """
    from core.rbm import RBMEmulator
    from core.lrom_dbmm import LROMEmulator
    from .plan_calibration import as_meta, calibration_for

    channels, model, theta_c, bounds = system.preset_fn()
    E = float(system.E_demo)

    cal = calibration_for(system.key)
    n_train = cal.Ns
    combos = ([dict(method="rbm", nb=nb) for nb in cal.ladder()[-n_rbm:]]
              + [dict(method="lrom", nb=nb, K_diag=Kd, K_coup=Kc)
                 for nb in cal.ladder()[-2:]
                 for (Kd, Kc) in cal.budgets(2)][-n_lrom:])

    th_valid = sample_theta(system, n_valid, seed=SEED_VALID, include_center=False)
    th_train = latin_hypercube(n_train, bounds, seed=SEED_TRAIN)
    _warm_coulomb(system, [E])

    # The reference, and four solver-side costs measured the same way:
    #   t_dbmm_asm   matrix already assembled for this theta -- a floor, not a
    #                baseline (a campaign changes theta at every evaluation).
    #   t_dbmm_V     K(E) kept, V(theta) re-evaluated and reassembled -- the
    #                affine split used by the solver itself; the honest baseline.
    #   t_dbmm_full  solver rebuilt from theta, kinetic block included.
    #   t_rmat       a straightforward R-matrix implementation, an UPPER BOUND on
    #                the method's cost: core/python_rmatrix.py rebuilds its
    #                theta-independent kinetic matrix in an O(N^2) Python loop
    #                and inverts the full matrix where Nc rhs would do (~90% of
    #                the n+40Ca cost). Ratios against it are upper bounds, not
    #                the gain.
    # The R-matrix mesh, boundary matrix and Lagrange values depend on E and
    # geometry but not theta, so they are hoisted out of the timing.
    from core import python_rmatrix as rmat
    rm_mesh, _ = rmat.compute_lagrange_mesh(system.N, system.R)
    rm_lag_b = rmat.eval_lag_at_boundary(rm_mesh, system.R)
    rm_B = rmat.make_boundary_matrix(rm_mesh, system.R, channels, E)

    def _rmatrix_once(th):
        V_diag, V_coup = model(th)
        if V_coup is None:
            def V_coup(r, ca, cb):
                return np.zeros_like(r, dtype=complex)
        H = rmat.compute_CC_ham(rm_mesh, system.R, channels, V_diag, V_coup)
        return rmat.compute_S_matrix(H + rm_B, rm_lag_b, channels, E,
                                     system.R, rm_mesh, compute_wfs=False)[0]

    S_ref = np.array([solver_for(system, th, energies=[E]).solve(E)
                      for th in th_valid])

    # The four solver-side costs, timed on n_time draws with MIN_REPEATS
    # repetitions each, not on all n_valid: none of them depends on theta (each
    # does fixed-size work), and the spread across draws equals the spread over
    # one theta repeated -- measurement scatter, not physics. So repetitions buy
    # more than draws here. These feed the reference rules on the figures; the
    # speedups come from the paired baseline in the configuration loop below.
    #
    # ERROR is still measured on all n_valid draws -- it varies with theta by up
    # to two decades, which the figure exists to show. n_valid = 2000 because
    # the p95 separates the two emulators, and a p95 from 150 draws moves by a
    # factor of two between seeds; the extra cost is ~0.07 s per draw.
    n_time = min(n_time, len(th_valid))
    th_time = th_valid[:n_time]
    sv_reuse = solver_for(system, theta_c, energies=[E])
    t_asm, t_V, t_full, t_rmat = [], [], [], []
    for th in th_time:
        sv = solver_for(system, th, energies=[E])
        t_asm.append(_timed(lambda sv=sv: sv.solve(E)))
        t_V.append(_timed(lambda th=th: (sv_reuse.set_potential(*model(th)),
                                         sv_reuse.solve(E))))
        t_full.append(_timed(lambda th=th: solver_for(system, th,
                                                     energies=[E]).solve(E)))
        t_rmat.append(_timed(lambda th=th: _rmatrix_once(th)))
    t_asm, t_V = np.array(t_asm), np.array(t_V)
    t_full, t_rmat = np.array(t_full), np.array(t_rmat)
    t_dbmm = t_V   # the baseline speedups are quoted against

    base = solver_for(system, theta_c, energies=[E])
    rows = []
    for c in combos:
        t0 = time.perf_counter()
        if c["method"] == "rbm":
            emu = RBMEmulator(base, E)
            emu.fit(model, th_train, eps_tol=1e-14, nb_max=c["nb"])
        else:
            emu = LROMEmulator(base, E)
            emu.fit(model, th_train, theta_c, K_diag=c["K_diag"],
                    K_coup=c["K_coup"], eps_tol=1e-14, nb_max=c["nb"])
        t_off = time.perf_counter() - t0
        ratio = float(getattr(emu, "overfit_ratio", np.inf) or np.inf)

        for i, th in enumerate(th_valid):
            t = _timed(lambda th=th: emu.predict(th))
            # The paired baseline: same theta, same protocol, measured
            # microseconds after the prediction it is divided by, so a common
            # clock drift divides out (absolute times here reproduce only to
            # tens of per cent; see MIN_REPEATS). Only the first n_time draws --
            # the baseline is the expensive half and does not depend on theta.
            t_pair = (_timed(lambda th=th: (sv_reuse.set_potential(*model(th)),
                                            sv_reuse.solve(E)))
                      if i < n_time else np.nan)
            S = emu.predict(th)
            with np.errstate(divide="ignore", invalid="ignore"):
                e00 = abs(S[0, 0] - S_ref[i, 0, 0]) / abs(S_ref[i, 0, 0])
                eab = (abs(S[0, 1] - S_ref[i, 0, 1]) / abs(S_ref[i, 0, 1])
                       if S.shape[0] > 1 and abs(S_ref[i, 0, 1]) > 0 else np.nan)
            rows.append((c["method"], c["nb"], c.get("K_diag", 0),
                         c.get("K_coup", 0), emu.nb, t, t_off, ratio,
                         float(e00), float(eab), t_pair))

    arr = np.array([r[4:] for r in rows], dtype=float)
    return _write(system, settings, "cat", dict(
        nb_kept=arr[:, 0], t_online=arr[:, 1], t_offline=arr[:, 2],
        overfit_ratio=arr[:, 3], err_S00=arr[:, 4], err_Sab=arr[:, 5],
        # The affine-split solver, re-timed beside every prediction. Same shape
        # as t_online, NaN past n_time, so a speedup is an elementwise ratio.
        t_dbmm_paired=arr[:, 6],
        method=np.array([r[0] for r in rows]),
        nb=np.array([r[1] for r in rows], dtype=float),
        K_diag=np.array([r[2] for r in rows], dtype=float),
        K_coup=np.array([r[3] for r in rows], dtype=float),
        t_dbmm=t_dbmm, th_valid=th_valid,
        t_dbmm_asm=t_asm, t_dbmm_V=t_V, t_dbmm_full=t_full, t_rmat=t_rmat,
    ), dict(
        E=E, n_valid=n_valid, n_time=n_time, n_train=n_train,
        seed_valid=SEED_VALID, min_repeats=MIN_REPEATS,
        seed_train=SEED_TRAIN, **as_meta(cal),
        n_channels=len(channels), N=system.N,
        t_dbmm_median=float(np.median(t_dbmm)),
        t_dbmm_asm_median=float(np.median(t_asm)),
        t_dbmm_V_median=float(np.median(t_V)),
        t_dbmm_full_median=float(np.median(t_full)),
        t_rmat_median=float(np.median(t_rmat)),
        # Spread of the affine-split baseline re-measured len(combos) times,
        # minutes apart: the reproducibility of an absolute timing here, and the
        # error bar every unpaired ratio carries.
        t_dbmm_paired_median=float(np.nanmedian(arr[:, 6]))
        if np.isfinite(arr[:, 6]).any() else float("nan"),
        note=("one row per (configuration, validation draw). Every time is the "
              "minimum over at least MIN_REPEATS repetitions clearing a 5 ms "
              "floor. The four solver baselines are measured on the first "
              "n_time draws only (they do not depend on theta). QUOTE SPEEDUPS "
              "FROM t_dbmm_paired / t_online -- an elementwise ratio of two "
              "timings taken next to each other; the standalone baselines drift "
              "against the prediction loop by tens of per cent."),
        timing_caveat=("t_rmat is an upper bound on the R-matrix cost, not the "
                       "cost of the method"),
    ))



def block_failure(system, settings, n_train=400, n_sweep=25, K_total=12, nb=32,
                  verbose=True):
    """Joint against split predictor budget, along the deformation axis.

    The diagonal blocks are one to two orders of magnitude larger than the
    coupling, so a single maxvol pass on the combined signature gives the
    diagonal essentially every point. On a rotational system, where the
    deformation lives in the coupling form factor and nowhere else, that starves
    the emulator without disturbing any elastic observable. Both variants get
    the same total budget K and training set; only the division differs.

    nb = 32: at nb = 16 the split budget is itself unconverged (7.4e-03 on the
    elastic phase shift, above the drawn tolerance), so the contrast -- a budget
    that CANNOT see the parameter (joint, K_coup = 0) against one that merely
    needs enough modes -- would be confounded with truncation error. At nb = 32
    the split budget reaches 5.6e-04 with overfit ratio 400/(12*33) = 1.01.
    """
    from core.lrom_dbmm import LROMEmulator
    from . import metrics

    channels, model, theta_c, bounds = system.preset_fn()
    E = float(system.E_demo)
    Nc = len(channels)
    # the coupling-only parameter, not the F1 sweep parameter.
    j = settings.failure_idx if settings.failure_idx >= 0 else settings.sweep_idx
    j_label = settings.failure_label or settings.sweep_label

    sweep = np.linspace(bounds[j, 0], bounds[j, 1], n_sweep)
    thetas = []
    for v in sweep:
        th = theta_c.copy()
        th[j] = v
        thetas.append(th)

    base = solver_for(system, theta_c, energies=[E])
    S_ref = np.array([solver_for(system, t, energies=[E]).solve(E)
                      for t in thetas])

    th_train = latin_hypercube(n_train, bounds, seed=SEED_TRAIN)

    def fit(Kd, Kc):
        emu = LROMEmulator(base, E)
        emu.fit(model, th_train, theta_c, K_diag=Kd, K_coup=Kc,
                eps_tol=1e-14, nb_max=nb)
        return emu

    joint = fit(K_total, 0)                       # the naive budget
    split = fit(K_total // 2, K_total - K_total // 2)

    S_joint = np.array([joint.predict(t) for t in thetas])
    S_split = np.array([split.predict(t) for t in thetas])

    def delta_of(S_grid):
        """The elastic phase shift, pointwise and branch-free:
        S_00 = eta * exp(2 i delta).

        Not the tracked eigenphase: the branch a grid latches onto depends on
        the grid, so a constant prediction (the joint budget is constant in
        beta_2) would be differenced against a different eigenvalue. The
        entrance-channel amplitude is what an elastic-only validation looks at.
        """
        return np.degrees(0.5 * np.angle(S_grid[:, 0, 0]))

    # The inelastic amplitude the elastic phase shift cannot see. Channel pair
    # (0, 1) is the ground state to first excited state transition.
    def s01_sq(S_grid):
        return np.abs(S_grid[:, 0, 1]) ** 2

    mask = metrics.coupling_mask(S_ref)

    # Where the budget actually put its points, which is the mechanism rather
    # than the symptom. Each entry of `points` is (kind, a, b, r, channels).
    def radii(emu, kind):
        return np.array([p[3] for p in emu.points if p[0] == kind], dtype=float)

    r_dense = np.linspace(1e-3, system.R, 400)
    V_diag_c, V_coup_c = model(theta_c)
    V_aa = np.abs(np.asarray(V_diag_c(r_dense, channels[0]), dtype=complex))
    V_ab = np.zeros_like(r_dense)
    if V_coup_c is not None and Nc > 1:
        V_ab = np.abs(np.asarray(V_coup_c(r_dense, channels[0], channels[1]),
                                 dtype=complex))

    return _write(system, settings, "failure", dict(
        sweep=sweep,
        delta_ref=delta_of(S_ref), delta_joint=delta_of(S_joint),
        delta_split=delta_of(S_split),
        s01_ref=s01_sq(S_ref), s01_joint=s01_sq(S_joint),
        s01_split=s01_sq(S_split),
        err_delta_joint=np.abs(delta_of(S_joint) - delta_of(S_ref))
        / np.maximum(np.abs(delta_of(S_ref)), 1e-30),
        err_delta_split=np.abs(delta_of(S_split) - delta_of(S_ref))
        / np.maximum(np.abs(delta_of(S_ref)), 1e-30),
        err_inel_joint=metrics.err_inelastic(S_joint, S_ref, mask=mask),
        err_inel_split=metrics.err_inelastic(S_split, S_ref, mask=mask),
        r_diag_joint=radii(joint, "diag"), r_coup_joint=radii(joint, "coup"),
        r_diag_split=radii(split, "diag"), r_coup_split=radii(split, "coup"),
        r_dense=r_dense, V_aa=V_aa, V_ab=V_ab,
    ), dict(
        E=E, n_train=n_train, n_sweep=n_sweep, K_total=K_total, nb=nb,
        seed_train=SEED_TRAIN, sweep_label=j_label,
        sweep_idx=j, n_channels=Nc,
        nb_joint=int(joint.nb), nb_split=int(split.nb),
        r_fit_joint=float(joint.overfit_ratio),
        r_fit_split=float(split.overfit_ratio),
        amplitude_ratio=float(np.nanmax(V_aa) / max(np.nanmax(V_ab), 1e-30)),
    ))


# --- F8: excitation functions across thresholds --------------------

def block_excitation(system, settings, methods=("rbm", "lrom", "rbm_et"),
                     n_E=N_ENERGY_NODES, n_resid=30, n_rmat=12, verbose=True):
    """The whole chain on the quantity astrophysics consumes, sigma(E).

    Two deliberate choices.

    The displayed curve is drawn at a VALIDATION parameter, never at theta_c.
    The LROM is exact at theta_c by construction, since its central gauge sets
    a(theta_c) = 0, so any figure drawn there shows an empty result.

    The residual panel is a distribution over n_resid draws rather than the
    single displayed one, so the figure says both what it looks like and how
    often.
    """
    from core.rbm import RBMEmulator
    from core.lrom_dbmm import LROMEmulator
    from . import metrics
    from .observables import energy_windows
    from .plan_calibration import as_meta, calibration_for

    channels, model, theta_c, bounds = system.preset_fn()
    lo, hi, _ = system.E_range
    E = np.linspace(lo, hi, n_E)
    Nc = len(channels)

    cal = calibration_for(system.key)
    nb, n_train = cal.nb, cal.Ns

    th_train = latin_hypercube(n_train, bounds, seed=SEED_TRAIN)
    th_valid = latin_hypercube(n_resid, bounds, seed=SEED_VALID)
    theta_show = th_valid[0]

    _warm_coulomb(system, E)
    base = solver_for(system, theta_c, energies=E)

    def sigma_of(S_grid, solver):
        return np.array([cross_section(solver, e, s, _xsec(system, settings))
                         for e, s in zip(E, S_grid)])

    # --- high-fidelity reference, for every validation draw ------------------
    def dbmm_run(theta):
        sv = solver_for(system, theta, energies=E)
        S = np.array([sv.solve(e) for e in E])
        return S, sigma_of(S, sv)

    ref = _pmap(dbmm_run, th_valid)
    S_ref = np.array([r[0] for r in ref])
    sigma_ref = np.array([r[1] for r in ref])

    # --- R-matrix, sparsely, as the independent check on the reference -------
    from core import python_rmatrix as rmat
    idx_r = np.linspace(0, n_E - 1, n_rmat).astype(int)
    sigma_rmat = np.full(n_E, np.nan)
    V_diag, V_coup = model(theta_show)
    if V_coup is None:
        def V_coup(r, ca, cb):
            return np.zeros_like(r, dtype=complex)
    mesh, _ = rmat.compute_lagrange_mesh(system.N, system.R)
    H = rmat.compute_CC_ham(mesh, system.R, channels, V_diag, V_coup)
    lag_b = rmat.eval_lag_at_boundary(mesh, system.R)
    for i in idx_r:
        B = rmat.make_boundary_matrix(mesh, system.R, channels, E[i])
        S, _ = rmat.compute_S_matrix(H + B, lag_b, channels, E[i], system.R,
                                     mesh, compute_wfs=False)
        sigma_rmat[i] = cross_section(base, E[i], S, _xsec(system, settings))

    # RBM and LROM: one offline stage per energy node. One fit per
    # (method, energy) is the real cost of a fixed-energy emulator on an
    # excitation function, and the dominant cost of the whole pipeline on this
    # suite (n_E fits of N_s snapshots, twice). The nodes are independent and
    # run in parallel: `base` is read-only during a fit (K_matrix copies, the
    # Coulomb caches are pure-function dicts, every emulator owns its basis).
    S_rbm = np.full_like(S_ref, np.nan)
    S_lrom = np.full_like(S_ref, np.nan)
    Kd, Kc = cal.budgets(1)[0]

    def fit_node(job):
        """One (method, energy) fit. Returns (predictions, seconds, failure).

        `failure` is None on success, the exception text otherwise, and the two
        are kept apart from a method that was never asked for: all three leave
        NaN behind, and only one of them is a result.
        """
        name, i, e = job
        t0 = time.perf_counter()
        try:
            if name == "rbm":
                emu = RBMEmulator(base, float(e))
                emu.fit(model, th_train, eps_tol=1e-14, nb_max=nb)
            else:
                # calibrated budget, same rank as the RBM: comparing two methods
                # means running them at one size, even though the larger nb
                # inflates the LROM least-squares system (K(nb+1) columns/node).
                emu = LROMEmulator(base, float(e))
                emu.fit(model, th_train, theta_c, K_diag=Kd, K_coup=Kc,
                        eps_tol=1e-14, nb_max=nb)
            out = np.array([emu.predict(th) for th in th_valid])
            return name, i, out, time.perf_counter() - t0, None
        except Exception as exc:                       # noqa: BLE001 - recorded
            return name, i, None, time.perf_counter() - t0, f"{type(exc).__name__}: {exc}"

    jobs = [(name, i, e) for name in ("rbm", "lrom") if name in methods
            for i, e in enumerate(E)]
    t_off = {"rbm": 0.0, "lrom": 0.0, "rbm_et": 0.0}
    failures = {}
    t_wall = time.perf_counter()
    for name, i, out, secs, failure in _pmap(fit_node, jobs):
        t_off[name] += secs
        if failure is not None:
            failures.setdefault(name, []).append(f"E[{i}]: {failure}")
            continue
        (S_rbm if name == "rbm" else S_lrom)[:, i] = out
    t_wall = time.perf_counter() - t_wall

    # --- RBM_ET: one basis per threshold-free window ------------------------
    S_et = np.full_like(S_ref, np.nan)
    if "rbm_et" in methods:
        t0 = time.perf_counter()
        try:
            from core.rbm_et import RBMEmulatorET, WindowedEmulatorET
            rng = np.random.default_rng(SEED_TRAIN)
            emus = []
            for (w_lo, w_hi) in energy_windows(system):
                E_tr = rng.uniform(w_lo, w_hi, size=len(th_train))
                e = RBMEmulatorET(base)
                e.fit(model, th_train, E_tr, eps_tol=1e-14, nb_max=nb)
                emus.append(e)
            windowed = WindowedEmulatorET(system.thresholds, emus)
            for k, th in enumerate(th_valid):
                for i, e_ in enumerate(E):
                    S_et[k, i] = windowed.predict(th, float(e_))
        except Exception as exc:                       # noqa: BLE001 - recorded
            failures["rbm_et"] = [f"{type(exc).__name__}: {exc}"]
        t_off["rbm_et"] = time.perf_counter() - t0

    def sigma_grid(S_all):
        out = np.full((S_all.shape[0], n_E), np.nan)
        for k in range(S_all.shape[0]):
            for i, e in enumerate(E):
                if np.all(np.isfinite(S_all[k, i])):
                    out[k, i] = cross_section(base, e, S_all[k, i],
                                              _xsec(system, settings))
        return out

    sigma_rbm, sigma_lrom, sigma_et = (sigma_grid(x) for x in
                                       (S_rbm, S_lrom, S_et))

    # Predictions dropped as unphysical, per method: cross_section returns NaN
    # when the largest singular value of S exceeds 1 + tol. The count tells a
    # reader whether a gap in a curve is a threshold or a failure.
    def n_unphysical(S_all, sig):
        got = np.all(np.isfinite(S_all), axis=(-2, -1))
        return int(np.sum(got & ~np.isfinite(sig)))

    # Only methods that produced a finite prediction get a provenance record, so
    # a failed fit is indistinguishable to the planner from one never computed
    # (a record on an all-NaN array would make force="if-changed" skip it).
    produced = {"rbm": S_rbm, "lrom": S_lrom, "rbm_et": S_et}
    method_meta = {}
    for m in methods:
        if m not in produced or not np.isfinite(produced[m]).any():
            continue
        rec = dict(settings=asdict(settings), system=system.fingerprint,
                   calibration=_cal_record(cal), seconds=round(t_off[m], 2))
        if failures.get(m):
            # Partially complete: some nodes are real, some are NaN. Recorded so
            # the gap is attributable rather than mysterious.
            rec["failed_nodes"] = len(failures[m])
        method_meta[m] = rec

    # Only the arrays of methods actually computed are written: cache.write
    # keeps whichever array a block omits, so an all-NaN array for an
    # unrequested method would overwrite a real result on merge.
    per_method = {}
    for m, suffix, sig in (("rbm", "rbm", sigma_rbm), ("lrom", "lrom", sigma_lrom),
                           ("rbm_et", "et", sigma_et)):
        if m in methods:
            per_method[f"sigma_{suffix}"] = sig
            per_method[f"err_{suffix}"] = metrics.err_observable(sig, sigma_ref)

    return _write(system, settings, "excitation", methods=method_meta,
                  merge=True, arrays=dict(
        E=E, idx_rmat=idx_r,
        sigma_ref=sigma_ref, sigma_rmat=sigma_rmat, **per_method,
        theta_show=theta_show, th_valid=th_valid,
    ), meta=dict(
        n_train=n_train, n_resid=n_resid, **as_meta(cal, n_E=n_E),
        seed_train=SEED_TRAIN, seed_valid=SEED_VALID,
        n_channels=Nc, N=system.N, xsec_kind=_xsec(system, settings),
        thresholds=system.thresholds,
        threshold_labels=system.threshold_labels,
        # Summed per-node fit durations, NOT a cost: the nodes run concurrently,
        # so each is inflated by contention. The amortisation figure takes its
        # offline constant from `cat`, which is timed serially.
        t_offline_rbm=t_off["rbm"], t_offline_lrom=t_off["lrom"],
        t_offline_et=t_off["rbm_et"],
        t_offline_is_wall_clock=False, t_wall=t_wall, n_workers=N_WORKERS,
        n_unphysical=dict(
            rbm=n_unphysical(S_rbm, sigma_rbm),
            lrom=n_unphysical(S_lrom, sigma_lrom),
            rbm_et=n_unphysical(S_et, sigma_et)),
        failures={m: v[:8] for m, v in failures.items()},
        n_windows=len(energy_windows(system)),
        note=("curves are drawn at a validation draw, never at theta_c, where "
              "the LROM central gauge is exact by construction"),
    ))


# --- is the split of the predictor budget worth its complexity ---------

def block_budget(system, settings, n_valid=40, n_sweep=9, verbose=True):
    """The split of K between diagonal and coupling, measured rather than argued.

    Strict A/B: same rank, training set, validation draws, and total budget
    K = K_diag + K_coup; only the division moves.

        joint   (K, 0)           one maxvol pass, the naive budget
        split   (K_diag, K_coup) two independent passes, what the code does

    Outcome: the out-of-sample elastic and inelastic error of each budget.
    Mechanism (`follows`): sweeping the coupling-only parameter, each budget's
    prediction travel over the solver's -- 1 means it tracks the parameter, 0
    that it cannot see it. `diag_moves` is how far the DIAGONAL travels under
    the same sweep, measured per parameter: where it is non-zero the parameter
    is not coupling-only and the split buys nothing.
    """
    from core.lrom_dbmm import LROMEmulator

    from . import metrics
    from .plan_calibration import calibration_for

    channels, model, theta_c, bounds = system.preset_fn()
    if len(channels) < 2:
        raise ValueError(f"{system.key}: uncoupled, there is no budget to split")
    E = float(system.E_demo)
    cal = calibration_for(system.key)

    r = np.linspace(1e-3, system.R, 200)
    V_diag_c, V_coup_c = model(theta_c)
    base_diag = [np.asarray(V_diag_c(r, c), dtype=complex) for c in channels]
    base_coup = np.asarray(V_coup_c(r, channels[0], channels[1]), dtype=complex)

    def travel(theta):
        """How far the diagonal and the coupling move, relative to their own size."""
        Vd, Vc = model(theta)
        d = max(np.max(np.abs(np.asarray(Vd(r, c), dtype=complex) - b))
                / max(np.max(np.abs(b)), 1e-30)
                for c, b in zip(channels, base_diag))
        cpl = (np.max(np.abs(np.asarray(Vc(r, channels[0], channels[1]),
                                        dtype=complex) - base_coup))
               / max(np.max(np.abs(base_coup)), 1e-30))
        return float(d), float(cpl)

    d_diag, d_coup = np.zeros(len(theta_c)), np.zeros(len(theta_c))
    for j in range(len(theta_c)):
        th = theta_c.copy()
        th[j] = bounds[j, 1]
        d_diag[j], d_coup[j] = travel(th)

    # The most nearly coupling-ONLY parameter, not the one that moves the
    # coupling most (on a rotational model R_0 moves the form factor further
    # than the deformation and also sits on the diagonal). Rank by how little
    # the diagonal moves; use coupling movement only to break ties and exclude
    # parameters that do not touch the coupling.
    live = np.flatnonzero(d_coup > 1e-12)
    j = int(min(live, key=lambda k: (d_diag[k], -d_coup[k])))

    th_train = latin_hypercube(cal.Ns, bounds, seed=SEED_TRAIN)
    th_valid = latin_hypercube(n_valid, bounds, seed=SEED_VALID)
    base = solver_for(system, theta_c, energies=[E])
    S_ref = np.array([solver_for(system, t, energies=[E]).solve(E)
                      for t in th_valid])
    mask = metrics.coupling_mask(S_ref)

    sweep = np.linspace(bounds[j, 0], bounds[j, 1], n_sweep)
    th_sweep = []
    for v in sweep:
        th = theta_c.copy()
        th[j] = v
        th_sweep.append(th)
    S_sweep_ref = np.array([solver_for(system, t, energies=[E]).solve(E)[0, 0]
                            for t in th_sweep])

    def spread(z):
        """Travel of a complex series about its own mean, relative to it."""
        z = np.asarray(z)
        return float(np.max(np.abs(z - z.mean())) / max(abs(z.mean()), 1e-30))

    out, meta_extra = {}, {}
    # The split arm uses the SPECTRAL allocation, not the applied one: after the
    # split rule the calibration spends nothing on the coupling for four systems
    # of five, so an applied-vs-joint test would compare a config with itself.
    kd_split = cal.K_diag_spec or cal.K_diag
    kc_split = cal.K_coup_spec if cal.K_coup_spec else cal.K_coup
    for name, (kd, kc) in (("joint", (kd_split + kc_split, 0)),
                           ("split", (kd_split, kc_split))):
        emu = LROMEmulator(base, E)
        emu.fit(model, th_train, theta_c, K_diag=kd, K_coup=kc,
                eps_tol=1e-14, nb_max=cal.nb)
        S_emu = np.array([emu.predict(t) for t in th_valid])
        out[f"err_el_{name}"] = metrics.err_elastic(S_emu, S_ref)
        out[f"err_in_{name}"] = metrics.err_inelastic(S_emu, S_ref, mask=mask)
        s = np.array([emu.predict(t)[0, 0] for t in th_sweep])
        out[f"sweep_{name}"] = s
        meta_extra[f"follows_{name}"] = spread(s) / max(spread(S_sweep_ref), 1e-30)
        meta_extra[f"nb_{name}"] = int(emu.nb)
        if verbose:
            print(f"  {system.key}/{name}: K=({kd},{kc}) "
                  f"elastic {np.nanmedian(out[f'err_el_{name}']):.2e} "
                  f"inelastic {np.nanmedian(out[f'err_in_{name}']):.2e}")

    return _write(system, settings, "budget", dict(
        d_diag=d_diag, d_coup=d_coup, sweep=sweep, sweep_ref=S_sweep_ref, **out,
    ), dict(
        E=E, nb=cal.nb, K_diag=kd_split, K_coup=kc_split,
        K_total=kd_split + kc_split,
        K_applied_diag=cal.K_diag, K_applied_coup=cal.K_coup,
        split_applied=bool(cal.split_needed),
        Ns=cal.Ns, n_valid=n_valid, n_sweep=n_sweep,
        seed_train=SEED_TRAIN, seed_valid=SEED_VALID,
        coup_idx=j, coup_label=(system.theta_labels or [])[j]
        if system.theta_labels else str(j),
        diag_moves=float(d_diag[j]), coup_moves=float(d_coup[j]),
        calibrated=True, **meta_extra,
    ))


BLOCK_FUNCTIONS = {
    "sweep": block_sweep,
    "validation": block_validation,
    "invariants": block_invariants,
    "manifold": block_manifold,
    "failure": block_failure,
    "excitation": block_excitation,
    "interior": block_interior,
    "cat": block_cat,
    "scaling": block_scaling,
    "budget": block_budget,
}
