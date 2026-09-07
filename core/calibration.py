"""SVD-driven hyperparameter selection (K_diag, K_coup, nb, n_train) for RBM and LROM."""
import numpy as np
from .rbm import latin_hypercube, RBMEmulator
from .lrom_dbmm import _select_predictor_points_diag, _select_predictor_points_coup


# The two tolerances everything else derives from, declared here so no block
# picks its own: eps_pot reads the predictor budget off the potential spectrum,
# eps_sol reads the basis rank off the solution spectrum. K_diag, K_coup, nb,
# N_s and the rank ladder all follow from these two.
EPS_POT = 5e-3
EPS_SOL = 1e-8



def _potential_svd(model, theta_c, bounds, solver, n_probe, seed, E):
    """SVD of diagonal and coupling potential delta-snapshots; singular values normalised to 1.

    Takes E because the selection weights each candidate radius by the central
    solution there, so the budget is read off the same weighted spectrum the
    selection ranks on.
    """
    theta_probe = latin_hypercube(n_probe, bounds, seed=seed)
    _, _, _, sv_d = _select_predictor_points_diag(
        model, theta_probe, theta_c, solver, n_probe, E)
    sv_d = sv_d / sv_d[0]

    Nc = solver.Nc
    if Nc > 1:
        _, _, _, sv_c = _select_predictor_points_coup(
            model, theta_probe, theta_c, solver, n_probe, E)
        sv_c = sv_c / sv_c[0] if len(sv_c) > 0 else np.zeros(0)
    else:
        sv_c = np.zeros(0)

    return sv_d, sv_c


def _k_from_spectral(sv, eps):
    """Smallest K (1-indexed) where sv[K] < eps (spectral criterion)."""
    if len(sv) == 0:
        return 0
    crossing = np.where(sv < eps)[0]
    return int(crossing[0]) if len(crossing) > 0 else len(sv)


def _nb_from_energy(sv_solution, eps):
    """Smallest nb s.t. cumulative energy fraction ≥ 1−eps (mirrors rbm.pod_basis)."""
    s = sv_solution
    energy = np.cumsum(s ** 2) / np.sum(s ** 2)
    nb = int(np.searchsorted(energy, 1.0 - eps) + 1)
    return min(nb, len(s))


def _solution_svd(model, theta_probe, solver, E):
    """SVD of the C-matrix snapshot matrix; returns singular values normalised to s[0]=1."""
    emu = RBMEmulator(solver, E)
    snaps = emu.snapshots(model, theta_probe)
    _, s, _ = np.linalg.svd(snaps, full_matrices=False)
    return s / s[0]


def _nb_multiE(model, theta_probe, solver, E_list, eps_sol):
    """Max nb over several energies — guards against E_demo being in a 'flat' regime."""
    return max(_nb_from_energy(_solution_svd(model, theta_probe, solver, E), eps_sol)
               for E in E_list)


def _log_range(lo, hi, n=6):
    """n integers log-spaced in [lo, hi], deduplicated and sorted."""
    pts = np.unique(np.round(np.exp(
        np.linspace(np.log(max(lo, 1)), np.log(max(hi, lo + 1)), n)
    )).astype(int))
    return tuple(int(x) for x in pts)


# A parameter is "absent from the diagonal" when pushing it to the edge of its
# box leaves every channel's diagonal potential unchanged -- an exact zero (it
# does not appear in V_diag), not a small number. Rotational systems, where
# static reorientation puts the deformation on the diagonal, travel 5e-4 to
# 4e-1, so any deep-middle threshold separates the two cases; 1e-9 sits far
# below the smallest non-zero travel and far above float noise.
DIAG_ABSENT = 1e-9


def _diagonal_travel(model, theta_c, bounds, solver):
    """How far the diagonal and the coupling move when each parameter is pushed
    to the top of its box, each relative to its own size.

    Returns (d_diag, d_coup), one entry per parameter.
    """
    r = np.asarray(solver.r)
    channels = solver.channels
    Vd0, Vc0 = model(theta_c)
    base_diag = [np.asarray(Vd0(r, c), dtype=complex) for c in channels]
    base_coup = (np.asarray(Vc0(r, channels[0], channels[1]), dtype=complex)
                 if len(channels) > 1 else np.zeros_like(r, dtype=complex))

    d_diag = np.zeros(len(theta_c))
    d_coup = np.zeros(len(theta_c))
    for j in range(len(theta_c)):
        th = np.asarray(theta_c, dtype=float).copy()
        th[j] = bounds[j, 1]
        Vd, Vc = model(th)
        d_diag[j] = max(
            np.max(np.abs(np.asarray(Vd(r, c), dtype=complex) - b))
            / max(np.max(np.abs(b)), 1e-30)
            for c, b in zip(channels, base_diag))
        if len(channels) > 1:
            d_coup[j] = (np.max(np.abs(np.asarray(
                Vc(r, channels[0], channels[1]), dtype=complex) - base_coup))
                / max(np.max(np.abs(base_coup)), 1e-30))
    return d_diag, d_coup


def _split_is_needed(model, theta_c, bounds, solver):
    """Does any parameter act on the coupling and not on the diagonal?

    Returns (needed, travel), travel the diagonal movement of the most nearly
    coupling-only parameter -- the number the decision is read on. Ranking by
    how LITTLE the diagonal moves is deliberate: on a rotational model R_0 moves
    the coupling form factor more than the deformation does yet also sits on the
    diagonal, where a joint budget already sees it.
    """
    d_diag, d_coup = _diagonal_travel(model, theta_c, bounds, solver)
    live = np.flatnonzero(d_coup > 1e-12)
    if live.size == 0:
        return False, float("nan")
    j = int(min(live, key=lambda k: (d_diag[k], -d_coup[k])))
    return bool(d_diag[j] <= DIAG_ABSENT), float(d_diag[j])


def auto_calibrate(
    model,
    theta_c,
    bounds,
    solver,
    E_demo,
    coupled,
    *,
    eps_pot=EPS_POT,
    eps_sol=EPS_SOL,
    n_probe=120,
    E_range=None,
    seed=0,
    verbose=True,
):
    sv_d, sv_c = _potential_svd(model, theta_c, bounds, solver, n_probe, seed, E_demo)
    K_diag = _k_from_spectral(sv_d, eps_pot)
    K_coup = _k_from_spectral(sv_c, eps_pot) if coupled and len(sv_c) > 0 else 0

    # safety: at least 2 predictors (maxvol needs K >= 2)
    K_diag = max(K_diag, 2)
    K_coup = max(K_coup, 2) if coupled else 0
    K_total = K_diag + K_coup

    # The spectral budgets say how many points each block needs; whether to
    # spend any on the coupling is separate. At fixed K every point moved to the
    # coupling is taken from the diagonal, so splitting when the parameter
    # already reaches the diagonal is a loss. Keep the spectral pair for the
    # block that tests both allocations; apply the split decision to the rest.
    K_diag_spec, K_coup_spec = K_diag, K_coup
    split_needed, diag_travel = (True, float("nan"))
    if coupled and K_coup > 0:
        split_needed, diag_travel = _split_is_needed(model, theta_c, bounds, solver)
        if not split_needed:
            K_diag, K_coup = K_total, 0
        if verbose:
            print(f"  [calibrate] coupling parameter moves the diagonal by "
                  f"{diag_travel:.3g}; budget "
                  f"{'split' if split_needed else 'kept joint'} "
                  f"({K_diag}+{K_coup})")

    theta_probe = latin_hypercube(n_probe, bounds, seed=seed + 1)

    # Probe at multiple energies: E_demo alone can be in a "flat" regime that
    # underestimates nb for systems with rich energy dependence (heavy-ion, thresholds).
    sv_sol = None
    if E_range is not None:
        E_lo, E_hi = float(E_range[0]), float(E_range[1])
        E_probe_list = np.unique([E_lo, 0.5 * (E_lo + E_hi), E_hi, float(E_demo)])
        nb = _nb_multiE(model, theta_probe, solver, E_probe_list, eps_sol)
    else:
        sv_sol = _solution_svd(model, theta_probe, solver, E_demo)
        nb = _nb_from_energy(sv_sol, eps_sol)

    # The snapshot matrix has n_probe * Nc columns, not n_probe: the solver
    # returns every incident channel at once. Compare nb to that, not to the
    # parameter count.
    n_cols = n_probe * getattr(solver, "Nc", 1)
    if nb > 0.7 * n_cols and verbose:
        print(f"  [calibrate] Warning: nb={nb} close to the {n_cols} snapshot "
              f"columns. Increase n_probe for a reliable estimate.")

    # n_train >= 5*nb: conservative oversampling (3x underfits complex/heavy-ion
    # systems).
    n_train_rbm  = max(int(np.ceil(5 * nb      / 10) * 10), 50)
    # n_train >= 6*K_total, and at least as large as n_train_rbm
    n_train_lrom = max(int(np.ceil(6 * K_total / 10) * 10), n_train_rbm)

    rbm_nb_list = _log_range(max(1, nb // 4), nb, n=6)
    lrom_K_list = _log_range(max(2, K_diag // 3), K_diag, n=6)

    result = dict(
        K_diag=K_diag, K_coup=K_coup,
        K_diag_spec=K_diag_spec, K_coup_spec=K_coup_spec,
        split_needed=bool(split_needed), diag_travel=float(diag_travel),
        nb=nb,
        n_train_rbm=n_train_rbm,
        n_train_lrom=n_train_lrom,
        rbm_nb_list=rbm_nb_list,
        lrom_K_list=lrom_K_list,
        sv_diag=sv_d, sv_coup=sv_c,
        sv_solution=sv_sol,
    )

    if verbose:
        print(f"  [calibrate] K_diag={K_diag}, K_coup={K_coup}  (eps_pot={eps_pot:.0e})")
        print(f"  [calibrate] nb={nb}  (eps_sol={eps_sol:.0e})")
        print(f"  [calibrate] n_train_rbm={n_train_rbm}, n_train_lrom={n_train_lrom}")
        print(f"  [calibrate] rbm_nb_list={rbm_nb_list}")
        print(f"  [calibrate] lrom_K_list={lrom_K_list}")

    return result
