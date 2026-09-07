"""Central-gauge residual-fit LROM emulator for the DBMM solver.
Reference: P. Giuliani, "Learning Reduced-Order Models" (clean central-gauge variant).

LROMEmulator: fixed-energy emulator over theta.
"""

import time
import warnings

import numpy as np

from .dbmm import (
    _channel_kinematics,
    _is_open,
    _coulomb_f_at_mesh,
    lagrange_basis_at,
    assemble_V_matrix,
)
from .rbm import (
    eval_potential_on_mesh,
    build_source,
    boundary_rows,
    s_matrix_from_boundary,
    observables_from_coeffs,
    _coulomb_source_factors,
)

#: (n_train, K, nb) triples already warned about. See LROMEmulator.fit.
_WARNED_UNDERDETERMINED = set()


#: Radii below this are not offered to the selection: the wave function goes as
#: r^(l+1) at the origin, so the potential there cannot move the solution.
R_MIN_PREDICTOR = 0.5


def _potential_signature_diag(Vnuc, Nc):
    """Complex signature vector from the diagonal blocks V_aa(r), one row per
    (channel, radius). Kept complex, not real+imag stacked: stacking lets two
    selected indices fold onto the same radius (7 distinct of 10 at K=10).
    """
    return np.concatenate([Vnuc[a] for a in range(Nc)])


def _potential_signature_coup(Vcoup, Nc):
    """Complex signature from V_ab(r), a<b. Returns (None, []) if no coupling."""
    pairs = [(a, b) for a in range(Nc) for b in range(a + 1, Nc)]
    if Vcoup is None or not pairs:
        return None, []
    return np.concatenate([Vcoup[a, b] for a, b in pairs]), pairs


def _decode_diag_index(idx, N, Nc):
    # one row per (channel, radius): nothing to fold.
    block, i = divmod(idx, N)
    return ('diag', block, None, i)


def _decode_coup_index(idx, N, pairs):
    block, i = divmod(idx, N)
    a, b = pairs[block]
    return ('coup', a, b, i)


def _greedy_maxvol_indices(basis):
    """Maxvol-style row selection: a forward greedy pass with a QR-based residual
    update, then up to 50 classical maxvol swaps (replace any row whose
    coefficient in the current submatrix inverse exceeds 1). `basis` is
    (n_rows, K); returns K sorted row indices.
    """
    n_rows, n_cols = basis.shape

    selected = [int(np.argmax(np.linalg.norm(basis, axis=1)))]
    for _ in range(1, n_cols):
        q, *_ = np.linalg.qr(basis[selected, :].T, mode="reduced")
        residual = basis - (basis @ q) @ q.T.conj()
        scores = np.linalg.norm(residual, axis=1)
        scores[selected] = -np.inf
        selected.append(int(np.argmax(scores)))

    for _ in range(50):
        sub = basis[selected, :]
        try:
            coeff = basis @ np.linalg.inv(sub)
        except np.linalg.LinAlgError:
            break
        abs_coeff = np.abs(coeff)
        abs_coeff[selected, :] = 0.0
        row, col = np.unravel_index(np.argmax(abs_coeff), abs_coeff.shape)
        if abs_coeff[row, col] <= 1.0 + 1e-10:
            break
        selected[col] = int(row)

    return np.array(sorted(selected), dtype=int)


def _channel_amplitudes(solver, E):
    """|psi_a(r)| of the central solution, per channel, normalised to its peak.

    Candidate rows are weighted by this before the SVD: a potential variation
    where the solution has no amplitude carries no information about delta c.
    |psi| beat |psi|^2 in measurement.
    """
    _r, psi, _inc = solver.wavefunction(E, total=True)
    amp = np.sqrt(np.sum(np.abs(np.atleast_3d(psi)) ** 2, axis=2))   # (Nc, N)
    peak = np.maximum(amp.max(axis=1, keepdims=True), 1e-300)
    return amp / peak


def _candidate_mask(r, blocks):
    """Rows of a `blocks`-block signature that the selection may choose."""
    return np.tile(r >= R_MIN_PREDICTOR, blocks)


def _select_predictor_points_diag(model, theta_train, theta_c, solver, K, E):
    Nc, N, r = solver.Nc, solver.N, solver.r
    Vnuc_c, _ = eval_potential_on_mesh(model, theta_c, solver)
    sig_c = _potential_signature_diag(Vnuc_c, Nc)

    deltas = []
    for theta in theta_train:
        Vnuc, _ = eval_potential_on_mesh(model, theta, solver)
        deltas.append(_potential_signature_diag(Vnuc, Nc) - sig_c)
    deltas = np.array(deltas).T

    amp = _channel_amplitudes(solver, E)
    deltas = deltas * np.concatenate([amp[a] for a in range(Nc)])[:, None]

    allowed = np.flatnonzero(_candidate_mask(r, Nc))
    U, s, _ = np.linalg.svd(deltas[allowed], full_matrices=False)
    K = min(K, U.shape[1])
    local = allowed[_greedy_maxvol_indices(U[:, :K])]

    points, center = [], []
    for idx in local:
        _, a, _, i = _decode_diag_index(idx, N, Nc)
        ri = r[i]
        ch = solver.channels[a]
        val = model(theta_c)[0](np.array([ri]), ch)[0]
        points.append(('diag', a, None, ri, ch))
        center.append(complex(val))
    center = np.array(center, dtype=complex)

    raw_train = np.array([_evaluate_points(model, theta, points) for theta in theta_train])
    scales = np.maximum(np.std(raw_train - center[np.newaxis, :], axis=0), 1e-12)

    return points, center, scales, s


def _select_predictor_points_coup(model, theta_train, theta_c, solver, K, E):
    Nc, N, r = solver.Nc, solver.N, solver.r
    _, Vcoup_c = eval_potential_on_mesh(model, theta_c, solver)
    sig_c, pairs = _potential_signature_coup(Vcoup_c, Nc)
    if sig_c is None or K <= 0:
        return [], np.zeros(0, dtype=complex), np.zeros(0, dtype=float), np.zeros(0)

    deltas = []
    for theta in theta_train:
        _, Vcoup = eval_potential_on_mesh(model, theta, solver)
        sig, _ = _potential_signature_coup(Vcoup, Nc)
        deltas.append(sig - sig_c)
    deltas = np.array(deltas).T

    # A coupling block moves the solution through both of its channels, so the
    # weight is the geometric mean of the two amplitudes.
    amp = _channel_amplitudes(solver, E)
    deltas = deltas * np.concatenate(
        [np.sqrt(amp[a] * amp[b]) for a, b in pairs])[:, None]

    allowed = np.flatnonzero(_candidate_mask(r, len(pairs)))
    U, s, _ = np.linalg.svd(deltas[allowed], full_matrices=False)
    K = min(K, U.shape[1])
    local = allowed[_greedy_maxvol_indices(U[:, :K])]

    points, center = [], []
    for idx in local:
        _, a, b, i = _decode_coup_index(idx, N, pairs)
        ri = r[i]
        ch_a, ch_b = solver.channels[a], solver.channels[b]
        val = model(theta_c)[1](np.array([ri]), ch_a, ch_b)[0]
        points.append(('coup', a, b, ri, (ch_a, ch_b)))
        center.append(complex(val))
    center = np.array(center, dtype=complex)

    raw_train = np.array([_evaluate_points(model, theta, points) for theta in theta_train])
    scales = np.maximum(np.std(raw_train - center[np.newaxis, :], axis=0), 1e-12)

    return points, center, scales, s


def _select_predictor_points(model, theta_train, theta_c, solver, K_diag, K_coup, E):
    """Delta-maxvol selection run separately on V_aa and V_ab: V_aa >> V_ab in
    amplitude, so a joint selection starves the coupling; separate budgets
    (K_diag, K_coup) fix this. Energy-dependent, through the amplitude weighting
    of _channel_amplitudes.
    """
    points_d, center_d, scales_d, sv_d = _select_predictor_points_diag(
        model, theta_train, theta_c, solver, K_diag, E)
    points_c, center_c, scales_c, sv_c = _select_predictor_points_coup(
        model, theta_train, theta_c, solver, K_coup, E)

    points = points_d + points_c
    center = np.concatenate([center_d, center_c])
    scales = np.concatenate([scales_d, scales_c])

    return points, center, scales, (sv_d, sv_c)


def _evaluate_points(model, theta, points):
    """Evaluate the K selected potential points for a single theta, no mesh.

    One array call per (kind, channel pair) group -- two or three, not K. A loop
    of K single-element calls carried a per-element constant ~30x the vectorised
    mesh pass (0.334 ms vs 0.049 ms for 64 mesh points on n+40Ca), enough to
    make the LROM query measure slower than the RBM.
    """
    V_diag, V_coup = model(theta)
    out = np.empty(len(points), dtype=complex)
    groups = {}
    for k, (kind, a, b, ri, ch) in enumerate(points):
        # Keyed on the channel INDICES, which are hashable; the channel objects
        # themselves need not be, and are carried along for the call.
        slot = groups.setdefault((kind, a, b), ([], [], ch))
        slot[0].append(k)
        slot[1].append(ri)
    for (kind, _a, _b), (idx, radii, ch) in groups.items():
        r = np.asarray(radii, dtype=float)
        if kind == 'diag':
            vals = V_diag(r, ch)
        else:
            ch_a, ch_b = ch
            vals = V_coup(r, ch_a, ch_b)
        out[np.asarray(idx, dtype=int)] = np.asarray(vals, dtype=complex)
    return out


def predictors(model, theta, points, center, scales):
    """Centered, scaled predictors p(theta) in (K,), p(theta_c) = 0 by construction."""
    raw = _evaluate_points(model, theta, points)
    return (raw - center) / scales


def fit_central_lrom(predictor_matrix, coeff_targets):
    """Fit [I + Σ_j p_j M_j] a = Σ_j p_j b_j by one lstsq [Giuliani, central RF-LROM, gauge M0=I].
    Returns (matrices (K,nb,nb), vectors (K,nb), info dict).

    The system decouples: the equation indexed (s, row) touches only row `row`
    of each M_j and component `row` of each b_j, and its coefficients (p_j a_s,
    -p_j) do not depend on `row`. The nb subproblems share one design matrix and
    differ only in their right-hand side, so the fit is a single lstsq with nb
    right-hand sides, n_samples x K(nb+1) instead of (n_samples nb) x K nb(nb+1)
    -- 0.165 s here against 47.8 s block-assembled at n_samples=400, K=nb=16,
    for a 1e-13 difference on the solution.
    """
    P = np.asarray(predictor_matrix, dtype=complex)
    A = np.asarray(coeff_targets, dtype=complex)
    n_samples, K = P.shape
    nb = A.shape[1]

    # Colonnes, par bloc j : [p_j a_0, ..., p_j a_{nb-1}, -p_j].
    design = np.empty((n_samples, K * (nb + 1)), dtype=complex)
    for j in range(K):
        design[:, j * (nb + 1): j * (nb + 1) + nb] = P[:, j, None] * A
        design[:, j * (nb + 1) + nb] = -P[:, j]

    t0 = time.perf_counter()
    solution, _res, rank, _sv = np.linalg.lstsq(design, -A, rcond=None)
    train_seconds = time.perf_counter() - t0

    matrices = np.zeros((K, nb, nb), dtype=complex)
    vectors = np.zeros((K, nb), dtype=complex)
    for j in range(K):
        blk = solution[j * (nb + 1): (j + 1) * (nb + 1), :]   # (nb+1, nb)
        # colonne row de blk = solution du second membre row = ligne row de M_j
        matrices[j] = blk[:nb, :].T
        vectors[j] = blk[nb, :]

    residual = design @ solution + A
    info = dict(train_seconds=train_seconds,
                residual_mse=float(np.mean(np.abs(residual) ** 2)),
                # rank of the decoupled design matrix (what overfit_ratio
                # compares against n_samples).
                rank=int(rank))
    return matrices, vectors, info


def predict_central_lrom(matrices, vectors, p_row):
    """Solve the fitted implicit system (*) for one predictor row p(theta)."""
    nb = vectors.shape[1]
    matrix = np.eye(nb, dtype=complex) + np.einsum('k,kij->ij', p_row, matrices)
    rhs = np.einsum('k,kj->j', p_row, vectors)
    return np.linalg.solve(matrix, rhs)


class LROMEmulator:
    """Central-gauge residual-fit LROM emulator over theta at fixed E."""

    def __init__(self, base_solver, E):
        self.base = base_solver
        self.E = float(E)
        self.N, self.R, self.Nc = base_solver.N, base_solver.R, base_solver.Nc

        self.K_op = base_solver.K_matrix(self.E)
        (self._sq, self._open, self._F,
         self._Vsc) = _coulomb_source_factors(base_solver, self.E)

        self.X_r = None
        self.theta_c = None
        self.C_c = None                # DBMM coefficients at theta_c, (Ntot, Nc)
        self.points = self.center = self.scales = None
        self._fits = {}                # b0 -> (matrices, vectors)
        self._stacked = None           # the same, batched for prediction
        self._model = None
        self.singular_values = None
        self.train_info = {}
        self.n_train = None

    def _snapshot(self, theta):
        Vnuc, Vcoup = eval_potential_on_mesh(self._model, theta, self.base)
        V = assemble_V_matrix(Vnuc, Vcoup, self.N, self.Nc)
        b = build_source(Vnuc, Vcoup, self.N, self.Nc,
                          self._sq, self._open, self._F, self._Vsc)
        return np.linalg.solve(self.K_op + V, b)

    def fit(self, model, theta_train, theta_c, K_diag=6, K_coup=6, eps_tol=1e-8,
            nb_max=None, predictor_points=None):
        """predictor_points: pass (points, center, scales) to skip re-selection.

        The selection is energy-dependent (it weights each candidate radius by
        the central solution there), so points passed in belong to the energy
        that chose them. The pipeline passes none: each energy node selects its own.
        """
        self._model = model
        self.theta_c = np.asarray(theta_c, dtype=float)
        self.n_train = len(theta_train)

        self.C_c = self._snapshot(self.theta_c)
        deltas = [self._snapshot(theta) - self.C_c for theta in theta_train]
        Delta = np.hstack(deltas)

        U, s, _ = np.linalg.svd(Delta, full_matrices=False)
        energy = np.cumsum(s ** 2) / np.sum(s ** 2)
        nb = int(np.searchsorted(energy, 1.0 - eps_tol) + 1)
        nb = min(nb, len(s))
        if nb_max is not None:
            nb = min(nb, nb_max)
        self.X_r = U[:, :nb]
        self.singular_values = s
        # central solution contracted with the boundary row (the basis is
        # contracted on demand by boundary_rows, so a lower-rank slice of X_r
        # stays correct).
        self.chi_c = np.stack([self.base.fj1 @ self.C_c[a * self.N:(a + 1) * self.N]
                               for a in range(self.Nc)]) / np.sqrt(self.R)

        if predictor_points is not None:
            self.points, self.center, self.scales = predictor_points
        else:
            self.points, self.center, self.scales, _sv = _select_predictor_points(
                model, theta_train, self.theta_c, self.base, K_diag, K_coup, self.E)

        P_train = np.array([predictors(model, th, self.points, self.center, self.scales)
                             for th in theta_train])
        open_channels = [a for a, ch in enumerate(self.base.channels) if _is_open(ch, self.E)]
        dC_reduced = [self.X_r.conj().T @ dC for dC in deltas]
        for b0 in open_channels:
            targets = np.array([dCr[:, b0] for dCr in dC_reduced])
            matrices, vectors, info = fit_central_lrom(P_train, targets)
            self._fits[b0] = (matrices, vectors)
            self._stacked = None            # refitting invalidates the batch
            self.train_info[b0] = info

        # Warn once per (n_train, K, nb), not once per fit. r_fit < 1 is the
        # calibrated pipeline's normal regime (N_s = c*nb collapses the ratio to
        # c/K, ~0.26-0.32 here); the value is carried as overfit_ratio, not
        # suppressed.
        ratio = self.overfit_ratio
        if ratio < 1.0:
            signature = (self.n_train, self.n_predictors, self.nb)
            if signature not in _WARNED_UNDERDETERMINED:
                _WARNED_UNDERDETERMINED.add(signature)
                warnings.warn(
                    f"LROMEmulator overfit ratio n_train / (K*(nb+1)) = {ratio:.3f} < 1: "
                    f"n_train={self.n_train}, K={self.n_predictors}, nb={self.nb}. "
                    "The residual-fit least-squares system is underdetermined and "
                    "lstsq returns the minimum-norm interpolant. Reported once per "
                    "(n_train, K, nb); see Calibration.r_fit.",
                    UserWarning, stacklevel=2)

        return self

    @property
    def nb(self):
        return 0 if self.X_r is None else self.X_r.shape[1]

    @property
    def n_predictors(self):
        return 0 if self.points is None else len(self.points)

    @property
    def overfit_ratio(self):
        """n_train / (n_predictors * (nb + 1)) -- should be >= 1 for a
        well-determined residual-fit least-squares system."""
        if self.n_train is None or self.n_predictors == 0 or self.nb == 0:
            return float('nan')
        return self.n_train / (self.n_predictors * (self.nb + 1))

    @property
    def open_channels(self):
        return list(self._fits.keys())

    def get_spline_data(self, nb_min=None):
        """Return (X_r, C_c, fits_dict) truncated to nb_min modes for LROMEnergyScan spline construction."""
        nb = min(self.nb, nb_min) if nb_min is not None else self.nb
        fits = {b0: (M[:, :nb, :nb].copy(), v[:, :nb].copy())
                for b0, (M, v) in self._fits.items()}
        return self.X_r[:, :nb].copy(), self.C_c.copy(), fits

    # -- prediction (K potential points, one stacked solve, one gemm) ------
    def _stacked_fits(self):
        """The per-channel fits, laid out so a query is two gemv and one solve.

        The fitted operator differs per entrance channel (no shared
        factorisation), but its dispatch batches: numpy solves the stacked
        (n_open, nb, nb) operand in one call and one gemm lifts every channel,
        where a Python loop paid n_open round trips.

        Layout is the real gain: summing p_j M_j contracts over K, so K first
        and contiguous makes it a gemv on a (K, n_open nb^2) matrix. That
        memory-bound object then streams at ~38 GB/s (0.82 ms) instead of
        ~8 GB/s (3.78 ms) for a channel-first einsum, on alpha+12C. Holds a
        second 32 MB copy of `_fits`.
        """
        if self._stacked is None:
            cols = list(self._fits.keys())
            # (K, n_open, nb, nb) and (K, n_open, nb): K leading and contiguous.
            M_kc = np.ascontiguousarray(
                np.stack([self._fits[b0][0] for b0 in cols], axis=1))
            v_kc = np.ascontiguousarray(
                np.stack([self._fits[b0][1] for b0 in cols], axis=1))
            self._stacked = (np.asarray(cols, dtype=int), M_kc, v_kc,
                             M_kc.reshape(M_kc.shape[0], -1),
                             v_kc.reshape(v_kc.shape[0], -1))
        return self._stacked

    def predict_coefficients(self, theta):
        p_row = predictors(self._model, theta, self.points, self.center, self.scales)
        cols, M_kc, _v_kc, M_flat, v_flat = self._stacked_fits()
        nopen, nb = M_kc.shape[1], M_kc.shape[2]
        A = (p_row @ M_flat).reshape(nopen, nb, nb)
        A += np.eye(nb, dtype=A.dtype)                  # the M0 = I gauge
        rhs = (p_row @ v_flat).reshape(nopen, nb)
        a = np.linalg.solve(A, rhs[:, :, None])[:, :, 0]        # (n_open, nb)
        C = self.C_c.copy()
        C[:, cols] = self.C_c[:, cols] + self.X_r @ a.T
        return C

    def predict(self, theta):
        """U at theta, without forming the full-mesh solution.

        Same reduced solve as predict_coefficients, but skips the lift: the
        central solution's boundary values are stored, so the update is
        G_r @ a on the open columns alone.
        """
        p_row = predictors(self._model, theta, self.points, self.center, self.scales)
        cols, M_kc, _v_kc, M_flat, v_flat = self._stacked_fits()
        nopen, nb = M_kc.shape[1], M_kc.shape[2]
        A = (p_row @ M_flat).reshape(nopen, nb, nb)
        A += np.eye(nb, dtype=A.dtype)                  # the M0 = I gauge
        rhs = (p_row @ v_flat).reshape(nopen, nb)
        a = np.linalg.solve(A, rhs[:, :, None])[:, :, 0]        # (n_open, nb)
        G = boundary_rows(self.X_r, self.base)
        chi_R = self.chi_c.copy()
        chi_R[:, cols] = self.chi_c[:, cols] + (G @ a.T) / np.sqrt(self.R)
        return s_matrix_from_boundary(chi_R, self.base, self.E)

    def predict_with_residual(self, theta):
        """Returns (S, unitarity_residual) where residual = ‖S†S − I‖_F.
        Values >> 0 indicate emulator failure (S-matrix unphysical)."""
        S = self.predict(theta)
        Nc = S.shape[0]
        res = float(np.linalg.norm(S.conj().T @ S - np.eye(Nc), 'fro'))
        return S, res

    def predict_observables(self, theta):
        """LROM counterpart of RBMEmulator.predict_observables / DBMMSolver.solve_observables."""
        return observables_from_coeffs(self.predict_coefficients(theta), self.base, self.E)

    def wavefunction(self, theta, r_eval=None, incident=None, total=False):
        """LROM counterpart of RBMEmulator.wavefunction / DBMMSolver.wavefunction."""
        N, R = self.N, self.R
        if r_eval is None:
            r_eval = self.base.r
        r_eval = np.atleast_1d(np.asarray(r_eval, dtype=float))
        basis = lagrange_basis_at(self.base.x, self.base.lam, r_eval / R)

        C = self.predict_coefficients(theta)
        incident_channels = (
            [incident] if incident is not None
            else [b0 for b0, ch in enumerate(self.base.channels) if _is_open(ch, self.E)])

        chi = np.zeros((self.Nc, len(r_eval), len(incident_channels)), dtype=complex)
        for k_idx, b0 in enumerate(incident_channels):
            for a in range(self.Nc):
                chi[a, :, k_idx] = (basis @ C[a * N:(a + 1) * N, b0]) / np.sqrt(R)

        if total:
            for k_idx, b0 in enumerate(incident_channels):
                ch_b0 = self.base.channels[b0]
                k, eta = _channel_kinematics(ch_b0, self.E)
                F = _coulomb_f_at_mesh(ch_b0['l'], eta, k, r_eval, self.base._f_cache)
                chi[b0, :, k_idx] += F

        return r_eval, chi, incident_channels

    def reaction_cross_section(self, theta, weight=None, S=None):
        if S is None:
            S = self.predict(theta)
        return self.base.reaction_cross_section(self.E, weight=weight, S=S)

    def differential_cross_section(self, theta, angles, entrance=0, S=None,
                                    eta=None, spin_half=False, j_minus_channel=None):
        if S is None:
            S = self.predict(theta)
        return self.base.differential_cross_section(
            self.E, angles, entrance=entrance, S=S,
            eta=eta, spin_half=spin_half, j_minus_channel=j_minus_channel)


