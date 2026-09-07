"""Joint (theta, E) POD-Galerkin RBM emulators.

RBMEmulatorET  — plain Galerkin; V(theta) projected on the full mesh online.
RBMEmulatorETP — DEIM predictor points for V_r and b_r, eliminating all full-mesh
                 evaluations at prediction time (O(K·nb²) vs O(N·Nc·nb)).

Both exploit the exact affine decomposition of K(E):

    K(E) = K0_static − Σ_a (E − E_a) D_a + Σ_a boundary_row_a(E)

so K_r(E) = X_r† K(E) X_r is assembled analytically at any E.
The reduced basis is trained on joint (theta, E) samples within a single
open-channel window (fit raises ValueError if a threshold is crossed).
"""

import numpy as np

from .dbmm import (
    _channel_kinematics,
    _is_open,
    _coulomb_f_at_mesh,
    _gamma_s,
    lagrange_basis_at,
    assemble_V_matrix,
)
from .rbm import (
    eval_potential_on_mesh,
    project_potential,
    project_dense,
    s_matrix_from_coeffs,
    observables_from_coeffs,
    _coulomb_source_factors,
    build_source,
    pod_basis,
)
from .lrom_dbmm import _greedy_maxvol_indices, predictors, _evaluate_points


class RBMEmulatorET:
    """Joint (E, theta) POD-Galerkin RBM emulator.

    K_r(E) is assembled analytically at any E from precomputed affine components
    (K0_r, Dr[a]) plus one O(N·nb) boundary-row reprojection per channel.
    Online cost: O(Nc·nb²) K_r + O(N·Nc·nb) V_r (full mesh projection) + O(nb³) solve.
    """

    def __init__(self, base_solver):
        self.base = base_solver
        self.N, self.R, self.Nc = base_solver.N, base_solver.R, base_solver.Nc
        self.X_r = self.K0_r = None
        self._Dr = None
        self._open = None
        self.singular_values = self.energy_curve = None
        self._model = None

    def _snapshot(self, theta, E):
        K = self.base.K_matrix(E)
        Vnuc, Vcoup = eval_potential_on_mesh(self._model, theta, self.base)
        V = assemble_V_matrix(Vnuc, Vcoup, self.N, self.Nc)
        sq, open_in, Fmesh, Vsc = _coulomb_source_factors(self.base, E)
        b = build_source(Vnuc, Vcoup, self.N, self.Nc, sq, open_in, Fmesh, Vsc)
        return np.linalg.solve(K + V, b), open_in

    def fit(self, model, theta_train, E_train, eps_tol=1e-8, nb_max=None):
        self._model = model
        C_list, open_ref = [], None
        for theta, E in zip(theta_train, E_train):
            C, open_i = self._snapshot(theta, E)
            if open_ref is None:
                open_ref = set(open_i)
            elif set(open_i) != open_ref:
                raise ValueError(
                    f"open-channel set changed at E={E} (got {sorted(open_i)}, "
                    f"expected {sorted(open_ref)}): restrict E_train to a single "
                    "open-channel window, or fit a separate RBMEmulatorET per window.")
            C_list.append(C)
        self._open = sorted(open_ref)

        Csnap = np.hstack(C_list)
        self.X_r, self.singular_values, self.energy_curve = pod_basis(
            Csnap, eps_tol=eps_tol, nb_max=nb_max, return_info=True)

        self.K0_r = project_dense(self.X_r, self.base._K0_full)
        N, Nc, nb = self.N, self.Nc, self.nb
        self._Dr = np.zeros((Nc, nb, nb), dtype=complex)
        for a in range(Nc):
            Xa = self.X_r[a * N: a * N + (N - 1), :]
            self._Dr[a] = Xa.conj().T @ Xa
        return self

    @property
    def nb(self):
        return 0 if self.X_r is None else self.X_r.shape[1]

    def _K_r(self, E):
        """K_r(E) = K0_r − Σ_a Erel_a·Dr[a] + boundary-row reprojections."""
        N, R = self.N, self.R
        Kr = self.K0_r.copy()
        for a, ch in enumerate(self.base.channels):
            Kr -= (E - ch['threshold']) * self._Dr[a]
            rs = a * N
            k, eta = _channel_kinematics(ch, E)
            Brow = self.base.pref[a] * (self.base.dfj1
                                        - R * _gamma_s(ch['l'], eta, k, R) * self.base.fj1)
            test_row = self.X_r[rs + N - 1, :].conj()
            trial_blk = self.X_r[rs: rs + N, :]
            Kr += np.outer(test_row, Brow @ trial_blk)
        return Kr

    def reduced_coefficients(self, theta, E):
        Vnuc, Vcoup = eval_potential_on_mesh(self._model, theta, self.base)
        V_r = project_potential(self.X_r, Vnuc, Vcoup, self.N, self.Nc)
        sq, open_in, Fmesh, Vsc = _coulomb_source_factors(self.base, E)
        b = build_source(Vnuc, Vcoup, self.N, self.Nc, sq, open_in, Fmesh, Vsc)
        b_r = self.X_r.conj().T @ b
        return np.linalg.solve(self._K_r(E) + V_r, b_r)

    def predict_coefficients(self, theta, E):
        return self.X_r @ self.reduced_coefficients(theta, E)

    def predict(self, theta, E):
        return s_matrix_from_coeffs(self.predict_coefficients(theta, E), self.base, E)

    def predict_observables(self, theta, E):
        return observables_from_coeffs(self.predict_coefficients(theta, E), self.base, E)

    def wavefunction(self, theta, E, r_eval=None, incident=None, total=False):
        N, R = self.N, self.R
        if r_eval is None:
            r_eval = self.base.r
        r_eval = np.atleast_1d(np.asarray(r_eval, dtype=float))
        basis = lagrange_basis_at(self.base.x, self.base.lam, r_eval / R)
        C = self.predict_coefficients(theta, E)
        incident_channels = (
            [incident] if incident is not None
            else [b0 for b0, ch in enumerate(self.base.channels) if _is_open(ch, E)])
        chi = np.zeros((self.Nc, len(r_eval), len(incident_channels)), dtype=complex)
        for k_idx, b0 in enumerate(incident_channels):
            for a in range(self.Nc):
                chi[a, :, k_idx] = (basis @ C[a * N:(a + 1) * N, b0]) / np.sqrt(R)
        if total:
            for k_idx, b0 in enumerate(incident_channels):
                ch_b0 = self.base.channels[b0]
                k, eta = _channel_kinematics(ch_b0, E)
                F = _coulomb_f_at_mesh(ch_b0['l'], eta, k, r_eval, self.base._f_cache)
                chi[b0, :, k_idx] += F
        return r_eval, chi, incident_channels

    def reaction_cross_section(self, theta, E, weight=None, S=None):
        if S is None:
            S = self.predict(theta, E)
        return self.base.reaction_cross_section(E, weight=weight, S=S)

    def differential_cross_section(self, theta, E, angles, entrance=0, S=None,
                                    eta=None, spin_half=False, j_minus_channel=None):
        if S is None:
            S = self.predict(theta, E)
        return self.base.differential_cross_section(
            E, angles, entrance=entrance, S=S,
            eta=eta, spin_half=spin_half, j_minus_channel=j_minus_channel)


# --- DEIM helpers for RBMEmulatorETP ------------------------------------

def _deim_select(model, theta_train, theta_c, solver, K_diag, K_coup):
    """Complex-SVD DEIM selection for diagonal and coupling potential blocks."""
    Nc, N, r = solver.Nc, solver.N, solver.r
    Vnuc_c, Vcoup_c = eval_potential_on_mesh(model, theta_c, solver)

    rows_d = []
    for theta in theta_train:
        Vnuc, _ = eval_potential_on_mesh(model, theta, solver)
        rows_d.append(np.concatenate([Vnuc[a] - Vnuc_c[a] for a in range(Nc)]))
    delta_d = np.array(rows_d).T          # (Nc*N, n_train), complex

    U_d, s_d, _ = np.linalg.svd(delta_d, full_matrices=False)
    K_d = min(K_diag, U_d.shape[1], max(1, int(np.sum(s_d > s_d[0] * 1e-12))))
    U_K_d = U_d[:, :K_d]
    P_d = _greedy_maxvol_indices(U_K_d)[:K_d]
    try:
        U_P_inv_d = np.linalg.inv(U_K_d[P_d, :])
    except np.linalg.LinAlgError:
        U_P_inv_d = np.linalg.pinv(U_K_d[P_d, :])

    points_d, center_d = [], []
    for idx in P_d:
        a, i = divmod(int(idx), N)
        ri = r[i]
        ch = solver.channels[a]
        val = complex(model(theta_c)[0](np.array([ri]), ch)[0])
        points_d.append(('diag', a, None, ri, ch))
        center_d.append(val)

    U_K_c = P_c = U_P_inv_c = None
    points_c, center_c = [], []
    if K_coup > 0 and Nc > 1 and Vcoup_c is not None:
        pairs = [(a, b) for a in range(Nc) for b in range(a + 1, Nc)]
        if pairs:
            rows_c = []
            for theta in theta_train:
                _, Vcoup = eval_potential_on_mesh(model, theta, solver)
                rows_c.append(np.concatenate(
                    [(Vcoup[a, b] - Vcoup_c[a, b]) for a, b in pairs]))
            delta_c = np.array(rows_c).T
            U_c, s_c, _ = np.linalg.svd(delta_c, full_matrices=False)
            K_c = min(K_coup, U_c.shape[1], max(1, int(np.sum(s_c > s_c[0] * 1e-12))))
            U_K_c = U_c[:, :K_c]
            P_c = _greedy_maxvol_indices(U_K_c)[:K_c]
            try:
                U_P_inv_c = np.linalg.inv(U_K_c[P_c, :])
            except np.linalg.LinAlgError:
                U_P_inv_c = np.linalg.pinv(U_K_c[P_c, :])
            for idx in P_c:
                pair_idx, i = divmod(int(idx), N)
                a, b = pairs[pair_idx]
                ri = r[i]
                ch_a, ch_b = solver.channels[a], solver.channels[b]
                val = complex(model(theta_c)[1](np.array([ri]), ch_a, ch_b)[0])
                points_c.append(('coup', a, b, ri, (ch_a, ch_b)))
                center_c.append(val)

    points = points_d + points_c
    center = np.array(center_d + center_c, dtype=complex)
    raw_train = np.array([_evaluate_points(model, theta, points) for theta in theta_train])
    scales = np.maximum(np.std(raw_train - center[np.newaxis, :], axis=0), 1e-12)
    return points, center, scales, U_K_d, P_d, U_P_inv_d, U_K_c, P_c, U_P_inv_c


def _precompute_deim_V_r(X_r, U_K):
    """M_k = X_r.H @ diag(U_K[:,k]) @ X_r — DEIM Galerkin matrix for diagonal potential."""
    K = U_K.shape[1]
    return [X_r.conj().T @ (U_K[:, k:k+1] * X_r) for k in range(K)]


def _precompute_deim_V_r_coup(X_r, U_K_c, pairs, N):
    """DEIM Galerkin matrices for coupling-potential modes.

    U_K_c shape: (num_pairs * N, K_c).  Block pair_idx*N:(pair_idx+1)*N is
    the k-th DEIM mode for pair (a, b) = pairs[pair_idx].
    V has off-diagonal blocks V[a*N:(a+1)*N, b*N:(b+1)*N] = diag(Vcoup[a,b]);
    by Hermiticity the (b,a) block is the conjugate, so:
        M_k += Xa† diag(u_k) Xb + Xb† diag(u_k*) Xa
    """
    K = U_K_c.shape[1]
    nb = X_r.shape[1]
    result = []
    for k in range(K):
        M_k = np.zeros((nb, nb), dtype=complex)
        for pair_idx, (a, b) in enumerate(pairs):
            uk = U_K_c[pair_idx * N:(pair_idx + 1) * N, k]
            Xa = X_r[a * N:(a + 1) * N, :]
            Xb = X_r[b * N:(b + 1) * N, :]
            M_k += Xa.conj().T @ (uk[:, None] * Xb)
            M_k += Xb.conj().T @ (uk.conj()[:, None] * Xa)
        result.append(M_k)
    return result


def _precompute_deim_b_r(X_r, U_K, sq, N, Nc):
    """DB[k][b0] = (nb, N-1) matrix for the diagonal DEIM source correction."""
    sq_inner = sq[:-1]
    K = U_K.shape[1]
    DB = []
    for k in range(K):
        DBk = []
        for b0 in range(Nc):
            Xb = X_r[b0 * N: b0 * N + N - 1, :]
            uk = U_K[b0 * N: b0 * N + N - 1, k]
            DBk.append(Xb.conj().T * (uk * sq_inner))
        DB.append(DBk)
    return DB


def _precompute_deim_b_r_coup(X_r, U_K_c, pairs, N, sq):
    """DB_c[k] = list of dicts with (nb, N-1) matrices for the coupling DEIM source correction.

    For pair (a, b) with a < b and mode k:
      mat_ab: applied to F_{b0=b}[:-1], corrects b_r[:, b] for coupling a←b0=b
      mat_ba: applied to F_{b0=a}[:-1], corrects b_r[:, a] for coupling b←b0=a  (Hermitian)
    """
    sq_inner = sq[:-1]
    K = U_K_c.shape[1]
    DB_c = []
    for k in range(K):
        entries = []
        for pair_idx, (a, b) in enumerate(pairs):
            uk = U_K_c[pair_idx * N: pair_idx * N + N - 1, k]
            Xa = X_r[a * N: a * N + N - 1, :]
            Xb = X_r[b * N: b * N + N - 1, :]
            entries.append({
                'a': a, 'b': b,
                'mat_ab': Xa.conj().T * (uk * sq_inner),         # corrects b_r[:, b]
                'mat_ba': Xb.conj().T * (uk.conj() * sq_inner),  # corrects b_r[:, a]
            })
        DB_c.append(entries)
    return DB_c


def _precompute_b_r0_mats(X_r, Vnuc_c, Vcoup_c, Vsc, sq, N, Nc):
    """Central source matrices: A_diag[b0] and A_coup[a][b0], shape (nb, N-1)."""
    sq_inner = sq[:-1]
    A_diag = []
    for b0 in range(Nc):
        Xb = X_r[b0 * N: b0 * N + N - 1, :]
        veff = (Vnuc_c[b0, :-1] + Vsc[b0, :-1]) * sq_inner
        A_diag.append(Xb.conj().T * veff)
    A_coup = None
    if Vcoup_c is not None:
        A_coup = [[None] * Nc for _ in range(Nc)]
        for a in range(Nc):
            for b0 in range(Nc):
                if a == b0:
                    continue
                Xa = X_r[a * N: a * N + N - 1, :]
                vcoup = Vcoup_c[a, b0, :-1] * sq_inner
                A_coup[a][b0] = Xa.conj().T * vcoup
    return A_diag, A_coup


class RBMEmulatorETP:
    """Joint (E, theta) POD-Galerkin RBM with DEIM for V_r and b_r.

    V(r; theta) is evaluated only at K sparse DEIM points at prediction time:
        c = inv(U_P) @ delta_V_P,  V_r = V_r_c + sum_k c_k M_k
    where M_k = X_r† diag(U_K[:,k]) X_r are precomputed.  Same for b_r.
    Exact reconstruction when rank(delta_V) <= K (holds for affine potentials).
    Online cost: O(Nc·nb²) K_r + O(K·nb²) V_r + O(K·Nc·nb·N) b_r + O(nb³) solve.
    Same open-channel constraint as RBMEmulatorET.
    """

    def __init__(self, base_solver):
        self.base = base_solver
        self.N, self.R, self.Nc = base_solver.N, base_solver.R, base_solver.Nc
        self.X_r = self.K0_r = None
        self._Dr = None
        self._open = None
        self.singular_values = self.energy_curve = None
        self._model = None
        self.points = self.center = self.scales = None
        self._V_r_c = None
        self._M_k = None
        self._U_P_inv_d = None
        self._K_d = 0
        self._M_k_c = None
        self._U_P_inv_c = None
        self._K_c = 0
        self._A_diag = None
        self._A_coup = None
        self._DB_d = None
        self._DB_c = None

    def _snapshot(self, theta, E):
        K = self.base.K_matrix(E)
        Vnuc, Vcoup = eval_potential_on_mesh(self._model, theta, self.base)
        V = assemble_V_matrix(Vnuc, Vcoup, self.N, self.Nc)
        sq, open_in, Fmesh, Vsc = _coulomb_source_factors(self.base, E)
        b = build_source(Vnuc, Vcoup, self.N, self.Nc, sq, open_in, Fmesh, Vsc)
        return np.linalg.solve(K + V, b), open_in

    def fit(self, model, theta_train, E_train, theta_c,
            eps_tol=1e-8, nb_max=None, K_diag=6, K_coup=0):
        self._model = model
        theta_c = np.asarray(theta_c, dtype=float)

        C_list, open_ref = [], None
        for theta, E in zip(theta_train, E_train):
            C, open_i = self._snapshot(theta, E)
            if open_ref is None:
                open_ref = set(open_i)
            elif set(open_i) != open_ref:
                raise ValueError(
                    f"Open-channel set changed at E={E:.3f} MeV "
                    f"(got {sorted(open_i)}, expected {sorted(open_ref)}). "
                    "Restrict E_train to a single open-channel window.")
            C_list.append(C)
        self._open = sorted(open_ref)

        Csnap = np.hstack(C_list)
        self.X_r, self.singular_values, self.energy_curve = pod_basis(
            Csnap, eps_tol=eps_tol, nb_max=nb_max, return_info=True)

        self.K0_r = project_dense(self.X_r, self.base._K0_full)
        N, Nc, nb = self.N, self.Nc, self.nb
        self._Dr = np.zeros((Nc, nb, nb), dtype=complex)
        for a in range(Nc):
            Xa = self.X_r[a * N: a * N + (N - 1), :]
            self._Dr[a] = Xa.conj().T @ Xa

        (self.points, self.center, self.scales,
         U_K_d, P_d, U_P_inv_d,
         U_K_c, P_c, U_P_inv_c) = _deim_select(
            model, theta_train, theta_c, self.base, K_diag, K_coup)

        self._K_d = len(P_d)
        self._U_P_inv_d = U_P_inv_d
        self._M_k = _precompute_deim_V_r(self.X_r, U_K_d)

        self._DB_c = None
        if U_K_c is not None:
            self._K_c = len(P_c)
            self._U_P_inv_c = U_P_inv_c
            pairs = [(a, b) for a in range(Nc) for b in range(a + 1, Nc)]
            self._M_k_c = _precompute_deim_V_r_coup(self.X_r, U_K_c, pairs, N)
        else:
            self._K_c = 0
            pairs = []

        sq = np.sqrt(self.R * self.base.lam)
        Vsc = self.base._V_sc
        Vnuc_c, Vcoup_c = eval_potential_on_mesh(model, theta_c, self.base)
        self._V_r_c = project_potential(self.X_r, Vnuc_c, Vcoup_c, N, Nc)
        self._A_diag, self._A_coup = _precompute_b_r0_mats(
            self.X_r, Vnuc_c, Vcoup_c, Vsc, sq, N, Nc)
        self._DB_d = _precompute_deim_b_r(self.X_r, U_K_d, sq, N, Nc)
        if U_K_c is not None and pairs:
            self._DB_c = _precompute_deim_b_r_coup(self.X_r, U_K_c, pairs, N, sq)
        return self

    @property
    def nb(self):
        return 0 if self.X_r is None else self.X_r.shape[1]

    @property
    def n_predictors(self):
        return 0 if self.points is None else len(self.points)

    def _K_r(self, E):
        N, R = self.N, self.R
        Kr = self.K0_r.copy()
        for a, ch in enumerate(self.base.channels):
            Kr -= (E - ch['threshold']) * self._Dr[a]
            rs = a * N
            k, eta = _channel_kinematics(ch, E)
            Brow = self.base.pref[a] * (self.base.dfj1
                                        - R * _gamma_s(ch['l'], eta, k, R) * self.base.fj1)
            test_row = self.X_r[rs + N - 1, :].conj()
            trial_blk = self.X_r[rs: rs + N, :]
            Kr += np.outer(test_row, Brow @ trial_blk)
        return Kr

    def _deim_coeffs(self, p):
        K_d = self._K_d
        c_d = self._U_P_inv_d @ (p[:K_d] * self.scales[:K_d])
        c_c = None
        if self._K_c > 0:
            c_c = self._U_P_inv_c @ (p[K_d:] * self.scales[K_d:])
        return c_d, c_c

    def _b_r(self, c_d, c_c, open_in, Fmesh):
        nb, Nc = self.nb, self.Nc
        b_r = np.zeros((nb, Nc), dtype=complex)
        for b0 in open_in:
            Fb = Fmesh[b0][:-1]
            b_r[:, b0] -= self._A_diag[b0] @ Fb
            if self._A_coup is not None:
                for a in range(Nc):
                    if a != b0 and self._A_coup[a][b0] is not None:
                        b_r[:, b0] -= self._A_coup[a][b0] @ Fb
        for b0 in open_in:
            Fb = Fmesh[b0][:-1]
            for k in range(self._K_d):
                b_r[:, b0] -= c_d[k] * (self._DB_d[k][b0] @ Fb)
        if c_c is not None and self._DB_c is not None:
            for k in range(self._K_c):
                for entry in self._DB_c[k]:
                    a, b_ch = entry['a'], entry['b']
                    if b_ch in open_in:
                        b_r[:, b_ch] -= c_c[k] * (entry['mat_ab'] @ Fmesh[b_ch][:-1])
                    if a in open_in:
                        b_r[:, a] -= c_c[k] * (entry['mat_ba'] @ Fmesh[a][:-1])
        return b_r

    def reduced_coefficients(self, theta, E):
        p = predictors(self._model, theta, self.points, self.center, self.scales)
        c_d, c_c = self._deim_coeffs(p)
        V_r = self._V_r_c + sum(c_d[k] * self._M_k[k] for k in range(self._K_d))
        if c_c is not None and self._M_k_c is not None:
            V_r = V_r + sum(c_c[k] * self._M_k_c[k] for k in range(self._K_c))
        _, open_in, Fmesh, _ = _coulomb_source_factors(self.base, E)
        b_r = self._b_r(c_d, c_c, open_in, Fmesh)
        return np.linalg.solve(self._K_r(E) + V_r, b_r)

    def predict_coefficients(self, theta, E):
        return self.X_r @ self.reduced_coefficients(theta, E)

    def predict(self, theta, E):
        return s_matrix_from_coeffs(self.predict_coefficients(theta, E), self.base, E)

    def predict_observables(self, theta, E):
        return observables_from_coeffs(self.predict_coefficients(theta, E), self.base, E)

    def wavefunction(self, theta, E, r_eval=None, incident=None, total=False):
        N, R = self.N, self.R
        if r_eval is None:
            r_eval = self.base.r
        r_eval = np.atleast_1d(np.asarray(r_eval, dtype=float))
        basis = lagrange_basis_at(self.base.x, self.base.lam, r_eval / R)
        C = self.predict_coefficients(theta, E)
        incident_channels = (
            [incident] if incident is not None
            else [b0 for b0, ch in enumerate(self.base.channels) if _is_open(ch, E)])
        chi = np.zeros((self.Nc, len(r_eval), len(incident_channels)), dtype=complex)
        for k_idx, b0 in enumerate(incident_channels):
            for a in range(self.Nc):
                chi[a, :, k_idx] = (basis @ C[a * N:(a + 1) * N, b0]) / np.sqrt(R)
        if total:
            for k_idx, b0 in enumerate(incident_channels):
                ch_b0 = self.base.channels[b0]
                k, eta = _channel_kinematics(ch_b0, E)
                F = _coulomb_f_at_mesh(ch_b0['l'], eta, k, r_eval, self.base._f_cache)
                chi[b0, :, k_idx] += F
        return r_eval, chi, incident_channels

    def reaction_cross_section(self, theta, E, weight=None, S=None):
        if S is None:
            S = self.predict(theta, E)
        return self.base.reaction_cross_section(E, weight=weight, S=S)

    def differential_cross_section(self, theta, E, angles, entrance=0, S=None,
                                    eta=None, spin_half=False, j_minus_channel=None):
        if S is None:
            S = self.predict(theta, E)
        return self.base.differential_cross_section(
            E, angles, entrance=entrance, S=S,
            eta=eta, spin_half=spin_half, j_minus_channel=j_minus_channel)


# --- Windowed dispatcher: one emulator per open-channel window ---------

class WindowedEmulatorET:
    """Route each prediction to the emulator for its open-channel window.

    The open-channel count jumps at every inelastic threshold, and one
    RBMEmulatorET / ETP cannot be trained across such a boundary.

    thresholds : strictly-increasing opening energies (MeV); N of them define
        N+1 windows (window 0: E < thresholds[0]; window k: in [thresholds[k-1],
        thresholds[k]); window N: E >= thresholds[-1]).
    emulators : len(thresholds)+1 fitted emulators, in window order.
    """

    def __init__(self, thresholds, emulators):
        thresholds = np.asarray(thresholds, dtype=float)
        if thresholds.ndim != 1 or (len(thresholds) > 1 and
                                     not np.all(np.diff(thresholds) > 0)):
            raise ValueError("thresholds must be a strictly-increasing 1-D array")
        if len(emulators) != len(thresholds) + 1:
            raise ValueError(
                f"need len(thresholds)+1 = {len(thresholds) + 1} emulators, "
                f"got {len(emulators)}")
        self.thresholds = thresholds
        self.emulators  = list(emulators)

    def _window(self, E):
        """Window index for scalar energy E."""
        return int(np.searchsorted(self.thresholds, E, side='right'))

    # -- public API (mirrors RBMEmulatorET / RBMEmulatorETP) --

    def predict(self, theta, E):
        """S-matrix (Nc × Nc) at (theta, E); zeros for closed-channel rows/cols."""
        return self.emulators[self._window(E)].predict(theta, E)

    def predict_observables(self, theta, E):
        """ScatteringObservables at (theta, E)."""
        return self.emulators[self._window(E)].predict_observables(theta, E)

    def predict_coefficients(self, theta, E):
        """Full-mesh coefficient block (Nc·N × Nc_open) at (theta, E)."""
        return self.emulators[self._window(E)].predict_coefficients(theta, E)


# --- Fixed-energy predictor-point emulator: the missing 2x2 cell -------

class RBMEmulatorP(RBMEmulatorETP):
    """Fixed-energy POD-Galerkin RBM with DEIM predictor points.

                     |  full-mesh projection  |  DEIM predictor points
        -------------+------------------------+------------------------
        fixed E      |  RBMEmulator           |  RBMEmulatorP  (this)
        joint (E, θ) |  RBMEmulatorET         |  RBMEmulatorETP

    Removes RBMEmulator's per-query loop over all N interior mesh points, as
    RBMEmulatorETP does, but with E fixed: K_r and the Coulomb source factors
    are assembled once at fit time, not from their affine parts per call. This
    is the configuration published fixed-energy emulators use.
    """

    def __init__(self, base_solver, E):
        super().__init__(base_solver)
        self.E = float(E)
        self._K_r_fixed = None
        self._src = None

    def fit(self, model, theta_train, theta_c, eps_tol=1e-8, nb_max=None,
            K_diag=6, K_coup=0):
        """Train at the single energy given to the constructor.

        theta_train is a plain (N_s, p) parameter sample: unlike the ET variants
        there is no energy column to draw.
        """
        theta_train = np.asarray(theta_train, dtype=float)
        E_train = np.full(len(theta_train), self.E)
        super().fit(model, theta_train, E_train, theta_c,
                    eps_tol=eps_tol, nb_max=nb_max,
                    K_diag=K_diag, K_coup=K_coup)
        # everything below is energy-independent once E is fixed
        self._K_r_fixed = super()._K_r(self.E)
        _, open_in, Fmesh, _ = _coulomb_source_factors(self.base, self.E)
        self._src = (open_in, Fmesh)
        return self

    def _K_r(self, E=None):
        return self._K_r_fixed

    def reduced_coefficients(self, theta, E=None):
        p = predictors(self._model, theta, self.points, self.center, self.scales)
        c_d, c_c = self._deim_coeffs(p)
        V_r = self._V_r_c + sum(c_d[k] * self._M_k[k] for k in range(self._K_d))
        if c_c is not None and self._M_k_c is not None:
            V_r = V_r + sum(c_c[k] * self._M_k_c[k] for k in range(self._K_c))
        open_in, Fmesh = self._src
        b_r = self._b_r(c_d, c_c, open_in, Fmesh)
        return np.linalg.solve(self._K_r_fixed + V_r, b_r)

    def predict_coefficients(self, theta, E=None):
        return self.X_r @ self.reduced_coefficients(theta)

    def predict(self, theta, E=None):
        return s_matrix_from_coeffs(self.predict_coefficients(theta), self.base, self.E)

    def predict_observables(self, theta, E=None):
        return observables_from_coeffs(self.predict_coefficients(theta), self.base, self.E)
