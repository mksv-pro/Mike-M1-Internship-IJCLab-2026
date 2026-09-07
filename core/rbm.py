"""POD-Galerkin emulator (RBM) for the DBMM solver.
Reference: Jin Lei, arXiv:2512.17687 (December 2025)

RBMEmulator: fixed-energy emulator over theta.
RBMEmulatorET and RBMEmulatorETP live in rbm_et.py.
"""

import weakref

import numpy as np
from scipy.linalg import lu_factor as _lu_factor, lu_solve as _lu_solve_raw

from .dbmm import (
    channel_velocities,
    _channel_kinematics,
    _is_open,
    _coulomb_boundary_cached,
    _coulomb_f_at_mesh,
    lagrange_basis_at,
    ScatteringObservables,
    assemble_V_matrix,
)


def latin_hypercube(n_samples, bounds, seed=0):
    """Latin hypercube sample of n_samples points in the box `bounds` (d x 2)."""
    rng    = np.random.default_rng(seed)
    bounds = np.atleast_2d(np.asarray(bounds, dtype=float))
    d      = bounds.shape[0]
    cuts   = (np.arange(n_samples)[None, :] + rng.random((d, n_samples))) / n_samples
    for i in range(d):
        cuts[i] = rng.permutation(cuts[i])
    lo, hi = bounds[:, 0][:, None], bounds[:, 1][:, None]
    return (lo + cuts * (hi - lo)).T



def pod_basis(snapshots, eps_tol=1e-8, nb_max=None, return_info=False):
    """POD of a snapshot matrix (Ntot x Ns): keep modes until the captured energy
    sum(s_i^2)/sum(s^2) exceeds 1 - eps_tol [Lei 2025, eqs. (14)-(15)]."""
    U, s, _ = np.linalg.svd(snapshots, full_matrices=False)
    energy  = np.cumsum(s**2) / np.sum(s**2)
    nb      = int(np.searchsorted(energy, 1.0 - eps_tol) + 1)
    nb      = min(nb, len(s))
    if nb_max is not None:
        nb = min(nb, nb_max)
    X_r = U[:, :nb]
    return (X_r, s, energy) if return_info else X_r



def eval_potential_on_mesh(model, theta, solver):
    """Evaluate model(theta) on the mesh. Returns (Vnuc (Nc,N), Vcoup (Nc,Nc,N) or None)."""
    V_diag, V_coup = model(theta)
    r, chans = solver.r, solver.channels
    Nc, N    = solver.Nc, solver.N

    # vectorised over the mesh when V_diag accepts an array, else scalar fallback.
    Vnuc = np.empty((Nc, N), dtype=complex)
    for a, ch in enumerate(chans):
        try:
            Vnuc[a] = np.asarray(V_diag(r, ch), dtype=complex)
        except (TypeError, ValueError):
            Vnuc[a] = [V_diag(ri, ch) for ri in r]

    Vcoup = None
    if V_coup is not None:
        Vcoup = np.zeros((Nc, Nc, N), dtype=complex)
        for a in range(Nc):
            for b in range(Nc):
                if a == b:
                    continue
                try:
                    Vcoup[a, b] = np.asarray(V_coup(r, chans[a], chans[b]), dtype=complex)
                except (TypeError, ValueError):
                    Vcoup[a, b] = [V_coup(ri, chans[a], chans[b]) for ri in r]
    return Vnuc, Vcoup


#: id(basis array) -> (weakref, key, value), for the two objects derived from a
#: reduced basis and reused at every query. Keyed on array identity (a numpy
#: array has no __dict__ to hang them on); a weakref callback drops the entry
#: when the basis is collected, so a rank sweep that rebinds emu.X_r does not
#: accumulate copies.
_BLOCK_CACHE: dict = {}
_BOUNDARY_CACHE: dict = {}


def _cached(store, X_r, key, build):
    """build() memoised against the identity of X_r and a validity key."""
    ident = id(X_r)
    hit = store.get(ident)
    if hit is not None and hit[0]() is X_r and hit[1] == key:
        return hit[2]

    def evict(dead, store=store, ident=ident):
        entry = store.get(ident)
        if entry is not None and entry[0] is dead:
            del store[ident]

    value = build()
    store[ident] = (weakref.ref(X_r, evict), key, value)
    return value


def _mesh_blocks(X_r, N, Nc):
    """(Xb, Xc): the (N-1, Nc, nb) view of the basis and its conjugate.

    Entry [i, a, :] of Xb is row a*N + i of X_r. Both projections below need the
    conjugate (a full basis copy, 2.8 MB on alpha+12C at n_b = 183) at every
    query, so it is built and cached once with Xb.
    """
    def build():
        nb = X_r.shape[1]
        Xb = np.ascontiguousarray(
            np.transpose(X_r.reshape(Nc, N, nb)[:, :N - 1, :], (1, 0, 2)))
        return Xb, np.ascontiguousarray(Xb.conj())

    return _cached(_BLOCK_CACHE, X_r, (N, Nc, X_r.shape), build)


def project_potential(X_r, Vnuc, Vcoup, N, Nc):
    """V_r = Σ_{i=0}^{N-2} X_r(i)^H V(r_i) X_r(i) without forming dense V [Lei 2025, eq. (23)].

    Batched (Nc,Nc)x(Nc,nb) product then one gemm, not a triple einsum (which
    keeps the n_b^2 contraction outside BLAS-3). Agrees with the einsum form to
    1e-15, up to summation order.
    """
    Xb, Xc = _mesh_blocks(X_r, N, Nc)                # (N-1, Nc, nb) each
    V = np.zeros((N - 1, Nc, Nc), dtype=complex)
    if Vcoup is not None:
        V[:] = np.transpose(Vcoup[:, :, :N - 1], (2, 0, 1))
    idx = np.arange(Nc)
    V[:, idx, idx] = Vnuc[:, :N - 1].T
    nb = X_r.shape[1]
    Y = np.matmul(V, Xb)                             # (N-1, Nc, nb)
    return Xc.reshape(-1, nb).T @ Y.reshape(-1, nb)


def project_dense(X_r, A):
    """Generic Galerkin projection X_r^H A X_r (for K and for validation)."""
    return X_r.conj().T @ A @ X_r



def _lu_solve(lu, B):
    """Solve with a stored LU factorisation, column block by column block."""
    return _lu_solve_raw(lu, B)


def _dS_from_dC(dC, solver, E):
    """dS from a coefficient-block derivative.

    S[a,b0] = delta_{a,b0} + 2i k psi_R/(k H+), and psi_R is linear in C, so the
    derivative is the same map with the constant delta dropped.
    """
    N, R, Nc = solver.N, solver.R, solver.Nc
    dS = np.zeros((Nc, Nc), dtype=complex)
    op = [a for a, ch in enumerate(solver.channels) if _is_open(ch, E)]
    vel = channel_velocities(solver.channels, E, op)
    for a in op:
        ch = solver.channels[a]
        k, eta = _channel_kinematics(ch, E)
        Hp, _ = _coulomb_boundary_cached(ch['l'], round(float(eta), 12),
                                         round(float(k * R), 12))
        for b0 in op:
            dpsi = np.dot(dC[a * N:(a + 1) * N, b0], solver.fj1) / np.sqrt(R)
            dS[a, b0] = 2j * np.sqrt(vel[a] / vel[b0]) * dpsi / Hp
    return dS


def boundary_rows(X_r, solver):
    """Project a reduced basis onto the channel radius: G = fj1 . X_r, (Nc, nb).

    U needs the interior solution only through psi(R) = fj1 . c per channel
    block, so G @ alpha replaces the (Ntot x nb) lift-then-contract:
    O(Nc^2 nb) instead of O(Nc^2 N nb), removing the only online term linear in
    the mesh size. The full lift stays for predict_coefficients and interior
    wave functions.

    Memoised on basis identity, not stored on the emulator: a rank sweep rebinds
    emu.X_r to a slice (a new array), which correctly misses and rebuilds once
    per rank. Key carries R, since two solvers of one size on different channel
    radii share neither fj1 nor G.
    """
    N, Nc = solver.N, solver.Nc
    return _cached(
        _BOUNDARY_CACHE, X_r, (N, Nc, solver.R, X_r.shape),
        lambda: np.stack([solver.fj1 @ X_r[a * N:(a + 1) * N, :]
                          for a in range(Nc)]))


def s_matrix_from_boundary(chi_R, solver, E):
    """(Nc x Nc) S-matrix from psi(R) already projected, chi_R : (Nc, Nc).

    The tail of s_matrix_from_coeffs, for a caller that already holds psi(R).
    """
    R, Nc = solver.R, solver.Nc
    S = np.zeros((Nc, Nc), dtype=complex)
    op = [a for a, ch in enumerate(solver.channels) if _is_open(ch, E)]
    vel = channel_velocities(solver.channels, E, op)
    for a in op:
        ch = solver.channels[a]
        k, eta = _channel_kinematics(ch, E)
        Hp, _ = _coulomb_boundary_cached(ch['l'], round(float(eta), 12),
                                         round(float(k * R), 12))
        for b0 in op:
            S[a, b0] = ((1.0 if a == b0 else 0.0)
                        + 2j * np.sqrt(vel[a] / vel[b0]) * chi_R[a, b0] / Hp)
    return S


def s_matrix_from_coeffs(C, solver, E):
    """(Nc x Nc) S-matrix from a coefficient block C : (Ntot, Nc) at energy E,
    replicating DBMMSolver.solve exactly."""
    N, R, Nc = solver.N, solver.R, solver.Nc
    S = np.zeros((Nc, Nc), dtype=complex)
    op = [a for a, ch in enumerate(solver.channels) if _is_open(ch, E)]
    vel = channel_velocities(solver.channels, E, op)
    for a in op:
        ch = solver.channels[a]
        k, eta = _channel_kinematics(ch, E)
        Hp, _  = _coulomb_boundary_cached(ch['l'], round(float(eta), 12),
                                          round(float(k * R), 12))
        for b0 in op:
            psi_R    = np.dot(C[a * N:(a + 1) * N, b0], solver.fj1) / np.sqrt(R)
            S[a, b0] = ((1.0 if a == b0 else 0.0)
                        + 2j * np.sqrt(vel[a] / vel[b0]) * psi_R / Hp)
    return S


def observables_from_coeffs(C, solver, E):
    """ScatteringObservables from coefficient block C : (Ntot, Nc); RBM counterpart of solve_observables."""
    N, R, Nc = solver.N, solver.R, solver.Nc
    S, f, chi_R = (np.zeros((Nc, Nc), dtype=complex) for _ in range(3))
    open_channels = [a for a, ch in enumerate(solver.channels) if _is_open(ch, E)]

    vel = channel_velocities(solver.channels, E, open_channels)
    for a in open_channels:
        ch     = solver.channels[a]
        k, eta = _channel_kinematics(ch, E)
        Hp, _  = _coulomb_boundary_cached(ch['l'], round(float(eta), 12),
                                          round(float(k * R), 12))
        for b0 in open_channels:
            psi_R        = np.dot(C[a * N:(a + 1) * N, b0], solver.fj1) / np.sqrt(R)
            chi_R[a, b0] = psi_R
            f[a, b0]     = psi_R / (k * Hp)
            S[a, b0]     = ((1.0 if a == b0 else 0.0)
                            + 2j * np.sqrt(vel[a] / vel[b0]) * k * f[a, b0])

    return ScatteringObservables(E=E, S=S, f=f, chi_sc_R=chi_R,
                                  coefficients=C, open_channels=open_channels)



def _coulomb_source_factors(solver, E):
    """The theta-independent half of the source, from the solver's own per-energy
    memo so emulator and solve share the arrays."""
    R   = solver.R
    sq  = np.sqrt(R * solver.lam)
    Vsc = solver._V_sc
    Fmesh = solver._F_on_mesh(E)
    open_in = [b0 for b0, ch in enumerate(solver.channels) if _is_open(ch, E)]
    return sq, open_in, Fmesh, Vsc


def build_source(Vnuc, Vcoup, N, Nc, sq, open_in, Fmesh, Vsc):
    """Full-mesh source vector, shape (Nc*N, Nc). Kept for the high-fidelity path
    and for validation; the emulator uses project_source, which never forms it."""
    b = np.zeros((Nc * N, Nc), dtype=complex)
    for b0 in open_in:
        Fb = Fmesh[b0]
        for a in range(Nc):
            if a == b0:
                Veff = Vnuc[a] + Vsc[a]                     # U_sr  [Lei eq. (6)]
            elif Vcoup is not None:
                Veff = Vcoup[a, b0]
            else:
                continue
            b[a * N:a * N + (N - 1), b0] -= (Veff * Fb * sq)[:-1]
    return b


def project_source(X_r, Vnuc, Vcoup, N, Nc, sq, open_in, Fmesh, Vsc):
    """Reduced source b_r = X_r^H b, computed without ever forming b.

    b is non-zero only on the N-1 interior rows of each channel block, so
    contracting directly against the mesh blocks costs O(N*Nc*nb) and allocates
    nothing large. All entrance channels go in one gemm, not one einsum each, so
    the basis is read once, not once per open channel (16.6 ms -> 2.6 ms on
    alpha+12C at n_b = 183).
    """
    _, Xc = _mesh_blocks(X_r, N, Nc)                 # (N-1, Nc, nb)
    nb = X_r.shape[1]
    w = sq[:N - 1]
    cols = list(open_in)
    W = np.zeros((N - 1, Nc, len(cols)), dtype=complex)
    for col, b0 in enumerate(cols):
        Fb = Fmesh[b0][:N - 1] * w                   # (N-1,)
        if Vcoup is not None:
            W[:, :, col] = Vcoup[:, b0, :N - 1].T
        W[:, b0, col] = (Vnuc[b0] + Vsc[b0])[:N - 1]
        W[:, :, col] *= Fb[:, None]
    b_r = np.zeros((nb, Nc), dtype=complex)
    if cols:
        b_r[:, cols] = -(Xc.reshape(-1, nb).T @ W.reshape(-1, len(cols)))
    return b_r



class RBMEmulator:
    """POD-Galerkin emulator over optical-potential parameters at fixed energy E."""

    def __init__(self, base_solver, E):
        self.base = base_solver
        self.E    = float(E)
        self.N, self.R, self.Nc = base_solver.N, base_solver.R, base_solver.Nc

        self.K = base_solver.K_matrix(self.E)                       # theta-indep.
        (self._sq, self._open, self._F,
         self._Vsc) = _coulomb_source_factors(base_solver, self.E)  # theta-indep.

        self.X_r = self.K_r = None
        self.singular_values = self.energy_curve = None
        self._model = None

    def snapshots(self, model, theta_train):
        cols = []
        for theta in theta_train:
            Vnuc, Vcoup = eval_potential_on_mesh(model, theta, self.base)
            V = assemble_V_matrix(Vnuc, Vcoup, self.N, self.Nc)
            b = build_source(Vnuc, Vcoup, self.N, self.Nc,
                             self._sq, self._open, self._F, self._Vsc)
            cols.append(np.linalg.solve(self.K + V, b))            # (Ntot, Nc)
        return np.hstack(cols)

    def fit(self, model, theta_train, eps_tol=1e-8, nb_max=None):
        self._model = model
        C_snap = self.snapshots(model, theta_train)
        self.eps_tol = eps_tol
        self.X_r, self.singular_values, self.energy_curve = pod_basis(
            C_snap, eps_tol=eps_tol, nb_max=nb_max, return_info=True)
        self.K_r = project_dense(self.X_r, self.K)
        return self

    @property
    def nb(self):
        return 0 if self.X_r is None else self.X_r.shape[1]

    # -- prediction (no special functions) ------------------------------------
    def reduced_coefficients(self, theta):
        Vnuc, Vcoup = eval_potential_on_mesh(self._model, theta, self.base)
        V_r = project_potential(self.X_r, Vnuc, Vcoup, self.N, self.Nc)
        b_r = project_source(self.X_r, Vnuc, Vcoup, self.N, self.Nc,
                             self._sq, self._open, self._F, self._Vsc)
        return np.linalg.solve(self.K_r + V_r, b_r)

    def predict_coefficients(self, theta):
        return self.X_r @ self.reduced_coefficients(theta)

    def predict(self, theta):
        """U at theta, without ever forming the full-mesh solution.

        alpha is (nb, Nc); G_r @ alpha is the boundary value of psi in every
        channel, which is all U needs. Identical to
        s_matrix_from_coeffs(X_r @ alpha) to roundoff, and free of the term
        linear in N.
        """
        alpha = self.reduced_coefficients(theta)
        G = boundary_rows(self.X_r, self.base)
        chi_R = (G @ alpha) / np.sqrt(self.R)
        return s_matrix_from_boundary(chi_R, self.base, self.E)

    # -- parameter sensitivities ----------------------------------------------
    def reduced_sensitivities(self, theta, h=None):
        """d(alpha)/d(theta_j) for every parameter, by the sensitivity equation.

        Differentiating (K_r + V_r(theta)) alpha = b_r(theta) gives

            (K_r + V_r) d_j alpha = d_j b_r - (d_j V_r) alpha,

        so every derivative reuses the factorisation built for alpha and costs
        one triangular solve; the whole gradient costs less than a second
        forward evaluation. d_j V(r_i, theta) is a central difference on the
        mesh (the model is an arbitrary callable) -- the only approximate step,
        and a derivative of the input, not of the solve.

        Returns (alpha, dalpha) with dalpha of shape (p, nb, Nc).
        """
        theta = np.asarray(theta, dtype=float)
        p = theta.size
        Vnuc, Vcoup = eval_potential_on_mesh(self._model, theta, self.base)
        A = self.K_r + project_potential(self.X_r, Vnuc, Vcoup, self.N, self.Nc)
        b_r = project_source(self.X_r, Vnuc, Vcoup, self.N, self.Nc,
                             self._sq, self._open, self._F, self._Vsc)
        lu = _lu_factor(A)
        alpha = _lu_solve(lu, b_r)

        if h is None:
            h = 1e-6 * np.maximum(np.abs(theta), 1.0)
        h = np.broadcast_to(np.asarray(h, dtype=float), (p,))

        rhs = np.empty((p, self.nb, self.Nc), dtype=complex)
        for j in range(p):
            tp, tm = theta.copy(), theta.copy()
            tp[j] += h[j]; tm[j] -= h[j]
            Vp, Cp = eval_potential_on_mesh(self._model, tp, self.base)
            Vm, Cm = eval_potential_on_mesh(self._model, tm, self.base)
            # difference on the mesh, then project once (not both sides separately).
            inv2h = 1.0 / (2.0 * h[j])
            dVnuc = (Vp - Vm) * inv2h
            dVcoup = None if Cp is None else (Cp - Cm) * inv2h
            dV = project_potential(self.X_r, dVnuc, dVcoup, self.N, self.Nc)
            db = project_source(self.X_r, dVnuc, dVcoup, self.N, self.Nc,
                                self._sq, self._open, self._F, self._Vsc)
            rhs[j] = db - dV @ alpha

        dalpha = np.empty_like(rhs)
        for j in range(p):
            dalpha[j] = _lu_solve(lu, rhs[j])
        return alpha, dalpha

    def predict_with_gradient(self, theta, h=None):
        """(S, dS/dtheta) at theta; dS has shape (p, Nc, Nc).

        The S-matrix is linear in the coefficient block, so its derivative is the
        same linear map applied to d(alpha), lifted through X_r.
        """
        alpha, dalpha = self.reduced_sensitivities(theta, h=h)
        S = s_matrix_from_coeffs(self.X_r @ alpha, self.base, self.E)
        dS = np.empty((dalpha.shape[0],) + S.shape, dtype=complex)
        for j in range(dalpha.shape[0]):
            dS[j] = _dS_from_dC(self.X_r @ dalpha[j], self.base, self.E)
        return S, dS

    # -- a posteriori error indicator -----------------------------------------
    def residual_indicator(self, theta, return_prediction=False):
        """Relative residual ||b - M X_r a|| / ||b|| of the reduced solution.

        Classical reduced-basis a posteriori quantity: ||c - X_r a|| <=
        ||r|| / beta(theta), with beta the smallest singular value of M. beta is
        not evaluated (SCM-type bound), so this is an indicator, not a
        certificate; empirically it tracks the true relative error to a
        correlation of 0.6-0.95 in log and within a factor of a few -- enough to
        flag an out-of-envelope query and to drive a greedy basis construction.
        Cost: one (Ntot x nb) mat-vec plus a norm; no special function.
        """
        Vnuc, Vcoup = eval_potential_on_mesh(self._model, theta, self.base)
        V_r = project_potential(self.X_r, Vnuc, Vcoup, self.N, self.Nc)
        b   = build_source(Vnuc, Vcoup, self.N, self.Nc,
                           self._sq, self._open, self._F, self._Vsc)
        a   = np.linalg.solve(self.K_r + V_r, self.X_r.conj().T @ b)

        Xa   = self.X_r @ a
        V    = assemble_V_matrix(Vnuc, Vcoup, self.N, self.Nc)
        res  = b - (self.K @ Xa + V @ Xa)
        eta  = np.linalg.norm(res) / np.linalg.norm(b)
        return (eta, Xa) if return_prediction else eta

    def predict_observables(self, theta):
        return observables_from_coeffs(self.predict_coefficients(theta),
                                        self.base, self.E)

    def wavefunction(self, theta, r_eval=None, incident=None, total=False):
        N, R = self.N, self.R
        if r_eval is None:
            r_eval = self.base.r
        r_eval = np.atleast_1d(np.asarray(r_eval, dtype=float))
        basis  = lagrange_basis_at(self.base.x, self.base.lam, r_eval / R)

        C = self.predict_coefficients(theta)
        incident_channels = (
            [incident] if incident is not None
            else [b0 for b0, ch in enumerate(self.base.channels) if _is_open(ch, self.E)])

        chi = np.zeros((self.Nc, len(r_eval), len(incident_channels)), dtype=complex)
        for k_idx, b0 in enumerate(incident_channels):
            for a in range(self.Nc):
                chi[a, :, k_idx] = (basis @ C[a*N:(a+1)*N, b0]) / np.sqrt(R)

        if total:
            for k_idx, b0 in enumerate(incident_channels):
                ch_b0  = self.base.channels[b0]
                k, eta = _channel_kinematics(ch_b0, self.E)
                F      = _coulomb_f_at_mesh(ch_b0['l'], eta, k, r_eval, self.base._f_cache)
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



# --- Greedy basis construction -------------------------------------------

def greedy_fit(base_solver, model, bounds, E, n_candidates=200, n_max=60,
               tol=1e-6, eps_tol=1e-12, nb_max=None, seed=0, n_seed=3,
               verbose=False):
    """Build the reduced basis greedily instead of from a fixed Latin hypercube.

    Start from a few seed points and repeatedly add the snapshot at the
    parameter where residual_indicator is worst; scanning a large candidate pool
    costs far less than one full solve. Gives the loop a stopping criterion: ask
    for a target accuracy, get back the number of snapshots actually needed.

    Returns (emulator, theta_train, history), history the per-iteration list of
    max-indicator values.
    """
    rng = np.random.default_rng(seed)
    candidates = latin_hypercube(n_candidates, bounds, seed=seed + 1)

    # seed the basis with a few well-separated points
    idx = list(rng.choice(len(candidates), size=n_seed, replace=False))
    theta_train = candidates[idx]

    history = []
    emu = None
    while True:
        emu = RBMEmulator(base_solver, E)
        emu.fit(model, theta_train, eps_tol=eps_tol, nb_max=nb_max)

        # score every candidate that is not already in the training set
        etas = np.empty(len(candidates))
        for i, th in enumerate(candidates):
            etas[i] = -np.inf if i in idx else emu.residual_indicator(th)
        worst = int(np.argmax(etas))
        eta_max = float(etas[worst])
        history.append(eta_max)
        if verbose:
            print(f"  greedy: N_s={len(theta_train):3d}  nb={emu.nb:3d}  "
                  f"max indicator {eta_max:.3e}")

        if eta_max < tol or len(theta_train) >= n_max:
            break
        idx.append(worst)
        theta_train = candidates[idx]

    return emu, theta_train, history
