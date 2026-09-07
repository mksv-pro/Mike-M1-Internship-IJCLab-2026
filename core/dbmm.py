"""Direct Boundary Matching Method for nuclear scattering.
Reference: Jin Lei, arXiv:2512.07111 (December 2025)

M(E) c = b with the outgoing-wave BC encoded directly in the last row of M.
Open channels (E>threshold): H^+_l boundary; closed channels: Whittaker W.
"""

import numpy as np
from contextlib import contextmanager
from functools import lru_cache
from dataclasses import dataclass, field

from .specfun import SF

# Re-exported from core.constants: several modules import them from here.
from .constants import AMU, E2, HBARC  # noqa: F401

_COULOMB_MESH_CACHE: dict = {}  # (l, eta_r, kRi) -> Coulomb value; shared across solvers

#: (l, eta, k, N, R) -> F_l on the whole mesh, shared across solvers. Keyed on
#: (N, R) since _gl_mesh is a deterministic function of them. Holds the array so
#: a per-theta solver rebuild does not reassemble it point by point (~2 ms on
#: alpha+12C, 8 channels x 120 points).
_COULOMB_ARRAY_CACHE: dict = {}


def _gl_mesh(N, R):
    """Gauss-Legendre nodes x_j in (0,1), weights lam_j, physical points r_j = R*x_j."""
    xgl, wgl = np.polynomial.legendre.leggauss(N)
    x   = 0.5 * (xgl + 1.0)
    lam = 0.5 * wgl
    return x, lam, R * x


def _kinetic_matrix_dimless(N, x):
    """Dimensionless kinetic matrix T'_ij [Lei 2025, eq. (18)]; multiply by hbar²/(2μR²) for MeV."""
    i2d = np.arange(N)[:, None]
    j2d = np.arange(N)[None, :]
    xi, xj = x[:, None], x[None, :]
    diff   = xj - xi

    with np.errstate(divide='ignore', invalid='ignore'):
        T = ((-1.0)**(i2d - j2d)
             * (xi + xj - 2*xi**2)
             / (xj * diff**2)
             * np.sqrt(xj*(1 - xj) / (xi*(1 - xi)**3)))

    np.fill_diagonal(T, (N*(N+1)*x*(1-x) - 3*x + 1) / (3*x**2*(1-x)**2))
    return T


def _boundary_values(N, x):
    """hat_f_j(1) and d/dx hat_f_j|_{x=1} [Lei 2025, eqs. (24)-(25)]."""
    j    = np.arange(N)
    base = (-1.0)**(N - 1 - j) / np.sqrt(x * (1 - x))
    return base, base * (N*(N+1) - x/(1 - x))


def _legendre_P(N, z):
    """Shifted Legendre polynomial P_N(z), z = 2x - 1, vectorised over z."""
    c    = np.zeros(N + 1)
    c[N] = 1.0
    return np.polynomial.legendre.legval(z, c)


def lagrange_basis_at(x_nodes, lam_nodes, x_eval):
    """hat_f_j(x_eval) for all j [Lei 2025, eq. (14)]; returns (Np, N).
    Node coincidences use the exact property hat_f_j(x_i) = delta_ij/sqrt(lam_j) [eq. (15)]."""
    N      = len(x_nodes)
    x_eval = np.atleast_1d(np.asarray(x_eval, dtype=float))
    PN     = _legendre_P(N, 2*x_eval - 1)                    # (Np,)
    j      = np.arange(1, N + 1)
    pref   = (-1.0)**(N - j) * np.sqrt((1 - x_nodes) / x_nodes)   # (N,)

    diff = x_eval[:, None] - x_nodes[None, :]                # (Np, N)
    with np.errstate(divide='ignore', invalid='ignore'):
        F = pref[None, :] * x_eval[:, None] * PN[:, None] / diff

    on_node = np.isclose(diff, 0.0, atol=1e-10)
    if on_node.any():
        rows, cols = np.where(on_node)
        F[rows, :]    = 0.0
        F[rows, cols] = 1.0 / np.sqrt(lam_nodes[cols])

    return F



_FLUX_FACTOR = True


@contextmanager
def flux_factor_disabled():
    """Reproduce the pre-flux-factor behaviour, for the invariant study only.

    Lei (2025) eq. (43) extracts the collision matrix without the velocity
    factor documented in `channel_velocities`: correct for a single channel and
    for degenerate thresholds, wrong otherwise, but invisible to every elastic
    observable. `channel_velocities` is the single funnel through which the
    solver and both emulators reconstruct U, so disabling it here disables it
    everywhere. Produces a physically wrong U by construction -- never use it
    outside the invariant study.
    """
    global _FLUX_FACTOR
    previous = _FLUX_FACTOR
    _FLUX_FACTOR = False
    try:
        yield
    finally:
        _FLUX_FACTOR = previous


def channel_velocities(channels, E, open_channels):
    """Relative velocity v = hbar k / mu for each open channel.

    The incident wave enters the DBMM source normalised as F_beta, that is with unit
    *amplitude* rather than unit *flux*, so turning the outgoing coefficients into a
    collision matrix requires the factor sqrt(v_alpha / v_beta):

        U[a, b] = delta_ab + 2i sqrt(v_a/v_b) psi_sc[a, b](R) / H+_a .

    It is unity on the diagonal, and unity whenever two channels share a threshold,
    which is why omitting it is invisible on elastic observables and on benchmarks
    without a genuine inelastic threshold. Omitting it breaks the symmetry of U that
    time-reversal invariance requires, and breaks flux conservation for a real
    potential. Every reconstruction of U, in the solver and in each emulator, must go
    through this function; see tests/test_invariants.py.
    """
    if not _FLUX_FACTOR:
        return {a: 1.0 for a in open_channels}
    return {a: _channel_kinematics(channels[a], E)[0].real / channels[a]['mu']
            for a in open_channels}


def _channel_kinematics(ch, E):
    """(k, eta) for channel ch at energy E; k is imaginary for closed channels."""
    mu   = ch['mu']
    Erel = E - ch['threshold']
    z1z2 = ch['z1z2']

    if Erel > 0.0:                                      # open channel
        k   = np.sqrt(2*mu*Erel / HBARC**2)
        eta = (mu * z1z2 * E2 / (HBARC**2 * k)) if z1z2 != 0 else 0.0
        return k, eta

    else:                                               # closed channel
        kappa = np.sqrt(-2*mu*Erel / HBARC**2)
        k     = 1j * kappa
        eta   = (mu * z1z2 * E2 / (HBARC**2 * kappa)) if z1z2 != 0 else 0.0
        return k, eta


def _is_open(ch, E):
    return E > ch['threshold']



@lru_cache(maxsize=32768)
def _coulomb_boundary_cached(l, eta_r, kR_r):
    """H^+_l(eta,kR) and d/d(kR)H^+_l via the DLMF 33.4.4 recurrence.

    Recurrence and backend live in core.specfun; this wrapper is only the lru_cache.
    """
    return SF.coulomb_hplus(l, eta_r, kR_r)


def _coulomb_f_at_mesh(l, eta, k, r, cache):
    """F_l(eta, k r) on the mesh, evaluating the cache misses in one call.

    Do not change the key format: pipeline.coulomb pickles this dict across runs
    and a mismatched key silently discards every stored entry.
    """
    eta_r = round(float(eta), 12)
    rho   = np.array([float(k * ri) for ri in r])
    keys  = [(l, eta_r, round(float(x), 10)) for x in rho]

    out  = np.empty(len(r))
    miss = []
    for i, key in enumerate(keys):
        val = cache.get(key)
        if val is None:
            miss.append(i)
        else:
            out[i] = val

    if miss:
        idx  = np.array(miss)
        vals = SF.coulomb_f_mesh(l, eta_r, rho[idx])
        out[idx] = vals
        for i, v in zip(miss, vals):
            cache[keys[i]] = float(v)

    return out



@lru_cache(maxsize=32768)
def _whittaker_boundary_cached(l, eta_kappa, two_kappa_R):
    """Whittaker W_{-eta_kappa, l+1/2}(z) and analytic dW/dz at z=2κR (closed channel)."""
    return SF.whittaker_w(l, eta_kappa, two_kappa_R)



def _coulomb_sphere_arr(r, z1z2, R_C):
    """V_C(r) = Z1Z2e²(3-r²/R_C²)/(2R_C) for r≤R_C, else Z1Z2e²/r; point-charge if R_C=None."""
    if z1z2 == 0:
        return np.zeros_like(r)
    if R_C is None:
        return E2 * z1z2 / r
    return np.where(r <= R_C,
                    E2 * z1z2 / (2*R_C) * (3 - r**2/R_C**2),
                    E2 * z1z2 / r)


def _coulomb_sc_correction_arr(r, z1z2, R_C):
    """Short-range Coulomb correction U^S_C = V_sphere(r) - Z1Z2e²/r; enters source as U_sr [Lei eq. (6)]."""
    if z1z2 == 0 or R_C is None:
        return np.zeros(len(r))
    return np.where(r <= R_C,
                    E2*z1z2/(2*R_C)*(3 - r**2/R_C**2) - E2*z1z2/r,
                    0.0)



def _gamma_s(l, eta, k, R):
    """Log-derivative of the asymptotic wave at R: k·H^+'/H^+ (open) or 2κ·W'/W (closed)."""
    if np.isreal(k):
        k_r = float(k.real) if hasattr(k, 'real') else float(k)
        Hp, dHp = _coulomb_boundary_cached(
            l, round(float(eta), 12), round(float(k_r*R), 12))
        return k_r * dHp / Hp

    else:
        kappa     = float(abs(k))
        eta_kappa = float(eta)
        z         = 2.0 * kappa * R
        W, dWdz   = _whittaker_boundary_cached(
            l, round(eta_kappa, 12), round(z, 12))
        return 2.0 * kappa * dWdz / W



@dataclass
class ScatteringObservables:
    """Bundle of (S, f, chi_sc_R, coefficients) from one DBMM solve [Lei 2025, eqs. (27)-(29), (42)-(43)]."""
    E: float
    S: np.ndarray
    f: np.ndarray
    chi_sc_R: np.ndarray
    coefficients: np.ndarray
    open_channels: list = field(default_factory=list)


class DBMMSolver:
    """DBMM solver. K0 (Coulomb/kinetic) and V0 (nuclear) stored separately so RBMEmulator
    can call K_matrix(E) without the nuclear potential."""

    def __init__(self, N, R, channels, V_diag, V_coup=None):
        self.N        = N
        self.R        = R
        self.Nc       = len(channels)
        self.channels = channels
        self._V_diag  = V_diag
        self._V_coup  = V_coup
        self._f_cache = _COULOMB_MESH_CACHE

        x, lam, r               = _gl_mesh(N, R)
        self.x, self.lam, self.r = x, lam, r
        self.mesh                = r
        self.fj1, self.dfj1     = _boundary_values(N, x)

        # E-dependent, theta-independent pieces memoised per solver: the boundary
        # rows and the regular Coulomb function on the mesh (0.008 ms vs 0.188 ms
        # on n+40Ca). Pure functions of (channel, E, mesh), so a concurrent fill
        # from several threads is safe -- a race recomputes the same value.
        self._E_cache = {}
        self._F_cache_by_E = {}

        self.pref = np.array([HBARC**2 / (2*ch['mu']*R**2) for ch in channels])

        self._V_nuc = np.empty((self.Nc, N), dtype=complex)
        for a, ch in enumerate(channels):
            try:
                self._V_nuc[a] = np.asarray(V_diag(r, ch), dtype=complex)
            except (TypeError, ValueError):
                self._V_nuc[a] = [V_diag(ri, ch) for ri in r]

        self._V_sc = np.array([
            _coulomb_sc_correction_arr(r, ch['z1z2'], ch.get('R_coulomb'))
            for ch in channels
        ])

        if V_coup is not None:
            self._V_coup_arr = np.zeros((self.Nc, self.Nc, N), dtype=complex)
            for a in range(self.Nc):
                for b in range(self.Nc):
                    if a == b:
                        continue
                    try:
                        self._V_coup_arr[a, b] = np.asarray(
                            V_coup(r, channels[a], channels[b]), dtype=complex)
                    except (TypeError, ValueError):
                        self._V_coup_arr[a, b] = [
                            V_coup(ri, channels[a], channels[b]) for ri in r]
        else:
            self._V_coup_arr = None

        self._build_H0()

    def set_potential(self, V_diag, V_coup=None):
        """Re-evaluate the potential at a new theta, keeping everything else.

        The kinetic block, the centrifugal and Coulomb diagonals and the Lagrange
        mesh do not depend on the interaction parameters: a campaign that changes
        theta re-evaluates V on the mesh, re-assembles the potential block and
        adds it to the cached K0. Solver-level counterpart of the affine split
        the emulators exploit.
        """
        N, Nc, r = self.N, self.Nc, self.r
        self._V_diag, self._V_coup = V_diag, V_coup

        for a, ch in enumerate(self.channels):
            try:
                self._V_nuc[a] = np.asarray(V_diag(r, ch), dtype=complex)
            except (TypeError, ValueError):
                self._V_nuc[a] = [V_diag(ri, ch) for ri in r]

        if V_coup is not None:
            if self._V_coup_arr is None:
                self._V_coup_arr = np.zeros((Nc, Nc, N), dtype=complex)
            for a in range(Nc):
                for b in range(Nc):
                    if a == b:
                        continue
                    try:
                        self._V_coup_arr[a, b] = np.asarray(
                            V_coup(r, self.channels[a], self.channels[b]), dtype=complex)
                    except (TypeError, ValueError):
                        self._V_coup_arr[a, b] = [
                            V_coup(ri, self.channels[a], self.channels[b]) for ri in r]
        else:
            self._V_coup_arr = None

        self._assemble_V0()
        return self

    def _assemble_V0(self):
        """V0 from the evaluated potential arrays, then H0 = K0 + V0."""
        N, Nc = self.N, self.Nc
        V0 = np.zeros((Nc*(N-1), Nc*N), dtype=complex)
        idx = np.arange(N-1)

        for a in range(Nc):
            rs, cs = a*(N-1), a*N
            V0[rs + idx, cs + idx] += self._V_nuc[a, :-1]

        if self._V_coup_arr is not None:
            for a in range(Nc):
                for b in range(Nc):
                    if a == b:
                        continue
                    rs, cs = a*(N-1), b*N
                    V0[rs + idx, cs + idx] += self._V_coup_arr[a, b, :-1]

        self.V0 = V0
        self.H0 = self.K0 + V0
        for a in range(Nc):
            rs, hs = a*N, a*(N-1)
            self._H0_full[rs:rs+(N-1), :] = self.H0[hs:hs+(N-1), :]

    def _build_H0(self):
        N, Nc   = self.N, self.Nc
        x, r    = self.x, self.r
        T_prime = _kinetic_matrix_dimless(N, x)

        K0 = np.zeros((Nc*(N-1), Nc*N), dtype=complex)
        V0 = np.zeros((Nc*(N-1), Nc*N), dtype=complex)

        for a, ch in enumerate(self.channels):
            mu   = ch['mu']
            pref = self.pref[a]
            T_a  = pref * T_prime

            V_coul    = _coulomb_sphere_arr(r, ch['z1z2'], ch.get('R_coulomb'))
            diag_coul = HBARC**2 * ch['l']*(ch['l']+1) / (2*mu*r**2) + V_coul

            Kblk = T_a.astype(complex)
            Kblk[np.arange(N), np.arange(N)] += diag_coul
            rs, cs = a*(N-1), a*N
            K0[rs:rs+(N-1), cs:cs+N] = Kblk[:-1, :]

            idx = np.arange(N-1)
            V0[rs + idx, cs + idx] += self._V_nuc[a, :-1]

        if self._V_coup_arr is not None:
            for a in range(Nc):
                for b in range(Nc):
                    if a == b:
                        continue
                    rs, cs = a*(N-1), b*N
                    idx = np.arange(N-1)
                    V0[rs + idx, cs + idx] += self._V_coup_arr[a, b, :-1]

        self.K0 = K0
        self.V0 = V0
        self.H0 = K0 + V0

        self._K0_full = np.zeros((Nc*N, Nc*N), dtype=complex)
        self._H0_full = np.zeros((Nc*N, Nc*N), dtype=complex)
        for a in range(Nc):
            rs, hs = a*N, a*(N-1)
            self._K0_full[rs:rs+(N-1), :] = K0[hs:hs+(N-1), :]
            self._H0_full[rs:rs+(N-1), :] = self.H0[hs:hs+(N-1), :]

    def _energy_part(self, E):
        """(E_rel per channel, boundary row per channel) at E. Theta-independent.

        The only part of M(E) that is not H0: an interior-diagonal shift and one
        row per channel carrying the outgoing-wave (or Whittaker) log-derivative.
        A campaign that changes theta at fixed E computes them once.
        """
        key = float(E)
        cached = self._E_cache.get(key)
        if cached is not None:
            return cached
        R = self.R
        erel, brows = [], []
        for a, ch in enumerate(self.channels):
            erel.append(E - ch['threshold'])
            k, eta = _channel_kinematics(ch, E)
            brows.append(self.pref[a]
                         * (self.dfj1 - R * _gamma_s(ch['l'], eta, k, R) * self.fj1))
        out = (erel, brows)
        self._E_cache[key] = out
        return out

    def _F_on_mesh(self, E):
        """{entrance channel -> F_l(eta, k r) on the mesh} for the open channels.

        Two caches: the per-solver dict spares one solve the work of the last;
        the module-level _COULOMB_ARRAY_CACHE spares one solver the work of the
        last, which is what a campaign does -- a new solver per theta, with F a
        function of (l, eta, k, mesh) and no interaction parameter. Without the
        second, a solver rebuild on alpha+12C spends ~2 ms reassembling arrays
        from the point cache.
        """
        key = float(E)
        cached = self._F_cache_by_E.get(key)
        if cached is not None:
            return cached
        out = {}
        for b0, ch in enumerate(self.channels):
            if not _is_open(ch, E):
                continue
            k, eta = _channel_kinematics(ch, E)
            shared = (int(ch['l']), round(float(eta), 12), round(float(k), 12),
                      self.N, self.R)
            arr = _COULOMB_ARRAY_CACHE.get(shared)
            if arr is None:
                arr = _coulomb_f_at_mesh(ch['l'], eta, k, self.r, self._f_cache)
                # Bounded: one N-vector of doubles per entry, < 40 MB at N = 120.
                if len(_COULOMB_ARRAY_CACHE) < 32768:
                    _COULOMB_ARRAY_CACHE[shared] = arr
            out[b0] = arr
        self._F_cache_by_E[key] = out
        return out

    def K_matrix(self, E):
        """(Nc*N, Nc*N) parameter-independent DBMM matrix: K0 − E_rel·I + boundary rows."""
        N, Nc = self.N, self.Nc
        K = self._K0_full.copy()
        erel, brows = self._energy_part(E)
        idx = np.arange(N - 1)

        for a in range(Nc):
            rs = a*N
            K[rs + idx, rs + idx] -= erel[a]
            K[rs+(N-1), rs:rs+N] = brows[a]

        return K

    def V_matrix(self):
        """(Nc*N, Nc*N) nuclear potential matrix (boundary rows = 0)."""
        N, Nc = self.N, self.Nc
        Vfull = np.zeros((Nc*N, Nc*N), dtype=complex)
        for a in range(Nc):
            rs, hs = a*N, a*(N-1)
            Vfull[rs:rs+(N-1), :] = self.V0[hs:hs+(N-1), :]
        return Vfull

    def source_vector(self, E):
        return self._build_rhs(E)

    def _build_matrix(self, E):
        N, Nc = self.N, self.Nc
        M = self._H0_full.copy()
        erel, brows = self._energy_part(E)
        idx = np.arange(N - 1)

        for a in range(Nc):
            rs = a*N
            M[rs + idx, rs + idx] -= erel[a]
            M[rs+(N-1), rs:rs+N] = brows[a]

        return M

    def _build_rhs(self, E):
        N, R, Nc = self.N, self.R, self.Nc
        B        = np.zeros((Nc*N, Nc), dtype=complex)
        sq       = np.sqrt(R * self.lam)
        F_by_channel = self._F_on_mesh(E)

        for b0, ch_b0 in enumerate(self.channels):
            if not _is_open(ch_b0, E):
                continue

            F = F_by_channel[b0]

            for a in range(Nc):
                if a == b0:
                    V_eff = self._V_nuc[a] + self._V_sc[a]   # U_sr [Lei eq. (6)]
                elif self._V_coup_arr is not None:
                    V_eff = self._V_coup_arr[a, b0]
                else:
                    continue
                rs = a*N
                B[rs:rs+(N-1), b0] -= (V_eff * F * sq)[:-1]

        return B

    def coefficients(self, E):
        """Solve M(E)*C = b(E). Returns C : (Nc*N, Nc); closed incident columns are zero."""
        return np.linalg.solve(self._build_matrix(E), self._build_rhs(E))

    get_internal_solution = coefficients

    def solve_observables(self, E):
        N, R, Nc = self.N, self.R, self.Nc
        C = self.coefficients(E)
        S      = np.zeros((Nc, Nc), dtype=complex)
        f      = np.zeros((Nc, Nc), dtype=complex)
        chi_R  = np.zeros((Nc, Nc), dtype=complex)
        open_channels = [a for a, ch in enumerate(self.channels) if _is_open(ch, E)]

        vel = channel_velocities(self.channels, E, open_channels)

        for a in open_channels:
            ch     = self.channels[a]
            k, eta = _channel_kinematics(ch, E)
            Hp, _  = _coulomb_boundary_cached(
                ch['l'], round(float(eta), 12), round(float(k*R), 12))

            for b0 in open_channels:
                psi_R       = np.dot(C[a*N:(a+1)*N, b0], self.fj1) / np.sqrt(R)
                chi_R[a, b0] = psi_R
                f[a, b0]     = psi_R / (k * Hp)
                flux         = np.sqrt(vel[a] / vel[b0])
                S[a, b0]     = ((1.0 if a == b0 else 0.0)
                                + 2j * flux * k * f[a, b0])

        return ScatteringObservables(E=E, S=S, f=f, chi_sc_R=chi_R,
                                      coefficients=C, open_channels=open_channels)

    def solve(self, E):
        return self.solve_observables(E).S

    def solve_single(self, E):
        """Scalar S-matrix element for single-channel problems."""
        assert self.Nc == 1, "Use solve() for Nc > 1"
        return self.solve(E)[0, 0]

    def wavefunction(self, E, r_eval=None, incident=None, total=False):
        """Radial wave function on r_eval [Lei 2025, eq. (16)].
        total=False: scattered wave chi^sc; total=True: full psi = F + chi^sc [eq. (8)/(31)].
        Returns (r_eval, chi (Nc,Np,n_inc), incident_channels)."""
        N, R = self.N, self.R
        if r_eval is None:
            r_eval = self.r
        r_eval = np.atleast_1d(np.asarray(r_eval, dtype=float))
        basis  = lagrange_basis_at(self.x, self.lam, r_eval / R)   # (Np, N)

        C = self.coefficients(E)
        incident_channels = (
            [incident] if incident is not None
            else [b0 for b0, ch in enumerate(self.channels) if _is_open(ch, E)])

        chi = np.zeros((self.Nc, len(r_eval), len(incident_channels)), dtype=complex)
        for k_idx, b0 in enumerate(incident_channels):
            for a in range(self.Nc):
                chi[a, :, k_idx] = (basis @ C[a*N:(a+1)*N, b0]) / np.sqrt(R)

        if total:
            for k_idx, b0 in enumerate(incident_channels):
                ch_b0  = self.channels[b0]
                k, eta = _channel_kinematics(ch_b0, E)
                F      = _coulomb_f_at_mesh(ch_b0['l'], eta, k, r_eval, self._f_cache)
                chi[b0, :, k_idx] += F

        return r_eval, chi, incident_channels

    def reaction_cross_section(self, E, weight=None, S=None):
        """σ_R(b0) = (π/k²)·w·(1−Σ|S[a,b0]|²) [Lei 2025, eq. (30)]. Returns (Nc,) array."""
        if S is None:
            S = self.solve(E)
        k_by_channel = np.array([_channel_kinematics(ch, E)[0] for ch in self.channels])
        Nc = self.Nc
        w  = np.ones(Nc) if weight is None else np.asarray(weight, dtype=float)
        sigma = np.full(Nc, np.nan)
        for b0, ch in enumerate(self.channels):
            if not _is_open(ch, E):
                continue
            flux_out  = np.sum(np.abs(S[:, b0])**2)
            sigma[b0] = (np.pi / k_by_channel[b0].real**2) * w[b0] * (1.0 - flux_out)
        return sigma

    def elastic_cross_section(self, E, weight=None, entrance=None, S=None):
        """σ_el(b0) = (π/k²)·w·Σ|δ_{ab0}−S[a,b0]|²; scalar if entrance given, else (Nc,) array."""
        if S is None:
            S = self.solve(E)
        k_by_channel = np.array([_channel_kinematics(ch, E)[0] for ch in self.channels])
        Nc = self.Nc
        w  = np.ones(Nc) if weight is None else np.asarray(weight, dtype=float)

        def _sigma(b0):
            flux = np.sum(np.abs(np.eye(Nc)[:, b0] - S[:, b0])**2)
            return (np.pi / k_by_channel[b0].real**2) * w[b0] * flux

        if entrance is not None:
            return _sigma(entrance)

        sigma = np.full(Nc, np.nan)
        for b0, ch in enumerate(self.channels):
            if _is_open(ch, E):
                sigma[b0] = _sigma(b0)
        return sigma

    def differential_cross_section(self, E, angles, entrance=0, S=None,
                                    eta=None, spin_half=False, j_minus_channel=None):
        """Single l-block DCS [fm²/sr]. For a physical DCS sum amplitudes across l via the module-level function."""
        if S is None:
            S = self.solve(E)
        ch_in = self.channels[entrance]
        if not _is_open(ch_in, E):
            raise ValueError(f"Entrance channel {entrance} is closed at E={E} MeV")
        l = ch_in['l']
        k, eta_ch = _channel_kinematics(ch_in, E)
        if eta is None:
            eta = float(eta_ch)

        if not spin_half:
            S_el = complex(S[entrance, entrance])
            pw = [(l, S_el, k)]
        else:
            # identify j+ (entrance) and j- channels
            if j_minus_channel is None:
                j_minus_channel = next(
                    (i for i, ch in enumerate(self.channels)
                     if ch['l'] == l and ch.get('j', l + 0.5) < l + 0.5),
                    None)
                if j_minus_channel is None:
                    raise ValueError(
                        "spin_half=True but no j=l-1/2 channel found; "
                        "set j_minus_channel explicitly.")
            S_plus  = complex(S[entrance, entrance])
            S_minus = complex(S[j_minus_channel, j_minus_channel])
            pw = [(l, S_plus, S_minus, k)]

        return differential_cross_section(pw, angles, eta=eta, spin_half=spin_half)

    def precompute_coulomb(self, energies):
        """Pre-fill every E-dependent piece so solve() evaluates no special
        function and does no per-point cache lookup.

        Fills both per-solver memos, not just the shared value cache: leaving the
        lookup in the critical path made _build_rhs a third of the solve cost.
        """
        for E in energies:
            self._energy_part(float(E))
            self._F_on_mesh(float(E))

    def clear_coulomb_cache(self):
        """Flush the shared Coulomb F_l mesh cache and this solver's own memos.

        Both: the per-solver dicts hold arrays derived from the shared cache, so
        dropping only the shared one leaves a solver answering from stale values.
        """
        _COULOMB_MESH_CACHE.clear()
        _COULOMB_ARRAY_CACHE.clear()
        self._E_cache.clear()
        self._F_cache_by_E.clear()


def assemble_V_matrix(Vnuc, Vcoup, N, Nc):
    """(Nc*N, Nc*N) nuclear potential from mesh arrays; same layout as DBMMSolver.V_matrix()."""
    V   = np.zeros((Nc*N, Nc*N), dtype=complex)
    idx = np.arange(N - 1)
    for a in range(Nc):
        rs = a*N
        V[rs + idx, rs + idx] += Vnuc[a, :-1]
        if Vcoup is not None:
            for b in range(Nc):
                if a == b:
                    continue
                cs = b*N
                V[rs + idx, cs + idx] += Vcoup[a, b, :-1]
    return V


# --- Differential cross section (partial-wave expansion) ---------------------

def _std_legendre(l, x):
    """Standard Legendre polynomial P_l(x), vectorised over x."""
    c = np.zeros(l + 1)
    c[l] = 1.0
    return np.polynomial.legendre.legval(x, c)


def _assoc_legendre1(l, x):
    """Associated Legendre polynomial P_l^1(x) = -sqrt(1-x^2) dP_l/dx.

    Spin-flip amplitude of spin-1/2 + spin-0 scattering. Real array; l=0 -> 0.
    """
    if l < 1:
        return np.zeros_like(x, dtype=float)
    c = np.zeros(l + 1)
    c[l] = 1.0
    dPl = np.polynomial.legendre.legder(c)
    return -np.sqrt(np.maximum(1.0 - x**2, 0.0)) * np.polynomial.legendre.legval(x, dPl)


@lru_cache(maxsize=4096)
def _coulomb_phase_shift_cached(l, eta_r):
    """Coulomb phase shift σ_l = arg Γ(l+1+iη) [radians], cached."""
    return float(SF.coulomb_sigma(l, eta_r))


def _coulomb_point_amplitude(k, eta, theta, sigma0):
    """Point-charge Coulomb scattering amplitude f_C(θ) [fm].

    Diverges at θ=0; callers must exclude θ=0 when eta != 0.
    """
    sin2 = np.sin(theta / 2.0) ** 2
    phase = -eta * np.log(sin2) + 2.0 * sigma0
    return -(eta / (2.0 * k)) / sin2 * np.exp(1j * phase)


def differential_cross_section(partial_waves, angles, eta=0.0, spin_half=False):
    """dσ/dΩ [fm²/sr] from partial-wave S-matrices.
    Spinless: f=f_C+(1/2ik)Σ(2l+1)(S_l−e^{2iσ_l})P_l; spin-half: A+B formula.
    DBMM S already encodes e^{2iσ_l} via H^+_l matching, so formula subtracts e^{2iσ_l} not 1.
    partial_waves: list of (l, S_l, k) spinless or (l, S_plus, S_minus, k) spin-half."""
    angles = np.atleast_1d(np.asarray(angles, dtype=float))
    cos_t  = np.cos(angles)
    Ntheta = len(angles)

    if not partial_waves:
        return np.zeros(Ntheta)

    eta_r = round(float(eta), 12)
    # entrance k from the first entry (last element of each tuple)
    k_in  = float(np.real(partial_waves[0][-1]))

    # --- Coulomb amplitude and per-l phase shifts ---
    if eta_r != 0.0:
        sigma_by_l = {pw[0]: _coulomb_phase_shift_cached(pw[0], eta_r)
                      for pw in partial_waves}
        sigma0 = _coulomb_phase_shift_cached(0, eta_r)
        f_amp  = _coulomb_point_amplitude(k_in, eta_r, angles, sigma0).astype(complex)
    else:
        sigma_by_l = {pw[0]: 0.0 for pw in partial_waves}
        f_amp      = np.zeros(Ntheta, dtype=complex)

    if not spin_half:
        # spinless: partial_waves = [(l, S_l, k), ...]
        for l, S_l, k in partial_waves:
            Pl    = _std_legendre(l, cos_t)
            e2s   = np.exp(2j * sigma_by_l[l])
            f_amp += (2*l + 1) / (2j * float(np.real(k))) * (complex(S_l) - e2s) * Pl
        return np.abs(f_amp) ** 2

    else:
        # spin-1/2: partial_waves = [(l, S_plus, S_minus, k), ...]
        g_amp = np.zeros(Ntheta, dtype=complex)
        for l, Sp, Sm, k in partial_waves:
            k_r  = float(np.real(k))
            e2s  = np.exp(2j * sigma_by_l[l])
            Pl   = _std_legendre(l, cos_t)
            Pl1  = _assoc_legendre1(l, cos_t)
            Sp_c, Sm_c = complex(Sp), complex(Sm)
            # non-spin-flip
            f_amp += ((l + 1)*(Sp_c - e2s) + l*(Sm_c - e2s)) / (2j * k_r) * Pl
            # spin-flip  (l=0 gives Pl1=0 automatically)
            g_amp += (Sm_c - Sp_c) / (2j * k_r) * (1j * Pl1)
        return np.abs(f_amp) ** 2 + np.abs(g_amp) ** 2



def solve_single(N, R, channel, V_nuc, E):
    """Single-channel DBMM solve (stateless)."""
    return DBMMSolver(N, R, [channel],
                      lambda r, ch: V_nuc(r), V_coup=None).solve_single(E)


def solve_coupled(N, R, channels, V_diag, V_coup, E):
    """Coupled-channel DBMM solve (stateless)."""
    return DBMMSolver(N, R, channels, V_diag, V_coup).solve(E)