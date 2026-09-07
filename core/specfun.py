"""The special functions the two solvers need, behind one swappable backend.

Four quantities, and only four:

    F_l(eta, rho) on the whole mesh          the DBMM source vector [Lei eq. (6)]
    H^+_l(eta, rho) and its derivative       the open-channel boundary row
    W_{-eta,l+1/2}(z) and its derivative     the closed-channel boundary row
    sigma_l = arg Gamma(l+1+i eta)           the Coulomb phase shift

A backend implements what it can and raises `Unsupported` for the rest;
`SpecialFunctions` walks `DEFAULT_CHAIN` in order and takes the first answer.
Default chain:

    ScipyBackend    eta = 0 (F_l(0,rho) = rho j_l(rho) exactly), plus Whittaker
                    and sigma_l for every eta. scipy is already a hard dependency.
    FlintBackend    charged Coulomb via Arb (C, wheel), with rigorous error
                    bounds -- wanted at a boundary every observable passes through.
    MpmathBackend   the reference `self_check` measures against, and the fallback.

NumbaBackend (double precision, ~100x faster on a mesh than Arb) exists but is
not in DEFAULT_CHAIN: its 2e-12 error against mpmath loses four digits to Arb.
"""
import math

import numpy as np


class Unsupported(Exception):
    """Raised by a backend for a case it declines to handle."""


# --- Backends --------------------------------------------------------------

class Backend:
    """One way of evaluating the four quantities.

    Any method may raise `Unsupported` to declare a gap for the chain to fill.
    `available` is checked once, at chain construction.
    """

    name = "backend"

    #: Import name of the library whose version changes what this backend
    #: returns. None when the backend has no external dependency worth pinning.
    library = None

    @classmethod
    def available(cls):
        return True

    @classmethod
    def version(cls):
        """The library version behind this backend, or '' if there is none.

        Part of the cache stamp: the same backend name under mpmath 1.3 and 1.4
        can produce values differing in their last digits.
        """
        if not cls.library:
            return ""
        try:
            import importlib
            return getattr(importlib.import_module(cls.library), "__version__", "?")
        except Exception:
            return "?"

    def coulomb_fg(self, l, eta, rho):
        """(F_l, G_l) at a single rho > 0."""
        raise Unsupported(self.name)

    def coulomb_f(self, l, eta, rho):
        """F_l on an array of rho >= 0; returns an array of the same shape."""
        raise Unsupported(self.name)

    def whittaker_w(self, l, eta_kappa, z):
        """(W, dW/dz) for W_{-eta_kappa, l+1/2}(z), z > 0."""
        raise Unsupported(self.name)

    def coulomb_sigma(self, l, eta):
        """sigma_l = arg Gamma(l+1+i eta), vectorised over l."""
        raise Unsupported(self.name)


class ScipyBackend(Backend):
    """Closed forms. Exact for eta = 0, and the only route for W and sigma_l.

    F_l(0,rho) is the Riccati-Bessel function rho j_l(rho) identically, not an
    approximation of the charged case.
    """

    name = "scipy"
    library = "scipy"

    def coulomb_fg(self, l, eta, rho):
        if eta != 0.0:
            raise Unsupported(self.name)
        from scipy.special import spherical_jn, spherical_yn
        return (rho * spherical_jn(l, rho), -rho * spherical_yn(l, rho))

    def coulomb_f(self, l, eta, rho):
        if eta != 0.0:
            raise Unsupported(self.name)
        from scipy.special import spherical_jn
        return rho * spherical_jn(l, rho)

    def whittaker_w(self, l, eta_kappa, z):
        """W_{-eta,l+1/2}(z) = e^{-z/2} z^{l+1} U(l+1+eta, 2l+2, z).

        Analytic derivative U'(a,b,z) = -a U(a+1,b+1,z). Prefactor carried in
        logs: only the ratio dW/W reaches the solver, so a shared underflow of
        W and dW to zero would be numerically fatal.
        """
        from scipy.special import hyperu
        a, b = l + 1.0 + eta_kappa, 2.0 * l + 2.0
        if not (z > 0.0 and a > 0.0):
            raise Unsupported(self.name)

        U = hyperu(a, b, z)
        if not np.isfinite(U) or U <= 0.0:
            raise Unsupported(self.name)
        dU = -a * hyperu(a + 1.0, b + 1.0, z)
        bracket = dU + (-0.5 + (l + 1.0) / z) * U
        if not np.isfinite(dU):
            raise Unsupported(self.name)

        log_pref = -0.5 * z + (l + 1.0) * math.log(z)
        W = math.exp(log_pref + math.log(U))
        dW = (math.copysign(math.exp(log_pref + math.log(abs(bracket))), bracket)
              if bracket != 0.0 else 0.0)
        if not (np.isfinite(W) and np.isfinite(dW)):
            raise Unsupported(self.name)
        return W, dW

    def coulomb_sigma(self, l, eta):
        """sigma_l = Im loggamma(l+1+i eta), the continuous branch.

        Differs from mpmath.arg (principal branch) by a multiple of 2 pi;
        invisible downstream, where sigma_l enters only through exp(2 i sigma_l).
        """
        from scipy.special import loggamma
        return np.imag(loggamma(np.add(l, 1.0) + 1j * eta))


class FlintBackend(Backend):
    """Charged Coulomb functions through Arb (python-flint), written in C.

    Arb computes F, G, H^+ and H^- together with a rigorous error bound on each.
    `prec` is Arb's working precision in bits; 64 leaves eight guard bits over
    double precision and the cost is flat in it (~0.28 ms at 53-80 bits).
    """

    name = "flint"
    library = "flint"
    prec = 64

    @classmethod
    def available(cls):
        try:
            import flint  # noqa: F401
        except ImportError:
            return False
        return True

    def __init__(self):
        from flint import acb, ctx
        self._acb, self._ctx = acb, ctx

    def coulomb_fg(self, l, eta, rho):
        if rho <= 0.0:
            raise Unsupported(self.name)
        self._ctx.prec = self.prec
        F, G, _Hp, _Hm = self._acb(float(rho)).coulomb(int(l), float(eta))
        return float(F.real), float(G.real)

    def coulomb_f(self, l, eta, rho):
        self._ctx.prec = self.prec
        acb, li, fe = self._acb, int(l), float(eta)
        out = np.empty(len(rho), dtype=float)
        for i, r in enumerate(rho):
            r = float(r)
            out[i] = 0.0 if r <= 0.0 else float(acb(r).coulomb_f(li, fe).real)
        return out


def _build_numba_kernel():
    """JIT the mesh kernel on first use, so importing this module stays free.

    Built in a closure because `f_mesh` calls `series`: `series` must be rebound
    to its jitted self (a Dispatcher free variable) before `f_mesh` compiles.
    """
    from numba import njit

    def series(l, eta, rho, cl):
        """F by its power series. Cancellation grows like e^rho, so the caller
        uses it only below rho_split."""
        a2, a1 = 1.0, eta / (l + 1.0)
        s = a2 + a1 * rho
        p = rho
        small = 0
        for k in range(2, 200):
            a = (2.0 * eta * a1 - a2) / (k * (k + 2.0 * l + 1.0))
            p *= rho
            term = a * p
            s += term
            # Two consecutive negligible terms, not one: at eta = 0 every other
            # term vanishes identically and a single test truncates the series.
            if abs(term) < 1e-18 * abs(s):
                small += 1
                if small == 2:
                    break
            else:
                small = 0
            a2, a1 = a1, a
        return cl * rho ** l * rho * s

    def f_mesh(l, eta, rho, cl, rho_split, h):
        """F on ascending rho: series near the origin, Numerov outward beyond.

        Outward is the stable direction for the regular solution: F grows going
        out, so an admixture of G decays. The same scheme applied to G is useless.
        """
        n = rho.shape[0]
        out = np.empty(n, dtype=np.float64)

        i = 0
        while i < n and rho[i] <= rho_split:
            out[i] = series(l, eta, rho[i], cl)
            i += 1
        if i == n:
            return out

        ngrid = int(np.ceil((rho[n - 1] - rho_split) / h)) + 6
        grid = rho_split + h * np.arange(-2, ngrid)
        m = grid.shape[0]

        w = np.empty(m)
        for j in range(m):
            g = grid[j]
            w[j] = l * (l + 1.0) / (g * g) + 2.0 * eta / g - 1.0

        u = np.empty(m)
        u[0] = series(l, eta, grid[0], cl)
        u[1] = series(l, eta, grid[1], cl)
        c = h * h / 12.0
        for j in range(1, m - 1):
            u[j + 1] = (2.0 * (1.0 + 5.0 * c * w[j]) * u[j]
                        - (1.0 - c * w[j - 1]) * u[j - 1]) / (1.0 - c * w[j + 1])

        # six-point Lagrange interpolation off the uniform grid onto the mesh
        for q in range(i, n):
            x = rho[q]
            j = int((x - grid[0]) / h)
            if j < 2:
                j = 2
            if j > m - 4:
                j = m - 4
            t = (x - grid[j]) / h
            acc = 0.0
            for a in range(-2, 4):
                num, den = 1.0, 1.0
                for b in range(-2, 4):
                    if a != b:
                        num *= (t - b)
                        den *= (a - b)
                acc += u[j + a] * num / den
            out[q] = acc
        return out

    series = njit(cache=True, nogil=True)(series)
    return njit(cache=True, nogil=True)(f_mesh)


class NumbaBackend(Backend):
    """The regular function on a mesh, in double precision.

    ~0.15 us per point against Arb's ~40 us (interval bookkeeping F does not
    need). Error 2e-12 against mpmath at the default step, so NOT in
    DEFAULT_CHAIN; enable it ahead of FlintBackend by passing a custom chain to
    SpecialFunctions if the mesh cost outweighs those four digits.

    Regular function only. Outside its validated domain, or on a non-finite
    result, it raises `Unsupported` and the chain falls through to Arb.
    """

    name = "numba"
    library = "numba"
    rho_split = 2.0
    step = 0.004          # 2e-12 against mpmath; 0.01 gives 3e-10 at half the cost
    l_max = 30
    eta_max = 50.0

    _kernel = None

    @classmethod
    def available(cls):
        try:
            import numba  # noqa: F401
        except ImportError:
            return False
        return True

    def coulomb_f(self, l, eta, rho):
        if l > self.l_max or abs(eta) > self.eta_max or rho.size == 0:
            raise Unsupported(self.name)
        if np.any(rho < 0.0):
            raise Unsupported(self.name)

        if NumbaBackend._kernel is None:
            NumbaBackend._kernel = _build_numba_kernel()

        from scipy.special import gammaln, loggamma
        # C_l(eta) = 2^l e^{-pi eta/2} |Gamma(l+1+i eta)| / Gamma(2l+2)
        cl = math.exp(l * math.log(2.0) - math.pi * eta / 2.0
                      + loggamma(l + 1 + 1j * eta).real - gammaln(2 * l + 2))
        if not np.isfinite(cl):
            raise Unsupported(self.name)

        order = np.argsort(rho)
        out = np.empty_like(rho)
        out[order] = NumbaBackend._kernel(
            int(l), float(eta), np.ascontiguousarray(rho[order]),
            float(cl), float(self.rho_split), float(self.step))
        if not np.all(np.isfinite(out)):
            raise Unsupported(self.name)
        return out


class MpmathBackend(Backend):
    """The reference. Slow, complete, and the yardstick for everything else."""

    name = "mpmath"
    library = "mpmath"

    def __init__(self):
        import mpmath
        self._mp = mpmath

    def coulomb_fg(self, l, eta, rho):
        mp = self._mp
        return (float(mp.coulombf(l, eta, rho)), float(mp.coulombg(l, eta, rho)))

    def coulomb_f(self, l, eta, rho):
        mp = self._mp
        return np.array([float(mp.coulombf(l, eta, float(r))) for r in rho])

    def whittaker_w(self, l, eta_kappa, z):
        mp = self._mp

        def _W(x):
            return mp.whitw(-eta_kappa, l + 0.5, x)

        return float(_W(z)), float(mp.diff(_W, z))

    def coulomb_sigma(self, l, eta):
        mp = self._mp
        f = np.vectorize(lambda li: float(mp.arg(mp.gamma(li + 1 + 1j * eta))))
        out = f(l)
        return float(out) if np.ndim(l) == 0 else out


DEFAULT_CHAIN = (ScipyBackend, FlintBackend, MpmathBackend)


# --- The facade -----------------------------------------------------------

class SpecialFunctions:
    """Dispatches each request to the first backend in the chain that takes it.

    Holds no cache: the solvers already cache these values (keyed on rounded
    arguments, persisted by `pipeline.coulomb`).
    """

    def __init__(self, chain=None):
        self.backends = [b() for b in (chain or DEFAULT_CHAIN) if b.available()]
        if not self.backends:
            raise RuntimeError("no special-function backend available")

    @property
    def names(self):
        return [b.name for b in self.backends]

    def _try(self, method, *args):
        last = None
        for b in self.backends:
            try:
                return getattr(b, method)(*args)
            except Unsupported as exc:
                last = exc
        raise RuntimeError(f"no backend handled {method}{args!r}") from last

    # --- the four quantities ------------------------------------------------

    def coulomb_f_mesh(self, l, eta, rho):
        """F_l(eta, rho) for an array of rho, in one call (not one per point)."""
        rho = np.asarray(rho, dtype=float)
        return np.asarray(self._try("coulomb_f", int(l), float(eta), rho),
                          dtype=float)

    def coulomb_hplus(self, l, eta, rho):
        """H^+_l(eta,rho) and d/drho H^+_l, via the DLMF 33.4.4 recurrence.

        Derivative from F and G at l and l+1 (exact), not a numerical derivative.
        """
        l, eta, rho = int(l), float(eta), float(rho)
        F, G = self._try("coulomb_fg", l, eta, rho)
        F1, G1 = self._try("coulomb_fg", l + 1, eta, rho)
        corr = (l + 1.0) / rho + eta / (l + 1.0)
        factor = math.sqrt((l + 1.0) ** 2 + eta ** 2) / (l + 1.0)
        return (complex(G, F), complex(corr * G - factor * G1,
                                       corr * F - factor * F1))

    def whittaker_w(self, l, eta_kappa, z):
        return self._try("whittaker_w", int(l), float(eta_kappa), float(z))

    def coulomb_sigma(self, l, eta):
        return self._try("coulomb_sigma", l, float(eta))

    # --- validation ---------------------------------------------------------

    def self_check(self, verbose=True):
        """Measure the active chain against mpmath, and return the worst error.

        Run this after reordering `DEFAULT_CHAIN` or adding a backend.
        """
        ref = SpecialFunctions(chain=(MpmathBackend,))
        cases_c = [(0, 0.0, 1.0), (2, 1.7, 12.3), (5, 3.4, 25.0), (0, 8.0, 2.0),
                   (3, 0.0, 0.5), (1, 2.5, 7.7), (7, 1.0, 30.0), (0, 0.5, 0.2)]
        cases_w = [(0, 1.3, 4.0), (2, 2.7, 9.5), (3, 0.0, 15.0), (1, 5.0, 2.0),
                   (4, 0.4, 30.0)]
        worst = 0.0
        rows = []

        for (l, eta, rho) in cases_c:
            got, dgot = self.coulomb_hplus(l, eta, rho)
            exp, dexp = ref.coulomb_hplus(l, eta, rho)
            e = max(abs(got - exp) / abs(exp), abs(dgot - dexp) / abs(dexp))
            rows.append(("H+ ", l, eta, rho, e))
            worst = max(worst, e)

        for (l, eta, rho) in cases_c:
            grid = np.linspace(0.05, max(rho, 1.0), 12)
            got = self.coulomb_f_mesh(l, eta, grid)
            exp = ref.coulomb_f_mesh(l, eta, grid)
            e = float(np.max(np.abs(got - exp)) / np.max(np.abs(exp)))
            rows.append(("F   ", l, eta, rho, e))
            worst = max(worst, e)

        for (l, eta, z) in cases_w:
            W, dW = self.whittaker_w(l, eta, z)
            Wr, dWr = ref.whittaker_w(l, eta, z)
            e = max(abs(W - Wr) / abs(Wr), abs(dW - dWr) / abs(dWr))
            rows.append(("W   ", l, eta, z, e))
            worst = max(worst, e)

        for (l, eta, _) in cases_c:
            got = float(self.coulomb_sigma(l, eta))
            exp = float(ref.coulomb_sigma(l, eta))
            # loggamma is the continuous branch, mpmath.arg the principal one
            e = abs((got - exp + math.pi) % (2 * math.pi) - math.pi)
            rows.append(("sig ", l, eta, 0.0, e))
            worst = max(worst, e)

        if verbose:
            print(f"chain: {' -> '.join(self.names)}")
            for kind, l, eta, x, e in rows:
                print(f"  {kind} l={l:<2d} eta={eta:5.2f} x={x:5.1f}   {e:.2e}")
            print(f"  worst: {worst:.2e}")
        return worst


SF = SpecialFunctions()
