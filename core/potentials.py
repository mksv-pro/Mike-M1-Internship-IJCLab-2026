"""Optical-potential building blocks: radial shapes, angular-momentum coupling,
channel builder, declarative model class, and the Nc-scalable rotational preset."""

import math
import numpy as np

from .constants import AMU  # noqa: F401  (one definition, core.constants)


# --- Radial shapes ---

def woods_saxon(r, R, a):
    return 1.0 / (1.0 + np.exp((r - R) / a))


def woods_saxon_surface(r, R, a):
    """Surface-derivative form -4a*d/dr[WS]; peaks to +1 at r=R."""
    x = (r - R) / a
    ex = np.exp(x)
    return 4.0 * ex / (1.0 + ex) ** 2


def thomas_so(r, R, a):
    """Thomas spin-orbit shape -(1/r)d/dr[WS], unnormalized (no hbar/m_pi factor)."""
    x = (r - R) / a
    ex = np.exp(x)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = ex / (1.0 + ex) ** 2 / (a * np.where(r == 0, np.inf, r))
    return np.where(r == 0, 0.0, out)


def gaussian(r, beta):
    return np.exp(-(r / beta) ** 2)


# --- Angular-momentum coupling ---

def ls_coupling(l, j, s=0.5):
    return 0.5 * (j * (j + 1) - l * (l + 1) - s * (s + 1))


def tensor_SD_matrix(J):
    """S12 matrix elements for S=1 channels at total J (l=J-1 and l=J+1 pair).

        <l_-|S12|l_-> = -2(J-1)/(2J+1)
        <l_+|S12|l_+> = -2(J+2)/(2J+1)
        <l_-|S12|l_+> = <l_+|S12|l_-> = 6*sqrt(J(J+1))/(2J+1)
    """
    if J < 1:
        raise ValueError("tensor_SD_matrix requires J >= 1")
    l_minus, l_plus = J - 1, J + 1
    c_mm = -2.0 * (J - 1) / (2 * J + 1)
    c_pp = -2.0 * (J + 2) / (2 * J + 1)
    c_pm = 6.0 * np.sqrt(J * (J + 1)) / (2 * J + 1)
    return {(l_minus, l_minus): c_mm, (l_plus, l_plus): c_pp,
            (l_minus, l_plus): c_pm, (l_plus, l_minus): c_pm}


def _wigner3j_000(j1, j2, j3):
    """Wigner 3j (j1 j2 j3; 0 0 0) for non-negative integer spins.

    Closed form (Edmonds): with g = (j1+j2+j3)/2,
        = (-1)^g * sqrt[(2g-2j1)!(2g-2j2)!(2g-2j3)!/(2g+1)!] * g!/[(g-j1)!(g-j2)!(g-j3)!]
    Evaluated in log-space so it stays stable for large spins (Nc~20 in preset_rotational_band).
    """
    j1, j2, j3 = int(round(j1)), int(round(j2)), int(round(j3))
    J = j1 + j2 + j3
    if J % 2 != 0:
        return 0.0
    if not (abs(j1 - j2) <= j3 <= j1 + j2):
        return 0.0
    g = J // 2
    def lf(n): return math.lgamma(n + 1)
    log_mag = 0.5 * (lf(2*g - 2*j1) + lf(2*g - 2*j2) + lf(2*g - 2*j3) - lf(2*g + 1))
    log_mag += lf(g) - lf(g - j1) - lf(g - j2) - lf(g - j3)
    return (-1.0) ** g * math.exp(log_mag)


def rotational_coupling_factor(I, Ip, lam=2):
    """sqrt((2I+1)(2I'+1)) * (I lam I'; 0 0 0), normalized so F(0, lam)=1.

    The reduced form, valid when the orbital angular momentum is the same in
    every channel. For a genuine J-coupled calculation, where one target spin I
    generates several values of l, use `rotational_xfac`.
    """
    return np.sqrt((2*I + 1) * (2*Ip + 1)) * _wigner3j_000(I, lam, Ip)


def _wigner6j(j1, j2, j3, j4, j5, j6):
    """{j1 j2 j3 ; j4 j5 j6} by the Racah formula, non-negative integer spins.

    Evaluated in log-space term by term, which is what keeps it stable once the
    orbital angular momenta reach l = 7 and the factorials in the denominator
    run past 20!.
    """
    def tri(a, b, c):
        if a + b < c or abs(a - b) > c:
            return None
        return 0.5 * (math.lgamma(a + b - c + 1) + math.lgamma(a - b + c + 1)
                      + math.lgamma(-a + b + c + 1) - math.lgamma(a + b + c + 2))

    js = [int(round(x)) for x in (j1, j2, j3, j4, j5, j6)]
    j1, j2, j3, j4, j5, j6 = js
    deltas = [tri(j1, j2, j3), tri(j1, j5, j6), tri(j4, j2, j6), tri(j4, j5, j3)]
    if any(d is None for d in deltas):
        return 0.0
    log_delta = sum(deltas)

    lo = max(j1 + j2 + j3, j1 + j5 + j6, j4 + j2 + j6, j4 + j5 + j3)
    hi = min(j1 + j2 + j4 + j5, j2 + j3 + j5 + j6, j1 + j3 + j4 + j6)
    total = 0.0
    for t in range(lo, hi + 1):
        log_term = (math.lgamma(t + 2)
                    - math.lgamma(t - j1 - j2 - j3 + 1)
                    - math.lgamma(t - j1 - j5 - j6 + 1)
                    - math.lgamma(t - j4 - j2 - j6 + 1)
                    - math.lgamma(t - j4 - j5 - j3 + 1)
                    - math.lgamma(j1 + j2 + j4 + j5 - t + 1)
                    - math.lgamma(j2 + j3 + j5 + j6 - t + 1)
                    - math.lgamma(j1 + j3 + j4 + j6 - t + 1))
        total += (-1.0) ** t * math.exp(log_term)
    return math.exp(log_delta) * total


def rotational_channels(J, spins, thresholds):
    """The (I, l) channel list of a J-coupled rotational model.

    One target spin I generates every l with |J - I| <= l <= J + I of the right
    parity, so a three-level band at J = 3 is an eight channel problem and not a
    three channel one. Returns level_specs ready for build_channels, each
    carrying its own index so the coupling matrix can be looked up by channel.
    """
    specs = []
    for I, th in zip(spins, thresholds):
        for l in range(abs(J - I), J + I + 1, 2):
            specs.append(dict(l=l, s=0, j=J, threshold=float(th), I=int(I),
                              idx=len(specs)))
    return specs


def rotational_xfac(specs, J, lam=2):
    """Full (Nc, Nc) angular coupling matrix of a J-coupled rotational model.

    Following Rhoades-Brown et al., PRC 21 (1980) 2417, as implemented in the
    reference package of Descouvemont:

        X = (-1)^{(|l-l'|/2)} (I' lam I; 000) (l lam l'; 000) {I l J; l' I' lam}
            sqrt[(2l+1)(2l'+1)(2I+1)(2I'+1)(2lam+1) / 4pi]

    with an overall sign (-1)^{J+lam}.

    The diagonal is NOT zero. A state of spin I >= lam couples to itself, which
    is the static reorientation term, and dropping it removes real physics from
    every excited channel.
    """
    n = len(specs)
    X = np.zeros((n, n))
    for a, sa in enumerate(specs):
        la, Ia = sa['l'], sa['I']
        for b, sb in enumerate(specs):
            lb, Ib = sb['l'], sb['I']
            fac = ((2*la + 1) * (2*lb + 1) * (2*Ia + 1) * (2*Ib + 1)
                   * (2*lam + 1))
            v = (_wigner3j_000(Ib, lam, Ia) * _wigner3j_000(la, lam, lb)
                 * _wigner6j(Ia, la, J, lb, Ib, lam)
                 * math.sqrt(fac / (4.0 * math.pi)))
            if (abs(la - lb) // 2) % 2 == 1:
                v = -v
            X[a, b] = v
    if (J + lam) % 2 == 1:
        X = -X
    return X


# --- Channel builder ---

def build_channels(A1, A2, level_specs, z1z2=None, R_coulomb=None, threshold=0.0,
                    uniform_sphere_coulomb=False):
    """Build channel dicts for A1+A2, one per entry in level_specs.

    z1z2: charge product (0 = no Coulomb).
    R_coulomb: explicit Coulomb radius [fm]; None = point charge.
    uniform_sphere_coulomb: if True, auto-set R_coulomb = 1.2*A2^(1/3) fm.
    Returns (channels, mu).
    """
    mu = A1 * A2 / (A1 + A2) * AMU
    z1z2 = 0 if z1z2 is None else z1z2
    if R_coulomb is None and z1z2 != 0 and uniform_sphere_coulomb:
        R_coulomb = 1.2 * A2 ** (1 / 3)
    channels = []
    for spec in level_specs:
        ch = dict(mu=mu, z1z2=z1z2, R_coulomb=R_coulomb)
        ch['threshold'] = spec.get('threshold', threshold)
        ch['l'] = spec['l']
        ch['s'] = spec.get('s', 0.0)
        ch['j'] = spec.get('j', spec['l'])
        for k, v in spec.items():
            if k not in ch:
                ch[k] = v
        channels.append(ch)
    return channels, mu


# --- Declarative model builder ---

class OpticalPotentialModel:
    """Declare a potential as a list of terms → model(theta) callable.

    Each term: {kind, part, depth, R, a, beta, channels}.
    kinds: 'volume', 'surface', 'spin_orbit', 'gaussian', 'tensor_gaussian'.
    parts: 'real', 'imag'.
    theta is a flat 1-D array; parameter order = first encounter across terms.
    """

    def __init__(self, channels, terms, center, box_frac=0.3):
        self.channels = channels
        self.terms = terms
        self.param_names = []
        for term in terms:
            for key in ('depth', 'R', 'a', 'beta'):
                name = term.get(key)
                if name is not None and name not in self.param_names:
                    self.param_names.append(name)
        missing = set(self.param_names) - set(center)
        if missing:
            raise ValueError(f"center is missing values for parameters: {missing}")
        self.theta_c = np.array([center[p] for p in self.param_names], dtype=float)
        self.bounds = np.array([[c * (1 - box_frac), c * (1 + box_frac)] if c >= 0
                                 else [c * (1 + box_frac), c * (1 - box_frac)]
                                 for c in self.theta_c])

    def set_bounds(self, bounds_override):
        for name, (lo, hi) in bounds_override.items():
            i = self.param_names.index(name)
            self.bounds[i] = [lo, hi]
        return self

    def _theta_dict(self, theta):
        return dict(zip(self.param_names, theta))

    def model(self, theta):
        p = self._theta_dict(theta)
        terms = self.terms

        def V_diag(r, ch):
            idx = self.channels.index(ch) if ch in self.channels else None
            out = np.zeros_like(np.atleast_1d(r), dtype=complex)
            for term in terms:
                kind = term['kind']
                if kind == 'tensor_gaussian':
                    continue
                chans = term['channels']
                if idx is None or idx not in chans:
                    continue
                depth = p[term['depth']]
                sign = 1j if term.get('part') == 'imag' else 1.0
                if kind == 'volume':
                    shape = woods_saxon(r, p[term['R']], p[term['a']])
                elif kind == 'surface':
                    shape = woods_saxon_surface(r, p[term['R']], p[term['a']])
                elif kind == 'spin_orbit':
                    shape = thomas_so(r, p[term['R']], p[term['a']]) * ls_coupling(ch['l'], ch['j'], ch.get('s', 0.5))
                elif kind == 'gaussian':
                    shape = gaussian(r, p[term['beta']])
                else:
                    raise ValueError(f"unknown term kind '{kind}'")
                out = out - sign * depth * shape
            for term in terms:
                if term['kind'] != 'tensor_gaussian' or idx is None:
                    continue
                J = term['J']
                coeffs = tensor_SD_matrix(J)
                l_minus, l_plus = J - 1, J + 1
                if ch['l'] not in (l_minus, l_plus):
                    continue
                c = coeffs[(ch['l'], ch['l'])]
                if c == 0.0:
                    continue
                out = out + c * (-p[term['depth']] * gaussian(r, p[term['beta']]))
            return out

        has_coupling = any(t['kind'] == 'tensor_gaussian' for t in terms)
        if not has_coupling:
            return V_diag, None

        def V_coup(r, ch_a, ch_b):
            out = np.zeros_like(np.atleast_1d(r), dtype=complex)
            for term in terms:
                if term['kind'] != 'tensor_gaussian':
                    continue
                J = term['J']
                coeffs = tensor_SD_matrix(J)
                key = (ch_a['l'], ch_b['l'])
                if key not in coeffs or ch_a['l'] == ch_b['l']:
                    continue
                out = out + coeffs[key] * (-p[term['depth']] * gaussian(r, p[term['beta']]))
            return out

        return V_diag, V_coup


# --- Nc-scalable rotational preset --------------------------------------

def preset_rotational_band(Nc=6, E2=0.100, A1=4, Z1=2, A2=154, Z2=62):
    """alpha + even-even rotor, K^pi=0+ band, Nc channels (I=0,2,...,2*(Nc-1)).
    Sweeps DBMM/RBM/LROM cost against channel count (scaling_study.py).
    theta = (V0, W0, Wd0, R0, a0, delta2).
    Level energies: E_I = E2*I*(I+1)/6 (rigid-rotor, fixed at construction time).
    """
    level_specs = [dict(l=0, threshold=E2 * I * (I + 1) / 6.0, I=I)
                   for I in range(0, 2 * Nc, 2)]
    channels, mu = build_channels(A1=A1, A2=A2, z1z2=Z1 * Z2,
                                   uniform_sphere_coulomb=True,
                                   level_specs=level_specs)

    def model(theta):
        V0, W0, Wd0, R0, a0, delta2 = theta

        def V_diag(r, ch):
            return (-(V0 + 1j * W0) * woods_saxon(r, R0, a0)
                    - 1j * Wd0 * woods_saxon_surface(r, R0, a0))

        def V_coup(r, ch_a, ch_b):
            Ia, Ib = ch_a['I'], ch_b['I']
            if abs(Ia - Ib) != 2:
                return np.zeros_like(r, dtype=complex)
            return -delta2 * rotational_coupling_factor(Ia, Ib, lam=2) * woods_saxon_surface(r, R0, a0)

        return V_diag, V_coup

    A2f = float(A2)
    theta_c = np.array([45.0, 5.0, 8.0, 1.2 * A2f ** (1 / 3), 0.65, 3.0])
    bounds = np.array([[0.7, 1.3]] * len(theta_c)) * theta_c[:, None]
    return channels, model, theta_c, bounds
