"""Every error quantity, defined once.

No physics, no I/O, no plotting: functions take the raw complex arrays from
compute.py and return real arrays, so a figure and its table cannot disagree
about what "the error" means.

The distinction that matters is elastic vs inelastic. The global measure
||S_emu - S||_F / ||S||_F is dominated by the order-one diagonal; off-diagonal
elements are 1e-1 to 1e-2, and an error of tens of per cent on them -- exactly
what propagates into a reaction cross section -- moves the global number by less
than 1e-3.
"""
import numpy as np

from .style import EPS_POST


# --- Structural zeros -------------------------------------------------------
# A relative error is meaningless where the reference is structurally zero: on
# n+40Ca the two channels are the j = l +- 1/2 spin-orbit blocks, solved
# together but never coupled, so every off-diagonal S is identically zero.

def coupling_mask(S_ref, rel_floor=1e-8):
    """Boolean (Nc, Nc): True where an off-diagonal element genuinely exists.

    Structural absence is judged on the ensemble, not on a single draw, because
    a physical coupling can pass through zero at one parameter point while
    being present everywhere else.
    """
    S_ref = np.asarray(S_ref)
    Nc = S_ref.shape[-1]
    off = ~np.eye(Nc, dtype=bool)
    if Nc < 2:
        return np.zeros((Nc, Nc), dtype=bool)
    mag = np.nanmax(np.abs(S_ref).reshape(-1, Nc, Nc), axis=0)
    scale = np.nanmax(mag)
    return off & (mag > rel_floor * scale)


def _safe_rel(num, den, floor=1e-30):
    return np.abs(num) / np.maximum(np.abs(den), floor)


# --- The three views of the S-matrix error ---------------------------

def err_global(S_emu, S_ref):
    """||S_emu - S_ref||_F / ||S_ref||_F over the trailing two axes.

    The measure the literature usually quotes; not a sufficient statistic here.
    """
    num = np.linalg.norm(S_emu - S_ref, axis=(-2, -1))
    den = np.linalg.norm(S_ref, axis=(-2, -1))
    return num / den


def err_elastic(S_emu, S_ref):
    """Worst relative error among the diagonal elements S_aa."""
    d_emu = np.diagonal(S_emu, axis1=-2, axis2=-1)
    d_ref = np.diagonal(S_ref, axis1=-2, axis2=-1)
    return np.nanmax(_safe_rel(d_emu - d_ref, d_ref), axis=-1)


def err_inelastic(S_emu, S_ref, mask=None):
    """Worst relative error among the off-diagonal elements that exist.

    Returns NaN where the system has no coupling at all, deliberately, so that
    a figure plotting it for n+40Ca shows a gap rather than a misleading number.
    """
    S_ref = np.asarray(S_ref)
    Nc = S_ref.shape[-1]
    if Nc < 2:
        return np.full(S_ref.shape[:-2], np.nan)
    if mask is None:
        mask = coupling_mask(S_ref)
    if not mask.any():
        return np.full(S_ref.shape[:-2], np.nan)
    rel = _safe_rel(S_emu - S_ref, S_ref)
    masked = np.where(mask, rel, np.nan)
    return np.nanmax(masked.reshape(*rel.shape[:-2], -1), axis=-1)


def err_report(S_emu, S_ref, mask=None):
    """The error a method is scored on: inelastic where coupling exists, elastic
    otherwise.

    Used by the cost-accuracy front, so a coupled system is never scored on a
    quantity that is optimistic by two orders of magnitude.
    """
    if mask is None:
        mask = coupling_mask(S_ref)
    if mask.any():
        return err_inelastic(S_emu, S_ref, mask=mask)
    return err_elastic(S_emu, S_ref)


def err_observable(x_emu, x_ref):
    """Relative error on a real observable, typically a cross section.

    Tracks the inelastic error, not the global norm.
    """
    return _safe_rel(np.asarray(x_emu) - np.asarray(x_ref), x_ref)


# --- Physical invariants: what no code comparison can check ---------------

def symmetry_violation(S):
    """max |S_ab - S_ba| over the trailing two axes.

    Time-reversal invariance requires S symmetric; a code-to-code comparison
    does not check it.
    """
    S = np.asarray(S)
    return np.nanmax(np.abs(S - np.swapaxes(S, -2, -1)), axis=(-2, -1))


def unitarity_violation(S, open_idx=None, closed_floor=1e-12):
    """||S^dagger S - I||_F, evaluated on the OPEN sub-block only.

    Meaningful only when the potential is real (absorption switched off), where
    flux conservation makes S unitary. With absorption on, S is sub-unitary by
    an amount that is physics, not error.

    Open channels only: a closed channel carries a zero row and column of S,
    contributing -1 to a diagonal entry of S^dagger S - I and a floor of exactly
    1.0 to the norm, whatever the physics. `open_idx` is taken from the solver
    when available, else detected as the channels with a non-negligible row.
    """
    S = np.asarray(S)
    Nc = S.shape[-1]
    if open_idx is None:
        scale = np.nanmax(np.abs(S))
        row = np.abs(S).reshape(-1, Nc, Nc).max(axis=0).max(axis=-1)
        open_idx = np.flatnonzero(row > max(closed_floor, 1e-10 * scale))
    open_idx = np.asarray(open_idx, dtype=int)
    if open_idx.size == 0:
        return np.full(S.shape[:-2], np.nan)
    sub = S[..., open_idx[:, None], open_idx[None, :]]
    prod = np.swapaxes(sub.conj(), -2, -1) @ sub
    return np.linalg.norm(prod - np.eye(open_idx.size), axis=(-2, -1))


# --- Summary statistics --------------------------------------------------

def residual_stats(x):
    """Summary of a residual sample spanning several decades.

    mean +- std is meaningless here (mean near the largest outlier); median and
    quantiles describe it, dispersion as std of log10.
    """
    x = np.asarray(x, dtype=float).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {k: np.nan for k in
                ("n", "median", "mean", "p95", "max", "min", "log10_std")} | {"n": 0}
    pos = x[x > 0]
    return dict(
        n=int(x.size),
        median=float(np.median(x)),
        mean=float(np.mean(x)),
        p95=float(np.percentile(x, 95)),
        max=float(np.max(x)),
        min=float(np.min(x)),
        log10_std=float(np.std(np.log10(pos))) if pos.size else np.nan,
    )


def wavefunction_residual(psi_a, psi_b):
    """max |psi_a - psi_b| normalised by max |psi_a|, per parameter draw.

    Assumes a common gauge: block_validation calibrates the R-matrix solution
    onto the DBMM normalisation (compute._rmatrix_interior) before caching,
    since the two codes fix the overall constant differently.
    """
    psi_a = np.atleast_2d(psi_a)
    psi_b = np.atleast_2d(psi_b)
    num = np.nanmax(np.abs(psi_a - psi_b), axis=-1)
    den = np.nanmax(np.abs(psi_a), axis=-1)
    return num / den


# --- Cost and accuracy, jointly ----------------------------------------

def pareto_front(cost, error):
    """Indices of the non-dominated (cost, error) points, ordered by cost.

    A configuration is kept when nothing else is both cheaper and more accurate.
    """
    cost = np.asarray(cost, dtype=float)
    error = np.asarray(error, dtype=float)
    ok = np.isfinite(cost) & np.isfinite(error)
    idx = np.argsort(np.where(ok, cost, np.inf))
    front, best = [], np.inf
    for i in idx:
        if not ok[i]:
            continue
        if error[i] < best:
            front.append(int(i))
            best = error[i]
    return np.array(front, dtype=int)


def break_even(t_offline_a, t_online_a, t_offline_b=0.0, t_online_b=None):
    """N at which method A overtakes B in T_total = T_offline + N * T_online.

    B defaults to the high-fidelity solver, which pays no offline cost. Returns
    inf when A is never cheaper, which is the honest answer for a method that
    only pays off in a regime nobody runs.
    """
    if t_online_b is None:
        raise ValueError("t_online_b is required")
    d_on = t_online_b - t_online_a
    if d_on <= 0:
        return np.inf
    return max((t_offline_a - t_offline_b) / d_on, 0.0)


def operating_point(errors, costs, tol=EPS_POST):
    """Index of the cheapest configuration whose error clears `tol`, or None.

    A method qualifies when ONE configuration is both accurate and cheap enough;
    selecting the point here rather than by eye prevents quoting one
    configuration's accuracy beside another's speed.
    """
    errors = np.asarray(errors, dtype=float)
    costs = np.asarray(costs, dtype=float)
    ok = np.isfinite(errors) & np.isfinite(costs) & (errors <= tol)
    if not ok.any():
        return None
    masked = np.where(ok, costs, np.inf)
    return int(np.argmin(masked))


def crossover(t_offline_a, t_online_a, t_offline_b, t_online_b):
    """N at which A becomes cheaper than B, or NaN when they never cross.

    Two emulators, each with an offline constant and a per-query slope: they
    cross only if the cheaper offline goes with the steeper slope. When one is
    better on both, NaN.
    """
    d_off = t_offline_b - t_offline_a
    d_on = t_online_a - t_online_b
    if d_on == 0 or d_off / d_on <= 0:
        return np.nan
    return d_off / d_on


def is_efficient(errors, costs, t_offline, t_online_ref, n_campaign, tol=EPS_POST):
    """Whether a method is efficient on a system: a SINGLE configuration clears
    `tol` AND has a break-even N* against the solver below `n_campaign`.

    Returns (verdict, detail_dict).
    """
    i = operating_point(errors, costs, tol=tol)
    if i is None:
        return False, dict(reason="no configuration reaches the tolerance",
                           index=None, n_star=np.inf)
    n_star = break_even(np.asarray(t_offline, float).ravel()[i]
                        if np.ndim(t_offline) else t_offline,
                        np.asarray(costs, float)[i],
                        t_online_b=t_online_ref)
    return bool(n_star < n_campaign), dict(
        reason="ok" if n_star < n_campaign else "break-even beyond the campaign",
        index=i, n_star=float(n_star), error=float(np.asarray(errors)[i]),
        cost=float(np.asarray(costs)[i]))
