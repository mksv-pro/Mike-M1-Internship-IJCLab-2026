"""Scalar observables derived from a collision matrix, and the sampling helpers.

Compute side only. Nothing here imports matplotlib, and neither views.py nor
draw.py imports this module: figures read arrays, they do not recompute
observables.
"""
import numpy as np

from core.dbmm import DBMMSolver
from core.rbm import latin_hypercube


# --- Construction ---------------------------------------------------

def solver_for(system, theta, energies=None, N=None, R=None):
    """A DBMMSolver at `theta`, with the Coulomb functions precomputed.

    The precompute matters: the Coulomb functions are still the most expensive
    thing in a solve, and in the critical path they would make every timing a
    measurement of the special-function backend rather than the method.
    """
    channels, model, _, _ = system.preset_fn()
    V_diag, V_coup = model(theta)
    solver = DBMMSolver(N or system.N, R or system.R, channels, V_diag, V_coup)
    if energies is not None:
        solver.precompute_coulomb(np.atleast_1d(energies))
    return solver


def sample_theta(system, n, seed, include_center=True):
    """n parameter vectors by Latin hypercube, theta_c first when included.

    Every block draws its parameters here, with the seed recorded in the cache
    metadata, so a figure regenerates identically.
    """
    _, _, theta_c, bounds = system.preset_fn()
    n_draw = n - 1 if include_center else n
    drawn = latin_hypercube(n_draw, bounds, seed=seed) if n_draw > 0 \
        else np.empty((0, theta_c.size))
    return np.vstack([theta_c[None, :], drawn]) if include_center else drawn


# --- Observables --------------------------------------------------

def is_physical(S, tol=0.1):
    """True when no singular value of S exceeds 1 + tol.

    A failed emulator can return a collision matrix that creates flux; every
    cross section below is guarded by this test.
    """
    if not np.all(np.isfinite(S)):
        return False
    return float(np.linalg.svd(S, compute_uv=False)[0]) <= 1.0 + tol


def cross_section(solver, E, S, kind):
    """Elastic or reaction cross section, or NaN when S is unphysical."""
    if not is_physical(S):
        return np.nan
    if kind == "elastic":
        return float(solver.elastic_cross_section(E, S=S)[0])
    return float(solver.reaction_cross_section(E, S=S)[0])


def eigenphase_branch(S_grid, entrance=0):
    """The eigenphase branch continuously connected to `entrance`, across a sweep.

    Branches are tracked by maximising eigenvector overlap with the previous
    energy, seeded by the eigenvector with the largest projection onto the
    entrance channel. Without this tracking the eigenphases of a coupled system
    swap labels at every avoided crossing and the curve becomes a sawtooth that
    is an artefact of numpy's eigenvalue ordering, not physics.

    Returns (delta in degrees, |eigenvalue|).
    """
    S_grid = np.asarray(S_grid)
    nE, Nc, _ = S_grid.shape
    if Nc == 1:
        s = S_grid[:, 0, 0]
        return np.degrees(0.5 * np.unwrap(np.angle(s))), np.abs(s)

    w0, v0 = np.linalg.eig(S_grid[0])
    idx = int(np.argmax(np.abs(v0[entrance, :])))
    branch = np.empty(nE, dtype=complex)
    branch[0] = w0[idx]
    prev = v0[:, idx]
    for i in range(1, nE):
        w, v = np.linalg.eig(S_grid[i])
        j = int(np.argmax(np.abs(prev.conj() @ v)))
        branch[i] = w[j]
        prev = v[:, j]
    return np.degrees(0.5 * np.unwrap(np.angle(branch))), np.abs(branch)


def open_channels(system, E):
    return [a for a, ch in enumerate(system.channels)
            if E > float(ch["threshold"])]


def energy_windows(system):
    """Intervals between thresholds, over which the set of open channels is fixed.

    Crossing a threshold changes the dimension of the object being represented,
    so a single reduced basis cannot span it. RBM_ET trains one basis per window
    and a dispatcher routes each query by its energy.
    """
    lo, hi, _ = system.E_range
    cuts = [lo] + [t for t in system.thresholds if lo < t < hi] + [hi]
    eps = 1e-9 * (hi - lo)
    return [(cuts[i] + eps, cuts[i + 1] - eps) for i in range(len(cuts) - 1)]
