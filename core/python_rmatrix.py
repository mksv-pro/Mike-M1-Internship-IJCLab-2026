"""Coupled-channel R-matrix reference solver (Lagrange-Legendre basis, Bloch BC).

Correct and deliberately unoptimised: a validation reference for the DBMM, not a
competitor, and optimising it would silently move published speedup ratios. The
avoidable costs, none fixed here:
  - compute_kinetic_matrix rebuilds an O(N^2) Python double loop per call though
    it depends only on (N, rmax, l, mu) -- as cacheable as the DBMM's K0, and
    ~89% of the n+40Ca cost;
  - compute_S_matrix inverts the full (Nc*N, Nc*N) matrix where only Cinv @ L,
    with L rank Nc, is used (196 ms vs 53 ms via np.linalg.solve on alpha+12C);
  - compute_potential_matrix evaluates the potential one mesh point at a time.
"""
import numpy as np
from functools import lru_cache

from .specfun import SF

from .constants import AMU, E2, HBARC  # noqa: F401  (defined in core.constants)


def compute_lagrange_mesh(N, rmax):
    """Gauss-Legendre quadrature mesh on [0, rmax]; returns (rmesh (N,), rweights (N,))."""
    x, w = np.polynomial.legendre.leggauss(N)
    rmesh    = 0.5 * rmax * (x + 1.0)
    rweights = 0.5 * rmax * w
    return rmesh, rweights


def compute_kinetic_matrix(mesh, rmax, channel):
    """(N,N) kinetic + centrifugal + Bloch surface matrix in MeV."""
    N         = len(mesh)
    mu        = channel['mu']
    l         = channel['l']
    prefactor = HBARC**2 / (2.0 * mu* rmax**2)

    xmesh = mesh / rmax

    Tmat = np.zeros((N, N))

    for i in range(N):
        Tmat[i,i] = ((4*N**2+4*N+3)*xmesh[i]*(1-xmesh[i])-6*xmesh[i]+1)/(3*xmesh[i]**2*(1-xmesh[i])**2)
        Tmat[i,i] += l*(l+1)/xmesh[i]**2

        for j in range(i+1,N):
            bracket=N**2+N+1+(xmesh[i]+xmesh[j]-2*xmesh[i]*xmesh[j])/(xmesh[i]-xmesh[j])**2-(1-xmesh[i])**(-1)-(1-xmesh[j])**(-1)
            Tmat[i,j]=(-1)**(i-j)/(xmesh[i]*xmesh[j]*(1-xmesh[i])*(1-xmesh[j]))**(0.5)*bracket
            Tmat[j,i]=Tmat[i,j]

    Tmat=prefactor*Tmat
    
    return Tmat


def compute_potential_matrix(mesh, channel, potential_func):
    N = len(mesh)
    Vmat = np.zeros((N, N), dtype=complex)

    for j in range(N):
        Vmat[j, j] = potential_func(mesh[j], channel)

    if channel['z1z2'] != 0:
        R_coulomb = channel.get('R_coulomb', None)
        for j in range(N):
            r = mesh[j]
            if R_coulomb is not None and r < R_coulomb:
                Vmat[j, j] += E2 * channel['z1z2'] * (3.0 - (r / R_coulomb) ** 2) / (2.0 * R_coulomb)
            else:
                Vmat[j, j] += E2 * channel['z1z2'] / r

    return Vmat


def compute_coupling_matrix(mesh, channel_i, channel_j, coupling_func):
    N = len(mesh)
    Vcoup = np.zeros((N, N), dtype=complex)

    for i in range(N):
        Vcoup[i, i] = coupling_func(mesh[i], channel_i, channel_j)

    return Vcoup


def compute_CC_ham(mesh, rmax, channels, potential_func, coupling_func):
    Nc   = len(channels)
    N    = len(mesh)
    Ntot = Nc * N

    Hmat = np.zeros((Ntot, Ntot), dtype=complex)

    for c, chan in enumerate(channels):
        s = c * N
        e = s + N
        Tmat = compute_kinetic_matrix(mesh, rmax, chan)
        Vmat = compute_potential_matrix(mesh, chan, potential_func)
        Hmat[s:e, s:e] = Tmat + Vmat

    for i in range(Nc):
        for j in range(i + 1, Nc):
            Vc = compute_coupling_matrix(mesh, channels[i], channels[j], coupling_func)
            si, ei = i * N, (i + 1) * N
            sj, ej = j * N, (j + 1) * N
            Hmat[si:ei, sj:ej] = Vc
            Hmat[sj:ej, si:ei] = Vc.T  

    return Hmat


def eval_lag_at_boundary(mesh, rmax):
    """f_j(rmax) for the Lagrange-Legendre basis; returns (N,) array."""
    N = len(mesh)
    lag_at_rhomax = np.zeros(N)
    xmesh = mesh / rmax
    for i in range(N):
        lag_at_rhomax[i] = (-1)**(i-1) * np.sqrt(1/(xmesh[i]*(1-xmesh[i])))
    return lag_at_rhomax / np.sqrt(rmax)


@lru_cache(maxsize=4096)
def _get_out_in_waves_cached(mu, l, threshold, z1z2, rmax, Ecm):
    """H^(+) = G_l + i F_l and its derivative at rho = k rmax, plus the conjugates.

    Both cases go through core.specfun; derivative analytic, from the DLMF
    33.4.4 recurrence.
    """
    E_rel = Ecm - threshold

    # Use |E_rel| so rho remains real for both open (E_rel>0) and closed (E_rel<0) channels.
    k     = np.sqrt(2.0 * mu * abs(E_rel) / HBARC**2)
    rho   = k * rmax
    eta   = mu * z1z2 * E2 / (HBARC**2 * k) if z1z2 != 0.0 else 0.0

    outwave, doutwave = SF.coulomb_hplus(l, eta, rho)

    return (complex(outwave), complex(doutwave),
            complex(np.conj(outwave)), complex(np.conj(doutwave)))


def get_out_in_waves(channel, rmax, Ecm):
    mu = float(channel['mu'])
    l = int(channel['l'])
    threshold = float(channel['threshold'])
    z1z2 = float(channel['z1z2'])
    return _get_out_in_waves_cached(mu, l, threshold, z1z2, float(rmax), float(Ecm))



def make_boundary_matrix(mesh, rmax, channels, E_cm):
    Nc      = len(channels)
    N       = len(mesh)
    Ntot    = N * Nc

    B_full = np.zeros((Ntot, Ntot), dtype=complex)

    xmesh = mesh / rmax

    # f_i(rmax) up to the common sqrt(rmax): the block below is rank one in it.
    lag = (-1.0)**np.arange(N) / np.sqrt(xmesh * (1.0 - xmesh))

    for nchan,chan in enumerate(channels):
        if E_cm < chan['threshold']:
            kappa=np.sqrt(2.0 * chan['mu'] * (chan['threshold']-E_cm) / HBARC**2)
            eta=chan['mu']*chan['z1z2'] * E2 / (HBARC**2 *kappa)
            l=chan['l']
            whittaker, dwhittaker = SF.whittaker_w(l, eta, 2*kappa*rmax)

            Bval=float(2*kappa*rmax*dwhittaker/whittaker)
            # B_ij = -Bval f_i f_j, a rank-one block.
            prefactor = HBARC**2/(2.0*chan['mu']*rmax**2)
            si, ei = N*nchan, N*(nchan+1)
            B_full[si:ei, si:ei] = (-Bval * prefactor) * np.outer(lag, lag)

    return B_full



def bloch_convergence(Hmat, max_E, channels, mesh, rmax,
                      max_iter=20, tol=1e-8):


    raw_eigenvalues, raw_eigenvectors = np.linalg.eigh(Hmat)
    numbound=len(raw_eigenvalues[raw_eigenvalues<max_E])

    converged_eigs= []
    converged_vecs= []
    for neig in range(numbound):
        trial_E=   raw_eigenvalues[neig]
        for it in range(1, max_iter + 1):
            bloch = make_boundary_matrix(mesh,rmax,channels, trial_E)

            eigvals, eigvecs = np.linalg.eigh(Hmat+bloch)

            if (abs(eigvals[neig]-trial_E) < tol):
                converged_eigs.append(eigvals[neig])
                converged_vecs.append(eigvecs[:,neig])
                break
            trial_E = eigvals[neig]
            
        if it == max_iter:
            print(f"Warning: Bloch convergence did not reach tol={tol} after {max_iter} iterations for eigenvalue {neig} (final trial energy {trial_E:.6f} MeV)")
    converged_eigs=np.array(converged_eigs)
    converged_vecs=np.array(converged_vecs).T
    return converged_eigs, converged_vecs, raw_eigenvalues, raw_eigenvectors



def compute_S_matrix(Hmat, lag_boundaries, channels, E_cm, rmax, mesh, compute_wfs=False):
    """(Nc,Nc) S-matrix from R-matrix. Per-channel energy shift Cinv=(Hmat−diag[E_cm−thresh_c])⁻¹;
    caller must add the Bloch matrix to Hmat first."""
    Nc = len(channels)
    N  = len(mesh)

    E_shift = np.repeat([E_cm - ch['threshold'] for ch in channels], N)
    Cinv    = np.linalg.inv(Hmat - np.diag(E_shift))

    open_idx = [c for c, ch in enumerate(channels) if E_cm >= ch['threshold']]
    n_open   = len(open_idx)

    if n_open == 0:
        empty = np.zeros((Nc, Nc), dtype=complex)
        return empty, (np.zeros((Nc, Nc, N), dtype=complex) if compute_wfs else None)

    kc_abs = np.array([
        np.sqrt(2.0 * ch['mu'] * abs(E_cm - ch['threshold']) / HBARC**2)
        for ch in channels
    ])
    waves = [get_out_in_waves(ch, rmax, E_cm) for ch in channels]

    Rmatrix     = np.zeros((Nc, Nc), dtype=complex)
    rightvector = np.zeros((Nc, Nc, N), dtype=complex)
    for nc, channeln in enumerate(channels):
        for mc, channelm in enumerate(channels):
            Rmatrix[nc, mc] = (lag_boundaries
                               @ Cinv[N*nc:N*(nc+1), N*mc:N*(mc+1)]
                               @ lag_boundaries)
            Rmatrix[nc, mc] *= HBARC**2 / (2.0 * np.sqrt(channeln['mu'] * channelm['mu']) * rmax)
            rightvector[nc, mc] = Cinv[N*nc:N*(nc+1), N*mc:N*(mc+1)] @ lag_boundaries

    Zmatrix  = np.zeros((n_open, n_open), dtype=complex)
    Z2matrix = np.zeros((n_open, n_open), dtype=complex)
    for i, nc in enumerate(open_idx):
        for j, mc in enumerate(open_idx):
            kc                   = kc_abs[mc]
            out, dout, inn, dinn = waves[mc]
            fac                  = (kc * rmax) ** (-0.5)
            base_out             = out if nc == mc else 0.0
            base_in              = inn if nc == mc else 0.0
            Zmatrix [i, j]       = (base_out - kc * rmax * Rmatrix[nc, mc] * dout) * fac
            Z2matrix[i, j]       = (base_in  - kc * rmax * Rmatrix[nc, mc] * dinn) * fac

    S_open = np.linalg.inv(Zmatrix) @ Z2matrix

    Smatrix = np.zeros((Nc, Nc), dtype=complex)
    for i, nc in enumerate(open_idx):
        for j, mc in enumerate(open_idx):
            Smatrix[nc, mc] = S_open[i, j]

    if not compute_wfs:
        return Smatrix, None

    wavefunction_int = np.zeros((Nc, Nc, N), dtype=complex)
    for mc, channelm in enumerate(channels):
        for ac, channela in enumerate(channels):
            kc                   = kc_abs[ac]
            out, dout, inn, dinn = waves[ac]
            amp                  = HBARC**2 / (2.0 * channela['mu'] * np.sqrt(HBARC * kc / channela['mu']))
            for nc in range(Nc):
                wavefunction_int[mc, ac] += (rightvector[nc, ac]
                                             * (dout * (1.0 if mc == ac else 0.0) - Smatrix[ac, mc] * dinn)
                                             * amp)
    return Smatrix, wavefunction_int