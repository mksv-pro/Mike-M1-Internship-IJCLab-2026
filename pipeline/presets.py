"""The interactions themselves: one preset function per benchmark. Physics only;
the registry is pipeline.systems."""
import numpy as np

from core.potentials import (build_channels, gaussian, ls_coupling,
                             rotational_channels,
                             rotational_coupling_factor, rotational_xfac,
                             tensor_SD_matrix, thomas_so, woods_saxon,
                             woods_saxon_surface)


# --- Presets ---------------------------------------------------------

def preset_p12C_gaussian():
    """p+12C, single channel, real Gaussian potential (Descouvemont 2016, Ex.2). theta = (V0, beta)."""
    channels, _ = build_channels(A1=1, A2=12, z1z2=6,
                                 level_specs=[dict(l=0, s=0.5, j=0.5)])

    def model(theta):
        V0, beta = theta
        def V_diag(r, ch): return -V0 * gaussian(r, beta)
        return V_diag, None

    theta_c = np.array([50.0, 2.5])
    bounds = np.array([[35.0, 65.0], [2.0, 3.0]])
    return channels, model, theta_c, bounds


def preset_n58Ni_vibrational():
    """n+58Ni, CH89 vibrational 2-channel (0+ g.s. + 2+_1 at 1.454 MeV).
    theta = (V0, Ws, Wv, delta2). Geometry fixed at CH89 values for A=58, E_lab=15 MeV.
    Ref: Varner et al., Phys. Rep. 201 (1991) 57; Raman et al., ADNDT 78 (2001) 1 (beta2=0.18)."""
    A2 = 58
    A13 = A2 ** (1 / 3)
    Rv = 1.250 * A13 - 0.225
    Rw = 1.33 * A13 - 0.42
    Rso = 1.34 * A13 - 1.2
    a0, aw, aso = 0.690, 0.69, 0.63
    Vso = 5.9

    channels, _ = build_channels(
        A1=1, A2=A2, z1z2=0,
        level_specs=[dict(l=1, s=0.5, j=1.5, threshold=0.000),
                     dict(l=1, s=0.5, j=1.5, threshold=1.454)])

    def model(theta):
        V0, Ws, Wv, delta2 = theta

        def V_diag(r, ch):
            return (-V0 * woods_saxon(r, Rv, a0)
                    - 1j * Wv * woods_saxon(r, Rw, aw)
                    - 1j * Ws * woods_saxon_surface(r, Rw, aw)
                    - Vso * thomas_so(r, Rso, aso) * ls_coupling(ch['l'], ch['j'], ch.get('s', 0.5)))

        def V_coup(r, ch_a, ch_b):
            if ch_a['threshold'] == ch_b['threshold']:
                return np.zeros_like(r, dtype=complex)
            return -delta2 * woods_saxon_surface(r, Rw, aw)

        return V_diag, V_coup

    theta_c = np.array([47.96, 5.99, 1.74, 0.830])
    bounds = np.array([[34.0, 62.0], [3.6, 8.4], [1.0, 2.5], [0.5, 1.2]])
    return channels, model, theta_c, bounds


def preset_p40Ca_ch89():
    """p+40Ca, CH89 global OMP (Varner et al. 1991, DOI 10.1016/0370-1573(91)90039-O).
    theta = (V0, Ws, Wv). Geometry fixed at CH89 values for A=40, E_lab=25 MeV."""
    A2 = 40
    A13 = A2 ** (1 / 3)
    Rv = 1.250 * A13 - 0.225
    Rw = 1.33 * A13 - 0.42
    Rso = 1.34 * A13 - 1.2
    a0, aw, aso = 0.690, 0.69, 0.63
    Vso = 5.9

    channels, _ = build_channels(A1=1, A2=A2, z1z2=20,
                                 R_coulomb=1.238 * A13 + 0.116,
                                 level_specs=[dict(l=0, s=0.5, j=0.5)])

    def model(theta):
        V0, Ws, Wv = theta
        def V_diag(r, ch):
            return (-V0 * woods_saxon(r, Rv, a0)
                    - 1j * Wv * woods_saxon(r, Rw, aw)
                    - 1j * Ws * woods_saxon_surface(r, Rw, aw)
                    - Vso * thomas_so(r, Rso, aso) * ls_coupling(ch['l'], ch['j'], ch.get('s', 0.5)))
        return V_diag, None

    theta_c = np.array([47.80, 6.25, 1.92])
    bounds = np.array([[33.0, 62.0], [3.7, 8.8], [1.1, 2.7]])
    return channels, model, theta_c, bounds


def preset_alpha_d_SD():
    """alpha+d, S-D tensor-coupled, J^pi=1+. theta = (Vc0, Vt0)."""
    channels, _ = build_channels(A1=4, A2=2, z1z2=2,
                                 level_specs=[dict(l=0, s=1, j=1), dict(l=2, s=1, j=1)])

    def model(theta):
        Vc0, Vt0 = theta
        coeffs = tensor_SD_matrix(J=1)

        def Vc(r): return -Vc0 * gaussian(r, 1 / np.sqrt(0.2))
        def Vt(r): return -Vt0 * gaussian(r, 1 / np.sqrt(1.12))

        def V_diag(r, ch):
            return Vc(r) + coeffs[(ch['l'], ch['l'])] * Vt(r)

        def V_coup(r, ch_a, ch_b):
            return coeffs[(ch_a['l'], ch_b['l'])] * Vt(r)

        return V_diag, V_coup

    theta_c = np.array([71.979, 27.0])
    bounds = np.array([[0.7, 1.3], [0.7, 1.3]]) * theta_c[:, None]
    return channels, model, theta_c, bounds


def preset_n40Ca_woods_saxon():
    """n+40Ca(n,n), ten free parameters, on Giuliani's numbers.

    theta = (Vv0, Wv0, Wd0, Vso, Rv0, Rd0, Rso, av0, ad0, aso).

    Form and parametrisation are Eq. (3) of Odell et al. (ROSE): real and
    imaginary volume sharing one geometry, an imaginary surface term with its
    OWN radius and diffuseness, and a Thomas spin-orbit term with a third pair.

    Central values and box are those of Giuliani's LROM package
    (cross_section_recompute.py): Koning-Delaroche on 40Ca at E_lab = 14.1 MeV,
    with a +-20 per cent Latin-hypercube box. Its ten parameters are ours term
    for term and in the same order.

    Three conventions had to be reconciled with that source.

    Volume. He writes (1j*wv - vv)*WS, we write -(Vv + 1j*Wv)*WS, so Wv is his
    |wv|. Surface. He writes -(4j*ad*wd)*WS', and since
    woods_saxon_surface(r,R,a) = -4a dWS/dr identically, that IS -1j*|wd| times
    our surface form: the mapping is exact and Wd0 is his |wd|. Spin-orbit. His
    Eq. carries the Thomas factor (hbar/m_pi c)^2 = 2.0 fm^2 explicitly and our
    thomas_so does not, so his strength is multiplied by it here; without that
    the term would be half strength and dimensionally short of a fm^2.

    The sign of the two imaginary parts is measured, not read off the source
    (ambiguous): both absorptive gives |U| = 0.51 at the central point, both
    emissive gives 1.95, and only the first conserves flux.

    The two channels are the spin-orbit partners j = l +- 1/2 at l = 1, solved
    together but uncoupled -- a choice of this work, so the system isolates
    parameter dimension from channel coupling.
    """
    #: (hbar/m_pi c)^2 in fm^2, carried by the source's spin-orbit term and
    #: absent from thomas_so, so it multiplies the strength not the shape.
    THOMAS_FM2 = 2.0

    channels, _ = build_channels(A1=1, A2=40, z1z2=0,
                                 level_specs=[dict(l=1, s=0.5, j=1.5), dict(l=1, s=0.5, j=0.5)])

    def model(theta):
        Vv0, Wv0, Wd0, Vso, Rv0, Rd0, Rso, av0, ad0, aso = theta

        def V_diag(r, ch):
            return (-(Vv0 + 1j * Wv0) * woods_saxon(r, Rv0, av0)
                    - 1j * Wd0 * woods_saxon_surface(r, Rd0, ad0)
                    - Vso * thomas_so(r, Rso, aso) * ls_coupling(ch['l'], ch['j'], ch.get('s', 0.5)))

        return V_diag, None

    theta_c = np.array([46.7238, 1.72334, 7.2357, 6.1 * THOMAS_FM2,
                        4.0538, 4.4055, 1.01 * 40 ** (1.0 / 3.0),
                        0.6718, 0.5379, 0.60])
    bounds = np.array([[0.80, 1.20]] * len(theta_c)) * theta_c[:, None]
    return channels, model, theta_c, bounds


def preset_n40Ca_coupled():
    """n+40Ca, vibrational CC: elastic+inelastic (40Ca 2+_1, Ex=3.90 MeV) for
    both j=3/2 and j=1/2 spin-orbit partners — 4 channels, block-diagonal in j.
    theta = (V0, W0, Wd0, Vso0, R0, a0, Rso, aso, delta2)."""
    Ex = 3.90
    channels, _ = build_channels(
        A1=1, A2=40, z1z2=0,
        level_specs=[
            dict(l=1, s=0.5, j=1.5, threshold=0.0,  label='p3/2_el'),
            dict(l=1, s=0.5, j=1.5, threshold=Ex,   label='p3/2_inel'),
            dict(l=1, s=0.5, j=0.5, threshold=0.0,  label='p1/2_el'),
            dict(l=1, s=0.5, j=0.5, threshold=Ex,   label='p1/2_inel'),
        ])

    def model(theta):
        V0, W0, Wd0, Vso0, R0, a0, Rso, aso, delta2 = theta

        def V_diag(r, ch):
            return (-(V0 + 1j * W0) * woods_saxon(r, R0, a0)
                    - 1j * Wd0 * woods_saxon_surface(r, R0, a0)
                    - Vso0 * thomas_so(r, Rso, aso) * ls_coupling(ch['l'], ch['j'], ch.get('s', 0.5)))

        def V_coup(r, ch_a, ch_b):
            if ch_a['j'] == ch_b['j'] and ch_a['threshold'] != ch_b['threshold']:
                return -delta2 * woods_saxon_surface(r, R0, a0)
            return np.zeros_like(r, dtype=complex)

        return V_diag, V_coup

    A2 = 40
    theta_c = np.array([45.0, 5.0, 8.0, 6.0, 1.2 * A2 ** (1/3), 0.65, 1.1 * A2 ** (1/3), 0.6, 3.0])
    bounds = np.array([[0.7, 1.3]] * len(theta_c)) * theta_c[:, None]
    return channels, model, theta_c, bounds


def preset_alpha24Mg_rotational():
    """alpha+24Mg, rigid-rotor CC: 0+/2+_1/4+_1 (1.369/4.123 MeV), lambda=2,4.
    theta = (V0, W0, Wd0, R0, a0, beta2, beta4).
    Refs: Neu et al., PRC 39 (1989) 2145; Gupta et al., PLB 806 (2020) 135473."""
    E_2plus, E_4plus = 1.369, 4.123
    J = 0

    # J-coupled channel list. At J = 0 with a spin-zero projectile the triangle
    # rule gives l = I for every level, so the band is three channels carrying
    # l = 0, 2, 4 rather than three channels all at l = 0.
    specs = rotational_channels(J, [0, 2, 4], [0.0, E_2plus, E_4plus])
    X2 = rotational_xfac(specs, J=J, lam=2)
    X4 = rotational_xfac(specs, J=J, lam=4)

    channels, _ = build_channels(
        A1=4, A2=24, z1z2=2 * 12,
        uniform_sphere_coulomb=True,
        level_specs=specs)

    A2 = 24

    def model(theta):
        V0, W0, Wd0, R0, a0, beta2, beta4 = theta

        def _xfac(a, b, R0):
            """The two multipoles, each with its own deformation length."""
            return X2[a, b] * beta2 * R0 + X4[a, b] * beta4 * R0

        def V_diag(r, ch):
            i = ch['idx']
            # The diagonal of X does not vanish for a level with 2I >= lambda:
            # that entry is the static reorientation, and it is real physics.
            return (-(V0 + 1j * W0) * woods_saxon(r, R0, a0)
                    - 1j * Wd0 * woods_saxon_surface(r, R0, a0)
                    - _xfac(i, i, R0) * woods_saxon_surface(r, R0, a0))

        def V_coup(r, ch_a, ch_b):
            return (-_xfac(ch_a['idx'], ch_b['idx'], R0)
                    * woods_saxon_surface(r, R0, a0))

        return V_diag, V_coup

    A13 = A2 ** (1 / 3)
    R0_c = 1.2 * A13
    theta_c = np.array([100.0, 10.0, 20.0, R0_c, 0.65, 0.43, -0.11])
    bounds = np.array([
        [ 70.0, 140.0],
        [  5.0,  20.0],
        [  8.0,  30.0],
        [R0_c * 0.85, R0_c * 1.15],
        [  0.50,   0.80],
        [  0.35,   0.51],
        [ -0.17,  -0.05],
    ])
    return channels, model, theta_c, bounds


def preset_n238U_rotational():
    """n+238U, simplified DCCOM rotational CC: 0+/2+/4+/6+/8+ ground band, lambda=2,4.
    theta = (V0, W0, Wd0, R0, a0, beta2, beta4). Fixed-energy WS (no dispersion).
    Depths are round values of the order of the reference; the radius uses the CH89
    form 1.250*A^(1/3) - 0.225 and not the reference's own rR = 1.245 fm.
    Refs: Soukhovitskii & Capote, J. Phys. G 30 (2004) 905; FRDM beta2=0.215, beta4=0.093."""
    thresholds = [0.0, 0.04491, 0.14803, 0.30722, 0.51831]
    spins_I    = [0,   2,       4,       6,       8      ]
    J = 0

    # As for alpha+24Mg: J-coupled channels, so l = I along the band, and the
    # full (l, I, J, lambda) coefficient, whose diagonal is the reorientation.
    # The projectile is treated as spinless, which is the standard spin-zero
    # rigid-rotor coupled-channel model and keeps the block at five channels.
    specs = rotational_channels(J, spins_I, thresholds)
    X2 = rotational_xfac(specs, J=J, lam=2)
    X4 = rotational_xfac(specs, J=J, lam=4)

    channels, _ = build_channels(A1=1, A2=238, z1z2=0, level_specs=specs)

    A2 = 238

    def model(theta):
        V0, W0, Wd0, R0, a0, beta2, beta4 = theta

        def _xfac(a, b, R0):
            return X2[a, b] * beta2 * R0 + X4[a, b] * beta4 * R0

        def V_diag(r, ch):
            i = ch['idx']
            return (-(V0 + 1j * W0) * woods_saxon(r, R0, a0)
                    - 1j * Wd0 * woods_saxon_surface(r, R0, a0)
                    - _xfac(i, i, R0) * woods_saxon_surface(r, R0, a0))

        def V_coup(r, ch_a, ch_b):
            return (-_xfac(ch_a['idx'], ch_b['idx'], R0)
                    * woods_saxon_surface(r, R0, a0))

        return V_diag, V_coup

    A13 = A2 ** (1 / 3)
    Rv = 1.250 * A13 - 0.225
    theta_c = np.array([50.0, 3.0, 8.0, Rv, 0.65, 0.215, 0.093])
    bounds = np.array([
        [ 35.0,  65.0],
        [  1.5,   6.0],
        [  4.0,  14.0],
        [Rv * 0.90, Rv * 1.10],
        [  0.50,   0.80],
        [  0.17,   0.26],
        [  0.05,   0.14],
    ])
    return channels, model, theta_c, bounds


def preset_alpha208Pb_optical():
    """alpha+208Pb, single channel, WS OMP. Goldring et al., PLB 32 (1970) 465.
    theta = (V0, W0, R0, a0). Point-charge Coulomb (z1z2=164, eta~7-16)."""
    A1, A2, Z1, Z2 = 4, 208, 2, 82
    R0_c = 1.1132 * (A2 ** (1/3) + A1 ** (1/3))
    a_c  = 0.5803

    channels, _ = build_channels(
        A1=A1, A2=A2, z1z2=Z1 * Z2,
        R_coulomb=None,
        level_specs=[dict(l=0, s=0, j=0)])

    def model(theta):
        V0, W0, R0, a0 = theta
        def V_diag(r, ch): return -(V0 + 1j * W0) * woods_saxon(r, R0, a0)
        return V_diag, None

    theta_c = np.array([100.0, 10.0, R0_c, a_c])
    bounds = np.array([
        [ 70.0, 140.0],
        [  5.0,  20.0],
        [R0_c * 0.85, R0_c * 1.15],
        [  0.45,   0.72],
    ])
    return channels, model, theta_c, bounds




def preset_o16Ca44_rotational():
    """16O+44Ca, 2-channel CC: 0+/2+_1(1.157 MeV), lambda=2.
    Heavy-ion (z1z2=160), above-barrier (34-55 MeV).
    theta = (V0, W0, R0, a0, beta2). Ref: Rhoades-Brown et al., PRC 21 (1980) 2417."""
    A1, A2, Z1, Z2 = 16, 44, 8, 20
    E_2plus = 1.157

    channels, _ = build_channels(
        A1=A1, A2=A2, z1z2=Z1 * Z2,
        uniform_sphere_coulomb=True,
        level_specs=[
            dict(l=0, s=0, j=0, threshold=0.0,    I=0),
            dict(l=0, s=0, j=0, threshold=E_2plus, I=2),
        ])

    R0_c = 1.2 * (A1 ** (1/3) + A2 ** (1/3))

    def model(theta):
        V0, W0, R0, a0, beta2 = theta
        delta2 = beta2 * R0

        def V_diag(r, ch):
            return -(V0 + 1j * W0) * woods_saxon(r, R0, a0)

        def V_coup(r, ch_a, ch_b):
            Ia, Ib = ch_a['I'], ch_b['I']
            if Ia == Ib:
                return np.zeros_like(r, dtype=complex)
            F = rotational_coupling_factor(Ia, Ib, lam=2)
            if F == 0.0:
                return np.zeros_like(r, dtype=complex)
            return -delta2 * F * woods_saxon_surface(r, R0, a0)

        return V_diag, V_coup

    theta_c = np.array([110.0, 20.0, R0_c, 0.5, 0.4])
    bounds = np.array([
        [ 77.0, 143.0],
        [ 10.0,  35.0],
        [R0_c * 0.85, R0_c * 1.15],
        [  0.38,   0.62],
        [  0.28,   0.52],
    ])
    return channels, model, theta_c, bounds


def preset_alpha12C_band():
    """alpha+12C at J = 3, rigid rotor 0+ / 2+ (4.44) / 4+ (14.08 MeV), lambda = 2.

    theta = (V0, W0, R0, a0, beta2), p = 5, Nc = 8.

    A faithful reproduction of Example 4 of the reference R-matrix package
    (Descouvemont, CPC 200 (2016) 199; example4.f), the only system here whose
    published output can be compared value by value. Faithful means:

      * Eight channels, not three. At J = 3 each target spin I generates every
        l from |J - I| to J + I in steps of two, so I = 0 gives l = 3, I = 2
        gives l = 1, 3, 5 and I = 4 gives l = 1, 3, 5, 7. Collapsing the band to
        one l = 0 channel per level removes the centrifugal barrier that keeps
        the projectile out of the absorptive interior, and changes the elastic
        amplitude by an order of magnitude.

      * The angular coupling carries 3j and 6j symbols in (l, I, J, lambda), and
        its diagonal does not vanish: a level with I >= lambda couples to itself
        through the static reorientation term.

      * Two distinct radii. The Woods-Saxon radius and the uniform-charge
        Coulomb radius are r0 (A_t^1/3 + A_p^1/3); the deformation length is
        built on the target alone, beta r0 A_t^1/3. They differ by a factor of
        1.7 here, and interchanging them mis-scales the whole coupling.

    The channel radius is 14 fm rather than the 9 to 11 fm of the reference
    runs. Below the highest threshold a closed channel has an exponential tail
    that is not extinguished at 11 fm, and the published amplitudes swing by
    factors of two to three across their own three radii there, while this
    calculation is stable to 3e-4 from 14 fm out to 30 fm. Above the highest
    threshold, where the reference is itself converged, the two agree to 1e-4.
    """
    A_P, A_T, Z_P, Z_T = 4, 12, 2, 6
    J, LAM = 3, 2
    E_2plus, E_4plus = 4.44, 14.08

    R0_c = 1.2 * (A_T ** (1 / 3) + A_P ** (1 / 3))
    # The deformation radius as a fixed fraction of the Woods-Saxon radius, so
    # that it equals r0 A_T^1/3 at the reference point and still scales sensibly
    # when R0 is varied as a parameter.
    DEFORM_RATIO = A_T ** (1 / 3) / (A_T ** (1 / 3) + A_P ** (1 / 3))

    specs = rotational_channels(J, [0, 2, 4], [0.0, E_2plus, E_4plus])
    X = rotational_xfac(specs, J=J, lam=LAM)

    channels, _ = build_channels(A1=A_P, A2=A_T, z1z2=Z_P * Z_T,
                                 R_coulomb=R0_c, level_specs=specs)

    def model(theta):
        V0, W0, R0, a0, beta2 = theta
        depth = V0 + 1j * W0
        delta = beta2 * DEFORM_RATIO * R0

        def form(r):
            """-d/dr of the Woods-Saxon shape, the deformation form factor."""
            x = np.exp((r - R0) / a0)
            return x / (a0 * (1.0 + x) ** 2)

        def V_diag(r, ch):
            i = ch['idx']
            return (-depth / (1.0 + np.exp((r - R0) / a0))
                    - X[i, i] * depth * delta * form(r))

        def V_coup(r, ch_a, ch_b):
            return -X[ch_a['idx'], ch_b['idx']] * depth * delta * form(r)

        return V_diag, V_coup

    theta_c = np.array([110.0, 20.0, R0_c, 0.5, 0.58])
    bounds = np.array([[77.0, 143.0], [10.0, 35.0],
                       [R0_c * 0.85, R0_c * 1.15], [0.38, 0.62], [0.40, 0.72]])
    return channels, model, theta_c, bounds


