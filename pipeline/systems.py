"""The single registry: one benchmark, one entry, physics and nothing else.

Training sizes, the ranks scanned and which parameter is swept live in
settings.py, not here. Preset functions are imported from presets.py, so there
is one definition of each potential.
"""
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

import numpy as np

from .presets import (preset_alpha12C_band, preset_alpha24Mg_rotational,
                      preset_alpha208Pb_optical, preset_alpha_d_SD,
                      preset_n40Ca_coupled, preset_n40Ca_woods_saxon,
                      preset_n58Ni_vibrational, preset_n238U_rotational,
                      preset_o16Ca44_rotational, preset_p12C_gaussian,
                      preset_p40Ca_ch89)


@dataclass(frozen=True)
class System:
    """One benchmark: its physical identity and the mesh that resolves it.

    No field here describes an experiment or a figure. If a field would have to
    take two values depending on what is being measured, it does not belong.
    """

    key: str
    label: str
    preset_fn: Callable
    N: int
    R: float
    E_demo: float
    E_range: Tuple[float, float, int]
    coupled: bool
    theta_labels: Optional[Sequence[str]] = None

    @property
    def E_grid(self):
        lo, hi, n = self.E_range
        return np.linspace(lo, hi, n)

    @property
    def channels(self):
        return self.preset_fn()[0]

    @property
    def n_channels(self):
        return len(self.channels)

    @property
    def fingerprint(self):
        """The physics a cached result was produced on: mesh, radius, energy
        grid, channel count. Invisible to a settings diff, so a block must not
        merge rows computed at different dim."""
        return dict(preset=self.preset_fn.__name__, N=int(self.N),
                    R=float(self.R), E_demo=float(self.E_demo),
                    E_range=list(self.E_range), channels=int(self.n_channels))

    @property
    def absorptive(self):
        """Whether the interaction is complex, i.e. a reaction cross section
        exists.

        Measured from the potential, not declared: with a real interaction the
        collision matrix is unitary and sigma_reaction is identically zero.
        Which cross section to *report* is a separate choice, in settings.py.
        """
        _, model, theta_c, _ = self.preset_fn()
        V_diag, _ = model(theta_c)
        r = np.linspace(0.1, 0.8 * self.R, 40)
        v = np.asarray(V_diag(r, self.channels[0]), dtype=complex)
        scale = float(np.max(np.abs(v.real))) or 1.0
        return bool(np.max(np.abs(v.imag)) > 1e-8 * scale)

    @property
    def n_params(self):
        """Number of model parameters."""
        return int(self.preset_fn()[2].size)

    @property
    def thresholds(self):
        """Distinct non-zero channel thresholds, sorted."""
        return sorted({float(c["threshold"]) for c in self.channels
                       if float(c.get("threshold", 0.0)) > 0.0})

    @property
    def threshold_labels(self):
        seen, out = set(), []
        for c in sorted(self.channels, key=lambda c: float(c.get("threshold", 0.0))):
            t = float(c.get("threshold", 0.0))
            if t <= 0.0 or t in seen:
                continue
            seen.add(t)
            I = c.get("I")
            out.append(rf"${I}^+$" if I is not None else f"{t:.2f}")
        return out


_WS8 = [r"$V_0$", r"$W_0$", r"$W_{d0}$", r"$V_{\rm so}$",
        r"$R_0$", r"$a_0$", r"$R_{\rm so}$", r"$a_{\rm so}$"]
#: The ROSE parametrisation: volume, surface and spin-orbit each carry their own
#: radius and diffuseness, which is what makes it ten parameters and not eight.
_WS10 = [r"$V_{v0}$", r"$W_{v0}$", r"$W_{d0}$", r"$V_{\rm so}$",
         r"$R_{v0}$", r"$R_{d0}$", r"$R_{\rm so}$",
         r"$a_{v0}$", r"$a_{d0}$", r"$a_{\rm so}$"]
_ROT7 = [r"$V_0$", r"$W_0$", r"$W_{d0}$", r"$R_0$", r"$a_0$",
         r"$\beta_2$", r"$\beta_4$"]
_ROT5 = [r"$V_0$", r"$W_0$", r"$R_0$", r"$a_0$", r"$\beta_2$"]


SYSTEMS = {s.key: s for s in [
    System("p12C", r"$p+^{12}$C", preset_p12C_gaussian,
           N=40, R=15.0, E_demo=5.0, E_range=(0.5, 15.0, 100),
           coupled=False,
           theta_labels=[r"$V_0$", r"$\beta$"]),

    System("n58Ni", r"$n+^{58}$Ni", preset_n58Ni_vibrational,
           N=60, R=15.0, E_demo=15.0, E_range=(5.0, 60.0, 100),
           coupled=True,
           theta_labels=[r"$V_0$", r"$W_s$", r"$W_v$", r"$\delta_2$"]),

    System("p40Ca", r"$p+^{40}$Ca", preset_p40Ca_ch89,
           N=60, R=15.0, E_demo=25.0, E_range=(10.0, 60.0, 100),
           coupled=False,
           theta_labels=[r"$V_0$", r"$W_s$", r"$W_v$"]),

    System("alpha_d", r"$\alpha+d$", preset_alpha_d_SD,
           N=60, R=15.0, E_demo=5.0, E_range=(0.5, 15.0, 100),
           coupled=True,
           theta_labels=[r"$V_{c0}$", r"$V_{t0}$"]),

    System("n40Ca", r"$n+^{40}$Ca", preset_n40Ca_woods_saxon,
           # E_demo is the c.m. energy of E_lab = 14.1 MeV, the single energy
           # Giuliani's LROM package runs this case at.
           N=64, R=15.0, E_demo=13.756, E_range=(1.0, 30.0, 100),
           coupled=False, theta_labels=_WS10),

    System("n40Ca_coupled", r"$n+^{40}$Ca (coupled)", preset_n40Ca_coupled,
           N=60, R=15.0, E_demo=10.0, E_range=(0.5, 30.0, 100),
           coupled=True,
           theta_labels=_WS8 + [r"$\delta_2$"]),

    System("alpha24Mg", r"$\alpha+^{24}$Mg", preset_alpha24Mg_rotational,
           N=100, R=15.0, E_demo=54.0, E_range=(10.0, 80.0, 100),
           coupled=True, theta_labels=_ROT7),

    System("n238U", r"$n+^{238}$U", preset_n238U_rotational,
           N=80, R=20.0, E_demo=7.0, E_range=(1.0, 20.0, 100),
           coupled=True, theta_labels=_ROT7),

    System("alpha208Pb", r"$\alpha+^{208}$Pb", preset_alpha208Pb_optical,
           N=120, R=20.0, E_demo=30.0, E_range=(15.0, 50.0, 100),
           coupled=False,
           theta_labels=[r"$V_0$", r"$W_0$", r"$R_0$", r"$a_0$"]),
    
    System("alpha12C_band", r"$\alpha+^{12}$C", preset_alpha12C_band,
           N=120, R=14.0, E_demo=16.0, E_range=(2.0, 22.0, 100),
           coupled=True, theta_labels=_ROT5),


    System("o16Ca44", r"$^{16}$O$+^{44}$Ca", preset_o16Ca44_rotational,
           N=120, R=22.0, E_demo=44.0, E_range=(34.0, 55.0, 100),
           coupled=True, theta_labels=_ROT5),
]}
