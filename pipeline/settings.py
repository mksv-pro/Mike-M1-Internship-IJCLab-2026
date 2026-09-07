"""What an experiment declares, once the spectra have decided the rest.

Settings is a flat dataclass; DECLARED holds what each system was computed with.

    from pipeline.settings import DECLARED, Settings
    import dataclasses

    st = DECLARED["alpha24Mg"]                       # as cached
    st = dataclasses.replace(st, sweep_idx=2)        # your own

Ranks, predictor budgets and training sizes are NOT here: they follow from the
two tolerances of core.calibration through pipeline.plan_calibration, so every
figure of a run shares one configuration.
"""
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class Settings:
    """What is varied and which mesh sizes are scanned. Nothing else.

    Ranks, predictor budgets and training sizes are derived per system from the
    two tolerances: see pipeline.plan_calibration.
    """

    # --- which parameter the experiment varies ---
    sweep_idx: int = 0
    sweep_label: str = r"$V_0$"
    # F7 sweeps the parameter living only in the coupling; -1 means no figure.
    failure_idx: int = -1
    failure_label: str = ""

    # --- mesh refinement, the one scan no spectrum decides ---
    N_scan: Tuple[int, ...] = (40, 60, 80, 100, 120)

    # --- which cross section is reported ---
    # None means "whatever the interaction allows": reaction where the potential
    # is absorptive, elastic where it is real. Set it explicitly to report the
    # other one -- on an absorptive system both exist and are comparable.
    xsec_kind: Optional[str] = None


_B2 = dict(failure_idx=5, failure_label=r"$\beta_2$")


#: What every cached result was computed with. Hand-written, and worth knowing
#: as such: `calibrate.compare(key)` shows what the physics would ask for.
DECLARED = {
    "n40Ca":         Settings(),
    "alpha12C_band": Settings(failure_idx=4, failure_label=r"$eta_2$"),
    "alpha24Mg":     Settings(**_B2),
    "n238U":         Settings(N_scan=(40, 60, 80, 100), **_B2),
    "p12C":          Settings(),
    # alpha+d has no V_0: its first parameter is the central depth V_c0.
    "alpha_d":       Settings(sweep_label=r"$V_{c0}$"),
    # n+58Ni carries one block, the budget test. Declared here so a run can
    # rebuild it (a system whose settings live only in a notebook cannot be
    # regenerated).
    "n58Ni":         Settings(),
}

#: The suite, in order of increasing complexity.
SUITE = ("n40Ca", "alpha12C_band", "alpha24Mg", "n238U")

#: The two control benchmarks -- a single-channel Gaussian resonance and a
#: two-channel toy with a tensor force, the systems the machinery was first
#: shown correct on. They get their own sheets, not two extra columns.
CONTROL = ("p12C", "alpha_d")

#: F7 is a one-system demonstration: alpha+24Mg is coupled, mid-sized, and
#: carries both deformations. Others have a coupling-only parameter too, but
#: only this one is drawn.
FAILURE_SYSTEM = "alpha24Mg"


def xsec_kind(system, settings=None):
    """Which cross section to report for this system under these settings.

    A real interaction has no reaction cross section, so elastic is forced;
    where both exist the choice lives here, not in the system registry.
    """
    if settings is not None and settings.xsec_kind is not None:
        return settings.xsec_kind
    return "reaction" if system.absorptive else "elastic"


def declared(key):
    """The settings a system's cache was built with.

    Raises for a system nothing has been run on, rather than handing back
    dataclass defaults that would read as a real configuration.
    """
    try:
        return DECLARED[key]
    except KeyError:
        raise KeyError(
            f"no declared settings for {key!r}: nothing has been computed on it. "
            f"Build your own with Settings(...), or derive them with "
            f"Declared: {', '.join(DECLARED)}"
        ) from None


def has_declared(key):
    """Whether a system has settings something was actually run with."""
    return key in DECLARED
