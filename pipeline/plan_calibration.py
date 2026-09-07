"""One calibration per system, one ladder, and every block reading the same one.

Probe once, build a ladder of ranks from the spectrum, and spend each figure on
as many TOP rungs as it needs: a spectrum scan takes all of it, a cost-accuracy
cloud the top two, an excitation function one.

    from pipeline.plan_calibration import calibration_for
    cal = calibration_for("alpha12C_band")
    print(cal)                       # nb, K_diag, K_coup, Ns, r_fit

What a run costs is `Run(...).plan()`, which prints an `est_s` per block and
reads the cache to say which of them will run.
"""
import json
from dataclasses import asdict, dataclass

import numpy as np

from core.calibration import EPS_POT, EPS_SOL

from . import cache
from .calibrate import probe
from .systems import SYSTEMS


# --- the calibration itself --------------------------------------------

@dataclass(frozen=True)
class Calibration:
    """What the physics asks for on one system, nothing written by hand.

    All four come from one probe: nb and the two predictor budgets from the
    spectra, Ns as a multiple of nb. With Ns = c*nb the overfit ratio collapses
    to r_fit = c*nb / [K(nb+1)] ~= c/K, a property of the rule (c = 5, K = 15 to
    23 here): r_fit >= 1 would mean c >= K, an offline stage K/5 times larger.
    """
    system: str
    nb: int             # rank the solution spectrum asks for, at eps_sol
    K_diag: int         # predictor points on the diagonal potential, at eps_pot
    K_coup: int         # ... and on the coupling; 0 when the system is uncoupled
    Ns: int             # training budget, derived as a multiple of nb
    eps_sol: float
    eps_pot: float
    n_probe: int
    # What the spectrum asked for before the split rule: K_diag/K_coup above are
    # what is spent, K_*_spec is the spectral reading, which the budget block
    # needs as a genuine split allocation to test against a joint one.
    K_diag_spec: int = 0
    K_coup_spec: int = 0
    split_needed: bool = True
    diag_travel: float = float("nan")

    @property
    def K(self):
        return self.K_diag + self.K_coup

    @property
    def r_fit(self):
        """Ns / [K(nb+1)] at the full rank.

        Below 1 the LROM least-squares system is wider than it is tall and lstsq
        returns the minimum-norm interpolant -- usable, but carried as a number
        rather than left to a warning.
        """
        return self.Ns / (self.K * (self.nb + 1))

    @property
    def well_posed_rank(self):
        """Largest rank this budget supports with r_fit >= 1."""
        return max(2, self.Ns // self.K - 1)

    def ladder(self, n=6):
        """n log-spaced ranks from nb/4 up to nb. A scan takes all of it, a
        cloud the top two, an operating point the top one."""
        lo, hi = max(1, self.nb // 4), max(self.nb, 2)
        pts = np.unique(np.round(np.exp(
            np.linspace(np.log(lo), np.log(hi), n))).astype(int))
        return tuple(int(x) for x in pts)

    def budgets(self, n=2):
        """The top n predictor budgets, as (K_diag, K_coup) pairs.

        K_coup is scaled at the calibrated ratio, not set equal to K_diag: the
        split is what the LROM contributes.
        """
        lo, hi = max(2, self.K_diag // 3), max(self.K_diag, 2)
        Ks = np.unique(np.round(np.exp(
            np.linspace(np.log(lo), np.log(hi), 6))).astype(int))
        out = []
        for K in sorted(Ks)[-n:]:
            if not self.K_coup:
                out.append((int(K), 0))
            else:
                out.append((int(K), max(2, int(round(K * self.K_coup / self.K_diag)))))
        return tuple(out)

    def __str__(self):
        return (f"{self.system}: nb={self.nb} K={self.K_diag}+{self.K_coup} "
                f"Ns={self.Ns} r_fit={self.r_fit:.2f} "
                f"(well-posed rank {self.well_posed_rank})")


_CACHE = {}


def store_path():
    """Where the frozen calibrations live: beside the blocks they configured.

    One per cache directory, because `cache.use("hires")` means a different mesh
    and therefore a different answer to the same question.
    """
    return cache.current() / "_calibration.json"


def _entry_key(system_key, probe_kw):
    """Everything the answer depends on, as one string: the tolerances, the
    probe size and seed, and the system fingerprint. A moved mesh re-probes."""
    return json.dumps(dict(
        system=system_key,
        eps_pot=float(probe_kw.get("eps_pot", EPS_POT)),
        eps_sol=float(probe_kw.get("eps_sol", EPS_SOL)),
        n_probe=int(probe_kw.get("n_probe", 120)),
        seed=int(probe_kw.get("seed", 0)),
        fingerprint=SYSTEMS[system_key].fingerprint,
    ), sort_keys=True)


def _read_store():
    p = store_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def calibration_for(system_key, **probe_kw):
    """Probe the physics; everything, budget included, follows from it.

    Memoised per process and frozen to results/_calibration.json: nb, K_diag,
    K_coup and Ns configure every block, and re-deriving them by SVD each
    session would make the whole experiment's hyperparameters a function of the
    LAPACK build. The stored entry is keyed on everything it depends on, so a
    moved mesh or preset re-probes and nothing else does.

    Delete the file to re-derive; `calibrations()` shows it, `recalibrate()`
    refreshes one system deliberately.
    """
    key = (system_key,) + tuple(sorted(probe_kw.items()))
    if key in _CACHE:
        return _CACHE[key]

    entry_key = _entry_key(system_key, probe_kw)
    store = _read_store()
    saved = store.get(entry_key)
    if saved is not None:
        try:
            cal = Calibration(**{f: saved[f] for f in Calibration.__annotations__})
        except (KeyError, TypeError):
            # entry written by an older field set: re-probe rather than guess
            # the missing field.
            saved = None
        else:
            _CACHE[key] = cal
            return cal

    r = probe(system_key, **probe_kw)
    coupled = SYSTEMS[system_key].coupled
    cal = Calibration(
        system=system_key,
        nb=int(r["nb"]),
        K_diag=int(r["K_diag"]),
        K_coup=int(r["K_coup"]) if coupled else 0,
        K_diag_spec=int(r.get("K_diag_spec", r["K_diag"])),
        K_coup_spec=int(r.get("K_coup_spec", r["K_coup"])) if coupled else 0,
        split_needed=bool(r.get("split_needed", True)),
        diag_travel=float(r.get("diag_travel", float("nan"))),
        Ns=int(r["n_train_rbm"]),
        eps_sol=float(probe_kw.get("eps_sol", EPS_SOL)),
        eps_pot=float(probe_kw.get("eps_pot", EPS_POT)),
        n_probe=int(probe_kw.get("n_probe", 120)),
    )
    _CACHE[key] = cal

    from .coulomb import chain_signature
    import numpy as _np
    store[entry_key] = dict(asdict(cal), _probed=dict(
        numpy=_np.__version__, specfun=chain_signature(),
        fingerprint=SYSTEMS[system_key].fingerprint))
    p = store_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(store, indent=1, sort_keys=True), encoding="utf-8")
    return cal


def recalibrate(system_key, **probe_kw):
    """Re-derive one system's calibration and overwrite the stored entry.

    A deliberate act, like api.freeze(): it changes every figure of that system.
    """
    store = _read_store()
    store.pop(_entry_key(system_key, probe_kw), None)
    store_path().parent.mkdir(parents=True, exist_ok=True)
    store_path().write_text(json.dumps(store, indent=1, sort_keys=True),
                            encoding="utf-8")
    _CACHE.pop((system_key,) + tuple(sorted(probe_kw.items())), None)
    return calibration_for(system_key, **probe_kw)


def calibrations():
    """Every frozen calibration, as rows. What the cache was configured by."""
    import pandas as pd
    rows = []
    for raw in _read_store().values():
        probed = raw.get("_probed", {})
        cal = Calibration(**{f: raw[f] for f in Calibration.__annotations__})
        rows.append(dict(system=cal.system, nb=cal.nb, K_diag=cal.K_diag,
                         K_coup=cal.K_coup, Ns=cal.Ns, r_fit=round(cal.r_fit, 3),
                         eps_sol=cal.eps_sol, eps_pot=cal.eps_pot,
                         n_probe=cal.n_probe, numpy=probed.get("numpy", ""),
                         specfun=probed.get("specfun", "")))
    if not rows:
        return pd.DataFrame(columns=["system"]).set_index("system")
    return pd.DataFrame(rows).set_index("system")


def as_meta(cal, n_E=1):
    """The calibration as a flat dict, for a block to store beside its arrays.

    `calibrated` is the flag views._provenance reads to print "autocalibrated"
    under a figure.
    """
    return dict(nb=cal.nb, K_diag=cal.K_diag, K_coup=cal.K_coup, Ns=cal.Ns,
                eps_sol=cal.eps_sol, eps_pot=cal.eps_pot,
                n_probe=cal.n_probe, n_E=int(n_E), r_fit=round(cal.r_fit, 4),
                calibrated=True)
