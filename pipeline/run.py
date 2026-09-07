"""A run, declared before it happens.

Everything that characterises a computation in one value: which systems, under
which hyperparameters, for which emulators, over which blocks, and what to do
about results that already exist. You look at `plan()`, and only then call
`execute()`.

    run = Run(systems=["n40Ca", "alpha24Mg"],
              settings={"n40Ca": Settings(sweep_idx=2)},
              methods=["rbm", "lrom"],
              blocks=["manifold", "cat"],
              force="if-changed")

    run.plan()      # a table: what will run, what will be skipped, and why
    run.execute()   # with a progress bar, and the real timings afterwards

`force` has three values and the middle one is the point of the design:

    "never"       compute only what is absent
    "if-changed"  and what was computed under different settings
    "always"      everything, regardless

"if-changed" is a comparison, not a guess: every block records the settings it
was produced under, per method, so the plan can say *which field* moved rather
than shrugging and recomputing.
"""
import time
from dataclasses import dataclass, field, fields
from typing import Any, Dict, Optional, Sequence

import pandas as pd

from . import cache, compute
from .settings import DECLARED, FAILURE_SYSTEM, SUITE, Settings, declared
from .systems import SYSTEMS

#: Emulators a block can be asked to fit. Only `excitation` and `interior`
#: (METHOD_BLOCKS) have a method axis; elsewhere the field is ignored and plan()
#: says so. __post_init__ rejects anything not in this tuple -- reaching RBM-ET
#: means calling compute.block_excitation(..., methods=("rbm_et",)) directly.
METHODS = ("rbm", "lrom")

#: Which blocks actually read the method list.
METHOD_BLOCKS = ("excitation", "interior")

#: Seconds per block at dim 300, scaled by dim**3 for other systems. An
#: order-of-magnitude estimate so a plan can say "this is minutes". `cat` and
#: `scaling` are absent on purpose: their cost is set by the repetition protocol
#: and draw count, not dim**3.
_COST_AT_DIM300 = {"sweep": 23, "validation": 13, "invariants": 0.8,
                   "manifold": 9, "failure": 4, "excitation": 16,
                   "interior": 18, "budget": 11}


#: Floor on the estimate: below dim ~ 150 the cubic term stops dominating (the
#: mesh Coulomb functions and the reference solver do), and a plan that prints
#: 0 s reads as broken.
_FLOOR_S = 5.0


def _estimate(system_key, block):
    dim = SYSTEMS[system_key].n_channels * SYSTEMS[system_key].N
    return max(_FLOOR_S, _COST_AT_DIM300.get(block, 10.0) * (dim / 300.0) ** 3)


def _settings_of(spec, key):
    """Resolve one system's settings from whatever the run declared."""
    if spec is None:
        return declared(key)
    if isinstance(spec, dict):
        spec = spec.get(key)
        if spec is None:
            return declared(key)
    if isinstance(spec, str):
        raise ValueError(f"settings must be a Settings or None; got {spec!r}. "
                         "Ranks and budgets are no longer a run-level choice: "
                         "every block reads pipeline.plan_calibration.")
    return spec


def _calibration_record(system_key):
    """The calibration a run would use, in the shape provenance stores it.

    Returns None when it cannot be obtained, so "unknown" does not read as
    "changed".
    """
    try:
        from .plan_calibration import calibration_for
        cal = calibration_for(system_key)
    except Exception:
        return None
    return dict(nb=cal.nb, K_diag=cal.K_diag, K_coup=cal.K_coup, Ns=cal.Ns,
                eps_sol=cal.eps_sol, eps_pot=cal.eps_pot, n_probe=cal.n_probe)


def _canon(v):
    """A value as it compares, not as it is typed: JSON has no tuples, so a
    sequence field round-trips to a list and would compare unequal to its tuple."""
    if isinstance(v, (list, tuple)):
        return tuple(_canon(x) for x in v)
    if isinstance(v, dict):
        return {k: _canon(x) for k, x in v.items()}
    return v


def _diff(a: Settings, b: Settings):
    """Fields that differ between two settings, as 'name: old -> new'."""
    out = []
    for f in fields(Settings):
        x, y = getattr(a, f.name), getattr(b, f.name)
        if _canon(x) != _canon(y):
            out.append(f"{f.name}: {x} -> {y}")
    return out


def _moved(record, st: Settings, fingerprint, label="", calibration=None):
    """What changed between a stored provenance record and the run being planned.

    Three axes:
      settings     the experiment, declared by hand;
      system       the physics -- mesh, radius, channel count;
      calibration  nb, K and N_s, derived, sizing the emulator blocks and
                   present in neither of the other two.

    A record written before an axis existed does not carry it and is not
    reported as changed on that account.
    """
    tag = f"{label}/" if label else ""
    out = []
    old_fp = record.get("system")
    if old_fp and _canon(old_fp) != _canon(fingerprint):
        changed = [f"{f}: {old_fp.get(f)} -> {fingerprint[f]}"
                   for f in fingerprint if _canon(old_fp.get(f)) != _canon(fingerprint[f])]
        out.append(f"{tag}system {'; '.join(changed[:2])}")
    old = record.get("settings")
    if old:
        try:
            out += [f"{tag}{d}" for d in _diff(Settings(**old), st)]
        except TypeError:
            out.append(f"{tag}settings format changed")
    old_cal = record.get("calibration")
    if old_cal and calibration:
        out += [f"{tag}calibration {f}: {old_cal[f]} -> {calibration[f]}"
                for f in calibration
                if f in old_cal and _canon(old_cal[f]) != _canon(calibration[f])]
    return out


@dataclass
class Run:
    systems: Sequence[str] = SUITE
    settings: Optional[Any] = None            # Settings | "calibrate" | {key: ...}
    methods: Sequence[str] = METHODS
    blocks: Optional[Sequence[str]] = None
    force: str = "if-changed"
    _resolved: Dict[str, Settings] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        bad = [m for m in self.methods if m not in METHODS]
        if bad:
            raise ValueError(f"unknown method(s) {bad}; known: {', '.join(METHODS)}")
        if self.force not in ("never", "if-changed", "always"):
            raise ValueError("force must be 'never', 'if-changed' or 'always'")
        unknown = [k for k in self.systems if k not in SYSTEMS]
        if unknown:
            raise KeyError(f"unknown system(s) {unknown}")
        self.blocks = list(self.blocks or cache.BLOCKS)

    def settings_for(self, key):
        """Resolved once per run, so 'calibrate' does not re-probe per block."""
        if key not in self._resolved:
            self._resolved[key] = _settings_of(self.settings, key)
        return self._resolved[key]

    # --- what would happen ---

    def plan(self):
        rows = []
        for key in self.systems:
            st = self.settings_for(key)
            for block in self.blocks:
                rows.append(self._plan_one(key, block, st))
        return pd.DataFrame(rows).set_index(["system", "block"])

    def _plan_one(self, key, block, st):
        """One row: what is on disk, what will run on it, and why.

        `methods` is what execute() will fit (differs from the request on an
        extend).
        """
        wanted = list(self.methods) if block in METHOD_BLOCKS else []
        row = dict(system=key, block=block,
                   methods="n/a" if block not in METHOD_BLOCKS else "",
                   cached="", action="", why="", est_s=round(_estimate(key, block)))

        def fit(names, action, why, est=None):
            if block in METHOD_BLOCKS:
                row["methods"] = ",".join(names)
            row.update(action=action, why=why)
            if est is not None:
                row["est_s"] = est
            return row

        if block == "failure" and key != FAILURE_SYSTEM:
            row.update(action="skip", why=f"drawn only for {FAILURE_SYSTEM}",
                       est_s=0, methods="n/a")
            return row

        # uncoupled system: K_coup is 0 by the physics, so the budget A/B has
        # nothing to compare. Skip rather than write a row of NaN.
        if block == "budget" and not SYSTEMS[key].coupled:
            row.update(action="skip", why="uncoupled: no budget to split",
                       est_s=0, methods="n/a")
            return row

        p = cache.path(key, block)
        if not p.exists():
            row["cached"] = "no"
            return fit(wanted, "compute", "absent")

        try:
            arrays, meta = cache.read(key, block)
        except Exception as exc:
            row["cached"] = "unreadable"
            return fit(wanted, "compute", str(exc)[:60])

        have = cache.methods_present(block, arrays, meta)
        row["cached"] = ",".join(have) if have else "yes"

        if self.force == "always":
            return fit(wanted, "recompute", "force='always'")

        missing = [m for m in wanted if m not in have]
        if missing:
            # only the missing methods are fitted; the block merges, so what is
            # already on disk survives.
            return fit(missing, "extend", f"missing {','.join(missing)}",
                       est=round(row["est_s"] * len(missing) / max(len(wanted), 1)))

        if self.force == "never":
            return fit([], "skip", "present")

        # if-changed: compare against the settings each method was made under; a
        # block with no method axis carries one record for the block itself.
        fp = SYSTEMS[key].fingerprint
        records = dict(meta.get("methods") or {})
        if not records and meta.get("provenance"):
            records = {"": meta["provenance"]}

        if not records:
            moved = ["no provenance recorded"]
        else:
            # read, not probed: a plan must not spend a calibration probe on a
            # system it may skip (calibration_for is frozen to disk).
            cal = _calibration_record(key)
            moved = [d for m, rec in records.items()
                     for d in _moved(rec, st, fp, m, calibration=cal)]

        if moved:
            return fit(wanted, "recompute", "; ".join(moved[:3]))
        return fit([], "skip", "settings unchanged")

    # --- doing it ---

    def execute(self, progress=True):
        planned = self.plan().reset_index()
        todo = planned[planned["action"].isin(("compute", "recompute", "extend"))]
        if todo.empty:
            # return the full plan so a no-op run says WHY it did nothing.
            out = planned.copy()
            out["state"] = "skipped: " + out["why"]
            out["seconds"] = 0.0
            return out.set_index(["system", "block"])[["state", "seconds"]]

        bar = None
        if progress:
            try:
                from tqdm.auto import tqdm
                bar = tqdm(total=float(todo["est_s"].sum()), unit="s",
                           bar_format="{l_bar}{bar}| {n:.0f}/{total:.0f}s est.")
            except ImportError:
                bar = None

        rows = []
        for _, r in todo.iterrows():
            key, block = r["system"], r["block"]
            if bar is not None:
                bar.set_description(f"{key}/{block}")
            st = self.settings_for(key)
            t0 = time.perf_counter()
            try:
                # the plan's own column, not self.methods: they differ on an extend.
                kw = ({"methods": tuple(r["methods"].split(","))}
                      if block in METHOD_BLOCKS else {})
                compute.BLOCK_FUNCTIONS[block](SYSTEMS[key], st, **kw)
                state = "done"
            except Exception as exc:
                state = f"{type(exc).__name__}: {exc}"
            dt = time.perf_counter() - t0
            rows.append(dict(system=key, block=block, state=state,
                             seconds=round(dt, 1), est_s=r["est_s"]))
            if bar is not None:
                bar.update(float(r["est_s"]))
                bar.write(f"  {key}/{block:11s} {state:10s} "
                          f"{dt:7.1f}s  (est {r['est_s']}s)")
        if bar is not None:
            bar.close()
        return pd.DataFrame(rows).set_index(["system", "block"])


def inventory(systems=None):
    """What the cache contains, read from the files themselves: the methods each
    block really holds, so a request that cannot be served is visible in advance."""
    rows = []
    for key in (systems or sorted({*SUITE, *DECLARED})):
        for block in cache.BLOCKS:
            p = cache.path(key, block)
            if not p.exists():
                rows.append(dict(system=key, block=block, present=False,
                                 methods="", written="", KiB=0))
                continue
            try:
                arrays, meta = cache.read(key, block)
            except Exception as exc:
                rows.append(dict(system=key, block=block, present=False,
                                 methods=f"unreadable: {exc}"[:40], written="",
                                 KiB=0))
                continue
            have = cache.methods_present(block, arrays, meta)
            rows.append(dict(system=key, block=block, present=True,
                             methods=",".join(have) if have else "-",
                             written=meta.get("written", "")[:19],
                             KiB=round(p.stat().st_size / 1024)))
    return pd.DataFrame(rows).set_index(["system", "block"])
