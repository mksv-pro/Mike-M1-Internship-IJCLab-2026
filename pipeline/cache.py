"""The boundary between computing and drawing.

compute.py writes results/<system>/<block>.npz; views.py reads them and nothing
else. No figure module imports a solver, no compute module imports matplotlib,
so text, tables and figures all read from the same array.

Every block carries its provenance in a `__meta__` JSON blob: seeds,
hyperparameters, sample sizes, timings, and a fingerprint of the core sources
that produced it. A stale cache announces itself instead of drawing last week's
result.
"""
import hashlib
import json
import platform
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

#: Where blocks are read and written. Not a constant: `use()` moves it, so one
#: checkout can hold several complete caches side by side.
RESULTS = Path("results")
DEFAULT_RESULTS = Path("results")


def use(name=None):
    """Point the whole pipeline at another cache directory, and return it.

        cache.use("hires")   ->  results-hires/     (created if absent)
        cache.use()          ->  results/           (the default)

    Everything downstream goes through path(): Run writes there, the views read
    there, and figures land in figures_out/<name>/ so two caches cannot
    overwrite each other's PNGs. Lets a second configuration be computed and
    drawn without destroying the first.
    """
    global RESULTS
    RESULTS = (DEFAULT_RESULTS if name in (None, "", "results")
               else Path(f"results-{name}"))
    RESULTS.mkdir(parents=True, exist_ok=True)
    return RESULTS


def current():
    """The cache in force, as a Path."""
    return RESULTS


def name():
    """Its short name: "results" for the default, else what use() was given."""
    return "" if RESULTS == DEFAULT_RESULTS else RESULTS.name[len("results-"):]


def available():
    """Every cache directory in the working directory, default first."""
    here = sorted(p.name for p in Path(".").glob("results-*") if p.is_dir())
    return (["results"] if DEFAULT_RESULTS.is_dir() else []) + here

# The blocks, and the figures each one feeds. Shown to the user when a figure
# finds its input missing. F4 and F5 read one block on purpose (both ask about
# error against rank).
BLOCKS = {
    "sweep":      "F1     benchmark suite: potential components and observable spread",
    "validation": "F2     DBMM against the R-matrix reference",
    "invariants": "F3     symmetry and unitarity against mesh refinement",
    "manifold":   "F4,F5  POD spectrum, truncation criterion, error by rank and by quantity",
    "failure":    "F7     the failure that passes an elastic-only validation",
    "excitation": "F8     excitation functions across thresholds",
    "interior":   "F9,F10 psi(r) and delta(E), solver against the emulators",
    "cat":        "F6b    cost-accuracy cloud, one point per validation draw",
    "scaling":    "F11    online cost against mesh size, RBM against LROM",
    "budget":     "       joint against split predictor budget, at fixed K",
}

#: Blocks whose contents depend on the MACHINE, not the physics -- the answer to
#: "the interpreter changed, what must I redo". Everything in the cache is
#: deterministic given the code and calibration except a measured duration, and
#: durations live only in `cat` (cost-accuracy cloud + solver baselines) and
#: `scaling` (online cost vs mesh). They are ~26% of a full run, so
#: `api.rebench()` re-measures a machine change without touching an accuracy
#: number. test_outputs.py asserts the invariant.
MACHINE_DEPENDENT = ("cat", "scaling")


def _core_fingerprint():
    """Short hash of every core/*.py file: solvers, emulators and benchmark
    definitions. A change to any of them invalidates a cached result."""
    h = hashlib.sha256()
    here = Path(__file__).resolve().parent
    sources = sorted((here.parent / "core").glob("*.py"))
    for p in sources:
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:12]


# The fingerprint is recorded, not acted on: Run's plan answers the sharper
# question ("made under different settings or mesh", not "a core/ file changed").


def path(system_key, block):
    return RESULTS / system_key / f"{block}.npz"


def write(system_key, block, arrays, meta=None, methods=None, merge=False):
    """Write one block. `arrays` is a dict of ndarrays, `meta` a JSON-able dict.

    `methods` records, per emulator, the settings and time it was produced under:

        {"rbm": {"settings": {...}, "seconds": 4.5}}

    With `merge=True` the block is read back first and the new arrays and method
    records are added to what is there, so computing LROM later does not destroy
    the RBM results. Provenance is per method: a merged block can hold RBM at
    200 samples beside LROM at 400, and the metadata says so.
    """
    if block not in BLOCKS:
        raise KeyError(f"unknown block {block!r}; known: {sorted(BLOCKS)}")
    out = path(system_key, block)
    out.parent.mkdir(parents=True, exist_ok=True)

    previous, prev_methods = {}, {}
    if merge and out.exists():
        try:
            previous, old_meta = read(system_key, block)
            prev_methods = old_meta.get("methods", {}) or {}
        except Exception:
            previous, prev_methods = {}, {}

    full_meta = dict(meta or {})
    if methods or prev_methods:
        full_meta["methods"] = {**prev_methods, **(methods or {})}
    from . import blas_threads
    from .coulomb import chain_signature
    full_meta.update(
        system=system_key,
        block=block,
        written=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        core_fingerprint=_core_fingerprint(),
        numpy=np.__version__,
        python=platform.python_version(),
        # which special-function backends produced the Coulomb values: swapping
        # the chain changes the numbers (default chain vs the faster 2e-12 mesh
        # kernel), so a cross-chain comparison is not a comparison.
        specfun=chain_signature(),
        # the threading policy: one BLAS thread against twelve is a factor of
        # forty on a validation loop, so a timing is meaningless without it.
        blas=blas_threads(),
    )
    # A machine-dependent block is all timing, so warn at write time (not in a
    # metadata field nobody reads) when the process did not get the one-thread
    # policy: its numbers cannot join the cache they are about to.
    if block in MACHINE_DEPENDENT:
        _threads = [n for api, n in (full_meta["blas"] or {}).items()
                    if isinstance(n, int) and api != "source"]
        if any(n > 1 for n in _threads):
            warnings.warn(
                f"{system_key}/{block}: written with {full_meta['blas']}, not "
                f"one BLAS thread. Every number in this block is a timing, and "
                f"oversubscription moves them by up to a factor of forty. "
                f"Import pipeline BEFORE numpy (or anything that imports it, "
                f"pandas included) and rerun.", stacklevel=2)
    payload = {k: np.asarray(v) for k, v in previous.items()}
    payload.update({k: np.asarray(v) for k, v in arrays.items()})
    payload["__meta__"] = np.array(json.dumps(full_meta, default=str))
    np.savez_compressed(out, **payload)
    return out


#: Where the two method-axis blocks (== run.METHOD_BLOCKS) keep each emulator's
#: result: one array per emulator, so a method can be added later without
#: touching the others. `cat` scans configurations, carrying a `method` column
#: rather than an array per emulator, so it is absent here.
_METHOD_ARRAYS = {"excitation": {"rbm": "sigma_rbm", "lrom": "sigma_lrom",
                                 "rbm_et": "sigma_et"},
                  "interior": {"rbm": "psi_rbm", "lrom": "psi_lrom"}}


def methods_present(block, arrays, meta=None):
    """Which emulators a block actually holds data for, read from the arrays.

    Not from the metadata (that is provenance and may be absent). The arrays
    answer "can this be drawn": a full-size all-NaN `sigma_et` would otherwise
    draw as an empty panel that reads as converged.
    """
    names = _METHOD_ARRAYS.get(block)
    if not names:
        return []
    out = []
    for m, name in names.items():
        if name not in arrays:
            continue
        # abs() first: psi_* is complex, and asarray(..., dtype=float) on a
        # complex array raises rather than answering the question.
        if np.isfinite(np.abs(np.asarray(arrays[name]))).any():
            out.append(m)
    return out


def read(system_key, block):
    """Return (arrays_dict, meta_dict). Raises FileNotFoundError with the fix."""
    p = path(system_key, block)
    if not p.exists():
        raise FileNotFoundError(
            f"missing {p}\n"
            f"    {BLOCKS.get(block, block)}\n"
            f"    produce it with:  Run(systems=[{system_key!r}], "
            f"blocks=[{block!r}]).execute()")
    with np.load(p, allow_pickle=False) as z:
        meta = json.loads(str(z["__meta__"]))
        arrays = {k: z[k] for k in z.files if k != "__meta__"}
    return arrays, meta


def require(system_keys, block):
    """Read one block for several systems at once.

    Reports every missing system in a single message rather than failing on the
    first, so one run of the pipeline tells the user everything still to compute.
    """
    loaded, missing = {}, []
    for key in system_keys:
        try:
            loaded[key] = read(key, block)
        except FileNotFoundError:
            missing.append(key)
    if missing:
        raise FileNotFoundError(
            f"block {block!r} missing for: {', '.join(missing)}\n"
            f"    {BLOCKS.get(block, block)}\n"
            f"    produce it with:  Run(systems={missing!r}, "
            f"blocks=[{block!r}]).execute()")
    return loaded
