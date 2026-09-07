"""Disk-persisted Coulomb cache, stamped with the backend chain that filled it.

The values depend only on (l, eta, k, r), never on the optical-potential
parameters, so the cost is paid once per (system, mesh, energy grid) and may as
well be paid across runs. Still cheap since core.specfun routes to scipy closed
forms and Arb, kept because it is free.

The stamp: the key (l, eta, rho) records neither the backend nor its library
versions, and F enters the DBMM source vector, so a cross-chain reuse
propagates straight into S and every reference the emulators are scored against.
The file carries the chain signature; a mismatch discards the store and the
needed keys are lazily refilled by the chain in force.
"""
import atexit
import pickle
from pathlib import Path

from core import dbmm
from core.specfun import SF

STORE = Path("results") / "_coulomb_cache.pkl"

#: Bumped whenever the on-disk layout changes. A file from an older format is
#: discarded, never reinterpreted.
FORMAT = 2

_loaded = False
_size_at_load = 0
_discarded = ""


def chain_signature():
    """What produced the values: backend names, library versions, and tuning.

    Names because a different backend is a different algorithm; tuning
    (FlintBackend.prec, NumbaBackend.step) because it changes the numbers
    without changing the backend; versions because they do too (the same name
    under mpmath 1.3 and 1.4 can differ in the last digits).
    """
    parts = []
    for b in SF.backends:
        tuning = [f"{k}={getattr(b, k)}" for k in ("prec", "step", "rho_split")
                  if hasattr(b, k)]
        version = type(b).version()
        parts.append(b.name
                     + (f"-{version}" if version else "")
                     + ("[" + ",".join(tuning) + "]" if tuning else ""))
    return "|".join(parts)


def warm():
    """Load the cache into core.dbmm's module-level dict, once, if it matches.

    Returns the number of entries adopted. A file written under a different
    special-function chain contributes nothing and is deleted, so the next
    save() writes a clean one.
    """
    global _loaded, _size_at_load, _discarded
    if _loaded:
        return len(dbmm._COULOMB_MESH_CACHE)
    _loaded = True
    if STORE.exists():
        try:
            with STORE.open("rb") as fh:
                blob = pickle.load(fh)
            if not isinstance(blob, dict) or blob.get("format") != FORMAT:
                raise ValueError("cache predates the backend stamp")
            if blob.get("chain") != chain_signature():
                raise ValueError(
                    f"written by {blob.get('chain')!r}, running {chain_signature()!r}")
            dbmm._COULOMB_MESH_CACHE.update(blob["entries"])
        except Exception as exc:
            # Never a correctness problem, always a cost one: the entries are
            # recomputed on demand by the chain in force.
            _discarded = f"{type(exc).__name__}: {exc}"
            STORE.unlink(missing_ok=True)
    _size_at_load = len(dbmm._COULOMB_MESH_CACHE)
    return _size_at_load


def status():
    """What warm() did, for a block or a manifest to record."""
    return dict(chain=chain_signature(), path=str(STORE),
                entries=len(dbmm._COULOMB_MESH_CACHE),
                adopted=_size_at_load, discarded=_discarded)


def save():
    """Write the cache back, if it grew. Registered to run at interpreter exit."""
    cache = dbmm._COULOMB_MESH_CACHE
    if not _loaded or len(cache) <= _size_at_load:
        return 0
    STORE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STORE.with_suffix(".tmp")
    with tmp.open("wb") as fh:
        pickle.dump({"format": FORMAT, "chain": chain_signature(),
                     "entries": dict(cache)},
                    fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(STORE)
    return len(cache)


atexit.register(save)
