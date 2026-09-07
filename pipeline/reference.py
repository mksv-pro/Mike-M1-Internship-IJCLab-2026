"""Hashes of the cached arrays: the regression reference.

cache.py's core_fingerprint says a result *might* have changed (different code);
it never says a number moved. This module hashes the array contents instead, so
any difference is a bug or a decision and both are seen.

    api.check()     compare the cache against the frozen reference
    api.freeze()    accept the current numbers as the new one

On arrays, not PNGs: an image diff moves with the matplotlib version, a font or
a driver.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from . import cache


def _results():
    """Whichever cache is in force -- see cache.use()."""
    return cache.current()

# Metadata fields that are part of a result's identity: the sampling that
# produced the numbers. Everything else changes run to run without anything
# having moved (timings, core_fingerprint).
META_KEYS = ("n_theta", "n_E", "n_train", "n_valid", "n_train_rbm",
             "n_train_lrom", "n_repeat", "seed", "seed_train", "seed_valid",
             "seed_box", "n_channels", "N", "R", "p", "E", "dim",
             "xsec_kind", "n_sweep", "n_box", "sweep_idx",
             # the derived half of the configuration: an error curve at nb = 38
             # is not the same measurement as one at nb = 51.
             "nb", "K_diag", "K_coup", "Ns", "eps_sol", "eps_pot", "n_probe",
             # which special-function chain produced the Coulomb values.
             "specfun")


def _hash_array(a):
    """Stable hash of an array: shape, dtype and bytes.

    tobytes() on a contiguous copy, so that a transposed view and its contiguous
    equivalent hash alike when they carry the same values.
    """
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()[:16]


#: Arrays that measure the machine, not the physics: real data, but they differ
#: by a few per cent every run, so hashing them would fire the guardrail on
#: every recompute. A prefix rule, not a list -- "t_" with the underscore, so
#: theta_c, th_valid, terms_diag and threshold arrays (physics) still hash.
TIMING_NAMES = frozenset({"seconds"})
TIMING_PREFIX = "t_"


def is_timing(name):
    """Whether an array measures the machine rather than the physics."""
    return name in TIMING_NAMES or name.startswith(TIMING_PREFIX)


def hash_block(path):
    """{array name -> hash} for one .npz, plus its identifying metadata."""
    out = {}
    with np.load(path, allow_pickle=True) as z:
        for name in sorted(z.files):
            if name == "__meta__" or is_timing(name):
                continue
            arr = z[name]
            if arr.dtype == object or arr.dtype.kind in "US":
                out[name] = _hash_array(np.array([str(x) for x in arr.ravel()],
                                                 dtype="S64"))
            else:
                out[name] = _hash_array(arr)
        meta = {}
        if "__meta__" in z.files:
            try:
                full = json.loads(str(z["__meta__"]))
                meta = {k: full[k] for k in META_KEYS if k in full}
            except Exception:
                meta = {}
    return {"arrays": out, "meta": meta}


def snapshot(results=None):
    """Fingerprint the whole cache, system by system and block by block."""
    out = {}
    for npz in sorted(Path(results or _results()).glob("*/*.npz")):
        key = f"{npz.parent.name}/{npz.stem}"
        try:
            out[key] = hash_block(npz)
        except Exception as exc:
            out[key] = {"error": f"{type(exc).__name__}: {exc}"}
    return out


def compare(ref, cur):
    """(missing, added, changed) between two snapshots."""
    missing = sorted(set(ref) - set(cur))
    added = sorted(set(cur) - set(ref))
    changed = []
    for key in sorted(set(ref) & set(cur)):
        r, c = ref[key], cur[key]
        if "error" in r or "error" in c:
            changed.append((key, ["unreadable"]))
            continue
        why = []
        for name in sorted(set(r["arrays"]) | set(c["arrays"])):
            a, b = r["arrays"].get(name), c["arrays"].get(name)
            if a is None:
                why.append(f"+{name}")
            elif b is None:
                why.append(f"-{name}")
            elif a != b:
                why.append(name)
        for k in sorted(set(r["meta"]) | set(c["meta"])):
            if r["meta"].get(k) != c["meta"].get(k):
                why.append(f"meta.{k}: {r['meta'].get(k)} -> {c['meta'].get(k)}")
        if why:
            changed.append((key, why))
    return missing, added, changed
