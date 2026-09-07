"""Pipeline package init. Its only job is to run before numpy does.

One BLAS thread per process: the solves here are small (a reduced system is
n_b x n_b, n_b ~ 100; the full one a few hundred a side) and OpenBLAS spreading
them over every core loses badly at that size -- 38.8 s against 1.0 s on one
validation loop. This must precede the first `import numpy` in the process:
OpenBLAS reads these variables when its shared library loads and ignores them
afterwards. Importing anything from this package is enough.
"""
import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")


def _probe_threads():
    """Ask the loaded runtimes how many threads they have. None if it fails."""
    try:
        from threadpoolctl import threadpool_info
        info = threadpool_info()
    except Exception:
        return None
    return {d["internal_api"]: int(d["num_threads"]) for d in info} or None


# Probed here, right after the variables are set and numpy loads its BLAS, and
# memoised: on this environment threadpoolctl raises once scipy.linalg is in
# memory, and every real pipeline process imports scipy.linalg. numpy is
# imported deliberately -- it is what makes the variables above binding.
import numpy as _np  # noqa: E402  (must follow the environment variables)

_OBSERVED = _probe_threads()


def blas_threads():
    """What the process actually got, for a block to record beside its timings.

    One BLAS thread against twelve is a factor of forty on a validation loop.
    The two shapes are deliberately distinguishable -- the second is a fallback
    that repeats what was requested, not a measurement:

        {"openblas": 1}                                   observed
        {"openblas_num_threads": "1", "source": "env"}    assumed
    """
    if _OBSERVED is not None:
        return dict(_OBSERVED)
    out = {v.lower(): os.environ.get(v)
           for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")}
    out["source"] = "env"           # requested, not verified
    return out
