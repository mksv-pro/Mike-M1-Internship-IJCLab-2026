"""Extend the channel axis of scaling_study.py to Nc = 16, 20, 30, 50.

Does the retained rank nb keep following the channel count, or saturate? The
existing sweep stops at Nc = 12 (nb = 29, 43, 57, 82, 101, 109, 110), last
increments 8 then 1 -- a plateau not yet established.

Identical to axis A of scaling_study.py (same preset, mesh N = 60, p = 6,
demonstration energy, calibration rule); only Nc moves. Caveat per row: at
E = 12 MeV the rotor thresholds close every level above I = 26, so beyond
Nc = 14 the OPEN-channel count stays at 14 while the matrix grows -- those
points probe the many-closed-channel regime.

Writes scaling_extend.json; does not touch scaling_study.json.
"""

# Run-from-anywhere: put this dir (siblings) then the package root (core/, pipeline/) on the path.
import sys as _sys
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve().parent
_sys.path[:0] = [str(_HERE), str(_HERE.parent)]
import json
import time

from scaling_study import one_point, E_DEMO, N_MESH

E2 = 0.100  # the rotor constant of the preset


def n_open(Nc):
    return sum(E2 * I * (I + 1) / 6.0 < E_DEMO for I in range(0, 2 * Nc, 2))


if __name__ == "__main__":
    # The whole axis is re-run, not just the new points. scaling_study.json was
    # written on 14 August against an older calibration API, before the
    # predictor selection started weighting each candidate radius by the central
    # solution. Mixing old and new points on one curve would compare two
    # different selections, so every point here is measured with today's code.
    rows = []
    for Nc in (2, 3, 4, 6, 8, 10, 12, 16, 20, 30, 50):
        t0 = time.perf_counter()
        print(f"--- Nc={Nc}, {n_open(Nc)} open of {Nc}, matrix {Nc*N_MESH}",
              flush=True)
        r = one_point(Nc, range(6), "channels_ext")
        r["n_open"] = n_open(Nc)
        r["wall_s"] = time.perf_counter() - t0
        print(f"    wall {r['wall_s']/60:.1f} min", flush=True)
        rows.append(r)
        with open(_HERE / "scaling_extend.json", "w") as f:
            json.dump(rows, f, indent=1)
    print("\nwritten scaling_extend.json", flush=True)
