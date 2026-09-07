"""The whole pipeline behind a handful of functions.

Meant to be called from a notebook: nothing parses arguments, nothing prints a
progress bar, every function returns a value to look at rather than a status code.

    from pipeline import api

    api.systems()                    what benchmarks exist, and their physics
    api.settings()                   what each cache was built with, per system
    api.figures()                    what can be drawn, and what is on disk
    api.can_draw("vs_rmatrix.suite") what a generation would leave out, and why
    api.figure("vs_rmatrix.suite")   draw one figure, get its path
    api.show("vs_rmatrix.suite")     draw it and display it inline
    api.view("vs_rmatrix", "n40Ca")  the numbers behind it, undrawn
    api.check()                      has any cached number moved
    api.tables()                     regenerate the tables from the cache
    api.stale()                      which outputs are older than their inputs
    api.campaign(1e5)                which emulator to use for a campaign, and what it costs
    api.rebench()                    re-measure the timings alone, after a machine change
    api.manifest()                   one file describing the whole state
    api.report()                     compute, draw, tabulate, record, check

Computing is `pipeline.run.Run` (declared, planned, then executed), not here.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import cache, reference
from .draw import VIEWS, audit as _audit, out_dir, render
from .settings import (CONTROL, DECLARED, SUITE, declared, has_declared,
                       xsec_kind as _xsec)
from .plan_calibration import calibration_for
from .recipes import RECIPES
from .systems import SYSTEMS

BLOCKS = tuple(cache.BLOCKS)


def _reference_path():
    """One reference file per cache: a hash frozen from `results` says nothing
    about `results-hires`."""
    return ("tests/reference_blocks.json" if not cache.name()
            else f"tests/reference_{cache.name()}.json")


# --- what exists ---------------------------------------------------------

def systems(keys=None):
    """The benchmarks: physics and mesh, nothing about any experiment.

    Training sizes are not a property of a system (the same benchmark runs with
    40 samples or 400); see api.settings().
    """
    rows = []
    for key in (keys or SYSTEMS):
        s = SYSTEMS[key]
        rows.append(dict(
            system=key, channels=s.n_channels, N=s.N, dim=s.n_channels * s.N,
            R=s.R, E_demo=s.E_demo, E_range=str(s.E_range), coupled=s.coupled,
            absorptive=s.absorptive, reports=_xsec(s, None), params=s.n_params, thresholds=len(s.thresholds),
            computed=has_declared(key)))
    return pd.DataFrame(rows).set_index("system")


def settings(keys=None):
    """What each system's cache was built with: the declared experiment, and
    beside it the calibration the physics derived.

    `swept` and `reports` are declared by hand in settings.DECLARED; nb, K, Ns
    and the rank ladder are probed from the system's own spectra. A record, not
    a control -- nothing reads it at compute time; pass Run(settings=...) to run
    something else.
    """
    rows = []
    for key in (keys or sorted(DECLARED)):
        st = declared(key)
        cal = calibration_for(key)
        rows.append(dict(system=key, nb=cal.nb, K_diag=cal.K_diag,
                         K_coup=cal.K_coup, Ns=cal.Ns,
                         r_fit=round(cal.r_fit, 2),
                         swept=st.sweep_label,
                         reports=_xsec(SYSTEMS[key], st),
                         ranks=str(cal.ladder())))
    return pd.DataFrame(rows).set_index("system")


def _scope(recipe, systems=None):
    """Which systems a figure covers: the caller's, else the recipe's, else all.

    F7 declares its own (a demonstration on one designated system); drawing it
    across the suite would ask for a block not computed elsewhere.
    """
    return list(systems or recipe.systems or SUITE)


def figures():
    """Every figure the pipeline can draw, and whether it is on disk."""
    rows = []
    for key, r in sorted(RECIPES.items()):
        outs = _paths(r, _scope(r))
        rows.append(dict(recipe=key, subject=key.split(".")[0],
                         scope="per-system" if r.per_system
                               else ("overlay" if r.overlay_systems else "suite"),
                         panels=len(r.rows),
                         drawn=sum(p.exists() for p in outs), of=len(outs),
                         file=r.stem + ".png"))
    return pd.DataFrame(rows).set_index("recipe")


def _paths(recipe, keys):
    """Where a generation would write -- must match where the renderer writes
    (flat, `<stem>.png` or `<system>_<stem>.png`)."""
    if recipe.per_system:
        return [out_dir() / f"{k}_{recipe.stem}.png" for k in keys]
    return [out_dir() / f"{recipe.stem}.png"]


def use_cache(name=None):
    """Switch every read and write to another cache, and say what is there.

        api.use_cache("hires")   ->  results-hires/ and figures_out/hires/
        api.use_cache()          ->  back to results/ and figures_out/

    Creates the directory if absent; copies and deletes nothing. The regression
    reference follows (tests/reference_<name>.json for a named cache).
    """
    cache.use(name)
    return pd.DataFrame([
        dict(cache=c, blocks=len(list((Path(".") / c).glob("*/*.npz"))),
             systems=len([p for p in (Path(".") / c).glob("*") if p.is_dir()]),
             active=(c == (cache.current().name)))
        for c in cache.available()]).set_index("cache")


def caches():
    """Every cache directory, how full it is, and which one is in force."""
    return use_cache(cache.name() or None)


# --- drawing -----------------------------------------------------------

def figure(recipe, systems=None, methods=None):
    """Draw one figure. Returns a path, or a list for a per-system recipe.

    `systems` and `methods` are display filters chosen per generation, not baked
    into the figure (RBM alone to stay readable, or everything to compare).
    """
    if recipe not in RECIPES:
        raise KeyError(f"unknown recipe {recipe!r}; known: "
                       f"{', '.join(sorted(RECIPES))}")
    r = RECIPES[recipe]
    return render(r, _scope(r, systems), methods=methods)


def can_draw(recipe, systems=None, methods=None):
    """What a generation would draw and what it would leave out, as a table.

    Look before drawing: a method asked for and silently missing produces an
    empty panel that reads like a converged emulator.
    """
    if recipe not in RECIPES:
        raise KeyError(f"unknown recipe {recipe!r}")
    r = RECIPES[recipe]
    return pd.DataFrame(_audit(r, _scope(r, systems), methods))


def show(recipe, systems=None, methods=None, report=True):
    """Draw it, display it, and say what was left out.

    `report` prints the methods asked for that had no data.
    """
    from IPython.display import Image, display
    if report and methods is not None:
        rows = _audit(RECIPES[recipe], _scope(RECIPES[recipe], systems), methods)
        missing = [r for r in rows if r["state"] != "drawn"]
        for r in missing:
            print(f"  not drawn: {r['system']}/{r['method']} -- {r['why']}")
    out = figure(recipe, systems=systems, methods=methods)
    for p in (out if isinstance(out, list) else [out]):
        display(Image(filename=str(p)))
    return out


def all_figures(systems=None):
    """Draw everything. Returns {recipe: path} or {recipe: the error text}."""
    out = {}
    for key in sorted(RECIPES):
        try:
            out[key] = figure(key, systems=systems)
        except Exception as exc:
            out[key] = f"{type(exc).__name__}: {exc}"
    return out


# --- looking underneath ------------------------------------------------

def view(name, system):
    """The numbers behind a figure, before anything is drawn.

    A suspicious curve can be read as an array without rendering; a panel's own
    summary numbers are in `.notes` next to the series they describe.
    """
    if name not in VIEWS:
        raise KeyError(f"unknown view {name!r}; known: {', '.join(sorted(VIEWS))}")
    return VIEWS[name](system)


def notes(name, system):
    """Just the summary numbers of every panel of a view, as a table."""
    rows = []
    for panel, data in view(name, system).items():
        for k, v in data.notes.items():
            rows.append(dict(panel=panel, note=k, value=v))
    return pd.DataFrame(rows)


def check(path=None):
    """Has any cached number moved since the reference was frozen."""
    path = path or _reference_path()
    ref = json.loads(Path(path).read_text(encoding="utf-8"))
    missing, added, changed = reference.compare(ref, reference.snapshot())
    rows = [dict(block=k, what=", ".join(w[:4])) for k, w in changed]
    rows += [dict(block=k, what="MISSING") for k in missing]
    rows += [dict(block=k, what="new") for k in added]
    if not rows:
        return f"{len(ref)} blocks identical to the reference"
    return pd.DataFrame(rows).set_index("block")


def freeze(path=None):
    """Accept the current numbers as the new reference. A deliberate act."""
    path = path or _reference_path()
    snap = reference.snapshot()
    Path(path).write_text(json.dumps(snap, indent=1, sort_keys=True),
                          encoding="utf-8")
    return f"{len(snap)} blocks frozen into {path}"


# --- the tables, and the state that produced everything ---------------

def tables(systems=None):
    """Regenerate the tables from the cache. Returns what was written.

    Each table is built from the same `views` call its neighbouring figure is
    drawn from, so the two cannot disagree.
    """
    from . import tables as _tables
    return _tables.write_all(list(systems or SUITE))


#: What turns arrays into an output, split by output kind. tables.py produces
#: .tex only, so an edit to it must not flag the PNGs as stale.
_SHARED_SOURCES = ("views.py", "metrics.py", "settings.py", "systems.py")
FIGURE_SOURCES = _SHARED_SOURCES + ("draw.py", "recipes.py", "style.py")
TABLE_SOURCES = _SHARED_SOURCES + ("tables.py",)


def stale():
    """What is older than what it was built from, at both levels of the chain:

        core/ + compute.py  ->  blocks  ->  views/draw/recipes  ->  png, tex

    `recompute` compares each block's recorded `core_fingerprint` (a content
    hash) against the current sources -- exact. `redraw` compares mtimes -- a
    warning, not a verdict: a docstring edit moves views.py without moving a
    pixel, so settle it by redrawing and comparing. `f7_failure.png` is expected
    here: its block is deliberately not computed.
    """
    from .tables import out_dir as _tables_dir

    here = Path(__file__).resolve().parent
    fingerprint = cache._core_fingerprint()

    recompute = []
    for npz in sorted(cache.current().glob("*/*.npz")):
        try:
            _, meta = cache.read(npz.parent.name, npz.stem)
        except Exception as exc:
            recompute.append(dict(block=f"{npz.parent.name}/{npz.stem}",
                                  why=f"unreadable: {exc}"[:50]))
            continue
        got = meta.get("core_fingerprint")
        if got and got != fingerprint:
            recompute.append(dict(block=f"{npz.parent.name}/{npz.stem}",
                                  why=f"core {got} -> {fingerprint}"))

    def newest_of(names):
        paths = [here / n for n in names if (here / n).exists()]
        return max((p.stat().st_mtime, p) for p in paths)

    newest_source = {".png": newest_of(FIGURE_SOURCES),
                     ".tex": newest_of(TABLE_SOURCES)}

    def newest_block(names):
        """Newest .npz among `names`, or None when the cache holds none of them.

        Scoped, not global: a figure is only out of date against the blocks it
        reads, so a block nothing draws (e.g. `budget`) must not age it.
        """
        qs = [q for q in cache.current().glob("*/*.npz") if q.stem in names]
        return max((q.stat().st_mtime, q) for q in qs) if qs else None

    from .draw import VIEW_BLOCKS
    drawn = newest_block(set(VIEW_BLOCKS.values()))

    outs = sorted(out_dir().glob("*.png"))
    if _tables_dir().exists():
        outs += sorted(_tables_dir().glob("*.tex"))

    def blocks_behind(p):
        """Which blocks this output is measured against: a table names its own
        in the header pipeline.tables writes; a figure, everything a view can
        read (the filename does not say which recipe made it)."""
        if p.suffix == ".png":
            return drawn
        head = p.read_text(encoding="utf-8", errors="replace")[:400]
        for line in head.splitlines():
            if line.startswith("% cache=") and "block=" in line:
                named = line.split("block=")[1].split("written=")[0]
                return newest_block({b.strip() for b in named.split("+")}
                                    & set(cache.BLOCKS))
        return drawn

    redraw = []
    for p in outs:
        m, why = p.stat().st_mtime, []
        block = blocks_behind(p)
        if block is not None and m < block[0]:
            why.append(f"block {block[1].parent.name}/{block[1].stem}")
        t_src, p_src = newest_source[p.suffix]
        if m < t_src:
            why.append(f"source {p_src.name}")
        if why:
            redraw.append(dict(output=p.name, older_than=", ".join(why)))

    if not recompute and not redraw:
        return (f"{len(outs)} outputs and "
                f"{len(list(cache.current().glob('*/*.npz')))} blocks current")
    return dict(
        recompute=pd.DataFrame(recompute).set_index("block") if recompute
        else "every block was made by the current solver sources",
        redraw=pd.DataFrame(redraw).set_index("output") if redraw
        else "every output is newer than its block and its renderer")


def campaign(n=1e5, systems=None, tol=None):
    """Which emulator to use for a campaign of `n` evaluations, and what it costs.

    Both emulators clear the posterior tolerance by orders of magnitude, so
    accuracy is a gate, not a ranking; what ranks them is the amortised total
    T(N) = T_offline + N * T_online at each one's operating point (cheapest
    configuration that clears the tolerance, from metrics.operating_point).

    `N_star` is where an emulator overtakes the split DBMM. `N_cross` is where
    the two emulators cross, which exists only when the cheaper offline has the
    steeper slope (once on this suite); unchanged if the campaign evaluates
    several energies per parameter draw.
    """
    from .style import EPS_POST
    from . import metrics
    tol = EPS_POST if tol is None else tol
    keys = list(systems or SUITE)
    rows = []
    for key in keys:
        d, meta = cache.read(key, "cat")
        method = np.asarray(d["method"]).astype(str)
        nb = np.asarray(d["nb_kept"], float)
        kd = np.asarray(d["K_diag"], float)
        kc = np.asarray(d["K_coup"], float)
        t_on = np.asarray(d["t_online"], float) * 1e3
        t_off = np.asarray(d["t_offline"], float)
        quantity = ("err_Sab" if np.isfinite(np.asarray(d["err_Sab"])).any()
                    else "err_S00")
        err = np.asarray(d[quantity], float)
        t_solver = float(np.median(np.asarray(d["t_dbmm_V"], float))) * 1e3

        picked = {}
        for m in ("rbm", "lrom"):
            combos, seen = [], set()
            for i in range(method.size):
                sig = (method[i], nb[i], kd[i], kc[i])
                if method[i] != m or sig in seen:
                    continue
                seen.add(sig)
                sel = ((method == m) & (nb == nb[i])
                       & (kd == kd[i]) & (kc == kc[i]))
                combos.append((
                    (f"nb {nb[i]:.0f}" if m == "rbm"
                     else f"nb {nb[i]:.0f}, K {kd[i]:.0f}"
                          + (f"+{kc[i]:.0f}" if kc[i] else "")),
                    float(np.nanmedian(err[sel])),
                    float(np.median(t_on[sel])), float(t_off[sel][0])))
            if not combos:
                continue
            i = metrics.operating_point([c[1] for c in combos],
                                        [c[2] for c in combos], tol=tol)
            clears = i is not None
            if not clears:
                i = int(np.argmin([c[1] for c in combos]))
            picked[m] = combos[i] + (clears,)

        for m, (lab, e, on, off, clears) in picked.items():
            rows.append(dict(
                system=key, method=m.upper(), config=lab, quantity=quantity,
                error=e, clears_tol=clears, t_on_ms=round(on, 3),
                t_off_s=round(off, 1),
                N_star=metrics.break_even(off, on * 1e-3,
                                          t_online_b=t_solver * 1e-3),
                T_min=round((off + n * on * 1e-3) / 60.0, 2),
                solver_T_min=round(n * t_solver * 1e-3 / 60.0, 2)))
        if len(picked) == 2:
            r, l = picked["rbm"], picked["lrom"]
            n_cross = metrics.crossover(r[3], r[2] * 1e-3, l[3], l[2] * 1e-3)
            cheaper = "RBM" if (r[3] + n * r[2] * 1e-3) < (l[3] + n * l[2] * 1e-3) \
                else "LROM"
            for row in rows[-2:]:
                row["N_cross"] = n_cross
                row["cheaper_at_N"] = cheaper
    return pd.DataFrame(rows).set_index(["system", "method"])


def rebench(systems=None, methods=None):
    """Re-measure what depends on the machine, and only that.

    A new interpreter, laptop or BLAS build moves every timing and no accuracy
    number. This recomputes `cache.MACHINE_DEPENDENT` (`cat`, `scaling`) and
    redraws only the figures whose view reads them, plus the two cost tables --
    ~13 min against ~52 for a full cache, without opening the physics blocks.
    `check()` compares array hashes (timings excluded by rule), so it stays
    silent afterwards.
    """
    from .run import Run
    keys = list(systems or (list(SUITE) + list(CONTROL)))
    blocks = list(cache.MACHINE_DEPENDENT)
    out = {"run": Run(systems=keys, blocks=blocks, force="always",
                      **({"methods": list(methods)} if methods else {})).execute()}

    from .draw import recipes_reading
    drawn = {}
    for key in recipes_reading(blocks):
        try:
            drawn[key] = str(figure(key, systems=[k for k in keys
                                                  if k in SUITE] or keys))
        except Exception as exc:
            drawn[key] = f"{type(exc).__name__}: {exc}"
    out["figures"] = pd.DataFrame(
        [dict(recipe=k, result=v) for k, v in drawn.items()]).set_index("recipe")
    out["tables"] = tables([k for k in keys if k in SUITE] or keys)
    out["manifest"] = manifest()
    out["check"] = check()
    return out


def manifest(path=None):
    """Write one file describing everything that produced the current outputs.

    Run-level provenance, joining the per-block records: interpreter, numpy,
    special-function chain, BLAS threads, each system's calibration, the hash of
    every cached array, and the figures and tables on disk. Written wherever the
    cache in force lives, so two configurations keep two manifests.
    """
    import platform
    from datetime import datetime, timezone
    import numpy as np
    from . import blas_threads, coulomb
    from .plan_calibration import store_path
    from .tables import out_dir as _tables_dir

    path = Path(path) if path else cache.current() / "_manifest.json"
    snap = reference.snapshot()
    doc = dict(
        written=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        cache=str(cache.current()),
        environment=dict(
            python=platform.python_version(), platform=platform.platform(),
            numpy=np.__version__, blas=blas_threads(),
            specfun=coulomb.chain_signature(), coulomb=coulomb.status()),
        calibration=json.loads(store_path().read_text(encoding="utf-8"))
        if store_path().exists() else {},
        blocks={k: v.get("arrays", {}) for k, v in snap.items()},
        block_meta={k: v.get("meta", {}) for k, v in snap.items()},
        figures=sorted(p.name for p in out_dir().glob("*.png")),
        tables=sorted(p.name for p in _tables_dir().glob("*.tex"))
        if _tables_dir().exists() else [],
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1, sort_keys=True), encoding="utf-8")
    return path


def report(systems=None, force="if-changed", compute=True, methods=None,
           blocks=None):
    """Configuration to final outputs, in one call and in order:

    compute -> draw -> tabulate -> record -> check. Tables use the same views as
    the figures; the manifest is written after both, so it describes what is on
    disk; the guardrail runs last. Returns a dict of tables, never a status
    code, and is never silent about failure (a raising block in `run`, a
    non-drawable figure as its error text in `figures`, likewise `tables`).

    `compute=False` redraws from the cache as it stands. `blocks` restricts what
    is computed -- pass everything but "failure" to leave F7 alone.
    """
    keys = list(systems or SUITE)
    out = {}
    if compute:
        from .run import Run
        kw = dict(systems=keys, force=force)
        if methods is not None:
            kw["methods"] = list(methods)
        if blocks is not None:
            kw["blocks"] = list(blocks)
        out["run"] = Run(**kw).execute()
    out["figures"] = pd.DataFrame(
        [dict(recipe=k, result=str(v)) for k, v in all_figures().items()]
    ).set_index("recipe")
    # Tables cover the suite even when the run also computed the controls: p+12C
    # and alpha+d are correctness benchmarks with their own `_control` sheets,
    # not table columns. Only a run touching no suite system falls back to what
    # it did compute.
    out["tables"] = tables([k for k in keys if k in SUITE] or keys)
    out["manifest"] = manifest()
    out["check"] = check()
    return out
