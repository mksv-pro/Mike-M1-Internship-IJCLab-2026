"""The renderer: one view, one recipe, one PNG.

The only module that imports matplotlib. Everything above it produces numbers,
everything here produces ink; the seam lets one computation be drawn as a
four-column suite sheet or as one sheet per system.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from . import cache
from . import style
from . import views

OUT = Path("figures_out")


def out_dir():
    """Where figures go for the cache in force: figures_out/ for the default
    cache, figures_out/<name>/ for any other."""
    return OUT if not cache.name() else OUT / cache.name()

VIEWS = {"vs_rmatrix": views.validation,
         "potential": views.potential,
         "error_distribution": views.error_distribution,
         "cat_cloud": views.cat_cloud,
         "cat_cloud_inelastic": views.cat_cloud_inelastic,
         "wavefunction_vs_emulators": views.wavefunction_vs_emulators,
         "phase_shift_vs_emulators": views.phase_shift_vs_emulators,
         "manifold_spread": views.manifold_spread,
         "online_vs_mesh": views.online_vs_mesh,
         "amortisation": views.amortisation,
         "coupling_budget_failure": views.coupling_budget_failure,
         "observable_vs_emulators": views.observable_vs_emulators,
         "invariants": views.invariants,
         "error_by_quantity": views.error_by_quantity,
         "pod_spectrum": views.pod_spectrum,
         "error_by_rank": views.error_by_rank}

#: Which cached block each view reads. Declared, not inferred: `api.rebench()`
#: needs it to know which figures a re-measured timing invalidates. Every view
#: reads exactly one block (only tables cross blocks). test_outputs.py asserts
#: this against the keys the views actually touch.
VIEW_BLOCKS = {
    "vs_rmatrix": "validation",
    "potential": "sweep",
    "manifold_spread": "sweep",
    "invariants": "invariants",
    "pod_spectrum": "manifold",
    "error_by_rank": "manifold",
    "error_by_quantity": "manifold",
    "observable_vs_emulators": "excitation",
    "error_distribution": "excitation",
    "wavefunction_vs_emulators": "interior",
    "phase_shift_vs_emulators": "interior",
    "cat_cloud": "cat",
    "cat_cloud_inelastic": "cat",
    "amortisation": "cat",
    "online_vs_mesh": "scaling",
    "coupling_budget_failure": "failure",
}


def recipes_reading(blocks):
    """Recipe keys whose view reads one of `blocks`. What a rebench must redraw."""
    from .recipes import RECIPES
    wanted = set(blocks)
    return sorted(k for k in RECIPES
                  if VIEW_BLOCKS.get(k.split(".")[0]) in wanted)

# Line weights for figures that overlay every system in one panel. There the
# colour is already spoken for -- it identifies the system -- so a curve's own
# identity has to come from its dash and its width.
WEIGHT_KW = {
    "thin":      dict(lw=0.8, ls="-", alpha=0.55),
    "thick":     dict(lw=1.8, ls="-"),
    "marked":    dict(lw=1.6, ls="-", marker="o", ms=4.0),
    "achieved":  dict(lw=1.8, ls="-"),
    "predicted": dict(lw=1.2, ls="--", alpha=0.8),
}

# A fourth colour channel, for figures whose curves are quantities rather than
# methods (style.py has three: method, system, swept parameter).
QUANTITY_KW = {
    "norm":      dict(color=style.GREY, lw=1.5, ls="-"),
    "elastic":   dict(color=style.BLUE, lw=1.6, ls="-"),
    "inelastic": dict(color=style.WINE, lw=1.6, ls="-"),
    "sigma":     dict(color=style.BLACK, lw=1.7, ls="--"),
    # the p5-p95 box envelope behind the sweep ribbon: not a curve, the region
    # the curves live in.
    "box":       dict(color=style.GREY, lw=1.0, ls="--", alpha=0.9),
    # non-entrance diagonal terms, thin so they read as context.
    "thin_grey": dict(color=style.GREY, lw=0.8, ls="-", alpha=0.45),
    # affine decomposition of the interaction: two thick totals, the thin terms
    # that sum to them, and the Coulomb tail.
    "total_re":  dict(color=style.BLACK, lw=2.0, ls="-"),
    "total_im":  dict(color=style.BRICK, lw=1.8, ls="--"),
    "term_a":    dict(color=style.BLUE, lw=1.0, ls="--", alpha=0.9),
    "term_b":    dict(color=style.OCHRE, lw=1.0, ls="--", alpha=0.9),
    "term_c":    dict(color=style.SAGE, lw=1.0, ls="-.", alpha=0.9),
    "term_d":    dict(color=style.MAUVE, lw=1.0, ls="-.", alpha=0.9),
    "term_e":    dict(color=style.GREY, lw=1.0, ls=":", alpha=0.9),
    "term_f":    dict(color=style.BLUE, lw=1.2, ls=(0, (3, 1, 1, 1)), alpha=0.9),
    "term_g":    dict(color=style.SAGE, lw=1.2, ls=(0, (5, 1)), alpha=0.9),
    "term_h":    dict(color=style.MAUVE, lw=1.2, ls=(0, (1, 1)), alpha=0.9),
    "coulomb":   dict(color=style.WINE, lw=1.6, ls=":"),
    # up to six configurations on one CAT sheet: colour AND marker differ, so
    # the clouds stay separable where they overlap.
    "combo_0":   dict(color=style.BLUE, marker="o"),
    "combo_1":   dict(color=style.WINE, marker="s"),
    "combo_2":   dict(color=style.SAGE, marker="^"),
    "combo_3":   dict(color=style.MAUVE, marker="D"),
    "combo_4":   dict(color=style.BRICK, marker="v"),
    "combo_5":   dict(color=style.MIST, marker="P"),
    "rank_lo":   dict(color=style.GREY, lw=1.3, ls=":"),
    "rank_mid":  dict(color=style.BLUE, lw=1.5, ls="--"),
    "rank_hi":   dict(color=style.BLACK, lw=1.8, ls="-"),
    # individual configurations behind an envelope: same hue, thin and faint.
    "rbm_faint":  dict(color=style.BLUE, lw=0.7, ls="-", alpha=0.35),
    "lrom_faint": dict(color=style.WINE, lw=0.7, ls="-", alpha=0.35),
    "lrom_split":      dict(color=style.BLUE, lw=1.6, ls="-."),
    "lrom_dash":       dict(color=style.WINE, lw=1.3, ls="--", alpha=0.9),
    "lrom_split_dash": dict(color=style.BLUE, lw=1.3, ls="--", alpha=0.9),
}


def _kw(name):
    """Ink only. The label belongs to the series, never to the style.

    style.METHOD carries a label beside its colour and dash, so it is stripped
    here: a view names its curves, a style says how they are drawn.
    """
    if name in QUANTITY_KW:
        kw = dict(QUANTITY_KW[name])
    else:
        try:
            kw = style.method_kw(name)
        except KeyError:
            kw = {}
    kw.pop("label", None)
    return kw


def mark_thresholds(ax, thresholds, labels=None, min_gap=0.045):
    """Thresholds, with labels that do not overprint each other.

    All lines are drawn, but a label is skipped when it would collide with the
    previous one (n+238U's four thresholds sit within 2% of the axis). The
    highest threshold's label always wins.
    """
    thresholds = [t for t in np.atleast_1d(thresholds) if t > 0]
    if not thresholds:
        return
    lo, hi = ax.get_xlim()
    span = hi - lo
    order = sorted(range(len(thresholds)), key=lambda i: thresholds[i])
    keep, last = [], -np.inf
    for i in reversed(order):                      # from the top down
        t = thresholds[i]
        if not (lo < t < hi):
            continue
        if last - t >= min_gap * span or not keep:
            keep.append(i)
            last = t
    for i in order:
        t = thresholds[i]
        if not (lo < t < hi):
            continue
        ax.axvline(t, color=style.GREY, ls="--", lw=0.8, alpha=0.75, zorder=1)
        if labels is not None and i < len(labels) and i in keep:
            ax.text(t, 1.005, labels[i], transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=6.5, color=style.INK)
    ax.set_xlim(lo, hi)


def _duration(seconds):
    """Seconds as a human reads them: 82800 s -> 23 h."""
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min"
    if seconds < 172800:
        return f"{seconds / 3600:.1f} h"
    return f"{seconds / 86400:.1f} days"


_CONFIG_MARKERS = ("o", "s", "^", "D", "v", "P", "X", "p", "*", "h")


def _draw_sweep(ax, sweeps):
    """A parameter sweep as filled ribbons, not a stack of lines.

    The band between consecutive values reads as one object deforming
    continuously, which is what a sweep is; the lines stay on top so an
    individual value can still be followed.
    """
    ramp = style.sweep_colors(len(sweeps))
    # low fill opacity so that where two consecutive curves cross their bands
    # read as overlapping; the lines underneath are unambiguous.
    for i in range(len(sweeps) - 1):
        ax.fill_between(sweeps[i].x, sweeps[i].y, sweeps[i + 1].y,
                        color=ramp[i], alpha=0.35, lw=0, zorder=2)
    for i, s in enumerate(sweeps):
        ax.plot(s.x, s.y, color=ramp[i], lw=1.1, alpha=0.95, zorder=3)


def _draw_series(ax, s, marker=None):
    # Nothing with style "sweep" reaches here: _render_columns pulls those out
    # for _draw_sweep, the only thing that knows the ramp.
    kw = _kw(s.style) if s.style else {}
    if marker is not None:
        kw["marker"] = marker
    if s.role == "residual":
        colour = kw.get("color", style.BRICK)
        if s.band is not None:
            ax.fill_between(s.x, s.band[0], s.band[1], color=colour,
                            alpha=0.18, lw=0, zorder=2)
        ax.plot(s.x, s.y, color=colour, lw=1.5, zorder=3)
    elif s.role == "cloud":
        # alpha-layered scatter: the whole cloud at low opacity so density shows
        # without a KDE, plus a deterministic tenth at full opacity for legibility.
        colour = kw.get("color", style.GREY)
        marker = kw.get("marker", "o")
        n = len(s.x)
        hi = np.zeros(n, dtype=bool)
        hi[::max(int(1 / 0.10), 1)] = True
        ax.scatter(s.x[~hi], s.y[~hi], color=colour, marker=marker, s=10,
                   alpha=0.10, linewidths=0, zorder=2)
        ax.scatter(s.x[hi], s.y[hi], color=colour, marker=marker, s=14,
                   alpha=1.0, linewidths=0, zorder=3, label=s.label)
        if s.band is not None and len(s.band) == 1 and s.band[0] < 1.0:
            ax.scatter([10 ** np.mean(np.log10(s.x))],
                       [10 ** np.mean(np.log10(s.y))],
                       s=110, facecolors="none", edgecolors=style.BRICK,
                       lw=1.3, marker="s", zorder=6)
    elif s.role == "dominated":
        # tried, kept on the sheet, not the thing to read.
        ax.plot(s.x, s.y, linestyle="none", marker="o", ms=4.0,
                mfc="none", mec=kw.get("color", style.GREY), mew=0.9,
                alpha=0.45, zorder=2)
    elif s.role == "front":
        colour = kw.get("color", style.GREY)
        ax.plot(s.x, s.y, color=colour, lw=1.3, ls="-", marker="o", ms=6.5,
                mfc=colour, mec="white", mew=0.8, label=s.label, zorder=4)
        for x, y, text in zip(s.x, s.y, s.point_labels or []):
            ax.annotate(text, (x, y), textcoords="offset points",
                        xytext=(0, 7), ha="center", fontsize=6.0, color=colour)
    elif s.role == "scatter":
        colour = kw.get("color", style.GREY)
        ax.plot(s.x, s.y, linestyle="none", marker=kw.get("marker", "o"),
                ms=7.0, mfc=colour, mec="white", mew=0.7,
                color=colour, label=s.label, zorder=3)
        # an underdetermined fit stays on the figure but is ringed, so it is
        # not read as an operating point.
        if s.band is not None and len(s.band) == 1 and s.band[0] < 1.0:
            ax.plot(s.x, s.y, linestyle="none", marker="s", ms=12.0,
                    mfc="none", mec=style.BRICK, mew=1.4, zorder=4)
    elif s.role == "shape":
        # a form factor, drawn for its shape: both curves normalised to their
        # own peak, the factor between them stated in the label.
        colour = kw.get("color", style.GREY)
        ax.fill_between(s.x, 0.0, s.y, color=colour, alpha=0.20, lw=0, zorder=1)
        ax.plot(s.x, s.y, color=colour, lw=1.0, alpha=0.7, label=s.label,
                zorder=2)
    elif s.role == "ticks":
        # predictor points as a rug; y only separates the two budgets into rows.
        if s.x.size:
            ax.plot(s.x, s.y, linestyle="none", marker="|", ms=11, mew=1.8,
                    color=kw.get("color", style.GREY), label=s.label, zorder=4)
    elif s.role == "span":
        # p5-p95 as a bar, median as a tick: two populations of matrix elements,
        # drawn where they live rather than reduced to a ratio.
        colour = kw.get("color", style.GREY)
        ax.plot(s.x, s.y, color=colour, lw=5.0, alpha=0.35,
                solid_capstyle="butt", zorder=2)
        if s.band is not None and np.isfinite(s.band[0]):
            ax.plot([s.band[0]], [s.y[0]], marker="|", ms=14, mew=2.0,
                    color=colour, zorder=3)
    else:
        # a band belongs to whichever curve carries one, not only residuals:
        # each point is a median over validation draws and the spread matters.
        if s.band is not None and len(s.band) == 2:
            ax.fill_between(s.x, s.band[0], s.band[1],
                            color=kw.get("color", style.GREY),
                            alpha=0.16, lw=0, zorder=1)
        ax.plot(s.x, s.y, label=s.label, **kw)


def _render_overlay(recipe, system_keys, methods=None):
    """One panel per quantity, every system drawn into each of them.

    Answers "how do the systems sit relative to one another" by putting them on
    one axis: colour means system, dash means quantity.
    """
    panels = {k: VIEWS[recipe.key.split(".")[0]](k) for k in system_keys}
    ncol = len(recipe.rows)
    fig, axes = plt.subplots(1, ncol, figsize=recipe.figsize,
                             constrained_layout=True, squeeze=False)

    for col, name in enumerate(recipe.rows):
        ax = axes[0][col]
        # a label that repeats across systems belongs to the quantity: name it once.
        labelled = set()
        for key in system_keys:
            data = panels[key][name]
            colour = style.SYSTEM_COLOR.get(key, style.GREY)
            for s in _keep(recipe, data.series, methods):
                kw = dict(WEIGHT_KW.get(s.style, {}))
                kw["color"] = colour
                # sparse markers (style.SYSTEM_MARKER) on the leading weight
                # only, so the thin companion curve stays unmarked.
                if kw.get("lw", 1.0) >= 1.2:
                    kw.update(marker=style.SYSTEM_MARKER.get(key, ""),
                              markevery=max(len(s.x) // 6, 1), ms=3.6,
                              mfc="white", mew=0.9)
                if s.band is not None and len(s.band) == 2:
                    ax.fill_between(s.x, s.band[0], s.band[1], color=colour,
                                    alpha=0.14, lw=0, zorder=1)
                label = s.label
                if label in labelled:
                    label = None
                elif label:
                    labelled.add(label)
                ax.plot(s.x, s.y, label=label, **kw)
        # the criterion, per system: after the curves so the marks sit on top,
        # before the limits are read so the label lands inside the axes.
        vkey = recipe.vlines.get(name)
        if vkey:
            for key in system_keys:
                value = panels[key][name].notes.get(vkey)
                if value is not None and np.isfinite(value):
                    ax.axvline(value, color=style.SYSTEM_COLOR.get(
                        key, style.GREY),
                        ls=":", lw=1.1, alpha=0.9, zorder=4)
        hspec = recipe.hlines.get(name)
        if hspec:
            value = panels[system_keys[0]][name].notes.get(hspec[0])
            if value is not None and np.isfinite(value):
                ax.axhline(value, color=style.BRICK, ls="--", lw=1.3,
                           alpha=0.9, zorder=4)
                ax.text(0.985, value, hspec[1], transform=ax.get_yaxis_transform(),
                        ha="right", va="bottom", fontsize=6.5,
                        color=style.BRICK)

        spec = recipe.end_labels.get(name)
        if spec:
            for key in system_keys:
                data = panels[key][name]
                value = data.notes.get(spec[0])
                if value is None or not data.series:
                    continue
                s = data.series[0]
                ax.annotate(spec[1].format(value), (s.x[-1], s.y[-1]),
                            textcoords="offset points", xytext=(5, 0),
                            fontsize=6.5, va="center",
                            color=style.SYSTEM_COLOR.get(key,
                                                         style.GREY))

        focus = [panels[k][name].notes.get("x_focus") for k in system_keys]
        focus = [f for f in focus if f is not None and np.isfinite(f)]
        if focus:
            ax.set_xlim(left=0, right=max(focus))

        first = panels[system_keys[0]][name]
        ax.set_title(first.title or "", fontsize=9)
        ax.set_xlabel(first.x_label)
        ax.set_ylabel(first.y_label)
        if recipe.y_scale and recipe.y_scale[col] == "log":
            ax.set_yscale("log")
        if recipe.x_scale:
            _set_x_scale(ax, recipe.x_scale[col], first)
        # overlay floor is the lowest any system asks for.
        floors = [panels[k][name].notes.get("y_floor") for k in system_keys]
        floors = [f for f in floors if f and np.isfinite(f)]
        if floors and ax.get_yscale() == "log":
            ax.set_ylim(bottom=min(floors))
        if name in recipe.tolerance_ladder:
            style.draw_tolerance_ladder(ax, which=("exp", "post"))
        if col == recipe.legend_in:
            leg = ax.legend(loc="best", fontsize=6.5, frameon=False)
            # the weight legend as its own artist so it does not interleave with
            # the system legend.
            if recipe.weight_legend:
                handles = [Line2D([], [], color=style.GREY,
                                  **{k: v for k, v in WEIGHT_KW.get(sk, {}).items()
                                     if k != "color"}, label=lab)
                           for sk, lab in recipe.weight_legend.items()]
                if vkey:
                    handles.append(Line2D([], [], color=style.GREY, ls=":",
                                          lw=1.1, label="selected rank"))
                ax.add_artist(leg)
                # unframed, placed in the empty corner made by the y_floor crop.
                ax.legend(handles=handles,
                          loc=recipe.legend_loc.get(name, "lower left"),
                          fontsize=6.5, frameon=False,
                          borderpad=0.3).set_zorder(6)

    if recipe.suptitle:
        fig.suptitle(recipe.suptitle, fontsize=11)
    # same one-line provenance as a column sheet; here "(by column)" means the
    # legend order, which is the system_keys order.
    text = _provenance_line(panels, system_keys, recipe.caption)
    if text:
        fig.supxlabel(_wrap_banner(text, recipe.figsize),
                      fontsize=6.6, color=style.INK)
    out = out_dir() / f"{recipe.stem}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


#: Style keys that name an emulator, as opposed to a reference or a quantity.
METHOD_STYLES = ("rbm", "lrom", "rbm_et", "rbm_etp")


def _keep(recipe, series, methods=None):
    """Series admitted by the recipe, or by a call-time override.

    `methods` is the per-generation display filter (RBM alone to stay readable,
    or everything to compare); the recipe supplies the default. References are
    never filtered.
    """
    wanted = tuple(methods) if methods is not None else recipe.methods
    if not wanted:
        return list(series)
    return [s for s in series
            if s.style in wanted or s.style not in METHOD_STYLES]


def audit(recipe, system_keys, methods=None):
    """What a generation would draw, what it would leave out, and why.

    Returned as rows for a table. Catches a method asked for and silently
    absent, which draws as an empty panel that looks converged.
    """
    view = VIEWS[recipe.key.split(".")[0]]
    wanted = list(methods if methods is not None else (recipe.methods or ()))
    rows = []
    for key in system_keys:
        try:
            panels = view(key)
        except Exception as exc:
            rows.append(dict(system=key, method="-", state="no data",
                             why=f"{type(exc).__name__}: {exc}"[:60]))
            continue
        present = {}
        for panel in panels.values():
            for s in panel.series:
                if s.style in METHOD_STYLES:
                    ok = bool(np.isfinite(np.asarray(s.y, dtype=float)).any())
                    present[s.style] = present.get(s.style, False) or ok
        for m in (wanted or sorted(present)):
            if present.get(m) is True:
                rows.append(dict(system=key, method=m, state="drawn", why=""))
            elif m in present:
                rows.append(dict(system=key, method=m, state="empty",
                                 why="all NaN: not computed for this system"))
            else:
                rows.append(dict(system=key, method=m, state="absent",
                                 why="this view has no series for it"))
    return rows


def render(recipe, system_keys, methods=None):
    if recipe.per_system:
        return [_render_columns(recipe, [k], methods) for k in system_keys]
    if recipe.overlay_systems:
        return _render_overlay(recipe, system_keys, methods)
    return _render_columns(recipe, system_keys, methods)


def _set_x_scale(ax, kind, data=None):
    """Apply a recipe's x scale, including the compressed-log of the CAT sheets.

    On a cost-accuracy cloud the useful data occupies one decade and the
    references reach two decades beyond it, so a plain log axis spends most of
    its width on the gap. "sqrtlog" keeps a log axis but re-parametrises it as
    u = sqrt(log10(x/x0)): monotone, so no point moves past another, and its
    derivative falls as x grows, so the cloud is stretched and the distance to
    the R-matrix is compressed without being hidden. The references stay in
    frame, which is the point of drawing them.
    """
    if kind == "log":
        ax.set_xscale("log")
    elif kind == "sqrtlog":
        x0 = getattr(data, "notes", {}).get("x_floor") if data else None
        x0 = float(x0) if x0 else 1e-3

        def fwd(x):
            x = np.asarray(x, dtype=float)
            return np.sqrt(np.maximum(np.log10(np.maximum(x, x0 * 1e-6) / x0), 0.0))

        def inv(u):
            u = np.asarray(u, dtype=float)
            return x0 * 10.0 ** (np.maximum(u, 0.0) ** 2)

        ax.set_xscale("function", functions=(fwd, inv))
        # decade ticks on the underlying time, so the labels stay milliseconds.
        lo, hi = ax.get_xlim()
        decades = np.arange(np.floor(np.log10(max(lo, x0))),
                            np.ceil(np.log10(max(hi, x0 * 10))) + 1)
        ax.set_xticks(10.0 ** decades)
        ax.set_xticks([], minor=True)
        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: (f"{v:g}" if v >= 0.1 else f"{v:.2g}")))


def _wrap_banner(text, figsize):
    """Wrap the provenance banner to the sheet width, so a one-column sheet does
    not inherit a four-column line length."""
    import textwrap
    nl = chr(10)
    width = max(58, int(15.0 * figsize[0]))
    return nl.join(nl.join(textwrap.wrap(part, width)) if part else ""
                   for part in text.split(nl))


def _provenance_line(panels, system_keys, caption=None):
    """One line. A field every system agrees on is printed once; a field that
    varies becomes a slash-separated list in column order (across the suite only
    N and E actually differ)."""
    records = []
    for key in system_keys:
        for panel in panels[key].values():
            if panel.provenance:
                records.append(panel.provenance)
                break
    if not records:
        return caption or ""
    names = list(records[0])
    for r in records[1:]:                       # only fields every system has
        names = [n for n in names if n in r]
    parts, varies = [], False
    for n in names:
        values = [str(r[n]) for r in records]
        if len(set(values)) == 1:
            parts.append(f"{n}={values[0]}")
        else:
            parts.append(f"{n}=" + "/".join(values))
            varies = True
    line = "   ".join(parts) + ("   (by column)" if varies else "")
    return f"{caption}\n{line}" if caption else line


def _draw_provenance(fig, recipe, panels, system_keys):
    """The numbers the figure was produced under, in one line under the sheet.

    From the block's metadata via PanelData.provenance, so it cannot drift from
    what was computed.
    """
    text = _provenance_line(panels, system_keys, recipe.caption)
    if text:
        fig.supxlabel(_wrap_banner(text, recipe.figsize),
                      fontsize=6.6, color=style.INK)


def _render_columns(recipe, system_keys, methods=None):
    panels = {k: VIEWS[recipe.key.split(".")[0]](k) for k in system_keys}
    nrow, ncol = len(recipe.rows), len(system_keys)
    fig, axes = plt.subplots(
        nrow, ncol, figsize=recipe.figsize, constrained_layout=True,
        squeeze=False, sharey="row" if recipe.share_y else False,
        gridspec_kw={"height_ratios": recipe.height_ratios or None})

    legended = set()
    for col, key in enumerate(system_keys):
        for row, row_name in enumerate(recipe.rows):
            ax = axes[row][col]
            data = panels[key][row_name]
            keep = _keep(recipe, data.series, methods)
            sweeps = [s for s in keep if s.style == "sweep"]
            if sweeps:
                _draw_sweep(ax, sweeps)
                keep = [s for s in keep if s.style != "sweep"]
            seen = {}
            for s in keep:
                mk = None
                if s.role == "scatter":
                    n = seen.get(s.style, 0)
                    seen[s.style] = n + 1
                    mk = _CONFIG_MARKERS[n % len(_CONFIG_MARKERS)]
                _draw_series(ax, s, marker=mk)
            if recipe.y_scale and recipe.y_scale[row] == "log":
                ax.set_yscale("log")
            if recipe.x_scale:
                _set_x_scale(ax, recipe.x_scale[row], data)
            # a view that knows its own data can say where the log-axis floor
            # belongs, so a curve going to zero does not waste half the panel.
            floor = data.notes.get("y_floor")
            if floor and np.isfinite(floor) and ax.get_yscale() == "log":
                ax.set_ylim(bottom=float(floor))
            if row == 0 and data.title:
                ax.set_title(data.title)
            if col == 0 and data.y_label:
                ax.set_ylabel(data.y_label)
            # label the bottom row, and any row whose x differs from the row
            # below (F7 stacks three beta_2 panels above one in r).
            below = (panels[key][recipe.rows[row + 1]].x_label
                     if row < nrow - 1 else None)
            if row == nrow - 1 or data.x_label != below:
                ax.set_xlabel(data.x_label)
            # the campaign cost at a chosen horizon, in wall-clock units.
            if "horizon" in data.notes:
                h = data.notes["horizon"]
                ax.axvline(h, color=style.GREY, ls=":", lw=1.0, zorder=1)
                lines = []
                # loop variable is `note`, not `key`: this is inside the
                # `for col, key in enumerate(system_keys)` loop.
                for note, tag in (("rmat_horizon", "R-matrix"),
                                  ("dbmm_full_horizon", "DBMM rebuild"),
                                  ("t_dbmm_horizon", "DBMM, $V$ only"),
                                  ("t_rbm_horizon", "RBM"),
                                  ("t_lrom_horizon", "LROM")):
                    v = data.notes.get(note)
                    if v is not None and np.isfinite(v):
                        lines.append(f"{tag} {_duration(v)}")
                if lines:
                    head = "at $N=10^{%d}$:" % int(np.log10(h))
                    ax.text(0.97, 0.03, "\n".join([head] + lines),
                            transform=ax.transAxes,
                            ha="right", va="bottom", fontsize=6.0,
                            color=style.INK, linespacing=1.35,
                            # opaque strip so the horizon rule does not cross the text.
                            bbox=dict(facecolor="white", edgecolor="none",
                                      alpha=0.82, pad=1.6))
            if {"t_dbmm_lo", "t_dbmm_hi"} <= set(data.notes):
                # the solver's cost as a band: a single line would overclaim
                # the measurement's precision.
                ax.axvspan(data.notes["t_dbmm_lo"], data.notes["t_dbmm_hi"],
                           color=style.GREY, alpha=0.12, zorder=0)
                ax.axvline(data.notes["t_dbmm"], color=style.GREY, ls="--",
                           lw=1.0, alpha=0.7, zorder=1)
                ax.text(data.notes["t_dbmm"], 0.985,
                        f"DBMM, $V$ only  {data.notes['t_dbmm']:.2f} ms",
                        transform=ax.get_xaxis_transform(), rotation=90,
                        ha="right", va="top", fontsize=6.2, color=style.INK)
                # the other two solver costs as thin lines, not bands: three
                # spans would swallow the cloud.
                for note, text in (("t_rebuild", "DBMM, full rebuild"),
                                   ("t_rmat", "R-matrix")):
                    if note not in data.notes:
                        continue
                    ax.axvline(data.notes[note], color=style.GREY, ls=":",
                               lw=0.9, alpha=0.85, zorder=1)
                    ax.text(data.notes[note], 0.985,
                            f"{text}  {data.notes[note]:.2f} ms",
                            transform=ax.get_xaxis_transform(), rotation=90,
                            ha="right", va="top", fontsize=6.2, color=style.INK)
            if recipe.mark_thresholds:
                mark_thresholds(ax, data.thresholds, data.threshold_labels)
            if row_name in recipe.tolerance_ladder:
                # rules on every panel, text on the first column only.
                style.draw_tolerance_ladder(ax, which=("exp", "post"),
                                            label=(col == 0))

            spans = [s for s in data.series if s.role == "span"]
            if spans:
                ax.set_ylim(-0.7, 1.7)
                ax.set_yticks([s.y[0] for s in spans])
                ax.set_yticklabels([s.label for s in spans] if col == 0
                                   else ["" for _ in spans], fontsize=6.5)
                ax.grid(axis="x", alpha=0.15, lw=0.6)

            hspec = recipe.hlines.get(row_name)
            if hspec and hspec[0] in data.notes:
                v = data.notes[hspec[0]]
                if np.isfinite(v):
                    ax.axhline(v, color=style.BLACK, ls="-", lw=1.4, zorder=4)
                    ax.text(0.98, v, hspec[1], transform=ax.get_yaxis_transform(),
                            ha="right", va="bottom", fontsize=6.2,
                            color=style.BLACK)
            vspec = recipe.vlines.get(row_name)
            if vspec and vspec[0] in data.notes:
                v = data.notes[vspec[0]]
                if np.isfinite(v):
                    # grey dashed reference mark, not data.
                    ax.axvline(v, color=style.GREY, ls="--", lw=1.0,
                               alpha=0.85, zorder=1)
                    # label inside the axes, low and rotated, on an opaque strip
                    # (above collides with the column title, top-left with the
                    # legend).
                    ax.text(v, 0.02, vspec[1].format(v) if "{" in vspec[1]
                            else vspec[1], transform=ax.get_xaxis_transform(),
                            ha="left", va="bottom", fontsize=6.0,
                            color=style.INK, rotation=90, zorder=6,
                            bbox=dict(boxstyle="square,pad=0.12", fc="white",
                                      ec="none", alpha=0.92))

            if row_name in recipe.orientation:
                ax.text(0.03, 0.10, recipe.orientation[row_name],
                        transform=ax.transAxes, fontsize=6.2,
                        color=style.INK, va="bottom")

            spec = recipe.headlines.get(row_name)
            if spec and spec[0] in data.notes:
                value = data.notes[spec[0]]
                if np.isfinite(value):
                    # optional third element of `spec` overrides the corner.
                    style.headline(ax, spec[1].format(value),
                                   loc=(spec[2] if len(spec) > 2 else "lower left"),
                                   fontsize=6.5)
    # Legends, once every panel is drawn. Handles are collected across the whole
    # ROW (a column may legitimately draw fewer curves than its neighbours), for
    # the top row and any later row that introduces new curves.
    legended = set()
    for row, row_name in enumerate(recipe.rows):
        # what was drawn WITH a label; first column to draw one wins the handle.
        union = {}
        for col in range(ncol):
            handles, labels = axes[row][col].get_legend_handles_labels()
            for handle, label in zip(handles, labels):
                union.setdefault(label, handle)
        if not union:
            continue
        if not (recipe.legend_all_columns
                or any(l not in legended for l in union)):
            continue
        targets = (range(ncol) if recipe.legend_all_columns
                   else [recipe.legend_in])
        for col in targets:
            if 0 <= col < ncol:
                axes[row][col].legend(
                    list(union.values()), list(union),
                    loc=recipe.legend_loc.get(row_name, "best"),
                    fontsize=6.5, frameon=False)
        legended.update(union)

    if recipe.suptitle:
        fig.suptitle(recipe.suptitle, fontsize=11)

    # One colorbar for the sheet, carrying DIRECTION only (low box to high):
    # each system sweeps its own parameter over its own bounds, so one numeric
    # scale across the sheet would be wrong. Per-column bounds are in the footer.
    swept = next((p for k in system_keys for p in panels[k].values()
                  if any(s.style == "sweep" for s in p.series)), None)
    if swept is not None:
        cb = style.sweep_colorbar(
            fig, [axes[r][-1] for r in range(nrow)], 0.0, 1.0,
            f"{swept.notes.get('label', 'swept')}, each within its own box",
            fraction=0.06)
        cb.set_ticks([0.0, 1.0])
        cb.set_ticklabels(["low", "high"])

    _draw_provenance(fig, recipe, panels, system_keys)

    if recipe.per_system:
        # flat, not one directory per system: the system is in the file name,
        # and a flat figures_out/ can be deleted and fully regenerated.
        out = out_dir() / f"{system_keys[0]}_{recipe.stem}.png"
    else:
        out = out_dir() / f"{recipe.stem}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out
