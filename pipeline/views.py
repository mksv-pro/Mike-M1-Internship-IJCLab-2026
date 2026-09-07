"""Cached arrays in, plot-ready series out. No matplotlib below this line.

A view knows physics: that this array is a relative error, that this axis is an
energy, that a residual is summarised by a median and a p5-p95 band and never by
a mean. It does not know how many panels the figure has, whether the y axis is
logarithmic, or which series the reader looks at first -- those are the recipe's
business. Keeping them apart makes a per-figure preference a one-line change,
and lets one view feed two recipes.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

import warnings

import numpy as np

from . import cache
from .systems import SYSTEMS


def _nanmin(x):
    """Smallest finite value, NaN when there is none. See `spread` below for
    why an all-NaN array is an expected input rather than a symptom."""
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)
    return float(np.min(x[finite])) if finite.any() else float("nan")


@dataclass
class Series:
    """One curve, plus what the recipe needs to place it."""
    x: np.ndarray
    y: np.ndarray
    role: str                      # "reference" | "comparison" | "residual"
    label: Optional[str] = None
    band: Optional[tuple] = None   # (low, high) envelope, already computed
    style: str = ""                # method key for style lookup
    # one short string per point, annotated beside it -- a cost-accuracy front is
    # a set of configurations, and a per-point label beats a fifteen-entry legend.
    point_labels: Optional[Sequence[str]] = None


@dataclass
class PanelData:
    """Everything one panel of one system needs."""
    key: str
    title: str
    series: Sequence[Series] = field(default_factory=list)
    x_label: str = ""
    y_label: str = ""
    thresholds: Sequence[float] = field(default_factory=list)
    threshold_labels: Sequence[str] = field(default_factory=list)
    notes: Dict[str, Any] = field(default_factory=dict)
    # the numbers the curves were produced under, as one line under the column,
    # built by _provenance below from the block's own metadata.
    provenance: dict = field(default_factory=dict)


def _provenance(system, meta, *fields):
    """{'N': 100, 'train': 300, 'valid': 40, 'E': '54'} -- a dict, not a string.

    The renderer assembles the line: it is the only thing that sees all systems
    at once, so a shared field prints once and only a varying one becomes a list.
    `fields` names meta keys to include, in order; an absent key is skipped, not
    printed as None.
    """
    short = {"n_train": "train", "n_train_rbm": "train", "n_valid": "valid",
             "n_theta": "theta", "n_repeat": "rep", "n_sweep": "sweep",
             "n_box": "box", "nb_probe": "nb", "K_total": "K",
             "n_train_lrom": "trainL", "n_resid": "resid",
             "nb": "nb", "K_diag": "Kd", "K_coup": "Kc"}
    out = {"N": system.N}
    # whether the hyperparameters were autocalibrated or hand-written, never
    # omitted when the block recorded it.
    if meta.get("calibrated"):
        out["hyperpar"] = "autocalibrated"
    for name in fields:
        value = meta.get(name)
        if value is None:
            continue
        if isinstance(value, float):
            value = f"{value:g}"
        elif isinstance(value, (list, tuple)):
            value = "[" + ",".join(str(v) for v in value) + "]"
        out[short.get(name, name)] = value
    return out


def _band(samples):
    """Median and p5-p95, the project's convention: these residuals span five
    decades, so a mean sits on the largest outlier."""
    samples = np.asarray(samples, dtype=float)
    if samples.ndim == 1:
        samples = samples[None, :]
    with np.errstate(invalid="ignore"):
        return (np.nanmedian(samples, axis=0),
                np.nanpercentile(samples, 5, axis=0),
                np.nanpercentile(samples, 95, axis=0))


def validation(system_key):
    """DBMM against the R-matrix on sigma(E), plus the residual distribution.

    Headline is the MEDIAN residual: the worst on S is set by whichever matrix
    element is nearest zero (9e-02 on alpha+24Mg where sigma agrees to 1e-06).
    The worst is still reported, next to the quantity it belongs to.
    """
    system = SYSTEMS[system_key]
    d, meta = cache.read(system_key, "validation")

    prov = _provenance(system, meta, "n_theta", "n_E")

    E = np.asarray(d["E"], dtype=float)
    sig_d = np.asarray(d["sigma_dbmm"], dtype=float)
    sig_r = np.asarray(d["sigma_rmat"], dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        resid = np.abs(sig_d - sig_r) / np.abs(sig_r)
    med, lo, hi = _band(resid)

    S_d, S_r = np.asarray(d["S_dbmm"]), np.asarray(d["S_rmat"])
    with np.errstate(divide="ignore", invalid="ignore"):
        err_S = (np.abs(S_d - S_r).max(axis=(2, 3))
                 / np.abs(S_d).max(axis=(2, 3)))
    psi_d, psi_r = np.asarray(d["psi_dbmm"]), np.asarray(d["psi_rmat"])
    with np.errstate(divide="ignore", invalid="ignore"):
        err_psi = (np.abs(psi_d - psi_r).max(axis=1)
                   / np.abs(psi_d).max(axis=1))

    # row 0 is theta_c: sample_theta puts it first
    main = PanelData(
        key=system_key, title=system.label,
        x_label=r"$E_{\rm c.m.}$ (MeV)", y_label=r"block contribution to $\sigma$ (fm$^2$)",
        thresholds=system.thresholds, threshold_labels=system.threshold_labels,
        series=[Series(E, sig_d[0], "reference", "DBMM", style="dbmm"),
                Series(E, sig_r[0], "comparison", "R-matrix", style="rmatrix")],
        notes=dict(median_sigma=float(np.nanmedian(resid)),
                   worst_sigma=float(np.nanmax(resid)),
                   median_S=float(np.nanmedian(err_S)),
                   worst_S=float(np.nanmax(err_S)),
                   median_psi=float(np.nanmedian(err_psi)),
                   n_theta=meta.get("n_theta"), n_E=meta.get("n_E")),
        provenance=prov,
    )
    residual = PanelData(
        key=system_key, title="",
        x_label=r"$E_{\rm c.m.}$ (MeV)", y_label=r"rel. residual on $\sigma$",
        thresholds=system.thresholds, threshold_labels=system.threshold_labels,
        series=[Series(E, med, "residual", band=(lo, hi), style="rmatrix")],
        notes=dict(median=float(np.nanmedian(resid))),
        provenance=prov,
    )
    return dict(main=main, residual=residual)


def pod_spectrum(system_key):
    """How fast the modes die, where the criterion cuts, and whether N_s suffices.

    sigma_k/sigma_1 measures how fast one mode dies; the truncation rule
    thresholds the accumulated discarded energy r(k), which is what eps_tol is
    compared against. Both are drawn, so the figure shows its own criterion.
    """
    system = SYSTEMS[system_key]
    d, meta = cache.read(system_key, "manifold")

    prov = _provenance(system, meta, "n_train", "n_valid", "nb_probe", "E")

    sv = np.asarray(d["singular_values"], dtype=float)
    disc = np.asarray(d["discarded"], dtype=float)
    k = np.arange(1, sv.size + 1, dtype=float)
    # from the block, never a literal: the criterion line and the rank beside it
    # must be read at the same tolerance.
    eps = float(meta.get("eps_sol", 1e-8))
    selected = int(np.argmax(disc < eps) + 1) if (disc < eps).any() else sv.size

    spectrum = PanelData(
        key=system_key,
        # no panel title: the sheet carries one, and thin/thick is the weight legend.
        title="",
        x_label="mode index $k$", y_label=r"$\sigma_k/\sigma_1$,  $r(k)$",
        series=[Series(k, sv / sv[0], "comparison", None, style="thin"),
                Series(k[:disc.size], np.maximum(disc, 1e-18), "comparison",
                       system.label, style="thick")],
        # x_focus / y_floor crop the axes: beyond a few times the selected rank,
        # and more than ~4 decades under the criterion, the curves are just the
        # machine-precision / SVD-noise floor.
        notes=dict(selected=selected, eps_tol=eps, n_modes=int(sv.size),
                   x_focus=3.0 * selected, y_floor=eps * 1e-4),
        provenance=prov,
    )

    ns = np.asarray(d["ns_scan"], dtype=float)
    nb_of_ns = np.asarray(d["nb_of_ns"], dtype=float)
    # The selected rank saturates: a tenfold increase in snapshots moves it by a
    # few per cent. Kept as an annotation on the spectrum -- the rank is a
    # property of the manifold, not an artefact of N_s.
    spectrum.notes["rank_drift"] = float(
        abs(nb_of_ns[-1] - nb_of_ns[0]) / max(nb_of_ns[-1], 1.0))
    spectrum.notes["ns_max"] = float(ns[-1])

    # On the cross section, not err_report: err_report is the inelastic error,
    # relative to matrix elements 3e-03 of the diagonal, so "does more training
    # help?" gets noise from a near-zero denominator.
    err_ns = np.asarray(d.get("err_sigma_of_ns", d["err_of_ns"]), dtype=float)
    if err_ns.ndim > 1:
        with np.errstate(invalid="ignore"):
            err_ns = np.nanmedian(err_ns.reshape(ns.size, -1), axis=1)
    # Measured at the operated rank, so the panel varies only the training size.
    # Both the probe rank and the criterion rank go in the label: the probe
    # alone cannot tell an under-resolved basis from a failing one.
    probe = meta.get("nb_probe", "?")
    error_vs_ns = PanelData(
        key=system_key,
        title="",   # the sheet title says it once
        x_label="training snapshots $N_s$",
        y_label=r"rel. error on $\sigma$ at that rank",
        series=[Series(ns, err_ns, "comparison",
                       rf"{system.label}  $n_b={probe}$ "
                       rf"(criterion: {selected})", style="marked")],
        notes=dict(nb_probe=probe, selected=selected, first=float(err_ns[0]),
                   last=float(err_ns[-1]),
                   under_resolved=float(probe) / selected if selected else float("nan")),
        provenance=prov,
    )
    return dict(spectrum=spectrum, error_vs_ns=error_vs_ns)


def error_by_rank(system_key):
    """Out-of-sample error against rank, beside what the criterion predicted.

    The criterion is about the training snapshots; the solid curve is what an
    unseen parameter costs. Where the gap is small the criterion can size a
    basis; where it is large it cannot.
    """
    system = SYSTEMS[system_key]
    d, meta = cache.read(system_key, "manifold")

    prov = _provenance(system, meta, "n_train", "n_valid", "E")

    nb = np.asarray(d["nb_scan"], dtype=float).ravel()
    v = np.asarray(d["err_global"], dtype=float)
    if v.ndim == 1:
        v = v.reshape(len(nb), -1)
    with np.errstate(invalid="ignore"):
        med = np.nanmedian(v, axis=1)
        lo = np.nanpercentile(v, 5, axis=1)
        hi = np.nanpercentile(v, 95, axis=1)

    disc = np.asarray(d["discarded"], dtype=float)
    idx = np.clip(nb.astype(int) - 1, 0, disc.size - 1)
    predicted = np.sqrt(np.maximum(disc[idx], 0.0))   # Eckart-Young, in norm

    return dict(error=PanelData(
        key=system_key, title="", x_label=r"rank $n_b$",
        y_label="rel. out-of-sample error",
        series=[Series(nb, med, "comparison", system.label, band=(lo, hi),
                       style="achieved"),
                # unlabelled: named once in the grey weight legend, not four
                # times in four system colours.
                Series(nb, predicted, "comparison", None, style="predicted")],
        # y_floor is set to the measured data: the Eckart-Young bound falls
        # 2-4 decades lower, and letting it leave the frame is the panel's point.
        notes=dict(n_valid=meta.get("n_valid"),
                   y_floor=float(np.nanmin(lo[np.isfinite(lo) & (lo > 0)])) * 0.3
                   if np.isfinite(lo).any() and (lo > 0).any() else None),
        provenance=prov,
    ))


def observable_vs_emulators(system_key):
    """The whole chain on the quantity astrophysics consumes: sigma(E).

    Reference as a line, R-matrix as sparse markers on top (the solver itself
    must be shown reproduced), emulators on the same axis, the row beneath
    turning the agreement into a number. Drawn at a VALIDATION parameter, never
    theta_c, where the LROM is exact by construction.
    """
    system = SYSTEMS[system_key]
    d, meta = cache.read(system_key, "excitation")

    prov = _provenance(system, meta, "n_train", "nb", "n_E", "n_resid")

    E = np.asarray(d["E"], dtype=float)
    idx = np.asarray(d["idx_rmat"], dtype=int)

    sigma = PanelData(
        key=system_key, title=system.label,
        x_label=r"$E_{\rm c.m.}$ (MeV)", y_label=r"block contribution to $\sigma$ (fm$^2$)",
        thresholds=system.thresholds, threshold_labels=system.threshold_labels,
        series=[Series(E, np.asarray(d["sigma_ref"])[0], "reference",
                       "DBMM", style="dbmm"),
                Series(E[idx], np.asarray(d["sigma_rmat"])[idx], "comparison",
                       "R-matrix", style="rmatrix"),
                Series(E, np.asarray(d["sigma_rbm"])[0], "comparison",
                       "RBM", style="rbm"),
                Series(E, np.asarray(d["sigma_lrom"])[0], "comparison",
                       "LROM", style="lrom"),
                ],
        notes=dict(n_resid=meta.get("n_resid"), nb=meta.get("nb")),
        provenance=prov,
    )

    def err(name, label, style_key):
        """Median and spread over the draws, silent where a method is undefined.

        RBM_ET is fitted per open-channel window, so its error is NaN over part
        of the grid by construction -- an all-NaN energy column is a column
        where the question does not apply.
        """
        v = np.asarray(d[name], dtype=float)
        defined = np.isfinite(v).any(axis=0)
        med = np.full(E.size, np.nan)
        lo = np.full(E.size, np.nan)
        hi = np.full(E.size, np.nan)
        if defined.any():
            with np.errstate(invalid="ignore"):
                med[defined] = np.nanmedian(v[:, defined], axis=0)
                lo[defined] = np.nanpercentile(v[:, defined], 5, axis=0)
                hi[defined] = np.nanpercentile(v[:, defined], 95, axis=0)
        return Series(E, med, "comparison", label, band=(lo, hi),
                      style=style_key), float(defined.mean())

    s_rbm, cov_rbm = err("err_rbm", "RBM", "rbm")
    s_lrom, cov_lrom = err("err_lrom", "LROM", "lrom")

    def median_of(name):
        v = np.asarray(d[name], dtype=float)
        return float(np.nanmedian(v)) if np.isfinite(v).any() else float("nan")

    error = PanelData(
        key=system_key, title="",
        x_label=r"$E_{\rm c.m.}$ (MeV)", y_label=r"rel. error on $\sigma$",
        thresholds=system.thresholds, threshold_labels=system.threshold_labels,
        series=[s_rbm, s_lrom],
        notes=dict(median_rbm=median_of("err_rbm"),
                   median_lrom=median_of("err_lrom")),
        provenance=prov,
    )
    return dict(sigma=sigma, error=error)


def invariants(system_key):
    """Properties the solution must have whatever code produced it.

    A code-to-code comparison can have both solvers agree and both be wrong;
    these are checks a wrong answer cannot pass. Both curves come from the same
    code, switched by flux_factor_disabled. The claim is read off the shape: a
    discretisation error falls with N, one that flattens is in the formulation.
    Four systems, not one, make it a result.
    """
    system = SYSTEMS[system_key]
    d, meta = cache.read(system_key, "invariants")
    prov = _provenance(system, meta, "E_sym", "E_open")
    N = np.asarray(d["N_scan"], dtype=float)

    def panel(on, off, title, ylab):
        return PanelData(
            key=system_key, title=title, x_label="$N$ mesh points",
            y_label=ylab,
            series=[Series(N, np.asarray(d[on], dtype=float), "comparison",
                           system.label, style="thick"),
                    Series(N, np.asarray(d[off], dtype=float), "comparison",
                           None, style="predicted")],
            notes=dict(floor=float(np.asarray(d[off], dtype=float)[-1]),
                       best=float(np.asarray(d[on], dtype=float)[-1])),
            provenance=prov,
        )

    # The ratio panel needs a coupled pair; on an uncoupled system the block
    # stores NaN and the panel draws nothing rather than zeros.
    ratio_S = np.asarray(d["ratio_S"], dtype=float)
    ratio_k = np.asarray(d["ratio_k"], dtype=float)
    E = np.asarray(d["E_ratio"], dtype=float)
    has_ratio = bool(np.any(np.isfinite(ratio_S)) and np.any(np.isfinite(ratio_k)))
    if has_ratio:
        with np.errstate(divide="ignore", invalid="ignore"):
            agree = float(np.nanmax(np.abs(ratio_S - ratio_k) / np.abs(ratio_k)))
    else:
        agree = float("nan")
        E = ratio_S = ratio_k = np.array([])

    return dict(
        symmetry=panel("sym_on", "sym_off", "time-reversal symmetry",
                       r"$\max|U_{ab}-U_{ba}|$"),
        # Im U == 0 by taking the real part of the interaction, whatever carries
        # it -- not by zeroing one parameter, which leaves a surface absorption
        # W_d0 in place on some systems.
        unitarity=panel("uni_on", "uni_off",
                        r"flux conservation, $\mathrm{Im}\,U \equiv 0$",
                        r"$\|U^\dagger U-\mathbb{1}\|_F$"),
        # measured OVER predicted, not two curves side by side: they agree to
        # 5e-04, so a ratio pinned to 1 shows the claim (the missing factor IS
        # k_a/k_b) where two overlaid curves would not.
        ratio=PanelData(
            key=system_key, title="the missing factor, named",
            x_label=r"$E_{\rm c.m.}$ (MeV)",
            y_label=r"$\left(|U_{ba}|/|U_{ab}|\right) \, / \, (k_a/k_b)$",
            series=[Series(E, np.divide(ratio_S, ratio_k,
                                        out=np.full_like(ratio_S, np.nan),
                                        where=np.abs(ratio_k) > 0),
                           "comparison", system.label, style="thick")],
            notes=dict(agree=agree, coupled=has_ratio),
            provenance=prov,
        ),
    )


def error_by_quantity(system_key):
    """Which quantity the emulator error is measured on -- and how large it is.

    Four error curves against retained rank (global norm, worst diagonal, worst
    off-diagonal, reaction cross section), plus a fifth panel that makes them
    readable: a relative error needs a denominator. On alpha+12C the
    off-diagonal elements are 3e-03 of the diagonal and reach 2e-08, so a 410%
    relative error there is an absolute 1e-07 -- drawn beside the magnitudes,
    the curve says the accuracy is being asked on a near-zero quantity.
    """
    system = SYSTEMS[system_key]
    d, meta = cache.read(system_key, "manifold")

    prov = _provenance(system, meta, "n_train", "n_valid", "E")

    nb = np.asarray(d["nb_scan"], dtype=float).ravel()

    def curve(name, role, label, style_key):
        """Median across validation draws, with the p5-p95 envelope.

        Each point is a median over n_valid draws whose errors span decades, so
        the envelope is what says the emulator has a distribution, not one error.
        """
        v = np.asarray(d[name], dtype=float)
        if v.ndim == 1:
            v = v.reshape(len(nb), -1)
        # An all-NaN row is the expected inelastic error of an uncoupled system
        # (structurally-zero off-diagonal S); numpy's warning about it is noise.
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(v, axis=1)
            lo = np.nanpercentile(v, 5, axis=1)
            hi = np.nanpercentile(v, 95, axis=1)
        return Series(nb, med, role, label, band=(lo, hi), style=style_key)

    errors = PanelData(
        key=system_key, title=system.label,
        x_label=r"retained rank $n_b$", y_label="relative error",
        series=[curve("err_global", "comparison",
                      r"$\|\Delta S\|_F/\|S\|_F$", "norm"),
                curve("err_elastic", "comparison", r"elastic $U_{aa}$", "elastic"),
                curve("err_inelastic", "comparison", r"inelastic $U_{ab}$",
                      "inelastic"),
                curve("err_sigma", "reference", r"$\sigma_{\rm reaction}$",
                      "sigma")],
        notes=dict(n_valid=meta.get("n_valid"), n_train=meta.get("n_train")),
        provenance=prov,
    )

    A = np.asarray(d["abs_S_ref"], dtype=float)
    mask = np.asarray(d["coupling_mask"], dtype=bool)
    Nc = A.shape[1]
    eye = np.eye(Nc, dtype=bool)
    diag = A[:, eye].ravel()
    off = A[:, mask & ~eye].ravel() if mask.any() else np.array([np.nan])

    def spread(v):
        v = v[np.isfinite(v) & (v > 0)]
        if v.size == 0:
            return (np.nan, np.nan, np.nan)
        return (np.percentile(v, 5), np.median(v), np.percentile(v, 95))

    lo_d, md_d, hi_d = spread(diag)
    lo_o, md_o, hi_o = spread(off)

    magnitude = PanelData(
        key=system_key, title="",
        x_label=r"$|S|$", y_label="",
        series=[Series(np.array([lo_d, hi_d]), np.array([1.0, 1.0]), "span",
                       r"diagonal $U_{aa}$", band=(md_d,), style="elastic"),
                Series(np.array([lo_o, hi_o]), np.array([0.0, 0.0]), "span",
                       r"off-diagonal $S_{ab}$", band=(md_o,), style="inelastic")],
        notes=dict(ratio=float(md_o / md_d) if md_d else float("nan"),
                   smallest=_nanmin(off)),
        provenance=prov,
    )
    return dict(errors=errors, magnitude=magnitude)


def potential(system_key):
    """The interaction at theta_c, term by term.

    The affine decomposition: each curve is theta_j * dU/dtheta_j for a
    parameter U is linear in, so the terms sum exactly to U and the figure shows
    the structure the emulator exploits. Geometry parameters enter nonlinearly
    and carry no term. The Coulomb potential is drawn too -- it lives in the
    channel definition, not the optical model, so a V_diag-only figure omits it.
    """
    system = SYSTEMS[system_key]
    d, meta = cache.read(system_key, "sweep")

    r = np.asarray(d["r"], dtype=float)
    V = np.asarray(d["V_channels"])
    Vc = np.asarray(d["V_couplings"])

    series = [Series(r, V[0].real, "reference", r"$\mathrm{Re}\,U$ (total)",
                     style="total_re"),
              Series(r, V[0].imag, "reference", r"$\mathrm{Im}\,U$ (total)",
                     style="total_im")]

    #: One pen per term. Thin and dashed: they are the pieces, and the two
    #: totals above are what the reader should see first.
    pens = ("term_a", "term_b", "term_c", "term_d", "term_e",
            "term_f", "term_g", "term_h")

    def add(terms, labels, prefix, scale=1.0, offset=0):
        # `offset` continues the pen cycle: coupling terms carry the same
        # parameters as diagonal ones, so restarting would reuse a pen.
        for i, (curve, name) in enumerate(zip(terms, labels), start=offset):
            curve = np.asarray(curve) * scale
            # each term is purely real or purely imaginary (-V0 f(r) or
            # -i W0 f(r)); draw the part that carries something.
            part, tag = ((curve.real, r"\mathrm{Re}")
                         if np.max(np.abs(curve.real)) >= np.max(np.abs(curve.imag))
                         else (curve.imag, r"\mathrm{Im}"))
            series.append(Series(r, part, "comparison",
                                 rf"{prefix}${tag}$ {name}",
                                 style=pens[i % len(pens)]))

    add(d.get("terms_diag", []), meta.get("terms_diag_labels", []), "")

    coulomb = np.asarray(d.get("V_coulomb", np.zeros_like(r)), dtype=float)
    if np.max(np.abs(coulomb)) > 0:
        series.append(Series(r, coulomb, "comparison", r"$V_C$", style="coulomb"))

    # the coupling on one scale factor so it is visible beside a diagonal ~100x
    # larger; the factor is stated in the panel, not a shared legend.
    scale, has_coupling = 1.0, False
    off = None
    if Vc.shape[0] > 1 and np.isfinite(Vc[0, 1]).any() and np.nanmax(np.abs(Vc[0, 1])) > 0:
        off = np.abs(np.nan_to_num(Vc[0, 1]))
        peak = np.max(np.abs(V[0])) or 1.0
        scale = float(10 ** np.round(np.log10(peak / max(off.max(), 1e-300))))
        has_coupling = True
        terms_c = d.get("terms_coup", [])
        labels_c = meta.get("terms_coup_labels", [])
        if len(terms_c):
            add(terms_c, labels_c, r"$V_{ab}$: ", scale=scale,
                offset=len(d.get("terms_diag", [])) + 1)
        else:
            series.append(Series(r, (np.nan_to_num(Vc[0, 1]) * scale).real,
                                 "comparison", r"$V_{ab}$", style="term_e"))

    return dict(potential=PanelData(
        key=system_key, title=system.label, x_label=r"$r$ (fm)",
        y_label=r"$U(r)$ (MeV)", series=series,
        notes=dict(coupling_scale=scale if has_coupling else float("nan"),
                   has_coupling=has_coupling),
        provenance=_provenance(system, meta),
    ))


def wavefunction_vs_emulators(system_key):
    """psi(r) at one energy, solver against the emulators, and the residual.

    sigma is a scalar reduction of S, and S is a boundary value of chi(r), so an
    emulator can land sigma while getting chi(r) wrong -- pointwise errors that
    cancel in the flux integral do not cancel in the wave function. Drawn at a
    validation parameter, never theta_c, where both emulators are near-exact.
    """
    system = SYSTEMS[system_key]
    d, meta = cache.read(system_key, "interior")

    r = np.asarray(d["r"], dtype=float)
    ref = np.asarray(d["psi_ref"])
    prov = _provenance(system, meta, "n_train", "nb", "E")

    main = [Series(r, ref.real, "reference", "DBMM", style="dbmm")]
    resid, notes = [], {}
    for name, label, style_key in (("rbm", "RBM", "rbm"), ("lrom", "LROM", "lrom")):
        key = f"psi_{name}"
        if key not in d:
            continue
        emu = np.asarray(d[key])
        main.append(Series(r, emu.real, "comparison", label, style=style_key))
        scale = np.max(np.abs(ref)) or 1.0
        # median and p5-p95 over the validation draws: the band says whether the
        # emulator is uniformly good or only good on average.
        spread = d.get(f"resid_psi_{name}")
        if spread is not None and np.asarray(spread).ndim == 2:
            med, lo, hi = _band(np.asarray(spread, dtype=float))
            resid.append(Series(r, med, "comparison", label, band=(lo, hi),
                                style=style_key))
            notes[f"worst_{name}"] = float(np.nanmax(hi))
            # median over r of the median over draws: computed here, not in the
            # table, so the figure and the table quote the same object.
            notes[f"median_{name}"] = float(np.nanmedian(med))
        else:
            pointwise = np.abs(emu - ref) / scale
            resid.append(Series(r, pointwise, "comparison", label,
                                style=style_key))
            notes[f"worst_{name}"] = float(np.max(pointwise))
            notes[f"median_{name}"] = float(np.nanmedian(pointwise))

    # the deviation vanishes at r=0, so without a floor the axis runs decades
    # below the plateau that carries the result.
    body = np.concatenate([np.asarray(sr.y, dtype=float) for sr in resid])         if resid else np.array([np.nan])
    body = body[np.isfinite(body) & (body > 0)]
    if body.size:
        notes["y_floor"] = float(np.percentile(body, 1)) * 0.3

    return dict(
        psi=PanelData(
            key=system_key, title=system.label, x_label=r"$r$ (fm)",
            y_label=r"$\mathrm{Re}\,\psi(r)$", series=main,
            notes=dict(E=meta.get("E"), **notes), provenance=prov),
        residual=PanelData(
            key=system_key, title="", x_label=r"$r$ (fm)",
            y_label=r"$|\Delta\psi| / \max|\psi|$", series=resid,
            notes=notes, provenance=prov),
    )


def phase_shift_vs_emulators(system_key):
    """delta(E) over the grid, solver against the emulators, and the residual.

    Each point needs its own emulator: RBM and LROM are fitted at one energy.
    The eigenphase branch is tracked the same way on every grid -- it is a
    property of the grid, not of a point -- so the curves are comparable.
    """
    system = SYSTEMS[system_key]
    d, meta = cache.read(system_key, "interior")

    E = np.asarray(d["E"], dtype=float)
    ref = np.asarray(d["delta_ref"], dtype=float)
    prov = _provenance(system, meta, "n_train", "nb", "n_E")

    main = [Series(E, ref, "reference", "DBMM", style="dbmm")]
    resid, notes = [], {}
    for name, label, style_key in (("rbm", "RBM", "rbm"), ("lrom", "LROM", "lrom")):
        key = f"delta_{name}"
        if key not in d:
            continue
        emu = np.asarray(d[key], dtype=float)
        main.append(Series(E, emu, "comparison", label, style=style_key))
        # absolute, in degrees: a relative error on a phase shift through zero
        # is unbounded.
        spread = d.get(f"resid_delta_{name}")
        if spread is not None and np.asarray(spread).ndim == 2:
            med, lo, hi = _band(np.asarray(spread, dtype=float))
            resid.append(Series(E, med, "comparison", label, band=(lo, hi),
                                style=style_key))
            notes[f"median_{name}"] = float(np.nanmedian(med))
        else:
            resid.append(Series(E, np.abs(emu - ref), "comparison", label,
                                style=style_key))
            notes[f"median_{name}"] = float(np.nanmedian(np.abs(emu - ref)))

    return dict(
        delta=PanelData(
            key=system_key, title=system.label,
            x_label=r"$E_{\rm c.m.}$ (MeV)", y_label=r"$\delta$ (deg)",
            thresholds=system.thresholds,
            threshold_labels=system.threshold_labels,
            series=main, notes=notes, provenance=prov),
        residual=PanelData(
            key=system_key, title="",
            x_label=r"$E_{\rm c.m.}$ (MeV)",
            y_label=r"$|\Delta\delta|$ (deg)",
            thresholds=system.thresholds,
            threshold_labels=system.threshold_labels,
            series=resid, notes=notes, provenance=prov),
    )


def error_distribution(system_key):
    """How the error is distributed, method against method.

    A median answers "how accurate on average", not "how often is it bad",
    which is what a campaign over the box asks: a median of 1e-04 with a tail at
    1e-01 and one with a tail at 2e-04 are different emulators. The sample is
    every (validation draw, energy) pair the excitation block holds. The
    solver's own disagreement with the R-matrix is drawn beside them as the
    scale to read against.
    """
    system = SYSTEMS[system_key]
    d, meta = cache.read(system_key, "excitation")

    samples = []
    for name, label, style_key in (("err_rbm", "RBM", "rbm"),
                                   ("err_lrom", "LROM", "lrom")):
        if name not in d:
            continue
        v = np.asarray(d[name], dtype=float).ravel()
        v = v[np.isfinite(v) & (v > 0)]
        if v.size:
            samples.append((label, style_key, v))

    ref = np.asarray(d["sigma_ref"], dtype=float)[0]
    rmat = np.asarray(d["sigma_rmat"], dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        floor = np.abs(rmat - ref) / np.abs(ref)
    floor = floor[np.isfinite(floor) & (floor > 0)]
    # a line, not a fourth histogram: 12 R-matrix energies against 1200 emulator
    # samples would bin into a spiky curve.
    floor_median = float(np.median(floor)) if floor.size else float("nan")

    allv = np.concatenate([v for _, _, v in samples] + ([floor] if floor.size else [])
                          ) if samples else np.array([1.0])
    bins = np.linspace(np.log10(allv.min()), np.log10(allv.max()) + 1e-12, 22)

    series, notes = [], {}
    for label, style_key, v in samples:
        counts, edges = np.histogram(np.log10(v), bins=bins)
        centres = 0.5 * (edges[:-1] + edges[1:])
        series.append(Series(10.0 ** centres, counts / v.size, "comparison",
                             label, style=style_key))
        tag = label.split()[0].lower()
        notes[f"median_{tag}"] = float(np.median(v))
        notes[f"p95_{tag}"] = float(np.percentile(v, 95))
    notes["n_sample"] = int(samples[0][2].size) if samples else 0
    notes["floor"] = floor_median

    # The paired win fraction, which a ratio of medians hides: every
    # (draw, energy) pair is a head-to-head trial, and "ahead on 97% of trials"
    # is a stronger claim than "ahead on the median". The per-energy breakdown
    # is reported too, since the trials are correlated within a draw and a
    # binomial on them would overclaim significance.
    if "err_rbm" in d and "err_lrom" in d:
        a = np.asarray(d["err_rbm"], dtype=float)
        b = np.asarray(d["err_lrom"], dtype=float)
        both = np.isfinite(a) & np.isfinite(b)
        if both.any():
            notes["win_rbm"] = float((a[both] < b[both]).mean())
            notes["n_trial"] = int(both.sum())
            # per column, only where the pair is defined: an energy where one
            # method predicts nothing is not an energy it lost.
            live = both.any(axis=0)
            med_a = np.full(a.shape[1], np.nan)
            med_b = np.full(b.shape[1], np.nan)
            med_a[live] = np.nanmedian(np.where(both, a, np.nan)[:, live], axis=0)
            med_b[live] = np.nanmedian(np.where(both, b, np.nan)[:, live], axis=0)
            notes["n_energy"] = int(live.sum())
            notes["n_energy_rbm"] = int((med_a[live] < med_b[live]).sum())

    return dict(distribution=PanelData(
        key=system_key, title=system.label,
        x_label=r"relative error on $\sigma$",
        y_label="fraction of the sample",
        series=series, notes=notes,
        provenance=_provenance(system, meta, "n_train", "nb", "n_E", "n_resid"),
    ))


def manifold_spread(system_key):
    """How far the observables travel when the parameter sweeps its box.

    A rainbow collapsing to one line would mean the box is too narrow and every
    later claim is about nothing. The headline is the factor the cross section
    spans over the full box.
    """
    system = SYSTEMS[system_key]
    d, meta = cache.read(system_key, "sweep")

    E = np.asarray(d["E"], dtype=float)
    delta = np.asarray(d["delta"], dtype=float)
    sigma = np.asarray(d["sigma"], dtype=float)
    box = np.asarray(d["sigma_box"], dtype=float)
    vals = np.asarray(d["sweep_vals"], dtype=float)

    # p95 over p5, not max over min: on p+12C the cross section passes through an
    # interference zero, where the extremes give a ratio of 1e9 that measures
    # how close a curve came to zero, not how far the observable travels.
    with np.errstate(invalid="ignore", divide="ignore"):
        span = (np.nanpercentile(box, 95, axis=0)
                / np.nanpercentile(box, 5, axis=0))

    # The envelope over the FULL box: p5-p95 over 200 draws with all p varying at
    # once, which is what the emulator must cover -- distinct from the rainbow,
    # which walks one parameter with the others fixed.
    with np.errstate(invalid="ignore"):
        box_med = np.nanmedian(box, axis=0)
        box_lo = np.nanpercentile(box, 5, axis=0)
        box_hi = np.nanpercentile(box, 95, axis=0)

    def rainbow(y, ylab, title, envelope=None):
        return PanelData(
            key=system_key, title=title, x_label=r"$E_{\rm c.m.}$ (MeV)",
            y_label=ylab,
            thresholds=system.thresholds,
            threshold_labels=system.threshold_labels,
            series=([] if envelope is None else [
                Series(E, envelope[0], "reference",
                       r"p5-p95, all $p$ varying", band=(envelope[1], envelope[2]),
                       style="box")])
            + [Series(E, y[i], "sweep", None, style="sweep",
                      band=None) for i in range(y.shape[0])],
            # median over the energy grid, not the maximum (which lands on an
            # interference dip).
            notes=dict(span=float(np.nanmedian(span)), lo=float(vals[0]),
                       hi=float(vals[-1]), label=meta.get("sweep_label", "")),
            # the shared colorbar carries direction only, so the per-column
            # sweep bounds go here.
            provenance={**_provenance(system, meta, "n_sweep", "n_box"),
                        meta.get("sweep_label", "swept"):
                            f"[{vals[0]:.3g},{vals[-1]:.3g}]"},
        )

    # column titles name the SYSTEM (the quantity is on the y axis, the box in
    # the suptitle).
    psi = np.asarray(d["psi_sweep"]) if "psi_sweep" in d else None
    r_psi = np.asarray(d["r_psi"], dtype=float) if "r_psi" in d else None

    def rainbow_r(y, ylab):
        """The same ribbon, against r rather than E."""
        return PanelData(
            key=system_key, title="", x_label=r"$r$ (fm)", y_label=ylab,
            series=[Series(r_psi, y[i], "sweep", None, style="sweep")
                    for i in range(y.shape[0])],
            notes=dict(label=meta.get("sweep_label", "")),
            provenance=_provenance(system, meta, "n_sweep", "n_box"),
        )

    return dict(
        eigenphase=rainbow(delta, r"$\delta$ (deg)", system.label),
        wavefunction=(rainbow_r(psi.real, r"$\mathrm{Re}\,\psi(r)$")
                      if psi is not None else None),
        # Only sigma has the box envelope: block_sweep draws the 200-point
        # Latin hypercube for the cross section alone, so claiming one for the
        # eigenphase would be drawing a band nobody computed.
        xsec=rainbow(sigma, r"$\sigma$ (fm$^2$)", "",
                     envelope=(box_med, box_lo, box_hi)),
    )


def cat_cloud(system_key, quantity="err_S00"):
    """Cost against accuracy as a CLOUD: one point per validation draw.

    A median error and cost per configuration would put an emulator whose error
    spans two decades and one that is uniform on the same spot of a Pareto
    front; every draw keeps its own time and error instead.

    The solver's cost is a p10-p90 BAND, but it is measurement scatter, not a
    distribution over the box: the solver does fixed-size work per theta, and
    timing one theta many times gives a band as wide as timing many. Same for
    the horizontal spread of each cloud. The VERTICAL spread is the figure -- the
    error genuinely varies over the box by up to two decades.
    """
    system = SYSTEMS[system_key]
    d, meta = cache.read(system_key, "cat")

    method = np.asarray(d["method"]).astype(str)
    t_on = np.asarray(d["t_online"], dtype=float) * 1e3        # ms
    err = np.asarray(d[quantity], dtype=float) * 100.0         # per cent
    nb = np.asarray(d["nb_kept"], dtype=float)
    kd = np.asarray(d["K_diag"], dtype=float)
    kc = np.asarray(d["K_coup"], dtype=float)
    ratio = np.asarray(d["overfit_ratio"], dtype=float)
    t_dbmm = np.asarray(d["t_dbmm"], dtype=float) * 1e3

    combos, seen = [], set()
    for i in range(method.size):
        key = (method[i], nb[i], kd[i], kc[i])
        if key not in seen:
            seen.add(key)
            combos.append((key, i))

    series, notes = [], {}
    for n, ((m, b, a, c), first) in enumerate(combos):
        sel = (method == m) & (nb == b) & (kd == a) & (kc == c)
        good = sel & np.isfinite(err) & (err > 0)
        if not good.any():
            continue
        label = (rf"RBM $n_b={b:.0f}$" if m == "rbm"
                 else rf"LROM $n_b={b:.0f}$, $K={a:.0f}"
                      + (rf"+{c:.0f}$" if c else r"$"))
        series.append(Series(t_on[good], err[good], "cloud", label,
                             style=f"combo_{n % 6}",
                             # underdetermined fit: reported and marked, not
                             # hidden, so it is not read as an operating point.
                             band=(float(ratio[first]),)))
        notes[f"median_{n}"] = float(np.median(err[good]))

    # Three solver-side references: the band is the affine-split path (the
    # honest baseline, what gains are quoted against); the two lines are the
    # full rebuild and the R-matrix code.
    notes["t_dbmm_lo"] = float(np.percentile(t_dbmm, 10))
    notes["t_dbmm_hi"] = float(np.percentile(t_dbmm, 90))
    notes["t_dbmm"] = float(np.median(t_dbmm))
    for key, note in (("t_dbmm_full", "t_rebuild"), ("t_rmat", "t_rmat")):
        if key in d:
            notes[note] = float(np.median(np.asarray(d[key], dtype=float))) * 1e3
    notes["n_valid"] = int(meta.get("n_valid", 0))
    # Anchor of the compressed x scale: the fastest prediction on the sheet, so
    # the cloud starts at the left edge and every decade beyond it is squeezed.
    finite = t_on[np.isfinite(t_on) & (t_on > 0)]
    if finite.size:
        notes["x_floor"] = float(np.min(finite)) * 0.9   # t_on is already ms

    label = (r"$|S_{00}|$" if quantity == "err_S00" else r"$|S_{ab}|$")
    return dict(cloud=PanelData(
        key=system_key, title=system.label,
        x_label="time per evaluation (ms)",
        y_label=rf"rel. error on {label} (%)",
        series=series, notes=notes,
        provenance=_provenance(system, meta, "n_train", "n_valid", "E"),
    ))


def cat_cloud_inelastic(system_key):
    """The same cloud, measured on the off-diagonal element."""
    return cat_cloud(system_key, quantity="err_Sab")


def online_vs_mesh(system_key):
    """Online cost of both emulators against the mesh size, physics fixed.

    The LROM is the slower of the two on the four suite systems. The term it
    removes -- evaluating and projecting the potential at all N-1 interior
    points -- grows with N, so its advantage is asymptotic and this suite (N
    40-120) is below it. Measured with MIN_REPEATS the LROM/RBM ratio closes
    with the mesh on all four (down to ~1.05 on alpha+12C) but stays at or above
    unity: the LROM is never the faster here.

    It closes slowly rather than crossing because both emulators pay a term
    linear in N -- `predict` lifts the reduced coefficients to the full mesh,
    X_r @ a at O(N Nc^2 nb), before contracting them with the boundary values.
    The two controls sit below unity, where nb and K are small enough that the
    LROM's extra per-query work is cheaper than the RBM's mesh loop. The solver's
    own cost is drawn beside the two.
    """
    system = SYSTEMS[system_key]
    d, meta = cache.read(system_key, "scaling")

    N = np.asarray(d["N"], dtype=float)
    t_rbm = np.asarray(d["t_rbm"], dtype=float) * 1e3
    t_lrom = np.asarray(d["t_lrom"], dtype=float) * 1e3
    t_V = np.asarray(d["t_dbmm_V"], dtype=float) * 1e3

    prov = _provenance(system, meta, "nb", "K_diag", "K_coup", "Ns", "E")
    # `provenance`, not `notes`: the renderer builds the banner from the former.
    # This figure's claim is "at fixed physics", so the rank and budget must show.
    cost = PanelData(
        key=system_key, title=system.label, x_label=r"mesh points $N$",
        y_label="online time (ms)", provenance=prov, series=[
            Series(N, t_V, "reference", r"DBMM, $V(\theta)$ only", style="dbmm"),
            Series(N, t_rbm, "comparison", "RBM", style="rbm"),
            Series(N, t_lrom, "comparison", "LROM", style="lrom"),
        ])
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = t_lrom / t_rbm
    band = PanelData(
        key=system_key, title="", x_label=r"mesh points $N$",
        y_label="LROM / RBM", notes=dict(
            ratio_lo=float(np.nanmin(ratio)), ratio_hi=float(np.nanmax(ratio)),
            unity=1.0),
        series=[Series(N, ratio, "residual", None, style="lrom")],
        provenance=prov)
    return {"online": cost, "ratio": band}


def amortisation(system_key, tol=5e-3, horizon=1e6):
    """From how many evaluations an emulator pays for itself.

    Every autocalibrated configuration is drawn, not just the best: a campaign
    of 1e4 and one of 1e6 do not want the same basis. The thick line is the
    envelope (total time of the best choice at each N); the thin lines are the
    configurations it is made of. The band comes from the timed predictions per
    configuration, since at large N the online stage dominates the total. A
    method with no configuration clearing `tol` is not drawn.
    """
    system = SYSTEMS[system_key]
    d, meta = cache.read(system_key, "cat")

    method = np.asarray(d["method"]).astype(str)
    nb = np.asarray(d["nb_kept"], dtype=float)
    kd = np.asarray(d["K_diag"], dtype=float)
    kc = np.asarray(d["K_coup"], dtype=float)
    t_on = np.asarray(d["t_online"], dtype=float)
    t_off = np.asarray(d["t_offline"], dtype=float)
    quantity = "err_Sab" if np.isfinite(np.asarray(d["err_Sab"])).any() else "err_S00"
    err = np.asarray(d[quantity], dtype=float)
    t_dbmm = float(np.median(np.asarray(d["t_dbmm"], dtype=float)))

    n = np.logspace(1, np.log10(horizon) + 0.4, 140)

    # Each baseline is a line through the origin, differing only in slope. Three
    # of them separate two gains: what the affine split buys the solver, and what
    # the reduction buys on top.
    refs = [("t_rmat", "R-matrix", "rmat"),
            ("t_dbmm_full", "DBMM, full rebuild", "dbmm_full"),
            (None, r"DBMM, $V(\theta)$ only", "dbmm")]
    series, notes = [], {}
    for key, label, style_key in refs:
        if key is None:
            t = t_dbmm
        elif key in d:
            t = float(np.median(np.asarray(d[key], dtype=float)))
        else:
            continue
        series.append(Series(n, t * n, "reference", label, style=style_key))
        notes[f"{style_key}_horizon"] = t * horizon
    notes.update({"t_dbmm": t_dbmm, "n_train": meta.get("n_train"),
                  "horizon": horizon, "t_dbmm_horizon": t_dbmm * horizon})
    absent = []

    for name, label, style_key in (("rbm", "RBM", "rbm"), ("lrom", "LROM", "lrom")):
        rows = np.where(method == name)[0]
        if not rows.size:
            continue
        curves, los, his = [], [], []
        for key in sorted({(nb[i], kd[i], kc[i]) for i in rows}):
            sel = rows[(nb[rows] == key[0]) & (kd[rows] == key[1])
                       & (kc[rows] == key[2])]
            if not np.isfinite(np.nanmedian(err[sel])) or np.nanmedian(err[sel]) > tol:
                continue
            off = t_off[sel][0]
            curves.append(off + np.median(t_on[sel]) * n)
            los.append(off + np.percentile(t_on[sel], 5) * n)
            his.append(off + np.percentile(t_on[sel], 95) * n)
            # one thin unlabelled line per configuration: the fan of choices.
            series.append(Series(n, curves[-1], "comparison", None,
                                 style=f"{style_key}_faint"))
        if not curves:
            absent.append(label)
            continue
        pick = np.argmin(np.array(curves), axis=0)
        best = np.array(curves)[pick, np.arange(n.size)]
        lo = np.array(los)[pick, np.arange(n.size)]
        hi = np.array(his)[pick, np.arange(n.size)]
        series.append(Series(n, best, "comparison", label, band=(lo, hi),
                             style=style_key))
        beats = best < t_dbmm * n
        notes[f"cross_{name}"] = float(n[np.argmax(beats)]) if beats.any() else np.nan
        notes[f"t_{name}_horizon"] = float(np.interp(horizon, n, best))
        notes[f"n_configs_{name}"] = int(len(curves))

    notes["absent"] = ", ".join(absent)
    return dict(breakeven=PanelData(
        key=system_key, title=system.label,
        x_label="forward evaluations $N$", y_label="total time (s)",
        series=series, notes=notes,
        provenance=_provenance(system, meta, "n_train", "n_valid", "E"),
    ))


def coupling_budget_failure(system_key):
    """The same predictor budget, divided two ways.

    Two LROM emulators, identical total K, identical training set, identical
    rank. The only difference is whether any predictor point is spent on the
    coupling blocks. The figure is read top down: what an elastic-only
    validation sees, then what it does not.
    """
    system = SYSTEMS[system_key]
    d, meta = cache.read(system_key, "failure")
    prov = _provenance(system, meta, "n_train", "n_sweep", "K_total", "nb", "E")
    x = np.asarray(d["sweep"], dtype=float)
    K = meta.get("K_total")

    # both budgets are LROM: the split one keeps an LROM colour, since the claim
    # is "the same method, the same budget, divided differently".
    def panel(ref, joint, split, ylab, title):
        return PanelData(
            key=system_key, title=title, x_label=meta.get("sweep_label", ""),
            y_label=ylab,
            series=[Series(x, np.asarray(d[ref], dtype=float), "reference",
                           "DBMM", style="dbmm"),
                    Series(x, np.asarray(d[joint], dtype=float), "comparison",
                           rf"joint  $K_{{\rm diag}}={K}$, $K_{{\rm coup}}=0$",
                           style="lrom"),
                    Series(x, np.asarray(d[split], dtype=float), "comparison",
                           rf"split  $K_{{\rm diag}}=K_{{\rm coup}}={K // 2}$",
                           style="lrom_split")],
            notes=dict(K_total=K, amplitude_ratio=meta.get("amplitude_ratio")),
            provenance=prov,
        )

    elastic = panel("delta_ref", "delta_joint", "delta_split",
                    r"elastic $\delta$ (deg)", "what an elastic check sees")
    inelastic = panel("s01_ref", "s01_joint", "s01_split",
                      r"$|S_{01}|^2$",
                      "the joint budget is constant in the parameter; "
                      "the split budget lies on the solver")

    with np.errstate(invalid="ignore"):
        elastic.notes["joint_err"] = float(
            np.nanmedian(np.asarray(d["err_delta_joint"], dtype=float)))
        inelastic.notes["joint_err"] = float(
            np.nanmedian(np.asarray(d["err_inel_joint"], dtype=float)))

    # The error panel and the predictor-position panel carry the argument: two
    # curves without a number, or a claim without a mechanism, otherwise.
    errors = PanelData(
        key=system_key, title="", x_label=meta.get("sweep_label", ""),
        y_label="relative error",
        series=[Series(x, np.asarray(d["err_delta_joint"], dtype=float),
                       "comparison", r"joint, on $\delta$", style="lrom_dash"),
                Series(x, np.asarray(d["err_inel_joint"], dtype=float),
                       "comparison", r"joint, on $S_{ab}$", style="lrom"),
                Series(x, np.asarray(d["err_delta_split"], dtype=float),
                       "comparison", r"split, on $\delta$", style="lrom_split_dash"),
                Series(x, np.asarray(d["err_inel_split"], dtype=float),
                       "comparison", r"split, on $S_{ab}$", style="lrom_split")],
        notes=dict(joint_delta=float(np.nanmedian(d["err_delta_joint"])),
                   joint_inel=float(np.nanmedian(d["err_inel_joint"])),
                   split_delta=float(np.nanmedian(d["err_delta_split"])),
                   split_inel=float(np.nanmedian(d["err_inel_split"])),
                   # the figure in one number: how far the elastic check
                   # under-reports the joint budget's real error.
                   understated=float(np.nanmedian(d["err_inel_joint"])
                                     / np.nanmedian(d["err_delta_joint"]))),
        provenance=prov,
    )

    r_dense = np.asarray(d["r_dense"], dtype=float)
    V_aa = np.asarray(d["V_aa"], dtype=float)
    V_ab = np.asarray(d["V_ab"], dtype=float)
    ratio = float(meta.get("amplitude_ratio", np.nan))
    predictors = PanelData(
        key=system_key, title="", x_label=r"$r$ (fm)", y_label="predictors",
        series=[Series(r_dense, V_aa / max(V_aa.max(), 1e-300), "shape",
                       r"$|V_{aa}|$", style="norm"),
                Series(r_dense, V_ab / max(V_ab.max(), 1e-300), "shape",
                       rf"$|V_{{ab}}|$ (${ratio:.0f}\times$ smaller)",
                       style="inelastic"),
                Series(np.asarray(d["r_diag_joint"], dtype=float),
                       np.full(np.asarray(d["r_diag_joint"]).size, 1.0),
                       "ticks", "joint", style="lrom"),
                Series(np.asarray(d["r_diag_split"], dtype=float),
                       np.full(np.asarray(d["r_diag_split"]).size, 0.0),
                       "ticks", "split, diagonal", style="lrom_split"),
                Series(np.asarray(d["r_coup_split"], dtype=float),
                       np.full(np.asarray(d["r_coup_split"]).size, -0.4),
                       "ticks", "split, coupling", style="inelastic")],
        provenance=prov,
        notes=dict(amplitude_ratio=ratio,
                   n_joint_diag=int(np.asarray(d["r_diag_joint"]).size),
                   n_joint_coup=int(np.asarray(d["r_coup_joint"]).size),
                   n_split_diag=int(np.asarray(d["r_diag_split"]).size),
                   n_split_coup=int(np.asarray(d["r_coup_split"]).size)),
    )
    return dict(elastic=elastic, inelastic=inelastic, errors=errors,
                predictors=predictors)
