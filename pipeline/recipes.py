"""How a figure looks, as data: every layout choice is a literal on a Recipe,
settled late and per figure. The recipe never touches the data -- the view
decided what the numbers mean, the recipe says how many panels and what goes
in them.
"""
import dataclasses
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

from .settings import CONTROL, FAILURE_SYSTEM


@dataclass(frozen=True)
class Recipe:
    key: str
    stem: str                                  # output filename stem
    rows: Tuple[str, ...]                      # which view panels, top to bottom
    height_ratios: Tuple[float, ...] = ()
    per_column: str = "system"                 # one column per system
    figsize: Tuple[float, float] = (9.0, 5.0)
    y_scale: Tuple[str, ...] = ()              # per row
    x_scale: Tuple[str, ...] = ()              # per row
    share_y: bool = False
    # False: one column per system, rows are the view's panels.
    # True: one panel per quantity, every system overlaid; colour = system, dash = quantity.
    overlay_systems: bool = False
    mark_thresholds: bool = True
    tolerance_ladder: Sequence[str] = ()       # rows that carry it
    # row -> (note key, format string): a figure's own number, printed in its
    # panel rather than the title so the quantity and its statistic cannot drift.
    headlines: dict = field(default_factory=dict)
    # Annotations a view computed that the recipe must ask for, named by meaning:
    #   hlines: row -> (note key, label)   one line, shared by all systems
    #   vlines: row -> note key            one per system, in its colour
    hlines: dict = field(default_factory=dict)
    vlines: dict = field(default_factory=dict)
    # weight_legend: style key -> what a line weight means (colour carries the
    #   system, weight the quantity, so a system-only legend explains half a sheet)
    # caption: what a band or marker is, inside the figure
    weight_legend: dict = field(default_factory=dict)
    caption: Optional[str] = None
    # row -> (note key, format): printed at the end of each system's curve, in
    # its colour -- a value that identifies a curve belongs on the curve.
    end_labels: dict = field(default_factory=dict)
    # row -> a short note saying which way is better (neither axis of a
    # cost-accuracy plane is self-evidently oriented).
    orientation: dict = field(default_factory=dict)
    # One sheet per system instead of one suite sheet: two recipes over one view.
    per_system: bool = False
    # Which methods survive into the sheet, by style key -- a subset is an
    # output, not a read-time filter.
    methods: Tuple[str, ...] = ()
    suptitle: Optional[str] = None
    legend_in: int = 0                         # which column carries the legend
    # True when every column names different curves (the potential figure:
    # n+40Ca decomposes into V_0/W_0/W_d0/V_so, alpha+24Mg into V_0/W_0/W_d0/beta_2).
    legend_all_columns: bool = False
    # row -> legend location, when "best" gets it wrong (matplotlib reads a panel
    # that is mostly filled area, like F7's form factors, as empty).
    legend_loc: dict = field(default_factory=dict)
    notes: dict = field(default_factory=dict)
    # Which systems this figure covers when the caller does not say; empty = the
    # whole suite. F7 is a one-system demonstration, so drawing it suite-wide
    # would ask for a block not computed elsewhere.
    systems: Tuple[str, ...] = ()


#: F2. Cross section and the residual on it, one column per system. Headline is
#: the median residual: the worst is a relative error on whichever matrix
#: element sits nearest zero (9e-02 for a solver that agrees to 1e-06).
VALIDATION_SUITE = Recipe(
    key="vs_rmatrix.suite",
    stem="f2_validation",
    rows=("main", "residual"),
    height_ratios=(1.7, 1.3),
    figsize=(9.5, 4.3),
    y_scale=("log", "log"),
    mark_thresholds=True,
    suptitle="DBMM against the R-matrix reference",
    headlines={"residual": ("median", "median {:.0e}")},
    legend_in=0,
)

#: F5. The four error curves, and beneath them the magnitudes those errors are
#: relative to -- one strip per system showing where diagonal and off-diagonal
#: elements live, so a 410% relative error on something worth 2e-08 reads as
#: what it is.
QUANTITY_SUITE = Recipe(
    key="error_by_quantity.suite",
    stem="f5_quantity",
    rows=("errors", "magnitude"),
    height_ratios=(3.6, 1.0),
    # one sheet per system, not four columns across the text width: two stacked
    # messages on four columns leave the labels and annotations unreadable.
    per_system=True,
    figsize=(4.6, 4.0),
    y_scale=("log", ""),
    x_scale=("log", "log"),
    mark_thresholds=False,
    tolerance_ladder=("errors",),
    suptitle="RBM error by quantity, beside the magnitude it is relative to",
    headlines={"magnitude": ("ratio", r"$|S_{{ab}}|/|S_{{aa}}| = ${:.0e}")},
    legend_in=0,
)

#: F4, first half. The spectrum with its criterion. Each quantity has its own
#: panel -- no shared double y axis.
SPECTRUM_SUITE = Recipe(
    key="pod_spectrum.suite",
    stem="f4_spectrum",
    rows=("spectrum",),
    overlay_systems=True,
    figsize=(5.6, 3.4),
    y_scale=("log", "log"),
    mark_thresholds=False,
    # label carries no literal: the rule is drawn at the block's own eps_sol, so
    # the figure cannot announce a tolerance it was not computed with.
    hlines={"spectrum": ("eps_tol", r"$\epsilon_{\rm tol}$")},
    vlines={"spectrum": "selected"},
    weight_legend={"thin": r"$\sigma_k/\sigma_1$  (one mode)",
                   "thick": r"$r(k)$  (discarded energy)"},
    # curves fall left to right, so lower right is the empty corner.
    legend_loc={"spectrum": "lower right"},
    caption=(r"colour: system   |   the criterion thresholds $r(k)$, not "
             r"$\sigma_k/\sigma_1$"),
    suptitle="how fast a mode dies",
    legend_in=0,
)

#: F4, training-size control. At the rank each system's criterion selects, does
#: adding snapshots move the error? A control on the offline stage, not a result
#: about the emulators.
SPECTRUM_NS_SUITE = Recipe(
    key="pod_spectrum.training",
    stem="f4_spectrum_ns",
    rows=("error_vs_ns",),
    overlay_systems=True,
    figsize=(5.6, 3.4),
    y_scale=("log",),
    mark_thresholds=False,
    tolerance_ladder=("error_vs_ns",),
    end_labels={"error_vs_ns": ("nb_probe", r"$n_b={}$")},
    legend_loc={"error_vs_ns": "lower left"},
    caption=(r"colour: system   |   each curve is probed at the rank its own "
             r"criterion selects"),
    suptitle="more snapshots, same rank",
    legend_in=0,
)

#: F4, second half. Achieved against predicted, on its own sheet.
ERROR_BY_RANK_SUITE = Recipe(
    key="error_by_rank.suite",
    stem="f4_error_by_rank",
    rows=("error",),
    overlay_systems=True,
    weight_legend={"achieved": "measured, out of sample",
                   "predicted": r"Eckart-Young bound $\sqrt{r(n_b)}$"},
    figsize=(5.6, 4.0),
    y_scale=("log",),
    x_scale=("log",),
    mark_thresholds=False,
    tolerance_ladder=("error",),
    suptitle="RBM accuracy against basis size, measured and predicted",
    legend_in=0,
)

#: F3, four systems. The claim -- an error that does not fall with N is in the
#: formulation, not the discretisation -- is read off the dashed curves, and is
#: stronger when four independent systems flatten at four different floors.
INVARIANTS_SUITE = Recipe(
    key="invariants.suite",
    stem="f3_invariants",
    rows=("symmetry", "unitarity", "ratio"),
    overlay_systems=True,
    # the three COUPLED systems only: n+40Ca has no off-diagonal element, so the
    # velocity factor is identically 1 and disabling it changes nothing but adds
    # a roundoff-level rising curve that reads as a divergence.
    systems=("alpha12C_band", "alpha24Mg", "n238U"),
    figsize=(10.0, 3.4),
    y_scale=("log", "log", "linear"),
    mark_thresholds=False,
    weight_legend={"thick": "with the flux factor",
                   "predicted": "without it"},
    legend_loc={"symmetry": "lower right"},
    caption=("colour: system   |   both curves come from the same code, "
             "switched by flux_factor_disabled   |   an error that flattens "
             "with $N$ is in the formulation, not in the mesh   |   the third "
             "panel is the measured ratio divided by the kinematic one, so "
             "agreement is the line at 1"),
    suptitle="solver invariants: what a code-to-code comparison cannot see",
    legend_in=0,
)

#: F8. End of the chain, on the reaction cross section. One column per system:
#: reference as a line, R-matrix as sparse markers, the three emulators on top;
#: the row beneath turns the agreement into a number against the tolerance ladder.
EXCITATION_SUITE = Recipe(
    key="observable_vs_emulators.suite",
    stem="f8_excitation",
    rows=("sigma", "error"),
    height_ratios=(1.7, 1.3),
    figsize=(9.5, 4.3),
    y_scale=("log", "log"),
    mark_thresholds=True,
    tolerance_ladder=("error",),
    suptitle="cross section end to end, at a parameter outside the training set",
    caption=("band: p5-p95 over the validation draws   |   markers: the "
             "R-matrix reference, shown so the solver is not taken on trust"),
    legend_in=0,
)

def _xs_per_system(key, stem, methods, title):
    """One sheet per system, one method subset, under the legacy filename.

    Only sigma(E) sheets can be built: the excitation block caches sigma(E)
    alone, and it fits RBM/LROM/ET but not ETP.
    """
    return Recipe(
        key=key, stem=stem, rows=("sigma", "error"),
        per_system=True, methods=methods,
        height_ratios=(1.7, 1.3), figsize=(5.6, 3.7),
        y_scale=("log", "log"), mark_thresholds=True,
        tolerance_ladder=("error",), suptitle=title,
        caption="band: p5-p95 over the validation draws",
        legend_in=0,
    )


XS_VS_RBM = _xs_per_system("observable_vs_emulators.xs_rbm", "xs_vs_rbm",
                           ("rbm",), "cross section: DBMM against RBM")
XS_VS_RBM_LROM = _xs_per_system("observable_vs_emulators.xs_rbm_lrom",
                                "xs_vs_rbm_lrom", ("rbm", "lrom"),
                                "cross section: DBMM against RBM and LROM")

#: F1. The premise: what the four systems contain and how far their observables
#: travel. Three columns, one row per system -- the transpose of the other suite
#: sheets.
SUITE_PREMISE = Recipe(
    key="manifold_spread.suite", stem="f1_suite",
    rows=("eigenphase", "wavefunction", "xsec"), figsize=(10.5, 7.6),
    y_scale=("linear", "linear", "log"), mark_thresholds=True,
    # raw string: "\times" in a plain string becomes TAB + "imes".
    headlines={"xsec": ("span", r"$\sigma$: $\times${:.0f} across the box",
                        "upper left")},
    suptitle="how far the observables travel when $V_0$ sweeps its box",
    legend_in=0,
)


#: F6, second half: where the offline stage is paid back.
AMORTISATION_SUITE = Recipe(
    # n_train marked on the axis: offline cost is the training solves to within
    # 20%, so the crossing sits just past the solves paid for it.
    vlines={"breakeven": ("n_train", "training solves")},
    key="amortisation.suite", stem="f6_amortisation",
    rows=("breakeven",), figsize=(9.5, 3.4),
    x_scale=("log",), y_scale=("log",), mark_thresholds=False,
    caption=("thin: every configuration the autocalibration selects that clears the "
             "posterior tolerance   |   thick: the best of them at each N, with "
             "the p5-p95 spread of its 150 timed predictions"),
    suptitle="amortisation: from how many evaluations an emulator pays for itself",
    legend_in=0,
)

#: F11. Online cost against mesh size, at fixed physics. Upper row: the solver's
#: own cost (without it a ratio between two emulators says nothing). Lower row:
#: the ratio, with the unity rule. Both emulators lift reduced coefficients to
#: the full mesh before extracting U, a term linear in N, so the ratio does not
#: fall towards one.
ONLINE_VS_MESH_SUITE = Recipe(
    key="online_vs_mesh.suite", stem="f11_online_vs_mesh",
    rows=("online", "ratio"), height_ratios=(2.0, 1.0), figsize=(9.5, 4.6),
    y_scale=("log", "linear"), mark_thresholds=False,
    hlines={"ratio": ("unity", "equal cost")},
    caption=("same rank and predictor budget at every N: only the mesh changes   |   "
             "both emulators lift to the full mesh before extracting U, so both "
             "carry a term linear in N and the ratio does not close"),
    suptitle="online cost against mesh size, at fixed physics",
    legend_in=0,
)

#: F7. One system by design: a demonstration, not a survey.
FAILURE_SINGLE = Recipe(
    key="coupling_budget_failure.single", stem="f7_failure",
    systems=(FAILURE_SYSTEM,),
    # no elastic panel: delta over a 0.1 deg window on a value of 31.75 makes a
    # budget accurate to 0.013% look like a failure. The "joint, on delta" curve
    # in the error panel carries the same point.
    rows=("inelastic", "errors", "predictors"),
    height_ratios=(2.6, 2.4, 1.3),
    figsize=(6.4, 6.2),
    y_scale=("log", "log", "linear"),
    mark_thresholds=False,
    tolerance_ladder=("errors",),
    legend_loc={"predictors": "upper right"},
    headlines={"errors": ("understated",
                          r"the elastic check understates by $\times${:.0f}")},
    caption=("same total predictor budget, same training set, same rank   |   "
             "bottom: where each budget spends its points, against where the "
             "interaction lives"),
    suptitle="LROM: the same predictor budget spent two ways",
    legend_in=0,
)

#: The interaction itself, one column per system. The premise of everything
#: after it: these are the objects a handful of modes is claimed to reproduce.
POTENTIAL_SUITE = Recipe(
    key="potential.suite", stem="f0_potential",
    rows=("potential",), figsize=(11.0, 3.8),
    mark_thresholds=False,
    headlines={"potential": ("coupling_scale", r"$V_{{ab}}$ shown $\times${:.0f}")},
    legend_all_columns=True,
    caption=("at the central parameters   |   the coupling is scaled to be "
             "visible beside the diagonal, by the factor stated in each panel"),
    suptitle="the interaction, term by term",
)

#: The error over the box as a distribution, at three ranks. A median with a
#: p5-p95 band says how accurate a basis is; this says whether it is accurate
#: everywhere, which is the question a campaign over the box actually asks.
ERROR_DISTRIBUTION_SUITE = Recipe(
    key="error_distribution.suite", stem="f5_distribution",
    rows=("distribution",), figsize=(9.5, 3.4),
    x_scale=("log",), mark_thresholds=False,
    # "best" collides with the bottom-left headline; a CDF leaves its upper-left
    # corner empty, so pin it upper right.
    legend_loc={"distribution": "upper right"},
    headlines={"distribution": ("median_rbm", r"RBM median {:.0e}", "upper left")},
    vlines={"distribution": ("floor", "solver floor")},
    caption=("every (validation draw, energy) pair, in decades of error   |   "
             "the line is the solver's own disagreement with the R-matrix code"),
    suptitle="how the emulator error is distributed, method against method",
    legend_in=0,
)

#: F9. The interior solution, solver against the emulators, at a parameter no
#: method has trained on.
WAVEFUNCTION_SUITE = Recipe(
    key="wavefunction_vs_emulators.suite", stem="f9_wavefunction",
    rows=("psi", "residual"), height_ratios=(1.8, 1.3),
    figsize=(9.5, 4.3), y_scale=("linear", "log"), mark_thresholds=False,
    headlines={"residual": ("worst_rbm", r"RBM worst {:.0e}")},
    caption="at one validation parameter, outside every training set",
    suptitle=r"the interior solution $\psi(r)$, solver against the emulators",
    legend_in=0,
)

#: F10. The same comparison on the phase shift, across the energy grid.
PHASE_SHIFT_SUITE = Recipe(
    key="phase_shift_vs_emulators.suite", stem="f10_phaseshift",
    rows=("delta", "residual"), height_ratios=(1.8, 1.3),
    figsize=(9.5, 4.3), y_scale=("linear", "log"), mark_thresholds=True,
    headlines={"residual": ("median_rbm", r"RBM median {:.0e} deg")},
    caption=("one emulator fitted per energy, at one validation parameter "
             "outside every training set"),
    suptitle=r"the phase shift $\delta(E)$, solver against the emulators",
    legend_in=0,
)

#: F6b. The cost-accuracy cloud, one sheet per system so each can be placed
#: independently. Two quantities, two recipes over one view.
CAT_ELASTIC = Recipe(
    key="cat_cloud.per_system", stem="cat_S00",
    rows=("cloud",), per_system=True, figsize=(6.0, 4.6),
    x_scale=("sqrtlog",), y_scale=("log",), mark_thresholds=False,
    orientation={"cloud": "← faster\n↓ more accurate"},
    caption=("one point per validation draw   |   band: the solver's own cost, "
             "p10-p90   |   a red square rings an underdetermined fit"),
    # no suptitle: the four sheets stack into one figure, where four copies of a
    # title read as an artefact.
    legend_loc={"cloud": "lower right"},
)

CAT_INELASTIC = Recipe(
    key="cat_cloud_inelastic.per_system", stem="cat_Sab",
    rows=("cloud",), per_system=True, figsize=(6.0, 4.6),
    x_scale=("sqrtlog",), y_scale=("log",), mark_thresholds=False,
    systems=("alpha12C_band", "alpha24Mg", "n238U"),
    orientation={"cloud": "← faster\n↓ more accurate"},
    caption=("one point per validation draw   |   band: the solver's own cost, "
             "p10-p90   |   coupled systems only: n+40Ca has no off-diagonal"),
    legend_loc={"cloud": "lower right"},
)

def _control_twin(recipe):
    """The same figure, drawn for the two control benchmarks instead.

    Generated, not hand-copied: only the systems covered and the output path
    differ.
    """
    return dataclasses.replace(
        recipe,
        key=recipe.key.replace(".", ".control_", 1) if "." in recipe.key
            else recipe.key + ".control",
        stem=recipe.stem + "_control",
        systems=CONTROL,
        figsize=(recipe.figsize[0] * 0.6, recipe.figsize[1]),
    )


RECIPES = {r.key: r for r in (SUITE_PREMISE, AMORTISATION_SUITE,
                              CAT_ELASTIC, CAT_INELASTIC,
                              WAVEFUNCTION_SUITE, PHASE_SHIFT_SUITE,
                              POTENTIAL_SUITE, ERROR_DISTRIBUTION_SUITE,
                              FAILURE_SINGLE, VALIDATION_SUITE, QUANTITY_SUITE,
                              SPECTRUM_SUITE, SPECTRUM_NS_SUITE,
                              ERROR_BY_RANK_SUITE,
                              INVARIANTS_SUITE, EXCITATION_SUITE,
                              XS_VS_RBM, XS_VS_RBM_LROM,
                              ONLINE_VS_MESH_SUITE)}

#: The same subjects again, for the two control benchmarks. F7 is excluded: it
#: is a one-system demonstration and neither control is that system.
_CONTROL_SUBJECTS = (SUITE_PREMISE, POTENTIAL_SUITE, VALIDATION_SUITE,
                     SPECTRUM_SUITE, ERROR_BY_RANK_SUITE, QUANTITY_SUITE,
                     ERROR_DISTRIBUTION_SUITE, EXCITATION_SUITE,
                     WAVEFUNCTION_SUITE, PHASE_SHIFT_SUITE,
                     AMORTISATION_SUITE, INVARIANTS_SUITE)

RECIPES.update({(t := _control_twin(r)).key: t for r in _CONTROL_SUBJECTS})
