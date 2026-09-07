"""The visual charter. Every figure imports from here and nowhere else.

Three visual channels, never more than one varying inside one panel:

    method   colour + marker + linestyle   DBMM, R-matrix, RBM, LROM, RBM_ET
    system   colour                        cross-system panels only
    sweep    sequential colormap           one parameter walking its box

One colour family for systems, another for methods, so a colour identifies
which kind of figure this is.

The tolerance ladder is the other half of the charter: accuracy claims are made
against three levels with three origins (experiment, posterior inference, solver
discretisation), drawn on every error axis by `draw_tolerance_ladder`.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# --- The charter's ten colours -----------------------------------------
# Two families that do not mix. The four benchmarks are muted and equal-weight;
# the two controls are lighter tints. Methods: near-black solver, slate R-matrix
# cross-check drawn on top of it (a distinct neutral, not the same black), one
# blue and one wine for the two emulators.
BLACK = "#1E1E20"     # DBMM
SLATE = "#63808C"     # R-matrix reference
BLUE  = "#2B5D95"     # RBM
WINE  = "#8C3560"     # LROM

NAVY  = "#3D4D6E"     # n + 40Ca
BRICK = "#AC484C"     # alpha + 12C
OCHRE = "#C09B19"     # alpha + 24Mg
SAGE  = "#527F65"     # n + 238U
CLAY  = "#8A6F4E"     # n + 58Ni, the vibrational budget test
MIST  = "#94A3BE"     # p + 12C, control
MAUVE = "#BE86A7"     # alpha + d, control

GREY  = "#7C8085"
INK   = "#4A4A4A"     # caption and banner text, darker than GREY for print


# --- Channel 1: the method -------------------------------------------
# One entry per method: colour, dash, weight. A method drawn as sparse markers
# on a reference curve (R-matrix on F2, F8) declares that in its own entry, so
# it keeps one identity whichever role it plays.

METHOD = {
    "dbmm": dict(label="DBMM", color=BLACK, ls="-", lw=1.6, zorder=3),
    # two further solver-side baselines, drawn only where evaluation cost is the
    # subject: same code, more work per parameter, so same colour, different dash.
    "dbmm_full": dict(label="DBMM, full rebuild", color=BLACK, ls="--",
                      lw=1.2, alpha=0.75, zorder=3),
    "rmat": dict(label="R-matrix", color=SLATE, ls=":", lw=1.4,
                 alpha=0.95, zorder=3),
    "rmatrix": dict(label="R-matrix", color=SLATE, ls="none", lw=0,
                    marker="o", ms=4.6, mfc="none", mew=1.4, zorder=6),
    "rbm": dict(label="RBM", color=BLUE, ls="--", lw=1.5, zorder=5),
    "lrom": dict(label="LROM", color=WINE, ls="-.", lw=1.5, zorder=5),
    "rbm_et": dict(label=r"RBM$_{ET}$", color=SAGE, ls=":", lw=1.8, zorder=5),
}

def method_kw(name, **override):
    """Style for one method. Raises KeyError for an unknown one: a defaulted pen
    is a curve nobody can attribute."""
    kw = dict(METHOD[name])
    kw.update(override)
    return kw


# --- Channel 2: the system -----------------------------------------
SYSTEM_COLOR = {
    "n40Ca":         NAVY,
    "alpha12C_band": BRICK,
    "alpha24Mg":     OCHRE,
    "n238U":         SAGE,
    "n58Ni":         CLAY,
    "p12C":          MIST,     # control
    "alpha_d":       MAUVE,    # control
}

# On overlay sheets colour is the only channel telling systems apart, and the
# palette is muted (brick and sage sit under the deuteranope-safe distance), so
# those panels carry a marker as well.
SYSTEM_MARKER = {
    "n40Ca": "o", "alpha12C_band": "s", "alpha24Mg": "^",
    "n238U": "D", "n58Ni": "X", "p12C": "v", "alpha_d": "P",
}


# --- Channel 3: a parameter sweeping its box ---
SWEEP_CMAP = "viridis"


def sweep_colors(n, lo=0.0, hi=0.92):
    return plt.get_cmap(SWEEP_CMAP)(np.linspace(lo, hi, n))


def sweep_colorbar(fig, ax, vmin, vmax, label, **kw):
    sm = plt.cm.ScalarMappable(cmap=SWEEP_CMAP,
                               norm=plt.Normalize(vmin=vmin, vmax=vmax))
    cb = fig.colorbar(sm, ax=ax, pad=kw.pop("pad", 0.02),
                      fraction=kw.pop("fraction", 0.046), **kw)
    cb.set_label(label, fontsize=8)
    cb.ax.tick_params(labelsize=7)
    return cb


# --- The tolerance ladder -------------------------------------------
# Three levels, three origins:
#   EPS_EXP    5e-2   experimental uncertainty on an elastic differential cross
#                     section; below it the emulator error is below the data.
#   EPS_POST   5e-3   the operating target: emulator invisible in the posterior
#                     (with sigma_emu = sigma_exp / 10 the posterior variance
#                     inflates by ~1%; stricter still, since the emulator error
#                     biases rather than merely broadens).
#   EPS_SOLVER 1e-4   the discretisation error of the solver itself (F3); below
#                     it the emulator is exact relative to its reference.

EPS_EXP = 5e-2
EPS_POST = 5e-3
EPS_SOLVER = 1e-4

TOLERANCES = {
    # all three grey (not measurements, must not compete for a hue), told apart
    # by dash; the operating target keeps the heaviest stroke.
    "exp":    (EPS_EXP,    r"$\epsilon_{\rm exp}$ 5%",    ":",  0.8),
    "post":   (EPS_POST,   r"$\epsilon_{\rm post}$ 0.5%", "--", 1.2),
    "solver": (EPS_SOLVER, r"$\epsilon_{\rm solver}$",    "-.", 0.8),
}


def draw_tolerance_ladder(ax, which=("exp", "post", "solver"), axis="y",
                          label=True, x=0.985, fontsize=6.5):
    """Draw the accuracy ladder on an error axis, on every panel that carries an
    error, so a claim like "clears the posterior tolerance at rank 48" reads off."""
    for name in which:
        value, text, ls, lw = TOLERANCES[name]
        line = ax.axhline if axis == "y" else ax.axvline
        emphasis = (name == "post")
        line(value, color=GREY, lw=lw, ls=ls,
             alpha=0.9 if emphasis else 0.6, zorder=1)
        if label:
            ax.text(x, value, text, transform=ax.get_yaxis_transform(),
                    ha="right", va="bottom", fontsize=fontsize, color=GREY,
                    alpha=0.95 if emphasis else 0.75, zorder=1)


# --- The number the figure exists for ---

def headline(ax, text, loc="upper right", fontsize=7.5, color=None):
    """Print the figure's own headline number inside the axes -- the same number
    the prose quotes, from the same cached array."""
    pos = {"upper right": (0.975, 0.955, "right", "top"),
           "upper left": (0.025, 0.955, "left", "top"),
           "lower right": (0.975, 0.045, "right", "bottom"),
           "lower left": (0.025, 0.045, "left", "bottom")}[loc]
    x, y, ha, va = pos
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va,
            fontsize=fontsize, color=color or GREY,
            # nearly opaque: at 0.72 the F6 amortisation lines show through the
            # digits.
            bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="none",
                      alpha=0.92))


# Figure size is a per-figure choice and lives on its Recipe, not here.


def setup():
    """Install the rcParams. Called on import; idempotent."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 9,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.grid": False,
        "axes.linewidth": 0.8,
        "lines.solid_capstyle": "round",
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "legend.frameon": False,
        "errorbar.capsize": 2,
    })


setup()
