"""The figures the memoir cites keep their filenames.

Every path here is \\includegraphics'd by Memoire/, so a stem is a published
interface: renaming one breaks a LaTeX build months later. Pinning them to the
recipes makes the rename fail here instead.

Extracted from the .tex with:

    grep -rhoE 'includegraphics[^{]*\\{[^}]*\\}' Memoire/ --include=*.tex
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.recipes import RECIPES

#: Suite sheets: one file per recipe, written flat as figures_out/<stem>.png.
CITED_SUITE = [
    "f0_potential", "f0_potential_control",
    "f1_suite_control",
    "f2_validation", "f2_validation_control",
    "f3_invariants",
    "f4_error_by_rank", "f4_spectrum", "f4_spectrum_control",
    "f5_distribution", "f5_quantity",
    "f6_amortisation",
    "f8_excitation", "f8_excitation_control",
    "f9_wavefunction", "f9_wavefunction_control",
    "f10_phaseshift", "f10_phaseshift_control",
]

#: Per-system sheets: figures_out/<system>_<stem>.png. The memoir cites the
#: elastic cloud for all four suite systems and the inelastic one for the three
#: coupled ones -- n+40Ca has no off-diagonal element, so cat_Sab does not exist
#: for it and is correctly not cited.
CITED_PER_SYSTEM = [
    ("n40Ca", "cat_S00"),
    ("alpha12C_band", "cat_S00"), ("alpha12C_band", "cat_Sab"),
    ("alpha24Mg", "cat_S00"), ("alpha24Mg", "cat_Sab"),
    ("n238U", "cat_S00"), ("n238U", "cat_Sab"),
]


def _owners(stem):
    return [key for key, r in RECIPES.items() if r.stem == stem]


def test_recipe_stems_are_unique():
    stems = [r.stem for r in RECIPES.values()]
    dupes = {s for s in stems if stems.count(s) > 1}
    assert not dupes, f"two recipes write the same file: {sorted(dupes)}"


# ── the dependency map, and the invariant it exists to support ───────────────

def _blocks_a_view_reads(name):
    """Which blocks a view really opens, recorded by wrapping cache.read.

    A declared map that has drifted from the code is worse than no map, because
    it is the one somebody will trust. This is what stops that.
    """
    from pipeline import cache, draw, views

    opened = set()
    real = cache.read

    def spy(system_key, block):
        opened.add(block)
        return real(system_key, block)

    cache.read = views.cache.read = spy
    try:
        for key in ("n40Ca", "alpha24Mg", "p12C"):
            try:
                draw.VIEWS[name](key)
            except FileNotFoundError:
                pass                      # a block this system has not computed
    finally:
        cache.read = views.cache.read = real
    return opened


@pytest.mark.parametrize("name", sorted(
    __import__("pipeline.draw", fromlist=["draw"]).VIEWS))
def test_declared_blocks_match_what_the_view_reads(name):
    from pipeline.draw import VIEW_BLOCKS
    opened = _blocks_a_view_reads(name)
    if not opened:
        pytest.skip(f"{name} opened no block (its data is not computed here)")
    assert opened == {VIEW_BLOCKS[name]}, (
        f"draw.VIEW_BLOCKS says {name} reads {VIEW_BLOCKS[name]!r}, "
        f"but it opened {sorted(opened)}")


def test_timings_live_only_in_the_machine_dependent_blocks():
    """No block outside cache.MACHINE_DEPENDENT may store a timing.

    This is what makes api.rebench() a complete answer to a machine change.
    """
    import numpy as np
    from pipeline import cache

    offenders = []
    for npz in sorted(cache.current().glob("*/*.npz")):
        if npz.stem in cache.MACHINE_DEPENDENT:
            continue
        with np.load(npz, allow_pickle=False) as z:
            bad = [n for n in z.files
                   if n.startswith("t_") or n == "seconds"]
        if bad:
            offenders.append(f"{npz.parent.name}/{npz.stem}: {bad}")
    assert not offenders, (
        "timings outside cache.MACHINE_DEPENDENT:\n  " + "\n  ".join(offenders))


@pytest.mark.parametrize("stem", CITED_SUITE)
def test_cited_suite_sheet_has_exactly_one_recipe(stem):
    owners = _owners(stem)
    assert len(owners) == 1, (
        f"figures_out/{stem}.png is \\includegraphics'd by the memoir and is "
        f"claimed by {len(owners)} recipes ({owners})")


@pytest.mark.parametrize("system,stem", CITED_PER_SYSTEM,
                         ids=lambda v: str(v))
def test_cited_per_system_sheet_has_a_recipe_that_covers_it(system, stem):
    owners = _owners(stem)
    assert len(owners) == 1, (
        f"figures_out/{system}_{stem}.png is cited by the memoir and is "
        f"claimed by {len(owners)} recipes ({owners})")
    recipe = RECIPES[owners[0]]
    assert recipe.per_system, (
        f"{owners[0]} writes one sheet for the whole suite, but the memoir "
        f"cites it as figures_out/{system}_{stem}.png")
    # An empty `systems` means the suite, which covers every cited system.
    assert not recipe.systems or system in recipe.systems, (
        f"{owners[0]} declares systems={recipe.systems}, so it never draws "
        f"{system} -- yet the memoir cites figures_out/{system}_{stem}.png")
