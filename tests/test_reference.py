"""No cached number has moved since the reference was frozen.

pipeline.cache's core_fingerprint only says a result *may* have changed
(different code). This compares the values: tests/reference_blocks.json holds
the hash of every cached array, and any difference here is a bug or a decision.

    api.freeze()        # re-freeze against the cache in force -- a deliberate act
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.reference import compare, snapshot

REFERENCE = Path(__file__).resolve().parent / "reference_blocks.json"
RESULTS = Path(__file__).resolve().parent.parent / "results"


@pytest.fixture(scope="module")
def diff():
    if not REFERENCE.exists():
        pytest.skip("no frozen reference; freeze one with api.freeze()")
    if not RESULTS.exists() or not any(RESULTS.glob("*/*.npz")):
        pytest.skip("no computed cache; run Run().execute() first")
    ref = json.loads(REFERENCE.read_text(encoding="utf-8"))
    return ref, compare(ref, snapshot(RESULTS))


def test_no_block_changed(diff):
    _, (_, _, changed) = diff
    assert not changed, "\n".join(
        [f"{k}: {', '.join(w[:4])}" for k, w in changed])


def test_no_block_vanished(diff):
    _, (missing, _, _) = diff
    assert not missing, (
        f"blocks in the reference but not in the cache: {missing}. "
        f"Either the cache is partial, or a block stopped being produced.")


def test_reference_is_not_empty(diff):
    ref, _ = diff
    n_arrays = sum(len(v.get("arrays", {})) for v in ref.values())
    assert len(ref) >= 20 and n_arrays >= 200, (
        f"the frozen reference holds {len(ref)} blocks / {n_arrays} arrays, "
        f"which is too little to be the full suite; it was probably snapshotted "
        f"from a partial cache.")
