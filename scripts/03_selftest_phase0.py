#!/usr/bin/env python3
"""
Phase 0 self-tests. Deterministic, synthetic, no network, no dataset, no
dependencies beyond what Phase 0 already needs.

Every fixture below is hand-constructed with a KNOWN correct answer. These tests
prove the inspection code behaves, not anything about the real dataset.

    python scripts/03_selftest_phase0.py

Exits 0 if all pass, 1 otherwise. Also runs under pytest if you have it.
"""

from __future__ import annotations

import csv
import importlib.util
import sys
import tempfile
from pathlib import Path

import numpy as np

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

# module name starts with a digit, so import it by path
_SPEC = importlib.util.spec_from_file_location(
    "inspect_dataset", Path(__file__).with_name("02_inspect_dataset.py"))
insp = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(insp)

COLUMNS = ["tweet_id", "author_id", "inbound", "created_at", "text",
           "response_tweet_id", "in_response_to_tweet_id"]
TS = "Mon Oct 02 10:00:00 +0000 2017"


def write_csv(rows: list[dict], path: Path) -> Path:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in COLUMNS})
    return path


def row(tid, author, inbound, text, parent=""):
    return {"tweet_id": tid, "author_id": author, "inbound": str(bool(inbound)),
            "created_at": TS, "text": text, "response_tweet_id": "",
            "in_response_to_tweet_id": parent}


def run_pipeline(rows, tmp: Path, name: str):
    csv_path = write_csv(rows, tmp / f"{name}.csv")
    a = insp.pass_a(csv_path, chunksize=3, sample_rows=None, quiet=True)
    th = insp.reconstruct_threads(a["graph"]["tweet_id"], a["graph"]["parent_id"])
    return a, th


# --------------------------------------------------------------- tests
def test_thread_depth_chain(tmp: Path):
    """root -> child -> grandchild -> great-grandchild  =>  depths 0,1,2,3"""
    rows = [
        row(1, "100", True, "root message"),
        row(2, "BrandX", False, "reply", parent=1),
        row(3, "100", True, "follow up", parent=2),
        row(4, "BrandX", False, "second reply", parent=3),
    ]
    a, th = run_pipeline(rows, tmp, "chain")
    got = th["depth"].tolist()
    assert got == [0, 1, 2, 3], f"expected [0,1,2,3], got {got}"
    assert th["stats"]["depth_max"] == 3, th["stats"]["depth_max"]
    # all four tweets must land in ONE conversation
    assert len(set(th["conversation_id"].tolist())) == 1
    assert th["stats"]["n_conversations"] == 1
    return "chain depths exact (0,1,2,3); single conversation"


def test_thread_depth_branching(tmp: Path):
    """A root with two children, one of which has its own child."""
    rows = [
        row(1, "100", True, "root"),
        row(2, "BrandX", False, "child a", parent=1),
        row(3, "BrandX", False, "child b", parent=1),
        row(4, "100", True, "grandchild of a", parent=2),
    ]
    a, th = run_pipeline(rows, tmp, "branch")
    got = th["depth"].tolist()
    assert got == [0, 1, 1, 2], f"expected [0,1,1,2], got {got}"
    return "branching depths exact (0,1,1,2)"


def test_deep_chain_not_logarithmic(tmp: Path):
    """The regression guard: the old pointer-jumping code reported 3 for a
    chain of length 7. Depth must be linear in edges walked."""
    rows = [row(1, "100", True, "t0")]
    for i in range(2, 9):
        rows.append(row(i, "BrandX" if i % 2 == 0 else "100",
                        i % 2 == 1, f"t{i}", parent=i - 1))
    a, th = run_pipeline(rows, tmp, "deep")
    got = th["depth"].tolist()
    assert got == list(range(8)), f"expected 0..7, got {got}"
    return "8-node chain gives depths 0..7 (not log2)"


def test_orphan_parent(tmp: Path):
    """A reply whose parent id is absent from the file is a root, not a crash."""
    rows = [
        row(1, "100", True, "normal root"),
        row(2, "BrandX", False, "reply to a tweet not in this file",
            parent=999999),
    ]
    a, th = run_pipeline(rows, tmp, "orphan")
    assert th["stats"]["n_orphan_replies"] == 1, th["stats"]
    assert th["stats"]["n_with_parent_field"] == 1
    assert th["stats"]["n_parent_resolvable"] == 0
    assert th["depth"].tolist() == [0, 0], th["depth"].tolist()
    assert th["stats"]["n_conversations"] == 2
    return "orphan reply counted, treated as its own root, depth 0"


def test_cycle_detection(tmp: Path):
    """Two tweets pointing at each other must be reported, never silently
    assigned a plausible-looking depth."""
    rows = [
        row(1, "100", True, "a", parent=2),
        row(2, "BrandX", False, "b", parent=1),
    ]
    a, th = run_pipeline(rows, tmp, "cycle")
    assert th["stats"]["n_unconverged_cycles"] == 2, th["stats"]
    assert th["depth"].tolist() == [-1, -1], th["depth"].tolist()
    # sentinel depths must be kept out of the histogram
    assert all(int(k) >= 0 for k in th["stats"]["depth_histogram"])
    return "2-cycle detected; depth set to -1 sentinel; excluded from histogram"


def test_self_parent(tmp: Path):
    """A tweet that is its own parent is a root with depth 0."""
    rows = [row(1, "100", True, "self referential", parent=1)]
    a, th = run_pipeline(rows, tmp, "selfparent")
    assert th["depth"].tolist() == [0], th["depth"].tolist()
    assert th["stats"]["n_unconverged_cycles"] == 0
    return "self-parent handled as root, depth 0, no false cycle"


def test_duplicate_tweet_ids(tmp: Path):
    rows = [
        row(1, "100", True, "hello"),
        row(1, "100", True, "hello"),          # exact duplicate row
        row(2, "BrandX", False, "reply", parent=1),
    ]
    a, th = run_pipeline(rows, tmp, "dupes")
    n_rows = a["acc"]["n_rows"]
    n_unique = len(np.unique(a["graph"]["tweet_id"]))
    assert n_rows == 3 and n_unique == 2, (n_rows, n_unique)
    uniq, counts = np.unique(a["graph"]["text_hash"], return_counts=True)
    assert int((counts > 1).sum()) == 1, counts.tolist()
    return "duplicate tweet_id and repeated normalised text both counted"


def test_deterministic_hash(tmp: Path):
    """Must be stable within a process, across processes, and across runs.
    Python's hash() is PYTHONHASHSEED-randomised and would fail the subprocess
    check below."""
    import subprocess
    s = "where is my bag it never arrived"
    a1, a2 = insp.hash64(s), insp.hash64(s)
    assert a1 == a2, "not stable within a process"
    assert insp.hash64("a") != insp.hash64("b")
    assert 0 <= a1 < 2 ** 63, a1

    # fixed expected value pins the algorithm: a change here is a silent
    # reproducibility break in duplicate detection
    expected = int.from_bytes(
        __import__("hashlib").blake2b(s.encode("utf-8"),
                                      digest_size=8).digest(), "big") & 0x7FFFFFFFFFFFFFFF
    assert a1 == expected, (a1, expected)

    # different process, different hash seed -> same value
    code = (f"import importlib.util,sys;"
            f"spec=importlib.util.spec_from_file_location('m',r'{Path(__file__).with_name('02_inspect_dataset.py')}');"
            f"m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);"
            f"print(m.hash64({s!r}))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, env={**__import__("os").environ,
                                         "PYTHONHASHSEED": "12345"})
    assert out.returncode == 0, out.stderr
    assert int(out.stdout.strip()) == a1, (out.stdout.strip(), a1)
    return f"blake2b-64 stable in-process and across PYTHONHASHSEED ({a1})"


def test_multi_brand_conversation(tmp: Path):
    """A thread where two brands both reply must NOT be credited to either in
    the selection statistics, and must still appear in the raw counts."""
    rows = [
        # conversation 1: single brand, clean
        row(1, "100", True, "@BrandA where is my bag"),
        row(2, "BrandA", False, "Bags are traced via our portal.", parent=1),
        # conversation 2: customer tags two brands, BOTH reply
        row(3, "200", True, "@BrandA @BrandB who is responsible here"),
        row(4, "BrandA", False, "Not us, try BrandB.", parent=3),
        row(5, "BrandB", False, "We can look into it.", parent=4),
    ]
    a, th = run_pipeline(rows, tmp, "multibrand")
    table, meta = insp.build_brand_table(
        a["graph"], th["conversation_id"], th["depth"],
        a["author_agg"], a["author_code_map"])

    assert meta["n_conversations_multibrand"] == 1, meta
    assert meta["n_conversations_unambiguous"] == 1, meta

    t = table.set_index("brand")
    # only the clean conversation is attributed, and only to BrandA
    assert int(t.loc["BrandA", "n_conversations_unambiguous"]) == 1
    assert int(t.loc["BrandB", "n_conversations_unambiguous"]) == 0
    # the multi-brand thread is retained against BOTH, attributed to NEITHER
    assert int(t.loc["BrandA", "n_multibrand_convs_participated"]) == 1
    assert int(t.loc["BrandB", "n_multibrand_convs_participated"]) == 1
    # author-level outbound counts are unaffected by attribution
    assert int(t.loc["BrandA", "n_outbound_tweets"]) == 2
    assert int(t.loc["BrandB", "n_outbound_tweets"]) == 1
    return "multi-brand thread excluded from both brands' selection stats, retained raw"


def test_deflection_and_script_metrics(tmp: Path):
    rows = [
        row(1, "100", True, "hi there"),
        row(2, "BrandA", False, "Sorry! Please DM us your booking ref.", parent=1),
        row(3, "100", True, "ok"),
        row(4, "BrandA", False,
            "Refunds return to the original payment method within 5-7 working "
            "days and you will get an email confirmation once processed.", parent=3),
    ]
    a, th = run_pipeline(rows, tmp, "deflect")
    agg = a["author_agg"]["BrandA"]
    assert agg["n_outbound"] == 2
    assert agg["n_deflect_loose"] == 1, agg
    assert agg["n_deflect_strict"] == 1, agg

    # script metric: names what it measures
    assert insp.predominantly_non_latin("आपका सामान कहाँ है") is True
    assert insp.predominantly_non_latin("amar bag kothay") is False   # Banglish = Latin
    assert insp.predominantly_non_latin("café") is False
    assert insp.predominantly_non_latin("ab") is False                # below min_letters
    return "deflection regex counts 1/2; non-Latin metric is script-based, not language"


TESTS = [
    test_thread_depth_chain,
    test_thread_depth_branching,
    test_deep_chain_not_logarithmic,
    test_orphan_parent,
    test_cycle_detection,
    test_self_parent,
    test_duplicate_tweet_ids,
    test_deterministic_hash,
    test_multi_brand_conversation,
    test_deflection_and_script_metrics,
]


def main() -> int:
    print("=" * 72)
    print("PHASE 0 SELF-TESTS (synthetic fixtures with known answers)")
    print("=" * 72)
    passed, failed = 0, []
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for fn in TESTS:
            name = fn.__name__.replace("test_", "")
            try:
                detail = fn(tmp)
                print(f"  PASS  {name:34s} {detail}")
                passed += 1
            except AssertionError as exc:
                print(f"  FAIL  {name:34s} {exc}")
                failed.append(name)
            except Exception as exc:
                print(f"  ERROR {name:34s} {type(exc).__name__}: {exc}")
                failed.append(name)
    print("-" * 72)
    print(f"{passed}/{len(TESTS)} passed" +
          (f" — FAILED: {', '.join(failed)}" if failed else ""))
    print("Note: these test the inspection code on synthetic data. They say "
          "nothing about the real dataset.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
