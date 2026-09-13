#!/usr/bin/env python3
"""
One command for the whole pipeline.

    python run_all.py

Stages run in order and are resumable — every LLM call is cached on disk, so
re-running after an interruption costs nothing and re-uses previous responses.

The pipeline pauses exactly once, at human labelling, because that is the one
step no model may do. Run the labelling command it prints, then run this again.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

STAGES = [
    ("20_build_brand_dataset.py", "build brand dataset + leak-safe splits"),
    ("21_discover_intents.py", "discover intent taxonomy from real DEV clusters"),
    ("22_make_gold_candidates.py", "sample + pre-annotate gold candidates"),
]
AFTER_LABELS = [
    ("24_run_evaluation.py", "agent + baselines + judge + metrics"),
    ("25_make_report.py", "failure analysis + REPORT.md"),
    ("99_audit.py", "final audit"),
]


def run(script: str, desc: str, extra=None) -> bool:
    print("\n" + "=" * 70)
    print(f"[{script}]  {desc}")
    print("=" * 70)
    t0 = time.time()
    cmd = [sys.executable, str(ROOT / "scripts" / script)] + (extra or [])
    rc = subprocess.call(cmd, cwd=ROOT)
    print(f"  -> exit {rc} in {time.time()-t0:.1f}s")
    return rc == 0


def splits_are_stale() -> bool:
    """Splits built before the strict-temporal-window fix (DECISIONS D11) have no
    `temporal_split` block in their manifest. Reusing them would silently keep the
    leakage the audit flagged, so they are rebuilt — along with everything derived
    from them."""
    mf = ROOT / "results" / "split_manifest.json"
    if not (ROOT / "data" / "processed" / "splits" / "history.csv").exists():
        return False
    if not mf.exists():
        return True
    try:
        return "temporal_split" not in json.loads(mf.read_text(encoding="utf-8"))
    except Exception:
        return True


def invalidate_downstream() -> None:
    """Anything sampled from stale splits is itself stale."""
    for p in (ROOT / "data" / "processed" / "taxonomy_v1.json",
              ROOT / "data" / "gold" / "gold_candidates.csv"):
        if p.exists():
            p.rename(p.with_suffix(p.suffix + ".stale"))
            print(f"  moved aside: {p.name} -> {p.name}.stale")
    gold = ROOT / "data" / "gold" / "golden_set.csv"
    if gold.exists():
        print(f"\n  !! {gold.name} holds HUMAN labels and was NOT touched.")
        print("  !! Its rows came from the old gold_pool, so after the rebuild some")
        print("  !! may no longer be in the pool. Re-run labelling for any that")
        print("  !! disappear; existing labels for surviving rows still count.\n")


def main() -> int:
    args = sys.argv[1:]
    force = "--rebuild" in args
    if force or splits_are_stale():
        why = "forced" if force else "built before the D11 temporal fix"
        print(f"[rebuild] splits are stale ({why}) — rebuilding from raw")
        invalidate_downstream()
        for p in (ROOT / "data" / "processed" / "splits").glob("*.csv"):
            p.unlink()

    for script, desc in STAGES:
        out = ROOT / "data" / "processed"
        if script.startswith("20") and (out / "splits" / "history.csv").exists():
            print(f"[skip] {script} — splits already built")
            continue
        if script.startswith("21") and (out / "taxonomy_v1.json").exists():
            print(f"[skip] {script} — taxonomy already discovered")
            continue
        if script.startswith("22") and (ROOT / "data" / "gold" / "gold_candidates.csv").exists():
            print(f"[skip] {script} — gold candidates already sampled")
            continue
        if not run(script, desc):
            print("\nStopped: that stage failed. Fix it and re-run; "
                  "completed work is cached.")
            return 1

    gold = ROOT / "data" / "gold" / "golden_set.csv"
    if not gold.exists():
        print("\n" + "=" * 70)
        print("PAUSED — human labelling required")
        print("=" * 70)
        print("""
This is the one step a model must not do for you. 150 candidates are ready.

    python scripts/23_label_cli.py

Two keypresses per message: intent number, then [a]uto-handle or [e]scalate.
Progress saves after every label, so you can stop and resume freely.
The model's guess stays hidden until after you commit, which is what makes
these human labels rather than a model grading itself.

Then run this again:

    python run_all.py
""")
        return 0

    passthrough = [a for a in args if a != "--rebuild"]
    for script, desc in AFTER_LABELS:
        if not run(script, desc, extra=passthrough):
            return 1

    print("\n" + "=" * 70)
    print("DONE — REPORT.md, DECISIONS.md, results/ are ready")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
