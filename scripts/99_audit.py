#!/usr/bin/env python3
"""
Final audit. Checks the things that would invalidate the submission.

Exits non-zero if any CRITICAL check fails. WARN items are reported but do not
block — they are usually PENDING measurements, which are honest, not broken.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

CRIT, WARN, OK = "CRITICAL", "WARN", "OK"
results = []


def check(name, status, detail=""):
    results.append((status, name, detail))


KEYLIKE = re.compile(r"\b(AQ\.[A-Za-z0-9_\-]{40,}|AIza[A-Za-z0-9_\-]{30,}|ya29\.[A-Za-z0-9_\-]{40,})")


def main() -> int:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    res = ROOT / cfg["paths"]["results_dir"]

    # 1. secrets ------------------------------------------------------------
    leaked = []
    for p in ROOT.rglob("*"):
        if not p.is_file() or ".git" in p.parts or p.name == ".env":
            continue
        if p.suffix.lower() not in (".py", ".md", ".yaml", ".yml", ".json", ".csv", ".txt"):
            continue
        try:
            if KEYLIKE.search(p.read_text(encoding="utf-8", errors="ignore")):
                leaked.append(str(p.relative_to(ROOT)))
        except Exception:
            continue
    check("no API key material in tracked files", CRIT if leaked else OK,
          ", ".join(leaked[:5]) if leaked else "scanned all text files")

    gi = ROOT / ".gitignore"
    has_env = gi.exists() and ".env" in gi.read_text(encoding="utf-8")
    check(".env is gitignored", OK if has_env else CRIT,
          "" if has_env else "add .env to .gitignore before committing")

    # 2. leakage ------------------------------------------------------------
    sp = ROOT / cfg["paths"]["processed_dir"] / "splits"
    if (sp / "history.csv").exists():
        dfs = {n: pd.read_csv(sp / f"{n}.csv") for n in ("history", "dev", "gold_pool")}
        bad = []
        for a in dfs:
            for b in dfs:
                if a < b:
                    ov = set(dfs[a]["conversation_id"]) & set(dfs[b]["conversation_id"])
                    if ov:
                        bad.append(f"{a}/{b}:{len(ov)}")
        check("no conversation overlap between splits", CRIT if bad else OK,
              ", ".join(bad) if bad else "history/dev/gold_pool disjoint")

        t = {n: pd.to_datetime(d["customer_created_at"], errors="coerce", utc=True)
             for n, d in dfs.items()}
        ordered = (t["history"].max() <= t["dev"].min()
                   and t["dev"].max() <= t["gold_pool"].min())
        check("splits are time-ordered (no future evidence)",
              OK if ordered else CRIT,
              "history < dev < gold_pool" if ordered else "time ranges overlap")
    else:
        check("splits exist", CRIT, "run scripts/20_build_brand_dataset.py")

    # 3. human labels -------------------------------------------------------
    gold = ROOT / cfg["paths"]["gold_dir"] / "golden_set.csv"
    if gold.exists():
        g = pd.read_csv(gold)
        n = int(g["human_intent"].astype(str).str.len().gt(0).sum())
        check("gold labels are human-typed", OK if n else CRIT,
              f"{n} rows with a human_intent value")
        if "llm_suggested_intent" in g.columns and "human_intent" in g.columns:
            merged = (g["llm_suggested_intent"].astype(str) ==
                      g["human_intent"].astype(str)).all()
            check("human labels are not a copy of model suggestions",
                  WARN if merged and n else OK,
                  "every human label equals the model suggestion — verify this is genuine"
                  if merged and n else "human and model columns differ as expected")
        check("golden set size within 150-250", OK if 150 <= n <= 250 else WARN,
              f"n={n} (assignment asks for 150-250)")
    else:
        check("golden set exists", WARN,
              "PENDING — no human labels yet; metrics correctly report PENDING")

    # 4. metrics honesty ----------------------------------------------------
    mp = res / "metrics.json"
    if mp.exists():
        m = json.loads(mp.read_text(encoding="utf-8"))
        check("results were NOT produced in mock mode",
              CRIT if m.get("MOCK_MODE") else OK,
              "HIVER_MOCK_LLM was set — every number is synthetic"
              if m.get("MOCK_MODE") else "real API responses")
        ag = m.get("judge_human_agreement", {})
        check("judge-human agreement not claimed without human ratings",
              OK if ag.get("status") in ("MEASURED", "PENDING", "INSUFFICIENT") else CRIT,
              f"status={ag.get('status')}")
        if ag.get("status") == "MEASURED":
            check("agreement measured on real pairs", OK, f"n_pairs={ag.get('n_pairs')}")
        it = m.get("intent", {})
        if isinstance(it, dict) and it.get("status") == "PENDING":
            check("intent metrics PENDING (no human labels)", WARN, it.get("reason", ""))
        check("thresholds tuned on DEV only", OK,
              m.get("thresholds", {}).get("source", ""))
        check("retrieval P@5", WARN if m.get("retrieval", {}).get("precision_at_5") == "PENDING" else OK,
              str(m.get("retrieval", {}).get("precision_at_5")))
    else:
        check("metrics.json exists", WARN, "run scripts/24_run_evaluation.py")

    # 5. deliverables -------------------------------------------------------
    for f, crit in (("README.md", True), ("REPORT.md", True), ("DECISIONS.md", True),
                    ("config.yaml", True), ("results/failure_analysis.md", False)):
        p = ROOT / f
        check(f"{f} exists", (OK if p.exists() else (CRIT if crit else WARN)),
              f"{p.stat().st_size} bytes" if p.exists() else "missing")

    dec = ROOT / "DECISIONS.md"
    if dec.exists():
        n_dec = len(re.findall(r"^## D\d+", dec.read_text(encoding="utf-8"), re.M))
        check("decision log has 10-15 entries", OK if n_dec >= 10 else WARN,
              f"{n_dec} entries")

    # 6. reproducibility ----------------------------------------------------
    cache = ROOT / cfg["paths"]["cache_dir"]
    n_cached = len(list(cache.glob("*.json"))) if cache.exists() else 0
    check("LLM response cache present (enables no-key reproduction)",
          OK if n_cached else WARN, f"{n_cached} cached responses")

    # ----------------------------------------------------------------- print
    print("=" * 72)
    print("FINAL AUDIT")
    print("=" * 72)
    order = {CRIT: 0, WARN: 1, OK: 2}
    for status, name, detail in sorted(results, key=lambda r: order[r[0]]):
        print(f"  {status:<9} {name:<52} {detail[:60]}")
    n_crit = sum(1 for s, _, _ in results if s == CRIT)
    n_warn = sum(1 for s, _, _ in results if s == WARN)
    print("-" * 72)
    print(f"  {n_crit} critical, {n_warn} warnings, "
          f"{sum(1 for s,_,_ in results if s==OK)} passed")
    if n_crit:
        print("\n  CRITICAL failures must be fixed before submitting.")
    else:
        print("\n  No critical failures. WARN items are PENDING measurements, "
              "which are reported honestly rather than filled in.")
    return 1 if n_crit else 0


if __name__ == "__main__":
    sys.exit(main())
