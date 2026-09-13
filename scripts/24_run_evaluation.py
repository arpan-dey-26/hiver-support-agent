#!/usr/bin/env python3
"""
Phases 4-8 — retrieval, agent, baselines, judge, metrics.

Runs the full system and all baselines over the evaluation rows, scores every
reply with the separate judge model, and computes metrics.

Honesty rules enforced in code, not just intended:
  * intent and escalation metrics are computed ONLY from human labels in
    golden_set.csv. With no human labels the result is PENDING, not a number.
  * judge-human agreement is PENDING unless human reply ratings exist.
  * judge parse failures are counted and reported, never silently dropped.
  * thresholds come from DEV statistics; gold is read by this script alone and
    only at scoring time.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import (agreement, agent as agent_mod, baselines, classify, judge,  # noqa: E402
                 llm, metrics, retrieval, taxonomy as tax_mod)

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="evaluate only the first N rows (smoke run)")
    ap.add_argument("--silver-n", type=int, default=400,
                    help="DEV messages to silver-label for the distilled classifier")
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--judge-n", type=int, default=60,
                    help="examples judged for the agent and B1. The judge is the "
                         "single biggest consumer of a 15 rpm quota, so it runs on "
                         "a defensible paired sample rather than every row.")
    ap.add_argument("--ablation-n", type=int, default=40,
                    help="rows to run the B2/B3 ablations and their judging on. "
                         "They are diagnostics, not headline numbers, so capping "
                         "them saves a large share of the API budget.")
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    if llm.mock_mode():
        print("!" * 70)
        print("!! HIVER_MOCK_LLM=1 — ALL LLM OUTPUT IS SYNTHETIC.")
        print("!! Results are for testing the pipeline ONLY and must never be")
        print("!! reported. The final audit will fail CRITICAL on this.")
        print("!" * 70)
    seed = cfg["project"]["seed"]
    rng = np.random.default_rng(seed)
    cache = ROOT / cfg["paths"]["cache_dir"]
    splits = ROOT / cfg["paths"]["processed_dir"] / "splits"
    res = ROOT / cfg["paths"]["results_dir"]
    res.mkdir(parents=True, exist_ok=True)

    tax = tax_mod.load(ROOT / "data" / "processed" / "taxonomy_v1.json")
    history = pd.read_csv(splits / "history.csv")
    dev = pd.read_csv(splits / "dev.csv")
    print(f"history {len(history):,} | dev {len(dev):,}")

    # ---------------------------------------------------------- retrieval
    print("building BM25 index over history...")
    index = retrieval.HistoryIndex(history)

    # ------------------------------------------- silver labels -> distilled clf
    print(f"silver-labelling {min(args.silver_n, len(dev))} DEV messages...")
    dev_s = dev.sample(n=min(args.silver_n, len(dev)), random_state=seed)
    silver = classify.SilverClassifier()
    try:
        sl = classify.llm_classify_batch(
            dev_s["customer_text"].astype(str).tolist(), tax,
            cfg["brand"]["selected"], cfg["models"]["generation"], cache,
            batch_size=25)
        labels = [s["intent"] for s in sl]
        silver.fit(dev_s["customer_text"].astype(str).tolist(), labels)
        print(f"  distilled classifier fitted on {len(labels)} silver labels")
    except Exception as exc:                                   # noqa: BLE001
        print(f"  silver labelling failed ({type(exc).__name__}: {exc})")
        labels = []

    # ------------------------------------- thresholds from DEV statistics only
    dev_probe = dev.sample(n=min(200, len(dev)), random_state=seed)
    top_scores = []
    for _, r in dev_probe.iterrows():
        h = index.search(str(r["customer_text"]),
                         query_time=r.get("customer_created_at"), k=5)
        top_scores.append(h[0]["score"] if h else 0.0)
    theta = float(np.percentile(top_scores, 25)) if top_scores else 1.0
    tau = 0.70
    m = 1
    cfg["retrieval"]["score_floor"] = round(theta, 4)
    cfg["escalation"]["tau_intent_confidence"] = tau
    cfg["escalation"]["theta_evidence_score"] = round(theta, 4)
    cfg["escalation"]["m_min_evidence_items"] = m
    (ROOT / "config.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"  thresholds from DEV: theta={theta:.3f} (25th pct of top-1 scores), "
          f"tau={tau}, m={m}")

    # ------------------------------------------------------- evaluation rows
    gold_path = ROOT / cfg["paths"]["gold_dir"] / "golden_set.csv"
    cand_path = ROOT / cfg["paths"]["gold_dir"] / "gold_candidates.csv"
    have_human = gold_path.exists()
    if have_human:
        ev = pd.read_csv(gold_path)
        ev = ev[ev["human_intent"].astype(str).str.len() > 0]
        print(f"HUMAN gold labels: {len(ev)}")
    else:
        ev = pd.read_csv(cand_path)
        print(f"NO human labels yet — running the system on {len(ev)} candidates; "
              f"intent/escalation metrics will be PENDING")
    if args.limit:
        ev = ev.head(args.limit)
    if "customer_created_at" not in ev.columns:
        ev = ev.merge(pd.read_csv(cand_path)[["customer_tweet_id", "customer_created_at",
                                              "conversation_id"]],
                      on="customer_tweet_id", how="left", suffixes=("", "_c"))
    # A label kept from before a split rebuild may no longer be in the pool. Its
    # timestamp merges as NaN, and a NaN query time disables retrieval's
    # "evidence strictly older than the query" filter -- a silent leakage path.
    # Drop such rows and say so rather than scoring them.
    stale = ev["customer_created_at"].isna()
    if stale.any():
        print(f"  ! dropping {int(stale.sum())} labelled row(s) with no timestamp in "
              f"the current gold pool (labelled against an older split); they would "
              f"bypass the retrieval time filter")
        ev = ev[~stale]
    eval_convs = set(ev["conversation_id"].dropna().astype(int))

    # --------------------------------------------------------------- systems
    allow = tax_mod.auto_allowed(tax)
    if not allow:
        print("  ! auto-handle allowlist is EMPTY: every intent in the taxonomy is")
        print("  ! marked account_specific, so automation coverage will be 0 by")
        print("  ! construction. That is a finding about this brand, not a bug.")
    else:
        print(f"  auto-handle allowlist: {allow}")

    ag = agent_mod.Agent(index=index, tax=tax, silver=silver, cfg=cfg, cache_dir=cache)
    b0 = baselines.B0Trivial(labels or [tax_mod.OTHER],
                             baselines.best_canned_reply(history))
    b1 = baselines.B1Simple(silver, index)

    runs: dict[str, list[dict]] = {"agent": [], "B0_trivial": [], "B1_simple": [],
                                   "B2_no_retrieval": [], "B3_random_evidence": []}
    print(f"\nrunning {len(ev)} examples through 5 systems...")
    for n, (_, r) in enumerate(ev.iterrows(), 1):
        msg = str(r["customer_text"])
        mid = int(r["customer_tweet_id"])
        t = r.get("customer_created_at")
        common = dict(message=msg, message_id=mid, created_at=t,
                      exclude_conversations=eval_convs)
        try:
            runs["agent"].append(ag.run(**common))
            if n <= args.ablation_n:
                runs["B2_no_retrieval"].append(ag.run(**common, use_retrieval=False))
                runs["B3_random_evidence"].append(ag.run(
                    **common, random_evidence_from=baselines.random_evidence(
                        index, ag.k, rng, before_time=t)))
        except Exception as exc:                               # noqa: BLE001
            print(f"\n  LLM call failed on row {n}: {type(exc).__name__}: {exc}")
            print("  stopping; rerun to resume from cache.")
            break
        out0 = b0.run(msg); out0["message_id"] = mid
        out1 = b1.run(msg, created_at=t, exclude_conversations=eval_convs)
        out1["message_id"] = mid
        runs["B0_trivial"].append(out0)
        runs["B1_simple"].append(out1)
        print(f"  {n}/{len(ev)}", end="\r", flush=True)
    print(" " * 30, end="\r")

    n_done = len(runs["agent"])
    ev = ev.head(n_done)
    for k, v in runs.items():
        pd.DataFrame(v).to_json(res / f"run_{k}.json", orient="records", indent=2)
    print(f"  completed {n_done} examples")

    # ----------------------------------------------------------------- judge
    judged: dict[str, list[dict]] = {}
    if not args.no_judge and n_done:
        jm = cfg["models"]["judge"]
        print(f"judging with {jm} (different model from generation)...")
        for name in ("agent", "B1_simple", "B0_trivial", "B3_random_evidence"):
            rows = []
            cap = (args.judge_n if name in ("agent", "B1_simple")
                   else min(args.ablation_n, args.judge_n // 2))
            for i, o in enumerate(runs[name][:cap]):
                try:
                    jr = judge.judge_reply(
                        message=str(ev.iloc[i]["customer_text"]),
                        reply=o.get("reply", ""), evidence=o.get("evidence", []),
                        intent=o.get("intent", ""), decision=o.get("decision", ""),
                        model=jm, cache_dir=cache)
                except Exception as exc:                        # noqa: BLE001
                    print(f"\n  judge failed: {type(exc).__name__}: {exc}")
                    jr = {"judge_failed": True, "judge_model": jm}
                jr["example_id"] = int(o["message_id"])
                rows.append(jr)
                print(f"  {name}: {i+1}/{len(runs[name])}", end="\r", flush=True)
            judged[name] = rows
            pd.DataFrame(rows).to_json(res / f"judge_{name}.json",
                                       orient="records", indent=2)
        print(" " * 40, end="\r")

    # --------------------------------------------------------------- metrics
    out = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "brand": cfg["brand"]["selected"],
        "models": {"generation": cfg["models"]["generation"],
                   "judge": cfg["models"]["judge"]},
        "n_evaluated": n_done,
        "thresholds": {"tau": tau, "theta": round(theta, 4), "m": m,
                       "source": "DEV statistics only; gold never used for tuning"},
        "llm_usage": llm.stats(),
        "api_limits": {
            "free_tier_rpm": int(__import__("os").environ.get("HIVER_RPM", llm.DEFAULT_RPM)),
            "note": ("Google Gemini free tier. The pipeline self-throttles to stay "
                     "under this rather than absorbing 429s. Sample sizes below are "
                     "set by this ceiling, not by what would be ideal."),
        },
        "evaluation_scope": {
            "n_agent_examples": n_done,
            "n_judged_per_system": {"agent": min(args.judge_n, n_done),
                                    "B1_simple": min(args.judge_n, n_done),
                                    "B0_trivial": min(args.ablation_n, args.judge_n // 2),
                                    "B3_random_evidence": min(args.ablation_n, args.judge_n // 2)},
            "n_ablation_examples": min(args.ablation_n, n_done),
            "n_silver_labels": args.silver_n,
            "claim_limitation": ("This is NOT a full-dataset LLM evaluation. Reply "
                                 "quality is judged on a subsample; intent and "
                                 "escalation metrics cover every human-labelled row."),
        },
        "MOCK_MODE": llm.mock_mode(),
        "retrieval": {
            "method": "bm25",
            "precision_at_5": "PENDING",
            "note": ("BM25 is the pre-registered default. The contingency that would "
                     "add dense retrieval requires a hand-judged Precision@5 on 50 DEV "
                     "queries, which has not been run. Retrieval quality is therefore "
                     "UNMEASURED and dense retrieval was correctly not added."),
        },
    }

    if have_human and n_done:
        gi = ev["human_intent"].astype(str).tolist()
        ge = ev["human_escalate"].astype(str).str.lower().isin(["true", "1", "yes"]).tolist()
        out["intent"] = {}
        out["escalation"] = {}
        for name, rows in runs.items():
            k = len(rows)
            if k == 0:
                continue
            pi = [r.get("intent", tax_mod.OTHER) for r in rows]
            pe = [r.get("decision") == "escalate" for r in rows]
            out["intent"][name] = metrics.intent_metrics(gi[:k], pi,
                                                         labels=tax_mod.names(tax))
            out["escalation"][name] = metrics.escalation_metrics(ge[:k], pe)
            if k < len(gi):
                out["intent"][name]["subsampled"] = (
                    f"ablation run on the first {k} of {len(gi)} rows "
                    f"(--ablation-n) — not comparable at full precision")
        conf = [r.get("intent_confidence", 0.0) for r in runs["agent"]]
        out["coverage_risk_curve"] = metrics.coverage_risk_curve(
            ge, conf, np.round(np.arange(0.0, 1.01, 0.05), 2).tolist())
        n_over = int(ev.get("changed_by_human", pd.Series(dtype=bool)).astype(str)
                     .str.lower().isin(["true", "1"]).sum()) if "changed_by_human" in ev else None
        out["human_label_provenance"] = {
            "n_human_labels": int(len(ev)),
            "n_overridden_llm_suggestion": n_over,
            "note": ("gold labels are the human_* columns typed in 23_label_cli.py, "
                     "where the LLM suggestion is hidden until after the human "
                     "commits. llm_suggested_* is kept in a separate column and is "
                     "never used as a label."),
        }
    else:
        out["intent"] = {"status": "PENDING",
                         "reason": "no human gold labels exist yet"}
        out["escalation"] = {"status": "PENDING",
                             "reason": "no human gold decisions exist yet"}

    if judged:
        out["reply_quality"] = {k: judge.aggregate(v) for k, v in judged.items()}
        out["reply_quality"]["placebo_note"] = (
            "B3_random_evidence receives randomly sampled evidence. If its "
            "groundedness score is close to the agent's, the judge is not "
            "measuring grounding and the agent's groundedness score means little.")

    hr = ROOT / cfg["paths"]["gold_dir"] / "human_reply_ratings.csv"
    human_ratings = []
    if hr.exists():
        human_ratings = pd.read_csv(hr).to_dict("records")
    out["judge_human_agreement"] = agreement.compare(
        human_ratings, judged.get("agent", []))

    (res / "metrics.json").write_text(json.dumps(out, indent=2, default=str),
                                      encoding="utf-8")
    print("\n" + "-" * 66)
    print(f"  wrote {res/'metrics.json'}")
    print(f"  intent metrics      : "
          f"{out['intent'].get('status', 'measured on ' + str(n_done) + ' human labels')}")
    print(f"  judge-human agreement: {out['judge_human_agreement'].get('status')}")
    print(f"  LLM usage           : {llm.stats()}")
    print("-" * 66)
    return 0


if __name__ == "__main__":
    sys.exit(main())
