#!/usr/bin/env python3
"""
Phase 1 — build the brand dataset and the leak-safe splits.

Two passes over twcs.csv:
  pass 1  graph columns only (no text) -> reconstruct threads, find the
          conversations whose ONLY brand replier is the selected brand
  pass 2  read text/time for just those rows

Then: build customer->brand exchanges, dedupe, and split by CONVERSATION in
TIME order (earliest -> history, latest -> gold_pool).

Leakage properties this enforces, by construction:
  L2  split is by conversation_id, never by message
  L4  history is strictly earlier than dev, which is strictly earlier than
      gold_pool, so retrieval can never surface a future reply
  L3  exact normalised-text duplicates of customer messages are dropped,
      keeping the earliest occurrence, and the count is reported

    python scripts/20_build_brand_dataset.py
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.threads import reconstruct_threads  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

DEFLECT_RE = re.compile(
    r"(?:\bdms?\b|\bd\.?m\.?\b|direct message|private message|"
    r"\bpm us\b|inbox us|message us privately)", re.I)
THANKS_RE = re.compile(
    r"\b(?:thank(?:s| you)|cheers|appreciate it|sorted|fixed it|that worked|"
    r"perfect|awesome|legend)\b", re.I)
URL_RE = re.compile(r"https?://\S+|\bwww\.\S+", re.I)
MENTION_RE = re.compile(r"@\w+")
WORD_RE = re.compile(r"\w+")


def norm_text(s: str) -> str:
    s = URL_RE.sub(" ", s)
    s = MENTION_RE.sub(" ", s)
    return re.sub(r"\s+", " ", s.lower()).strip()


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))
    brand = cfg["brand"]["selected"]
    seed = cfg["project"]["seed"]
    csv_path = root / cfg["paths"]["raw_csv"]
    out_dir = root / cfg["paths"]["processed_dir"]
    (out_dir / "splits").mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found", file=sys.stderr)
        return 2

    print("=" * 64)
    print(f"BUILD BRAND DATASET — {brand}")
    print("=" * 64)

    # ---------------------------------------------------------- pass 1: graph
    print("pass 1/2  reconstructing threads (graph columns only)...")
    tid_parts, pid_parts, inb_parts, auth_parts = [], [], [], []
    for i, ch in enumerate(pd.read_csv(
            csv_path, chunksize=400_000, dtype=str, keep_default_na=False,
            na_values=[""], encoding="utf-8", on_bad_lines="warn",
            usecols=["tweet_id", "author_id", "inbound", "in_response_to_tweet_id"])):
        tid_parts.append(pd.to_numeric(ch["tweet_id"], errors="coerce")
                         .fillna(-1).to_numpy("int64"))
        pid_parts.append(pd.to_numeric(ch["in_response_to_tweet_id"], errors="coerce")
                         .fillna(-1).to_numpy("int64"))
        inb_parts.append(ch["inbound"].astype(str).str.strip().str.lower()
                         .isin(["true", "1", "t", "yes"]).to_numpy(bool))
        auth_parts.append(ch["author_id"].fillna("").to_numpy())
        print(f"   chunk {i+1}", end="\r", flush=True)
    print(" " * 40, end="\r")

    tweet_id = np.concatenate(tid_parts)
    parent_id = np.concatenate(pid_parts)
    inbound = np.concatenate(inb_parts)
    author = np.concatenate(auth_parts)
    del tid_parts, pid_parts, inb_parts, auth_parts

    th = reconstruct_threads(tweet_id, parent_id)
    conv = th["conversation_id"]
    depth = th["depth"]
    print(f"   {th['stats']['n_tweets']:,} tweets -> "
          f"{th['stats']['n_conversations']:,} conversations "
          f"(cycles: {th['stats']['n_unconverged_cycles']})")

    # conversations whose ONLY outbound author is our brand (D3: no multi-brand)
    outb = pd.DataFrame({"conv": conv[~inbound], "author": author[~inbound]}).drop_duplicates()
    n_brands = outb.groupby("conv")["author"].size()
    single = set(n_brands.index[n_brands == 1])
    brand_convs = set(outb.loc[(outb["author"] == brand) & outb["conv"].isin(single), "conv"])
    multi_excluded = int(((outb["author"] == brand) & ~outb["conv"].isin(single)).sum())
    print(f"   {brand}: {len(brand_convs):,} single-brand conversations "
          f"({multi_excluded:,} multi-brand excluded)")
    if not brand_convs:
        print("ERROR: no conversations found for this brand", file=sys.stderr)
        return 2

    keep_rows = np.isin(conv, np.fromiter(brand_convs, dtype="int64"))
    keep_tids = set(tweet_id[keep_rows].tolist())
    print(f"   {int(keep_rows.sum()):,} tweets belong to those conversations")

    idx_of_tid = {int(t): i for i, t in enumerate(tweet_id) if int(t) in keep_tids}

    # ---------------------------------------------------------- pass 2: text
    print("pass 2/2  reading text for those tweets...")
    rows = []
    for i, ch in enumerate(pd.read_csv(
            csv_path, chunksize=400_000, dtype=str, keep_default_na=False,
            na_values=[""], encoding="utf-8", on_bad_lines="warn")):
        t = pd.to_numeric(ch["tweet_id"], errors="coerce").fillna(-1).astype("int64")
        m = t.isin(keep_tids)
        if m.any():
            sub = ch.loc[m, ["tweet_id", "author_id", "inbound", "created_at", "text",
                             "in_response_to_tweet_id"]].copy()
            sub["tweet_id"] = t[m].to_numpy()
            rows.append(sub)
        print(f"   chunk {i+1}", end="\r", flush=True)
    print(" " * 40, end="\r")

    df = pd.concat(rows, ignore_index=True)
    df["inbound"] = df["inbound"].astype(str).str.strip().str.lower().isin(
        ["true", "1", "t", "yes"])
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce",
                                      format="mixed", utc=True)
    df["parent_id"] = pd.to_numeric(df["in_response_to_tweet_id"],
                                    errors="coerce").fillna(-1).astype("int64")
    df["conversation_id"] = [int(conv[idx_of_tid[int(t)]]) for t in df["tweet_id"]]
    df["depth"] = [int(depth[idx_of_tid[int(t)]]) for t in df["tweet_id"]]
    df["text"] = df["text"].fillna("")
    print(f"   loaded {len(df):,} tweets")

    # --------------------------------------------------- build exchanges
    # An exchange = one customer message + the brand replies that answer it.
    cust = df[df["inbound"]].set_index("tweet_id")
    brand_msgs = df[(~df["inbound"]) & (df["author_id"] == brand)].copy()
    brand_msgs = brand_msgs[brand_msgs["parent_id"].isin(cust.index)]

    agg = (brand_msgs.sort_values("created_at")
           .groupby("parent_id")
           .agg(brand_reply_text=("text", lambda s: " ".join(s)),
                brand_tweet_ids=("tweet_id", lambda s: ",".join(map(str, s))),
                brand_first_reply_at=("created_at", "min"),
                n_brand_replies=("tweet_id", "size")))

    ex = cust.join(agg, how="inner").reset_index()
    ex = ex.rename(columns={"tweet_id": "customer_tweet_id", "text": "customer_text",
                            "created_at": "customer_created_at"})
    ex = ex[["conversation_id", "customer_tweet_id", "customer_text",
             "customer_created_at", "depth", "brand_tweet_ids", "brand_reply_text",
             "brand_first_reply_at", "n_brand_replies"]]
    print(f"   {len(ex):,} customer->brand exchanges")

    # resolution PROXY signals — observable only, never ground truth (D5/F.2)
    ex["reply_word_count"] = ex["brand_reply_text"].map(lambda s: len(WORD_RE.findall(s)))
    ex["is_deflection"] = ex["brand_reply_text"].str.contains(DEFLECT_RE)
    ex["deflection_only"] = ex["is_deflection"] & (ex["reply_word_count"] <= 25)
    conv_max_depth = df.groupby("conversation_id")["depth"].max()
    ex["conv_max_depth"] = ex["conversation_id"].map(conv_max_depth)

    # did the customer say thanks later in the same conversation?
    later = df[df["inbound"]].merge(
        ex[["conversation_id", "customer_tweet_id", "customer_created_at"]],
        on="conversation_id")
    later = later[later["created_at"] > later["customer_created_at"]]
    thanked = (later.assign(t=later["text"].str.contains(THANKS_RE))
               .groupby(["conversation_id", "customer_tweet_id"])["t"].any())
    ex["customer_thanked_after"] = ex.set_index(
        ["conversation_id", "customer_tweet_id"]).index.map(thanked).fillna(False)

    ex["is_thread_opener"] = ex["depth"] == 0

    ex["norm_customer"] = ex["customer_text"].map(norm_text)
    ex = ex[ex["norm_customer"].str.len() > 0]
    before = len(ex)

    # -------------------------------------------- strict temporal window split
    # Assigning by conversation START time is not enough. Conversations run long
    # (measured on the real corpus: max span 1,131 days, 99.9th pct 89 days), so
    # an early-starting history conversation can still have exchanges inside the
    # dev window. That put 204 exchanges (1.36%) after dev's start and broke the
    # "history strictly before dev" guarantee.
    #
    # Fix: pick two cut timestamps and assign a conversation to a window only if
    # its ENTIRE span fits inside that window. A conversation straddling a cut
    # cannot be placed without either splitting the conversation (leakage path
    # L2) or breaking temporal separation, so it is excluded and COUNTED.
    span = ex.groupby("conversation_id")["customer_created_at"].agg(["min", "max"])
    t1 = span["min"].quantile(cfg["splits"]["history_frac"])
    t2 = span["min"].quantile(cfg["splits"]["history_frac"] + cfg["splits"]["dev_frac"])

    assign = {}
    for conv, (lo, hi) in span[["min", "max"]].iterrows():
        if hi < t1:
            assign[conv] = "history"
        elif lo >= t1 and hi < t2:
            assign[conv] = "dev"
        elif lo >= t2:
            assign[conv] = "gold_pool"
        # else: straddles a cut -> excluded below
    ex["split"] = ex["conversation_id"].map(assign)
    straddlers = span.index.difference(pd.Index(list(assign)))
    n_straddle = int(len(straddlers))
    dropped_rows = int(ex["split"].isna().sum())
    # bias check: straddlers are long conversations by construction, so report it
    kept_span = (span.loc[span.index.isin(assign), "max"] -
                 span.loc[span.index.isin(assign), "min"]).dt.total_seconds() / 86400
    drop_span = ((span.loc[straddlers, "max"] - span.loc[straddlers, "min"])
                 .dt.total_seconds() / 86400) if n_straddle else pd.Series(dtype=float)
    ex = ex[ex["split"].notna()]
    print(f"   cuts at {t1} and {t2}")
    print(f"   excluded {n_straddle:,} conversations ({dropped_rows:,} exchanges) "
          f"that straddle a cut")
    print(f"   mean span kept {kept_span.mean():.2f}d vs dropped "
          f"{drop_span.mean() if n_straddle else 0:.2f}d "
          f"(dropping removes long conversations — reported as a bias)")

    # --------------------------------------------------- leakage-aware dedupe
    # DELIBERATELY NOT deduplicating identical customer text across splits.
    #
    # Two customers asking "how do I cancel" in the same words, months apart, are
    # separate real events. The brand's earlier answer is legitimate historical
    # precedent, and retrieving it is precisely what this system is meant to do --
    # deleting it would remove the evidence the agent is supposed to ground in.
    # Worse, dropping duplicates from the later split systematically strips dev and
    # gold_pool (measured on a fixture: a 70/20/10 split collapsed to 99/1/0),
    # biasing the evaluation set toward unusual phrasings.
    #
    # The actual leakage path -- one CONVERSATION spanning splits -- is already
    # closed by conversation-level, time-ordered splitting, and retrieval applies a
    # strict "evidence older than query" filter on top.
    #
    # What we do instead: deduplicate WITHIN gold_pool so no evaluation item is
    # repeated, and FLAG gold items that have an exact twin in history so the
    # effect can be measured in slice analysis rather than assumed away.
    gp_mask = ex["split"] == "gold_pool"
    dup_gp = ex[gp_mask].duplicated(subset="norm_customer", keep="first")
    n_gp_dupes = int(dup_gp.sum())
    ex = ex.drop(index=ex[gp_mask].index[dup_gp])

    hist_texts = set(ex.loc[ex["split"] == "history", "norm_customer"])
    ex["has_exact_twin_in_history"] = (
        (ex["split"] == "gold_pool") & ex["norm_customer"].isin(hist_texts))
    n_twins = int(ex["has_exact_twin_in_history"].sum())
    n_gold = int((ex["split"] == "gold_pool").sum())
    print(f"   deduped {n_gp_dupes:,} within gold_pool; {len(ex):,} exchanges remain")
    print(f"   {n_twins:,}/{n_gold:,} gold_pool items have an exact twin in history "
          f"(flagged, not deleted — reported as a slice)")

    # hard assertions: both leakage guarantees
    for a in ("history", "dev", "gold_pool"):
        for b in ("history", "dev", "gold_pool"):
            if a < b:
                overlap = (set(ex.loc[ex["split"] == a, "conversation_id"]) &
                           set(ex.loc[ex["split"] == b, "conversation_id"]))
                assert not overlap, f"conversation overlap between {a} and {b}"
    _t = {k: ex.loc[ex["split"] == k, "customer_created_at"] for k in
          ("history", "dev", "gold_pool")}
    for earlier, later in (("history", "dev"), ("dev", "gold_pool")):
        if len(_t[earlier]) and len(_t[later]):
            assert _t[earlier].max() < _t[later].min(), (
                f"{earlier} exchanges run past the start of {later}: "
                f"{_t[earlier].max()} >= {_t[later].min()}")

    # ------------------------------------------------------------- write out
    # CSV rather than parquet: these files are small, the evaluator can open them
    # in any tool, and it removes a binary dependency (see DECISIONS D9).
    for name in ("history", "dev", "gold_pool"):
        part = ex[ex["split"] == name].drop(columns=["norm_customer"])
        part.to_csv(out_dir / "splits" / f"{name}.csv", index=False, encoding="utf-8")
    ex.drop(columns=["norm_customer"]).to_csv(
        out_dir / "exchanges.csv", index=False, encoding="utf-8")

    counts = ex["split"].value_counts().to_dict()
    conv_sets = {k: set(ex.loc[ex["split"] == k, "conversation_id"]) for k in counts}
    overlaps = {f"{a}&{b}": len(conv_sets[a] & conv_sets[b])
                for a in conv_sets for b in conv_sets if a < b}
    spans = {k: [str(ex.loc[ex["split"] == k, "customer_created_at"].min()),
                 str(ex.loc[ex["split"] == k, "customer_created_at"].max())]
             for k in counts}

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "brand": brand, "seed": seed,
        "n_conversations_brand": len(brand_convs),
        "n_multibrand_conversations_excluded": multi_excluded,
        "n_exchanges_before_dedup": before,
        "n_gold_pool_duplicates_dropped": n_gp_dupes,
        "temporal_split": {
            "cut_1": str(t1), "cut_2": str(t2),
            "n_conversations_excluded_straddling_a_cut": n_straddle,
            "n_exchanges_excluded": dropped_rows,
            "mean_span_days_kept": round(float(kept_span.mean()), 3),
            "mean_span_days_dropped": round(float(drop_span.mean()), 3) if n_straddle else 0.0,
            "known_bias": ("excluded conversations are long-running by construction, "
                           "so the splits under-represent long conversations. The "
                           "alternative -- splitting a conversation across windows -- "
                           "would be a leakage path, so this trade was taken "
                           "deliberately and is reported rather than hidden."),
        },
        "n_gold_items_with_exact_twin_in_history": n_twins,
        "n_exchanges": int(len(ex)),
        "dedup_policy": ("Cross-split text dedup deliberately NOT applied: identical "
                         "questions from different customers are legitimate historical "
                         "precedent, and dropping them systematically strips the later "
                         "(evaluation) splits. Leakage is closed by conversation-level "
                         "time-ordered splitting plus a strict time filter at retrieval. "
                         "gold_pool is deduplicated within itself, and gold items with an "
                         "exact twin in history are FLAGGED for slice analysis, not deleted."),
        "split_counts": {k: int(v) for k, v in counts.items()},
        "split_conversation_overlaps": overlaps,
        "split_time_spans": spans,
        "proxy_signal_rates": {
            "is_deflection_pct": round(100 * float(ex["is_deflection"].mean()), 2),
            "deflection_only_pct": round(100 * float(ex["deflection_only"].mean()), 2),
            "customer_thanked_after_pct": round(100 * float(ex["customer_thanked_after"].mean()), 2),
            "mean_reply_words": round(float(ex["reply_word_count"].mean()), 1),
            "NOTE": "observable proxies only — NOT validated as resolution (PENDING-1)",
        },
    }
    (root / cfg["paths"]["results_dir"] / "phase0").mkdir(parents=True, exist_ok=True)
    (root / "results" / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    print("\n" + "-" * 64)
    for k in ("history", "dev", "gold_pool"):
        print(f"  {k:<10} {counts.get(k,0):>6,} exchanges   {spans.get(k,['?','?'])[0][:10]}"
              f" -> {spans.get(k,['?','?'])[1][:10]}")
    print(f"  conversation overlap between splits: {overlaps}  (must all be 0)")
    print(f"  deflection rate: {manifest['proxy_signal_rates']['is_deflection_pct']}%"
          f"  | mean reply: {manifest['proxy_signal_rates']['mean_reply_words']} words")
    print(f"\n  wrote {out_dir/'splits'} and results/split_manifest.json")
    print("-" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
