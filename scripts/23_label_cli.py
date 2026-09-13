#!/usr/bin/env python3
"""
Phase 3b — the HUMAN labelling interface. Two keypresses per example.

Anti-anchoring: the LLM's suggestion is hidden until AFTER you have typed your
own intent. That costs a little speed and is the reason the gold labels can
honestly be called human labels.

Every answer is written to disk immediately, so quitting and resuming is safe.

    python scripts/23_label_cli.py            # label / resume
    python scripts/23_label_cli.py --round 2  # blind re-label for reliability
    python scripts/23_label_cli.py --status   # how many are done
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import taxonomy as tax_mod  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def getkey(valid: str) -> str:
    """Single keypress where the platform allows it; Enter-terminated otherwise."""
    try:
        import msvcrt                                        # Windows
        while True:
            ch = msvcrt.getch().decode("utf-8", "ignore").lower()
            if ch in valid:
                print(ch)
                return ch
            if ch in ("\x03",):
                raise KeyboardInterrupt
    except ImportError:
        pass
    try:
        import termios, tty                                  # POSIX
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            while True:
                ch = sys.stdin.read(1).lower()
                if ch in valid:
                    print(ch)
                    return ch
                if ch == "\x03":
                    raise KeyboardInterrupt
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
    except Exception:
        while True:
            ch = input().strip().lower()[:1]
            if ch in valid:
                return ch


def save_rows(rows: list, path: Path) -> "pd.DataFrame":
    """Write labels to disk after every single answer.

    Three things this guarantees:
      * every row is a plain dict before the DataFrame is built — mixing pandas
        Series (loaded from an earlier session) with dicts (typed this session)
        is what caused the 'dict' object has no attribute 'dtype' crash;
      * one row per customer_tweet_id, keeping the most recent answer, so a
        re-label overwrites rather than duplicating;
      * the write is atomic — a temp file is replaced into position, so an
        interruption mid-write cannot truncate labels already earned.
    """
    normalised = [dict(r) if not isinstance(r, dict) else r for r in rows]
    df = pd.DataFrame(normalised)
    if "customer_tweet_id" in df.columns:
        df = df.drop_duplicates(subset="customer_tweet_id", keep="last")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8")
    tmp.replace(path)
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    gold_dir = ROOT / cfg["paths"]["gold_dir"]
    cand_path = gold_dir / "gold_candidates.csv"
    out_path = gold_dir / ("golden_set.csv" if args.round == 1
                           else f"relabel_round{args.round}.csv")
    if not cand_path.exists():
        print("ERROR: run scripts/22_make_gold_candidates.py first", file=sys.stderr)
        return 2

    tax = tax_mod.load(ROOT / "data" / "processed" / "taxonomy_v1.json")
    intents = [i["name"] for i in tax["intents"]]
    keys = "".join(str(i + 1) for i in range(min(9, len(intents)))) + "0"

    cand = pd.read_csv(cand_path)
    done = {}
    if out_path.exists():
        # keep_default_na=False so an empty cell stays "" and never round-trips
        # through NaN into the literal string "nan" on the next save.
        prev = pd.read_csv(out_path, keep_default_na=False)
        # .to_dict() matters: iterrows() yields Series, and a list mixing Series
        # with the plain dicts appended below makes pandas raise
        # "'dict' object has no attribute 'dtype'" from nested_data_to_arrays.
        done = {int(r["customer_tweet_id"]): r.to_dict()
                for _, r in prev.iterrows()}

    todo = [i for i in cand.index
            if int(cand.at[i, "customer_tweet_id"]) not in done]
    if args.round == 2:
        rng = __import__("numpy").random.default_rng(cfg["project"]["seed"] + 1)
        n = int(cfg["golden_set"]["relabel_subset_n"])
        todo = [int(x) for x in rng.choice(cand.index, size=min(n, len(cand)),
                                           replace=False)
                if int(cand.at[int(x), "customer_tweet_id"]) not in done]

    print("=" * 70)
    print(f"GOLD LABELLING — round {args.round}")
    print("=" * 70)
    print(f"  {len(done)} already labelled, {len(todo)} remaining")
    if args.status:
        return 0
    if not todo:
        print("  nothing left to label.")
        return 0
    if args.limit:
        todo = todo[:args.limit]

    print("\nIntents:")
    print(tax_mod.render_for_human(tax))
    print("\n  Two keypresses per message:")
    print("    1) intent number   2) [a]uto-handle or [e]scalate")
    print("    [s] skip   [q] save and quit")
    print("\n  'escalate' = a human must handle it: needs account access, a refund,")
    print("  or involves legal / safety / financial-harm content.\n")
    input("  Press Enter to start...")

    rows = list(done.values())
    try:
        for n, i in enumerate(todo, 1):
            r = cand.loc[i]
            print("\n" + "=" * 70)
            print(f"[{n}/{len(todo)}]  stratum: {r['stratum']}")
            print("-" * 70)
            print(f"  {str(r['customer_text'])[:600]}")
            print("-" * 70)
            t0 = time.perf_counter()

            print(f"  intent [{keys}] (0=other, s=skip, q=quit): ", end="", flush=True)
            k = getkey(keys + "sq")
            if k == "q":
                break
            if k == "s":
                continue
            intent = tax_mod.OTHER if k == "0" else intents[int(k) - 1]

            # suggestion revealed only AFTER the human has committed
            sug = str(r.get("llm_suggested_intent", "") or "")
            if sug:
                mark = "same" if sug == intent else f"model said: {sug}"
                print(f"    you: {intent}    ({mark})")

            print("  [a]uto-handle or [e]scalate: ", end="", flush=True)
            k2 = getkey("aeq")
            if k2 == "q":
                break
            secs = round(time.perf_counter() - t0, 1)

            rows.append({
                "customer_tweet_id": int(r["customer_tweet_id"]),
                "conversation_id": int(r["conversation_id"]),
                "customer_text": r["customer_text"],
                "stratum": r["stratum"],
                "natural_weight": r.get("natural_weight", 1.0),
                "llm_suggested_intent": sug,
                "llm_suggested_escalate": r.get("llm_suggested_escalate", ""),
                "human_intent": intent,
                "human_escalate": (k2 == "e"),
                "changed_by_human": bool(sug) and (sug != intent),
                "label_timestamp": datetime.now(timezone.utc).isoformat(),
                "seconds_spent": secs,
                "round": args.round,
            })
            save_rows(rows, out_path)
    except KeyboardInterrupt:
        print("\n  interrupted — progress saved.")

    if rows:
        df = save_rows(rows, out_path)
        overrode = df["changed_by_human"].sum() if "changed_by_human" in df else 0
        with_sug = int(df["llm_suggested_intent"].astype(str).str.len().gt(0).sum())
        print("\n" + "-" * 70)
        print(f"  {len(df)} labelled -> {out_path}")
        print(f"  escalate: {int(df['human_escalate'].sum())} / {len(df)}")
        if with_sug:
            print(f"  you overrode the model on {int(overrode)}/{with_sug} "
                  f"({100*overrode/with_sug:.0f}%) — this is the evidence the labels are human")
        print(f"  median time per label: {df['seconds_spent'].median():.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
