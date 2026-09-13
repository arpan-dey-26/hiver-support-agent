#!/usr/bin/env python3
"""
Human reply-quality ratings — the input to judge-human agreement.

The assignment asks for "evidence showing how well the judge agrees with a
human". That evidence cannot exist without a human rating the same replies on
the same rubric, which is what this does.

Blind by construction: the judge's scores for these replies are never shown, and
this script does not read judge_*.json at all. Rate first, compare afterwards.

Six keypresses per reply (five 1-5 dimensions, then one y/n flag). Saves after
every reply, resumable.

    python scripts/26_rate_replies.py            # rate / resume
    python scripts/26_rate_replies.py --n 30     # how many to rate (default 30)
    python scripts/26_rate_replies.py --status
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

DIMS = [
    ("relevance", "does it address what the customer actually asked?"),
    ("groundedness", "is every claim traceable to the evidence shown?"),
    ("helpfulness", "does the customer know what happens next?"),
    ("tone_fit", "does it match the register of the brand's real replies?"),
    ("completeness", "is the main ask fully addressed?"),
]


def getkey(valid: str) -> str:
    try:
        import msvcrt
        while True:
            ch = msvcrt.getch().decode("utf-8", "ignore").lower()
            if ch in valid:
                print(ch)
                return ch
            if ch == "\x03":
                raise KeyboardInterrupt
    except ImportError:
        pass
    try:
        import termios, tty
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


def save_rows(rows: list, path: Path) -> pd.DataFrame:
    """Same normalise/dedupe/atomic-write discipline as the label CLI."""
    df = pd.DataFrame([dict(r) if not isinstance(r, dict) else r for r in rows])
    if "example_id" in df.columns:
        df = df.drop_duplicates(subset="example_id", keep="last")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8")
    tmp.replace(path)
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    res = ROOT / cfg["paths"]["results_dir"]
    out_path = ROOT / cfg["paths"]["gold_dir"] / "human_reply_ratings.csv"
    run_path = res / "run_agent.json"
    if not run_path.exists():
        print("ERROR: run scripts/24_run_evaluation.py first", file=sys.stderr)
        return 2

    runs = json.loads(run_path.read_text(encoding="utf-8"))
    gold = pd.read_csv(ROOT / cfg["paths"]["gold_dir"] / "golden_set.csv",
                       keep_default_na=False)
    text_by_id = dict(zip(gold["customer_tweet_id"].astype(int),
                          gold["customer_text"].astype(str)))

    done = {}
    if out_path.exists():
        prev = pd.read_csv(out_path, keep_default_na=False)
        done = {int(r["example_id"]): r.to_dict() for _, r in prev.iterrows()}

    # rate the same replies the judge scored: the first judge-n agent rows
    judged_n = int(cfg.get("judge", {}).get("agreement_study_n", 50))
    pool = [o for o in runs if int(o["message_id"]) not in done][:max(args.n, 0)]

    print("=" * 70)
    print("HUMAN REPLY RATING  (blind — the judge's scores are not shown)")
    print("=" * 70)
    print(f"  {len(done)} already rated, {len(pool)} to go")
    if args.status:
        return 0
    if not pool:
        print("  nothing left to rate.")
        return 0

    print("\n  For each reply score five dimensions 1-5, then one y/n flag.")
    print("  1 = bad, 5 = excellent.  [q] saves and quits.\n")
    input("  Press Enter to start...")

    rows = list(done.values())
    try:
        for n, o in enumerate(pool, 1):
            mid = int(o["message_id"])
            ev = o.get("evidence") or []
            print("\n" + "=" * 70)
            print(f"[{n}/{len(pool)}]  tweet_id {mid}")
            print("-" * 70)
            print(f"CUSTOMER: {text_by_id.get(mid, '(text unavailable)')[:400]}")
            print("-" * 70)
            if ev:
                for h in ev[:3]:
                    print(f"EVIDENCE [{h['rank']}] brand replied: "
                          f"{str(h['brand_reply_snippet'])[:180]}")
            else:
                print("EVIDENCE: (none retrieved)")
            print("-" * 70)
            print(f"DRAFT REPLY: {str(o.get('reply', ''))[:400]}")
            print(f"  (system decided: {o.get('decision')})")
            print("-" * 70)

            t0 = time.perf_counter()
            rec = {"example_id": mid}
            quit_now = False
            for dim, hint in DIMS:
                print(f"  {dim:<14} 1-5  ({hint}): ", end="", flush=True)
                k = getkey("12345q")
                if k == "q":
                    quit_now = True
                    break
                rec[dim] = int(k)
            if quit_now:
                break
            print("  unsupported claim? (fact not in the evidence) [y/n]: ",
                  end="", flush=True)
            k = getkey("ynq")
            if k == "q":
                break
            rec["unsupported_claims"] = 1 if k == "y" else 0
            rec["seconds_spent"] = round(time.perf_counter() - t0, 1)
            rec["rated_at"] = datetime.now(timezone.utc).isoformat()
            rows.append(rec)
            save_rows(rows, out_path)
    except KeyboardInterrupt:
        print("\n  interrupted — progress saved.")

    if rows:
        df = save_rows(rows, out_path)
        print("\n" + "-" * 70)
        print(f"  {len(df)} replies rated -> {out_path}")
        print(f"  median time per reply: {df['seconds_spent'].median():.1f}s")
        if len(df) >= 8:
            print("\n  Now re-run to compute agreement (cached, no API quota):")
            print("    python scripts/24_run_evaluation.py")
            print("    python scripts/25_make_report.py")
        else:
            print(f"\n  {8 - len(df)} more needed before agreement is computable.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
