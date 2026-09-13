#!/usr/bin/env python3
"""
Regression test for the labelling CLI's save path.

The bug: rows loaded from an earlier session came back as pandas Series
(iterrows()), while rows typed this session were plain dicts. Building a
DataFrame from that mixed list raises

    AttributeError: 'dict' object has no attribute 'dtype'

from pandas' nested_data_to_arrays — and only after the FIRST NEW label, since
until then every element was a Series.

    python tests/test_label_cli_save.py
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "label_cli", ROOT / "scripts" / "23_label_cli.py")
cli = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cli)

ROW = {
    "customer_tweet_id": 111, "conversation_id": 9, "customer_text": "cant log in",
    "stratum": "core_proportional", "natural_weight": 0.1,
    "llm_suggested_intent": "account_access", "llm_suggested_escalate": True,
    "human_intent": "billing_or_charges", "human_escalate": True,
    "changed_by_human": True, "label_timestamp": "2026-09-13T00:00:00+00:00",
    "seconds_spent": 6.2, "round": 1,
}

passed, failed = 0, []


def check(name, cond, detail=""):
    global passed
    if cond:
        print(f"  PASS  {name}")
        passed += 1
    else:
        print(f"  FAIL  {name}  {detail}")
        failed.append(name)


def main() -> int:
    print("=" * 68)
    print("REGRESSION — labelling CLI save path")
    print("=" * 68)

    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "golden_set.csv"

        # --- 1. reproduce the original failure mode directly
        pd.DataFrame([ROW]).to_csv(out, index=False, encoding="utf-8")
        prev = pd.read_csv(out, keep_default_na=False)
        series_rows = [r for _, r in prev.iterrows()]          # the old code path
        new_dict = dict(ROW, customer_tweet_id=222, human_intent="playback_issue")
        mixed = series_rows + [new_dict]
        try:
            pd.DataFrame(mixed)
            reproduced = False
            err = "no exception raised"
        except Exception as exc:                                # noqa: BLE001
            reproduced = isinstance(exc, AttributeError) and "dtype" in str(exc)
            err = f"{type(exc).__name__}: {exc}"
        check("original bug reproduces on a mixed Series+dict list", reproduced, err)

        # --- 2. save_rows survives exactly that input
        try:
            df = cli.save_rows(mixed, out)
            ok, err = True, ""
        except Exception as exc:                                # noqa: BLE001
            df, ok, err = None, False, f"{type(exc).__name__}: {exc}"
        check("save_rows handles mixed Series+dict", ok, err)
        check("both rows written", ok and len(df) == 2, f"len={len(df) if ok else '-'}")

        # --- 3. the pre-existing human label is preserved byte-for-byte
        back = pd.read_csv(out, keep_default_na=False)
        orig = back[back["customer_tweet_id"] == 111].iloc[0]
        check("existing human_intent preserved",
              orig["human_intent"] == ROW["human_intent"],
              f"got {orig['human_intent']!r}")
        check("existing human_escalate preserved",
              str(orig["human_escalate"]).lower() == "true",
              f"got {orig['human_escalate']!r}")
        check("no NaN leaked in as the string 'nan'",
              not any(str(v) == "nan" for v in orig.values), str(list(orig.values)))

        # --- 4. re-labelling overwrites rather than duplicating
        again = cli.save_rows(mixed + [dict(new_dict, human_intent="device_support")],
                              out)
        check("re-label overwrites, no duplicate row", len(again) == 2, f"len={len(again)}")
        check("re-label keeps the latest answer",
              again[again["customer_tweet_id"] == 222].iloc[0]["human_intent"]
              == "device_support")

        # --- 5. resume: reload -> append -> save, the real crash sequence
        reloaded = pd.read_csv(out, keep_default_na=False)
        resumed = [r.to_dict() for _, r in reloaded.iterrows()]
        resumed.append(dict(ROW, customer_tweet_id=333, human_intent="app_bug"))
        try:
            df3 = cli.save_rows(resumed, out)
            ok3, err3 = True, ""
        except Exception as exc:                                # noqa: BLE001
            df3, ok3, err3 = None, False, f"{type(exc).__name__}: {exc}"
        check("resume-then-append saves cleanly", ok3, err3)
        check("all three labels present", ok3 and len(df3) == 3,
              f"len={len(df3) if ok3 else '-'}")

        # --- 6. atomic write leaves no stray temp file
        check("no .tmp left behind",
              not list(Path(td).glob("*.tmp")),
              str([p.name for p in Path(td).glob("*.tmp")]))

    print("-" * 68)
    print(f"{passed} passed" + (f", FAILED: {', '.join(failed)}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
