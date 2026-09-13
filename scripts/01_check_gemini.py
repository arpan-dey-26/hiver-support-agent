#!/usr/bin/env python3
"""
Phase 0b — Gemini capability, quota and budget verification.

Does four things, in order, and stops early if one fails:

  1. LIST      live ListModels against the real key: what this project can call.
  2. SMOKE     one tiny structured-JSON call per candidate model (synthetic
               fixtures, NOT dataset examples) -> API access, structured-JSON
               compatibility, schema validity, latency, token usage.
               It does NOT measure model quality. Nothing here is evidence that
               a model is good enough for the task; that is decided on DEV in
               Phase 5, never here.
  3. PROBE     a small burst on the chosen model to observe real rate-limit
               behaviour and any 429s.  (--skip-probe to omit)
  4. BUDGET    projects total calls/tokens for the whole project and compares
               them against the limits YOU read off AI Studio.

No model ID is hard-coded. Candidates are derived from the live list and ranked
by a cost heuristic. The cheapest candidate whose API call SUCCEEDS and returns
valid structured JSON is proposed as a starting point -- a compatibility result,
not a quality verdict, and revisable once DEV measurements exist.

Usage:
    python scripts/01_check_gemini.py --rpm 15 --rpd 1000
    python scripts/01_check_gemini.py --list-only
    python scripts/01_check_gemini.py --candidates models/a,models/b

--rpm / --rpd are YOUR observed free-tier limits from
https://aistudio.google.com/rate-limit  — this script cannot discover them.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):      # Windows cp1252 console guard
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

# ---------------------------------------------------------------- synthetic fixtures
# Hand-written, generic, NOT from the Twitter dataset. They exist only to check
# that the API returns valid structured JSON, not to measure task quality.
SMOKE_LABELS = ["billing_problem", "delivery_delay", "login_issue",
                "cancellation_request", "other"]

SMOKE_CASES = [
    ("I was charged twice for my order this month, can you sort it out?", "billing_problem"),
    ("where is my parcel? it was meant to arrive three days ago", "delivery_delay"),
    ("cant sign in, it keeps saying my password is wrong", "login_issue"),
    ("please cancel my subscription, I don't want it any more", "cancellation_request"),
    ("your ad on the radio this morning was hilarious", "other"),
    ("hi", "other"),
]

SMOKE_PROMPT = (
    "Classify each customer support message into exactly one label.\n"
    f"Labels: {', '.join(SMOKE_LABELS)}\n\n"
    "Return ONLY JSON of the form:\n"
    '{"results": [{"id": 0, "label": "<label>", "confidence": 0.0}]}\n\n'
    "Messages:\n"
)

# model families we never want for text classification / generation
EXCLUDE_PAT = re.compile(
    r"(embedding|imagen|veo|tts|transcribe|image|vision-only|aqa|answer)", re.I
)
PREVIEW_PAT = re.compile(r"(preview|exp|experimental)", re.I)


def cheapness_rank(name: str) -> tuple:
    """Lower is cheaper. Heuristic over the LIVE model name — no fixed IDs.
    Prefers *-lite, then *-flash, then everything else; newer minor version
    first within a tier (newer flash-lite is usually cheaper than older flash)."""
    n = name.lower()
    if "lite" in n:
        tier = 0
    elif "flash" in n:
        tier = 1
    elif "pro" in n:
        tier = 3
    else:
        tier = 2
    nums = [float(x) for x in re.findall(r"(\d+\.\d+|\d+)", n)]
    version = -(max(nums) if nums else 0.0)  # newer first
    return (tier, version, len(name))


def load_dotenv(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def get_client():
    try:
        from google import genai  # type: ignore
    except ImportError:
        print("ERROR: google-genai not installed.  pip install google-genai",
              file=sys.stderr)
        raise SystemExit(2)
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        print("ERROR: no GEMINI_API_KEY / GOOGLE_API_KEY in environment or .env",
              file=sys.stderr)
        raise SystemExit(2)
    return genai.Client(api_key=key)


def supports_generate(model) -> bool:
    for attr in ("supported_actions", "supported_generation_methods", "actions"):
        vals = getattr(model, attr, None)
        if vals:
            return any("generatecontent" in str(v).lower().replace("_", "")
                       for v in vals)
    return True  # unknown -> let the smoke test decide


def list_models(client) -> list[dict]:
    out = []
    for m in client.models.list():
        name = getattr(m, "name", "") or ""
        out.append({
            "name": name,
            "display_name": getattr(m, "display_name", None),
            "input_token_limit": getattr(m, "input_token_limit", None),
            "output_token_limit": getattr(m, "output_token_limit", None),
            "supports_generate_content": supports_generate(m),
            "excluded_family": bool(EXCLUDE_PAT.search(name)),
            "preview": bool(PREVIEW_PAT.search(name)),
        })
    return out


def build_smoke_contents() -> str:
    lines = [f"{i}: {text}" for i, (text, _) in enumerate(SMOKE_CASES)]
    return SMOKE_PROMPT + "\n".join(lines)


def smoke_test(client, model_name: str, timeout_note: dict) -> dict:
    """One structured-JSON call. Records latency, tokens, JSON validity and how
    many synthetic fixtures came back with a valid in-vocabulary label."""
    from google.genai import types  # type: ignore

    result = {"model": model_name, "ok": False}
    cfg_kwargs = {"temperature": 0.0, "response_mime_type": "application/json"}
    t0 = time.perf_counter()
    try:
        resp = client.models.generate_content(
            model=model_name,
            contents=build_smoke_contents(),
            config=types.GenerateContentConfig(**cfg_kwargs),
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["latency_s"] = round(time.perf_counter() - t0, 2)
        timeout_note.setdefault("errors", []).append(result["error"])
        return result

    result["latency_s"] = round(time.perf_counter() - t0, 2)
    usage = getattr(resp, "usage_metadata", None)
    result["tokens"] = {
        "prompt": getattr(usage, "prompt_token_count", None),
        "output": getattr(usage, "candidates_token_count", None),
        "total": getattr(usage, "total_token_count", None),
    }
    text = (getattr(resp, "text", None) or "").strip()
    result["raw_len"] = len(text)
    try:
        parsed = json.loads(text)
        rows = parsed.get("results", parsed if isinstance(parsed, list) else [])
        labels = {int(r["id"]): str(r.get("label", "")) for r in rows
                  if isinstance(r, dict) and "id" in r}
        in_vocab = sum(1 for v in labels.values() if v in SMOKE_LABELS)
        matches = sum(1 for i, (_, gold) in enumerate(SMOKE_CASES)
                      if labels.get(i) == gold)
        result.update({
            "json_valid": True,
            "rows_returned": len(labels),
            "labels_in_vocabulary": in_vocab,
            "fixtures_total": len(SMOKE_CASES),
            # sanity signal only on synthetic fixtures — NOT a task metric
            "synthetic_fixture_matches": matches,
            "ok": len(labels) == len(SMOKE_CASES) and in_vocab == len(labels),
        })
    except Exception as exc:
        result["json_valid"] = False
        result["parse_error"] = f"{type(exc).__name__}: {exc}"
        result["raw_preview"] = text[:300]
    return result


def quota_probe(client, model_name: str, n: int) -> dict:
    """Small burst to observe real rate-limit behaviour. Deliberately tiny —
    this spends quota, so keep n low."""
    from google.genai import types  # type: ignore

    lat, errors, rate_limited = [], [], 0
    for i in range(n):
        t0 = time.perf_counter()
        try:
            client.models.generate_content(
                model=model_name,
                contents="Reply with the single word: ok",
                config=types.GenerateContentConfig(
                    temperature=0.0, max_output_tokens=5),
            )
            lat.append(time.perf_counter() - t0)
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            errors.append(msg)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg.upper() or "quota" in msg.lower():
                rate_limited += 1
    return {
        "calls_attempted": n,
        "calls_succeeded": len(lat),
        "rate_limited": rate_limited,
        "other_errors": len(errors) - rate_limited,
        "latency_s_mean": round(statistics.mean(lat), 2) if lat else None,
        "latency_s_max": round(max(lat), 2) if lat else None,
        "observed_throughput_rpm": (
            round(60.0 / statistics.mean(lat), 1) if lat else None),
        "error_samples": errors[:3],
        "note": ("observed_throughput_rpm is what THIS sequential loop achieved; "
                 "it is not the account's RPM limit. Read the real limit from "
                 "AI Studio and pass it as --rpm."),
    }


def budget(args, tokens_per_call: dict | None) -> dict:
    """Project the whole evaluation. Every count is an explicit, editable input
    so the assumptions are auditable rather than buried."""
    g = args.gold_n
    silver_calls = -(-args.dev_silver_n // max(args.silver_batch_size, 1))

    items = [
        ("gold pre-annotation (candidates)",       args.gold_candidates),
        ("silver labels for B1 (batched)",         silver_calls),
        ("agent: intent classification",           g),
        ("agent: grounded generation",             g),
        ("B2 zero-shot: classification",           g),
        ("B2 zero-shot: generation",               g),
        ("B3 random-evidence placebo: generation", g),
        ("judge: systems x gold",                  g * args.judge_systems),
        ("judge: agreement subset re-run",         args.agreement_n),
    ]
    subtotal = sum(n for _, n in items)
    retries = int(subtotal * args.retry_overhead)
    total = subtotal + retries

    est_tok = None
    if tokens_per_call and tokens_per_call.get("total"):
        per = tokens_per_call["total"]
        # generation + judge calls carry retrieved evidence -> assume heavier
        est_tok = int(total * per * args.token_multiplier)

    verdict, notes = "UNKNOWN", []
    if args.rpd:
        days = total / args.rpd
        notes.append(f"{total} calls / {args.rpd} RPD = {days:.2f} day(s) of quota")
        if days <= 0.5:
            verdict = "FEASIBLE"
        elif days <= 1.0:
            verdict = "TIGHT — fits one day with no room for mistakes"
        else:
            verdict = "NOT FEASIBLE AS SPECIFIED"
            notes.append("Reduce: judge_systems, dev_silver_n, or raise "
                         "silver_batch_size. Cached calls never re-spend quota.")
    else:
        notes.append("Pass --rpd (from AI Studio) to get a feasibility verdict.")
    if args.rpm:
        notes.append(f"at {args.rpm} RPM the wall-clock floor is "
                     f"~{total / args.rpm / 60:.1f} h of continuous calling")

    return {
        "line_items": [{"item": k, "calls": v} for k, v in items],
        "subtotal_calls": subtotal,
        "retry_overhead_calls": retries,
        "total_calls": total,
        "estimated_total_tokens": est_tok,
        "assumptions": {
            "gold_n": g, "gold_candidates": args.gold_candidates,
            "dev_silver_n": args.dev_silver_n,
            "silver_batch_size": args.silver_batch_size,
            "judge_systems": args.judge_systems,
            "agreement_n": args.agreement_n,
            "retry_overhead": args.retry_overhead,
            "token_multiplier": args.token_multiplier,
        },
        "observed_limits": {"rpm": args.rpm, "rpd": args.rpd},
        "verdict": verdict,
        "notes": notes,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("results/phase0"))
    ap.add_argument("--list-only", action="store_true")
    ap.add_argument("--candidates", type=str, default=None,
                    help="comma-separated model names to smoke-test instead of auto-ranking")
    ap.add_argument("--max-candidates", type=int, default=3)
    ap.add_argument("--include-preview", action="store_true")
    ap.add_argument("--skip-probe", action="store_true")
    ap.add_argument("--probe-n", type=int, default=8)
    # observed limits (from AI Studio — this script cannot discover them)
    ap.add_argument("--rpm", type=int, default=None)
    ap.add_argument("--rpd", type=int, default=None)
    # budget inputs
    ap.add_argument("--gold-n", type=int, default=150)
    ap.add_argument("--gold-candidates", type=int, default=200)
    ap.add_argument("--dev-silver-n", type=int, default=1500)
    ap.add_argument("--silver-batch-size", type=int, default=20)
    ap.add_argument("--judge-systems", type=int, default=3)
    ap.add_argument("--agreement-n", type=int, default=50)
    ap.add_argument("--retry-overhead", type=float, default=0.15)
    ap.add_argument("--token-multiplier", type=float, default=3.0,
                    help="evidence-bearing calls are heavier than the smoke call")
    args = ap.parse_args()

    load_dotenv()
    args.out.mkdir(parents=True, exist_ok=True)
    report = {"generated_at_utc": datetime.now(timezone.utc).isoformat()}

    print("=" * 68)
    print("PHASE 0b — GEMINI CAPABILITY, QUOTA AND BUDGET")
    print("=" * 68)

    client = get_client()

    # ---- 1. LIST
    try:
        models = list_models(client)
    except Exception as exc:
        print(f"ERROR listing models: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    report["models"] = models

    usable = [m for m in models
              if m["supports_generate_content"] and not m["excluded_family"]
              and (args.include_preview or not m["preview"])]
    usable.sort(key=lambda m: cheapness_rank(m["name"]))

    print(f"\n{len(models)} models visible to this key; "
          f"{len(usable)} usable for text generation.\n")
    print(f"  {'rank':<5}{'model':<44}{'in_tok':>9}{'out_tok':>9}")
    for i, m in enumerate(usable[:12]):
        print(f"  {i:<5}{m['name'][:43]:<44}"
              f"{str(m['input_token_limit'] or '-'):>9}"
              f"{str(m['output_token_limit'] or '-'):>9}")
    if not usable:
        print("  none — check the key's permissions.")
        return 2

    if args.list_only:
        (args.out / "gemini_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nWritten: {args.out / 'gemini_report.json'}")
        return 0

    # ---- 2. SMOKE
    if args.candidates:
        cand = [c.strip() for c in args.candidates.split(",") if c.strip()]
    else:
        cand = [m["name"] for m in usable[:args.max_candidates]]

    print(f"\n--- smoke test ({len(cand)} model(s), 1 call each, synthetic fixtures) ---")
    smoke, notes = [], {}
    for name in cand:
        r = smoke_test(client, name, notes)
        smoke.append(r)
        if r.get("ok"):
            print(f"  PASS  {name}  {r['latency_s']}s  "
                  f"tokens={r['tokens']['total']}  "
                  f"fixtures_matched={r['synthetic_fixture_matches']}/{r['fixtures_total']}")
        else:
            print(f"  FAIL  {name}  "
                  f"{r.get('error') or r.get('parse_error') or 'incomplete output'}")
    report["smoke_test"] = smoke
    report["smoke_test_note"] = (
        "WHAT THIS CHECKS: API access, structured-JSON compatibility, output "
        "schema validity, latency and token usage, on six synthetic generic "
        "fixtures. WHAT IT DOES NOT CHECK: model quality, task suitability, or "
        "classification accuracy. 'synthetic_fixture_matches' is a sanity "
        "signal only; it is NOT used to rank or select models, and must not be "
        "cited as evidence that a model is good enough. Model quality is "
        "measured on the DEV split in Phase 5.")

    passed = [r for r in smoke if r.get("ok")]
    if not passed:
        print("\nNo candidate passed the smoke test. Stopping before budgeting.")
        (args.out / "gemini_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8")
        return 1

    chosen = passed[0]  # candidates were already cheapness-ordered
    judge = next((r["model"] for r in passed[1:] if r["model"] != chosen["model"]), None)
    report["recommendation"] = {
        "starting_generation_model": chosen["model"],
        "starting_judge_model": judge,
        "basis": ("cheapest candidate whose call succeeded and returned valid "
                  "structured JSON — a COMPATIBILITY result, not a quality "
                  "verdict"),
        "status": "PROVISIONAL — revisit after DEV measurement in Phase 5",
        "not_established_here": [
            "that this model is accurate enough for intent classification",
            "that this model writes acceptable grounded replies",
            "that this model is a reliable judge",
        ],
        "warning": (None if judge else
                    "only one model passed — judge would share the generation "
                    "model, so self-preference bias cannot be reduced. Report this."),
    }
    print(f"\n  provisional generation model : {chosen['model']}")
    print(f"  provisional judge model      : {judge or 'NONE — see warning'}")
    print("  (compatibility only — quality is NOT assessed here; DEV decides)")

    # ---- 3. PROBE
    if not args.skip_probe:
        print(f"\n--- quota probe ({args.probe_n} small calls on "
              f"{chosen['model']}) ---")
        probe = quota_probe(client, chosen["model"], args.probe_n)
        report["quota_probe"] = probe
        print(f"  succeeded {probe['calls_succeeded']}/{probe['calls_attempted']}  "
              f"rate_limited={probe['rate_limited']}  "
              f"mean_latency={probe['latency_s_mean']}s")
        if probe["error_samples"]:
            for e in probe["error_samples"]:
                print(f"    ! {e[:140]}")

    # ---- 4. BUDGET
    b = budget(args, chosen.get("tokens"))
    report["budget"] = b
    print("\n--- projected budget for the full project ---")
    for li in b["line_items"]:
        print(f"  {li['item']:<42}{li['calls']:>8}")
    print(f"  {'subtotal':<42}{b['subtotal_calls']:>8}")
    print(f"  {'retries (+%d%%)' % int(args.retry_overhead * 100):<42}"
          f"{b['retry_overhead_calls']:>8}")
    print(f"  {'TOTAL CALLS':<42}{b['total_calls']:>8}")
    if b["estimated_total_tokens"]:
        print(f"  {'estimated tokens (rough)':<42}"
              f"{b['estimated_total_tokens']:>8,}")
    print(f"\n  VERDICT: {b['verdict']}")
    for n in b["notes"]:
        print(f"    - {n}")

    out_path = args.out / "gemini_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWritten: {out_path}")
    print("These model IDs are a PROVISIONAL starting point for config.yaml. "
          "They record what the API can do, not what works for the task.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
