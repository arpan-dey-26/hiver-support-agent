"""LLM-as-judge for reply quality.

Bias controls actually applied:
  * judge model differs from the generation model (config.models.judge)
  * absolute scoring, not pairwise — removes position bias entirely
  * the judge never sees which system produced a reply, nor the gold labels
  * justification is written before the score, which reduces anchoring
  * temperature 0 and a cached response, so the same reply always scores the same

Not claimed: that these remove self-preference bias. Both models are Gemini, so
family bias remains, and the report says so.

The hallucination flag is reported separately and is never averaged into the
composite — folding a safety flag into a quality mean hides it.
"""

from __future__ import annotations

from pathlib import Path

from . import llm, prompts
from .retrieval import format_evidence

DIMS = ["relevance", "groundedness", "helpfulness", "tone_fit", "completeness"]
FLAGS = ["unsupported_claims", "escalation_appropriate"]


def judge_reply(*, message: str, reply: str, evidence: list[dict], intent: str,
                decision: str, model: str, cache_dir: Path) -> dict:
    prompt = prompts.JUDGE_V1.format(
        message=message[:1500], evidence=format_evidence(evidence),
        reply=(reply or "(empty reply)")[:1500], intent=intent, decision=decision)
    obj, raw = llm.generate_json(prompt, model, cache_dir, temperature=0.0,
                                 max_output_tokens=900)

    out = {"judge_failed": False, "judge_model": model}
    if not isinstance(obj, dict):
        out["judge_failed"] = True
        return out

    ok = True
    for d in DIMS:
        node = obj.get(d)
        try:
            s = int(round(float(node["score"] if isinstance(node, dict) else node)))
        except (TypeError, ValueError, KeyError):
            ok = False
            break
        if not 1 <= s <= 5:
            ok = False
            break
        out[d] = s
        out[f"{d}_why"] = (node.get("why", "") if isinstance(node, dict) else "")[:300]
    if not ok:
        return {"judge_failed": True, "judge_model": model}

    for f in FLAGS:
        node = obj.get(f)
        try:
            v = int(node["flag"] if isinstance(node, dict) else node)
        except (TypeError, ValueError, KeyError):
            v = 0
        out[f] = 1 if v == 1 else 0
        out[f"{f}_why"] = (node.get("why", "") if isinstance(node, dict) else "")[:300]

    # composite deliberately EXCLUDES unsupported_claims
    out["composite"] = round(sum(out[d] for d in DIMS) / len(DIMS), 3)
    return out


def aggregate(rows: list[dict]) -> dict:
    """Per-dimension means with the judge-failure rate reported, never hidden."""
    total = len(rows)
    ok = [r for r in rows if not r.get("judge_failed")]
    if not ok:
        return {"n": total, "n_scored": 0,
                "judge_failure_rate": 1.0 if total else 0.0,
                "note": "every judge call failed to parse; no scores available"}
    agg = {"n": total, "n_scored": len(ok),
           "judge_failure_rate": round((total - len(ok)) / total, 4) if total else 0.0,
           "excluded_failed_rows": total - len(ok)}
    for d in DIMS:
        vals = [r[d] for r in ok if d in r]
        agg[d] = round(sum(vals) / len(vals), 3) if vals else None
    agg["composite"] = round(sum(r["composite"] for r in ok) / len(ok), 3)
    for f in FLAGS:
        vals = [r[f] for r in ok if f in r]
        agg[f"{f}_rate"] = round(sum(vals) / len(vals), 4) if vals else None
    agg["note"] = ("composite is the mean of the five 1-5 dimensions and EXCLUDES "
                   "unsupported_claims, which is reported separately as a safety flag")
    return agg
