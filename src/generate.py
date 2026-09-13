"""Grounded reply generation.

Every factual claim must be attributable to a retrieved precedent. The model is
asked to enumerate its own claims and cite which precedent supports each; claims
citing nothing are counted as unsupported and escalate the message.

That self-report is a weak signal on its own, which is why the LLM judge scores
groundedness independently, and why a random-evidence placebo run (B3) tests
whether the judge is sensitive to grounding at all.
"""

from __future__ import annotations

from pathlib import Path

from . import llm, prompts
from .retrieval import format_evidence

HOLDING_TEMPLATE = (
    "Thanks for reaching out — I'm passing this to a member of our team who can "
    "look into it properly. {ask}")

HOLDING_ASKS = {
    "account_action_required": "They'll need to verify your account before making any changes.",
    "sensitive_topic": "Someone will follow up with you directly on this.",
    "unknown_intent": "Could you tell us a little more about what you need?",
    "low_quality_input": "Could you share a few more details so we can help?",
    "insufficient_evidence": "They'll be able to give you a definitive answer.",
    "conflicting_precedents": "They'll confirm the right answer for your situation.",
    "ungrounded_draft": "They'll be able to give you a definitive answer.",
    "low_confidence": "Could you tell us a little more about the issue?",
    "classifier_disagreement": "Could you tell us a little more about the issue?",
    "multi_intent": "It looks like there are a couple of things here — a colleague will pick both up.",
    "intent_not_auto_allowed": "They'll be able to help with this directly.",
}


def generate_reply(message: str, intent: str, hits: list[dict], brand: str,
                   model: str, cache_dir: Path, temperature: float = 0.0) -> dict:
    prompt = prompts.GENERATE_V1.format(
        brand=brand, message=message[:1500], intent=intent,
        evidence=format_evidence(hits))
    obj, raw = llm.generate_json(prompt, model, cache_dir,
                                 temperature=temperature, max_output_tokens=700)
    if not isinstance(obj, dict) or not str(obj.get("reply", "")).strip():
        return {"reply": "", "sufficient_evidence": False, "evidence_used": [],
                "claims": [], "unsupported_claims": 0, "parse_failed": True}
    claims = obj.get("claims") or []
    unsupported = sum(
        1 for c in claims
        if isinstance(c, dict) and not (c.get("supported_by") or []))
    return {
        "reply": str(obj["reply"]).strip(),
        "sufficient_evidence": bool(obj.get("sufficient_evidence", False)),
        "evidence_used": [e for e in (obj.get("evidence_used") or [])
                          if isinstance(e, int)],
        "claims": claims,
        "unsupported_claims": unsupported,
        "parse_failed": False,
    }


def analyze(message: str, hits: list[dict], tax: dict, brand: str,
            model: str, cache_dir: Path, temperature: float = 0.0) -> dict:
    """Classification AND drafting in ONE call.

    Split across two calls this costs 2 requests per message. On a 15 rpm free
    tier that doubles wall-clock for no analytical gain: the retrieval step does
    not depend on the intent, and the TF-IDF second opinion still provides the
    disagreement signal. Merging is a quota decision, recorded in DECISIONS D12.
    """
    from . import prompts as _p, taxonomy as _t
    prompt = _p.ANALYZE_V1.format(
        brand=brand, taxonomy=_t.render(tax), message=message[:1500],
        evidence=format_evidence(hits))
    obj, raw = llm.generate_json(prompt, model, cache_dir,
                                 temperature=temperature, max_output_tokens=900)
    valid = set(_t.names(tax))
    if not isinstance(obj, dict):
        return {"intent": _t.OTHER, "confidence": 0.0, "runner_up": None,
                "multi_intent": False, "reply": "", "sufficient_evidence": False,
                "evidence_used": [], "claims": [], "unsupported_claims": 0,
                "parse_failed": True}
    intent = obj.get("intent") if obj.get("intent") in valid else _t.OTHER
    try:
        conf = max(0.0, min(1.0, float(obj.get("confidence", 0.0))))
    except (TypeError, ValueError):
        conf = 0.0
    claims = obj.get("claims") or []
    unsupported = sum(1 for c in claims
                      if isinstance(c, dict) and not (c.get("supported_by") or []))
    return {
        "intent": intent, "confidence": conf, "runner_up": obj.get("runner_up"),
        "multi_intent": bool(obj.get("multi_intent", False)),
        "reply": str(obj.get("reply", "")).strip(),
        "sufficient_evidence": bool(obj.get("sufficient_evidence", False)),
        "evidence_used": [e for e in (obj.get("evidence_used") or [])
                          if isinstance(e, int)],
        "claims": claims, "unsupported_claims": unsupported,
        "parse_failed": not str(obj.get("reply", "")).strip(),
    }


def holding_reply(primary_reason_code: str) -> str:
    return HOLDING_TEMPLATE.format(
        ask=HOLDING_ASKS.get(primary_reason_code,
                             "They'll follow up with you shortly."))
