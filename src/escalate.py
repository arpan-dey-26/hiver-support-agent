"""Escalation policy — deterministic rules over signals produced upstream.

The LLM supplies signals. This module decides. That split is deliberate: the
decision is then auditable, unit-testable, and explainable line by line in an
interview, and it cannot drift with model temperature.

Design bias: trust over coverage. Auto-handling requires an intent to be on an
explicit ALLOWLIST — an unseen or newly added intent escalates by default rather
than being silently automated.

Pure functions, no I/O.
"""

from __future__ import annotations

import re

# Ordered: the first rule that fires becomes the headline reason.
SENSITIVE_RE = re.compile(
    r"\b(?:lawsuit|sue|suing|legal action|attorney|lawyer|court|subpoena|"
    r"fraud|fraudulent|unauthorou?ised|stolen|identity theft|hacked|"
    r"discriminat\w*|racist|harass\w*|threat\w*|"
    r"died|death|passed away|funeral|hospital|medical|disabilit\w*|"
    r"self.?harm|suicid\w*|kill myself|"
    r"data breach|gdpr|ombudsman|regulator|complaint to)\b", re.I)

ACCOUNT_ACTION_RE = re.compile(
    r"\b(?:refund|charge ?back|reimburse|credit my|cancel my (?:account|subscription|plan)|"
    r"close my account|delete my account|reset my password|change my (?:email|address|card)|"
    r"update my (?:card|billing|payment)|bill(?:ed|ing) me|double.?charged|"
    r"unsubscribe me|downgrade my|upgrade my)\b", re.I)

LOW_QUALITY_MIN_TOKENS = 4


def _is_low_quality(text: str) -> bool:
    stripped = re.sub(r"https?://\S+|@\w+|[^\w\s]", " ", str(text)).strip()
    return len(stripped.split()) < LOW_QUALITY_MIN_TOKENS


def evidence_conflict(hits: list[dict], threshold: float = 0.45) -> bool:
    """Rough disagreement check across retrieved brand replies: if the top
    replies share very little vocabulary, the brand has no consistent historical
    answer and a human should decide."""
    texts = [h.get("brand_reply_snippet", "") for h in hits[:3]]
    sets = [set(re.findall(r"[a-z']+", t.lower())) for t in texts if t]
    if len(sets) < 2:
        return False
    sims = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            u = sets[i] | sets[j]
            sims.append(len(sets[i] & sets[j]) / len(u) if u else 0.0)
    return (sum(sims) / len(sims)) < threshold * 0.35


def decide(*, customer_text: str, intent: str, intent_confidence: float,
           classifier_agreement: bool, evidence: list[dict],
           sufficient_evidence: bool, unsupported_claims: int,
           auto_allowed_intents, tau: float, theta: float, m: int,
           unknown_label: str = "other", multi_intent: bool = False) -> dict:
    """Return {decision, reason_codes, reason}."""
    codes: list[str] = []

    if SENSITIVE_RE.search(customer_text or ""):
        codes.append("sensitive_topic")
    if ACCOUNT_ACTION_RE.search(customer_text or ""):
        codes.append("account_action_required")
    if intent == unknown_label:
        codes.append("unknown_intent")
    if _is_low_quality(customer_text):
        codes.append("low_quality_input")
    if intent_confidence < tau:
        codes.append("low_confidence")
    if not classifier_agreement:
        codes.append("classifier_disagreement")

    strong = [h for h in evidence if h.get("score", 0.0) >= theta]
    if len(strong) < m:
        codes.append("insufficient_evidence")
    if evidence_conflict(evidence):
        codes.append("conflicting_precedents")
    if not sufficient_evidence or unsupported_claims > 0:
        codes.append("ungrounded_draft")
    if multi_intent:
        codes.append("multi_intent")
    if intent not in set(auto_allowed_intents) and intent != unknown_label:
        codes.append("intent_not_auto_allowed")

    if not codes:
        return {
            "decision": "auto_handle",
            "reason_codes": ["confident_intent", "strong_retrieval_support",
                             "no_account_action_required"],
            "reason": (f"auto_handle — confident_intent: '{intent}' at "
                       f"{intent_confidence:.2f} (>= {tau}), {len(strong)} historical "
                       f"precedents above similarity {theta}, and no account action "
                       f"or sensitive content detected."),
        }

    primary = codes[0]
    detail = {
        "sensitive_topic": "message contains legal, financial-harm, safety or medical language",
        "account_action_required": "request needs a change to the customer's account, which this system must never perform",
        "unknown_intent": "message does not fall into any intent the taxonomy covers",
        "low_quality_input": f"message has fewer than {LOW_QUALITY_MIN_TOKENS} content words, too little to act on",
        "low_confidence": f"intent confidence {intent_confidence:.2f} is below the {tau} threshold",
        "classifier_disagreement": "the LLM and the distilled TF-IDF classifier chose different intents",
        "insufficient_evidence": (f"only {len(strong)} of {len(evidence)} retrieved precedents "
                                  f"scored above {theta}; the brand has no consistent historical response"),
        "conflicting_precedents": "retrieved historical replies disagree with each other",
        "ungrounded_draft": "the draft could not be fully supported by the retrieved evidence",
        "multi_intent": "message contains more than one distinct request",
        "intent_not_auto_allowed": f"intent '{intent}' is not on the auto-handle allowlist",
    }[primary]
    return {
        "decision": "escalate",
        "reason_codes": codes,
        "reason": f"escalate — {primary}: {detail}. Needs a human.",
    }
