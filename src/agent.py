"""The support agent: classify -> retrieve -> ground -> decide.

Ordering is deliberate. Retrieval happens after classification so evidence can be
filtered by intent; the escalation decision happens last so it can use every
upstream signal, including whether a grounded reply could actually be produced.
"""

from __future__ import annotations

import time
from pathlib import Path

from . import classify, escalate, generate, taxonomy as tax_mod


class Agent:
    def __init__(self, *, index, tax: dict, silver, cfg: dict, cache_dir: Path):
        self.index = index
        self.tax = tax
        self.silver = silver
        self.cfg = cfg
        self.cache_dir = Path(cache_dir)
        self.brand = cfg["brand"]["selected"]
        self.gen_model = cfg["models"]["generation"]
        self.auto_allowed = tax_mod.auto_allowed(tax)
        r = cfg["retrieval"]
        self.k = int(r.get("k") or 5)
        self.theta = float(r.get("score_floor") or 0.0)
        e = cfg["escalation"]
        self.tau = float(e.get("tau_intent_confidence") or 0.0)
        self.m = int(e.get("m_min_evidence_items") or 1)

    def run(self, *, message: str, message_id, created_at=None,
            exclude_conversations=None, use_retrieval: bool = True,
            random_evidence_from=None) -> dict:
        t0 = time.perf_counter()

        # Retrieval first: BM25 needs no intent, so one combined LLM call can do
        # classification and drafting together (DECISIONS D12, quota).
        if random_evidence_from is not None:            # B3 placebo
            hits = random_evidence_from
        elif use_retrieval:
            hits = self.index.search(message, query_time=created_at, k=self.k,
                                     score_floor=self.theta,
                                     exclude_conversations=exclude_conversations)
        else:                                            # B2 no-retrieval ablation
            hits = []

        g = generate.analyze(message, hits, self.tax, self.brand,
                             self.gen_model, self.cache_dir)
        c = g
        silver_label = self.silver.predict([message])[0] if self.silver.fitted else None
        agreement = (silver_label is None) or (silver_label == c["intent"])

        d = escalate.decide(
            customer_text=message, intent=c["intent"],
            intent_confidence=c["confidence"], classifier_agreement=agreement,
            evidence=hits, sufficient_evidence=g["sufficient_evidence"],
            unsupported_claims=g["unsupported_claims"],
            auto_allowed_intents=self.auto_allowed, tau=self.tau,
            theta=self.theta, m=self.m, unknown_label=tax_mod.OTHER,
            multi_intent=c.get("multi_intent", False))

        if d["decision"] == "escalate":
            reply = generate.holding_reply(d["reason_codes"][0])
            reply_type = "holding_ack"
        else:
            reply = g["reply"]
            reply_type = "draft_resolution"

        return {
            "message_id": message_id,
            "intent": c["intent"],
            "intent_confidence": round(c["confidence"], 3),
            "intent_runner_up": c.get("runner_up"),
            "silver_intent": silver_label,
            "classifier_agreement": bool(agreement),
            "decision": d["decision"],
            "reason_codes": d["reason_codes"],
            "reason": d["reason"],
            "reply": reply,
            "reply_type": reply_type,
            "draft_reply": g["reply"],
            "evidence": hits,
            "grounding_check": {"claims": len(g["claims"]),
                                "unsupported": g["unsupported_claims"]},
            "sufficient_evidence": g["sufficient_evidence"],
            "parse_failures": {"analyze": g.get("parse_failed", False)},
            "policy_version": self.cfg["escalation"]["policy_version"],
            "model": self.gen_model,
            "latency_ms": int((time.perf_counter() - t0) * 1000),
        }
