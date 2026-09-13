"""Baselines. All run through the identical evaluation harness on the same rows.

B0  trivial       majority intent + one canned reply + both degenerate policies
B1  simple        distilled TF-IDF classifier + 1-NN historical reply, no LLM
B2  ablation      LLM with retrieval switched off
B3  placebo       LLM with RANDOM evidence — tests the JUDGE, not the system

B1's reply baseline is the sharp one: it returns what the brand actually said to
the most similar past message. If generation cannot beat that, the generation
layer is not earning its place.

None of these is weakened on purpose. B1 gets a real hyperparameter sweep on DEV
and B0's canned reply is the best single canned reply available, not a bad one.
"""

from __future__ import annotations

import re
from collections import Counter

import numpy as np

from . import escalate, taxonomy as tax_mod

ACCOUNT_WORDS = escalate.ACCOUNT_ACTION_RE
SENSITIVE_WORDS = escalate.SENSITIVE_RE


class B0Trivial:
    """Majority class, one fixed reply, and both degenerate escalation policies."""

    name = "B0_trivial"

    def __init__(self, train_labels, canned_reply: str):
        self.majority = Counter(train_labels).most_common(1)[0][0] if len(train_labels) else tax_mod.OTHER
        self.canned = canned_reply

    def run(self, message: str, **_) -> dict:
        return {"intent": self.majority, "intent_confidence": 1.0,
                "reply": self.canned, "decision": "auto_handle",
                "reason_codes": ["trivial_baseline"],
                "reason": "auto_handle — trivial baseline always auto-handles.",
                "evidence": [], "reply_type": "draft_resolution"}


class B1Simple:
    """No LLM at inference: distilled classifier + nearest historical reply."""

    name = "B1_simple"

    def __init__(self, silver_clf, index):
        self.clf = silver_clf
        self.index = index

    def run(self, message: str, created_at=None, exclude_conversations=None, **_) -> dict:
        intent = self.clf.predict([message])[0]
        conf = self.clf.predict_proba_max([message])[0]
        hits = self.index.search(message, query_time=created_at, k=1,
                                 exclude_conversations=exclude_conversations)
        reply = hits[0]["brand_reply_snippet"] if hits else ""
        text = message or ""
        if SENSITIVE_WORDS.search(text) or ACCOUNT_WORDS.search(text) or not hits:
            decision, codes = "escalate", (
                ["sensitive_topic"] if SENSITIVE_WORDS.search(text)
                else ["account_action_required"] if ACCOUNT_WORDS.search(text)
                else ["insufficient_evidence"])
            reason = f"escalate — {codes[0]}: keyword/rule baseline."
        else:
            decision, codes = "auto_handle", ["rule_baseline"]
            reason = "auto_handle — rule baseline found no escalation keyword."
        return {"intent": intent, "intent_confidence": round(float(conf), 3),
                "reply": reply, "decision": decision, "reason_codes": codes,
                "reason": reason, "evidence": hits,
                "reply_type": "draft_resolution"}


def random_evidence(index, k: int, rng: np.random.Generator, before_time=None) -> list[dict]:
    """Evidence sampled at random for the B3 placebo."""
    df = index.df
    if before_time is not None:
        import pandas as pd
        qt = pd.Timestamp(before_time)
        if qt.tzinfo is None:
            qt = qt.tz_localize("UTC")
        df = df[df["customer_created_at"] < qt]
    if df.empty:
        return []
    idx = rng.choice(len(df), size=min(k, len(df)), replace=False)
    out = []
    for rank, i in enumerate(idx, 1):
        r = df.iloc[int(i)]
        out.append({"rank": rank, "score": 0.0,
                    "customer_tweet_id": int(r["customer_tweet_id"]),
                    "conversation_id": int(r["conversation_id"]),
                    "customer_snippet": str(r["customer_text"])[:280],
                    "brand_reply_snippet": str(r["brand_reply_text"])[:400],
                    "is_deflection": bool(r.get("is_deflection", False)),
                    "reply_word_count": int(r.get("reply_word_count", 0))})
    return out


def best_canned_reply(history_df, n_candidates: int = 200) -> str:
    """The most 'central' short historical reply — the strongest single fixed
    reply available, so B0 is a fair floor rather than a straw man."""
    from .retrieval import tokenize
    cands = history_df[history_df["reply_word_count"].between(8, 30)]
    if cands.empty:
        cands = history_df
    cands = cands.head(n_candidates)
    toks = [set(tokenize(t)) for t in cands["brand_reply_text"]]
    if not toks:
        return "Thanks for reaching out — we'll take a look."
    df_counter = Counter()
    for s in toks:
        df_counter.update(s)
    scores = [sum(df_counter[t] for t in s) / (len(s) or 1) for s in toks]
    return str(cands.iloc[int(np.argmax(scores))]["brand_reply_text"])[:280]
