"""BM25 retrieval over the brand's historical customer->reply exchanges.

BM25 is implemented here rather than pulled from a library: it is ~40 lines, it
removes a dependency, and every term contribution can be shown when explaining a
retrieval result.

    score(q, d) = sum over terms t in q of
                  idf(t) * f(t,d) * (k1+1) / (f(t,d) + k1*(1 - b + b*|d|/avgdl))
    idf(t)      = ln(1 + (N - n(t) + 0.5) / (n(t) + 0.5))

Two filters are applied at query time and are the leakage guarantee (K/L4):
  * evidence must come from the HISTORY split only
  * evidence must be strictly OLDER than the query message
"""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np
import pandas as pd

TOKEN_RE = re.compile(r"[a-z0-9']+")
STOP = set("""a an and are as at be but by for from has have i in is it its of on or
that the this to was were will with you your my me we our not no do does did can
could would should have had im ive dont cant just get got so if what when how why
""".split())


def tokenize(text: str) -> list[str]:
    text = re.sub(r"https?://\S+|@\w+", " ", str(text).lower())
    return [t for t in TOKEN_RE.findall(text) if t not in STOP and len(t) > 1]


class BM25:
    def __init__(self, corpus_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.N = len(corpus_tokens)
        self.doc_len = np.array([len(d) for d in corpus_tokens], dtype="float64")
        self.avgdl = float(self.doc_len.mean()) if self.N else 0.0
        self.tf: list[Counter] = [Counter(d) for d in corpus_tokens]
        df = Counter()
        for d in corpus_tokens:
            df.update(set(d))
        self.idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5))
                    for t, n in df.items()}
        # inverted index so scoring touches only documents containing a query term
        self.postings: dict[str, list[int]] = {}
        for i, d in enumerate(corpus_tokens):
            for t in set(d):
                self.postings.setdefault(t, []).append(i)

    def scores(self, query_tokens: list[str]) -> np.ndarray:
        out = np.zeros(self.N, dtype="float64")
        for t in set(query_tokens):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i in self.postings[t]:
                f = self.tf[i][t]
                denom = f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / (self.avgdl or 1))
                out[i] += idf * f * (self.k1 + 1) / denom
        return out


class HistoryIndex:
    """Index of historical exchanges, with the time and split filters baked in."""

    def __init__(self, history: pd.DataFrame):
        self.df = history.reset_index(drop=True).copy()
        self.df["customer_created_at"] = pd.to_datetime(
            self.df["customer_created_at"], errors="coerce", utc=True)
        self.tokens = [tokenize(t) for t in self.df["customer_text"]]
        self.bm25 = BM25(self.tokens)
        # Compare times as integer nanoseconds. Mixing tz-aware pandas
        # Timestamps with numpy datetime64 raises; integers cannot.
        ts = self.df["customer_created_at"]
        self.times_ns = ts.astype("int64").to_numpy()
        self.time_known = ts.notna().to_numpy()

    def search(self, query_text: str, query_time=None, k: int = 5,
               score_floor: float = 0.0, exclude_conversations=None) -> list[dict]:
        q = tokenize(query_text)
        if not q:
            return []
        s = self.bm25.scores(q)

        if query_time is not None and not pd.isna(query_time):
            qt = pd.Timestamp(query_time)
            qt = qt.tz_localize("UTC") if qt.tzinfo is None else qt.tz_convert("UTC")
            # strictly older, and an unparseable timestamp is never eligible
            eligible = self.time_known & (self.times_ns < qt.value)
            s = np.where(eligible, s, -1.0)
        if exclude_conversations:
            bad = self.df["conversation_id"].isin(exclude_conversations).to_numpy()
            s = np.where(bad, -1.0, s)

        order = np.argsort(-s)[: max(k * 4, k)]
        hits = []
        for i in order:
            if s[i] <= score_floor:
                continue
            r = self.df.iloc[i]
            hits.append({
                "rank": len(hits) + 1,
                "score": round(float(s[i]), 4),
                "customer_tweet_id": int(r["customer_tweet_id"]),
                "conversation_id": int(r["conversation_id"]),
                "customer_snippet": str(r["customer_text"])[:280],
                "brand_reply_snippet": str(r["brand_reply_text"])[:400],
                "is_deflection": bool(r.get("is_deflection", False)),
                "reply_word_count": int(r.get("reply_word_count", 0)),
            })
            if len(hits) >= k:
                break
        return hits


def format_evidence(hits: list[dict]) -> str:
    if not hits:
        return "(no historical precedent found)"
    return "\n".join(
        f"[E{h['rank']}] (similarity {h['score']})\n"
        f"  Customer: {h['customer_snippet']}\n"
        f"  {'Brand'} replied: {h['brand_reply_snippet']}"
        for h in hits)
