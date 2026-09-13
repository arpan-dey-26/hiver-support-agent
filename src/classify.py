"""Intent classification: an LLM classifier, and a distilled TF-IDF model.

The distilled model has two jobs. It is baseline B1 — testing whether an LLM is
needed at inference at all — and it is a free second opinion at runtime:
disagreement between the two is an uncertainty signal that costs no extra API
call, which matters on a free-tier quota.

Whether that signal actually predicts errors is measured on gold, not assumed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from . import llm, prompts, taxonomy as tax_mod


def llm_classify(message: str, tax: dict, brand: str, model: str,
                 cache_dir: Path, temperature: float = 0.0) -> dict:
    prompt = prompts.CLASSIFY_V1.format(
        brand=brand, taxonomy=tax_mod.render(tax), message=message[:1500])
    obj, raw = llm.generate_json(prompt, model, cache_dir,
                                 temperature=temperature, max_output_tokens=300)
    valid = set(tax_mod.names(tax))
    if not isinstance(obj, dict) or obj.get("intent") not in valid:
        return {"intent": tax_mod.OTHER, "confidence": 0.0, "runner_up": None,
                "multi_intent": False, "parse_failed": True}
    try:
        conf = float(obj.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    return {"intent": obj["intent"], "confidence": max(0.0, min(1.0, conf)),
            "runner_up": obj.get("runner_up"),
            "multi_intent": bool(obj.get("multi_intent", False)),
            "parse_failed": False}


def llm_classify_batch(messages: list[str], tax: dict, brand: str, model: str,
                       cache_dir: Path, batch_size: int = 10) -> list[dict]:
    """Batched path, used ONLY for silver labelling on DEV where volume matters.

    The evaluation run classifies one message per call so that examples cannot
    influence each other. Batching here is a quota trade-off, documented as such.
    """
    valid = set(tax_mod.names(tax))
    out: list[dict] = []
    for start in range(0, len(messages), batch_size):
        chunk = messages[start:start + batch_size]
        listing = "\n".join(f"{i}: {m[:400]}" for i, m in enumerate(chunk))
        prompt = prompts.CLASSIFY_BATCH_V1.format(
            brand=brand, taxonomy=tax_mod.render(tax), messages=listing)
        obj, _ = llm.generate_json(prompt, model, cache_dir,
                                   max_output_tokens=min(2048, 90 * len(chunk) + 200))
        got = {}
        if isinstance(obj, dict):
            for r in obj.get("results", []):
                if isinstance(r, dict) and "id" in r:
                    try:
                        got[int(r["id"])] = r
                    except (TypeError, ValueError):
                        continue
        for i in range(len(chunk)):
            r = got.get(i)
            if not r or r.get("intent") not in valid:
                out.append({"intent": tax_mod.OTHER, "confidence": 0.0,
                            "parse_failed": True})
            else:
                try:
                    c = float(r.get("confidence", 0.0))
                except (TypeError, ValueError):
                    c = 0.0
                out.append({"intent": r["intent"],
                            "confidence": max(0.0, min(1.0, c)),
                            "parse_failed": False})
        print(f"    silver labels {min(start+batch_size, len(messages))}/{len(messages)}",
              end="\r", flush=True)
    print(" " * 50, end="\r")
    return out


class SilverClassifier:
    """TF-IDF + logistic regression distilled from LLM labels on DEV."""

    def __init__(self):
        self.pipe = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2,
                                      sublinear_tf=True, strip_accents="unicode")),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                       C=4.0)),
        ])
        self.fitted = False

    def fit(self, texts, labels):
        if len(set(labels)) < 2:
            return self
        self.pipe.fit(list(texts), list(labels))
        self.fitted = True
        return self

    def predict(self, texts) -> list[str]:
        if not self.fitted:
            return [tax_mod.OTHER] * len(list(texts))
        return list(self.pipe.predict(list(texts)))

    def predict_proba_max(self, texts) -> list[float]:
        if not self.fitted:
            return [0.0] * len(list(texts))
        return [float(np.max(p)) for p in self.pipe.predict_proba(list(texts))]
