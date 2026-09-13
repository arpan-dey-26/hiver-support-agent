#!/usr/bin/env python3
"""
Phase 3a — sample the golden-set candidates and pre-annotate them.

Sampling is stratified by UNSUPERVISED CLUSTER, not by the system's own
prediction: stratifying on the output of the model under test would bias the
evaluation set toward whatever that model already finds easy.

The LLM pre-annotation is written to SEPARATE columns (llm_suggested_*). It is
not a label. The gold label is whatever the human types in 23_label_cli.py, and
the two are never merged.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import classify, escalate, taxonomy as tax_mod  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

TARGET = 150
STRATA = {                       # deliberately harder than the natural mix
    "core_proportional": 75,
    "rare_cluster": 20,
    "short": 12,
    "long_or_multiturn": 12,
    "likely_sensitive": 12,
    "likely_account_action": 10,
    "junk_or_offtopic": 9,
}


def main() -> int:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    seed = cfg["project"]["seed"]
    rng = np.random.default_rng(seed)
    gp_path = ROOT / cfg["paths"]["processed_dir"] / "splits" / "gold_pool.csv"
    tax_path = ROOT / "data" / "processed" / "taxonomy_v1.json"
    if not gp_path.exists() or not tax_path.exists():
        print("ERROR: run scripts/20 and 21 first", file=sys.stderr)
        return 2

    gp = pd.read_csv(gp_path).reset_index(drop=True)
    tax = tax_mod.load(tax_path)
    gp["n_words"] = gp["customer_text"].astype(str).str.split().str.len()
    print(f"gold pool: {len(gp):,} exchanges")

    # cluster the pool (unsupervised -> unbiased strata)
    texts = gp["customer_text"].astype(str)
    k = max(2, min(10, len(gp) // 15))
    try:
        X = TfidfVectorizer(ngram_range=(1, 2), min_df=2, stop_words="english",
                            sublinear_tf=True).fit_transform(texts)
        gp["cluster"] = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(X)
    except ValueError:
        gp["cluster"] = 0

    taken: set[int] = set()

    def take(mask, n, label):
        avail = [i for i in gp.index[mask] if i not in taken]
        if not avail:
            return []
        pick = rng.choice(avail, size=min(n, len(avail)), replace=False)
        taken.update(int(p) for p in pick)
        return [(int(p), label) for p in pick]

    chosen = []
    sizes = gp["cluster"].value_counts()
    rare = set(sizes.index[sizes <= sizes.median()])
    # proportional core
    for c, n in (sizes / sizes.sum() * STRATA["core_proportional"]).round().items():
        chosen += take(gp["cluster"] == c, int(n), "core_proportional")
    chosen += take(gp["cluster"].isin(rare), STRATA["rare_cluster"], "rare_cluster")
    chosen += take(gp["n_words"] <= 6, STRATA["short"], "short")
    chosen += take((gp["n_words"] >= 25) | (gp["conv_max_depth"] >= 3),
                   STRATA["long_or_multiturn"], "long_or_multiturn")
    chosen += take(gp["customer_text"].astype(str).str.contains(
        escalate.SENSITIVE_RE), STRATA["likely_sensitive"], "likely_sensitive")
    chosen += take(gp["customer_text"].astype(str).str.contains(
        escalate.ACCOUNT_ACTION_RE), STRATA["likely_account_action"], "likely_account_action")
    chosen += take(gp["n_words"] <= 3, STRATA["junk_or_offtopic"], "junk_or_offtopic")
    # top up to TARGET from anything left
    if len(chosen) < TARGET:
        chosen += take(pd.Series(True, index=gp.index), TARGET - len(chosen), "topup")

    idx = [i for i, _ in chosen][:TARGET]
    strat = {i: s for i, s in chosen}
    cand = gp.loc[idx].copy()
    cand["stratum"] = [strat[i] for i in idx]
    cand["natural_weight"] = 1.0 / cand.groupby("stratum")["stratum"].transform("size")
    print(f"sampled {len(cand)} candidates")
    print(cand["stratum"].value_counts().to_string())

    # ---- LLM PRE-ANNOTATION — suggestions only, never labels
    print("\npre-annotating with the LLM (suggestions only, NOT labels)...")
    try:
        sug = classify.llm_classify_batch(
            cand["customer_text"].astype(str).tolist(), tax,
            cfg["brand"]["selected"], cfg["models"]["pre_annotation"],
            ROOT / cfg["paths"]["cache_dir"], batch_size=10)
    except Exception as exc:                                 # noqa: BLE001
        print(f"  pre-annotation unavailable ({type(exc).__name__}); "
              f"continuing with blanks — the human still labels everything.")
        sug = [{"intent": "", "confidence": 0.0}] * len(cand)

    cand["llm_suggested_intent"] = [s.get("intent", "") for s in sug]
    cand["llm_suggested_confidence"] = [round(float(s.get("confidence", 0.0)), 3) for s in sug]
    txt = cand["customer_text"].astype(str)
    cand["llm_suggested_escalate"] = (
        txt.str.contains(escalate.SENSITIVE_RE) |
        txt.str.contains(escalate.ACCOUNT_ACTION_RE) |
        cand["llm_suggested_intent"].isin(["", tax_mod.OTHER]))

    # ---- HUMAN columns: empty by construction. Filled only by 23_label_cli.py
    for col in ("human_intent", "human_escalate", "human_escalation_reason",
                "human_difficulty", "human_ambiguous", "human_notes",
                "label_timestamp", "seconds_spent", "round"):
        cand[col] = ""

    out_dir = ROOT / cfg["paths"]["gold_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "gold_candidates.csv"
    cand.to_csv(out, index=False, encoding="utf-8")
    print(f"\n  wrote {out}")
    print(f"  {len(cand)} candidates ready for HUMAN labelling")
    print("  human_* columns are empty by design — run scripts/23_label_cli.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
