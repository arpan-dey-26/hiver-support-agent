#!/usr/bin/env python3
"""
Phase 2 — discover the intent taxonomy from REAL data.

Clusters customer messages on the DEV split only (gold_pool is never read here —
leakage path L5), shows each cluster's distinctive terms and real samples to the
LLM, and asks it to propose a small taxonomy grounded in those clusters.

The result is recorded as LLM-PROPOSED FROM REAL CLUSTERS, human-reviewable. It
is never described as hand-crafted. A review file is written alongside it with
the actual cluster samples, so the proposal can be checked against the data.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import llm, prompts, taxonomy as tax_mod  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

N_CLUSTERS = 14
SAMPLES_PER_CLUSTER = 12
LO, HI = 7, 9


def main() -> int:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    seed = cfg["project"]["seed"]
    brand = cfg["brand"]["selected"]
    dev_path = ROOT / cfg["paths"]["processed_dir"] / "splits" / "dev.csv"
    if not dev_path.exists():
        print("ERROR: run scripts/20_build_brand_dataset.py first", file=sys.stderr)
        return 2

    dev = pd.read_csv(dev_path)
    msgs = dev["customer_text"].astype(str)
    msgs = msgs[msgs.str.split().str.len() >= 3].drop_duplicates()
    print(f"clustering {len(msgs):,} DEV messages into {N_CLUSTERS} groups...")

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.6,
                          stop_words="english", sublinear_tf=True,
                          strip_accents="unicode")
    X = vec.fit_transform(msgs)
    km = KMeans(n_clusters=min(N_CLUSTERS, max(2, X.shape[0] // 20)),
                random_state=seed, n_init=10).fit(X)
    terms = np.array(vec.get_feature_names_out())

    rng = np.random.default_rng(seed)
    blocks, review = [], []
    for c in range(km.n_clusters):
        idx = np.flatnonzero(km.labels_ == c)
        if idx.size == 0:
            continue
        top = terms[np.argsort(-km.cluster_centers_[c])[:12]]
        pick = rng.choice(idx, size=min(SAMPLES_PER_CLUSTER, idx.size), replace=False)
        samples = [msgs.iloc[int(i)][:220] for i in pick]
        blocks.append(f"GROUP {c} ({idx.size} messages)\n"
                      f"  distinctive terms: {', '.join(top)}\n"
                      + "\n".join(f"  - {s}" for s in samples))
        review.append({"cluster": int(c), "size": int(idx.size),
                       "top_terms": top.tolist(), "samples": samples})

    prompt = prompts.TAXONOMY_DISCOVERY_V1.format(
        brand=brand, n_clusters=len(blocks), clusters="\n\n".join(blocks),
        lo=LO, hi=HI)
    print("asking the model to propose a taxonomy from those clusters...")
    obj, raw = llm.generate_json(prompt, cfg["models"]["generation"],
                                 ROOT / cfg["paths"]["cache_dir"],
                                 max_output_tokens=3000)
    if not isinstance(obj, dict) or not obj.get("intents"):
        print("ERROR: taxonomy proposal did not parse. Raw head:\n"
              + (raw or "")[:600], file=sys.stderr)
        return 1

    intents = []
    seen = set()
    for i in obj["intents"]:
        if not isinstance(i, dict) or not i.get("name"):
            continue
        name = str(i["name"]).strip().lower().replace(" ", "_").replace("-", "_")
        if name in seen or name == tax_mod.OTHER:
            continue
        seen.add(name)
        intents.append({
            "name": name,
            "definition": str(i.get("definition", ""))[:300],
            "includes": str(i.get("includes", ""))[:300],
            "excludes": str(i.get("excludes", ""))[:300],
            "account_specific": bool(i.get("account_specific", True)),
            "example_messages": [str(e)[:220] for e in (i.get("example_messages") or [])][:3],
        })
    intents = intents[:HI]

    tax = {
        "version": "v1",
        "brand": brand,
        "provenance": ("LLM-proposed from KMeans clusters of real DEV-split customer "
                       "messages. Human-reviewable; NOT hand-crafted. Discovered on "
                       "DEV only — gold_pool was never read (leakage path L5)."),
        "discovery": {"n_dev_messages": int(len(msgs)),
                      "n_clusters": int(km.n_clusters),
                      "samples_per_cluster": SAMPLES_PER_CLUSTER,
                      "model": cfg["models"]["generation"],
                      "prompt_version": prompts.VERSIONS["taxonomy_discovery"],
                      "seed": seed},
        "other_label": tax_mod.OTHER,
        "intents": intents,
    }
    out = ROOT / "data" / "processed" / "taxonomy_v1.json"
    tax_mod.save(out, tax)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "taxonomy_review.json").write_text(
        json.dumps({"clusters": review, "proposed": tax}, indent=2,
                   ensure_ascii=False), encoding="utf-8")

    cfg["taxonomy"]["n_intents"] = len(intents)
    cfg["taxonomy"]["intents"] = [i["name"] for i in intents]
    cfg["escalation"]["auto_allowed_intents"] = tax_mod.auto_allowed(tax)
    (ROOT / "config.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")

    print(f"\n{len(intents)} intents discovered (+ '{tax_mod.OTHER}'):")
    print(tax_mod.render_for_human(tax))
    print(f"\n  auto-handle allowlist (non-account-specific): "
          f"{tax_mod.auto_allowed(tax) or '(none)'}")
    print(f"  wrote {out}")
    print(f"  cluster evidence for review: results/taxonomy_review.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
