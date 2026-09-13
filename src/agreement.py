"""Judge-human agreement.

This is the deliverable most easily faked, so the rule enforced here is simple:
if there are no human ratings, this module returns a PENDING record. It never
substitutes model ratings for human ones, and it never estimates agreement from
anything other than real paired observations.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

from .judge import DIMS

MIN_PAIRS = 8          # below this, correlation is noise, not evidence


def compare(human_rows: list[dict], judge_rows: list[dict]) -> dict:
    """human_rows / judge_rows: aligned lists keyed by the same example ids."""
    if not human_rows:
        return {"status": "PENDING",
                "reason": ("no human reply-quality ratings exist. Judge-human "
                           "agreement has NOT been measured and must not be "
                           "reported as a number."),
                "n_pairs": 0}

    hj = {r["example_id"]: r for r in judge_rows if not r.get("judge_failed")}
    pairs = [(h, hj[h["example_id"]]) for h in human_rows if h["example_id"] in hj]
    if len(pairs) < MIN_PAIRS:
        return {"status": "INSUFFICIENT",
                "reason": (f"only {len(pairs)} paired human/judge ratings; at least "
                           f"{MIN_PAIRS} are needed before a correlation means anything"),
                "n_pairs": len(pairs)}

    out = {"status": "MEASURED", "n_pairs": len(pairs), "per_dimension": {}}
    for d in DIMS:
        h = [p[0].get(d) for p in pairs]
        j = [p[1].get(d) for p in pairs]
        keep = [(a, b) for a, b in zip(h, j) if a is not None and b is not None]
        if len(keep) < MIN_PAIRS:
            out["per_dimension"][d] = {"status": "INSUFFICIENT", "n": len(keep)}
            continue
        ha = np.array([a for a, _ in keep], dtype=float)
        ja = np.array([b for _, b in keep], dtype=float)
        rho, p = spearmanr(ha, ja)
        try:
            kappa = cohen_kappa_score(ha.astype(int), ja.astype(int),
                                      weights="quadratic")
        except Exception:
            kappa = float("nan")
        out["per_dimension"][d] = {
            "n": len(keep),
            "spearman_rho": None if np.isnan(rho) else round(float(rho), 3),
            "spearman_p": None if np.isnan(p) else round(float(p), 4),
            "quadratic_weighted_kappa": None if np.isnan(kappa) else round(float(kappa), 3),
            "exact_agreement": round(float((ha == ja).mean()), 3),
            "within_one": round(float((np.abs(ha - ja) <= 1).mean()), 3),
            "mean_abs_deviation": round(float(np.abs(ha - ja).mean()), 3),
            "judge_bias": round(float((ja - ha).mean()), 3),
        }

    hf = [p[0].get("unsupported_claims") for p in pairs]
    jf = [p[1].get("unsupported_claims") for p in pairs]
    keep = [(a, b) for a, b in zip(hf, jf) if a is not None and b is not None]
    if len(keep) >= MIN_PAIRS:
        ha = np.array([a for a, _ in keep], dtype=int)
        ja = np.array([b for _, b in keep], dtype=int)
        try:
            k = float(cohen_kappa_score(ha, ja))
        except Exception:
            k = float("nan")
        out["unsupported_claims_flag"] = {
            "n": len(keep),
            "cohen_kappa": None if np.isnan(k) else round(k, 3),
            "agreement": round(float((ha == ja).mean()), 3),
            "note": "the flag that matters most for trust",
        }
    out["interpretation_note"] = (
        "judge_bias is mean(judge - human): positive means the judge scores more "
        "generously than the human. Agreement is reported per dimension because "
        "judges typically agree on relevance and diverge on groundedness.")
    return out
