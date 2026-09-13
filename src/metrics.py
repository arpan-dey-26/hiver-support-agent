"""Metrics with uncertainty attached.

At n=150 a bare point estimate is misleading, so every headline number here
carries an interval. Safety rates use Wilson intervals (correct near 0, where a
normal approximation is not), and everything else uses the bootstrap.
"""

from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)

RNG_SEED = 42


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Binomial CI that stays sensible at 0 and 1, unlike the normal approx."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    d = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def bootstrap_ci(y_true, y_pred, fn, n_boot: int = 2000, seed: int = RNG_SEED,
                 ci: float = 0.95):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n = len(y_true)
    if n == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            vals.append(fn(y_true[idx], y_pred[idx]))
        except Exception:
            continue
    if not vals:
        return (float("nan"), float("nan"))
    lo = float(np.percentile(vals, (1 - ci) / 2 * 100))
    hi = float(np.percentile(vals, (1 + ci) / 2 * 100))
    return (lo, hi)


def intent_metrics(y_true, y_pred, labels=None, n_boot: int = 2000) -> dict:
    y_true, y_pred = list(y_true), list(y_pred)
    if not y_true:
        return {"n": 0, "note": "no human gold labels available"}
    labels = labels or sorted(set(y_true) | set(y_pred))
    acc = accuracy_score(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
    rep = classification_report(y_true, y_pred, labels=labels, zero_division=0,
                                output_dict=True)
    low_support = [l for l in labels if rep.get(l, {}).get("support", 0) < 8]
    return {
        "n": len(y_true),
        "accuracy": round(float(acc), 4),
        "accuracy_ci95": [round(v, 4) for v in bootstrap_ci(
            y_true, y_pred, lambda a, b: accuracy_score(a, b), n_boot)],
        "macro_f1": round(float(mf1), 4),
        "macro_f1_ci95": [round(v, 4) for v in bootstrap_ci(
            y_true, y_pred,
            lambda a, b: f1_score(a, b, average="macro", labels=labels, zero_division=0),
            n_boot)],
        "per_class": {l: {k: round(float(v), 4) for k, v in rep[l].items()}
                      for l in labels if l in rep},
        "confusion_matrix": {"labels": labels,
                             "matrix": confusion_matrix(y_true, y_pred,
                                                        labels=labels).tolist()},
        "labels_with_support_under_8": low_support,
        "caveat": ("per-class F1 for labels listed in labels_with_support_under_8 "
                   "is computed on fewer than 8 examples and should not be reported "
                   "as a reliable estimate"),
    }


def escalation_metrics(gold_escalate, pred_escalate) -> dict:
    """Primary safety metric is the unsafe auto-handle rate: cases the human said
    must escalate that the system auto-handled anyway."""
    g = np.asarray(list(gold_escalate), dtype=bool)
    p = np.asarray(list(pred_escalate), dtype=bool)
    if len(g) == 0:
        return {"n": 0, "note": "no human gold decisions available"}
    tp = int((g & p).sum()); fp = int((~g & p).sum())
    fn = int((g & ~p).sum()); tn = int((~g & ~p).sum())
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if prec and rec and not math.isnan(prec + rec) else float("nan")
    unsafe_n, unsafe_d = fn, int(g.sum())
    lo, hi = wilson_ci(unsafe_n, unsafe_d) if unsafe_d else (float("nan"),) * 2
    cov_lo, cov_hi = wilson_ci(int((~p).sum()), len(p))
    return {
        "n": len(g),
        "unsafe_auto_handle_rate": round(unsafe_n / unsafe_d, 4) if unsafe_d else None,
        "unsafe_auto_handle_ci95_wilson": [round(lo, 4), round(hi, 4)] if unsafe_d else None,
        "unsafe_auto_handle_count": f"{unsafe_n}/{unsafe_d}",
        "escalation_precision": None if math.isnan(prec) else round(prec, 4),
        "escalation_recall": None if math.isnan(rec) else round(rec, 4),
        "escalation_f1": None if math.isnan(f1) else round(f1, 4),
        "automation_coverage": round(float((~p).mean()), 4),
        "automation_coverage_ci95_wilson": [round(cov_lo, 4), round(cov_hi, 4)],
        "counts": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "primary_metric_note": ("unsafe_auto_handle_rate is the safety headline; "
                                "automation_coverage without it is meaningless, and "
                                "both move together as the confidence threshold moves"),
    }


def coverage_risk_curve(gold_escalate, confidences, taus) -> list[dict]:
    """Sweeping the threshold shows automation is a dial, not a capability."""
    g = np.asarray(list(gold_escalate), dtype=bool)
    c = np.asarray(list(confidences), dtype=float)
    out = []
    for t in taus:
        auto = c >= t
        unsafe_d = int(g.sum())
        unsafe = int((g & auto).sum())
        out.append({
            "tau": round(float(t), 3),
            "automation_coverage": round(float(auto.mean()), 4),
            "unsafe_auto_handle_rate": round(unsafe / unsafe_d, 4) if unsafe_d else None,
            "unsafe_count": f"{unsafe}/{unsafe_d}",
        })
    return out
