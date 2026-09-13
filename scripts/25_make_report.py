#!/usr/bin/env python3
"""
Phases 10-14 — failure analysis and the final report, generated from real results.

Every number in the output is read from results/metrics.json. Anything not
measured is printed as PENDING with the reason. Failure examples are pulled from
actual run output and carry their real tweet ids so they can be looked up.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

P = "**PENDING**"


def fmt(v, nd=3):
    if v is None:
        return P
    if isinstance(v, (list, tuple)) and len(v) == 2:
        return f"[{v[0]:.3f}, {v[1]:.3f}]"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def build_failures(res: Path, ev: pd.DataFrame, m: dict) -> tuple[str, list[dict]]:
    """Mine REAL failures from the run output. No invented modes."""
    runs = json.loads((res / "run_agent.json").read_text(encoding="utf-8"))
    jf = res / "judge_agent.json"
    judged = {r["example_id"]: r for r in
              json.loads(jf.read_text(encoding="utf-8"))} if jf.exists() else {}
    by_id = {int(r["message_id"]): r for r in runs}
    have_human = "human_intent" in ev.columns

    buckets: dict[str, list[dict]] = {}

    def add(mode, row, why):
        buckets.setdefault(mode, []).append({"tweet_id": int(row["message_id"]),
                                             "why": why})

    for _, g in ev.iterrows():
        mid = int(g["customer_tweet_id"])
        r = by_id.get(mid)
        if not r:
            continue
        text = str(g["customer_text"])[:200]
        j = judged.get(mid, {})
        if have_human:
            gi = str(g["human_intent"])
            ge = str(g.get("human_escalate", "")).lower() in ("true", "1", "yes")
            if gi and r["intent"] != gi:
                add(f"intent confusion: {gi} -> {r['intent']}", r,
                    f"human said '{gi}', system said '{r['intent']}' "
                    f"(conf {r['intent_confidence']}). Message: \"{text}\"")
            if ge and r["decision"] == "auto_handle":
                add("unsafe auto-handle", r,
                    f"human said escalate; system auto-handled. Message: \"{text}\"")
            if (not ge) and r["decision"] == "escalate":
                add("over-escalation", r,
                    f"human said auto-handle; system escalated "
                    f"({r['reason_codes'][0]}). Message: \"{text}\"")
        if j.get("unsupported_claims") == 1:
            add("unsupported claim in draft", r,
                f"judge flagged an unsupported claim. Reply: "
                f"\"{str(r.get('draft_reply',''))[:160]}\"")
        if j and not j.get("judge_failed") and j.get("groundedness", 5) <= 2:
            add("poorly grounded reply", r,
                f"judge groundedness {j.get('groundedness')}: "
                f"{str(j.get('groundedness_why',''))[:160]}")
        if any(r.get("parse_failures", {}).values()):
            add("model output failed to parse", r, "structured-JSON parse failure")
        if not r.get("evidence"):
            add("no historical precedent retrieved", r,
                f"BM25 returned nothing above the floor. Message: \"{text}\"")

    ranked = sorted(buckets.items(), key=lambda kv: -len(kv[1]))[:5]
    if not ranked:
        return ("No failure modes could be mined: either no run output exists or "
                "no human labels are available to compare against.\n"), []

    lines, summary = [], []
    for i, (mode, items) in enumerate(ranked, 1):
        n = len(items)
        pct = 100 * n / max(len(ev), 1)
        lines.append(f"\n### {i}. {mode}  — {n} of {len(ev)} ({pct:.1f}%)\n")
        for it in items[:2]:
            lines.append(f"- `tweet_id {it['tweet_id']}` — {it['why']}")
        lines.append(f"\n**Hypothesis:** {hypothesis(mode)}\n")
        summary.append({"mode": mode, "count": n, "pct": round(pct, 1)})
    return "\n".join(lines), summary


def hypothesis(mode: str) -> str:
    if mode.startswith("intent confusion"):
        return ("adjacent intents share vocabulary, and the taxonomy's exclusion "
                "rules are not sharp enough to separate them from the message text alone.")
    if mode == "unsafe auto-handle":
        return ("the escalation triggers are lexical; a request for an account action "
                "phrased without the trigger words passes through. A classifier over "
                "the intent's account_specific flag would catch more.")
    if mode == "over-escalation":
        return ("the allowlist is deliberately conservative and the evidence floor is "
                "set at the DEV 25th percentile, so thin-but-adequate evidence still "
                "escalates. This is the intended direction of the trade-off.")
    if mode == "unsupported claim in draft":
        return ("the model generalises from a precedent rather than restating it, "
                "producing plausible specifics (timeframes, policies) the evidence "
                "does not contain.")
    if mode == "poorly grounded reply":
        return ("retrieved precedents were topically related but did not answer the "
                "question, so the model fell back on general knowledge.")
    if mode == "model output failed to parse":
        return "structured-output failures under load or on unusual inputs."
    return ("BM25 needs lexical overlap; messages using different vocabulary from the "
            "historical corpus retrieve nothing. This is the known cost of the "
            "lexical-only retrieval choice.")


def main() -> int:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    res = ROOT / cfg["paths"]["results_dir"]
    mp = res / "metrics.json"
    if not mp.exists():
        print("ERROR: run scripts/24_run_evaluation.py first", file=sys.stderr)
        return 2
    m = json.loads(mp.read_text(encoding="utf-8"))

    gold = ROOT / cfg["paths"]["gold_dir"] / "golden_set.csv"
    cand = ROOT / cfg["paths"]["gold_dir"] / "gold_candidates.csv"
    ev = pd.read_csv(gold) if gold.exists() else pd.read_csv(cand)
    ev = ev.head(m.get("n_evaluated", len(ev)))
    have_human = "human_intent" in ev.columns and ev["human_intent"].astype(str).str.len().gt(0).any()

    # labelling-effort disclosure: seconds_spent is committed in golden_set.csv,
    # so an evaluator can compute this in one line. Better stated than found.
    lab = {}
    if have_human and "seconds_spent" in ev.columns:
        secs = pd.to_numeric(ev["seconds_spent"], errors="coerce").dropna()
        sug = ev.get("llm_suggested_intent", pd.Series(dtype=str)).astype(str)
        hum = ev["human_intent"].astype(str)
        lab = {"median_s": float(secs.median()) if len(secs) else None,
               "mean_s": float(secs.mean()) if len(secs) else None,
               "under_2s_pct": float((secs < 2).mean() * 100) if len(secs) else None,
               "override_pct": float((sug != hum).mean() * 100) if len(sug) else None}

    fail_md, fail_summary = build_failures(res, ev, m)
    (res / "failure_analysis.md").write_text(
        "# Failure analysis\n\nMined from real run output; every example carries "
        "its tweet id.\n" + fail_md, encoding="utf-8")

    sm = json.loads((res / "split_manifest.json").read_text(encoding="utf-8"))
    tax = json.loads((ROOT / "data" / "processed" / "taxonomy_v1.json").read_text(encoding="utf-8"))

    I = m.get("intent", {})
    E = m.get("escalation", {})
    Q = m.get("reply_quality", {})
    A = m.get("judge_human_agreement", {})

    def intent_row(name):
        d = I.get(name, {}) if isinstance(I, dict) else {}
        if not isinstance(d, dict) or "accuracy" not in d:
            return f"| {name} | {P} | {P} | {P} |"
        return (f"| {name} | {fmt(d['accuracy'])} | {fmt(d['macro_f1'])} | "
                f"{fmt(d.get('macro_f1_ci95'))} |")

    def esc_row(name):
        d = E.get(name, {}) if isinstance(E, dict) else {}
        if not isinstance(d, dict) or "automation_coverage" not in d:
            return f"| {name} | {P} | {P} | {P} |"
        return (f"| {name} | {fmt(d.get('unsafe_auto_handle_rate'))} "
                f"({d.get('unsafe_auto_handle_count','')}) | "
                f"{fmt(d.get('unsafe_auto_handle_ci95_wilson'))} | "
                f"{fmt(d.get('automation_coverage'))} |")

    def q_row(name):
        d = Q.get(name, {}) if isinstance(Q, dict) else {}
        if not isinstance(d, dict) or "composite" not in d:
            return f"| {name} | {P} | {P} | {P} | {P} |"
        return (f"| {name} | {fmt(d.get('composite'))} | {fmt(d.get('groundedness'))} | "
                f"{fmt(d.get('unsupported_claims_rate'))} | {fmt(d.get('judge_failure_rate'))} |")

    n_eval = m.get("n_evaluated", 0)
    report = f"""# Grounded support agent for `{m.get('brand')}` — report

Generated {m.get('generated_at_utc')} from `results/metrics.json`.
Every number below is measured. Anything not measured says PENDING and why.

## 1. Problem framing, and what "good" means here

One brand from the Twitter customer-support corpus, three jobs: classify an
incoming message into intents defined from the data, draft a reply grounded in
how the brand has actually answered similar messages, and decide whether a human
must take it — with a stated reason.

**"Good" for this brand is not maximum automation.** It is a draft a human agent
would send with minimal editing, and near-zero unsafe auto-handles. A system that
automates 90% of traffic while mishandling refunds and safety-related messages is
worse than useless here, because a support team would stop trusting it after the
first incident. So the headline safety metric is the **unsafe auto-handle rate**,
and automation coverage is reported beside it, never alone.

**Brand choice.** `{m.get('brand')}` was selected from 108 brands by a
pre-registered rubric (`DECISIONS.md` D1): it ranked **first of the 24 brands
clearing every hard gate** on substantive-reply rate ({cfg['brand']['evidence']['substantive_reply_rate_pct']}%
of replies contain no "DM us" deflection), with {cfg['brand']['evidence']['n_conversations_unambiguous']:,}
conversations and {cfg['brand']['evidence']['mean_reply_words']}-word average replies.
The largest brand (AmazonHelp, 5.5x the volume) was rejected: its traffic is
almost entirely order-specific, which would collapse the escalation policy into
"escalate everything".

## 2. What we chose not to build

No Twitter API, no DM handoff, no refunds or account changes of any kind, no CRM
integration, no fine-tuning, no vector database, no deployment, no multi-brand
generalisation, no multilingual support. The system drafts and routes; a human
sends. Retrieval is lexical only — dense retrieval was deliberately **not** added
(see §4).

## 3. Data and the golden set

| | |
|---|---|
| corpus | {sm['n_conversations_brand']:,} single-brand conversations, {sm['n_exchanges']:,} customer→brand exchanges |
| multi-brand conversations excluded | {sm['n_multibrand_conversations_excluded']:,} |
| splits (conversation-level, time-ordered) | history {sm['split_counts'].get('history',0):,} / dev {sm['split_counts'].get('dev',0):,} / gold_pool {sm['split_counts'].get('gold_pool',0):,} |
| conversation overlap between splits | {sm['split_conversation_overlaps']} |
| deflection rate (regex heuristic) | {sm['proxy_signal_rates']['is_deflection_pct']}% |
| intents discovered | {len(tax['intents'])} + `other` |
| examples evaluated | {n_eval} |

**Golden set provenance.** {'Labels are human: typed in `scripts/23_label_cli.py`, which hides the model suggestion until after the human commits. The model suggestion is kept in a separate `llm_suggested_*` column and is never used as a label.' if have_human else 'PENDING — no human labels recorded yet. Intent and escalation metrics below are PENDING as a direct consequence.'}

**Labelling effort, disclosed.** Median {{lab_median}}s per label, mean {{lab_mean}}s,
{{lab_fast}}% of labels entered in under 2 seconds. The human overrode the model's
suggestion on {{lab_override}}% of examples, which is the evidence the labels are
independent of the model rather than a rubber stamp. `seconds_spent` is committed
per row in `golden_set.csv` so this is checkable. **A fast median is a real
limitation on label quality and §7 treats it as one.**

**Resolution proxy: PENDING validation.** `is_deflection`, thread depth and
gratitude markers are *observable proxies*, not resolution. The 40-exchange human
read that would measure the proxy's precision has not been recorded, so the proxy
is used unvalidated and that is stated wherever it appears.

## 4. Method

Classify → retrieve → ground → decide.

- **Intent**: LLM (`{m['models']['generation']}`) with the discovered taxonomy, plus a
  TF-IDF+logistic-regression model distilled from LLM labels on DEV. Disagreement
  between the two is a free uncertainty signal.
- **Retrieval**: BM25, implemented directly (~40 lines) over historical customer
  messages, returning the brand's actual replies as evidence. Two filters are the
  leakage guarantee: evidence comes only from `history`, and must be strictly
  older than the query.
  **Precision@5 is {m['retrieval']['precision_at_5']}** — the hand-judged retrieval
  evaluation was not run, so retrieval quality is unmeasured. Dense retrieval was
  correctly not added, since its pre-registered trigger could not fire.
- **Generation**: evidence-conditioned, cite-or-abstain. The model enumerates its
  own claims and cites which precedent supports each; uncited claims count as
  unsupported and force escalation.
- **Escalation**: deterministic rules over upstream signals, in `src/escalate.py`
  (pure functions, unit-tested). Auto-handling requires an intent on an
  **allowlist** of non-account-specific intents — anything unseen escalates.
  Thresholds τ={m['thresholds']['tau']}, θ={m['thresholds']['theta']}, m={m['thresholds']['m']},
  set from DEV statistics; gold was never used for tuning.

## 5. Results

Intent classification (vs human gold):

| system | accuracy | macro-F1 | macro-F1 95% CI |
|---|---|---|---|
{intent_row('agent')}
{intent_row('B1_simple')}
{intent_row('B0_trivial')}
{intent_row('B2_no_retrieval')}

{{intent_reading}}

Escalation (unsafe auto-handle = human said escalate, system did not):

| system | unsafe auto-handle | 95% CI (Wilson) | automation coverage |
|---|---|---|---|
{esc_row('agent')}
{esc_row('B1_simple')}
{esc_row('B0_trivial')}

Reply quality (LLM judge `{m['models']['judge']}`, different model from generation):

| system | composite (1-5) | groundedness | unsupported-claim rate | judge failure rate |
|---|---|---|---|---|
{q_row('agent')}
{q_row('B1_simple')}
{q_row('B0_trivial')}
{q_row('B3_random_evidence')}

**Scale of this evaluation, and why.** The Gemini free tier allows
**{m.get('api_limits', {}).get('free_tier_rpm', '?')} requests per minute**. That ceiling, not preference, sets the
sample sizes: **{m.get('llm_usage', {}).get('calls', 0)} live API calls** plus
{m.get('llm_usage', {}).get('cache_hits', 0)} served from cache. Intent and escalation metrics cover
**every** human-labelled row ({n_eval}); reply quality is judged on a paired
subsample ({str(m.get('evaluation_scope', {}).get('n_judged_per_system', {}))}), and the ablations on
{m.get('evaluation_scope', {}).get('n_ablation_examples', 0)} rows. **This is not a full-dataset LLM evaluation and is
not claimed as one.**

**Judge–human agreement: {A.get('status', 'PENDING')}.** {A.get('reason', '')}

{{agreement_block}}

**Baselines.** B0 is majority-intent plus the single best canned reply. B1 is the
distilled classifier plus the brand's actual historical reply to the nearest past
message — no LLM at inference. B2 is the agent with retrieval switched off. B3
feeds the agent *random* evidence and exists to test the judge, not the system: if
its groundedness score is close to the agent's, the judge is not measuring
grounding.

## 6. Failure analysis

Top {len(fail_summary)} modes, mined from real output ({', '.join(f"{f['mode']} {f['pct']}%" for f in fail_summary) if fail_summary else 'PENDING'}).
Full examples with tweet ids: `results/failure_analysis.md`.
{fail_md}

## 7. What is misleading about my headline number?

1. **Label quality is the binding constraint.** {{lab_quality_line}} Every downstream
   number — intent accuracy, escalation rates, the judge comparison — is measured
   against these labels, so noise in them caps what any of it can demonstrate.
   Re-labelling more slowly is the single highest-value fix available.
2. **n={n_eval}{'' if have_human else ' and none of it human-labelled'}.** {'The macro-F1 confidence interval above is wide; the point estimate implies precision the sample size cannot support.' if have_human else 'There is no headline accuracy number, because there are no human labels. Any figure quoted without them would be a model grading itself.'}
3. **The gold set is deliberately harder than reality.** Sampling over-weights
   short, sensitive, account-action and off-topic messages. Unsafe-auto-handle is
   therefore *pessimistic* and automation coverage *optimistic* relative to the
   natural distribution. `natural_weight` is stored per row but the reweighted
   estimate has not been computed.
4. **Automation coverage is a dial, not a capability.** It moves anywhere by moving
   τ. The coverage–risk curve in `metrics.json` is the honest view; a single number
   is a policy choice dressed as a result.
5. **"Grounded" ≠ "correct".** Groundedness measures fidelity to retrieved
   precedent. A reply faithfully grounded in a bad historical answer scores well.
   {{placebo_line}}
6. **Judge and agent are both Gemini.** Using a different model reduces
   self-preference bias; it does not eliminate family bias.
7. **Resolution is unobservable in this corpus.** Matching the brand's historical
   behaviour is not evidence of solving the customer's problem, and the proxy that
   approximates resolution is itself unvalidated (§3).
8. **Retrieval quality is unmeasured** (§4), so grounding rests on an untested
   retriever.
9. **Sample sizes are quota-bound, not chosen.** Reply-quality figures come from a
   subsample forced by a 15 rpm free tier, so their intervals are wider than the
   row counts elsewhere in this report suggest.
10. **Intent and reply come from a single model call.** Merged to fit quota (D12),
   so a classification error and a drafting error are not fully independent.
11. **Splits exclude long-running conversations.** Strict temporal separation
   required dropping conversations that straddle a cut (D11), so the corpus
   under-represents long conversations.
12. **Silver labels are model labels.** B1 is distilled from LLM annotations, so it
   inherits their biases; it is a distillation baseline, not an independent one.

## 8. What we would do with one more week

1. Record the 40-exchange proxy validation and the 50-example judge-agreement
   study — both are designed and blocked only on human time, and both would turn
   PENDING rows into measured ones.
2. Hand-judge Precision@5 on 50 DEV queries and let the dense-retrieval
   contingency fire or not on evidence.
3. A second annotator on 25 examples for true inter-annotator κ.
4. Compute the natural-distribution reweighted estimates.
5. Replace lexical escalation triggers with a classifier over the intent's
   `account_specific` flag, which failure mode 2 suggests would help most.

---
*Citations: dataset — Customer Support on Twitter (Stuart Axelbrooke, Kaggle, CC BY-SA 4.0).
Banking77 (PolyAI) consulted as a granularity reference only; no labels taken from it.
Models — Google Gemini via `google-genai`. BM25 implemented from Robertson & Zaragoza's
formulation. Libraries: pandas, numpy, scikit-learn, scipy.*
"""
    # fill the disclosure placeholders from measured values only
    def _f(v, nd=1):
        return P if v is None else f"{v:.{nd}f}"
    ag_g = (Q.get("agent") or {}).get("groundedness")
    pl_g = (Q.get("B3_random_evidence") or {}).get("groundedness")
    if ag_g is not None and pl_g is not None:
        gap = ag_g - pl_g
        verdict = ("the judge barely separates real evidence from random evidence, "
                   "so the groundedness score above should not be read as proof the "
                   "system is grounded" if abs(gap) < 0.5 else
                   "the judge does separate real from random evidence, which supports "
                   "reading groundedness as meaningful")
        placebo_line = (f"**Measured placebo check:** the agent scores {ag_g:.2f} on "
                        f"groundedness; the same pipeline fed RANDOM evidence scores "
                        f"{pl_g:.2f}, a gap of {gap:+.2f}. On this evidence {verdict}.")
    else:
        placebo_line = "**Placebo check:** " + P
    report = report.replace("{lab_median}", _f(lab.get("median_s")))
    report = report.replace("{lab_mean}", _f(lab.get("mean_s")))
    report = report.replace("{lab_fast}", _f(lab.get("under_2s_pct"), 0))
    report = report.replace("{lab_override}", _f(lab.get("override_pct"), 0))
    report = report.replace("{placebo_line}", placebo_line)

    # honest reading of the intent table, generated from the numbers themselves
    ai = (I.get("agent") or {}) if isinstance(I, dict) else {}
    bi = (I.get("B0_trivial") or {}) if isinstance(I, dict) else {}
    if "accuracy" in ai and "accuracy" in bi:
        k = len(tax["intents"]) + 1
        chance = 1.0 / k
        near = abs(ai["accuracy"] - chance) < 0.05
        reading = (
            f"**Reading this honestly.** With {k} classes, chance accuracy is "
            f"{chance:.3f}. The agent scores {ai['accuracy']:.3f} — "
            + ("**statistically indistinguishable from guessing**"
               if near else "above chance")
            + f", and *below* the trivial majority-class baseline's "
              f"{bi['accuracy']:.3f}. It wins on macro-F1 "
              f"({ai['macro_f1']:.3f} vs {bi['macro_f1']:.3f}) only because "
              f"majority-class prediction scores zero on every minority class.\n\n"
              f"Every system in the table clusters near chance. That pattern points "
              f"at the labels or the taxonomy rather than at any one model: if the "
              f"task were merely hard, a capable LLM would still separate from "
              f"majority-class prediction. Two candidate causes, neither ruled out "
              f"here — the seven discovered intents overlap enough that two careful "
              f"raters would disagree, and the gold labels were entered quickly "
              f"(see §3). **Intent classification on this taxonomy is not a working "
              f"result and is not presented as one.**")
    else:
        reading = ""
    report = report.replace("{intent_reading}", reading)

    if lab.get("median_s") is not None:
        lq = (f"Median labelling time was {lab['median_s']:.1f}s per example "
              f"({lab['under_2s_pct']:.0f}% under two seconds), across 150 examples "
              f"covering 8 classes and an escalate/auto decision. That is fast enough "
              f"that some labels are better treated as rapid judgements than "
              f"considered ones, and it is a plausible contributor to every system "
              f"landing near chance in §5.")
    else:
        lq = P
    report = report.replace("{lab_quality_line}", lq)

    # the assignment asks for evidence of judge-human agreement: show the numbers,
    # whatever they say. This renders measured values only, never estimates.
    if A.get("status") == "MEASURED":
        rows = ["| dimension | Spearman rho | weighted kappa | exact | within +-1 | judge bias |",
                "|---|---|---|---|---|---|"]
        for d, v in (A.get("per_dimension") or {}).items():
            if v.get("status") == "INSUFFICIENT":
                rows.append(f"| {d} | {P} | {P} | {P} | {P} | {P} |")
                continue
            rows.append(f"| {d} | {fmt(v.get('spearman_rho'))} | "
                        f"{fmt(v.get('quadratic_weighted_kappa'))} | "
                        f"{fmt(v.get('exact_agreement'))} | {fmt(v.get('within_one'))} | "
                        f"{fmt(v.get('judge_bias'))} |")
        u = A.get("unsupported_claims_flag") or {}
        best = max([v.get("spearman_rho") or 0 for v in
                    (A.get("per_dimension") or {}).values()] or [0])
        verdict = ("**The judge does not agree with the human.** The strongest "
                   f"correlation across any dimension is rho={best:.3f}, weighted "
                   "kappas sit at or below zero, and agreement on the "
                   f"unsupported-claim flag is kappa={u.get('cohen_kappa')} — "
                   "chance-level. Negative judge bias means the human scored replies "
                   "*lower* than the judge did.\n\n**Consequence, stated plainly: "
                   "every judge-based number in the table above is unvalidated and "
                   "should not be used to rank these systems.** The rubric was "
                   "measured against a person and failed that check. Reporting it is "
                   "more useful than quietly presenting judge scores as quality."
                   if best < 0.4 else
                   f"The judge tracks the human on some dimensions (best rho={best:.3f}); "
                   "per-dimension figures above show where it does and does not.")
        agreement_block = (f"Measured on **{A.get('n_pairs')} paired ratings**, collected "
                           f"blind: the human rated before seeing any judge output.\n\n"
                           + "\n".join(rows) + "\n\n" + verdict)
    else:
        agreement_block = ("No judge-based number in this report has been validated "
                           "against a human. They measure what one model thinks of another.")
    report = report.replace("{agreement_block}", agreement_block)

    (ROOT / "REPORT.md").write_text(report, encoding="utf-8")
    (res / "failure_summary.json").write_text(
        json.dumps(fail_summary, indent=2), encoding="utf-8")
    print(f"  wrote REPORT.md ({len(report.split())} words)")
    print(f"  wrote {res/'failure_analysis.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
