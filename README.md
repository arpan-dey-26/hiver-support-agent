# Grounded customer-support agent — `hulu_support`

An AI support agent built from the [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
corpus. For an incoming customer message it: classifies the intent, retrieves how
the brand has actually answered similar messages, drafts a reply grounded in those
precedents, and decides whether a human must take it — with a stated reason.

**The proof matters more than the system.** The evaluation harness, the leakage
guarantees and the honest accounting of what is *not* measured are the point.
See [`REPORT.md`](REPORT.md) and [`DECISIONS.md`](DECISIONS.md).

---

## Reproduce the headline results in under 15 minutes

Every LLM response is cached on disk and committed, so the headline numbers
reproduce **with no API key and no quota**:

```bash
pip install -r requirements.txt
python scripts/24_run_evaluation.py      # replays from cache/llm/
python scripts/25_make_report.py
python scripts/99_audit.py
```

The cache holds **recorded real responses**, not hand-written ones. To verify,
delete `cache/llm/`, add your own `GEMINI_API_KEY`, and re-run — the pipeline will
call the API live.

## Run from scratch

```bash
pip install -r requirements.txt
python scripts/00_check_env.py                  # checks deps, creates folders
python scripts/10_setup_key.py                  # writes .env (hidden input)
# place twcs.csv in data/raw/
python run_all.py                               # everything, in order
```

`run_all.py` pauses exactly once, for human labelling:

```bash
python scripts/23_label_cli.py                  # ~150 labels, 2 keypresses each
python run_all.py                               # resumes
```

Useful flags: `--limit N` (smoke run), `--judge-n N` (judged sample size),
`--ablation-n N` (cap the diagnostic runs), `--no-judge` (skip judging).

### API quota

The Gemini free tier allows **15 requests per minute**. `src/llm.py` self-throttles
to stay under it rather than absorbing 429s, and every response is cached, so an
interrupted run resumes for free. Override with `HIVER_RPM=30` if you have a
higher limit.

The pipeline is sized for that ceiling: roughly **426 calls end to end** (~29 min).
Classification and drafting share one call, silver labels are batched 25 at a time,
and the judge runs on a paired subsample rather than every row — see `DECISIONS.md`
D12. Retrieval, clustering, the distilled classifier, the escalation policy and all
metrics are local and cost nothing.

**This is not a full-dataset LLM evaluation.** Intent and escalation metrics cover
every human-labelled row; reply quality is judged on a subsample, recorded in
`results/metrics.json` under `evaluation_scope`.

---

## How it works

```
customer message
  ├─ intent classification     LLM + a TF-IDF model distilled from LLM labels;
  │                            disagreement is a free uncertainty signal
  ├─ retrieval (BM25)          history split only, evidence strictly older than
  │                            the query — the two leakage filters
  ├─ grounded generation       cite-or-abstain; the model enumerates its claims
  │                            and cites a precedent for each
  └─ escalation policy         deterministic rules over those signals
```

The LLM produces signals; `src/escalate.py` (pure functions, unit-tested) makes
the decision. Auto-handling requires the intent to be on an **allowlist** of
non-account-specific intents — anything unseen escalates by default.

## Leakage guarantees, asserted in code

| risk | control |
|---|---|
| same conversation in two splits | split by `conversation_id`, asserted in `20_build_brand_dataset.py` and re-checked in `99_audit.py` |
| retrieval surfacing a future reply | splits are time-ordered; retrieval filters evidence to strictly older than the query |
| gold examples inside the index | index built from `history` only; eval conversation ids excluded at query time |
| thresholds tuned on gold | τ, θ, m derived from DEV statistics; `golden_set.csv` is read only by the scoring script |
| retrieval surfacing a reply written after the query | splits use strict temporal windows: a conversation joins a window only if its whole span fits. Conversations straddling a cut are excluded and counted (`DECISIONS.md` D11) |
| model labels passed off as human | `llm_suggested_*` and `human_*` are separate columns; the suggestion is hidden in the CLI until the human commits |

Cross-split *text* dedup is deliberately **not** applied — see `DECISIONS.md` D10
for why the obvious guard was the less honest one.

## Layout

```
config.yaml            all knobs; nulls mean "not yet measured"
run_all.py             one-command pipeline
src/
  threads.py           thread reconstruction (exact depth, not log-depth)
  retrieval.py         BM25, implemented directly
  classify.py          LLM classifier + distilled TF-IDF model
  generate.py          grounded, cite-or-abstain generation
  escalate.py          the policy — pure functions
  agent.py             orchestration
  baselines.py         B0-B3
  judge.py             LLM-as-judge rubric
  metrics.py           bootstrap + Wilson intervals
  agreement.py         judge-human agreement (PENDING without human ratings)
scripts/00..99         numbered, run in order
data/gold/             golden_set.csv and the labelling note
results/               metrics.json, failure_analysis.md, run_*.json
```

## Measured, and what it showed

**Judge–human agreement: measured on 30 blind paired ratings — and the judge
failed the check.** Spearman rho runs from **-0.337 to +0.279** across the five
rubric dimensions, weighted kappas sit at or below zero on four of five, and
agreement on the unsupported-claim flag is **kappa = 0.043**, which is chance.

A second measurement agrees: the placebo run fed deliberately **random** evidence
scores 2.20 on groundedness against the agent's 2.37 — a judge separating real
from random evidence by 0.17 points is not measuring grounding.

Consequence, applied throughout: judge-based reply-quality numbers are reported
but explicitly demoted, and must not be used to rank the systems. See `REPORT.md`
§5 for the full table and `DECISIONS.md` D13 for the reasoning.

Collected with `python scripts/26_rate_replies.py --n 30` (blind — it never reads
the judge's output), then replayed from cache at no API cost.

## Known-unmeasured

The audit reports these as WARN rather than filling them in:

- **Retrieval Precision@5** — the hand-judged 50-query evaluation was not run, so
  the dense-retrieval contingency could not fire either way. Retrieval quality is
  unmeasured, and the grounding claim rests on an untested retriever.
- **Resolution-proxy precision** — the 40 sampled exchanges were read but the
  R/D/N/U label sequence was never recorded, so `is_deflection` and the
  thread-depth signals remain unvalidated heuristics.

Neither is required by the assignment. Both are stated rather than filled in.

## Citations

- Dataset: Customer Support on Twitter — Stuart Axelbrooke, Kaggle 2017, CC BY-SA 4.0
- [Banking77](https://huggingface.co/datasets/PolyAI/banking77) (PolyAI) — consulted
  as a label-granularity reference only; no labels taken from it
- Models: Google Gemini via the `google-genai` SDK
- BM25: Robertson & Zaragoza, *The Probabilistic Relevance Framework: BM25 and
  Beyond* (2009) — implemented directly in `src/retrieval.py`
- Libraries: pandas, NumPy, scikit-learn, SciPy, PyYAML
- AI coding assistance was used throughout, as the assignment permits.

## Secrets

`.env` holds the API key and is gitignored. `99_audit.py` scans every tracked text
file for key-shaped strings and fails CRITICAL if it finds one.
