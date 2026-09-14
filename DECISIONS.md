# Decision Log

Non-obvious decisions, why they were made, and what was rejected.
Every entry states what evidence supports it, or marks itself as an assumption.

---

## D1 — Brand: `hulu_support`

**Evidence (measured, `results/phase0/brand_profile_raw.csv`, 108 brands profiled):**

| | value | gate | result |
|---|---|---|---|
| inbound messages in its conversations | 25,889 | ≥ 20,000 | PASS |
| reconstructable conversations | 14,871 | ≥ 8,000 | PASS |
| multi-turn share (max depth ≥ 2) | 38.61% | ≥ 25% | PASS |
| date span | 1,213 days | ≥ 90 | PASS |
| substantive-reply rate (100 − loose DM-deflection) | **99.46%** | report; low disqualifies | highest of all 24 eligible brands |
| mean reply length | 25.0 words | — | joint-longest in top tier |

24 of 108 brands passed all four hard gates. Ranked by the blueprint's measurable
ranking term (non-deflection reply rate), `hulu_support` came first.

**Rejected alternatives:**
- `AmazonHelp` — ranked #2 (99.35%) with 5.5× the volume, but rejected on
  separability: Amazon support traffic is overwhelmingly order-specific, which
  would collapse the auto-handle/escalate policy into "escalate everything".
  Also its deflection number is the least trustworthy (see D5).
- `ChipotleTweets` — #3 (98.86%) but mean reply only 13.8 words; thin grounding material.
- `GWRHelp` / `VirginTrains` — #4/#5, highest multi-turn rates (63.4% / 60.3%), but
  dominated by real-time disruption queries that cannot be answered from history.
- `British_Airways` — #7 (86.00%); the best-structured intent space of any candidate
  and the designated fallback if D2 fails.

**Trade-off accepted:** moderate volume (gold pool ≈ 1,487 conversations) in exchange
for the cleanest historical reply corpus in the dataset.

---

## D2 — Separability is an ASSUMPTION, not a measurement

The blueprint's "auto/escalate separability" criterion (≥20% of messages in each
camp) **was not measured at Phase 0** — it requires reading messages, which happens
in Phase 2.

The brand choice assumes Hulu's traffic splits into policy-answerable questions
(device support, plan differences, content availability) and account-specific ones
(double charges, account access, cancellations).

**If Phase 2 taxonomy work refutes this**, the documented fallback is
`British_Airways`, and the switch is cheap because no Hulu-specific code exists.
This assumption and its outcome will be reported either way.

---

## D3 — Multi-brand conversations excluded from brand-selection statistics

3,066 of 798,197 conversations (0.38%) have more than one brand replying. Crediting
the first outbound author — the obvious implementation — would silently attribute
another brand's traffic. These conversations are excluded from every per-brand
selection statistic and counted separately against each participating brand.

**Rejected:** first-author attribution (silent misattribution).

---

## D4 — Exact thread depth, not pointer-jumping

The first implementation used pointer-jumping union-find and incremented depth once
per doubling step, returning ⌈log₂(depth)⌉ — a chain of 8 reported depths
`0,0,1,2,2,3,3,3` instead of `0..7`. Roots were correct; depth was not. Because
multi-turn share is a brand-selection gate, the rubric was being fed corrupted input.

Replaced with a compacted single-edge ancestor walk. Verified on a synthetic fixture
with a known distribution: multi-turn share moved from 21% (wrong) to ~50% (true).
Regression test: `scripts/03_selftest_phase0.py::deep_chain_not_logarithmic`.

**Cost:** O(sum of depths) instead of O(n log depth). Negligible — measured max depth
is 649, mean 2.025 over 2.8M rows.

---

## D5 — The deflection metric measures DM-asks, not all deflection

`deflection_rate_*` matches DM-style phrasing only (`dm`, `direct message`,
`private message`, `pm us`, `inbox us`). A brand that deflects by posting a support
**link**, or by saying "shoot us a note", scores artificially low.

This is why `AmazonHelp`'s 0.55% is treated with suspicion — Amazon commonly routes
to a URL, which this regex cannot see.

**Consequence:** the ranking term in D1 is currently **unvalidated**. The 40-example
human read is what validates it (see PENDING-1). If human labels show the regex
badly under-counts deflection, D1 is revisited.

---

## D6 — Deterministic hashing for duplicate detection

Python's `hash()` is randomised per process (`PYTHONHASHSEED`), so duplicate counts
would differ between runs. Replaced with BLAKE2b-64. Collision probability over ~3M
strings ≈ 5e-7 — may overstate duplicates by under one row, reported as a bounded
approximation rather than claimed exact.

---

## D7 — Non-ASCII and non-Latin reported as two separate metrics

`n_contains_non_ascii_char` (548,632 rows) is not a language or script claim — emoji
and curly quotes trigger it. A separate script-based `n_predominantly_non_latin_script`
(21,280 rows, 0.76%) measures >50% of alphabetic characters outside Latin ranges.

Neither is language detection: romanised Hindi or Bangla counts as Latin. English
share per brand remains **UNKNOWN**.

---

## D8 — Gemini auth-key format; a stale assumption caught by the author

The key-shape diagnostic initially treated a ~39-character `AIza…` string as the only
valid Gemini API key. That came from a model prior, not from the documentation — and
it was wrong for the current date.

Google's documentation states that new keys created in AI Studio are automatically
issued as **auth keys** (bound to a service account, restricted to the Gemini API),
and that the API **rejects legacy Standard keys from September 2026**. The project's
`AQ.…` key is therefore the required type, not an anomaly.

The author caught this by checking the live documentation. Both `10_setup_key.py` and
`11_check_key_shape.py` were corrected to recognise `AQ.` auth keys, flag `AIza` as
legacy, and still reject OAuth tokens (`ya29.`) and JWTs (`eyJ…`).

**Wider lesson applied throughout:** model IDs, key formats and quota limits are
discovered from the live API or current documentation, never assumed from priors.
This is why `config.yaml` ships with `models.* = null`.

---

## D9 — CSV, not Parquet, for processed splits

The processed files are small (thousands of rows). CSV removes a binary dependency
that can fail to install, and lets the evaluator open the splits in any tool.

**Rejected:** Parquet. Better dtype fidelity, but not worth a dependency for a file
an evaluator should be able to read directly.

---

## D10 — Cross-split text deduplication deliberately NOT applied

The obvious leakage guard — drop a customer message if the same text appears in
another split — is wrong here, for two reasons.

**It deletes the evidence the system exists to use.** Two customers asking "how do I
cancel" in the same words, months apart, are separate real events. The brand's
earlier answer is legitimate historical precedent, and retrieving it is the entire
point of grounding.

**It systematically strips the evaluation splits.** Keeping the earliest occurrence
means later splits lose their duplicates — and the later splits are dev and
gold_pool. Measured on a synthetic fixture, a 70/20/10 split collapsed to **99/1/0**.
That would have silently produced an almost-empty gold pool biased toward unusual
phrasings.

**What is done instead:**
1. Splitting is by `conversation_id` and time-ordered, which closes the real leakage
   path (one conversation appearing in two splits) — asserted in code.
2. Retrieval applies a strict "evidence strictly older than query" filter.
3. `gold_pool` is deduplicated *within itself*, so no evaluation item repeats.
4. Gold items that do have an exact twin in history are **flagged**
   (`has_exact_twin_in_history`) and reported as a slice, so the effect is measured
   rather than assumed away.

This is a case where the more aggressive-looking guard was the less honest one.

---

## D11 — Strict temporal windows, and 175 conversations excluded to get them

The first split assigned conversations by their START time. That is not enough:
conversations run long (measured on the real Hulu corpus — max span **1,131 days**,
99.9th percentile 89 days), so an early-starting `history` conversation still had
exchanges inside the `dev` window.

Measured on the real data before the fix:

| | |
|---|---|
| history exchanges at/after dev start | **204 (1.36%)** across 138 conversations |
| dev exchanges at/after gold_pool start | **57 (1.35%)** across 37 conversations |
| conversation overlap between splits | 0 — conversation-level splitting was already correct |

So the leakage was temporal, not structural. The audit was right to call it CRITICAL:
retrieval could surface a reply written *after* the query it was answering.

**Fix.** Two cut timestamps taken from the quantiles of conversation start times. A
conversation joins a window only if its **entire span** fits inside that window.
A conversation straddling a cut cannot be placed without either splitting it across
windows (leakage path L2) or breaking temporal separation, so it is excluded.

**Cost, stated plainly:** the excluded conversations are long-running *by
construction*, so the splits under-represent long conversations. `split_manifest.json`
records the exact counts and the mean span of kept versus dropped conversations.
No timestamps were altered and no data was dropped to make a number look better.

Assertions now enforce both guarantees in `20_build_brand_dataset.py`, and
`99_audit.py` re-checks them independently.

---

## D12 — Redesigned to fit a 15 requests/minute free tier

The measured ceiling for `gemini-3.5-flash-lite` is **15 RPM**. The original plan
needed roughly 960 calls — over an hour of pure waiting, with 429s throughout.

| change | calls saved |
|---|---|
| classification and drafting merged into ONE call | ~150 |
| silver-label batch size 10 → 25, and 600 → 400 messages | ~44 |
| judge on a paired sample (60 agent / 60 B1 / 30 B0 / 30 B3) rather than every row | ~220 |
| ablations (B2, B3) capped at 40 rows | ~120 |
| BM25, TF-IDF, clustering, escalation, metrics all local | — |

Result: **~426 calls**, roughly 29 minutes at 15 RPM, and cached calls never
re-spend quota.

Merging classify and generate is safe *here* specifically because retrieval is
BM25 over the message text and does not depend on the predicted intent, and the
TF-IDF second opinion still supplies the disagreement signal. It does mean intent
and reply come from one model pass, which is noted in the report.

`src/llm.py` self-throttles to stay under the ceiling rather than absorbing 429s.

**What this costs in honesty terms:** reply quality is judged on a **subsample**,
not the full evaluation set. The report says so explicitly and `metrics.json`
records `evaluation_scope`. No claim of full-dataset LLM evaluation is made
anywhere.

---

## D13 — The LLM judge was validated against a human, and failed that check

The assignment asks for "evidence showing how well the judge agrees with a human".
That evidence was collected: **30 (message, reply) pairs rated by the author on the
judge's own rubric**, blind — `scripts/26_rate_replies.py` never reads the judge's
output, so the human rated first and the comparison happened afterwards.

The result is negative:

| dimension | Spearman rho | quadratic-weighted kappa |
|---|---|---|
| relevance | 0.262 | 0.226 |
| groundedness | -0.119 | -0.078 |
| helpfulness | 0.279 | 0.101 |
| tone_fit | -0.019 | -0.008 |
| completeness | -0.337 | -0.121 |

Agreement on the unsupported-claim flag is **kappa = 0.043** — chance. Judge bias is
negative on four of five dimensions, meaning the human scored replies *lower* than
the judge did.

**Decision taken as a result:** judge-based reply-quality numbers are reported but
explicitly demoted. REPORT.md §5 states that they are unvalidated and must not be
used to rank the systems. The alternative — presenting judge scores as quality
because they are the only reply metric available — would have been the easier and
less honest option.

A second measurement points the same way: the placebo run (B3), fed deliberately
**random** evidence, scores 2.20 on groundedness against the agent's 2.37. A judge
that separates real evidence from random evidence by 0.17 points is not measuring
grounding.

---

# STATUS — what is complete, and what genuinely is not

Last updated after the full evaluation run: 150 human gold labels, 366 live API
calls, 0 errors, final audit 0 critical.

### Complete

- **Model selection and quota** (was PENDING-2). `scripts/01_check_gemini.py` was run
  against the live API. `config.yaml` carries real IDs: `gemini-3.5-flash-lite` for
  generation, `gemini-3.1-flash-lite` for judging — deliberately different models.
  Measured free-tier ceiling 15 RPM; the client self-throttles (561s of waiting in
  the final run) rather than absorbing 429s.
- **Evaluation run.** 150 human-labelled examples through the agent and four
  baselines; intent, escalation and coverage-risk metrics computed with bootstrap
  and Wilson intervals.
- **Judge-human agreement** (see D13). Measured on 30 blind paired ratings.
- **Reproducibility cache.** 478 recorded responses, ~0.6 MB, committed so the
  headline numbers replay with no API key. Each entry stores the model, generation
  parameters, a SHA-256 of the prompt and the response — **never the API key and
  never the raw prompt text**.

### Genuinely still pending — not resolved, not estimated

- **Resolution-proxy validation (was PENDING-1).** The 40 sampled exchanges in
  `results/phase0/samples_for_manual_reading.md` were read, but the R/D/N/U label
  sequence **was never recorded**. No proxy-precision number exists, so
  `is_deflection` and the thread-depth signals remain unvalidated heuristics
  wherever they appear. These must be genuine human labels; they will not be
  generated, estimated or reconstructed by a model.
- **Retrieval Precision@5.** The hand-judged 50-query evaluation was not run.
  Retrieval quality is unmeasured, which is also why the dense-retrieval contingency
  in D-retrieval never fired — its trigger could not be evaluated either way.

Both are reported as PENDING in REPORT.md and as WARN by `scripts/99_audit.py`.
Neither is required by the assignment; both are limitations this project chose to
state rather than fill in.
