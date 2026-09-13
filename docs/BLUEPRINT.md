# Project Blueprint — AI Customer Support Agent (Hiver SDE Intern take-home)

**Status: design document. Nothing here has been experimentally validated.**
**Revision 3 — see `## Revision history`.**

---

## Standing rules for this document

1. Every dataset statement is tagged `[DOC-ONLY]` (third-party documentation, **not inspected**) or
   `[VERIFIED]` (measured by our own inspection script, with the output committed).
2. **No number in this document is a result.** Results live in `results/` and only appear after the
   code that produced them has run.
3. The following are **UNDECIDED** and must not be finalised before the evidence named beside them:

   | Decision | Unblocked by |
   |---|---|
   | Brand | `02_inspect_dataset.py` + `03_profile_brands.py` output |
   | Intent count & taxonomy | Cluster reading on the DEV split |
   | Retrieval method | DEV retrieval evaluation (BM25 is the default; see §F) |
   | Generation / judge model IDs | Live `ListModels` + quota probe + smoke test (§P0) |
   | Escalation thresholds τ, θ, m | DEV tuning sweep |
   | "Resolved exchange" definition | Manual validation of the proxy (§F) |

4. **No deadline is assumed.** The assignment specification is the only source of truth for scope.
   Phases below are ordered by dependency and sized by relative effort (S/M/L), not by a schedule.

---

## Revision history

**Rev 3 — Phase 0 implementation correction pass (code, not design):**

1. **Thread depth was wrong.** The pointer-jumping reconstruction incremented
   depth once per doubling step, yielding `ceil(log2(depth))` — a chain of 8
   reported depths `0,0,1,2,2,3,3,3` instead of `0..7`. Roots were correct;
   depth was not. Since `pct_multiturn_depth_ge_2` is a **brand-selection gate**
   (§C), the rubric was being fed corrupted input. Replaced with a compacted
   single-edge ancestor walk (exact, still chunk-friendly). Verified on a
   synthetic fixture with a known generator distribution: multi-turn share moved
   from 21% (wrong) to ~50% (matching the fixture's true 50%).
2. **Duplicate detection was not reproducible.** `hash()` is
   PYTHONHASHSEED-randomised. Replaced with BLAKE2b-64; pinned by a self-test
   that re-hashes in a subprocess under a different seed.
3. **`n_non_ascii` overclaimed.** Renamed to `n_contains_non_ascii_char`, and a
   real script-based `n_predominantly_non_latin_script` added alongside. Both
   definitions ship in the output JSON. Neither is language detection.
4. **Smoke test cannot imply quality.** Wording corrected everywhere: it checks
   API access, structured-JSON compatibility, schema validity, latency and
   tokens. Model choice out of Phase 0 is explicitly `PROVISIONAL` and
   compatibility-only; quality is measured on DEV in P5.
5. **Multi-brand conversations were misattributed.** The old code credited the
   first outbound author in a thread, so a customer tagging two brands inflated
   one brand's counts. Selection statistics now use single-brand conversations
   only; multi-brand threads are counted separately against every participating
   brand and attributed to none.
6. **Self-test suite added** (`scripts/03_selftest_phase0.py`): depth, orphan
   parent, self-parent, cycle, duplicate IDs, deterministic hash, multi-brand,
   deflection/script metrics. 10/10 passing.
7. **Windows**: UTF-8 stdout guard in every script (cp1252 consoles raise on
   tweet text), directory creation moved into `00_check_env.py`, and
   `python -m compileall scripts` replaces the Unix-only `py_compile scripts/*.py`.

**Rev 2 corrections applied:**

1. All wall-clock deadline assumptions removed. Phases carry relative effort sizes only.
2. No hard-coded model IDs. Phase 0 enumerates live models and picks the cheapest sufficient one;
   all IDs live in `config.yaml`.
3. Phase 0 now includes a Gemini smoke test **and** a call/token budget estimate that must clear the
   measured quota before the evaluation design is locked.
4. Retrieval starts and stays at BM25 unless the DEV retrieval evaluation shows it insufficient.
   Dense/hybrid is contingent, not planned.
5. "Resolved exchange" is explicitly a proxy, validated on manually inspected examples **before** it
   is used to build the corpus, with a defined fallback if the brand is mostly deflection.
6. Brand, intent count, retrieval method, model, thresholds and resolution definition are all listed
   as UNDECIDED above and gated on evidence.
7. All anti-leakage, human-gold, judge-agreement, baseline, failure-analysis and reproducibility
   requirements preserved unchanged.

---

# A. REQUIREMENT CHECKLIST

| # | Requirement | Build | Evidence needed | Repo location | Report § | Verification |
|---|---|---|---|---|---|---|
| R1 | Pick ONE brand | Brand profiling script | Real per-brand counts table | `scripts/03_profile_brands.py` → `results/tables/brand_profile.csv` | §1 | Table committed; regenerable |
| R2 | Classify intent into self-defined set | LLM classifier + taxonomy | Per-intent P/R/F1 vs human gold | `src/classify.py`, `src/taxonomy.py` | §5 | Metrics on gold + bootstrap CI |
| R3 | Draft reply grounded in brand history | Retrieval + generation | Retrieved tweet IDs in every output | `src/retrieval.py`, `src/generate.py` | §4, §5 | Every reply carries `evidence[]`; judge groundedness |
| R4 | Auto-handle vs escalate **with reason** | Rule policy + reason codes | Unsafe-auto-handle rate, coverage–risk curve | `src/escalate.py` | §5, §7 | Gold decision labels; policy unit tests |
| R5 | Runnable repo | `Makefile` / `run_all.py` | Clean-clone run log | root | README | Fresh clone → `make reproduce` |
| R6 | **README reproduces headline in <15 min** | Cached-replay mode | Timed run, recorded | `README.md`, `cache/llm/` | README | Stopwatch a clean clone; record it |
| R7 | **150–250 hand-labelled gold examples** | Labelling CLI | `changed_by_human` column, timestamps | `data/gold/golden_set.csv` | §3 | Row count, human-edit rate |
| R8 | **Note: how sampled + how labelled** | Written note | Sampling code + guidelines | `data/gold/golden_set_notes.md`, `annotation_guidelines.md` | §3 | Note matches the actual script |
| R9 | Automated metrics | Metrics module | Accuracy, macro-F1, confusion, CIs | `src/metrics.py` | §5 | Unit tests on synthetic input |
| R10 | **LLM-as-judge rubric** | Judge + rubric | Rubric text, scores, parse-failure rate | `src/judge.py`, `prompts/judge_v1.txt` | §4, §5 | Rubric versioned & committed |
| R11 | **Evidence judge agrees with human** | Agreement study | Spearman ρ + weighted κ per dimension | `src/agreement.py`, `data/gold/human_reply_ratings.csv` | §5 | Human ratings collected *before* judge output is seen |
| R12 | ≥2 baselines (1 trivial, 1 simple) | 4 baselines | Same gold, same metrics | `src/baselines.py` | §5 | Identical eval path for all systems |
| R13 | Failure analysis: top 5 + real examples + hypotheses | Error mining | Verbatim failing examples w/ IDs | `results/failure_analysis.md` | §6 | Each example traceable to a tweet_id |
| R14 | **"What is misleading about my headline number?"** | Written section | CIs, bias directions, ablations | `REPORT.md` | §7 | Each claim backed by a measured number |
| R15 | What we'd do with one more week | Written section | — | `REPORT.md` | §8 | — |
| R16 | **Decision log, 10–15 non-obvious** | `DECISIONS.md` | — | root | — | ≥10 entries, each with rejected alternative |
| R17 | Report ≤6 pages | `REPORT.md` + PDF | — | root | — | Page count on export |
| R18 | Cite anything borrowed | Citations section | — | `README.md#citations` | — | Every library/dataset/prompt/paper listed |
| R19 | Keys out of repo | `.env` + `.gitignore` | — | `.env.example` | — | Secret scan clean |
| R20 | Subsample, not 3M rows | Sampling script w/ seed | Row counts at each stage | `src/data_prep.py` | §2 | `run_manifest.json` records all N |

**Easy to miss — where marks are lost:**

1. **R11 is the trap.** Most candidates build a judge and never validate it. Human ratings must be
   collected **blind and first**.
2. **R8** — the sampling/labelling note is a graded deliverable, not a footnote.
3. **R6's 15-minute clock** includes data prep. Solved by committing a processed brand subset plus
   the LLM response cache.
4. **R4's "stated reason"** — a boolean is not enough; the reason must be human-readable *and*
   machine-auditable.
5. **R14** is where the evaluator learns whether you are honest. Most important section in the report.
6. **R12 "fair baselines"** — a deliberately crippled baseline is a visible tell.
7. Escalation **gold labels** are a second labelling pass over the same rows. Budget for it.

---

# B. DATASET FINDINGS

**Status: `[DOC-ONLY]`. NOT INSPECTED. Phase 0 replaces this entire section with `[VERIFIED]` facts.**

Cloud-environment constraint discovered during blueprinting: this session's egress policy blocks
`kaggle.com`, `huggingface.co`, `raw.githubusercontent.com` **and**
`generativelanguage.googleapis.com`. Consequence: **all data download, all inspection and all Gemini
calls run on the local machine.** The cloud session is used for authoring code and reviewing outputs.

### `[DOC-ONLY]` Reported schema — `twcs.csv`, 7 columns

| Column | Documented meaning |
|---|---|
| `tweet_id` | Unique anonymised tweet ID |
| `author_id` | Unique anonymised user ID (brand accounts reportedly readable names, customers numeric — **verify**) |
| `inbound` | `True` = customer→company; `False` = company reply |
| `created_at` | Timestamp |
| `text` | Tweet content; phone numbers / emails reportedly masked |
| `response_tweet_id` | Comma-separated IDs of replies to this tweet |
| `in_response_to_tweet_id` | ID of the parent tweet |

`[DOC-ONLY]` ~517 MB CSV; one third-party analysis reports 2,811,774 rows; creator Stuart
Axelbrooke, 2017; licence CC BY-SA 4.0. **No verified brand list and no verified counts exist yet.**

### Inspection protocol (implemented in `scripts/02_inspect_dataset.py`)

The script **discovers** the schema rather than assuming it, and fails loudly if the real columns
differ from the documented ones.

1. **Schema truth** — dtypes, exact column names, 5 raw rows printed verbatim.
2. **Scale** — exact row count; nulls per column; duplicate `tweet_id` count.
3. **Roles** — `inbound` distribution; distinct outbound authors; whether they are readable handles.
4. **Brand table** — per outbound author: n_outbound, n_inbound directed at them, n_threads, date range.
   *Raw counts only. No selection happens here.* **Conversations in which more than one brand replies
   are excluded from every per-brand selection statistic** and reported separately against each
   participating brand, so a customer tagging two brands cannot inflate either one's counts.
5. **Thread reconstruction** — root resolution over `in_response_to_tweet_id` via a sorted index, then
   a compacted single-edge ancestor walk for **exact** depth (not pointer-jumping, which returns
   log₂ depth). Reports root-resolution rate, true depth histogram, orphan rate, cycle count.
   Nodes in a cycle get depth `-1` and are excluded from the histogram rather than given a
   plausible-looking number.
6. **Text pathologies** — empty/whitespace, <5 chars, URL-only, mention-only, emoji-only, mask tokens,
   `n_contains_non_ascii_char` (literally that), `n_predominantly_non_latin_script` (>50% of
   alphabetic chars outside Latin ranges — **script, not language**; romanised Bangla/Hindi counts as
   Latin), and duplicate normalised text via a deterministic BLAKE2b-64 hash.
7. **Time** — parse failures, min/max, replies preceding parents.
8. **Resolution proxy candidates** — thread-length distribution, who ends the thread, and the
   **deflection rate** (share of brand replies that are essentially "DM us"). Plus a dump of sampled
   exchanges for manual reading.

Step 8 gates the whole grounding design — see §F and RK-02.

### Data-quality issues to measure (not assert)

Duplicate tweets · orphaned replies · multi-parent / fan-out threads · brand-to-brand and bot traffic ·
empty or emoji-only messages · non-English · intent imbalance · **threads ending in "please DM us"** ·
retweet/quote noise · customers tagging multiple brands in one tweet.

---

# C. BRAND SELECTION

**UNDECIDED. No brand is recommended. Naming one before inspection would be fabrication.**

What is fixed in advance is the *rule*, so the choice cannot be rationalised after the fact.

### Pre-registered scoring rubric (`03_profile_brands.py`, all brands ranked)

| Criterion | Why it matters | Gate |
|---|---|---|
| Inbound customer messages | Need train + retrieval corpus + gold pool, disjoint | ≥ 20,000 |
| Reconstructable threads | Grounding needs message→reply pairs | ≥ 8,000 |
| Multi-turn share (depth ≥ 3) | Real support, not one-shot noise | ≥ 25% |
| Brand-reply rate | Corpus density | ≥ 60% of inbound get a reply |
| Intent concentration (top-8 clusters ≥70%) | A small taxonomy must be possible | ≥ 70% |
| Intent diversity (≥6 clusters, ≥300 msgs each) | Can't have 2 real intents | ≥ 6 |
| Auto/escalate separability | Needs both policy-answerable and account-specific messages | ≥ 20% each |
| **Non-deflection reply rate** | Replies must contain content to ground on | **report; low = disqualifying** |
| English share | Single-language scope | ≥ 85% |
| Date span | Time-based split must be possible | ≥ 3 months |

Selection = pass all gates, then rank by (intent concentration × non-deflection reply rate).
Ties break toward the most balanced auto/escalate mix, **not** the largest brand.

If no brand clears every gate, volume gates relax first; the **separability** and **non-deflection**
gates relax last, and any relaxation is logged in `DECISIONS.md`.

### `[PROVISIONAL HYPOTHESIS — to be confirmed or discarded by data, never reported as a finding]`

A full-service airline support account plausibly scores well, because airline support mixes
policy-answerable questions (baggage allowance, check-in windows, receipt requests, pet policy) with
irreducibly account-specific ones (rebook me, where is my bag, compensation). That mix is what makes
an auto-handle/escalate policy demonstrable rather than degenerate.

Counter-hypothesis: a consumer-software/streaming account may have more self-serve troubleshooting but
a flatter, mushier intent space. Known risk for any brand: telco/e-commerce accounts may be almost
entirely account-specific, collapsing the policy into "escalate everything", which the assignment
explicitly warns against.

**This hypothesis has zero weight in the selection.** The rubric decides.

---

# D. SYSTEM ARCHITECTURE

```
customer message (+ prior turns in thread)
        │
        ├─► [1] normalise / guard   → junk, non-English, too short → escalate(low_quality_input)
        │
        ├─► [2] intent classification
        │       LLM (structured JSON) ─┐
        │       TF-IDF + LR (silver)  ─┴─► agreement → cheap confidence signal
        │
        ├─► [3] retrieval over HISTORY split only
        │       BM25 (default), time-filtered (evidence strictly older),
        │       gold conversation_ids excluded, near-dupes dropped
        │
        ├─► [4] grounded reply generation (evidence-conditioned, cite-or-abstain)
        │
        └─► [5] escalation policy (deterministic rules over signals from 2–4)
                      │
                      ▼
                 AgentOutput
```

### Output schema

```jsonc
{
  "message_id": "1234567",
  "intent": "baggage_delayed",
  "intent_confidence": 0.78,           // calibration measured, never assumed
  "intent_runner_up": "refund_request",
  "classifier_agreement": true,         // LLM label == TF-IDF-LR label
  "decision": "auto_handle",            // auto_handle | escalate
  "reason": "High-confidence baggage query answerable from the brand's standard delayed-baggage guidance; 4 close historical precedents agree.",
  "reason_codes": ["confident_intent", "strong_retrieval_support", "no_account_action_required"],
  "reply": "...",
  "reply_type": "draft_resolution",     // draft_resolution | holding_ack
  "evidence": [
    {"tweet_id": "998877", "score": 0.71, "customer_snippet": "...", "brand_reply_snippet": "..."}
  ],
  "grounding_check": {"claims": 3, "unsupported": 0},
  "policy_version": "escalation_v1",
  "prompt_versions": {"classify": "v1", "generate": "v1"},
  "model": "<from config.yaml>",
  "seed_config": {"temperature": 0.0},
  "latency_ms": 1420
}
```

**Four deliberate improvements over the spec's schema:**

1. `reason_codes[]` beside prose `reason` — prose is for humans, codes are what escalation precision
   can actually be computed over. Free-text reasons are unevaluable.
2. `evidence[]` with real `tweet_id`s — makes "grounded" a checkable claim. Any evaluator can look a
   reply up.
3. `reply_type` — escalated messages still get a **holding acknowledgement** stating what the human
   agent will need. Real support does this, and it means escalation cannot be used to dodge
   reply-quality measurement.
4. `classifier_agreement` — a second, independent, free confidence signal. LLM self-reported
   confidence is notoriously uncalibrated; disagreement between a distilled TF-IDF model and the LLM
   is a cheap uncertainty proxy. Whether it actually predicts errors is **an empirical question we
   will test**, not an assumption.

### What we will NOT build

Twitter/X API integration · DM handoff · refunds, rebooking, credits, any account mutation · CRM or
ticketing integration · multi-brand generalisation · multilingual support · fine-tuning · agentic
tool-use loops or function calling · vector DB, service deployment, containers, web UI · cross-session
conversation state · real-time streaming · human-agent interface · PII detection beyond the dataset's
existing masking · anything that attempts to *resolve* rather than *draft*.

The system is a **draft-and-route assistant with a human in the loop**, evaluated offline.

---

# E. INTENT TAXONOMY PLAN

**Intent count is UNDECIDED.** Working range 7–9 plus `other`, justified as follows and revisable on
evidence: fewer than 6 makes the task trivial; more than ~10 makes per-class F1 unmeasurable at a gold
set of 150 (fewer than 15 examples per class). The final number comes from cluster reading.

### Discovery process (DEV split only)

1. Sample ~4,000 **first inbound messages** of threads for the chosen brand (seed fixed), DEV only.
2. Cluster: TF-IDF + KMeans over k ∈ {8,12,16,20} as the default. (An embedding-based clustering pass
   is optional and only added if TF-IDF clusters are unreadable — same restraint principle as §F.)
3. For each cluster: top terms + 20 random members. **You read these.** This step is what makes the
   taxonomy yours rather than the model's.
4. Independently, the LLM proposes a taxonomy from a ~300-message sample.
5. **Merge manually**: keep a label only if it appears in the cluster reading *and* has ≥2% support;
   collapse near-duplicates; drop anything you cannot define crisply.
6. Write `annotation_guidelines.md`: per intent — one-line definition, **inclusion rule, exclusion
   rule, 3 positive examples, 2 near-misses with the reason they're near-misses**.
7. **Pilot**: label 30 fresh DEV messages against the draft. If >20% felt genuinely ambiguous, revise
   the definitions once and re-pilot.
8. **Freeze** as `taxonomy_v1`. No changes after gold labelling starts; if a change becomes
   unavoidable it is logged and affected rows are re-labelled.

### Rules

- **`other` is a real, mandatory class** — off-topic, praise/complaint with no request, spam,
  multi-brand, genuinely novel issues. `other` → always escalate. `other` recall is a headline safety
  number.
- **Rare intents** (<2% of DEV) fold into `other` for v1 and are listed as known coverage gaps. We do
  not keep a class we cannot measure.
- **Multi-intent messages**: single primary label = what the customer most wants acted on; secondary
  in a notes field; `multi_intent=true` for slice analysis.
- **Banking77**: used only as a granularity reference and prompt-format reference. **Zero Banking77
  labels enter our taxonomy.** Cited either way.

### Label-leakage prevention

- Taxonomy discovered **exclusively on DEV**. Gold-pool messages are never read during design.
- Cluster IDs/centroids are not features at inference time.
- Thresholds and prompt wording tuned on DEV. Gold is opened **once**, at the end.

---

# F. HISTORICAL GROUNDING PLAN

## F.1 Retrieval — start simple, escalate only on evidence

**Default and starting point: BM25** (`rank_bm25`, pure Python, CPU, no model download, no API cost,
deterministic, and explainable term-by-term in a live interview).

**Dense and hybrid retrieval are NOT planned. They are contingent.**

The contingency is defined in advance so it cannot become a sophistication grab:

> After BM25 is built, evaluate it on a **DEV retrieval relevance set** — 50 DEV queries × top-5
> results, hand-judged relevant/not (~30 min of reading). Compute Precision@5 and MRR.
>
> **If BM25 Precision@5 ≥ 0.6** → BM25 is final. Dense retrieval is not built. This is logged as a
> decision with the measured number as its justification.
>
> **If BM25 Precision@5 < 0.6** → inspect the failures first. If they are dominated by vocabulary
> mismatch/paraphrase (e.g. "luggage" vs "bag", "won't load" vs "buffering"), *then* add a local
> dense retriever (`all-MiniLM-L6-v2`, offline, deterministic) and a reciprocal-rank-fusion hybrid,
> and re-run the same 50-query evaluation. Adopt whichever wins on the same measurement.
>
> If the failures are instead dominated by a bad index (deflection-only replies, broken threads), the
> fix is the corpus, **not** the retriever. Fix the corpus and re-measure BM25 before adding anything.

Cost of the contingency if triggered: `sentence-transformers` pulls PyTorch (~2 GB). If that install
is impractical on the target machine, BM25 stays and the limitation is reported honestly rather than
worked around.

| Method | Simplicity | Repro | Explainable | Cost | Role |
|---|---|---|---|---|---|
| Keyword/boolean | ★★★ | ★★★ | ★★★ | 0 | too weak |
| TF-IDF cosine | ★★★ | ★★★ | ★★★ | 0 | fallback if `rank_bm25` unavailable |
| **BM25** | ★★★ | ★★★ | ★★★ | 0 | **default, primary** |
| Dense (MiniLM, local) | ★★ | ★★★ | ★★ | ~2 GB install | contingent only |
| Hybrid RRF | ★★ | ★★★ | ★★ | as above | contingent only |

## F.2 "Resolved exchange" — an observable proxy, validated before use

**This dataset contains no resolution label. `resolution_score` is a proxy. It is never called ground
truth, in code, in the report, or in conversation.**

Candidate observable signals (weights UNDECIDED until inspection):

- brand replied at least once (**required**)
- reply is substantive rather than pure deflection (measured, see below)
- thread depth ≥ 2
- customer's final message contains gratitude/positive markers
- thread terminates rather than escalating in-channel

### Mandatory validation gate — runs *before* the corpus is built

1. Compute the proxy over the brand's threads.
2. **Sample 50 exchanges the proxy marks "resolved" and 25 it marks "not resolved". Read all 75 by
   hand.** Record, for each, whether the label looks right.
3. Report **proxy precision** (of those marked resolved, how many plausibly were) and the failure
   modes seen.
4. **Gate:** if proxy precision on the manual sample is below ~0.7, the proxy is not fit to define the
   corpus. Either revise the signals and re-validate, or drop the "resolved" filter entirely and index
   *all* exchanges with a brand reply — reporting that we could not identify resolution reliably.
5. Whatever happens, the measured proxy precision and the sample size go in the report. A weak proxy
   is a finding, not something to hide.

### Deflection contingency (RK-02) — decided in advance

The inspection script measures, per brand, the share of brand replies that are essentially
"DM us / send us a direct message" with no substantive content.

- **If the chosen brand's deflection rate is low** → proceed as designed; grounding means grounding in
  actual resolutions.
- **If it is high across the shortlist** → two responses, in order:
  1. **Reconsider the brand.** Non-deflection reply rate is already a ranking term in §C; if the
     top-ranked brand is deflection-heavy, re-rank and pick one that isn't, and log why.
  2. **If every viable brand is deflection-heavy**, narrow the claim honestly rather than overstate
     it. The system is then framed throughout — §1 framing, §4 method, §7 misleading-numbers — as
     *"drafts the correct information request and routes correctly, grounded in the brand's historical
     handling pattern"*, and we state plainly that **resolution is not observable in this dataset** so
     matching brand behaviour is not evidence of solving the customer's problem.

Option 2 is a legitimate, defensible product. Pretending option 1 happened when it didn't is not.

## F.3 Index construction

Unit = one exchange: `(customer_message, brand_reply_chain, thread_meta, resolution_score)`.
Indexed text = the customer message (must match the query distribution). Payload = the brand reply
chain, timestamps, depth, proxy score. Built **only from the HISTORY split**.

## F.4 Evidence → LLM

Top-k (k UNDECIDED, start 4–5, tuned on DEV) exchanges as `[E1] Customer: ... → Brand: ...`.
Prompt instruction: answer only from E1..Ek; every factual claim must be attributable to an evidence
item; if evidence is insufficient, say so and set `sufficient_evidence=false`. The model returns the
evidence IDs it used, checked against `evidence[]`.

## F.5 Filtering, conflicts, leakage

- **Filtering**: hard score floor; drop evidence whose intent ≠ predicted intent unless score is very
  high; prefer substantive replies over deflection-only ones when both are available.
- **Conflicts**: if retrieved brand replies materially disagree (pairwise embedding or lexical
  distance above a threshold), that is `conflicting_precedents` → **escalate**. Conflict is a signal,
  not noise to suppress.
- **Leakage** (enforced in code + unit tests): gold `conversation_id`s removed from the index;
  **time filter** — evidence `created_at` strictly earlier than the query; near-duplicate guard via
  normalised-text hash (plus MinHash) between query and candidates, with the drop count reported.

## F.6 Proving grounding is real

1. Every reply ships `evidence[]` tweet IDs — auditable by anyone.
2. **Ablation**: same pipeline, retrieval disabled. If judge groundedness/helpfulness doesn't drop,
   retrieval is decorative and we say so.
3. **Placebo**: feed *random* evidence. Scores should fall toward the no-retrieval condition. If they
   don't, the judge is not measuring grounding — and that goes straight into §7 of the report.

---

# G. ESCALATION PLAN

Deterministic, versioned rules over upstream signals. **No LLM makes the final call** — the LLM
supplies signals, the policy decides. Auditable and explainable live.

### Escalate if ANY fires (ordered — first match leads the reason)

| Code | Trigger |
|---|---|
| `sensitive_topic` | safety, injury, medical, death/bereavement, legal, fraud/unauthorised charge, discrimination, minors, self-harm, regulatory complaint |
| `account_action_required` | refund, cancel, rebook, charge, credentials, address/PII change — anything mutating an account |
| `unknown_intent` | intent == `other` |
| `low_confidence` | `intent_confidence < τ` |
| `classifier_disagreement` | LLM label ≠ TF-IDF-LR label |
| `insufficient_evidence` | fewer than *m* evidence items above score θ |
| `conflicting_precedents` | retrieved brand replies materially disagree |
| `ungrounded_draft` | `sufficient_evidence=false`, or an unsupported claim detected |
| `low_quality_input` | <5 tokens, non-English, emoji/URL only, unparseable |
| `multi_intent` | two intents above the confidence floor |

**Auto-handle** only if none fire **and** the intent is on an explicit `AUTO_ALLOWED` allowlist — not
a denylist. Unseen intents default to escalate.

**τ, θ, m are UNDECIDED** and tuned on DEV only.

### Reason format

```
escalate — insufficient_evidence: only 1 of 5 retrieved precedents scored above
threshold (best 0.31 < 0.45); the brand has no consistent historical response to
this request. Needs a human.
```

Template: `{decision} — {primary_code}: {explanation with the actual numbers}`. Numbers make the
reason verifiable.

### Evaluation

- Gold `should_escalate` labelled by a human on all gold rows.
- **Primary safety metric: unsafe auto-handle rate** = P(auto_handled | gold says escalate), with a
  Wilson CI, never a bare point estimate.
- Escalation precision/recall; **automation coverage**.
- **Coverage–risk curve**: sweep τ, plot coverage vs unsafe-auto-handle rate. A single automation
  number is a policy choice dressed as a capability.
- **Anti-degenerate guard**: always report the `always_escalate` baseline. Perfect safety, zero value.
  If our system isn't clearly better on coverage at matched risk, we say so.

---

# H. GOLDEN SET PLAN

**150 examples** (spec floor; the range is 150–250). Highest-value deliverable, and the only one no
model can produce.

### Sampling (`05_make_gold_candidates.py`, fixed seed, code committed)

Drawn from **GOLD_POOL** — conversations never used for taxonomy discovery, never indexed, never used
for baseline training.

Stratified by **unsupervised cluster ID**, not by model prediction — stratifying on the
system-under-test's output biases the eval set toward what the system already finds easy.

| Stratum | n | Purpose |
|---|---|---|
| Proportional across clusters | 75 | Core; supports per-intent F1 |
| Rare-cluster oversample | 20 | Tail coverage |
| Hard: short (<8 tokens) | 12 | Underspecified input |
| Hard: long / multi-turn context | 12 | Context handling |
| Hard: multi-intent | 10 | Label ambiguity |
| Likely-sensitive (keyword-seeded, hand-verified) | 12 | Escalation safety |
| Off-topic / junk / praise | 9 | `other` + `low_quality_input` |

**This is deliberately harder than the natural distribution.** That biases metrics in known
directions, stated explicitly in §7 rather than quietly benefited from. Each row records its
natural-distribution weight so a **reweighted estimate** can be reported alongside the raw one.

### Labelling protocol

Each row: `gold_intent`, `gold_should_escalate`, `gold_escalation_reason_code`, `difficulty`,
`ambiguous`, `notes`.

**Two column families, never merged:**

- `llm_suggested_intent`, `llm_suggested_escalate` — LLM pre-annotation.
- `human_intent`, `human_escalate` — **the human's. These are the gold labels.**
- `changed_by_human` (derived), `label_timestamp`, `seconds_spent`.

`06_label_cli.py` shows one example at a time with thread context and the taxonomy definitions, and
reveals the LLM suggestion **only after the human has typed their own label**. That single UI choice
is what makes the human-in-the-loop claim true rather than cosmetic.

The **LLM–human agreement rate on the gold set is then reported as a result** — free, honest, and it
quantifies how much the human actually contributed.

### Quality checks

- **Intra-annotator reliability**: re-label a random 30 after a gap, blind to round 1. Report Cohen's
  κ, described precisely as **self-consistency, not correctness**.
- **Stretch**: a second person labels 25 → true inter-annotator κ. Named in the report if obtained.
- **Disagreement resolution** (round 1 vs 2): re-read the guideline; silence in the guideline is a
  guideline defect — amend `annotation_guidelines.md`, note the amendment, re-check affected rows.
- **Sanity audit**: intents with <8 gold examples get a warning in the results table. We do not report
  a confident per-class F1 on 5 examples.

### Independence from tuning

`golden_set.csv` is read by exactly one module, `src/evaluate.py`. A unit test asserts no
training/tuning script imports it. Taxonomy, thresholds, prompts and baselines are frozen before the
first gold run; `run_manifest.json` records the freeze commit hash. **One evaluation run.** Any change
after seeing gold results is disclosed in the report as a second look.

---

# I. BASELINES

Four, all through the identical harness on the identical gold rows.

### B0 — Trivial
- **Intent**: majority class.
- **Reply**: one fixed canned response (the brand's most common reply pattern).
- **Escalation**: reported both ways — `always_escalate` and `always_auto_handle`.

*Tests:* what the metric looks like when the system knows nothing. If our macro-F1 isn't far above
this, the framing is broken. If the canned reply scores well with the judge, **the judge is the
problem** — and that belongs in §7.

### B1 — Simple, non-agentic
- **Intent**: TF-IDF (1–2 grams) + Logistic Regression trained on **silver labels** (LLM-labelled DEV
  messages), evaluated on human gold. Honest framing: distillation of an LLM annotator into a tiny model.
- **Reply**: **1-NN retrieval** — return the brand's actual historical reply to the most similar past
  customer message, verbatim. No generation.
- **Escalation**: keyword + rule only.

*Tests:* (i) do we need an LLM at inference time at all? (ii) does generation beat replaying what the
brand actually said? (iii) how much of the escalation policy is just keyword matching?

B1's reply baseline is the sharp one: if the LLM can't beat *literally what the brand already replied
to a near-identical message*, the generation layer isn't earning its place.

### B2 — LLM zero-shot, no retrieval
Same model, same prompt, `evidence=[]`. The retrieval ablation expressed as a baseline.

### B3 — Random-evidence placebo
Same pipeline, randomly sampled evidence. *Tests the judge, not the system.*

### Fairness commitments
Same preprocessing, same rows, same metrics, same judge prompt, same parse-failure handling. B1's
classifier gets a real hyperparameter sweep on DEV. B0's canned reply is the *best* single canned
reply by DEV judge score, not a bad one.

---

# J. EVALUATION + LLM JUDGE PLAN

### A. Intent classification
Accuracy · **macro-F1 (headline)** · per-class P/R/F1 with support · confusion matrix · `other` recall ·
**bootstrap 95% CI (10k resamples)** on every headline number. At n=150 the CI is wide; showing it is
the point.

### B. Reply quality
LLM judge (below) + human ratings on a subset. **ROUGE/BLEU against the historical reply appears in
the appendix only, labelled a similarity descriptor, never a quality metric** — the historical reply
is one of many acceptable answers and is often a deflection, so rewarding similarity rewards the wrong
thing.

### C. Escalation
**Unsafe auto-handle rate (primary, Wilson CI)** · escalation precision/recall/F1 · automation
coverage · **coverage–risk curve over τ** · reason-code accuracy · comparison against
`always_escalate` / `always_auto_handle`.

### D. Slice analysis
Intent · message length tertiles · confidence bin (plus a **reliability diagram** — whether
self-reported confidence is calibrated is an open empirical question) · max retrieval score bin ·
thread depth · `difficulty` · `ambiguous` · `multi_intent` · `changed_by_human` (do rows where the
human overrode the LLM behave differently? probes gold-label anchoring).

### LLM-as-judge design

**Input:** customer message, thread context, the `evidence[]` actually shown to the generator, the
candidate reply, the system's intent and decision. **Blind**: no system name, no gold labels, system
order randomised, identical formatting across systems.

**Rubric — `prompts/judge_v1.txt`, committed and versioned:**

| Dimension | Scale | Anchors |
|---|---|---|
| Relevance | 1–5 | 1 = ignores the request; 5 = addresses exactly what was asked |
| Groundedness | 1–5 | 1 = contradicts/ignores evidence; 5 = every claim traceable to an evidence item |
| Helpfulness | 1–5 | 1 = no actionable next step; 5 = customer knows exactly what happens next |
| Tone fit | 1–5 | 1 = off-brand/robotic/rude; 5 = matches the brand's historical register |
| Completeness | 1–5 | 1 = main ask unanswered; 5 = fully addressed |
| **Unsupported claims** | 0/1 + list | 1 if any factual claim is unsupported by evidence (**hard flag, reported separately, never averaged in**) |
| Escalation appropriateness | 0/1 | was auto_handle vs escalate right for this message? |

Each dimension requires a one-sentence justification **before** the score (reason-then-score reduces
anchoring).

**Aggregation:** per-dimension reporting. A composite exists (mean of the 1–5 dims) but the
hallucination flag is **never folded into it** — averaging a safety flag into a quality score hides
it. Headline reply number = mean composite with CI, with unsupported-claim rate stated alongside.

**Malformed outputs:** strict JSON schema, temperature 0, 2 retries with a repair prompt, then
`judge_failed=true`. **Failure rate reported.** Failed rows are never silently dropped.

**Model & bias controls:** judge model **must differ from** the generation model, selected in Phase 0
(§P0). If only same-family models are available on the free tier, self-preference bias is *reduced,
not eliminated*, and the report says exactly that. Temperature 0, fixed prompt hash, every call cached.

### Human agreement study — not skippable

1. Sample **50** (message, reply) pairs, stratified across systems and intents.
2. **The human rates them first**, same rubric, same CLI, **blind to judge output**.
3. Only then does the judge run on those 50.
4. Report per dimension: **Spearman ρ**, **quadratic-weighted Cohen's κ**, exact-match %, within-±1 %,
   mean absolute deviation, and the sign/size of systematic bias (does the judge run generous?).
5. **Per dimension, not aggregate.** Judges typically agree on relevance and diverge on groundedness;
   that contrast is informative.
6. Also report agreement on the **binary unsupported-claims flag** (κ) — the one that matters most for
   trust.

**Hard rule: no judge-based number appears anywhere until this study has run, and the result is
reported even if it is bad.** If ρ is weak, the judge is demoted to secondary and human ratings carry
the headline.

---

# K. DATA LEAKAGE RISKS

| # | Leakage path | Mitigation | Test |
|---|---|---|---|
| L1 | Gold examples inside the retrieval index | Index from HISTORY only; gold `conversation_id`s hard-excluded | assert `set(gold_convs) ∩ set(index_convs) == ∅` |
| L2 | Same conversation split across sets | **Split by `conversation_id`, never by message** | assert pairwise-disjoint conversation sets |
| L3 | Near-duplicate tweets across splits | Normalised-text hash + MinHash(0.9) before splitting; cross-split dupes counted and reported | report `n_near_dupes_removed`; assert 0 exact dupes across splits |
| L4 | Future evidence answering a past query | Hard filter `evidence.created_at < query.created_at`; time-ordered split | unit test with a synthetic future doc |
| L5 | Taxonomy built by reading gold messages | Discovery reads DEV only | script asserts its input path is the DEV parquet |
| L6 | Thresholds tuned on gold | Tuning loads DEV only; gold read by `evaluate.py` alone | grep test: no tuning module imports `golden_set.csv` |
| L7 | Few-shot examples drawn from gold | Few-shots from DEV, IDs committed in `prompts/fewshots.json` | assert few-shot IDs ∉ gold IDs |
| L8 | LLM pre-annotation anchoring the human gold | Suggestion hidden until the human has typed; `changed_by_human` logged; blind re-label of 30 | report override rate + intra-annotator κ |
| L9 | Judge seeing gold labels or system identity | Judge prompt built from a field whitelist; systems anonymised and shuffled | snapshot-test the rendered judge prompt |
| L10 | Brand chosen by peeking at eval results | Brand chosen by the §C rubric before any modelling | rubric + brand table committed at an earlier commit |
| L11 | Silver labels derived from gold-pool messages | B1 trains on DEV silver only | assert training IDs ∉ gold IDs |
| L12 | Iterating on the system after seeing gold | One evaluation run; any second look disclosed | `run_manifest.json` logs every gold-eval invocation with a timestamp |

L12 is enforced by discipline, not code — so it is stated explicitly in the report.

---

# L. REPOSITORY STRUCTURE

```
hiver-twitter-support-agent/
├── README.md                  # quickstart, <15-min repro, citations, limitations
├── REPORT.md                  # the ≤6-page report (+ report.pdf)
├── DECISIONS.md               # decision log
├── requirements.txt           # pinned
├── .env.example               # GEMINI_API_KEY=   (real .env gitignored)
├── .gitignore                 # data/raw/, .env, __pycache__
├── Makefile                   # make reproduce | make fresh | make label | make test
├── config.yaml                # brand, seeds, k, τ/θ/m, MODEL IDS, split ratios
│
├── data/
│   ├── raw/                   # gitignored — twcs.csv
│   ├── processed/
│   │   ├── brand_threads.parquet          # committed (small)
│   │   └── splits/ history / dev / gold_pool .parquet
│   └── gold/
│       ├── golden_set.csv                 # THE deliverable — committed
│       ├── golden_set_notes.md            # sampling + labelling note (R8)
│       ├── annotation_guidelines.md
│       ├── relabel_round2.csv
│       └── human_reply_ratings.csv
│
├── prompts/                   # every prompt, versioned, plain text
│   ├── classify_v1.txt  generate_v1.txt  judge_v1.txt  pre_annotate_v1.txt
│   └── fewshots.json
│
├── src/
│   ├── config.py        # loads config.yaml, seeds everything
│   ├── data_prep.py     # load → filter brand → rebuild threads → dedup → split
│   ├── taxonomy.py      # intent enum + definitions (single source of truth)
│   ├── llm.py           # Gemini client: on-disk cache, retry/backoff, JSON repair
│   ├── retrieval.py     # BM25 (+ contingent dense/hybrid) + time & leakage filters
│   ├── classify.py      # LLM classifier + TF-IDF-LR
│   ├── generate.py      # evidence-conditioned reply generation
│   ├── escalate.py      # the policy — pure functions, no I/O, fully testable
│   ├── agent.py         # orchestration → AgentOutput
│   ├── baselines.py     # B0–B3
│   ├── judge.py         # rubric call + parsing + failure accounting
│   ├── metrics.py       # metrics + bootstrap/Wilson CIs
│   ├── agreement.py     # ρ, weighted κ, MAD
│   └── evaluate.py      # ONLY module that reads golden_set.csv
│
├── scripts/             # numbered, run in order
│   ├── 00_check_env.py            01_check_gemini.py
│   ├── 02_inspect_dataset.py      03_profile_brands.py
│   ├── 04_discover_intents.py     05_make_gold_candidates.py
│   ├── 06_label_cli.py            07_run_agent.py
│   ├── 08_run_baselines.py        09_run_judge.py
│   ├── 10_agreement_study.py      11_make_report_tables.py
│
├── cache/llm/           # COMMITTED response cache → repro with no API key
├── results/
│   ├── phase0/  tables/  figures/  failure_analysis.md  run_manifest.json
└── tests/
    ├── test_splits_no_leak.py     test_escalation_rules.py
    ├── test_schema.py             test_judge_prompt_snapshot.py
```

**Key architectural choice: `cache/llm/` is committed.** Every LLM call is keyed by
`sha256(model + prompt + params)` and stored as JSON. `make reproduce` replays from cache — no API
key, no quota, no nondeterminism. `make fresh` hits the live API. The README states plainly that the
cache holds **recorded real responses** and how to verify by re-running fresh.

`src/escalate.py` is pure functions with no I/O specifically so it can be walked through and
unit-tested live in an interview.

---

# M. REPORT STRUCTURE (≤6 pages)

| § | Content | Backed by |
|---|---|---|
| **1. Problem framing & brand choice** | Brand + why, from the pre-registered rubric. **What "good" means**: a draft a human agent would send with minimal edits, and near-zero unsafe auto-handles — explicitly *not* max automation | `brand_profile.csv` |
| **2. What we chose not to build** | The §D exclusion list + why each was out of scope | — |
| **3. Data & golden set** | Verified counts at each stage, thread reconstruction rate, **proxy validation result**, split sizes, sampling table, labelling protocol, human-override rate, intra-annotator κ | `results/phase0/*`, `golden_set_notes.md` |
| **4. Method** | Architecture, **retrieval choice with the measured P@5 that justified it** (including the decision *not* to add dense retrieval, if that's what the data said), escalation rules, output schema | `retrieval_comparison.csv` |
| **5. Results** | Our system vs B0/B1/B2 on intent (acc, macro-F1 + CI), reply (judge dims + unsupported-claim rate), escalation (unsafe-auto-handle + coverage). **Coverage–risk curve. Judge–human agreement table.** Ablations incl. random-evidence placebo | `results/tables/*` |
| **6. Failure analysis** | Top 5 modes, each with a real verbatim example + `tweet_id`, frequency in gold, hypothesis, proposed fix | `failure_analysis.md` |
| **7. What is misleading about my headline number?** | See below | measured numbers only |
| **8. One more week** | Ranked by expected value | — |
| Appendix | Full per-class table, rubric text, guidelines, ROUGE descriptor | — |

### §7 — candidate limitations, each to be **verified and quantified** before it appears

1. **n=150, single annotator** → macro-F1 CI is wide; the point estimate implies precision we lack.
2. **Gold was LLM-pre-annotated** → anchoring may inflate system–gold agreement. Quantified by the
   human-override rate.
3. **Judge and agent may share a model family** → self-preference bias reduced, not removed.
4. **"Grounded" ≠ "correct"** — a reply can be faithfully grounded in a historically bad brand reply.
5. **Automation rate is a dial, not a capability** — one number hides the coverage–risk curve.
6. **Resolution is unobservable.** If the brand's replies are largely deflections, matching brand
   behaviour is not evidence of solving the problem. *(Measure the deflection rate and say so.)*
7. **The gold set is deliberately harder than reality** → unsafe-auto-handle rate is pessimistic,
   coverage optimistic on the natural distribution. Report the reweighted estimate too.
8. **Macro-F1 vs accuracy diverge** under imbalance; rare classes dominate variance.
9. **2017 data** — brand policies and tone have moved; this measures fit to a historical snapshot.

---

# N. DECISION LOG PLAN

**DECISIONS TO BE MADE / DOCUMENTED** — none are decided. Each entry: decision, why, **alternative
rejected**, cost.

1. Brand choice — which, and the rubric score that beat the runners-up.
2. Subsample size and strategy; why not all ~2.8M rows.
3. Thread reconstruction algorithm; orphan/fan-out/cycle handling and how many rows were dropped.
4. **"Resolved exchange" definition, its measured proxy precision, and the admission it is a proxy.**
5. Intent count and granularity; which candidate labels were merged or killed.
6. Fate of rare intents and the coverage cost.
7. **Retrieval method, justified by measured P@5 — including the decision NOT to add dense retrieval.**
8. k, score floor θ, conflict threshold.
9. Conservative escalation allowlist rather than denylist.
10. τ chosen at which point on the coverage–risk curve, and why.
11. Holding-reply-on-escalate design.
12. Gold set size and what it costs in per-class precision.
13. Stratified (harder-than-natural) gold sampling and the reweighting response.
14. LLM pre-annotation with the suggestion hidden until after the human types.
15. **Model selection: which Gemini models, and the two-stage basis — Phase 0 established
    compatibility and cost only; DEV measurement established sufficiency.**
15b. **Multi-brand conversations excluded from per-brand selection statistics** (and the measured share
    of conversations that were ambiguous).
16. Judge model ≠ agent model; absolute not pairwise scoring; hallucination flag kept out of composite.
17. Committing the LLM cache, and what that does and does not prove.
18. Temperature 0 everywhere; what remains nondeterministic anyway.
19. Banking77 as granularity reference only.
20. Excluding ROUGE/BLEU from headline metrics.
21. Anything changed after first seeing gold results.

Trim to the 12–15 strongest at write-up. An entry without a rejected alternative is a weak entry.

---

# O. TOP PROJECT RISKS

| ID | Risk | Impact | Mitigation | Test |
|---|---|---|---|---|
| RK-01 | **Free-tier quota cannot support the evaluation** | Project stops | **Phase 0 smoke test + budget estimate before the design locks** (§P0); cache everything; cheapest sufficient model; reduce judge coverage or gold-set eval breadth if the budget doesn't clear | `01_check_gemini.py` budget report vs measured quota |
| RK-02 | **Brand replies are mostly "DM us" deflections** | Guts the grounding claim | Deflection rate is a §C ranking term; §F.2 contingency: re-rank the brand first, narrow the claim second | Deflection-rate column in `brand_profile.csv` |
| RK-03 | **Resolution proxy is wrong** | Corpus full of unresolved junk | **Mandatory 75-example manual validation gate before corpus build**; fallback to indexing all replied exchanges | Measured proxy precision, reported |
| RK-04 | Thread reconstruction errors | Wrong pairs poison the index | Union-find + explicit orphan/cycle/fan-out handling; hand-verify 30 threads | Reconstruction rate + 30-thread audit |
| RK-05 | Severe class imbalance | Macro-F1 unstable | Report support per class; warn on <8; report macro and weighted F1 | Support column everywhere |
| RK-06 | **Judge–human agreement is poor** | Judge headline unusable | Run the study **before** writing results; if ρ weak, demote judge, lead with human ratings; report either way | The study itself |
| RK-07 | Judge insensitive to grounding | "Grounded" claim is empty | **Random-evidence placebo (B3)** | If placebo ≈ real, report as a negative result |
| RK-08 | LLM confidence uncalibrated | τ meaningless | Reliability diagram on gold; fall back to `classifier_disagreement` | Calibration plot |
| RK-09 | Hallucinated resolutions (invented policies, refund promises, fake links) | Trust failure | Cite-or-abstain prompt; unsupported-claim flag; `ungrounded_draft` → escalate | Unsupported-claim rate per system |
| RK-10 | Over-escalation | Safe but worthless | Always report `always_escalate` baseline + coverage | Coverage metric |
| RK-11 | Under-escalation on sensitive content | Real harm | Sensitive rules fire first; sensitive stratum in gold; unsafe-auto-handle is the primary metric | Recall on the sensitive stratum |
| RK-12 | Annotation inconsistency | Noisy gold | Guidelines written before labelling; blind re-label of 30; report κ; label in short blocks | Intra-annotator κ |
| RK-13 | Nondeterminism breaks repro | Evaluator gets different numbers | temperature 0, fixed seeds, committed cache, `run_manifest.json` | Run `make reproduce` twice, diff |
| RK-14 | **Scope creep / unnecessary sophistication** | Unexplainable system, wasted effort | Contingency gates (§F.1) require a measured trigger before complexity is added | Every added component cites the number that justified it |
| RK-15 | Leakage found late | Results invalid | Leakage tests written in P2, before results exist | `pytest tests/` green at every phase gate |
| RK-16 | Windows path/encoding issues (emoji, CRLF) | Silent corruption | `encoding='utf-8'` everywhere, `pathlib`, no shell-specific Makefile commands | Round-trip an emoji tweet through the pipeline |

---

# P. IMPLEMENTATION ROADMAP

Ordered by dependency. Effort is relative (S/M/L), **not a schedule**. P0–P7 constitutes a complete,
honest submission; P8–P9 is upside.

### P0 — Environment, API and data verification `[S–M]`

**Objective:** replace every `[DOC-ONLY]` claim with a `[VERIFIED]` one, and prove the LLM budget is
feasible before anything is designed around it.

Creates: `scripts/00_check_env.py`, `scripts/01_check_gemini.py`,
`scripts/02_inspect_dataset.py`, `scripts/03_selftest_phase0.py`, `config.example.yaml`.

Outputs:
- **P0.a Environment** — Python version, package availability, disk, UTF-8 sanity.
- **P0.b Gemini capability & budget** —
  1. Live `ListModels` against the real API key: which models this project can actually call.
  2. **Smoke test**: one tiny structured-JSON call per candidate, on synthetic generic fixtures.
     Checks **API access, structured-JSON compatibility, output-schema validity, latency and token
     usage — and nothing else.** It is not a quality measurement, the fixture match count does not
     rank models, and no claim that a model is "good enough" may be made from it.
  3. **Quota probe**: a short burst to observe the real rate-limit behaviour and any `429`s.
  4. **Budget estimate**: projected calls and tokens for classification + generation + judge +
     baselines + agreement, against the limits read off AI Studio.
  5. **Provisional model choice**: the cheapest candidate whose call succeeds and returns valid
     structured JSON, written to `config.yaml` as a **starting point**. A **different** model for the
     judge where available. **Model quality is decided on DEV in P5**, and the choice is revised there
     if DEV says so.
- **P0.c Dataset** — the 8-point protocol in §B, including the deflection rate and a sample dump for
  manual reading.
- **P0.d Self-tests** — `03_selftest_phase0.py` on synthetic fixtures with known answers: exact thread
  depth, orphan parent, self-parent, cycle detection, duplicate IDs, deterministic hashing,
  multi-brand attribution, deflection and script metrics.

**Checkpoint:** self-tests 10/10; verified schema and counts printed; budget estimate clears the
measured quota; provisional model IDs written to config.
**Must NOT do yet:** pick a brand, define intents, set thresholds, choose a retrieval method, treat
the model choice as final, write prompts, or build anything.

### P1 — Brand selection + splits `[M]`
Apply the §C rubric to the verified brand table. Conversation-level, time-ordered
HISTORY/DEV/GOLD_POOL splits; dedup report.
**Checkpoint:** `pytest tests/test_splits_no_leak.py` green.
**Must NOT:** open GOLD_POOL messages.

### P2 — Taxonomy `[M]`
Cluster reading on DEV → intents + `other`, with definitions, inclusion/exclusion, examples. Pilot on
30 DEV messages.
**Checkpoint:** ambiguity rate <20%; taxonomy frozen.
**Must NOT:** look at the gold pool; build the agent.

### P3 — Golden set `[L]`
Candidate sampling + labelling CLI + human labelling + blind re-label of 30 + notes.
**Checkpoint:** ≥150 rows; κ computed; `golden_set_notes.md` written.
**Must NOT:** run any model against it.

### P4 — Retrieval corpus + baselines `[M]`
**Proxy validation gate first** (75 hand-read exchanges) → then build the BM25 index → 50-query
retrieval evaluation → BM25 final or contingency triggered. B0/B1 running.
**Checkpoint:** proxy precision reported; P@5 measured; retrieval decision logged with its number.
**Must NOT:** tune on gold; add dense retrieval without a triggering measurement.

### P5 — Agent `[M–L]`
`llm.py` (+cache), `classify.py`, `generate.py`, `escalate.py`, `agent.py`. Eyeball 20 DEV outputs.
τ/θ/m tuned on DEV.
**Checkpoint:** schema valid; escalation unit tests green.
**Must NOT:** touch gold; optimise for a metric.

### P6 — Evaluation run `[S–M]`
One gold run: system + B0/B1/B2. Intent, escalation, coverage–risk curve, CIs.
**Checkpoint:** `run_manifest.json` written once.
**Must NOT:** tweak the system after seeing this — and if you do, log it.

### P7 — Judge + agreement study `[M]`
**Human rates 50 blind first**, then the judge runs; ρ/κ per dimension; placebo test (B3).
**Checkpoint:** agreement reported before any judge headline.
**Must NOT:** report judge numbers before this completes.

### P8 — Failure analysis + report `[M]`
Top 5 modes with real examples; §7 written from measured numbers; `DECISIONS.md`.
**Must NOT:** invent a failure mode that wasn't observed.

### P9 — Final audit `[S]`
Clean-clone → `make reproduce`, **timed**; secret scan; citations.
**Must NOT:** add features.

**Phase gate rule:** `pytest tests/` passes before advancing. If effort must be cut, drop ablations
and slice analysis — **never** P3 or P7.

---

# Q. BEFORE-WE-CODE CHECKLIST

**Blocking**
1. [ ] `twcs.csv` downloaded and unzipped locally.
2. [ ] Inspection route chosen: run `02_inspect_dataset.py` locally and share the output, **or**
   connect the folder here (note: the raw CSV exceeds the 400 MB staging cap, so a pre-sampled slice
   would be staged instead).
3. [ ] Gemini API key created and placed in `.env` (never in the repo).
4. [ ] `01_check_gemini.py` run → model list, smoke test, quota probe and budget estimate reviewed
   **before** the evaluation design is locked.
5. [ ] Python 3.10+; `pip install -r requirements-phase0.txt` succeeds.
6. [ ] Git repo created (public, or private with access granted to the evaluator).

**Design confirmations**
7. [ ] Holding replies on escalated messages — agreed.
8. [ ] Committing the LLM response cache for reproducibility — agreed.
9. [ ] Blind-first labelling UI (human types before seeing the suggestion) — agreed.
10. [ ] Blind reply-quality rating session in P7 — agreed, non-skippable.

**Optional**
11. [ ] A second person double-labels 25 examples → true inter-annotator κ.

---

## Citations (to be maintained in `README.md`)

- Customer Support on Twitter dataset — Stuart Axelbrooke, Kaggle, 2017, CC BY-SA 4.0.
  https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter
- Banking77 — PolyAI. https://huggingface.co/datasets/PolyAI/banking77 (granularity reference only)
- Schema documentation consulted before inspection: openBIGdata.org resource page;
  cfrivera/Twitter_Customer_Support_Analysis (GitHub). Both `[DOC-ONLY]`.
- Google Gemini API docs — https://ai.google.dev/gemini-api/docs/models ,
  https://ai.google.dev/gemini-api/docs/rate-limits
- Libraries: pandas, scikit-learn, rank_bm25, scipy, matplotlib, google-genai
  (+ sentence-transformers only if the §F.1 contingency triggers).
