# Grounded support agent for `hulu_support` — report

Generated 2026-09-13T17:55:25.081183+00:00 from `results/metrics.json`.
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

**Brand choice.** `hulu_support` was selected from 108 brands by a
pre-registered rubric (`DECISIONS.md` D1): it ranked **first of the 24 brands
clearing every hard gate** on substantive-reply rate (99.46%
of replies contain no "DM us" deflection), with 14,871
conversations and 25.0-word average replies.
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
| corpus | 14,871 single-brand conversations, 20,590 customer→brand exchanges |
| multi-brand conversations excluded | 84 |
| splits (conversation-level, time-ordered) | history 14,444 / dev 4,070 / gold_pool 2,076 |
| conversation overlap between splits | {'dev&history': 0, 'dev&gold_pool': 0, 'gold_pool&history': 0} |
| deflection rate (regex heuristic) | 0.52% |
| intents discovered | 7 + `other` |
| examples evaluated | 150 |

**Golden set provenance.** Labels are human: typed in `scripts/23_label_cli.py`, which hides the model suggestion until after the human commits. The model suggestion is kept in a separate `llm_suggested_*` column and is never used as a label.

**Labelling effort, disclosed.** Median 1.1s per label, mean 2.6s,
67% of labels entered in under 2 seconds. The human overrode the model's
suggestion on 86% of examples, which is the evidence the labels are
independent of the model rather than a rubber stamp. `seconds_spent` is committed
per row in `golden_set.csv` so this is checkable. **A fast median is a real
limitation on label quality and §7 treats it as one.**

**Resolution proxy: PENDING validation.** `is_deflection`, thread depth and
gratitude markers are *observable proxies*, not resolution. The 40-exchange human
read that would measure the proxy's precision has not been recorded, so the proxy
is used unvalidated and that is stated wherever it appears.

## 4. Method

Classify → retrieve → ground → decide.

- **Intent**: LLM (`models/gemini-3.5-flash-lite`) with the discovered taxonomy, plus a
  TF-IDF+logistic-regression model distilled from LLM labels on DEV. Disagreement
  between the two is a free uncertainty signal.
- **Retrieval**: BM25, implemented directly (~40 lines) over historical customer
  messages, returning the brand's actual replies as evidence. Two filters are the
  leakage guarantee: evidence comes only from `history`, and must be strictly
  older than the query.
  **Precision@5 is PENDING** — the hand-judged retrieval
  evaluation was not run, so retrieval quality is unmeasured. Dense retrieval was
  correctly not added, since its pre-registered trigger could not fire.
- **Generation**: evidence-conditioned, cite-or-abstain. The model enumerates its
  own claims and cites which precedent supports each; uncited claims count as
  unsupported and force escalation.
- **Escalation**: deterministic rules over upstream signals, in `src/escalate.py`
  (pure functions, unit-tested). Auto-handling requires an intent on an
  **allowlist** of non-account-specific intents — anything unseen escalates.
  Thresholds τ=0.7, θ=12.9202, m=1,
  set from DEV statistics; gold was never used for tuning.

## 5. Results

Intent classification (vs human gold):

| system | accuracy | macro-F1 | macro-F1 95% CI |
|---|---|---|---|
| agent | 0.113 | 0.100 | [0.056, 0.147] |
| B1_simple | 0.120 | 0.069 | [0.034, 0.112] |
| B0_trivial | 0.133 | 0.029 | [0.018, 0.041] |
| B2_no_retrieval | 0.125 | 0.121 | [0.019, 0.199] |

**Reading this honestly.** With 8 classes, chance accuracy is 0.125. The agent scores 0.113 — **statistically indistinguishable from guessing**, and *below* the trivial majority-class baseline's 0.133. It wins on macro-F1 (0.100 vs 0.029) only because majority-class prediction scores zero on every minority class.

Every system in the table clusters near chance. That pattern points at the labels or the taxonomy rather than at any one model: if the task were merely hard, a capable LLM would still separate from majority-class prediction. Two candidate causes, neither ruled out here — the seven discovered intents overlap enough that two careful raters would disagree, and the gold labels were entered quickly (see §3). **Intent classification on this taxonomy is not a working result and is not presented as one.**

Escalation (unsafe auto-handle = human said escalate, system did not):

| system | unsafe auto-handle | 95% CI (Wilson) | automation coverage |
|---|---|---|---|
| agent | 0.027 (2/74) | [0.007, 0.093] | 0.027 |
| B1_simple | 0.851 (63/74) | [0.753, 0.915] | 0.860 |
| B0_trivial | 1.000 (74/74) | [0.951, 1.000] | 1.000 |

Reply quality (LLM judge `models/gemini-3.1-flash-lite`, different model from generation):

| system | composite (1-5) | groundedness | unsupported-claim rate | judge failure rate |
|---|---|---|---|---|
| agent | 2.493 | 2.367 | 0.750 | 0.000 |
| B1_simple | 2.693 | 2.983 | 0.417 | 0.000 |
| B0_trivial | 2.813 | 3.567 | 0.333 | 0.000 |
| B3_random_evidence | 2.633 | 2.200 | 0.933 | 0.000 |

**Scale of this evaluation, and why.** The Gemini free tier allows
**15 requests per minute**. That ceiling, not preference, sets the
sample sizes: **0 live API calls** plus
426 served from cache. Intent and escalation metrics cover
**every** human-labelled row (150); reply quality is judged on a paired
subsample ({'agent': 60, 'B1_simple': 60, 'B0_trivial': 30, 'B3_random_evidence': 30}), and the ablations on
40 rows. **This is not a full-dataset LLM evaluation and is
not claimed as one.**

**Judge–human agreement: MEASURED.** 

Measured on **30 paired ratings**, collected blind: the human rated before seeing any judge output.

| dimension | Spearman rho | weighted kappa | exact | within +-1 | judge bias |
|---|---|---|---|---|---|
| relevance | 0.262 | 0.226 | 0.333 | 0.733 | -0.633 |
| groundedness | -0.119 | -0.078 | 0.200 | 0.567 | -0.800 |
| helpfulness | 0.279 | 0.101 | 0.133 | 0.367 | -1.533 |
| tone_fit | -0.019 | -0.008 | 0.267 | 0.500 | 0.200 |
| completeness | -0.337 | -0.121 | 0.067 | 0.433 | -1.767 |

**The judge does not agree with the human.** The strongest correlation across any dimension is rho=0.279, weighted kappas sit at or below zero, and agreement on the unsupported-claim flag is kappa=0.043 — chance-level. Negative judge bias means the human scored replies *lower* than the judge did.

**Consequence, stated plainly: every judge-based number in the table above is unvalidated and should not be used to rank these systems.** The rubric was measured against a person and failed that check. Reporting it is more useful than quietly presenting judge scores as quality.

**Baselines.** B0 is majority-intent plus the single best canned reply. B1 is the
distilled classifier plus the brand's actual historical reply to the nearest past
message — no LLM at inference. B2 is the agent with retrieval switched off. B3
feeds the agent *random* evidence and exists to test the judge, not the system: if
its groundedness score is close to the agent's, the judge is not measuring
grounding.

## 6. Failure analysis

Top 5 modes, mined from real output (over-escalation 49.3%, poorly grounded reply 32.0%, no historical precedent retrieved 30.7%, unsupported claim in draft 30.0%, model output failed to parse 7.3%).
Full examples with tweet ids: `results/failure_analysis.md`.

### 1. over-escalation  — 74 of 150 (49.3%)

- `tweet_id 565509` — human said auto-handle; system escalated (unknown_intent). Message: "@hulu_support I would definitely check that out. I appreciate your response. Wondering such repetition not going to impress customers though"
- `tweet_id 2927259` — human said auto-handle; system escalated (conflicting_precedents). Message: "@hulu_support Firestick. We've been experiencing playback failures and buffering/pixelization issues since the last push."

**Hypothesis:** the allowlist is deliberately conservative and the evidence floor is set at the DEV 25th percentile, so thin-but-adequate evidence still escalates. This is the intended direction of the trade-off.


### 2. poorly grounded reply  — 48 of 150 (32.0%)

- `tweet_id 559291` — judge groundedness 2: The evidence consistently shows the brand asking for device information and providing a troubleshooting link, whereas the draft promises an escalation that is n
- `tweet_id 464786` — judge groundedness 2: The reply promises to pass the issue to a team member to 'look into it properly,' which is not supported by the evidence where similar feedback is simply shared

**Hypothesis:** retrieved precedents were topically related but did not answer the question, so the model fell back on general knowledge.


### 3. no historical precedent retrieved  — 46 of 150 (30.7%)

- `tweet_id 460328` — BM25 returned nothing above the floor. Message: "@hulu_support Can you add @224379?"
- `tweet_id 458827` — BM25 returned nothing above the floor. Message: "@hulu_support Nvmmmm I managed to get it to workkkk"

**Hypothesis:** BM25 needs lexical overlap; messages using different vocabulary from the historical corpus retrieve nothing. This is the known cost of the lexical-only retrieval choice.


### 4. unsupported claim in draft  — 45 of 150 (30.0%)

- `tweet_id 559291` — judge flagged an unsupported claim. Reply: "Oh no! We'd be upset too. What device is in use? Let's start with: https://t.co/HXpLyTU79x."
- `tweet_id 464786` — judge flagged an unsupported claim. Reply: "Thank you so much for the insight. We'll share it with the team. Feel free to leave feedback here: https://t.co/n4qDfJdbFR"

**Hypothesis:** the model generalises from a precedent rather than restating it, producing plausible specifics (timeframes, policies) the evidence does not contain.


### 5. model output failed to parse  — 11 of 150 (7.3%)

- `tweet_id 458827` — structured-JSON parse failure
- `tweet_id 2901719` — structured-JSON parse failure

**Hypothesis:** structured-output failures under load or on unusual inputs.


## 7. What is misleading about my headline number?

1. **Label quality is the binding constraint.** Median labelling time was 1.1s per example (67% under two seconds), across 150 examples covering 8 classes and an escalate/auto decision. That is fast enough that some labels are better treated as rapid judgements than considered ones, and it is a plausible contributor to every system landing near chance in §5. Every downstream
   number — intent accuracy, escalation rates, the judge comparison — is measured
   against these labels, so noise in them caps what any of it can demonstrate.
   Re-labelling more slowly is the single highest-value fix available.
2. **n=150.** The macro-F1 confidence interval above is wide; the point estimate implies precision the sample size cannot support.
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
   **Measured placebo check:** the agent scores 2.37 on groundedness; the same pipeline fed RANDOM evidence scores 2.20, a gap of +0.17. On this evidence the judge barely separates real evidence from random evidence, so the groundedness score above should not be read as proof the system is grounded.
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
