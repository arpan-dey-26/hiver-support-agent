# Phase 0 — environment, API and data verification

Four scripts. Run them in order and send me the output. Nothing here builds the
project; it only replaces guesses with measurements.

**Why on your laptop:** this cloud session's egress policy blocks `kaggle.com`,
`huggingface.co`, `raw.githubusercontent.com` and
`generativelanguage.googleapis.com`. Data download and every Gemini call have to
happen locally. I author and review here.

All commands below work identically on Windows (PowerShell or cmd), macOS and
Linux. No Unix-only shell is used anywhere in the workflow.

---

## Setup

```
pip install -r requirements-phase0.txt
python scripts/00_check_env.py
```

`00_check_env.py` creates the directory tree for you (`data/raw`,
`data/processed`, `data/gold`, `results/phase0`, `cache/llm`) — there is no
`mkdir -p` step, which would not work on Windows cmd. Pass `--no-make-dirs` to
skip it.

Then download the dataset from Kaggle and unzip it so `data/raw/twcs.csv` exists:

```
kaggle datasets download -d thoughtvector/customer-support-on-twitter
```

(or download through the website — same file)

Create `.env` in the repo root — **never commit this**:

```
GEMINI_API_KEY=your_key_here
```

Add `.env` to `.gitignore` before your first commit.

---

## 0a — environment

```
python scripts/00_check_env.py --csv data/raw/twcs.csv
```

Checks Python version, packages, disk, and a UTF-8 round-trip. That last one
matters on Windows: cp1252 consoles raise `UnicodeEncodeError` on emoji in tweet
text. Every script now forces UTF-8 on stdout, but the round-trip confirms file
I/O too. Also reports whether the CSV is where it should be and whether a key is
present — it never reads or prints the key's value.

Exits non-zero on a blocking problem. → `results/phase0/env_report.json`

---

## 0b — self-tests (run before trusting any dataset output)

```
python -m compileall -q scripts
python scripts/03_selftest_phase0.py
```

Ten synthetic fixtures with known correct answers: exact thread depth (chain,
branching, and an 8-node chain that the previous implementation got wrong),
orphan parent, self-parent, 2-cycle detection, duplicate tweet IDs, deterministic
hashing across a different `PYTHONHASHSEED` in a subprocess, multi-brand
conversation attribution, and the deflection/script metrics.

Expect `10/10 passed`. If anything fails, stop and send me the output — the
dataset numbers would not be trustworthy.

*(`python -m compileall` is used rather than `python -m py_compile scripts/*.py`
because Windows cmd does not expand the glob.)*

---

## 0c — Gemini capability, quota and budget

First open <https://aistudio.google.com/rate-limit> and read your actual
free-tier RPM and RPD. The API does not expose them; only you can see them.

```
python scripts/01_check_gemini.py --rpm <your_rpm> --rpd <your_rpd>
```

Four steps, stopping early if one fails:

1. **LIST** — live `ListModels`, ranked by a cost heuristic derived from the live
   names (`*-lite` < `*-flash` < `*-pro`, newer first). No model ID is hard-coded.
2. **SMOKE** — one structured-JSON call per candidate (3 by default) on six
   synthetic generic fixtures. This verifies **API access, structured-JSON
   compatibility, output-schema validity, latency and token usage.** It does
   **not** measure model quality, and the fixture match count does not rank or
   select models. Task quality is measured on DEV in Phase 5.
3. **PROBE** — 8 tiny calls to observe real rate-limit behaviour and any 429s.
   `--skip-probe` if you'd rather not spend the quota.
4. **BUDGET** — projects total calls for pre-annotation, silver labels, the
   agent, three baselines, the judge and the agreement study, against the
   RPM/RPD you supplied. Verdict: FEASIBLE / TIGHT / NOT FEASIBLE.

→ `results/phase0/gemini_report.json`, plus a **provisional** generation model and
a **different** judge model. If only one model passes, the report flags that
self-preference bias cannot be reduced — that limitation goes in the final report
rather than being hidden.

If the verdict is NOT FEASIBLE, the levers in order: raise
`--silver-batch-size`, cut `--dev-silver-n`, then cut `--judge-systems`. Cached
calls never re-spend quota.

---

## 0d — dataset

Trial run first (fast, confirms the file parses):

```
python scripts/02_inspect_dataset.py --csv data/raw/twcs.csv --sample-rows 200000
```

Then the full pass (streams in chunks; never loads the whole file):

```
python scripts/02_inspect_dataset.py --csv data/raw/twcs.csv
```

The schema is **discovered, not assumed** — it prints the real columns and warns
loudly if they differ from the documented seven.

Covers all eight points: schema · scale and duplicates · inbound/outbound roles ·
per-brand raw counts · thread reconstruction · text pathologies · timestamps ·
resolution-proxy candidates including the **deflection rate**.

Three things worth knowing about how it counts:

- **Depth is exact.** Distance in parent edges from the conversation root.
  Conversations in a reply cycle get depth `-1` and are excluded from the
  histogram rather than given a plausible-looking number.
- **Multi-brand threads are not attributed.** If a customer tags two brands and
  both reply, that conversation is excluded from *both* brands' selection
  statistics and counted separately in `n_multibrand_convs_participated`. The
  printed summary shows what share of conversations this affects.
- **`n_contains_non_ascii_char` means exactly that.** The separate
  `n_predominantly_non_latin_script` is script-based, not language detection —
  romanised Bangla or Hindi counts as Latin. Definitions ship in the JSON.

Outputs:

| File | What it is |
|---|---|
| `results/phase0/dataset_facts.json` | every measured number, with method notes |
| `results/phase0/brand_profile_raw.csv` | per-brand counts — **raw, no selection made** |
| `results/phase0/samples_for_manual_reading.md` | 40 real exchanges for you to read |

**The samples file is the important one.** Read it and tick the boxes. It is how
we find out whether "resolved" is observable in this data at all, and whether the
brand's replies are mostly "please DM us". That answer decides the grounding
design (blueprint §F.2) — no model can hand it to us.

---

## Send me

1. terminal output of all four scripts (especially `10/10 passed`)
2. `results/phase0/dataset_facts.json`
3. `results/phase0/brand_profile_raw.csv`
4. `results/phase0/gemini_report.json` — safe to share, it contains no key
5. your ticked `samples_for_manual_reading.md`

Then we do brand selection on evidence, and I fill in `config.yaml` from the
measured values.

**Still undecided, and not to be decided before this output:** brand, intent
count and taxonomy, retrieval method, final model IDs, escalation thresholds,
and the definition of a resolved exchange.
