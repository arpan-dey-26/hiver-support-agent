#!/usr/bin/env python3
"""
Phase 0c — dataset verification for the Customer Support on Twitter corpus.

Reports FACTS ONLY. It does not select a brand, define intents, or decide
anything. Every downstream decision is gated on reading this output.

The schema is DISCOVERED, not assumed: the script prints the real columns and
warns loudly if they differ from the documented seven.

Covers the 8-point protocol in BLUEPRINT §B:
  1 schema  2 scale  3 roles  4 brand table (raw counts)  5 thread
  reconstruction  6 text pathologies  7 time  8 resolution-proxy candidates
  (incl. the deflection rate, which gates the grounding design -- see §F.2)

Usage:
    python scripts/02_inspect_dataset.py --csv data/raw/twcs.csv
    python scripts/02_inspect_dataset.py --csv data/raw/twcs.csv --sample-rows 200000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Windows consoles default to cp1252 and raise UnicodeEncodeError when this
# script prints tweet text or brand handles. Force UTF-8 on stdout/stderr.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

DOCUMENTED_COLUMNS = [
    "tweet_id", "author_id", "inbound", "created_at", "text",
    "response_tweet_id", "in_response_to_tweet_id",
]

# "please DM us" style deflection. Deliberately broad; the report shows both a
# loose (contains a DM ask) and a strict (short AND contains a DM ask) count so
# the number can be sanity-checked by reading samples rather than trusted blind.
DEFLECT_RE = re.compile(
    r"(\bdms?\b|\bd\.?m\.?\b|direct message|private message|"
    r"\bpm us\b|inbox us|message us privately)", re.I)
URL_RE = re.compile(r"https?://\S+|\bwww\.\S+", re.I)
MENTION_RE = re.compile(r"@\w+")
MASK_RE = re.compile(r"__\w+__|\[(email|phone|url)\]", re.I)
WORD_RE = re.compile(r"\w+")


def is_emoji_only(s: str) -> bool:
    stripped = "".join(ch for ch in s if not ch.isspace())
    if not stripped:
        return False
    return all(unicodedata.category(ch) in ("So", "Sk", "Cn") for ch in stripped)


def norm_text(s: str) -> str:
    s = URL_RE.sub(" ", s)
    s = MENTION_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s.lower()).strip()
    return s


def hash64(s: str) -> int:
    """Deterministic 63-bit hash of a normalised string.

    Python's built-in hash() is randomised per process (PYTHONHASHSEED), so it
    cannot be used for reproducible duplicate detection -- two runs would report
    different duplicate sets. BLAKE2b is deterministic across processes,
    platforms and Python versions, and is in the standard library.

    Digest truncated to 63 bits so it fits numpy int64 without sign issues.
    Collision probability over ~3M distinct strings is ~n^2 / 2^64 ~= 5e-7,
    i.e. duplicate counts may be overstated by well under one row. That is
    reported as a known, bounded approximation rather than treated as exact.
    """
    d = hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(d, "big") & 0x7FFFFFFFFFFFFFFF


# Latin-script codepoint ranges: Basic Latin letters, Latin-1 Supplement,
# Latin Extended-A/B, IPA Extensions, Latin Extended Additional, Latin
# Extended-C/D. Used for a SCRIPT check only -- this is not language detection.
_LATIN_RANGES = (
    (0x0041, 0x005A), (0x0061, 0x007A), (0x00C0, 0x024F),
    (0x1E00, 0x1EFF), (0x2C60, 0x2C7F), (0xA720, 0xA7FF),
)


def _is_latin_letter(ch: str) -> bool:
    o = ord(ch)
    return any(lo <= o <= hi for lo, hi in _LATIN_RANGES)


def predominantly_non_latin(s: str, min_letters: int = 3,
                            threshold: float = 0.5) -> bool:
    """True when a message has at least `min_letters` alphabetic characters and
    more than `threshold` of them fall outside Latin script ranges.

    This measures SCRIPT, not language: Hindi written in Devanagari counts,
    Hindi written in Latin script (Hinglish) does not, and neither does English.
    Language identification is a Phase 1 concern and is not attempted here.
    """
    letters = [ch for ch in s if ch.isalpha()]
    if len(letters) < min_letters:
        return False
    non_latin = sum(1 for ch in letters if not _is_latin_letter(ch))
    return (non_latin / len(letters)) > threshold


# ------------------------------------------------------------------ pass A
def pass_a(csv: Path, chunksize: int, sample_rows: int | None,
           quiet: bool = False) -> dict:
    """One chunked sweep: schema, counts, pathologies, per-author aggregates,
    and the id graph columns needed for thread reconstruction."""
    acc = {
        "n_rows": 0,
        "columns": None,
        "dtypes_first_chunk": None,
        "head_rows": None,
        "nulls": defaultdict(int),
        "text": defaultdict(int),
        "time": {"parse_failures": 0, "min": None, "max": None},
        "ids": {"tweet_id_non_numeric": 0, "parent_non_numeric": 0,
                "parent_missing": 0},
    }
    author_agg: dict[str, dict] = {}
    tweet_ids, parent_ids, inbound_flags, author_codes = [], [], [], []
    text_hashes = []
    author_code_map: dict[str, int] = {}

    read_kwargs = dict(chunksize=chunksize, dtype=str, keep_default_na=False,
                       na_values=[""], encoding="utf-8", on_bad_lines="warn")
    if sample_rows:
        read_kwargs["nrows"] = sample_rows

    for chunk_no, chunk in enumerate(pd.read_csv(csv, **read_kwargs)):
        if acc["columns"] is None:
            acc["columns"] = list(chunk.columns)
            acc["dtypes_first_chunk"] = {c: str(chunk[c].dtype) for c in chunk.columns}
            acc["head_rows"] = chunk.head(5).to_dict(orient="records")
            missing = [c for c in DOCUMENTED_COLUMNS if c not in chunk.columns]
            extra = [c for c in chunk.columns if c not in DOCUMENTED_COLUMNS]
            acc["schema_matches_documentation"] = not missing and not extra
            acc["columns_missing_vs_doc"] = missing
            acc["columns_extra_vs_doc"] = extra
            if missing:
                print(f"!! documented columns absent: {missing}", file=sys.stderr)
                print("!! schema differs from documentation — downstream code "
                      "must be adapted before use.", file=sys.stderr)

        acc["n_rows"] += len(chunk)
        for c in chunk.columns:
            acc["nulls"][c] += int(chunk[c].isna().sum())

        # --- ids
        tid = pd.to_numeric(chunk.get("tweet_id"), errors="coerce")
        acc["ids"]["tweet_id_non_numeric"] += int(tid.isna().sum())
        pid_raw = chunk.get("in_response_to_tweet_id")
        pid = pd.to_numeric(pid_raw, errors="coerce")
        acc["ids"]["parent_missing"] += int(pid_raw.isna().sum())
        acc["ids"]["parent_non_numeric"] += int(
            (pid.isna() & pid_raw.notna()).sum())
        tweet_ids.append(tid.fillna(-1).to_numpy(dtype="int64"))
        parent_ids.append(pid.fillna(-1).to_numpy(dtype="int64"))

        # --- roles
        inb_raw = chunk.get("inbound", pd.Series([""] * len(chunk)))
        inb = inb_raw.astype(str).str.strip().str.lower().isin(
            ["true", "1", "t", "yes"])
        inbound_flags.append(inb.to_numpy(dtype=bool))

        authors = chunk.get("author_id", pd.Series([""] * len(chunk))).fillna("")
        codes = np.empty(len(chunk), dtype="int32")
        for i, a in enumerate(authors.to_numpy()):
            c = author_code_map.get(a)
            if c is None:
                c = len(author_code_map)
                author_code_map[a] = c
            codes[i] = c
        author_codes.append(codes)

        # --- text
        txt = chunk.get("text", pd.Series([""] * len(chunk))).fillna("")
        t = txt.to_numpy()
        acc["text"]["n_empty"] += int(sum(1 for s in t if not s.strip()))
        acc["text"]["n_under_5_chars"] += int(sum(1 for s in t if len(s.strip()) < 5))
        acc["text"]["n_url_only"] += int(
            sum(1 for s in t if s.strip() and not URL_RE.sub("", s).strip()))
        acc["text"]["n_mention_only"] += int(
            sum(1 for s in t if s.strip() and not MENTION_RE.sub("", s).strip()))
        acc["text"]["n_emoji_only"] += int(sum(1 for s in t if is_emoji_only(s)))
        acc["text"]["n_with_mask_token"] += int(sum(1 for s in t if MASK_RE.search(s)))
        # two DIFFERENT things, named for what they actually measure
        non_ascii = [s for s in t if any(ord(ch) > 127 for ch in s)]
        acc["text"]["n_contains_non_ascii_char"] += len(non_ascii)
        acc["text"]["n_predominantly_non_latin_script"] += int(
            sum(1 for s in non_ascii if predominantly_non_latin(s)))
        text_hashes.append(np.fromiter((hash64(norm_text(s)) for s in t),
                                       dtype="int64", count=len(t)))

        # --- time
        ts = pd.to_datetime(chunk.get("created_at"), errors="coerce",
                            format="mixed", utc=True)
        acc["time"]["parse_failures"] += int(ts.isna().sum())
        if ts.notna().any():
            lo, hi = ts.min(), ts.max()
            acc["time"]["min"] = lo if acc["time"]["min"] is None else min(acc["time"]["min"], lo)
            acc["time"]["max"] = hi if acc["time"]["max"] is None else max(acc["time"]["max"], hi)

        # --- per-author aggregates (outbound only = candidate brand accounts)
        out_mask = ~inb.to_numpy()
        if out_mask.any():
            sub_auth = authors.to_numpy()[out_mask]
            sub_txt = t[out_mask]
            sub_ts = ts.to_numpy()[out_mask]
            for a, s, when in zip(sub_auth, sub_txt, sub_ts):
                d = author_agg.get(a)
                if d is None:
                    d = author_agg[a] = {
                        "n_outbound": 0, "n_deflect_loose": 0,
                        "n_deflect_strict": 0, "chars": 0, "words": 0,
                        "first_seen": None, "last_seen": None,
                    }
                d["n_outbound"] += 1
                nwords = len(WORD_RE.findall(s))
                d["chars"] += len(s)
                d["words"] += nwords
                if DEFLECT_RE.search(s):
                    d["n_deflect_loose"] += 1
                    if nwords <= 25:
                        d["n_deflect_strict"] += 1
                if when is not np.datetime64("NaT"):
                    if d["first_seen"] is None or when < d["first_seen"]:
                        d["first_seen"] = when
                    if d["last_seen"] is None or when > d["last_seen"]:
                        d["last_seen"] = when

        if not quiet:
            print(f"  ...chunk {chunk_no + 1}, rows so far {acc['n_rows']:,}",
                  end="\r", flush=True)

    if not quiet:
        print(" " * 60, end="\r")
    acc["nulls"] = dict(acc["nulls"])
    acc["text"] = dict(acc["text"])
    for k in ("min", "max"):
        v = acc["time"][k]
        acc["time"][k] = None if v is None else str(v)

    graph = {
        "tweet_id": np.concatenate(tweet_ids),
        "parent_id": np.concatenate(parent_ids),
        "inbound": np.concatenate(inbound_flags),
        "author_code": np.concatenate(author_codes),
        "text_hash": np.concatenate(text_hashes),
    }
    return {"acc": acc, "graph": graph, "author_agg": author_agg,
            "author_code_map": author_code_map}


# ------------------------------------------------- thread reconstruction
def build_parent_index(tweet_id: np.ndarray, parent_id: np.ndarray) -> tuple:
    """Map each row to its parent ROW INDEX via a sorted search (no 2.8M-key
    dict). Rows whose parent id is absent from the file (orphans) and rows with
    no parent both point at themselves, marking them as roots."""
    n = len(tweet_id)
    order = np.argsort(tweet_id, kind="stable")
    sorted_ids = tweet_id[order]
    pos = np.clip(np.searchsorted(sorted_ids, parent_id), 0, max(n - 1, 0))
    resolvable = (parent_id >= 0) & (sorted_ids[pos] == parent_id) if n else \
        np.zeros(n, dtype=bool)
    self_idx = np.arange(n)
    parent_idx = np.where(resolvable, order[pos], self_idx)
    return parent_idx, resolvable, self_idx


def reconstruct_threads(tweet_id: np.ndarray, parent_id: np.ndarray,
                        max_depth: int = 1000) -> dict:
    """Resolve each tweet's conversation root and its TRUE depth from that root.

    Depth is the number of parent edges walked to reach the root. An earlier
    version used pointer-jumping (root = root[root]) and incremented depth once
    per doubling step, which yields ceil(log2(depth)) -- it reported depth 3 for
    a chain of length 7. Roots were correct; depth was not.

    This walks one parent edge at a time, which is exact, and compacts the
    active set each step so total work is sum-of-depths rather than
    n * max_depth. Support threads are shallow, so this is a handful of passes
    over a rapidly shrinking array.

    Nodes still walking at `max_depth` are in a cycle (or a pathologically deep
    chain). They are counted, and their depth is set to -1 so a corrupt value
    can never be mistaken for a real one.
    """
    n = len(tweet_id)
    if n == 0:
        return {"conversation_id": np.array([], dtype="int64"),
                "depth": np.array([], dtype="int32"), "stats": {}}

    parent_idx, resolvable, self_idx = build_parent_index(tweet_id, parent_id)

    depth = np.zeros(n, dtype="int32")
    root = self_idx.copy()

    alive = np.flatnonzero(parent_idx != self_idx)   # rows that have a real parent
    cur = parent_idx[alive]
    depth[alive] = 1
    root[alive] = cur

    steps = 1
    while alive.size and steps < max_depth:
        nxt = parent_idx[cur]
        moving = nxt != cur                          # cur is a root -> stop
        alive, cur = alive[moving], nxt[moving]
        if alive.size == 0:
            break
        depth[alive] += 1
        root[alive] = cur
        steps += 1

    n_cycle = int(alive.size)
    if n_cycle:
        depth[alive] = -1                            # never silently plausible

    valid_depth = depth[depth >= 0]
    sizes = np.bincount(root, minlength=n)
    conv_sizes = sizes[sizes > 0]
    n_replies = int((parent_id >= 0).sum())
    n_orphans = int(((parent_id >= 0) & ~resolvable).sum())

    return {
        "conversation_id": root,
        "depth": depth,
        "stats": {
            "n_tweets": int(n),
            "n_with_parent_field": n_replies,
            "n_parent_resolvable": int(resolvable.sum()),
            "n_orphan_replies": n_orphans,
            "orphan_rate_of_replies": float(n_orphans / max(n_replies, 1)),
            "n_conversations": int((sizes > 0).sum()),
            "n_unconverged_cycles": n_cycle,
            "max_walk_steps_used": int(steps),
            "depth_max": int(valid_depth.max()) if valid_depth.size else 0,
            "depth_mean": float(valid_depth.mean()) if valid_depth.size else 0.0,
            "conv_size_mean": float(conv_sizes.mean()) if conv_sizes.size else 0.0,
            "conv_size_median": float(np.median(conv_sizes)) if conv_sizes.size else 0.0,
            "conv_size_max": int(conv_sizes.max()) if conv_sizes.size else 0,
            "conv_size_histogram": {
                str(k): int(v) for k, v in
                zip(*np.unique(np.clip(conv_sizes, 0, 15), return_counts=True))
            },
            "depth_histogram": {
                str(k): int(v) for k, v in
                zip(*np.unique(np.clip(valid_depth, 0, 15), return_counts=True))
            },
        },
    }


# --------------------------------------------------------- brand table
def build_brand_table(graph: dict, conv_id: np.ndarray, depth: np.ndarray,
                      author_agg: dict, code_map: dict) -> tuple:
    inv = {v: k for k, v in code_map.items()}
    df = pd.DataFrame({
        "conv": conv_id,
        "inbound": graph["inbound"],
        "author_code": graph["author_code"],
        "depth": depth,
    })

    # (conversation, outbound author) pairs -> how many distinct brands replied
    outb = df.loc[~df["inbound"], ["conv", "author_code"]].drop_duplicates()
    brands_per_conv = outb.groupby("conv").size().rename("n_brands_in_conv")

    # A conversation is attributable to ONE brand only if exactly one distinct
    # outbound author appears in it. Picking the first author in a multi-brand
    # thread (the previous behaviour) silently credits one brand with another's
    # traffic, which would corrupt the selection rubric. Those conversations are
    # excluded from per-brand selection statistics and reported separately.
    unambiguous_convs = brands_per_conv.index[brands_per_conv == 1]
    ambiguous_convs = brands_per_conv.index[brands_per_conv > 1]
    conv_brand = (outb[outb["conv"].isin(unambiguous_convs)]
                  .set_index("conv")["author_code"].rename("brand_code"))

    conv_stats = df.groupby("conv").agg(
        conv_size=("inbound", "size"),
        n_inbound=("inbound", "sum"),
        max_depth=("depth", "max"),
    ).join(brands_per_conv)
    conv_stats = conv_stats.join(conv_brand)

    unamb = conv_stats.dropna(subset=["brand_code"]).copy()
    unamb["brand_code"] = unamb["brand_code"].astype("int64")

    # raw retention: for multi-brand conversations, record every participating
    # brand, without attributing the conversation to any one of them
    amb_pairs = outb[outb["conv"].isin(ambiguous_convs)]
    amb_convs_per_brand = amb_pairs.groupby("author_code").size()
    amb_inbound = (amb_pairs.merge(
        conv_stats[["n_inbound"]], left_on="conv", right_index=True)
        .groupby("author_code")["n_inbound"].sum()) if len(amb_pairs) else \
        pd.Series(dtype="int64")

    per_brand = {code: g for code, g in unamb.groupby("brand_code")}
    all_codes = (set(per_brand) | set(amb_convs_per_brand.index)
                 | {c for name, c in code_map.items() if name in author_agg})

    rows = []
    for code in all_codes:
        name = inv.get(code, f"<code {code}>")
        if name not in author_agg:           # not an outbound author at all
            continue
        a = author_agg[name]
        n_out = a.get("n_outbound", 0)
        g = per_brand.get(code)
        has = g is not None and len(g) > 0
        rows.append({
            "brand": name,
            # author-level, exact, unaffected by conversation attribution
            "n_outbound_tweets": n_out,
            "deflection_rate_loose_pct": round(
                100 * a.get("n_deflect_loose", 0) / n_out, 2) if n_out else None,
            "deflection_rate_strict_pct": round(
                100 * a.get("n_deflect_strict", 0) / n_out, 2) if n_out else None,
            "mean_reply_words": round(a.get("words", 0) / n_out, 1) if n_out else None,
            # selection statistics: single-brand conversations ONLY
            "n_conversations_unambiguous": int(len(g)) if has else 0,
            "n_inbound_in_unambiguous_convs": int(g["n_inbound"].sum()) if has else 0,
            "n_multiturn_unambiguous_depth_ge_2": int((g["max_depth"] >= 2).sum()) if has else 0,
            "pct_multiturn_unambiguous_depth_ge_2": round(
                float((g["max_depth"] >= 2).mean() * 100), 2) if has else None,
            "mean_conversation_size_unambiguous": round(
                float(g["conv_size"].mean()), 2) if has else None,
            # raw, retained, NOT attributed to this brand
            "n_multibrand_convs_participated": int(
                amb_convs_per_brand.get(code, 0)),
            "n_inbound_in_multibrand_convs_participated": int(
                amb_inbound.get(code, 0)),
            "first_seen": str(a.get("first_seen")) if a.get("first_seen") is not None else None,
            "last_seen": str(a.get("last_seen")) if a.get("last_seen") is not None else None,
        })

    table = (pd.DataFrame(rows)
             .sort_values("n_conversations_unambiguous", ascending=False)
             .reset_index(drop=True))
    meta = {
        "n_conversations_with_a_brand_reply": int(len(brands_per_conv)),
        "n_conversations_unambiguous": int(len(unambiguous_convs)),
        "n_conversations_multibrand": int(len(ambiguous_convs)),
        "pct_conversations_multibrand": round(
            100 * len(ambiguous_convs) / max(len(brands_per_conv), 1), 2),
        "note": ("Selection statistics use single-brand conversations only. "
                 "Multi-brand conversations are counted separately per "
                 "participating brand and are NOT attributed to any one brand."),
    }
    return table, meta


# ------------------------------------------------------- sample dump
def dump_samples(csv: Path, chunksize: int, wanted_ids: set[int],
                 out_path: Path, pairs: list[tuple[int, int, str]]) -> None:
    """Second targeted pass to fetch the text of specific tweet ids, so
    exchanges can be read by hand (BLUEPRINT §F.2 proxy validation)."""
    texts: dict[int, str] = {}
    for chunk in pd.read_csv(csv, chunksize=chunksize, dtype=str,
                             keep_default_na=False, na_values=[""],
                             encoding="utf-8", on_bad_lines="warn",
                             usecols=lambda c: c in ("tweet_id", "text")):
        tid = pd.to_numeric(chunk["tweet_id"], errors="coerce").fillna(-1).astype("int64")
        mask = tid.isin(list(wanted_ids))
        if mask.any():
            for i, t in zip(tid[mask].to_numpy(), chunk.loc[mask, "text"].to_numpy()):
                texts[int(i)] = t
        if len(texts) >= len(wanted_ids):
            break

    lines = [
        "# Sampled exchanges for manual reading",
        "",
        "Purpose: validate the 'resolved exchange' proxy and judge the",
        "deflection rate by eye before either is used (BLUEPRINT §F.2).",
        "Read these and mark each one yourself. Nothing here is a label.",
        "",
    ]
    for n, (cust_id, brand_id, brand) in enumerate(pairs, 1):
        lines += [
            f"## {n}. brand: {brand}",
            f"- customer tweet_id `{cust_id}`:",
            f"  > {texts.get(cust_id, '<text not found>')}",
            f"- brand reply tweet_id `{brand_id}`:",
            f"  > {texts.get(brand_id, '<text not found>')}",
            "",
            "  your call — resolved? [ ] yes  [ ] no  [ ] deflection only  [ ] unclear",
            "",
        ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("results/phase0"))
    ap.add_argument("--chunksize", type=int, default=250_000)
    ap.add_argument("--sample-rows", type=int, default=None,
                    help="read only the first N rows (fast trial run)")
    ap.add_argument("--top-brands", type=int, default=25)
    ap.add_argument("--sample-exchanges", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not args.csv.exists():
        print(f"ERROR: {args.csv} not found.", file=sys.stderr)
        return 2
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print("=" * 68)
    print("PHASE 0c — DATASET INSPECTION (facts only, no decisions)")
    print("=" * 68)
    if args.sample_rows:
        print(f"TRIAL RUN: first {args.sample_rows:,} rows only. "
              f"Numbers are NOT dataset-wide.")
    print(f"Reading {args.csv} ...")

    a = pass_a(args.csv, args.chunksize, args.sample_rows)
    acc, graph = a["acc"], a["graph"]

    print("\n--- 1. schema (discovered) ---")
    print(f"  columns: {acc['columns']}")
    print(f"  matches documented 7 columns: {acc.get('schema_matches_documentation')}")
    if acc.get("columns_missing_vs_doc"):
        print(f"  MISSING vs doc: {acc['columns_missing_vs_doc']}")
    if acc.get("columns_extra_vs_doc"):
        print(f"  extra vs doc  : {acc['columns_extra_vs_doc']}")
    print("\n  first row verbatim:")
    if acc["head_rows"]:
        for k, v in acc["head_rows"][0].items():
            shown = "<empty/NaN>" if (v is None or (isinstance(v, float) and v != v)) \
                else str(v)[:90]
            print(f"    {k:26s} = {shown!r}")

    print("\n--- 2. scale ---")
    print(f"  rows: {acc['n_rows']:,}")
    uniq_h, counts_h = np.unique(graph["text_hash"], return_counts=True)
    dup_text = int((counts_h > 1).sum())
    dup_tid = int(acc["n_rows"] - len(np.unique(graph["tweet_id"])))
    print(f"  duplicate tweet_id values        : {dup_tid:,}")
    print(f"  normalised-text values seen >1x  : {dup_text:,}")
    print(f"  nulls per column: {acc['nulls']}")

    print("\n--- 3. roles ---")
    n_in = int(graph["inbound"].sum())
    n_out = int((~graph["inbound"]).sum())
    n_out_authors = len(a["author_agg"])
    print(f"  inbound (customer->brand): {n_in:,} ({100*n_in/max(acc['n_rows'],1):.1f}%)")
    print(f"  outbound (brand->customer): {n_out:,}")
    print(f"  distinct outbound authors : {n_out_authors:,}")
    sample_authors = list(a["author_agg"].keys())[:10]
    print(f"  sample outbound author_ids: {sample_authors}")
    readable = sum(1 for s in a["author_agg"] if not str(s).isdigit())
    print(f"  outbound authors that are non-numeric (readable handles): "
          f"{readable:,}/{n_out_authors:,}")

    print("\n--- 5. thread reconstruction ---")
    th = reconstruct_threads(graph["tweet_id"], graph["parent_id"])
    for k, v in th["stats"].items():
        if isinstance(v, dict):
            continue
        print(f"  {k:34s}: {v:,}" if isinstance(v, int) else f"  {k:34s}: {v}")
    for label, key in (("conversation size", "conv_size_histogram"),
                       ("reply depth", "depth_histogram")):
        hist = th["stats"][key]
        bars = "  ".join(f"{k}:{v:,}" for k, v in
                         sorted(hist.items(), key=lambda kv: int(kv[0])))
        print(f"  {label + ' histogram (15+ clipped)':34s}: {bars}")

    print("\n--- 4. brand table (raw counts — NO SELECTION MADE) ---")
    brand_df, brand_meta = build_brand_table(
        graph, th["conversation_id"], th["depth"],
        a["author_agg"], a["author_code_map"])
    brand_csv = args.out / "brand_profile_raw.csv"
    brand_df.to_csv(brand_csv, index=False, encoding="utf-8")
    print(f"  conversations with a brand reply : "
          f"{brand_meta['n_conversations_with_a_brand_reply']:,}")
    print(f"  single-brand (attributable)      : "
          f"{brand_meta['n_conversations_unambiguous']:,}")
    print(f"  multi-brand (NOT attributed)     : "
          f"{brand_meta['n_conversations_multibrand']:,} "
          f"({brand_meta['pct_conversations_multibrand']}%)")
    cols = ["brand", "n_conversations_unambiguous",
            "n_inbound_in_unambiguous_convs",
            "pct_multiturn_unambiguous_depth_ge_2",
            "n_multibrand_convs_participated",
            "deflection_rate_strict_pct", "mean_reply_words"]
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(brand_df.head(args.top_brands)[cols].to_string(index=False))

    print("\n--- 6. text pathologies ---")
    for k, v in acc["text"].items():
        print(f"  {k:24s}: {v:,}  ({100*v/max(acc['n_rows'],1):.2f}%)")

    print("\n--- 7. time ---")
    print(f"  parse failures: {acc['time']['parse_failures']:,}")
    print(f"  range: {acc['time']['min']}  ->  {acc['time']['max']}")
    print(f"  ids non-numeric: {acc['ids']}")

    print("\n--- 8. resolution-proxy candidates ---")
    overall_defl = sum(v["n_deflect_strict"] for v in a["author_agg"].values())
    print(f"  brand replies matching a strict DM-deflection pattern: "
          f"{overall_defl:,} / {n_out:,} "
          f"({100*overall_defl/max(n_out,1):.1f}% of all outbound)")
    print("  NOTE: this is a regex heuristic. Validate it by reading the "
          "sampled exchanges before relying on it (BLUEPRINT §F.2).")

    # sample exchanges for manual reading
    conv = th["conversation_id"]
    inb = graph["inbound"]
    tid = graph["tweet_id"]
    parent = graph["parent_id"]
    reply_rows = np.flatnonzero((~inb) & (parent >= 0))
    pairs = []
    if reply_rows.size:
        pick = rng.choice(reply_rows,
                          size=min(args.sample_exchanges, reply_rows.size),
                          replace=False)
        inv = {v: k for k, v in a["author_code_map"].items()}
        for r in pick:
            pairs.append((int(parent[r]), int(tid[r]),
                          inv.get(int(graph["author_code"][r]), "?")))
    wanted = {p for pr in pairs for p in pr[:2]}
    samples_path = args.out / "samples_for_manual_reading.md"
    if wanted:
        dump_samples(args.csv, args.chunksize, wanted, samples_path, pairs)

    facts = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_csv": str(args.csv),
        "trial_run_rows": args.sample_rows,
        "schema": {
            "columns": acc["columns"],
            "matches_documentation": acc.get("schema_matches_documentation"),
            "missing_vs_doc": acc.get("columns_missing_vs_doc"),
            "extra_vs_doc": acc.get("columns_extra_vs_doc"),
            "dtypes_first_chunk": acc["dtypes_first_chunk"],
        },
        "scale": {
            "n_rows": acc["n_rows"],
            "duplicate_tweet_ids": dup_tid,
            "normalised_text_values_repeated": dup_text,
            "nulls_per_column": acc["nulls"],
        },
        "roles": {
            "n_inbound": n_in, "n_outbound": n_out,
            "n_distinct_outbound_authors": n_out_authors,
            "n_readable_outbound_authors": readable,
        },
        "threads": th["stats"],
        "brand_attribution": brand_meta,
        "text_pathologies": acc["text"],
        "text_pathology_definitions": {
            "n_contains_non_ascii_char": "at least one codepoint > 127 (NOT a language or script claim)",
            "n_predominantly_non_latin_script": (">50% of alphabetic characters outside Latin ranges, "
                                                 "min 3 letters. Script only; Latin-script Hinglish or "
                                                 "romanised text counts as Latin. Not language detection."),
        },
        "time": acc["time"],
        "ids": acc["ids"],
        "hashing": {
            "algorithm": "blake2b-64 (truncated to 63 bits)",
            "why": "deterministic across processes; Python hash() is PYTHONHASHSEED-randomised",
            "known_approximation": "collisions ~5e-7 over 3M strings may overstate duplicates by <1 row",
        },
        "deflection": {
            "strict_matches_outbound": overall_defl,
            "strict_rate_pct": round(100 * overall_defl / max(n_out, 1), 2),
            "method": "regex heuristic, unvalidated — see samples file",
        },
        "artifacts": {
            "brand_profile_csv": str(brand_csv),
            "samples_for_manual_reading": str(samples_path) if wanted else None,
        },
        "DISCLAIMER": ("Facts only. No brand selected, no intents defined, "
                       "no thresholds set. All such decisions remain open."),
    }
    facts_path = args.out / "dataset_facts.json"
    facts_path.write_text(json.dumps(facts, indent=2, default=str), encoding="utf-8")

    print("\n" + "-" * 68)
    print("NO BRAND SELECTED. NO INTENTS DEFINED. NO THRESHOLDS SET.")
    print(f"  facts   : {facts_path}")
    print(f"  brands  : {brand_csv}")
    if wanted:
        print(f"  samples : {samples_path}  <- read these by hand next")
    print("-" * 68)
    return 0


if __name__ == "__main__":
    sys.exit(main())
