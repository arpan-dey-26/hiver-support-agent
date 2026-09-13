#!/usr/bin/env python3
"""
Build the submission ZIP.

Packaging only — reads the project, writes one archive, changes nothing else.

It refuses to produce an archive containing anything key-shaped. The scan runs
over the exact bytes destined for the ZIP, not over the source tree, so a file
cannot pass the check and then be included in a different form.

    python scripts/90_package_submission.py
    python scripts/90_package_submission.py --include-docs
"""

from __future__ import annotations

import argparse
import re
import sys
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

# Exactly the requested contents.
INCLUDE_FILES = ["README.md", "REPORT.md", "DECISIONS.md", "config.yaml",
                 "requirements.txt", "run_all.py", ".gitignore"]
INCLUDE_DIRS = ["src", "scripts", "tests", "data/gold", "data/processed/splits",
                "results", "cache/llm"]
INCLUDE_EXTRA = ["data/processed/taxonomy_v1.json"]
OPTIONAL_DOCS = ["docs/BLUEPRINT.md", "PHASE0_README.md"]

EXCLUDE_NAMES = {".env", "twcs.csv"}
EXCLUDE_DIR_PARTS = {"__pycache__", ".git", ".ipynb_checkpoints", "raw"}
EXCLUDE_SUFFIX = {".pyc", ".pyo", ".stale", ".tmp", ".bak", ".swp"}

# Google credential shapes: current auth keys, legacy keys, OAuth tokens, JWTs.
SECRET_PATTERNS = [
    ("gemini auth key", re.compile(rb"AQ\.[A-Za-z0-9_\-]{40,}")),
    ("legacy google api key", re.compile(rb"AIza[A-Za-z0-9_\-]{30,}")),
    ("oauth access token", re.compile(rb"ya29\.[A-Za-z0-9_\-]{40,}")),
    ("jwt", re.compile(rb"eyJ[A-Za-z0-9_\-]{20,}\.[A-Za-z0-9_\-]{20,}\.")),
    ("assigned api key var", re.compile(rb"(?:GEMINI|GOOGLE)_API_KEY\s*[=:]\s*['\"]?[A-Za-z0-9_\-.]{20,}")),
    ("private key block", re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
]
TEXT_SUFFIX = {".py", ".md", ".yaml", ".yml", ".json", ".csv", ".txt", ".cfg", ".ini", ""}


def excluded(rel: Path) -> str | None:
    if rel.name in EXCLUDE_NAMES:
        return f"excluded name ({rel.name})"
    if rel.suffix in EXCLUDE_SUFFIX:
        return f"excluded suffix ({rel.suffix})"
    for part in rel.parts:
        if part in EXCLUDE_DIR_PARTS:
            return f"excluded directory ({part})"
    return None


def collect(include_docs: bool) -> tuple[list[Path], list[tuple[str, str]]]:
    keep, skipped = [], []

    def add_file(p: Path):
        rel = p.relative_to(ROOT)
        why = excluded(rel)
        if why:
            skipped.append((str(rel).replace("\\", "/"), why))
        else:
            keep.append(p)

    for name in INCLUDE_FILES + INCLUDE_EXTRA + (OPTIONAL_DOCS if include_docs else []):
        p = ROOT / name
        if p.is_file():
            add_file(p)
    for d in INCLUDE_DIRS:
        base = ROOT / d
        if not base.is_dir():
            continue
        for p in sorted(base.rglob("*")):
            if p.is_file():
                add_file(p)
    # de-duplicate while preserving order
    seen, uniq = set(), []
    for p in keep:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq, skipped


def scan(paths: list[Path]) -> list[tuple[str, str]]:
    """Scan the exact bytes headed for the archive."""
    hits = []
    for p in paths:
        try:
            data = p.read_bytes()
        except Exception:
            continue
        if p.suffix.lower() not in TEXT_SUFFIX and len(data) > 2_000_000:
            continue
        for label, pat in SECRET_PATTERNS:
            if pat.search(data):
                hits.append((str(p.relative_to(ROOT)).replace("\\", "/"), label))
                break
    return hits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-docs", action="store_true",
                    help="also include docs/BLUEPRINT.md and PHASE0_README.md")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    print("=" * 72)
    print("SUBMISSION PACKAGING")
    print("=" * 72)

    files, skipped = collect(args.include_docs)
    if not files:
        print("ERROR: nothing collected — wrong working directory?", file=sys.stderr)
        return 2

    print(f"\ncollected {len(files)} files, skipped {len(skipped)}")
    groups: dict[str, list[int]] = {}
    for p in files:
        rel = p.relative_to(ROOT)
        key = rel.parts[0] if len(rel.parts) > 1 else "(root)"
        g = groups.setdefault(key, [0, 0])
        g[0] += 1
        g[1] += p.stat().st_size
    for k in sorted(groups):
        n, b = groups[k]
        print(f"  {k:<26} {n:>5} files   {b/1e6:>8.2f} MB")

    print("\nexclusions applied:")
    seen_reasons: dict[str, int] = {}
    for _, why in skipped:
        seen_reasons[why] = seen_reasons.get(why, 0) + 1
    for why, n in sorted(seen_reasons.items()):
        print(f"  {why:<40} {n}")
    for must in (".env", "data/raw/twcs.csv"):
        inside = any(str(p.relative_to(ROOT)).replace("\\", "/") == must for p in files)
        print(f"  {must:<40} {'IN ARCHIVE — ABORT' if inside else 'not in archive'}")
        if inside:
            return 1

    print("\nsecret scan over archive bytes...")
    hits = scan(files)
    if hits:
        print("  ABORTED — credential-shaped content found:")
        for rel, label in hits[:20]:
            print(f"    {rel}  ({label})")
        print("\n  No archive was written.")
        return 1
    print(f"  clean — {len(files)} files scanned, 0 matches")

    stamp = datetime.now().strftime("%Y%m%d")
    out = args.out or (ROOT / f"hiver-support-agent-submission-{stamp}.zip")
    if out.exists():
        out.unlink()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in files:
            z.write(p, arcname=str(Path("hiver-support-agent") /
                                   p.relative_to(ROOT)).replace("\\", "/"))

    with zipfile.ZipFile(out) as z:
        names = z.namelist()
        bad = [n for n in names
               if n.endswith((".pyc", ".stale", ".tmp"))
               or "__pycache__" in n or n.endswith("/.env")
               or n.endswith("twcs.csv")]
        total_raw = sum(i.file_size for i in z.infolist())
    print("\narchive verification:")
    print(f"  entries        : {len(names)}")
    print(f"  uncompressed   : {total_raw/1e6:.2f} MB")
    print(f"  forbidden      : {len(bad)}" + (f"  {bad[:5]}" if bad else "  (none)"))
    if bad:
        out.unlink()
        print("  archive deleted.")
        return 1

    size = out.stat().st_size
    print("\n" + "-" * 72)
    print(f"  ZIP PATH : {out}")
    print(f"  ZIP SIZE : {size:,} bytes ({size/1e6:.2f} MB)")
    print("-" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
