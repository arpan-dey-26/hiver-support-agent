#!/usr/bin/env python3
"""
Phase 0a — environment verification.

Reports facts about the machine this project will run on. Makes no changes and
installs nothing. Writes results/phase0/env_report.json and prints a summary.

Usage:
    python scripts/00_check_env.py
    python scripts/00_check_env.py --csv data/raw/twcs.csv
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):      # Windows cp1252 console guard
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

MIN_PY = (3, 10)

# created by this script so the workflow never needs `mkdir -p` (Unix-only)
PROJECT_DIRS = ["data/raw", "data/processed", "data/gold",
                "results/phase0", "cache/llm"]

# (module name, pip name, required?)
PACKAGES = [
    ("pandas", "pandas", True),
    ("numpy", "numpy", True),
    ("sklearn", "scikit-learn", True),
    ("scipy", "scipy", True),
    ("yaml", "PyYAML", True),
    ("pyarrow", "pyarrow", True),
    ("rank_bm25", "rank_bm25", True),
    ("tqdm", "tqdm", False),
    ("matplotlib", "matplotlib", False),
    ("google.genai", "google-genai", True),
    # contingent only — see BLUEPRINT §F.1. Absence is NOT a failure.
    ("sentence_transformers", "sentence-transformers", False),
]


def check_python() -> dict:
    ok = sys.version_info[:2] >= MIN_PY
    return {
        "version": platform.python_version(),
        "executable": sys.executable,
        "meets_minimum": ok,
        "minimum_required": ".".join(map(str, MIN_PY)),
    }


def check_packages() -> dict:
    found, missing_required, missing_optional = {}, [], []
    for mod, pip_name, required in PACKAGES:
        try:
            m = importlib.import_module(mod)
            found[mod] = getattr(m, "__version__", "unknown")
        except Exception:
            (missing_required if required else missing_optional).append(pip_name)
    return {
        "installed": found,
        "missing_required": missing_required,
        "missing_optional": missing_optional,
    }


def check_disk(where: Path) -> dict:
    target = where
    while not target.exists() and target != target.parent:
        target = target.parent
    total, used, free = shutil.disk_usage(target)
    gb = 1024 ** 3
    return {
        "path_checked": str(target),
        "total_gb": round(total / gb, 1),
        "free_gb": round(free / gb, 1),
        # raw csv ~0.5GB + parquet + caches; 5GB is a comfortable floor
        "sufficient_for_project": free / gb >= 5.0,
    }


def check_encoding() -> dict:
    """Windows consoles and cp1252 defaults silently mangle tweet text. Prove
    a UTF-8 round-trip works before we process 2.8M rows of emoji."""
    probe = "café — naïve 🙂 ✈️ ①"
    result = {
        "stdout_encoding": getattr(sys.stdout, "encoding", None),
        "preferred_encoding": __import__("locale").getpreferredencoding(False),
        "filesystem_encoding": sys.getfilesystemencoding(),
    }
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False,
                                         encoding="utf-8") as fh:
            fh.write(probe)
            path = fh.name
        with open(path, encoding="utf-8") as fh:
            result["utf8_roundtrip_ok"] = fh.read() == probe
        os.unlink(path)
    except Exception as exc:
        result["utf8_roundtrip_ok"] = False
        result["error"] = repr(exc)
    return result


def check_dataset(csv_path: Path | None) -> dict:
    if csv_path is None:
        return {"checked": False, "note": "no --csv path supplied"}
    info = {"checked": True, "path": str(csv_path), "exists": csv_path.exists()}
    if csv_path.exists():
        size = csv_path.stat().st_size
        info["size_bytes"] = size
        info["size_mb"] = round(size / (1024 ** 2), 1)
        try:
            with open(csv_path, encoding="utf-8", errors="replace") as fh:
                info["header_line"] = fh.readline().rstrip("\n")
        except Exception as exc:
            info["header_error"] = repr(exc)
    return info


def check_api_key() -> dict:
    """Reports only whether a key is present. NEVER prints or stores the value."""
    env_file = Path(".env")
    in_env_file = False
    if env_file.exists():
        try:
            for line in env_file.read_text(encoding="utf-8").splitlines():
                key = line.split("=", 1)[0].strip()
                if key in ("GEMINI_API_KEY", "GOOGLE_API_KEY") and "=" in line:
                    in_env_file = bool(line.split("=", 1)[1].strip())
        except Exception:
            pass
    return {
        "dotenv_file_present": env_file.exists(),
        "key_present_in_dotenv": in_env_file,
        "key_present_in_environment": bool(
            os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        ),
        "note": "value never read, printed or stored by this script",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=None,
                    help="path to twcs.csv, if already downloaded")
    ap.add_argument("--out", type=Path, default=Path("results/phase0"))
    ap.add_argument("--no-make-dirs", action="store_true",
                    help="skip creating the project directory tree")
    args = ap.parse_args()

    created = []
    if not args.no_make_dirs:
        for d in PROJECT_DIRS:
            p = Path(d)
            if not p.exists():
                p.mkdir(parents=True, exist_ok=True)
                created.append(str(p))

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "directories_created": created,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "python": check_python(),
        "packages": check_packages(),
        "disk": check_disk(args.csv.parent if args.csv else Path.cwd()),
        "encoding": check_encoding(),
        "dataset": check_dataset(args.csv),
        "api_key": check_api_key(),
    }

    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / "env_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    p, pk, d, e = (report["python"], report["packages"],
                   report["disk"], report["encoding"])
    print("=" * 68)
    print("PHASE 0a — ENVIRONMENT REPORT")
    print("=" * 68)
    print(f"Platform     : {report['platform']['system']} "
          f"{report['platform']['release']} ({report['platform']['machine']})")
    print(f"Python       : {p['version']}  "
          f"[{'OK' if p['meets_minimum'] else 'TOO OLD, need >= ' + p['minimum_required']}]")
    print(f"Disk free    : {d['free_gb']} GB at {d['path_checked']}  "
          f"[{'OK' if d['sufficient_for_project'] else 'LOW — need ~5 GB'}]")
    print(f"UTF-8 I/O    : {'OK' if e.get('utf8_roundtrip_ok') else 'FAILED'}  "
          f"(stdout={e['stdout_encoding']}, locale={e['preferred_encoding']})")
    if created:
        print(f"Created dirs : {', '.join(created)}")

    print("\nPackages installed:")
    for mod, ver in sorted(pk["installed"].items()):
        print(f"  {mod:24s} {ver}")
    if pk["missing_required"]:
        print("\n  MISSING (required):")
        for name in pk["missing_required"]:
            print(f"    - {name}")
        print(f"\n    pip install {' '.join(pk['missing_required'])}")
    if pk["missing_optional"]:
        print("\n  missing (optional / contingent only):")
        for name in pk["missing_optional"]:
            print(f"    - {name}")

    ds = report["dataset"]
    print("\nDataset:")
    if not ds["checked"]:
        print("  not checked (pass --csv path/to/twcs.csv)")
    elif ds["exists"]:
        print(f"  FOUND  {ds['path']}  ({ds['size_mb']} MB)")
        print(f"  header: {ds.get('header_line', '')[:120]}")
    else:
        print(f"  NOT FOUND at {ds['path']}")

    ak = report["api_key"]
    print("\nGemini API key:")
    print(f"  .env present        : {ak['dotenv_file_present']}")
    print(f"  key set in .env     : {ak['key_present_in_dotenv']}")
    print(f"  key set in env vars : {ak['key_present_in_environment']}")

    blocking = []
    if not p["meets_minimum"]:
        blocking.append(f"Python >= {p['minimum_required']} required")
    if pk["missing_required"]:
        blocking.append(f"missing required packages: {', '.join(pk['missing_required'])}")
    if not e.get("utf8_roundtrip_ok"):
        blocking.append("UTF-8 round-trip failed")
    if not d["sufficient_for_project"]:
        blocking.append("less than 5 GB free disk")

    print("\n" + "-" * 68)
    if blocking:
        print("BLOCKING ISSUES:")
        for b in blocking:
            print(f"  - {b}")
    else:
        print("No blocking environment issues found.")
    print(f"Written: {out_path}")
    print("-" * 68)
    return 1 if blocking else 0


if __name__ == "__main__":
    sys.exit(main())
