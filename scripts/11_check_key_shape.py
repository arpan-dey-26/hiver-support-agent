#!/usr/bin/env python3
"""
Diagnose what is in .env WITHOUT revealing it.

Prints only structural facts — length, a 4-character prefix (which is identical
for every key of a given type and so reveals nothing), delimiter counts, and a
best-guess classification. The key value is never printed, logged or returned.

    python scripts/11_check_key_shape.py
"""

from __future__ import annotations

import sys
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def classify(k: str) -> tuple[str, str]:
    if k.startswith("AIza") and 35 <= len(k) <= 45:
        return ("Google API key", "This is the right thing.")
    if k.startswith("ya29."):
        return ("OAuth2 ACCESS TOKEN",
                "This is a short-lived login token, not an API key. It expires in ~1 hour "
                "and will not work. Copy the API key from aistudio.google.com/apikey instead.")
    if k.count(".") == 2 and k.startswith("eyJ"):
        return ("JWT / identity token",
                "This is a signed login token, not an API key. Copy the API key from "
                "aistudio.google.com/apikey instead.")
    if k.startswith("http://") or k.startswith("https://"):
        return ("a URL", "A web address was pasted instead of the key.")
    if k.count("AIza") > 1:
        return ("MULTIPLE keys concatenated",
                "Looks like the key was pasted more than once. Re-run the setup and paste once.")
    if "-----BEGIN" in k:
        return ("a private key / PEM block",
                "This is a service-account credential, not a Gemini API key.")
    if len(k) > 100:
        return ("unknown, but far too long",
                "Gemini API keys are about 39 characters. Something else was copied.")
    return ("unknown", "Does not match any known credential shape.")


def main() -> int:
    path = Path(".env")
    if not path.exists():
        print("No .env found. Are you in the project folder?")
        return 2

    raw = path.read_bytes()
    text = raw.decode("utf-8", errors="replace")
    lines = [ln for ln in text.splitlines() if ln.strip()]

    print("=" * 60)
    print(".env STRUCTURE CHECK  (no secret values are shown)")
    print("=" * 60)
    print(f"  file size        : {len(raw)} bytes")
    print(f"  non-empty lines  : {len(lines)}")
    has_bom = raw.startswith(b"\xef\xbb\xbf")
    print(f"  starts with BOM  : {has_bom}")

    key = None
    for ln in lines:
        if ln.startswith("GEMINI_API_KEY="):
            key = ln.split("=", 1)[1].strip()
            break
    if key is None:
        print("\n  ! No GEMINI_API_KEY= line found.")
        print(f"  ! First line begins: {lines[0][:14]!r}..." if lines else "  ! File is empty.")
        return 1

    kind, advice = classify(key)
    print(f"\n  key length       : {len(key)} characters")
    print(f"  first 4 chars    : {key[:4]!r}   (identical for all keys of this type)")
    print(f"  contains '.'     : {key.count('.')}")
    print(f"  contains '/'     : {key.count('/')}")
    print(f"  contains ':'     : {key.count(':')}")
    print(f"  all alphanumeric : {key.replace('-', '').replace('_', '').isalnum()}")
    print(f"\n  LOOKS LIKE       : {kind}")
    print(f"  WHAT TO DO       : {advice}")
    print("=" * 60)
    return 0 if kind == "Google API key" else 1


if __name__ == "__main__":
    sys.exit(main())
