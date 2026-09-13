#!/usr/bin/env python3
"""
Safely write the Gemini API key into .env — Windows-friendly.

Why this exists instead of a shell one-liner or Notepad:

  * The key is never part of a command, so it never lands in PowerShell history.
  * Input is HIDDEN while you type/paste it, so it never appears on screen
    and never ends up in a screenshot or screen recording.
  * .gitignore is written FIRST, so .env is protected before it exists —
    there is no window in which an unprotected key file is on disk.
  * The file is written as plain ASCII with no byte-order mark. Notepad's
    default encoding prepends an invisible marker that silently corrupts the
    first line, which is a genuinely nasty bug to chase later.
  * Confirmation is masked: it reports the length and a format check, and
    never prints any part of the key.

    python scripts/10_setup_key.py
"""

from __future__ import annotations

import getpass
import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):          # Windows cp1252 guard
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

GITIGNORE_ENTRIES = [
    ".env",
    ".env.*",
    "data/raw/",
    "__pycache__/",
    "*.pyc",
    ".ipynb_checkpoints/",
]

PLACEHOLDERS = {"", "your_key_here", "paste_your_key_here", "xxx",
                "gemini_api_key", "none", "null"}


def ensure_gitignore(root: Path) -> list[str]:
    """Create or top up .gitignore. Never rewrites lines that already exist."""
    path = root / ".gitignore"
    existing = []
    if path.exists():
        existing = [ln.strip() for ln in
                    path.read_text(encoding="utf-8", errors="replace").splitlines()]
    missing = [e for e in GITIGNORE_ENTRIES if e not in existing]
    if missing:
        with open(path, "a", encoding="utf-8", newline="\n") as fh:
            if existing and existing[-1] != "":
                fh.write("\n")
            fh.write("# added by scripts/10_setup_key.py\n")
            for entry in missing:
                fh.write(entry + "\n")
    return missing


def read_key() -> str | None:
    print("\nPaste your Gemini API key, then press Enter.")
    print("Nothing will appear as you paste — that is deliberate. Paste with a")
    print("right-click, then press Enter even though the line still looks empty.\n")
    try:
        key = getpass.getpass("Key (hidden): ")
    except Exception:
        print("  ! Hidden input is not supported by this terminal.")
        print("  ! Falling back to visible input — your key WILL show on screen.")
        ans = input("  Continue anyway? [y/N]: ").strip().lower()
        if ans != "y":
            return None
        key = input("Key (visible): ")
    return key.strip().strip('"').strip("'")


def validate(key: str) -> list[str]:
    problems = []
    if key.lower() in PLACEHOLDERS:
        problems.append("that looks like placeholder text, not a real key")
    if len(key) < 20:
        problems.append(f"too short ({len(key)} characters) — keys are much longer")
    if any(c.isspace() for c in key):
        problems.append("contains a space or line break — the paste may have broken")
    if key.upper().startswith("GEMINI_API_KEY"):
        problems.append("includes the 'GEMINI_API_KEY=' prefix — paste only the key itself")
    if not key.isascii():
        problems.append("contains non-ASCII characters — likely a copy/paste artefact")
    return problems


def main() -> int:
    root = Path.cwd()
    print("=" * 62)
    print("SECURE KEY SETUP")
    print("=" * 62)
    print(f"Project folder: {root}")

    if not (root / "scripts").is_dir():
        print("\n! You do not appear to be in the project folder.")
        print("! Expected to find a 'scripts' folder here.")
        print("! Open the project folder in File Explorer, type 'powershell' in")
        print("! the address bar, and run this again.")
        return 2

    added = ensure_gitignore(root)
    print(f"\n.gitignore : {'created/updated — added ' + ', '.join(added) if added else 'already covers .env'}")

    env_path = root / ".env"
    if env_path.exists():
        print(f"\nAn existing .env was found. It will be REPLACED.")
        if input("Continue? [y/N]: ").strip().lower() != "y":
            print("Cancelled. Nothing changed.")
            return 1

    key = read_key()
    if key is None:
        print("Cancelled. Nothing changed.")
        return 1

    problems = validate(key)
    if problems:
        print("\nThat does not look like a valid key:")
        for p in problems:
            print(f"  - {p}")
        print("\nNothing was written. Copy the key again from AI Studio and retry.")
        return 1

    # plain ASCII, no BOM, unix newline
    with open(env_path, "w", encoding="ascii", newline="\n") as fh:
        fh.write(f"GEMINI_API_KEY={key}\n")

    print("\n" + "-" * 62)
    print("DONE")
    print(f"  wrote      : {env_path}")
    print(f"  key length : {len(key)} characters")
    if key.startswith("AQ."):
        fmt = "Gemini auth key (current AI Studio format) — correct"
    elif key.startswith("AIza"):
        fmt = "legacy Standard key — works only if your project still accepts them"
    elif key.startswith(("ya29.", "eyJ")):
        fmt = "WARNING: looks like a login token, not an API key"
    else:
        fmt = "unrecognised prefix — verify it came from AI Studio's API keys table"
    print(f"  format     : {fmt}")
    print("  the key itself was not printed, and is not in your shell history")
    print("-" * 62)
    return 0


if __name__ == "__main__":
    sys.exit(main())
