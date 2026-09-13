"""Intent taxonomy: load, save, render.

The taxonomy is DISCOVERED from real clustered messages on the DEV split (never
gold_pool — that is leakage path L5) and proposed by an LLM from those clusters.
It is LLM-proposed, human-reviewable, and the file records which. It is not
presented anywhere as hand-crafted.
"""

from __future__ import annotations

import json
from pathlib import Path

OTHER = "other"


def load(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path: Path, obj: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False),
                          encoding="utf-8")


def names(tax: dict, include_other: bool = True) -> list[str]:
    ns = [i["name"] for i in tax["intents"]]
    return ns + [OTHER] if include_other else ns


def auto_allowed(tax: dict) -> list[str]:
    """Intents answerable from published policy or general troubleshooting.
    Account-specific intents are never auto-handled — that is the allowlist."""
    return [i["name"] for i in tax["intents"] if not i.get("account_specific", True)]


def render(tax: dict) -> str:
    lines = []
    for i in tax["intents"]:
        lines.append(f"- {i['name']}: {i['definition']}")
        if i.get("includes"):
            lines.append(f"    includes: {i['includes']}")
        if i.get("excludes"):
            lines.append(f"    excludes: {i['excludes']}")
    return "\n".join(lines)


def render_for_human(tax: dict) -> str:
    """Compact reference shown in the labelling CLI."""
    out = []
    for n, i in enumerate(tax["intents"], 1):
        flag = "ACCOUNT" if i.get("account_specific") else "policy"
        out.append(f"  [{n}] {i['name']:<28} ({flag})  {i['definition'][:70]}")
    out.append(f"  [0] {OTHER:<28} (ACCOUNT)  none of the above / unclear / off-topic")
    return "\n".join(out)
