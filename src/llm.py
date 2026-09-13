"""Gemini client with an on-disk response cache.

Why the cache matters, beyond speed:
  * the evaluator reproduces headline numbers with NO API key and NO quota
  * results are stable across runs despite model nondeterminism
  * a rate-limit interruption resumes instead of restarting

Cache key = sha256(model | prompt | generation params). Responses are stored as
JSON under cache/llm/. They are RECORDED REAL RESPONSES, never hand-written.

The API key is read from .env into memory and is never logged, printed, or
written to the cache.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path

_CLIENT = None
_MOCK = None          # test hook; see tests/
_STATS = {"calls": 0, "cache_hits": 0, "errors": 0, "retries": 0,
          "prompt_tokens": 0, "output_tokens": 0, "throttle_seconds": 0.0}

# Measured free-tier ceiling for gemini-3.5-flash-lite. Override with HIVER_RPM.
DEFAULT_RPM = 15
_CALL_TIMES: list[float] = []


def _throttle() -> None:
    """Keep under the requests-per-minute ceiling by waiting, rather than
    letting the API return 429 and burning a retry. Cache hits do not count —
    they never touch the network."""
    rpm = int(os.environ.get("HIVER_RPM", DEFAULT_RPM))
    if rpm <= 0:
        return
    now = time.monotonic()
    _CALL_TIMES[:] = [t for t in _CALL_TIMES if now - t < 60.0]
    if len(_CALL_TIMES) >= rpm:
        wait = 60.0 - (now - _CALL_TIMES[0]) + 0.25
        if wait > 0:
            _STATS["throttle_seconds"] += wait
            print(f"    [rate limit] {rpm} rpm reached, waiting {wait:.0f}s",
                  file=sys.stderr)
            time.sleep(wait)
        now = time.monotonic()
        _CALL_TIMES[:] = [t for t in _CALL_TIMES if now - t < 60.0]
    _CALL_TIMES.append(time.monotonic())


MOCK_MARKER = "__MOCK_LLM_OUTPUT__"


def mock_mode() -> bool:
    """Offline pipeline testing only. Set HIVER_MOCK_LLM=1.

    Everything produced in this mode is SYNTHETIC and must never be reported as
    a result: metrics.json records MOCK_MODE and the audit fails CRITICAL on it.
    """
    return os.environ.get("HIVER_MOCK_LLM") == "1"


def _builtin_mock(prompt: str, model: str) -> str:
    """Deterministic, structurally valid responses for each prompt type, so the
    pipeline's plumbing can be exercised without an API key."""
    h = int(hashlib.sha256(prompt.encode()).hexdigest()[:8], 16)
    if "Propose a SMALL taxonomy" in prompt:
        names = ["playback_issue", "billing_or_charges", "account_access",
                 "content_availability", "plan_or_subscription",
                 "device_support", "app_bug"]
        return json.dumps({"intents": [
            {"name": n, "definition": f"mock definition for {n}",
             "includes": "mock", "excludes": "mock",
             "account_specific": i in (1, 2, 4),
             "example_messages": ["mock example"]} for i, n in enumerate(names)]})
    if '"results"' in prompt:
        ids = re.findall(r"^(\d+):", prompt, re.M)
        pool = re.findall(r"^- (\w+):", prompt, re.M) or ["other"]
        return json.dumps({"results": [
            {"id": int(i), "intent": pool[(h + int(i)) % len(pool)],
             "confidence": 0.5 + ((h + int(i)) % 50) / 100,
             "runner_up": pool[0], "multi_intent": False} for i in ids]})
    if '"intent"' in prompt and '"sufficient_evidence"' in prompt:
        pool = re.findall(r"^- (\w+):", prompt, re.M) or ["other"]
        has_ev = "[E1]" in prompt
        return json.dumps({
            "intent": pool[h % len(pool)], "confidence": 0.4 + (h % 60) / 100,
            "runner_up": pool[0], "multi_intent": False,
            "reply": f"{MOCK_MARKER} mock drafted reply.",
            "sufficient_evidence": bool(has_ev and h % 4),
            "evidence_used": [1] if has_ev else [],
            "claims": [{"claim": "mock", "supported_by": [1] if has_ev else []}]})
    if "Return ONLY JSON:\n{\"intent\"" in prompt or '{"intent":' in prompt:
        pool = re.findall(r"^- (\w+):", prompt, re.M) or ["other"]
        return json.dumps({"intent": pool[h % len(pool)],
                           "confidence": 0.4 + (h % 60) / 100,
                           "runner_up": pool[0], "multi_intent": False})
    if '"sufficient_evidence"' in prompt:
        has_ev = "[E1]" in prompt
        return json.dumps({
            "reply": f"{MOCK_MARKER} mock drafted reply.",
            "sufficient_evidence": bool(has_ev and h % 4),
            "evidence_used": [1] if has_ev else [],
            "claims": [{"claim": "mock claim", "supported_by": [1] if has_ev else []}]})
    if '"relevance"' in prompt:
        def s(o):
            return 2 + (h >> o) % 4
        return json.dumps({
            "relevance": {"why": "mock", "score": s(0)},
            "groundedness": {"why": "mock", "score": s(3)},
            "helpfulness": {"why": "mock", "score": s(6)},
            "tone_fit": {"why": "mock", "score": s(9)},
            "completeness": {"why": "mock", "score": s(12)},
            "unsupported_claims": {"why": "mock", "flag": h % 5 == 0},
            "escalation_appropriate": {"why": "mock", "flag": h % 3 != 0}})
    return json.dumps({MOCK_MARKER: True})


def set_mock_responder(fn):
    """Install a deterministic stand-in for the API (offline testing only).
    Never used when a real key is present and HIVER_MOCK_LLM is unset."""
    global _MOCK
    _MOCK = fn


def stats() -> dict:
    return dict(_STATS)


def load_dotenv(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _client():
    global _CLIENT
    if _CLIENT is None:
        from google import genai  # imported lazily so offline tests need no SDK
        load_dotenv()
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("No GEMINI_API_KEY found. Run scripts/10_setup_key.py")
        _CLIENT = genai.Client(api_key=key)
    return _CLIENT


def _cache_key(model: str, prompt: str, params: dict) -> str:
    blob = json.dumps({"m": model, "p": prompt, "c": params},
                      sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def generate(prompt: str, model: str, cache_dir: Path,
             temperature: float = 0.0, max_output_tokens: int = 1024,
             json_mode: bool = True, max_retries: int = 4,
             use_cache: bool = True) -> str:
    """Return raw model text. Cached by content; retried with backoff on
    rate limits and transient server errors."""
    params = {"t": temperature, "mo": max_output_tokens, "j": json_mode}
    key = _cache_key(model, prompt, params)
    path = Path(cache_dir) / f"{key}.json"

    if use_cache and path.exists():
        _STATS["cache_hits"] += 1
        return json.loads(path.read_text(encoding="utf-8"))["response"]

    if _MOCK is not None or mock_mode():
        text = (_MOCK or _builtin_mock)(prompt, model)
    else:
        from google.genai import types
        cfg = {"temperature": temperature, "max_output_tokens": max_output_tokens}
        if json_mode:
            cfg["response_mime_type"] = "application/json"
        text, last_err = None, None
        for attempt in range(max_retries):
            try:
                _throttle()
                resp = _client().models.generate_content(
                    model=model, contents=prompt,
                    config=types.GenerateContentConfig(**cfg))
                text = (getattr(resp, "text", None) or "").strip()
                u = getattr(resp, "usage_metadata", None)
                _STATS["prompt_tokens"] += getattr(u, "prompt_token_count", 0) or 0
                _STATS["output_tokens"] += getattr(u, "candidates_token_count", 0) or 0
                break
            except Exception as exc:                      # noqa: BLE001
                last_err = f"{type(exc).__name__}: {exc}"
                transient = any(s in last_err.lower() for s in
                                ("429", "resource_exhausted", "quota", "503",
                                 "500", "unavailable", "deadline", "timeout"))
                if not transient or attempt == max_retries - 1:
                    _STATS["errors"] += 1
                    raise
                _STATS["retries"] += 1
                sleep = (2 ** attempt) + random.random()
                print(f"    rate limited, retrying in {sleep:.1f}s "
                      f"({attempt+1}/{max_retries})", file=sys.stderr)
                time.sleep(sleep)
        if text is None:
            raise RuntimeError(last_err or "generation failed")

    _STATS["calls"] += 1
    if use_cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"model": model, "params": params,
                                    "prompt_sha256": hashlib.sha256(
                                        prompt.encode()).hexdigest(),
                                    "response": text}, ensure_ascii=False),
                        encoding="utf-8")
    return text


_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.S)


def parse_json(text: str):
    """Best-effort JSON extraction. Returns None on failure — callers must
    count failures rather than silently substituting a default."""
    if not text:
        return None
    for candidate in (text, (_FENCE.search(text) or [None, None])[1] if _FENCE.search(text) else None):
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except Exception:
            pass
    # last resort: first balanced {...} or [...]
    for opener, closer in (("{", "}"), ("[", "]")):
        i, j = text.find(opener), text.rfind(closer)
        if 0 <= i < j:
            try:
                return json.loads(text[i:j + 1])
            except Exception:
                pass
    return None


def generate_json(prompt: str, model: str, cache_dir: Path, **kw):
    """Generate and parse. Returns (obj_or_None, raw_text)."""
    raw = generate(prompt, model, cache_dir, json_mode=True, **kw)
    return parse_json(raw), raw
