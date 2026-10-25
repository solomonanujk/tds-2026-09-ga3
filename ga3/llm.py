"""Minimal OpenAI-compatible model client plus a content-addressed cache.

Configuration (all optional; the agents stay correct without a key by falling
back to their deterministic analysers):

  LLM_BASE_URL  default https://aipipe.org/openrouter/v1
  LLM_API_KEY   or AIPIPE_TOKEN / OPENAI_API_KEY
  LLM_MODEL     default openai/gpt-4.1-nano
  LLM_TIMEOUT   seconds per request (default 20)

Decisions are cached by canonical *content* fingerprint, never by batch,
message, evaluation or task id, so later Checks reuse the stable core corpora
without repeating model work.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional

from . import store
from .canon import digest

BASE_URL = os.environ.get("LLM_BASE_URL", "https://aipipe.org/openrouter/v1").rstrip("/")
API_KEY = (
    os.environ.get("LLM_API_KEY")
    or os.environ.get("AIPIPE_TOKEN")
    or os.environ.get("OPENAI_API_KEY")
    or ""
).strip()
MODEL = os.environ.get("LLM_MODEL", "openai/gpt-4.1-nano")
TIMEOUT = float(os.environ.get("LLM_TIMEOUT", "20"))

LAST_ERROR: Optional[str] = None
CALL_COUNT = 0


def available() -> bool:
    return bool(API_KEY)


def model_name() -> str:
    return MODEL


def _extract_json(text: str) -> Optional[Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
        text = re.sub(r"```\s*$", "", text).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    # Tolerate a prose wrapper around the object/array.
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = text.find(opener), text.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except Exception:
                continue
    return None


def chat_json(
    messages: List[Dict[str, str]],
    *,
    max_tokens: int = 4096,
    temperature: float = 0.0,
    timeout: Optional[float] = None,
    attempts: int = 2,
) -> Optional[Any]:
    """One chat completion, parsed as JSON. ``None`` on any failure."""
    global LAST_ERROR, CALL_COUNT
    if not API_KEY:
        LAST_ERROR = "no api key configured"
        return None
    body = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    deadline = time.time() + (timeout or TIMEOUT)
    for attempt in range(attempts):
        remaining = deadline - time.time()
        if remaining <= 1:
            break
        request = urllib.request.Request(
            f"{BASE_URL}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            CALL_COUNT += 1
            with urllib.request.urlopen(request, timeout=remaining) as response:
                payload = json.loads(response.read().decode("utf-8"))
            content = payload["choices"][0]["message"]["content"]
            parsed = _extract_json(content)
            if parsed is not None:
                LAST_ERROR = None
                return parsed
            LAST_ERROR = "model returned unparseable content"
        except urllib.error.HTTPError as exc:  # pragma: no cover - network
            detail = exc.read()[:400].decode("utf-8", "replace")
            LAST_ERROR = f"http {exc.code}: {detail}"
            if exc.code in (400, 401, 403, 404) and "response_format" in detail:
                body.pop("response_format", None)
                continue
            if exc.code in (401, 403, 404):
                break
        except Exception as exc:  # pragma: no cover - network
            LAST_ERROR = f"{type(exc).__name__}: {exc}"
        time.sleep(0.4 * (attempt + 1))
    return None


def cached_decision(
    namespace: str,
    fingerprint: Any,
    producer: Callable[[], Any],
    *,
    wait: float = 30.0,
) -> Any:
    """Compute ``producer()`` once per content fingerprint, then reuse it.

    Concurrent callers for the same fingerprint block on the winner instead of
    issuing a second model call.
    """
    key = digest(fingerprint)
    hit = store.get(namespace, key)
    if hit is not None:
        return hit
    won, existing = store.claim(namespace, key)
    if not won:
        if existing is not None:
            return existing
        waited = store.await_ready(namespace, key, timeout=wait)
        if waited is not None:
            return waited
        return producer()
    try:
        value = producer()
    except BaseException:
        store.release(namespace, key)
        raise
    store.put(namespace, key, value)
    return value
