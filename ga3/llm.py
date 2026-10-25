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
MODEL = os.environ.get("LLM_MODEL", "gpt-5-nano")
# Provider catalogues change under us: a deprecated or unknown model is
# retried against the next candidate instead of failing the request.
FALLBACK_MODELS = [
    name.strip()
    for name in os.environ.get(
        "LLM_MODEL_FALLBACKS", "gpt-4o-mini,gpt-4.1-mini,gpt-5-mini,gpt-5-nano,gpt-4o"
    ).split(",")
    if name.strip()
]
ACTIVE_MODEL = MODEL
TIMEOUT = float(os.environ.get("LLM_TIMEOUT", "20"))
MAX_OUTPUT_TOKENS = int(os.environ.get("LLM_MAX_OUTPUT_TOKENS", "3000"))
USER_AGENT = os.environ.get(
    "LLM_USER_AGENT", "tds-ga3-agent/1.0 (+https://github.com/solomonanujk/tds-2026-09-ga3)"
)

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


def _candidates() -> List[str]:
    ordered = [MODEL] + [name for name in FALLBACK_MODELS if name != MODEL]
    if ACTIVE_MODEL in ordered:
        index = ordered.index(ACTIVE_MODEL)
        ordered = ordered[index:] + ordered[:index]
    return ordered


def _repair_body(status: int, detail: str, body: Dict[str, Any]) -> bool:
    """Adapt the request to a provider that rejects one of its parameters.

    Newer OpenAI models want ``max_completion_tokens`` and a fixed temperature;
    several cheap models reject structured-output mode. Each is repaired once
    rather than losing the batch to the fallback analyser.
    """
    if status not in (400, 404, 422):
        return False
    lowered = detail.lower()
    if "max_tokens" in lowered and "max_completion_tokens" in lowered and "max_tokens" in body:
        body["max_completion_tokens"] = body.pop("max_tokens")
        return True
    if "temperature" in lowered and "temperature" in body:
        body.pop("temperature")
        return True
    if "response_format" in body and ("response_format" in lowered or "json" in lowered):
        body.pop("response_format")
        return True
    if "response_format" in body:
        body.pop("response_format")
        return True
    return False


def _is_unknown_model(status: int, detail: str) -> bool:
    lowered = detail.lower()
    return status in (400, 404) and (
        "model_not_found" in lowered
        or "deprecated" in lowered
        or "pricing unknown" in lowered
        or ("model" in lowered and ("not found" in lowered or "does not exist" in lowered))
    )


def chat_json(
    messages: List[Dict[str, str]],
    *,
    max_tokens: int = 4096,
    temperature: float = 0.0,
    timeout: Optional[float] = None,
    attempts: int = 2,
) -> Optional[Any]:
    """One chat completion, parsed as JSON. ``None`` on any failure."""
    global LAST_ERROR, CALL_COUNT, ACTIVE_MODEL
    if not API_KEY:
        LAST_ERROR = "no api key configured"
        return None
    deadline = time.time() + (timeout or TIMEOUT)

    for model in _candidates():
        body = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": min(max_tokens, MAX_OUTPUT_TOKENS),
            "response_format": {"type": "json_object"},
        }
        unknown_model = False
        repairs = 0
        for attempt in range(attempts + 3):
            remaining = deadline - time.time()
            if remaining <= 1:
                return None
            request = urllib.request.Request(
                f"{BASE_URL}/chat/completions",
                data=json.dumps(body).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    # Without an explicit agent, Cloudflare in front of the
                    # provider rejects the default urllib one with error 1010.
                    "User-Agent": USER_AGENT,
                    "HTTP-Referer": "https://github.com/solomonanujk/tds-2026-09-ga3",
                    "X-Title": "TDS 2026-09 GA3 agents",
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
                    ACTIVE_MODEL = model
                    return parsed
                LAST_ERROR = "model returned unparseable content"
            except urllib.error.HTTPError as exc:  # pragma: no cover - network
                detail = exc.read()[:400].decode("utf-8", "replace")
                LAST_ERROR = f"http {exc.code} ({model}): {detail}"
                if _is_unknown_model(exc.code, detail):
                    unknown_model = True
                    break
                if exc.code in (401, 403):
                    return None
                if repairs < 3 and _repair_body(exc.code, detail, body):
                    repairs += 1
                    continue
                if exc.code == 429:
                    time.sleep(1.5)
                    continue
            except Exception as exc:  # pragma: no cover - network
                LAST_ERROR = f"{type(exc).__name__}: {exc}"
            time.sleep(0.4 * (attempt + 1))
        if not unknown_model:
            return None
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
