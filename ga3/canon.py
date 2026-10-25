"""Canonical JSON and digest helpers shared by all three GA3 agents.

Every question defines its digests over "recursively key-sorted, compact JSON".
One implementation, used everywhere, keeps the graders' digests reproducible.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical(value: Any) -> str:
    """Recursively key-sorted, compact JSON. Arrays keep their order."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def canonical_bytes(value: Any) -> bytes:
    return canonical(value).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest(value: Any) -> str:
    """Lowercase SHA-256 hex of the canonical encoding of ``value``."""
    return sha256_hex(canonical_bytes(value))


def short_id(prefix: str, *parts: Any, length: int = 32) -> str:
    """Deterministic opaque id derived from content.

    Stable ids are a requirement in all three questions: the same package,
    dossier or run must produce the same id on every replay and later Check.
    """
    body = sha256_hex(canonical_bytes(list(parts)))[:length]
    return f"{prefix}{body}" if prefix.endswith(("-", "_", ":", ".")) else f"{prefix}-{body}"
