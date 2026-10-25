"""OTLP span building for the incident agent (GA3 Q4).

Only metadata is exported: run id, public marker, action/call ids, attempt
numbers, receipt ids and nonces, HTTP method and status. Transcripts, prompts,
tool arguments, tool results and anything under ``sensitive``/``doNotExport``
never reach a span.
"""
from __future__ import annotations

import os
import secrets
import time
from typing import Any, Dict, List, Optional

SPAN_KIND_INTERNAL = 1
SPAN_KIND_SERVER = 2
SPAN_KIND_CLIENT = 3

STATUS_UNSET = 0
STATUS_OK = 1
STATUS_ERROR = 2


def new_trace_id() -> str:
    while True:
        value = secrets.token_hex(16)
        if value.strip("0"):
            return value


def new_span_id() -> str:
    while True:
        value = secrets.token_hex(8)
        if value.strip("0"):
            return value


def now_nanos() -> int:
    return time.time_ns()


def parse_traceparent(header: Optional[str]) -> Optional[Dict[str, str]]:
    """Return the trace/span ids of a valid W3C traceparent, else ``None``."""
    if not header:
        return None
    parts = header.strip().split("-")
    if len(parts) < 4:
        return None
    version, trace_id, span_id, flags = parts[0], parts[1], parts[2], parts[3]
    if version == "ff" or len(version) != 2:
        return None
    if len(trace_id) != 32 or len(span_id) != 16 or len(flags) != 2:
        return None
    hexdigits = set("0123456789abcdef")
    if not set(trace_id) <= hexdigits or not set(span_id) <= hexdigits:
        return None
    if not trace_id.strip("0") or not span_id.strip("0"):
        return None
    return {"traceId": trace_id, "spanId": span_id, "flags": flags}


def traceparent(trace_id: str, span_id: str) -> str:
    return f"00-{trace_id}-{span_id}-01"


def attribute(key: str, value: Any) -> Dict[str, Any]:
    if isinstance(value, bool):
        return {"key": key, "value": {"boolValue": value}}
    if isinstance(value, int):
        return {"key": key, "value": {"intValue": value}}
    if isinstance(value, float):
        return {"key": key, "value": {"doubleValue": value}}
    return {"key": key, "value": {"stringValue": str(value)}}


def span(
    *,
    trace_id: str,
    span_id: str,
    name: str,
    kind: int,
    start: int,
    end: int,
    parent_span_id: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
    status_code: int = STATUS_UNSET,
    status_message: str = "",
    links: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "traceId": trace_id,
        "spanId": span_id,
        "name": name,
        "kind": kind,
        "startTimeUnixNano": str(start),
        "endTimeUnixNano": str(end),
        "attributes": [attribute(k, v) for k, v in (attributes or {}).items()],
    }
    if parent_span_id:
        out["parentSpanId"] = parent_span_id
    status: Dict[str, Any] = {"code": status_code}
    if status_message:
        status["message"] = status_message
    out["status"] = status
    if links:
        out["links"] = [
            {"traceId": link["traceId"], "spanId": link["spanId"]} for link in links
        ]
    return out


def envelope(spans: List[Dict[str, Any]], *, service_name: str) -> Dict[str, Any]:
    return {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": [
                        attribute("service.name", service_name),
                        attribute("telemetry.sdk.language", "python"),
                        attribute("telemetry.sdk.name", "ga3-incident-agent"),
                    ]
                },
                "scopeSpans": [
                    {
                        "scope": {
                            "name": "ga3.incident",
                            "version": os.environ.get("GA3_VERSION", "1.0.0"),
                        },
                        "spans": spans,
                    }
                ],
            }
        ]
    }
