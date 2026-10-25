"""GA3 Q4 — Observable Incident Triage.

Routes (relative to the submitted base URL):

    POST /v2/incidents
    POST /v2/incidents/{runId}/receipts
    GET  /v2/incidents/{runId}

The run is a receipt-driven state machine: diagnose, dispatch the minimal
diagnostics, honour one 503 retry, suppress an effect whose prerequisite timed
out, gate destructive effects behind approval, then store the terminal result
with an OTLP trace that matches every dispatch and receipt.
"""
from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from . import planner, store, trace
from .canon import digest, short_id
from .llm import model_name

PROFILE = "ga5-incident-agent/v2"
NS_RUN = "incident.run"
NS_REPLAY = "incident.replay"

router = APIRouter()

_locks: Dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _lock_for(run_id: str) -> threading.Lock:
    with _locks_guard:
        lock = _locks.get(run_id)
        if lock is None:
            lock = _locks[run_id] = threading.Lock()
        return lock


def _error(status: int, code: str, message: str) -> JSONResponse:
    """Errors never echo request content: no transcript, no sensitive values."""
    return JSONResponse({"error": {"code": code, "message": message}}, status_code=status)


# --------------------------------------------------------------------------- #
# response views
# --------------------------------------------------------------------------- #

def _dispatch_view(action: Dict[str, Any], attempt: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    view = {
        "actionId": action["actionId"],
        "callId": action["callId"],
        "phase": action["phase"],
        "toolName": action["toolName"],
        "arguments": action["arguments"],
        "evidence": action["evidence"],
        "attempt": attempt["attempt"],
        "traceparent": trace.traceparent(state["trace"]["traceId"], attempt["clientSpanId"]),
    }
    tracestate = state["trace"].get("tracestate")
    if tracestate:
        view["tracestate"] = tracestate
    if action.get("approvalId"):
        view["approvalId"] = action["approvalId"]
        view["approvalNonce"] = action.get("approvalNonce")
    return view


def _pending_dispatches(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for action in state["actions"]:
        for attempt in action["attempts"]:
            if attempt.get("receipt") is None:
                out.append(_dispatch_view(action, attempt, state))
    return out


def _pending_approvals(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    approval = state.get("approval")
    if approval and approval.get("decision") is None:
        return [
            {
                "approvalId": approval["approvalId"],
                "actionId": approval["actionId"],
                "toolName": approval["toolName"],
                "argumentsDigest": approval["argumentsDigest"],
            }
        ]
    return []


def _waiting_response(
    state: Dict[str, Any],
    dispatches: Optional[List[Dict[str, Any]]] = None,
    approvals: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    return {
        "runId": state["runId"],
        "status": "waiting",
        "diagnosis": state["diagnosis"],
        "dispatches": dispatches if dispatches is not None else _pending_dispatches(state),
        "approvals": approvals if approvals is not None else _pending_approvals(state),
    }


def _terminal_response(state: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "runId": state["runId"],
        "status": state["status"],
        "diagnosis": state["diagnosis"],
        "chosenEffect": state.get("chosenEffect"),
        "suppressed": state.get("suppressed", []),
        "actionLog": state.get("actionLog", []),
        "receiptLog": state.get("receiptLog", []),
        "otlp": build_otlp(state),
        "dispatches": [],
        "approvals": [],
    }


def _view(state: Dict[str, Any]) -> Dict[str, Any]:
    if state["status"] in {"completed", "failed"}:
        return _terminal_response(state)
    return _waiting_response(state)


# --------------------------------------------------------------------------- #
# OTLP
# --------------------------------------------------------------------------- #

def build_otlp(state: Dict[str, Any]) -> Dict[str, Any]:
    run_id = state["runId"]
    marker = state["publicMarker"]
    trace_id = state["trace"]["traceId"]
    base = {"ga5.run.id": run_id, "ga5.public.marker": marker}
    start = state["timing"]["start"]
    end = state["timing"].get("end") or trace.now_nanos()

    spans: List[Dict[str, Any]] = []
    spans.append(
        trace.span(
            trace_id=trace_id,
            span_id=state["trace"]["serverSpanId"],
            parent_span_id=state["trace"].get("parentSpanId"),
            name="POST /v2/incidents",
            kind=trace.SPAN_KIND_SERVER,
            start=start,
            end=end,
            attributes={
                **base,
                "http.request.method": "POST",
                "url.path": "/v2/incidents",
                "http.response.status_code": 200,
            },
            status_code=trace.STATUS_OK,
        )
    )
    spans.append(
        trace.span(
            trace_id=trace_id,
            span_id=state["trace"]["agentSpanId"],
            parent_span_id=state["trace"]["serverSpanId"],
            name=f"invoke_agent {state['agentName']}",
            kind=trace.SPAN_KIND_INTERNAL,
            start=start + 1000,
            end=end - 1000,
            attributes={
                **base,
                "gen_ai.operation.name": "invoke_agent",
                "gen_ai.agent.name": state["agentName"],
            },
            status_code=trace.STATUS_OK,
        )
    )
    model = state["modelSpan"]
    spans.append(
        trace.span(
            trace_id=trace_id,
            span_id=model["spanId"],
            parent_span_id=state["trace"]["agentSpanId"],
            name="chat incident-plan",
            kind=trace.SPAN_KIND_CLIENT,
            start=model["start"],
            end=model["end"],
            attributes={
                **base,
                "gen_ai.operation.name": "chat",
                "gen_ai.request.model": model["model"],
                "gen_ai.provider.name": model.get("provider", "openai"),
            },
            status_code=trace.STATUS_OK,
        )
    )

    diagnostic_span_ids: List[str] = []
    for action in state["actions"]:
        attempts = action["attempts"]
        first = attempts[0]
        last = attempts[-1]
        tool_start = first["dispatchedAt"]
        tool_end = last.get("settledAt") or last["dispatchedAt"] + 1000
        failed = action["state"] == "failed"
        spans.append(
            trace.span(
                trace_id=trace_id,
                span_id=action["internalSpanId"],
                parent_span_id=state["trace"]["agentSpanId"],
                name=f"execute_tool {action['toolName']}",
                kind=trace.SPAN_KIND_INTERNAL,
                start=tool_start,
                end=tool_end,
                attributes={
                    **base,
                    "ga5.action.id": action["actionId"],
                    "ga5.action.phase": action["phase"],
                    "gen_ai.tool.name": action["toolName"],
                    "gen_ai.tool.call.id": action["callId"],
                    "gen_ai.operation.name": "execute_tool",
                },
                status_code=trace.STATUS_ERROR if failed else trace.STATUS_OK,
                status_message="tool call failed" if failed else "",
            )
        )
        if action["phase"] == "diagnostic":
            diagnostic_span_ids.append(action["internalSpanId"])

        for attempt in attempts:
            receipt = attempt.get("receipt") or {}
            attributes: Dict[str, Any] = {
                **base,
                "ga5.action.id": action["actionId"],
                "ga5.attempt": attempt["attempt"],
                "http.request.method": "POST",
                "http.request.resend_count": attempt["attempt"] - 1,
                "url.path": f"tool/{action['toolName']}",
            }
            if receipt.get("receiptId"):
                attributes["ga5.receipt.id"] = receipt["receiptId"]
            if receipt.get("nonce"):
                attributes["ga5.receipt.nonce"] = receipt["nonce"]
            status_code = trace.STATUS_UNSET
            message = ""
            http_status = receipt.get("status")
            if isinstance(http_status, int) and http_status > 0:
                attributes["http.response.status_code"] = http_status
            if receipt.get("errorType"):
                attributes["error.type"] = receipt["errorType"]
                status_code = trace.STATUS_ERROR
                message = str(receipt["errorType"])
            elif isinstance(http_status, int) and http_status >= 400:
                attributes["error.type"] = str(http_status)
                status_code = trace.STATUS_ERROR
                message = f"tool returned {http_status}"
            elif isinstance(http_status, int) and 200 <= http_status < 300:
                status_code = trace.STATUS_OK
            spans.append(
                trace.span(
                    trace_id=trace_id,
                    span_id=attempt["clientSpanId"],
                    parent_span_id=action["internalSpanId"],
                    name=f"POST tool/{action['toolName']}",
                    kind=trace.SPAN_KIND_CLIENT,
                    start=attempt["dispatchedAt"],
                    end=attempt.get("settledAt") or attempt["dispatchedAt"] + 1000,
                    attributes=attributes,
                    status_code=status_code,
                    status_message=message,
                )
            )

    join = state.get("joinSpan")
    if join and len(diagnostic_span_ids) >= 2:
        spans.append(
            trace.span(
                trace_id=trace_id,
                span_id=join["spanId"],
                parent_span_id=state["trace"]["agentSpanId"],
                name="incident.join",
                kind=trace.SPAN_KIND_INTERNAL,
                start=join["start"],
                end=join.get("end") or join["start"] + 1000,
                attributes={**base, "ga5.join.count": len(diagnostic_span_ids)},
                status_code=trace.STATUS_OK,
                links=[{"traceId": trace_id, "spanId": sid} for sid in diagnostic_span_ids],
            )
        )

    approval = state.get("approval")
    if approval:
        attributes = {
            **base,
            "ga5.approval.id": approval["approvalId"],
            "ga5.action.id": approval["actionId"],
            "gen_ai.tool.name": approval["toolName"],
            "ga5.approval.decision": approval.get("decision") or "pending",
        }
        if approval.get("nonce"):
            attributes["ga5.approval.nonce"] = approval["nonce"]
            attributes["ga5.receipt.nonce"] = approval["nonce"]
        if approval.get("receiptId"):
            attributes["ga5.receipt.id"] = approval["receiptId"]
        spans.append(
            trace.span(
                trace_id=trace_id,
                span_id=approval["spanId"],
                parent_span_id=state["trace"]["agentSpanId"],
                name="approval_gate",
                kind=trace.SPAN_KIND_INTERNAL,
                start=approval["start"],
                end=approval.get("end") or approval["start"] + 1000,
                attributes=attributes,
                status_code=trace.STATUS_OK,
            )
        )

    return trace.envelope(spans, service_name=state["agentName"])


# --------------------------------------------------------------------------- #
# state transitions
# --------------------------------------------------------------------------- #

def _new_action(
    state: Dict[str, Any], tool_name: str, phase: str, arguments: Dict[str, Any], evidence: List[str]
) -> Dict[str, Any]:
    action_id = short_id("act", state["runId"], phase, tool_name, arguments)
    action = {
        "actionId": action_id,
        "callId": short_id("call", state["runId"], phase, tool_name, arguments),
        "toolName": tool_name,
        "phase": phase,
        "arguments": arguments,
        "evidence": evidence,
        "state": "pending",
        "internalSpanId": trace.new_span_id(),
        "attempts": [],
    }
    state["actions"].append(action)
    return action


def _dispatch(state: Dict[str, Any], action: Dict[str, Any]) -> Dict[str, Any]:
    attempt = {
        "attempt": len(action["attempts"]) + 1,
        "clientSpanId": trace.new_span_id(),
        "dispatchedAt": trace.now_nanos(),
        "receipt": None,
    }
    action["attempts"].append(attempt)
    view = _dispatch_view(action, attempt, state)
    state["actionLog"].append(view)
    return view


def _find_action(state: Dict[str, Any], action_id: Any, call_id: Any) -> Optional[Dict[str, Any]]:
    for action in state["actions"]:
        if action["actionId"] == action_id and action["callId"] == call_id:
            return action
    return None


def _effect_requires_approval(state: Dict[str, Any]) -> bool:
    effect = state["plan"].get("effect") or {}
    return effect.get("toolName") in (state["policy"].get("approvalRequiredFor") or [])


def _request_approval(state: Dict[str, Any]) -> Dict[str, Any]:
    effect = state["plan"]["effect"]
    arguments = effect["arguments"]
    action_id = short_id("act", state["runId"], "effect", effect["toolName"], arguments)
    approval = {
        "approvalId": short_id("apr", state["runId"], effect["toolName"], arguments),
        "actionId": action_id,
        "toolName": effect["toolName"],
        "argumentsDigest": digest(arguments),
        "spanId": trace.new_span_id(),
        "start": trace.now_nanos(),
        "decision": None,
        "nonce": None,
        "receiptId": None,
    }
    state["approval"] = approval
    state["phase"] = "awaiting_approval"
    return {
        "approvalId": approval["approvalId"],
        "actionId": approval["actionId"],
        "toolName": approval["toolName"],
        "argumentsDigest": approval["argumentsDigest"],
    }


def _dispatch_effect(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    effect = state["plan"]["effect"]
    action = _new_action(state, effect["toolName"], "effect", effect["arguments"], effect["evidence"])
    approval = state.get("approval")
    if approval and approval.get("decision") == "approved":
        # The approval request reserved this logical action id for the effect.
        action["actionId"] = approval["actionId"]
        action["approvalId"] = approval["approvalId"]
        action["approvalNonce"] = approval["nonce"]
    state["phase"] = "effect"
    state["chosenEffect"] = effect["toolName"]
    return [_dispatch(state, action)]


def _suppress_effect(state: Dict[str, Any], reason: str) -> None:
    effect = (state["plan"].get("effect") or {}).get("toolName")
    if effect and effect not in state["suppressed"]:
        state["suppressed"].append(effect)
    state["chosenEffect"] = None
    state["status"] = "failed"
    state["phase"] = "terminal"
    state["failureReason"] = reason
    state["timing"]["end"] = trace.now_nanos()


def _finish(state: Dict[str, Any], status: str) -> None:
    state["status"] = status
    state["phase"] = "terminal"
    state["timing"]["end"] = trace.now_nanos()


def _diagnostics_done(state: Dict[str, Any]) -> bool:
    diagnostics = [a for a in state["actions"] if a["phase"] == "diagnostic"]
    return bool(diagnostics) and all(a["state"] == "succeeded" for a in diagnostics)


def _advance_after_diagnostics(state: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Decide what, if anything, to send once every diagnostic has settled."""
    if not (state["plan"].get("effect") or {}).get("toolName"):
        _finish(state, "completed")
        return [], []
    if _effect_requires_approval(state):
        approval = state.get("approval")
        if approval is None:
            return [], [_request_approval(state)]
        if approval.get("decision") is None:
            return [], _pending_approvals(state)
        if approval.get("decision") != "approved":
            _suppress_effect(state, "approval_denied")
            return [], []
    if any(a["phase"] == "effect" for a in state["actions"]):
        return [], []
    return _dispatch_effect(state), []


# --------------------------------------------------------------------------- #
# routes
# --------------------------------------------------------------------------- #

@router.post("/v2/incidents")
async def create_incident(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        return _error(400, "INVALID_JSON", "body must be JSON")
    if not isinstance(body, dict):
        return _error(400, "INVALID_BODY", "body must be a JSON object")
    if body.get("profile") != PROFILE:
        return _error(400, "UNSUPPORTED_PROFILE", f"profile must be {PROFILE}")
    run_id = body.get("runId")
    incident = body.get("incident")
    if not isinstance(run_id, str) or not run_id.strip():
        return _error(422, "INVALID_RUN_ID", "runId must be a nonempty string")
    if not isinstance(incident, dict) or not isinstance(incident.get("transcript"), str):
        return _error(422, "INVALID_INCIDENT", "incident.transcript is required")

    request_digest = digest(body)
    lock = _lock_for(run_id)
    with lock:
        stored = store.get(NS_RUN, run_id)
        if stored is not None:
            if stored["requestDigest"] != request_digest:
                return _error(409, "RUN_CONTENT_CONFLICT", "runId already exists with different content")
            replay = store.get(NS_REPLAY, f"{run_id}:create:{request_digest}")
            return JSONResponse(replay if replay is not None else _view(stored), status_code=200)

        incoming = trace.parse_traceparent(request.headers.get("traceparent"))
        trace_id = incoming["traceId"] if incoming else trace.new_trace_id()
        tracestate = request.headers.get("tracestate") if incoming else None
        started = trace.now_nanos()

        policy = body.get("policy") if isinstance(body.get("policy"), dict) else {}
        catalog = body.get("toolCatalog") if isinstance(body.get("toolCatalog"), list) else []

        model_start = trace.now_nanos()
        plan = planner.plan_incident(incident, catalog, policy)
        model_end = trace.now_nanos()

        state: Dict[str, Any] = {
            "runId": run_id,
            "profile": PROFILE,
            "requestDigest": request_digest,
            "agentName": body.get("agentName") or "incident-response",
            "publicMarker": body.get("publicMarker") or "",
            "policy": policy,
            "plan": plan,
            "diagnosis": {"rootCause": plan["rootCause"], "evidence": plan["evidence"]},
            "status": "waiting",
            "phase": "diagnostics",
            "actions": [],
            "actionLog": [],
            "receiptLog": [],
            "receipts": {},
            "approval": None,
            "suppressed": [],
            "chosenEffect": None,
            "joinSpan": None,
            "modelSpan": {
                "spanId": trace.new_span_id(),
                "start": model_start,
                "end": model_end,
                "model": model_name(),
                "provider": "openai-compatible",
            },
            "timing": {"start": started},
            "trace": {
                "traceId": trace_id,
                "serverSpanId": trace.new_span_id(),
                "agentSpanId": trace.new_span_id(),
                "parentSpanId": incoming["spanId"] if incoming else None,
                "tracestate": tracestate,
            },
        }

        dispatches = []
        for diagnostic in plan["diagnostics"]:
            action = _new_action(
                state, diagnostic["toolName"], "diagnostic", diagnostic["arguments"], diagnostic["evidence"]
            )
            dispatches.append(_dispatch(state, action))
        if len(dispatches) >= 2:
            state["joinSpan"] = {"spanId": trace.new_span_id(), "start": trace.now_nanos()}

        response = _waiting_response(state, dispatches=dispatches, approvals=[])
        store.put(NS_RUN, run_id, state)
        store.put(NS_REPLAY, f"{run_id}:create:{request_digest}", response)
        return JSONResponse(response, status_code=200)


@router.post("/v2/incidents/{run_id}/receipts")
async def post_receipts(run_id: str, request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        return _error(400, "INVALID_JSON", "body must be JSON")
    if not isinstance(body, dict):
        return _error(400, "INVALID_BODY", "body must be a JSON object")
    if body.get("profile") is not None and body.get("profile") != PROFILE:
        return _error(400, "UNSUPPORTED_PROFILE", f"profile must be {PROFILE}")
    receipt_id = body.get("receiptId")
    outcomes = body.get("outcomes")
    approvals = body.get("approvals")
    if not isinstance(receipt_id, str) or not receipt_id.strip():
        return _error(422, "INVALID_RECEIPT_ID", "receiptId must be a nonempty string")
    if outcomes is None and approvals is None:
        return _error(422, "EMPTY_RECEIPT", "receipt needs outcomes or approvals")
    if outcomes is not None and not isinstance(outcomes, list):
        return _error(422, "INVALID_OUTCOMES", "outcomes must be an array")
    if approvals is not None and not isinstance(approvals, list):
        return _error(422, "INVALID_APPROVALS", "approvals must be an array")

    body_digest = digest(body)
    lock = _lock_for(run_id)
    with lock:
        state = store.get(NS_RUN, run_id)
        if state is None:
            return _error(404, "UNKNOWN_RUN", "no such run")
        previous = state["receipts"].get(receipt_id)
        if previous is not None:
            if previous != body_digest:
                return _error(409, "RECEIPT_CONTENT_CONFLICT", "receiptId already seen with different content")
            replay = store.get(NS_REPLAY, f"{run_id}:receipt:{body_digest}")
            return JSONResponse(replay if replay is not None else _view(state), status_code=200)

        terminal_before = state["status"] in {"completed", "failed"}
        applied = 0
        dispatches: List[Dict[str, Any]] = []
        new_approvals: List[Dict[str, Any]] = []

        for outcome in outcomes or []:
            if not isinstance(outcome, dict):
                continue
            action = _find_action(state, outcome.get("actionId"), outcome.get("callId"))
            if action is None or action["state"] != "pending":
                continue
            attempt_number = outcome.get("attempt")
            attempt = None
            for candidate in action["attempts"]:
                if candidate.get("receipt") is None and (
                    attempt_number is None or candidate["attempt"] == attempt_number
                ):
                    attempt = candidate
                    break
            if attempt is None:
                continue
            status_code = outcome.get("status")
            error_type = outcome.get("errorType")
            record = {
                "receiptId": receipt_id,
                "status": status_code,
                "resultClass": outcome.get("resultClass"),
                "nonce": outcome.get("nonce"),
                "errorType": error_type,
            }
            attempt["receipt"] = record
            attempt["settledAt"] = trace.now_nanos()
            applied += 1
            log_entry = {
                "receiptId": receipt_id,
                "actionId": action["actionId"],
                "callId": action["callId"],
                "attempt": attempt["attempt"],
                "status": status_code,
                "resultClass": outcome.get("resultClass"),
                "nonce": outcome.get("nonce"),
            }
            if error_type:
                log_entry["errorType"] = error_type
            state["receiptLog"].append(log_entry)

            retryable = status_code == 503 and not error_type
            if isinstance(status_code, int) and 200 <= status_code < 300 and not error_type:
                action["state"] = "succeeded"
            elif retryable and len(action["attempts"]) < 2:
                dispatches.append(_dispatch(state, action))
            else:
                action["state"] = "failed"

        for approval_in in approvals or []:
            if not isinstance(approval_in, dict):
                continue
            approval = state.get("approval")
            if approval is None or approval["approvalId"] != approval_in.get("approvalId"):
                continue
            if approval.get("decision") is not None:
                continue
            approval["decision"] = approval_in.get("decision")
            approval["nonce"] = approval_in.get("nonce")
            approval["receiptId"] = receipt_id
            approval["end"] = trace.now_nanos()
            applied += 1
            state["receiptLog"].append(
                {
                    "receiptId": receipt_id,
                    "approvalId": approval["approvalId"],
                    "decision": approval_in.get("decision"),
                    "nonce": approval_in.get("nonce"),
                }
            )

        if applied == 0:
            if terminal_before:
                return _error(409, "INVALID_STATE", "run is already terminal")
            return _error(422, "NO_PENDING_MATCH", "receipt does not match any pending call")

        failed_diagnostic = any(
            a["phase"] == "diagnostic" and a["state"] == "failed" for a in state["actions"]
        )
        effect_actions = [a for a in state["actions"] if a["phase"] == "effect"]

        if state["status"] == "waiting":
            if dispatches:
                # A retry is in flight: never attach an approval request to it,
                # and never run the effect before its prerequisite settles.
                pass
            elif failed_diagnostic:
                _suppress_effect(state, "diagnostic_failed")
            elif effect_actions:
                if all(a["state"] != "pending" for a in effect_actions):
                    _finish(state, "completed" if effect_actions[-1]["state"] == "succeeded" else "failed")
            elif _diagnostics_done(state):
                dispatches, new_approvals = _advance_after_diagnostics(state)

        if state["status"] in {"completed", "failed"}:
            response = _terminal_response(state)
        else:
            response = _waiting_response(state, dispatches=dispatches, approvals=new_approvals)

        state["receipts"][receipt_id] = body_digest
        store.put(NS_RUN, run_id, state)
        store.put(NS_REPLAY, f"{run_id}:receipt:{body_digest}", response)
        return JSONResponse(response, status_code=200)


@router.get("/v2/incidents/{run_id}")
async def get_incident(run_id: str) -> JSONResponse:
    state = store.get(NS_RUN, run_id)
    if state is None:
        return _error(404, "UNKNOWN_RUN", "no such run")
    return JSONResponse(_view(state), status_code=200)
