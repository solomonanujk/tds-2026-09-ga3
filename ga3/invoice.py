"""GA3 Q2 — A2A 1.0 invoice action agent (HTTP+JSON binding).

    GET  {origin}/.well-known/agent-card.json      public
    POST {base}/message:send                       Bearer
    GET  {base}/tasks/{id}                         Bearer
    GET  {base}/tasks                              Bearer
    POST {base}/tasks/{id}:cancel                  Bearer

A proposal is not permission to act: the first message produces proposals in
TASK_STATE_INPUT_REQUIRED, and only a matching result continuation carrying the
grader's unpredictable receipt nonces can add the receipts artifact and complete
the task. Every Bearer token is a separate user.
"""
from __future__ import annotations

import hashlib
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from time import monotonic
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from . import invoice_policy as policy
from . import llm, store
from .canon import digest

A2A_MEDIA = "application/a2a+json"
BATCH_MEDIA = "application/vnd.ga5.invoice-claim-batch+json"
PROPOSALS_MEDIA = "application/vnd.ga5.invoice-action-proposals+json"
RECEIPTS_MEDIA = "application/vnd.ga5.invoice-action-receipts+json"
RESULTS_MEDIA = "application/vnd.ga5.invoice-action-results+json"

NS_TASK = "invoice.task"
NS_MSG = "invoice.msg"
NS_DECISION = "invoice.decision"

BATCH_SIZE = int(os.environ.get("INVOICE_BATCH_SIZE", "12"))
MAX_WORKERS = int(os.environ.get("INVOICE_WORKERS", "4"))
DECISION_BUDGET = float(os.environ.get("INVOICE_BUDGET", "34"))

router = APIRouter()

_locks: Dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _lock_for(key: str) -> threading.Lock:
    with _locks_guard:
        lock = _locks.get(key)
        if lock is None:
            lock = _locks[key] = threading.Lock()
        return lock


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _json(payload: Any, status: int = 200) -> JSONResponse:
    return JSONResponse(payload, status_code=status, media_type=A2A_MEDIA)


def _error(status: int, code: str, message: str) -> JSONResponse:
    """Generic bodies only: never echo another principal's task id or existence."""
    return JSONResponse(
        {"error": {"code": code, "message": message}}, status_code=status, media_type=A2A_MEDIA
    )


# --------------------------------------------------------------------------- #
# request guards
# --------------------------------------------------------------------------- #

def principal_of(request: Request) -> Optional[str]:
    header = request.headers.get("authorization") or ""
    if not header.lower().startswith("bearer "):
        return None
    token = header[7:].strip()
    if not token:
        return None
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:32]


def guard(request: Request, *, body: bool) -> Optional[JSONResponse]:
    if principal_of(request) is None:
        return _error(401, "UNAUTHENTICATED", "a Bearer token is required")
    version = request.headers.get("a2a-version")
    if version is not None and version.strip() != "1.0":
        return _error(400, "UNSUPPORTED_VERSION", "A2A-Version must be 1.0")
    if body:
        content_type = (request.headers.get("content-type") or "").split(";")[0].strip().lower()
        if content_type and content_type not in {A2A_MEDIA, "application/json"}:
            return _error(415, "UNSUPPORTED_MEDIA_TYPE", f"use {A2A_MEDIA}")
    return None


def base_url(request: Request) -> str:
    configured = os.environ.get("A2A_BASE_URL")
    if configured:
        return configured
    scheme = (request.headers.get("x-forwarded-proto") or request.url.scheme or "https").split(",")[0].strip()
    host = (request.headers.get("x-forwarded-host") or request.headers.get("host") or request.url.netloc)
    if scheme == "http" and host and not host.startswith(("localhost", "127.0.0.1")):
        scheme = "https"
    return f"{scheme}://{host}/a2a/"


# --------------------------------------------------------------------------- #
# agent card
# --------------------------------------------------------------------------- #

def agent_card(base: str) -> Dict[str, Any]:
    variants = [base]
    if base.endswith("/"):
        variants.append(base[:-1])
    else:
        variants.append(base + "/")
    return {
        "protocolVersion": "1.0",
        "name": "GA3 Invoice Action Agent",
        "description": (
            "Reads an invoice claim batch, proposes exactly one typed action per package with "
            "the decisive evidence references, and executes only the proposals the caller "
            "accepts, storing task state so retries never repeat a payment."
        ),
        "version": os.environ.get("GA3_VERSION", "1.0.0"),
        "url": base,
        "preferredTransport": "HTTP+JSON",
        "provider": {"organization": "TDS 2026-09 GA3", "url": base},
        "capabilities": {
            "streaming": False,
            "pushNotifications": False,
            "stateTransitionHistory": True,
            "extensions": [],
        },
        "defaultInputModes": [BATCH_MEDIA, RESULTS_MEDIA, A2A_MEDIA],
        "defaultOutputModes": [PROPOSALS_MEDIA, RECEIPTS_MEDIA, A2A_MEDIA],
        "securitySchemes": {
            "bearer": {"type": "http", "scheme": "bearer", "description": "per-user Bearer token"}
        },
        "security": [{"bearer": []}],
        "supportedInterfaces": [
            {"url": variant, "protocolBinding": "HTTP+JSON", "protocolVersion": "1.0"}
            for variant in variants
        ],
        "skills": [
            {
                "id": "invoice_action_agent",
                "name": "invoice_action_agent",
                "displayName": "Invoice action agent",
                "description": (
                    "Decides settle_invoice, request_approval, hold_invoice, reject_duplicate or "
                    "open_exception for each invoice package, cites the three decisive bracketed "
                    "references, and executes only accepted proposals against grader receipts."
                ),
                "tags": ["invoice", "accounts-payable", "action-gate", "evidence", "a2a"],
                "examples": ["Decide one action per package in an invoice claim batch."],
                "inputModes": [BATCH_MEDIA, RESULTS_MEDIA],
                "outputModes": [PROPOSALS_MEDIA, RECEIPTS_MEDIA],
            }
        ],
    }


@router.get("/.well-known/agent-card.json")
async def well_known_card(request: Request) -> JSONResponse:
    return JSONResponse(agent_card(base_url(request)), media_type="application/json")


@router.get("/a2a/.well-known/agent-card.json")
async def nested_card(request: Request) -> JSONResponse:
    return JSONResponse(agent_card(base_url(request)), media_type="application/json")


# --------------------------------------------------------------------------- #
# decisions
# --------------------------------------------------------------------------- #

def _decide_batch(batch: List[Dict[str, Any]], keys: Dict[str, str], timeout: float) -> None:
    decisions = policy.model_decisions(batch, timeout=timeout) if llm.available() else {}
    for package in batch:
        package_id = str(package.get("packageId"))
        store.put(NS_DECISION, keys[package_id], policy.assemble(package, decisions.get(package_id)))


def decide_packages(packages: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """One validated decision per package, cached by canonical package content."""
    keys = {str(p.get("packageId")): digest(policy.fingerprint(p)) for p in packages}
    out: Dict[str, Dict[str, Any]] = {}
    misses: List[Dict[str, Any]] = []
    for package in packages:
        package_id = str(package.get("packageId"))
        cached = store.get(NS_DECISION, keys[package_id])
        if cached is not None:
            out[package_id] = cached
        else:
            misses.append(package)

    if misses:
        started = monotonic()
        batches = [misses[i : i + BATCH_SIZE] for i in range(0, len(misses), BATCH_SIZE)]
        with ThreadPoolExecutor(max_workers=max(1, MAX_WORKERS)) as pool:
            futures = [
                pool.submit(
                    _decide_batch,
                    batch,
                    keys,
                    max(5.0, min(30.0, DECISION_BUDGET - (monotonic() - started))),
                )
                for batch in batches
            ]
            for future in futures:
                try:
                    future.result(timeout=max(1.0, DECISION_BUDGET - (monotonic() - started)))
                except Exception:
                    continue
        for package in misses:
            package_id = str(package.get("packageId"))
            cached = store.get(NS_DECISION, keys[package_id])
            if cached is None:
                cached = policy.assemble(package, None)
                store.put(NS_DECISION, keys[package_id], cached)
            out[package_id] = cached
    return out


# --------------------------------------------------------------------------- #
# task construction
# --------------------------------------------------------------------------- #

def artifact(name: str, media: str, data: Dict[str, Any], seed: Any) -> Dict[str, Any]:
    return {
        "artifactId": f"art-{digest(seed)[:32]}",
        "name": name,
        "description": name.replace("-", " "),
        "parts": [{"mediaType": media, "data": data}],
    }


def build_proposals(batch: Dict[str, Any], task_id: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    packages = [p for p in (batch.get("packages") or []) if isinstance(p, dict)]
    decisions = decide_packages(packages)
    proposals: List[Dict[str, Any]] = []
    seen_actions = set()
    for index, package in enumerate(packages):
        package_id = str(package.get("packageId") or f"package-{index}")
        body = decisions.get(package_id) or policy.assemble(package, None)
        action_id = f"act-{digest([batch.get('batchId'), package_id, body['action']])[:40]}"
        while action_id in seen_actions:
            action_id = f"act-{digest([action_id, index])[:40]}"
        seen_actions.add(action_id)
        proposals.append(
            {
                "packageId": package_id,
                "actionId": action_id,
                "action": body["action"],
                "facts": body["facts"],
                "evidenceRefs": body["evidenceRefs"],
                "rationale": body["rationale"],
            }
        )
    data = {"batchId": batch.get("batchId"), "proposals": proposals}
    return data, {p["packageId"]: p for p in proposals}


def task_envelope(record: Dict[str, Any]) -> Dict[str, Any]:
    return record["task"]


# --------------------------------------------------------------------------- #
# message:send
# --------------------------------------------------------------------------- #

def _validate_message(message: Any) -> Optional[JSONResponse]:
    if not isinstance(message, dict):
        return _error(400, "INVALID_MESSAGE", "message must be an object")
    if not isinstance(message.get("messageId"), str) or not message["messageId"].strip():
        return _error(400, "INVALID_MESSAGE_ID", "message.messageId is required")
    if message.get("role") not in {"ROLE_USER", "ROLE_AGENT"}:
        return _error(400, "INVALID_ROLE", "message.role must be a ProtoJSON Role")
    parts = message.get("parts")
    if not isinstance(parts, list) or not parts:
        return _error(400, "INVALID_PARTS", "message.parts must be a nonempty array")
    for part in parts:
        if not isinstance(part, dict):
            return _error(400, "INVALID_PART", "each part must be an object")
    return None


def _part_of(message: Dict[str, Any], media: str) -> Optional[Dict[str, Any]]:
    for part in message.get("parts") or []:
        if not isinstance(part, dict):
            continue
        part_media = part.get("mediaType") or part.get("mimeType") or ""
        data = part.get("data")
        if part_media == media and isinstance(data, dict):
            return data
    return None


def _any_data_part(message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for part in message.get("parts") or []:
        if isinstance(part, dict) and isinstance(part.get("data"), dict):
            return part["data"]
    return None


def _start_task(principal: str, message: Dict[str, Any], batch: Dict[str, Any]) -> Dict[str, Any]:
    task_id = f"task-{digest([principal, message['messageId'], batch.get('batchId')])[:32]}"
    context_id = f"ctx-{digest([principal, batch.get('batchId')])[:32]}"
    data, proposals = build_proposals(batch, task_id)
    task = {
        "id": task_id,
        "contextId": context_id,
        "kind": "task",
        "status": {"state": "TASK_STATE_INPUT_REQUIRED", "timestamp": now_iso()},
        "history": [message],
        "artifacts": [artifact("invoice-action-proposals", PROPOSALS_MEDIA, data, [task_id, "proposals"])],
        "metadata": {"batchId": batch.get("batchId"), "policyRevision": batch.get("policyRevision")},
    }
    record = {
        "principal": principal,
        "task": task,
        "batchId": batch.get("batchId"),
        "proposals": proposals,
        "executions": [],
    }
    store.put(NS_TASK, f"{principal}:{task_id}", record)
    return record


def _apply_results(
    principal: str, message: Dict[str, Any], results_data: Dict[str, Any]
) -> Tuple[Optional[Dict[str, Any]], Optional[JSONResponse]]:
    task_id = message.get("taskId")
    context_id = message.get("contextId")
    if not isinstance(task_id, str) or not task_id.strip():
        return None, _error(400, "INVALID_CONTINUATION", "taskId is required to continue a task")
    with _lock_for(f"task:{task_id}"):
        record = store.get(NS_TASK, f"{principal}:{task_id}")
        if record is None:
            # Generic: an outsider learns nothing about another principal's task.
            return None, _error(404, "TASK_NOT_FOUND", "no such task for this principal")
        task = record["task"]
        state = task["status"]["state"]
        if state in {"TASK_STATE_COMPLETED", "TASK_STATE_CANCELED", "TASK_STATE_FAILED", "TASK_STATE_REJECTED"}:
            if state == "TASK_STATE_COMPLETED" and record.get("resultMessageId") == message.get("messageId"):
                return record, None
            return None, _error(409, "TASK_NOT_CONTINUABLE", "task is already terminal")
        if context_id is not None and context_id != task["contextId"]:
            return None, _error(400, "CONTEXT_MISMATCH", "contextId does not match the task")
        if results_data.get("batchId") != record["batchId"]:
            return None, _error(400, "BATCH_MISMATCH", "batchId does not match the task")
        results = results_data.get("results")
        if not isinstance(results, list) or not results:
            return None, _error(400, "INVALID_RESULTS", "results must be a nonempty array")

        executions: List[Dict[str, Any]] = []
        for result in results:
            if not isinstance(result, dict):
                return None, _error(400, "INVALID_RESULT", "each result must be an object")
            proposal = record["proposals"].get(result.get("packageId"))
            if proposal is None:
                return None, _error(400, "UNKNOWN_PACKAGE", "result does not match a stored proposal")
            if result.get("actionId") != proposal["actionId"] or result.get("action") != proposal["action"]:
                return None, _error(400, "ACTION_MISMATCH", "result does not match the stored proposal")
            outcome = result.get("outcome")
            if outcome not in {"ACCEPTED", "REJECTED"}:
                return None, _error(400, "INVALID_OUTCOME", "outcome must be ACCEPTED or REJECTED")
            if outcome != "ACCEPTED":
                continue
            nonce = result.get("receiptNonce")
            if not isinstance(nonce, str) or not nonce.strip():
                return None, _error(400, "MISSING_RECEIPT", "an accepted result needs its receiptNonce")
            executions.append(
                {
                    "packageId": proposal["packageId"],
                    "actionId": proposal["actionId"],
                    "action": proposal["action"],
                    "receiptNonce": nonce,
                    "facts": proposal["facts"],
                    "evidenceRefs": proposal["evidenceRefs"],
                }
            )

        receipts = {"batchId": record["batchId"], "executions": executions}
        task["history"] = list(task.get("history") or []) + [message]
        task["artifacts"] = [
            a for a in task["artifacts"] if a["parts"][0]["mediaType"] != RECEIPTS_MEDIA
        ] + [artifact("invoice-action-receipts", RECEIPTS_MEDIA, receipts, [task["id"], "receipts"])]
        task["status"] = {"state": "TASK_STATE_COMPLETED", "timestamp": now_iso()}
        record["task"] = task
        record["executions"] = executions
        record["resultMessageId"] = message.get("messageId")
        store.put(NS_TASK, f"{principal}:{task['id']}", record)
        return record, None


@router.post("/a2a/message:send")
async def message_send(request: Request) -> JSONResponse:
    blocked = guard(request, body=True)
    if blocked is not None:
        return blocked
    principal = principal_of(request)
    try:
        body = await request.json()
    except Exception:
        return _error(400, "INVALID_JSON", "body must be JSON")
    if not isinstance(body, dict):
        return _error(400, "INVALID_BODY", "body must be a JSON object")
    message = body.get("message")
    invalid = _validate_message(message)
    if invalid is not None:
        return invalid

    # Deduplicate on the message alone: configuration, key order and
    # returnImmediately must not change the identity of the request.
    message_hash = digest(message)
    idem_key = f"{principal}:{message['messageId']}"

    won, existing = store.claim(NS_MSG, idem_key)
    if not won:
        record = existing if existing is not None else store.await_ready(NS_MSG, idem_key, timeout=40.0)
        if record is not None:
            if record.get("hash") != message_hash:
                return _error(409, "IDEMPOTENCY_CONFLICT", "this messageId was used with different content")
            stored = store.get(NS_TASK, f"{principal}:{record['taskId']}")
            if stored is not None:
                return _json({"task": task_envelope(stored)})
            return _error(409, "IDEMPOTENCY_CONFLICT", "stored task is unavailable")
        return _error(409, "CONCURRENT_REQUEST", "an equivalent request is still in flight")

    try:
        results_data = _part_of(message, RESULTS_MEDIA)
        if results_data is None and message.get("taskId"):
            candidate = _any_data_part(message)
            if isinstance(candidate, dict) and isinstance(candidate.get("results"), list):
                results_data = candidate
        if results_data is not None:
            record, error = _apply_results(principal, message, results_data)
            if error is not None:
                store.release(NS_MSG, idem_key)
                return error
            store.put(NS_MSG, idem_key, {"hash": message_hash, "taskId": record["task"]["id"]})
            return _json({"task": task_envelope(record)})

        batch = _part_of(message, BATCH_MEDIA)
        if batch is None:
            candidate = _any_data_part(message)
            if isinstance(candidate, dict) and isinstance(candidate.get("packages"), list):
                batch = candidate
        if batch is None or not isinstance(batch.get("packages"), list) or not batch["packages"]:
            store.release(NS_MSG, idem_key)
            return _error(400, "UNSUPPORTED_INPUT", f"send a {BATCH_MEDIA} part with packages")
        record = _start_task(principal, message, batch)
        store.put(NS_MSG, idem_key, {"hash": message_hash, "taskId": record["task"]["id"]})
        return _json({"task": task_envelope(record)})
    except BaseException:
        store.release(NS_MSG, idem_key)
        raise


# --------------------------------------------------------------------------- #
# task reads and cancellation
# --------------------------------------------------------------------------- #

@router.post("/a2a/tasks/{task_id}:cancel")
async def cancel_task(task_id: str, request: Request) -> JSONResponse:
    blocked = guard(request, body=False)
    if blocked is not None:
        return blocked
    principal = principal_of(request)
    with _lock_for(f"task:{task_id}"):
        record = store.get(NS_TASK, f"{principal}:{task_id}")
        if record is None:
            return _error(404, "TASK_NOT_FOUND", "no such task for this principal")
        task = record["task"]
        if task["status"]["state"] in {
            "TASK_STATE_COMPLETED",
            "TASK_STATE_CANCELED",
            "TASK_STATE_FAILED",
            "TASK_STATE_REJECTED",
        }:
            return _error(409, "TASK_NOT_CANCELABLE", "task is already terminal")
        task["status"] = {"state": "TASK_STATE_CANCELED", "timestamp": now_iso()}
        record["task"] = task
        store.put(NS_TASK, f"{principal}:{task_id}", record)
        return _json(task)


@router.get("/a2a/tasks")
async def list_tasks(request: Request) -> JSONResponse:
    blocked = guard(request, body=False)
    if blocked is not None:
        return blocked
    principal = principal_of(request)
    records = store.list_ns(NS_TASK, prefix=f"{principal}:")
    return _json({"tasks": [record["task"] for record in records]})


@router.get("/a2a/tasks/{task_id}")
async def get_task(task_id: str, request: Request) -> JSONResponse:
    blocked = guard(request, body=False)
    if blocked is not None:
        return blocked
    principal = principal_of(request)
    record = store.get(NS_TASK, f"{principal}:{task_id}")
    if record is None:
        return _error(404, "TASK_NOT_FOUND", "no such task for this principal")
    return _json(task_envelope(record))
