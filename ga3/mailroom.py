"""GA3 Q3 — Safe AI Mailroom: one endpoint, two operations.

    POST <BASE>/mailroom   {"operation": "propose" | "commit", ...}

Proposals are rebuilt from the frozen action schemas and cached by canonical
dossier content, so the stable core costs one model pass ever. Commits verify
every Ed25519 receipt against the verifier key supplied with that evaluation's
propose request before any outcome is recorded.
"""
from __future__ import annotations

import base64
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from time import monotonic
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from . import mailroom_policy as policy
from . import llm, store
from .canon import canonical, canonical_bytes, digest, sha256_hex

PROFILE = "ga5-mailroom-action-gate/v2"
NS_EVAL = "mailroom.eval"
NS_DECISION = "mailroom.decision"
NS_REPLAY = "mailroom.replay"

BATCH_SIZE = int(os.environ.get("MAILROOM_BATCH_SIZE", "6"))
MAX_WORKERS = int(os.environ.get("MAILROOM_WORKERS", "6"))
DECISION_BUDGET = float(os.environ.get("MAILROOM_BUDGET", "42"))

router = APIRouter()

_locks: Dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _lock_for(key: str) -> threading.Lock:
    with _locks_guard:
        lock = _locks.get(key)
        if lock is None:
            lock = _locks[key] = threading.Lock()
        return lock


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        {"profile": PROFILE, "error": {"code": code, "message": message}}, status_code=status
    )


# --------------------------------------------------------------------------- #
# digests and signatures
# --------------------------------------------------------------------------- #

def input_digest(dossiers: Any) -> str:
    return sha256_hex(canonical_bytes(dossiers))


def proposal_digest(proposal: Dict[str, Any]) -> str:
    normalised = {
        "dossierId": proposal["dossierId"],
        "callId": proposal["callId"],
        "action": proposal["action"],
        "target": proposal.get("target") if proposal.get("target") is not None else None,
        "payload": proposal["payload"],
        "evidence": sorted(proposal["evidence"]),
    }
    return digest(normalised)


def _b64decode(value: str) -> bytes:
    """Accept base64 or base64url, padded or not.

    ``validate=False`` silently drops ``-`` and ``_``, which would quietly
    decode a base64url key into the wrong bytes, so the alphabet is chosen from
    the input instead.
    """
    cleaned = "".join(value.split())
    padding = "=" * (-len(cleaned) % 4)
    if "-" in cleaned or "_" in cleaned:
        return base64.urlsafe_b64decode(cleaned + padding)
    try:
        return base64.b64decode(cleaned + padding, validate=True)
    except Exception:
        return base64.urlsafe_b64decode(cleaned + padding)


def verify_receipt_signature(
    verifier: Dict[str, Any], evaluation_id: str, digest_value: str, receipt: Dict[str, Any]
) -> bool:
    """Ed25519 check over the exact signed shape the question specifies."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    signature = receipt.get("receiptSignature")
    if not isinstance(signature, str) or not signature.strip():
        return False
    jwk = (verifier or {}).get("publicKeyJwk") or {}
    if jwk.get("kty") != "OKP" or jwk.get("crv") != "Ed25519" or not isinstance(jwk.get("x"), str):
        return False
    try:
        public_key = Ed25519PublicKey.from_public_bytes(_b64decode(jwk["x"]))
    except Exception:
        return False
    inner = {k: v for k, v in receipt.items() if k != "receiptSignature"}
    message = canonical_bytes(
        {
            "profile": PROFILE,
            "evaluationId": evaluation_id,
            "inputDigest": digest_value,
            "receipt": inner,
        }
    )
    try:
        public_key.verify(_b64decode(signature), message)
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# decisions
# --------------------------------------------------------------------------- #

def _decide_batch(batch: List[Dict[str, Any]], keys: Dict[str, str], timeout: float) -> None:
    decisions = policy.model_decisions(batch, timeout=timeout) if llm.available() else {}
    for dossier in batch:
        dossier_id = str(dossier.get("dossierId"))
        choice = decisions.get(dossier_id)
        if not choice or choice.get("action") not in policy.ALLOWED_ACTIONS:
            choice = policy.heuristic_decision(dossier)
        assembled = policy.assemble(dossier, choice["action"], choice.get("fields") or {}, choice.get("evidence") or [])
        if not assembled["evidence"]:
            fallback = policy.heuristic_decision(dossier)
            assembled = policy.assemble(dossier, assembled["action"], fallback.get("fields") or {}, fallback["evidence"])
        if not assembled["evidence"]:
            ids = policy.line_ids(dossier)
            assembled["evidence"] = ids[:1]
        store.put(NS_DECISION, keys[dossier_id], assembled)


def decide_all(
    dossiers: List[Dict[str, Any]]
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str]]:
    """One validated proposal body per dossier, cached by canonical content."""
    keys = {str(d.get("dossierId")): digest(policy.fingerprint(d)) for d in dossiers}
    out: Dict[str, Dict[str, Any]] = {}
    misses: List[Dict[str, Any]] = []
    for dossier in dossiers:
        dossier_id = str(dossier.get("dossierId"))
        cached = store.get(NS_DECISION, keys[dossier_id])
        if cached is not None:
            out[dossier_id] = cached
        else:
            misses.append(dossier)

    if misses:
        started = monotonic()
        batches = [misses[i : i + BATCH_SIZE] for i in range(0, len(misses), BATCH_SIZE)]
        with ThreadPoolExecutor(max_workers=max(1, MAX_WORKERS)) as pool:
            futures = []
            for batch in batches:
                remaining = DECISION_BUDGET - (monotonic() - started)
                futures.append(pool.submit(_decide_batch, batch, keys, max(4.0, min(25.0, remaining))))
            for future in futures:
                try:
                    future.result(timeout=max(1.0, DECISION_BUDGET - (monotonic() - started)))
                except Exception:
                    continue
        for dossier in misses:
            dossier_id = str(dossier.get("dossierId"))
            cached = store.get(NS_DECISION, keys[dossier_id])
            if cached is None:
                choice = policy.heuristic_decision(dossier)
                cached = policy.assemble(dossier, choice["action"], choice.get("fields") or {}, choice["evidence"])
                if not cached["evidence"]:
                    cached["evidence"] = policy.line_ids(dossier)[:1]
                store.put(NS_DECISION, keys[dossier_id], cached)
            out[dossier_id] = cached
    return out, keys


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #

def _validate_propose(body: Dict[str, Any]) -> Optional[JSONResponse]:
    evaluation_id = body.get("evaluationId")
    dossiers = body.get("dossiers")
    if not isinstance(evaluation_id, str) or not evaluation_id.strip():
        return _error(422, "INVALID_EVALUATION_ID", "evaluationId must be a nonempty string")
    if not isinstance(dossiers, list) or not dossiers:
        return _error(422, "INVALID_DOSSIERS", "dossiers must be a nonempty array")
    seen = set()
    for dossier in dossiers:
        if not isinstance(dossier, dict):
            return _error(422, "INVALID_DOSSIER", "each dossier must be an object")
        dossier_id = dossier.get("dossierId")
        if not isinstance(dossier_id, str) or not dossier_id.strip():
            return _error(422, "INVALID_DOSSIER_ID", "dossierId must be a nonempty string")
        if dossier_id in seen:
            return _error(400, "DUPLICATE_DOSSIER_ID", "dossierId repeated in this request")
        seen.add(dossier_id)
        sources = dossier.get("sources")
        if not isinstance(sources, list) or not sources:
            return _error(422, "INVALID_SOURCES", "each dossier needs a nonempty sources array")
        line_ids = set()
        for source in sources:
            if not isinstance(source, dict) or not isinstance(source.get("lines"), list):
                return _error(422, "INVALID_SOURCE", "each source needs a lines array")
            for line in source["lines"]:
                if not isinstance(line, dict) or not isinstance(line.get("lineId"), str):
                    return _error(422, "INVALID_LINE", "each line needs a string lineId")
                if line["lineId"] in line_ids:
                    return _error(400, "DUPLICATE_LINE_ID", "lineId repeated within a dossier")
                line_ids.add(line["lineId"])
        if not line_ids:
            return _error(422, "EMPTY_DOSSIER", "dossier has no lines")
    return None


# --------------------------------------------------------------------------- #
# operations
# --------------------------------------------------------------------------- #

def _propose(body: Dict[str, Any]) -> JSONResponse:
    invalid = _validate_propose(body)
    if invalid is not None:
        return invalid
    evaluation_id = body["evaluationId"]
    dossiers = body["dossiers"]
    computed_digest = input_digest(dossiers)

    with _lock_for(f"eval:{evaluation_id}"):
        stored = store.get(NS_EVAL, evaluation_id)
        if stored is not None:
            if stored["inputDigest"] != computed_digest:
                return _error(409, "EVALUATION_CONTENT_CONFLICT", "evaluationId already used with different content")
            return JSONResponse(stored["proposeResponse"], status_code=200)

        decisions, keys = decide_all(dossiers)
        proposals: List[Dict[str, Any]] = []
        for dossier in dossiers:
            dossier_id = str(dossier["dossierId"])
            body_fields = decisions[dossier_id]
            proposal = {
                "dossierId": dossier_id,
                "callId": f"call-{keys[dossier_id][:40]}",
                "action": body_fields["action"],
                "target": body_fields["target"],
                "payload": body_fields["payload"],
                "evidence": body_fields["evidence"],
            }
            proposals.append(proposal)

        response = {
            "profile": PROFILE,
            "evaluationId": evaluation_id,
            "status": "awaiting_receipts",
            "inputDigest": computed_digest,
            "proposals": proposals,
        }
        record = {
            "evaluationId": evaluation_id,
            "inputDigest": computed_digest,
            "receiptVerifier": body.get("receiptVerifier") or {},
            "proposals": {p["callId"]: p for p in proposals},
            "proposalOrder": [p["callId"] for p in proposals],
            "proposalDigests": {p["callId"]: proposal_digest(p) for p in proposals},
            "proposeResponse": response,
            "commit": None,
        }
        store.put(NS_EVAL, evaluation_id, record)
        return JSONResponse(response, status_code=200)


def _commit(body: Dict[str, Any]) -> JSONResponse:
    evaluation_id = body.get("evaluationId")
    supplied_digest = body.get("inputDigest")
    receipts = body.get("receipts")
    if not isinstance(evaluation_id, str) or not evaluation_id.strip():
        return _error(422, "INVALID_EVALUATION_ID", "evaluationId must be a nonempty string")
    if not isinstance(receipts, list) or not receipts:
        return _error(422, "INVALID_RECEIPTS", "receipts must be a nonempty array")
    for receipt in receipts:
        if not isinstance(receipt, dict):
            return _error(422, "INVALID_RECEIPT", "each receipt must be an object")
        for field in ("dossierId", "callId", "action", "proposalDigest", "receiptId", "receiptSignature"):
            if not isinstance(receipt.get(field), str) or not receipt[field].strip():
                return _error(422, "INVALID_RECEIPT_FIELD", f"receipt.{field} must be a nonempty string")
        if not isinstance(receipt.get("accepted"), bool):
            return _error(422, "INVALID_RECEIPT_FIELD", "receipt.accepted must be a boolean")
    if not isinstance(supplied_digest, str) or not supplied_digest.strip():
        return _error(422, "INVALID_INPUT_DIGEST", "inputDigest must be a nonempty string")

    with _lock_for(f"eval:{evaluation_id}"):
        record = store.get(NS_EVAL, evaluation_id)
        if record is None:
            return _error(404, "UNKNOWN_EVALUATION", "no proposal stored for this evaluation")
        if record["inputDigest"] != supplied_digest:
            return _error(409, "INPUT_DIGEST_MISMATCH", "inputDigest does not match the stored proposal")

        receipts_key = sha256_hex(canonical_bytes(receipts))
        if record.get("commit"):
            if record["commit"]["receiptsKey"] == receipts_key:
                return JSONResponse(record["commit"]["response"], status_code=200)
            return _error(409, "COMMIT_CONTENT_CONFLICT", "this evaluation was already committed with different receipts")

        # Duplicated or relocated receipts invalidate the whole commit.
        call_ids = [r["callId"] for r in receipts]
        if len(set(call_ids)) != len(call_ids):
            return _error(400, "DUPLICATE_RECEIPT", "two receipts for the same callId")
        signatures = [r["receiptSignature"] for r in receipts]
        if len(set(signatures)) != len(signatures):
            return _error(400, "DUPLICATE_SIGNATURE", "the same signature appears twice")
        receipt_ids = [r["receiptId"] for r in receipts]
        if len(set(receipt_ids)) != len(receipt_ids):
            return _error(400, "DUPLICATE_RECEIPT_ID", "the same receiptId appears twice")

        verifier = record.get("receiptVerifier") or {}
        for receipt in receipts:
            if not verify_receipt_signature(verifier, evaluation_id, supplied_digest, receipt):
                return _error(400, "INVALID_RECEIPT_SIGNATURE", "a receipt signature did not verify")

        for receipt in receipts:
            proposal = record["proposals"].get(receipt["callId"])
            if proposal is None:
                return _error(409, "UNKNOWN_CALL_ID", "receipt callId does not match a stored proposal")
            if receipt["dossierId"] != proposal["dossierId"] or receipt["action"] != proposal["action"]:
                return _error(409, "RECEIPT_MISMATCH", "receipt does not match the stored proposal")
            if receipt["proposalDigest"] != record["proposalDigests"][receipt["callId"]]:
                return _error(409, "PROPOSAL_DIGEST_MISMATCH", "receipt proposalDigest does not match")

        by_call = {r["callId"]: r for r in receipts}
        outcomes = []
        for call_id in record["proposalOrder"]:
            receipt = by_call.get(call_id)
            if receipt is None:
                continue
            outcomes.append(
                {
                    "dossierId": receipt["dossierId"],
                    "callId": call_id,
                    "action": receipt["action"],
                    "proposalDigest": receipt["proposalDigest"],
                    "receiptId": receipt["receiptId"],
                    "status": "executed" if receipt["accepted"] else "rejected",
                }
            )

        response = {
            "profile": PROFILE,
            "evaluationId": evaluation_id,
            "status": "completed",
            "inputDigest": supplied_digest,
            "outcomes": outcomes,
        }
        record["commit"] = {"receiptsKey": receipts_key, "response": response}
        store.put(NS_EVAL, evaluation_id, record)
        return JSONResponse(response, status_code=200)


async def _handle(request: Request) -> JSONResponse:
    if int(request.headers.get("content-length") or 0) > 8 * 1024 * 1024:
        return _error(413, "BODY_TOO_LARGE", "request body too large")
    try:
        body = await request.json()
    except Exception:
        return _error(400, "INVALID_JSON", "body must be JSON")
    if not isinstance(body, dict):
        return _error(400, "INVALID_BODY", "body must be a JSON object")
    if body.get("profile") != PROFILE:
        return _error(400, "UNSUPPORTED_PROFILE", f"profile must be {PROFILE}")
    operation = body.get("operation")
    if operation == "propose":
        return _propose(body)
    if operation == "commit":
        return _commit(body)
    return _error(400, "UNSUPPORTED_OPERATION", "operation must be propose or commit")


@router.post("/mailroom")
async def mailroom(request: Request) -> JSONResponse:
    return await _handle(request)


@router.post("/mailroom/")
async def mailroom_slash(request: Request) -> JSONResponse:
    return await _handle(request)
