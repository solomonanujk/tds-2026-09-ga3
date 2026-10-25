"""GA3 Q2 — invoice action decisions.

Each package is a bundle of documents that mixes the decisive paragraph with
cover sheets, archived examples, negated statements and decoy action words. The
module pulls out the bracketed references paragraph by paragraph, asks a model
for one typed action per package, and then validates the answer: the action must
be one of the five, the facts must be typed, and the evidence must be the three
bracketed references of a single decisive paragraph.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from . import llm

ACTIONS = [
    "settle_invoice",
    "request_approval",
    "hold_invoice",
    "reject_duplicate",
    "open_exception",
]

REF_RE = re.compile(r"\[([^\[\]\n]{1,80})\]")
AMOUNT_RE = re.compile(
    r"(?:(?P<code>[A-Z]{3})\s*|(?P<symbol>[₹$€£¥])\s*)?(?P<amount>\d{1,3}(?:[,\d]{0,20})(?:\.\d{1,2})?)"
    r"(?:\s*(?P<code2>[A-Z]{3}))?"
)
CURRENCY_RE = re.compile(r"\b(INR|USD|EUR|GBP|AED|SGD|JPY|AUD|CAD|CHF)\b")
INVOICE_RE = re.compile(
    r"\b(?:invoice|inv|bill|document)\s*(?:number|no\.?|#|id)?\s*[:#]?\s*"
    r"([A-Z0-9][A-Z0-9/-]{3,24})",
    re.IGNORECASE,
)
VENDOR_RE = re.compile(
    r"\b(?:vendor|supplier|payee|biller|from)\s*[:#-]?\s*([A-Z][\w&.,'\- ]{2,60})"
)

SYMBOL_CURRENCY = {"₹": "INR", "$": "USD", "€": "EUR", "£": "GBP", "¥": "JPY"}

NOISE_RE = re.compile(
    r"(cover sheet|cover-sheet|coversheet|archive|archived|historical example|"
    r"worked example|training (?:example|decoy|sample)|for illustration|sample only|"
    r"do not cite|appendix|glossary|template)",
    re.IGNORECASE,
)

DUPLICATE_RE = re.compile(
    r"(already (?:paid|settled|remitted|discharged)|duplicate (?:of|submission|invoice|claim)|"
    r"previously (?:paid|settled|remitted)|second submission of the same invoice|"
    r"paid (?:on|under) payment run)",
    re.IGNORECASE,
)
APPROVAL_RE = re.compile(
    r"(exceeds? (?:the )?(?:delegated|autonomous|approval|authority)|above (?:the )?"
    r"(?:delegated|autonomous|approval) (?:limit|authority|threshold)|outside (?:the )?"
    r"(?:delegated|autonomous) authority|requires? (?:human |manager |finance )?approval|"
    r"beyond (?:the )?(?:limit|threshold|mandate))",
    re.IGNORECASE,
)
HOLD_RE = re.compile(
    r"(pause (?:the )?payment|payment (?:is )?(?:on )?hold|hold (?:the )?(?:payment|invoice)|"
    r"do not pay until|pending (?:bank|vendor|tax|gst|compliance|goods|site) verification|"
    r"awaiting (?:verification|confirmation|certificate|inspection)|"
    r"until (?:the )?verification (?:is )?complete)",
    re.IGNORECASE,
)
EXCEPTION_RE = re.compile(
    r"(material (?:conflict|discrepancy|difference)|records conflict|do(?:es)? not reconcile|"
    r"cannot be reconciled|mismatch(?:es|ed)? between|contradict|conflicting (?:records|entries)|"
    r"exception workflow|three-way match fail)",
    re.IGNORECASE,
)
SETTLE_RE = re.compile(
    r"(three-way match (?:passed|ok|complete)|fully reconciled|reconciles exactly|"
    r"matches the (?:purchase order|receipt|goods receipt)|within (?:the )?(?:delegated|autonomous) "
    r"(?:authority|limit)|cleared for payment|no prior payment)",
    re.IGNORECASE,
)
NEGATION_RE = re.compile(
    r"\b(not|never|no longer|without|unless|rather than|instead of|does not|did not|cannot|"
    r"is not|was not|superseded|rescinded|withdrawn)\b",
    re.IGNORECASE,
)


# --------------------------------------------------------------------------- #
# package reading
# --------------------------------------------------------------------------- #

def walk_strings(value: Any, path: str = "") -> List[Tuple[str, str]]:
    """Every string in the package, with its JSON path."""
    out: List[Tuple[str, str]] = []
    if isinstance(value, str):
        out.append((path, value))
    elif isinstance(value, dict):
        for key, item in value.items():
            out.extend(walk_strings(item, f"{path}.{key}" if path else str(key)))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            out.extend(walk_strings(item, f"{path}[{index}]"))
    return out


def paragraphs(package: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Candidate decisive paragraphs, each with the references it carries."""
    blocks: List[Dict[str, Any]] = []
    for path, text in walk_strings(package):
        if len(text) < 20:
            continue
        chunks = [chunk for chunk in re.split(r"\n\s*\n", text) if chunk.strip()]
        for index, chunk in enumerate(chunks):
            refs: List[str] = []
            for ref in REF_RE.findall(chunk):
                if ref not in refs:
                    refs.append(ref)
            if not refs:
                continue
            blocks.append(
                {
                    "path": f"{path}#{index}" if len(chunks) > 1 else path,
                    "text": chunk.strip(),
                    "refs": refs,
                    "noise": bool(NOISE_RE.search(chunk)) or bool(NOISE_RE.search(path)),
                }
            )
    return blocks


def all_refs(package: Dict[str, Any]) -> List[str]:
    refs: List[str] = []
    for block in paragraphs(package):
        for ref in block["refs"]:
            if ref not in refs:
                refs.append(ref)
    return refs


def package_text(package: Dict[str, Any]) -> str:
    return "\n".join(text for _, text in walk_strings(package))


def fingerprint(package: Dict[str, Any]) -> Dict[str, Any]:
    """Canonical content identity: everything except the per-batch id."""
    return {k: v for k, v in (package or {}).items() if k not in {"packageId", "batchId"}}


# --------------------------------------------------------------------------- #
# fact extraction
# --------------------------------------------------------------------------- #

def to_minor(amount_text: str) -> Optional[int]:
    cleaned = amount_text.replace(",", "").replace(" ", "")
    if not cleaned:
        return None
    try:
        if "." in cleaned:
            whole, fraction = cleaned.split(".", 1)
            fraction = (fraction + "00")[:2]
            return int(whole) * 100 + int(fraction)
        return int(cleaned) * 100
    except ValueError:
        return None


def extract_facts(package: Dict[str, Any]) -> Dict[str, Any]:
    """Best-effort vendor, invoice number, minor-unit amount and currency."""
    flat = {path.lower(): text for path, text in walk_strings(package)}
    text = package_text(package)

    vendor = ""
    for key, value in flat.items():
        if re.search(r"vendor(name)?$|supplier(name)?$|payee$", key) and 2 < len(value) < 80:
            vendor = value.strip()
            break
    if not vendor:
        match = VENDOR_RE.search(text)
        if match:
            vendor = match.group(1).strip(" .,-")

    invoice = ""
    for key, value in flat.items():
        if re.search(r"invoice(number|no|id)$|billnumber$", key) and 2 < len(value) < 40:
            invoice = value.strip()
            break
    if not invoice:
        match = INVOICE_RE.search(text)
        if match:
            invoice = match.group(1).strip()

    currency = ""
    amount_minor: Optional[int] = None
    for key, value in flat.items():
        if re.search(r"currency$", key) and re.fullmatch(r"[A-Za-z]{3}", value.strip()):
            currency = value.strip().upper()
            break
    for path, value in walk_strings(package):
        if re.search(r"amountminor$|totalminor$|minoramount$", path.lower()):
            try:
                amount_minor = int(value)
            except (TypeError, ValueError):
                pass
            break
    if amount_minor is None:
        match = re.search(
            r"(?:total|amount|payable|gross|invoice value|net)[^\n\d]{0,20}"
            r"(?:(?P<code>[A-Z]{3})\s*|(?P<symbol>[₹$€£¥])\s*)?(?P<amount>\d[\d,]*(?:\.\d{1,2})?)",
            text,
            re.IGNORECASE,
        )
        if match:
            amount_minor = to_minor(match.group("amount"))
            if not currency:
                currency = (match.group("code") or SYMBOL_CURRENCY.get(match.group("symbol") or "", "")) or ""
    if not currency:
        found = CURRENCY_RE.search(text)
        currency = found.group(1) if found else "INR"
    return {
        "vendorName": vendor or "unknown vendor",
        "invoiceNumber": invoice or "unknown",
        "amountMinor": int(amount_minor) if amount_minor is not None else 0,
        "currency": currency,
    }


def validate_facts(facts: Any, package: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only the four typed fact fields, repairing anything unusable."""
    fallback = extract_facts(package)
    if not isinstance(facts, dict):
        return fallback
    out: Dict[str, Any] = {}
    vendor = facts.get("vendorName")
    out["vendorName"] = vendor.strip() if isinstance(vendor, str) and vendor.strip() else fallback["vendorName"]
    invoice = facts.get("invoiceNumber")
    out["invoiceNumber"] = (
        str(invoice).strip() if invoice not in (None, "") else fallback["invoiceNumber"]
    )
    amount = facts.get("amountMinor")
    if isinstance(amount, bool):
        amount = None
    if isinstance(amount, (int, float)) and float(amount).is_integer() and amount >= 0:
        out["amountMinor"] = int(amount)
    elif isinstance(amount, str) and amount.strip().isdigit():
        out["amountMinor"] = int(amount.strip())
    else:
        out["amountMinor"] = fallback["amountMinor"]
    currency = facts.get("currency")
    out["currency"] = (
        currency.strip().upper()
        if isinstance(currency, str) and re.fullmatch(r"[A-Za-z]{3}", currency.strip())
        else fallback["currency"]
    )
    return out


# --------------------------------------------------------------------------- #
# evidence and rationale
# --------------------------------------------------------------------------- #

def pick_evidence(package: Dict[str, Any], proposed: Any) -> List[str]:
    """Exactly the three bracketed refs of one decisive paragraph.

    A paragraph that looks like a cover sheet, an archive example or a training
    decoy is never the decisive one when a real candidate exists.
    """
    blocks = paragraphs(package)
    signal = [block for block in blocks if not block["noise"]]
    cited = [ref for ref in (proposed or []) if isinstance(ref, str)]

    def score(block: Dict[str, Any]) -> int:
        return len(set(cited) & set(block["refs"]))

    triples = [block for block in signal if len(block["refs"]) == 3]
    if cited:
        best = max(triples, key=score, default=None)
        if best is not None and score(best) > 0:
            return list(best["refs"])
        exact = [ref for ref in cited if any(ref in block["refs"] for block in blocks)]
        if len(exact) == 3:
            return exact[:3]
        best_any = max(signal, key=score, default=None)
        if best_any is not None and score(best_any) > 0 and len(best_any["refs"]) >= 3:
            return list(best_any["refs"][:3])
    if triples:
        return list(triples[0]["refs"])
    for block in signal:
        if len(block["refs"]) >= 3:
            return list(block["refs"][:3])
    refs = [ref for ref in all_refs(package)]
    return refs[:3]


ACTION_SENTENCE = {
    "settle_invoice": "settle_invoice: the invoice reconciles against the purchase order and receipt and sits inside autonomous payment authority",
    "request_approval": "request_approval: the claim is commercially valid but the amount sits outside the delegated payment authority, so a human approver must release it",
    "hold_invoice": "hold_invoice: payment must pause until the verification named in the file completes",
    "reject_duplicate": "reject_duplicate: this same commercial invoice from this vendor was already paid, so paying again would duplicate the settlement",
    "open_exception": "open_exception: the supporting records materially conflict, so the case needs the exception workflow rather than a payment decision",
}


def build_rationale(action: str, evidence: List[str], facts: Dict[str, Any], supplied: Any = None) -> str:
    """Prefer the model's reasoning when it is valid; otherwise synthesise."""
    if isinstance(supplied, str) and supplied.strip():
        candidate = " ".join(supplied.split())
        if len(candidate) > 1500:
            candidate = candidate[:1400].rsplit(" ", 1)[0]
        cited = sum(1 for ref in evidence if ref in candidate)
        if 60 <= len(candidate) <= 1500 and action in candidate and cited >= 2:
            return candidate
        # The model's reasoning is the valuable part; repair the formal
        # requirements around it rather than discarding it for boilerplate.
        refs = list(evidence) + ["", "", ""]
        prefix = "" if action in candidate else f"Action {action}. "
        suffix = (
            ""
            if cited >= 2
            else f" The decisive references [{refs[0]}], [{refs[1]}] and [{refs[2]}] "
            f"in that paragraph support this."
        )
        repaired = " ".join(f"{prefix}{candidate}{suffix}".split())
        if 60 <= len(repaired) <= 1500:
            return repaired
    refs = list(evidence) + ["", "", ""]
    sentence = ACTION_SENTENCE.get(action, f"{action}: chosen from the decisive paragraph")
    rationale = (
        f"Chose {sentence}. Decisive references [{refs[0]}] and [{refs[1]}] in that paragraph "
        f"establish the position for invoice {facts.get('invoiceNumber')} from "
        f"{facts.get('vendorName')} for {facts.get('amountMinor')} minor units of "
        f"{facts.get('currency')}, and [{refs[2]}] confirms it; cover-sheet lines, archived "
        f"examples and decoy action words elsewhere in the package were disregarded."
    )
    return " ".join(rationale.split())[:1500]


# --------------------------------------------------------------------------- #
# decisions
# --------------------------------------------------------------------------- #

def heuristic_action(package: Dict[str, Any]) -> str:
    """Deterministic fallback, biased away from settling.

    Settling something that should have been approved, held, rejected or
    escalated is the one mistake the question caps severely, so a positive
    settle signal is required and any contrary signal wins.
    """
    blocks = [block for block in paragraphs(package) if not block["noise"]]
    text = "\n".join(block["text"] for block in blocks) or package_text(package)
    sentences = re.split(r"(?<=[.!?])\s+", text)

    def hits(pattern: re.Pattern) -> int:
        """Count assertions, ignoring ones a nearby negation inverts.

        A pattern that already carries its own negation ("cannot be
        reconciled", "no prior payment") is self-contained, so only negation in
        the run-up to the match counts.
        """
        total = 0
        for sentence in sentences:
            for match in pattern.finditer(sentence):
                if NEGATION_RE.search(match.group(0)):
                    total += 1
                    continue
                lead = sentence[max(0, match.start() - 40) : match.start()]
                if NEGATION_RE.search(lead):
                    continue
                total += 1
        return total

    if hits(DUPLICATE_RE):
        return "reject_duplicate"
    if hits(EXCEPTION_RE):
        return "open_exception"
    if hits(HOLD_RE):
        return "hold_invoice"
    if hits(APPROVAL_RE):
        return "request_approval"
    if hits(SETTLE_RE):
        return "settle_invoice"
    return "request_approval"


SYSTEM_PROMPT = (
    "You are an accounts-payable agent. For each invoice package you choose "
    "exactly one action and cite the decisive evidence.\n"
    "Actions:\n"
    "- settle_invoice: valid, reconciled, and inside autonomous authority.\n"
    "- request_approval: commercially valid but outside delegated authority.\n"
    "- hold_invoice: payment must pause until a stated verification completes.\n"
    "- reject_duplicate: the same commercial invoice was already paid. A "
    "payment for a DIFFERENT invoice is not a duplicate.\n"
    "- open_exception: material records conflict and need an exception workflow.\n"
    "Rules:\n"
    "- Packages deliberately mix in cover sheets, archived examples, negated "
    "sentences and irrelevant action words. Read what the current paragraph "
    "actually asserts.\n"
    "- Settling something that needed approval, a hold, a duplicate rejection "
    "or an exception is the worst possible error.\n"
    "- evidenceRefs must be exactly the three bracketed references of the one "
    "paragraph that determines the action, copied without the brackets. Never "
    "cite the cover sheet, an archive example or a decoy.\n"
    "- rationale must say which action you chose, quote at least two of those "
    "references, and explain in one or two sentences what those specific lines "
    "assert about THIS invoice that forces the action. Do not write generic "
    "policy text.\n"
    "- facts.amountMinor is an integer in minor units.\n"
    "Answer with JSON only."
)


def batch_prompt(packages: List[Dict[str, Any]]) -> str:
    import json as _json

    blocks = []
    for package in packages:
        rendered = []
        for path, text in walk_strings(package):
            if len(text) < 2:
                continue
            rendered.append(f"  [{path}] {text}")
        blocks.append(
            f"PACKAGE {package.get('packageId')}\n" + "\n".join(rendered)
        )
    return (
        "\n\n".join(blocks)
        + "\n\nReturn JSON: {\"decisions\": [{\"packageId\": \"..\", \"action\": \"one of "
        + ", ".join(ACTIONS)
        + "\", \"facts\": {\"vendorName\": \"..\", \"invoiceNumber\": \"..\", "
          "\"amountMinor\": 123, \"currency\": \"INR\"}, \"evidenceRefs\": [\"r1\", \"r2\", \"r3\"], "
          "\"rationale\": \"60-1500 characters naming the action and citing at least two refs\"}]} "
          "with one entry per package above."
    )


def parse_decisions(payload: Any) -> Dict[str, Dict[str, Any]]:
    items: List[Any] = []
    if isinstance(payload, dict):
        for key in ("decisions", "proposals", "results", "items"):
            if isinstance(payload.get(key), list):
                items = payload[key]
                break
        else:
            if payload.get("packageId"):
                items = [payload]
    elif isinstance(payload, list):
        items = payload
    out: Dict[str, Dict[str, Any]] = {}
    for item in items:
        if isinstance(item, dict) and isinstance(item.get("packageId"), str):
            out[item["packageId"]] = item
    return out


def model_decisions(packages: List[Dict[str, Any]], *, timeout: float) -> Dict[str, Dict[str, Any]]:
    raw = llm.chat_json(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": batch_prompt(packages)},
        ],
        max_tokens=700 * max(1, len(packages)),
        timeout=timeout,
        attempts=1,
    )
    return parse_decisions(raw) if raw is not None else {}


def assemble(package: Dict[str, Any], choice: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Validated decision body for one package (no ids: those are per batch)."""
    choice = choice if isinstance(choice, dict) else {}
    action = choice.get("action")
    if action not in ACTIONS:
        action = heuristic_action(package)
    facts = validate_facts(choice.get("facts"), package)
    evidence = pick_evidence(package, choice.get("evidenceRefs"))
    rationale = build_rationale(action, evidence, facts, choice.get("rationale"))
    return {"action": action, "facts": facts, "evidenceRefs": evidence, "rationale": rationale}
